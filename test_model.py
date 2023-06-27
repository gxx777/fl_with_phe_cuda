import os
import argparse, json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Cifar_2NN, Cifar_CNN, Mnist_2NN, Mnist_CNN, RestNet18
from clients import ClientsGroup, client


def test_model(path):
    model = torch.load(path)
    return model


# 中间参数保存路径
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def add_noise(parameters, dp, dev):
    noise = None
    # 不加噪声
    if dp == 0:
        return parameters
    # 拉普拉斯噪声
    elif dp == 1:
        noise = torch.tensor(np.random.laplace(0, sigma, parameters.shape),dtype=torch.float32).to(dev)
    # 高斯噪声
    else:
        noise = torch.cuda.FloatTensor(parameters.shape).normal_(0, sigma)

    return parameters.add_(noise)


# 标签one-hot编码
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


if __name__ == "__main__":

    # 定义解析器
    parser = argparse.ArgumentParser(description='FedAvg')
    parser.add_argument('-c', '--conf', dest='conf')
    arg = parser.parse_args()

    # 解析器解析json文件
    with open(arg.conf, 'r') as f:
        args = json.load(f)

    # 创建中间参数保存目录
    test_mkdir(args['save_path'])

    # 使用gpu or cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args["mps"]:
        dev = torch.device("mps")
        print("use_mps")

    # 定义使用模型(全连接 or 简单卷积)
    # net = None
    # if args['model_name'] == 'mnist_2nn':
    #     net = Mnist_2NN()
    # elif args['model_name'] == 'mnist_cnn':
    #     net = Mnist_CNN()
    # elif args['model_name'] == 'cifar_cnn':
    #     net = Cifar_CNN()
    # elif args['model_name'] == 'resnet18':
    #     net = RestNet18()
    # elif args['model_name'] == 'cifar_2nn':
    #     net = Cifar_2NN()

    #     # 如果gpu设备不止一个，并行计算
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = torch.nn.DataParallel(net)
    # net = net.to(dev)

    # # 定义损失函数和优化器
    # loss_func = F.cross_entropy
    # opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    # 定义数据集
    type = args['type']

    # 定义多个参与方，导入训练、测试数据集
    myClients = ClientsGroup(type, args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader
    trainDataLoader = myClients.train_data_loader

    # 每轮迭代的参与方个数
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # # 初始化全局参数
    # global_parameters = {}
    # for key, var in net.state_dict().items():
    #     global_parameters[key] = var.clone()

    # 定义噪声的类型和幅度
    dp = args['noise']
    sigma = args['sigma']

    # 保存训练集accuracy和验证集accuracy
    train_acc = []
    val_acc = []

    # 保存训练集loss
    train_loss = []
    sum_accu = 0
    num = 0
    net = test_model("./checkpoints/mnist_cnn_num_comm4_E3_B100_lr0.005_num_clients2_cf1")
       # 遍历每个测试数据
    for data, label in testDataLoader:
    # 转成gpu数据
        data, label = data.to(dev), label.to(dev)
        # 预测（返回结果是概率向量）
        preds = net(data)
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
        print('val_accuracy: {}'.format(sum_accu / num))
        val_acc.append((sum_accu / num).cpu())
    print('val_accuracy: {}'.format(sum_accu / num))
    # val_acc.append((sum_accu / num).cpu())

    # # 全局迭代轮次
    # for i in range(args['num_comm']):
    #     print("communicate round {}".format(i + 1))

    #     # 打乱排序，确定num_in_comm个参与方
    #     order = np.random.permutation(args['num_of_clients'])
    #     clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

    #     sum_parameters = None

    #     # 可视化进度条对选中参与方local_epoch
    #     for client in tqdm(clients_in_comm):
    #         # 本地梯度下降
    #         local_parameters = myClients.clients_set[client].localUpdate(args['local_epoch'], args['batch_size'], net,
    #                                                                      loss_func, opti, global_parameters)
            
    #         # 初始化sum_parameters
    #         if sum_parameters is None:
    #             sum_parameters = {}
    #             for key, var in local_parameters.items():
    #                 sum_parameters[key] = var.clone()
    #                 sum_parameters[key] = add_noise(sum_parameters[key], dp, dev)

    #         else:
    #             for key in sum_parameters:
    #                 sum_parameters[key].add_(add_noise(local_parameters[key], dp, dev))

    #     # 更新全局梯度参数
    #     for var in global_parameters:
    #         global_parameters[var] = (sum_parameters[var] / num_in_comm)

    #     # 不进行计算图构建（无需反向传播）
    #     with torch.no_grad():
    #         # 满足评估的条件，用测试集进行数据评估
    #         if (i + 1) % args['val_freq'] == 0:
    #             # strict表示key、val严格重合才能执行（false不对齐部分默认初始化）
    #             net.load_state_dict(global_parameters, strict=True)
    #             sum_accu = 0
    #             num = 0
    #             # 遍历每个测试数据
    #             for data, label in testDataLoader:
    #                 # 转成gpu数据
    #                 data, label = data.to(dev), label.to(dev)
    #                 # 预测（返回结果是概率向量）
    #                 preds = net(data)
    #                 # 取最大概率label
    #                 preds = torch.argmax(preds, dim=1)
    #                 sum_accu += (preds == label).float().mean()
    #                 num += 1
    #             print('val_accuracy: {}'.format(sum_accu / num))
    #             val_acc.append((sum_accu / num).cpu())

    #             total_loss = 0
    #             # 遍历每个训练数据
    #             for data, label in trainDataLoader:
    #                 # 转成gpu数据
    #                 data, label = data.to(dev), label.to(dev)
    #                 # 预测（返回结果是概率向量）
    #                 preds = net(data)
    #                 target = dense_to_one_hot(label.cpu().numpy())
    #                 target = torch.tensor(target,dtype=torch.float32).to(dev)
    #                 total_loss += torch.nn.functional.cross_entropy(preds, target,
    #                                                                 reduction='sum').item()
    #                 # 取最大概率label
    #                 preds = torch.argmax(preds, dim=1)
    #                 sum_accu += (preds == label).float().mean()
    #                 num += 1
    #             print('train_accuracy: {}'.format(sum_accu / num))
    #             total_loss = total_loss / num
    #             train_acc.append((sum_accu / num).cpu())
    #             train_loss.append(total_loss)

       
