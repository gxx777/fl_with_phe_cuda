import os
import argparse, json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Cifar_2NN, Cifar_CNN, Mnist_2NN, Mnist_CNN, RestNet18
from clients import ClientsGroup, client
from phe import paillier

os.environ['CUDA_VISIBLE_DEVICES'] = 'gpu'
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#if args["mps"]:
#        print("use_mps")
#        dev = torch.device("mps")
    # 定义使用模型(全连接 or 简单卷积)
print("Let's use", torch.cuda.device_count(), "GPUs!")
