

## INSTALL

```
1. git clone https://github.com/pybind/pybind11.git
2. git clone https://github.com/NVlabs/CGBN.git
3. pip3 install phe-cuda
apt-get install libgmp-dev
```



## 编译gpu_lib.so 文件

需要配置`CMakeLists.txt` 文件

```
set(PYTHON_INCLUDE_DIRS "/root/miniconda3/include/python3.8")
set(PYTHON_LIBRARIES "/root/miniconda3/lib")
set(CGBN_INCLUDES "/root/CGBN/include")
set(PYBIND11_INCLUDE "/root/pybind11/include")
```





## Usage

```
python3 server_encrypt_gpu.py -c ./utils/conf.json
```







