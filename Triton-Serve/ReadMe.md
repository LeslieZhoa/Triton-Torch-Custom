## run Triton-serving
[english_version](https://github.com/LeslieZhoa/Triton-Torch-Custom/blob/main/Triton-Serve/ReadMe.md)
[中文版本](https://github.com/LeslieZhoa/Triton-Torch-Custom/blob/main/Triton-Serve/ReadMe-chinese.md)
1. build docker
```
sudo bash build.sh
sudo bash run_tensorrt_serving.sh
```
2. test（It can be run in the existing triton environment on your own server）
```
python generator_trt_rpc_client.py
```