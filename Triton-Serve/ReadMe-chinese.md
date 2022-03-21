## 运行Triton服务
1. 构建docker
```
sudo bash build.sh
sudo bash run_tensorrt_serving.sh
```
2. 测试（可在自己服务器已有triton环境运行）
```
python generator_trt_rpc_client.py
```
