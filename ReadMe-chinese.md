## Triton-Pytorch自定义算子教程
### 前提
可以按照nvidia-docker-install.sh安装nvidia-docker
### 说明
本实例以stylegan fused_bias_act为例，pytorch1.6,cuda11
### 结构
- Make-Model<br>
    - 编译动态链接库
    - 打包模型.pt文件
- Triton-Serve<br>
    - 服务部署

### 使用步骤
- 进入Make-Model文件夹打包模型相关
- 进入Triton-Serve文件夹部署服务
### 新增Swap2p打包
在Swap2P中
