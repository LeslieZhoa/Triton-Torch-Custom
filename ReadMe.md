## Triton-Pytorch Custom operator tutorial
teach you how to run pytorch Custom operator in triton. <br>
[english_version](https://github.com/LeslieZhoa/Triton-Torch-Custom/blob/main/ReadMe.md)
[中文版本](https://github.com/LeslieZhoa/Triton-Torch-Custom/blob/main/ReadMe-chinese.md)
### precondition
you can install nvidia-docker by nvidia-docker-install.sh
### Discription
stylegan fused_bias_act,pytorch1.6,cuda11 For example
### structure
- Make-Model<br>
    - Compiling dynamic link libraries
    - Packaging model .pt， file
- Triton-Serve<br>
    - Service deployment

### Use steps
- enter Make-Model folder and package related models
- enter Triton-Serve folder and deployment service
### ADD Swap2p 
enter Swap2P
