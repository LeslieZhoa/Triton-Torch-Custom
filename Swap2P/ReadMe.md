## Swap2p Custom operator
[english_version](https://github.com/LeslieZhoa/Triton-Torch-Custom/blob/main/Swap2P/ReadMe.md)
[中文版本](https://github.com/LeslieZhoa/Triton-Torch-Custom/blob/main/Swap2P/ReadMe-chinese.md)
1. run docker
```
sudo bash build.sh
sudo bash run_pytorch.sh
```
2. enter docker,Package the model and copy it out
```
python swapAE_export_pt.py
```
3. run triton
- refer to Triton-Serve/Dockerfile,Chain both .so's in, such as ENV LD_PRELOAD="/libfusebias.so:/libupfirdn.so"
