## Compile the model
[english_version](https://github.com/LeslieZhoa/Triton-Torch-Custom/blob/main/Make-Model2/ReadMe.md)
[中文版本](https://github.com/LeslieZhoa/Triton-Torch-Custom/blob/main/Make-Model2/ReadMe-chinese.md)
1. run pytorch docker and enter docker to compile the model
```
sudo bash build.sh
sudo bash run_pytorch.sh
```
2. cmake to generate makefile, Compile dynamic link library
```shell
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../
make 
```
3. generate pt
```
cd ..
python convert_op.py
```
4. Switch out of docker to copy model files
```
sudo docker ps  # check doker id like 1bdff20dd069
sudo docker cp 1bdff20dd069:/test_model/model.pt ../Triton-Serve/models/selfop/1/model.pt
sudo docker cp 1bdff20dd069:/test_model/build/libfusebias.so ../Triton-Serve/
```
## Some instructions
1. ```cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../```is to specify the torch_cmake path
2. If you don't find Python.h when make, add ``` include_directories(/opt/conda/include/python3.6m/)``` in CMakeLists
3. IF you find the questions about ">=" when make，ADD set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_NO_HALF_OPERATORS__" ) in CMakeLists
4. IF you find that you need convert int to int64_t and float to double when make，you should let all int64_t,float be double in cpp and cu,except 
    ```
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    ``` in cu
5. When you find undefined symbol:THPVariableClass. There is .so not connected,Find to manually add target_link_libraries(fusebias /opt/conda/lib/python3.6/site-packages/torch/lib/libtorch_python.so) in CMakeLists
6. Model code, don't use backpropagation code, refer to convert_op.py