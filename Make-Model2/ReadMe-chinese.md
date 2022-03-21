## 编译模型
1. 运行pytorch docker,进入docker编译模型
```
sudo bash build.sh
sudo bash run_pytorch.sh
```
2. cmake生成makefile, 编译动态链接库
```shell
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../
make 
```
3. 生成pt
```
cd ..
python convert_op.py
```
4. 切换出docker复制模型文件
```
sudo docker ps  # 查看doker id 如1bdff20dd069
sudo docker cp 1bdff20dd069:/test_model/model.pt ../Triton-Serve/models/selfop/1/model.pt
sudo docker cp 1bdff20dd069:/test_model/build/libfusebias.so ../Triton-Serve/
```
## 一些说明
1. ```cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../```是指定torch_cmake路径
2. make是发现未找到Python.h 所以在CMakeLists里添加``` include_directories(/opt/conda/include/python3.6m/)```
3. make时>=等符号重载不明确，所以在CMakeLists里添加set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_NO_HALF_OPERATORS__" )
4. make出错提示int转成int64_t,float转成double，将cpp和cu里的所有int64_t,float转成double，除了cu里的
    ```
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    ```
5. 推理时出错undefined symbol:THPVariableClass，有.so未连接上，查找到在CMakeLists里手动添加target_link_libraries(fusebias /opt/conda/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
6. 模型代码，不要使用反向传播代码，参考convert_op.py