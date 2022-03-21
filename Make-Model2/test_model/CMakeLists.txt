cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# fix your own nvcc path，or delete this if can work。
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(upfirdn LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_NO_HALF_OPERATORS__" )

# fix your own python path，or delete this if can work
include_directories(/opt/conda/include/python3.6m/)

# target_compile_features(add2 PRIVATE cxx_std_14)
set(SRCS upfirdn2d.cpp upfirdn2d_kernel.cu)
add_library(upfirdn SHARED ${SRCS})
target_compile_features(upfirdn PRIVATE cxx_std_14)
target_link_libraries(upfirdn "${TORCH_LIBRARIES}")
target_link_libraries(upfirdn /opt/conda/lib/python3.6/site-packages/torch/lib/libtorch_python.so)
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "") 
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
