cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(obds_torch_extension)

set(CMAKE_PREFIX_PATH "/home/wiser-renjie/libtorch")
set(pybind11_DIR "/home/wiser-renjie/anaconda3/envs/blockcopy/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(obds_torch_extension OBDS_pytorch.cpp)
target_link_libraries(obds_torch_extension PRIVATE "${TORCH_LIBRARIES}")
