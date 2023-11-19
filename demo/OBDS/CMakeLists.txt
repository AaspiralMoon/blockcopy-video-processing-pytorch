cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(OBDS_zoo)

set(CMAKE_PREFIX_PATH "/home/wiser-renjie/libtorch")
set(pybind11_DIR "/home/wiser-renjie/anaconda3/envs/blockcopy/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
set(Torch_DIR "/home/wiser-renjie/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(OBDS_zoo OBDS_zoo.cpp)
target_link_libraries(OBDS_zoo PRIVATE "${TORCH_LIBRARIES}")
