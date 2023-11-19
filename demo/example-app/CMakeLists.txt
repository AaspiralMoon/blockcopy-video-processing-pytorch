cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

set(CMAKE_PREFIX_PATH "/home/wiser-renjie/libtorch")
set(pybind11_DIR "/home/wiser-renjie/anaconda3/envs/blockcopy/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)