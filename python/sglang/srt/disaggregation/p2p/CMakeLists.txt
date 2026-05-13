cmake_minimum_required(VERSION 3.14)
project(p2p_transfer)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

# Manually set CUDA path
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

pybind11_add_module(cuda_p2p_transfer bindings.cpp p2p_transfer_engine.cpp)

target_link_libraries(cuda_p2p_transfer PRIVATE ${CUDA_LIBRARIES} cudart)