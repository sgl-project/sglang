cmake_minimum_required(VERSION 3.17)
project(p2p_transfer)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(cuda_p2p_transfer bindings.cpp p2p_transfer_engine.cpp)

target_include_directories(cuda_p2p_transfer PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cuda_p2p_transfer PRIVATE CUDA::cudart)
