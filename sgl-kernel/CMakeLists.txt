cmake_minimum_required(VERSION 3.18)
project(sgl-kernel LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
message(STATUS "TORCH_CMAKE_PATH: ${TORCH_CMAKE_PATH}")

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

find_package(Torch REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

add_library(warp_reduce SHARED
    src/sgl-kernel/csrc/warp_reduce.cc
    src/sgl-kernel/csrc/warp_reduce_kernel.cu
)

target_include_directories(warp_reduce PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(warp_reduce PRIVATE
    ${TORCH_LIBRARIES}
    ${PYTHON_LIBRARIES}
)

set_target_properties(warp_reduce PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
