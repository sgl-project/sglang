cmake_minimum_required(VERSION 3.18)
project(sgl_kernel_cpu_bench CXX)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

set(CPU_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SGL_KERNEL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# Source files for CPU kernels (same as the main build)
file(GLOB CPU_SOURCES
    ${CPU_KERNEL_DIR}/*.cpp
    ${CPU_KERNEL_DIR}/mamba/*.cpp
    ${CPU_KERNEL_DIR}/model/*.cpp
)
list(FILTER CPU_SOURCES EXCLUDE REGEX "torch_extension_cpu.cpp")
list(FILTER CPU_SOURCES EXCLUDE REGEX "interface.cpp")

find_library(NUMA_LIB numa)

add_executable(bench_cpu_kernels bench_cpu_kernels.cpp ${CPU_SOURCES})
target_include_directories(bench_cpu_kernels PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CPU_KERNEL_DIR}
    ${SGL_KERNEL_ROOT}
    ${SGL_KERNEL_ROOT}/include
)
target_link_libraries(bench_cpu_kernels PRIVATE ${TORCH_LIBRARIES} ${NUMA_LIB})
target_compile_options(bench_cpu_kernels PRIVATE -march=native -fopenmp)
target_link_options(bench_cpu_kernels PRIVATE -fopenmp)
