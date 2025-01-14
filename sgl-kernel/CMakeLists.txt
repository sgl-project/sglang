cmake_minimum_required(VERSION 3.18)
project(sgl-kernel LANGUAGES CXX CUDA)

# Basic settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUTLASS_DIR "3rdparty/cutlass")
set(CUB_DIR "3rdparty/cub")

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

find_package(Torch REQUIRED)

# Warp Reduce library
add_library(_kernels SHARED
    src/sgl-kernel/csrc/trt_reduce_internal.cu
    src/sgl-kernel/csrc/trt_reduce_kernel.cu
    src/sgl-kernel/csrc/moe_align_kernel.cu
    src/sgl-kernel/csrc/int8_gemm_kernel.cu
    src/sgl-kernel/csrc/sampling_scaling_penalties.cu
    src/sgl-kernel/csrc/rms_norm_kernel.cu
    src/sgl-kernel/csrc/sgl_kernel_ops.cu
)

target_include_directories(_kernels
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sgl-kernel/csrc
        ${CUDA_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
        ${CUB_DIR}/cub
)

target_link_libraries(_kernels
    PRIVATE
        ${TORCH_LIBRARIES}
        Python3::Python
)

# Set common properties for both libraries
foreach(target _kernels)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        PREFIX ""
        SUFFIX ".so"
    )
endforeach()
