# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cmake_minimum_required(VERSION 3.18)
project(sgl_kernel LANGUAGES CXX CUDA)

# Set C++ and CUDA standards.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable ccache if available.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Find required packages.
find_package(Python REQUIRED COMPONENTS 
    Interpreter 
    Development 
    Development.Module 
    Development.Embed
)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Print build information.
message(STATUS "Build configuration:")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  Python version: ${Python_VERSION}")
message(STATUS "  Torch version: ${TORCH_VERSION}")
message(STATUS "  CUDA version: ${CUDAToolkit_VERSION}")

# Define operator namespace.
set(OPERATOR_NAMESPACE "sgl_kernel")

# Set paths for third-party libraries.
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass")
set(FLASHINFER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/flashinfer")
set(DEEPGEMM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/deepgemm")

# Include directories.
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${FLASHINFER_DIR}/include
    ${FLASHINFER_DIR}/include/gemm
    ${FLASHINFER_DIR}/csrc
    ${DEEPGEMM_DIR}/deep_gemm/include
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file(
    ${DEEPGEMM_DIR}/deep_gemm/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/deep_gemm/__init__.py
    COPYONLY
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/deep_gemm/include
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/deep_gemm/include
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CUTLASS_DIR}/include/cute
        ${CMAKE_CURRENT_BINARY_DIR}/deep_gemm/include/cute
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CUTLASS_DIR}/include/cutlass
        ${CMAKE_CURRENT_BINARY_DIR}/deep_gemm/include/cutlass
    COMMENT "Creating symbolic links for third-party headers"
)

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/deep_gemm/include
)

# Determine CUDA capabilities.
function(get_cuda_version MAJOR MINOR)
    if(DEFINED TORCH_CUDA_VERSION)
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)" CUDA_VERSION_MATCH ${TORCH_CUDA_VERSION})
        set(${MAJOR} ${CMAKE_MATCH_1} PARENT_SCOPE)
        set(${MINOR} ${CMAKE_MATCH_2} PARENT_SCOPE)
    else()
        set(${MAJOR} 0 PARENT_SCOPE)
        set(${MINOR} 0 PARENT_SCOPE)
    endif()
endfunction()

get_cuda_version(CUDA_VERSION_MAJOR CUDA_VERSION_MINOR)

# Set CUDA architectures.
set(CMAKE_CUDA_ARCHITECTURES "75;80;89;90")

# Set compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG \
    -DOPERATOR_NAMESPACE=${OPERATOR_NAMESPACE} \
    -Xcompiler -fPIC \
    -std=c++17 \
    -DFLASHINFER_ENABLE_F16 \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    -DCUTLASS_VERSIONS_GENERATED \
    -DCUTE_USE_PACKED_TUPLE=1 \
    -DCUTLASS_TEST_LEVEL=0 \
    -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 \
    -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
    --ptxas-options=-v \
    --expt-relaxed-constexpr \
    -Xcompiler=-Wconversion \
    -Xcompiler=-fno-strict-aliasing"
)

# Add architecture-specific flags.
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${ARCH},code=sm_${ARCH}")
endforeach()

# Check for SM90A support.
if(CUDA_VERSION_MAJOR GREATER_EQUAL 12 AND CUDA_VERSION_MINOR GREATER_EQUAL 0)
    if((DEFINED ENV{SGL_KERNEL_ENABLE_SM90A}) AND ("$ENV{SGL_KERNEL_ENABLE_SM90A}" STREQUAL "1"))
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90a,code=sm_90a")
        add_compile_definitions(ENABLE_SM90A)
    endif()
endif()

# Check for SM100A support.
if(CUDA_VERSION_MAJOR GREATER_EQUAL 12 AND CUDA_VERSION_MINOR GREATER_EQUAL 8)
    if((DEFINED ENV{SGL_KERNEL_ENABLE_SM100A}) AND ("$ENV{SGL_KERNEL_ENABLE_SM100A}" STREQUAL "1"))
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_100,code=sm_100")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_100a,code=sm_100a")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
    endif()
endif()

# Auto-enable FP8 based on compute capability.
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 8)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.cuda.get_device_capability()[0]*10+torch.cuda.get_device_capability()[1] if torch.cuda.is_available() else 0)"
        OUTPUT_VARIABLE SM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(SM_VERSION GREATER_EQUAL 90)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFLASHINFER_ENABLE_FP8 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2")
        add_compile_definitions(ENABLE_FP8)
    endif()
endif()

# Check for BF16 support.
if((DEFINED ENV{SGL_KERNEL_ENABLE_BF16}) AND ("$ENV{SGL_KERNEL_ENABLE_BF16}" STREQUAL "1"))
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFLASHINFER_ENABLE_BF16")
    add_compile_definitions(ENABLE_BF16)
endif()

# Remove problematic CUDA flags.
string(REPLACE "-D__CUDA_NO_HALF_OPERATORS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF2_OPERATORS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

add_compile_definitions(TORCH_EXTENSION_NAME=\"sgl_kernel.common_ops\")

# Collect source files.
set(SOURCES
    csrc/allreduce/trt_reduce_internal.cu
    csrc/allreduce/trt_reduce_kernel.cu
    csrc/attention/lightning_attention_decode_kernel.cu
    csrc/elementwise/activation.cu
    csrc/elementwise/fused_add_rms_norm_kernel.cu
    csrc/elementwise/rope.cu
    csrc/gemm/bmm_fp8.cu
    csrc/gemm/cublas_grouped_gemm.cu
    csrc/gemm/awq_kernel.cu
    csrc/gemm/fp8_gemm_kernel.cu
    csrc/gemm/fp8_blockwise_gemm_kernel.cu
    csrc/gemm/int8_gemm_kernel.cu
    csrc/gemm/per_token_group_quant_fp8.cu
    csrc/gemm/per_token_quant_fp8.cu
    csrc/gemm/per_tensor_quant_fp8.cu
    csrc/moe/moe_align_kernel.cu
    csrc/moe/moe_topk_softmax_kernels.cu
    csrc/speculative/eagle_utils.cu
    csrc/speculative/speculative_sampling.cu
    csrc/speculative/packbit.cu
    csrc/torch_extension.cc
    3rdparty/flashinfer/csrc/norm.cu
    3rdparty/flashinfer/csrc/renorm.cu
    3rdparty/flashinfer/csrc/sampling.cu
)

# Rename library to match setup.py extension name.
add_library(common_ops MODULE ${SOURCES})

target_link_libraries(common_ops PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)

set_target_properties(common_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../torch/lib"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sgl_kernel"
    OUTPUT_NAME "common_ops"
    PREFIX ""
    SUFFIX ".so"
)

if(${Python_VERSION} VERSION_GREATER_EQUAL "3.9")
    set_property(TARGET common_ops PROPERTY 
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_DEFINITIONS "Py_LIMITED_API=0x03090000"
    )
endif()

install(TARGETS common_ops 
    LIBRARY DESTINATION sgl_kernel
)

install(
    DIRECTORY ${DEEPGEMM_DIR}/deep_gemm/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/deep_gemm
)

install(
    DIRECTORY ${CUTLASS_DIR}/include/cute
    DESTINATION ${CMAKE_INSTALL_PREFIX}/deep_gemm/include
)

install(
    DIRECTORY ${CUTLASS_DIR}/include/cutlass
    DESTINATION ${CMAKE_INSTALL_PREFIX}/deep_gemm/include
)
