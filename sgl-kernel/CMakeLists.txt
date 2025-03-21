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

# Find pybind11.
find_package(pybind11 CONFIG REQUIRED)

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
    ${DEEPGEMM_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
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
if((CUDA_VERSION_MAJOR GREATER_EQUAL 12 AND CUDA_VERSION_MINOR GREATER_EQUAL 0) OR DEFINED ENV{ENABLE_SM90A})
    if("$ENV{ENABLE_SM90A}" STREQUAL "ON" OR "$ENV{ENABLE_SM90A}" STREQUAL "1")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90a,code=sm_90a")
        add_compile_definitions(ENABLE_SM90A)
    endif()
endif()

# Check for SM100A support.
if((CUDA_VERSION_MAJOR GREATER_EQUAL 12 AND CUDA_VERSION_MINOR GREATER_EQUAL 8) OR DEFINED ENV{ENABLE_SM100A})
    if("$ENV{ENABLE_SM100A}" STREQUAL "ON" OR "$ENV{ENABLE_SM100A}" STREQUAL "1")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_100,code=sm_100")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_100a,code=sm_100a")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
    endif()
endif()

# Check for FP8 support.
if(DEFINED ENV{ENABLE_FP8})
    if("$ENV{ENABLE_FP8}" STREQUAL "ON" OR "$ENV{ENABLE_FP8}" STREQUAL "1")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
            -DFLASHINFER_ENABLE_FP8 \
            -DFLASHINFER_ENABLE_FP8_E4M3 \
            -DFLASHINFER_ENABLE_FP8_E5M2"
        )
        add_compile_definitions(ENABLE_FP8)
    endif()
endif()

# Check for BF16 support.
if(DEFINED ENV{ENABLE_BF16})
    if("$ENV{ENABLE_BF16}" STREQUAL "ON" OR "$ENV{ENABLE_BF16}" STREQUAL "1")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFLASHINFER_ENABLE_BF16")
        add_compile_definitions(ENABLE_BF16)
    endif()
endif()

# Remove problematic CUDA flags.
string(REPLACE "-D__CUDA_NO_HALF_OPERATORS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF2_OPERATORS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# Create a more complete pytorch_extension_utils.h file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/pytorch_extension_utils.h" "#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <sstream>
#include <vector>

// Common check macros
#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x \" must be a CPU tensor\")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x \" must be a CUDA tensor\")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x \" must be contiguous\")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_SHAPE(x, ...) AT_ASSERTM(check_shape(x, {__VA_ARGS__}), #x \" must have shape \" #__VA_ARGS__)
#define CHECK_SHAPES(x, y) AT_ASSERTM(check_same_shape(x, y), #x \" and \" #y \" must have the same shape\")
#define CHECK_DIM(dim, x) AT_ASSERTM(x.dim() == dim, #x \" must have dimension \" #dim)
#define CHECK_EQ(x, y) AT_ASSERTM(x == y, #x \" must be equal to \" #y)

// Dispatch macros
#define _DISPATCH_CASE_F16(TYPE, TYPE_NAME, ...) \
    case at::ScalarType::Half: { using TYPE_NAME = at::Half; __VA_ARGS__; break; }

#define _DISPATCH_CASE_BF16(TYPE, TYPE_NAME, ...) \
    case at::ScalarType::BFloat16: { using TYPE_NAME = at::BFloat16; __VA_ARGS__; break; }

#define _DISPATCH_CASE_FP32(TYPE, TYPE_NAME, ...) \
    case at::ScalarType::Float: { using TYPE_NAME = float; __VA_ARGS__; break; }

#define _DISPATCH_CASE_FP64(TYPE, TYPE_NAME, ...) \
    case at::ScalarType::Double: { using TYPE_NAME = double; __VA_ARGS__; break; }

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(DTYPE, TYPE_NAME, ...) \
    [&] { \
        switch (DTYPE) { \
            _DISPATCH_CASE_F16(DTYPE, TYPE_NAME, return __VA_ARGS__) \
            _DISPATCH_CASE_FP32(DTYPE, TYPE_NAME, return __VA_ARGS__) \
            default: { \
                AT_ERROR(\"Unsupported dtype for FP16 dispatch: \", DTYPE); \
            } \
        } \
        return torch::Tensor(); \
    }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(DTYPE, TYPE_NAME, ...) \
    [&] { \
        switch (DTYPE) { \
            case at::ScalarType::Char: { using TYPE_NAME = int8_t; return __VA_ARGS__; } \
            case at::ScalarType::Byte: { using TYPE_NAME = uint8_t; return __VA_ARGS__; } \
            default: { \
                AT_ERROR(\"Unsupported dtype for FP8 dispatch: \", DTYPE); \
            } \
        } \
        return torch::Tensor(); \
    }()

inline bool check_shape(const torch::Tensor& x, std::vector<int64_t> shape) {
    if (x.dim() != shape.size()) {
        return false;
    }
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] != -1 && shape[i] != x.size(i)) {
            return false;
        }
    }
    return true;
}

inline bool check_same_shape(const torch::Tensor& x, const torch::Tensor& y) {
    if (x.dim() != y.dim()) {
        return false;
    }
    for (int i = 0; i < x.dim(); i++) {
        if (x.size(i) != y.size(i)) {
            return false;
        }
    }
    return true;
}

// Get the current CUDA stream
inline cudaStream_t torch_current_stream() {
    return at::cuda::getCurrentCUDAStream();
}
")

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

# Create Python extension module.
pybind11_add_module(sgl_kernel_core ${SOURCES})
target_link_libraries(sgl_kernel_core PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)

# Set rpath for the extension module.
set_target_properties(sgl_kernel_core PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../torch/lib"
)

# Install target.
install(TARGETS sgl_kernel_core DESTINATION .)
