cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(sgl-kernel LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

message(STATUS "Detected CUDA_VERSION=${CUDA_VERSION}")
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.8")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.4")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.1")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.1")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "11.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 11.8")
endif()

find_package(Torch REQUIRED)

include(FetchContent)

FetchContent_Declare(
    repo-cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass
    GIT_TAG        62750a2b75c802660e4894434dc55e839f322277
    GIT_SHALLOW    ON
)
FetchContent_MakeAvailable(repo-cutlass)

FetchContent_Declare(
    repo-deepgemm
    GIT_REPOSITORY https://github.com/deepseek-ai/DeepGEMM
    GIT_TAG        c57699ac933a93651c34d365797c2d8b41a4765b
    GIT_SHALLOW    ON
)
FetchContent_MakeAvailable(repo-deepgemm)

FetchContent_Declare(
    repo-flashinfer
    GIT_REPOSITORY https://github.com/flashinfer-ai/flashinfer
    GIT_TAG        e5a3befbe3e63025f0158bc96b218a9c5f402ac7
    GIT_SHALLOW    OFF
)
FetchContent_MakeAvailable(repo-flashinfer)

FetchContent_Declare(
    repo-flash-attention
    GIT_REPOSITORY https://github.com/Dao-AILab/flash-attention
    GIT_TAG        2f9ef0879a0935c3ca852f7a6a7b7a9c24f41e96
    GIT_SHALLOW    OFF
)
FetchContent_MakeAvailable(repo-flash-attention)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
    ${repo-flashinfer_SOURCE_DIR}/include
    ${repo-flashinfer_SOURCE_DIR}/csrc
    # ${repo-flash-attention_SOURCE_DIR}/hopper
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CUDA_NVCC_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=sgl-kernel"
    "-O3"
    "-Xcompiler"
    "-fPIC"
    "-gencode=arch=compute_75,code=sm_75"
    "-gencode=arch=compute_80,code=sm_80"
    "-gencode=arch=compute_89,code=sm_89"
    "-gencode=arch=compute_90,code=sm_90"
    "-std=c++17"
    "-DFLASHINFER_ENABLE_F16"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "--expt-relaxed-constexpr"
    "-Xcompiler=-Wconversion"
    "-Xcompiler=-fno-strict-aliasing"
)

option(SGL_KERNEL_ENABLE_SM100A "Enable SM100A" OFF)
option(SGL_KERNEL_ENABLE_SM90A  "Enable SM90A" OFF)
option(SGL_KERNEL_ENABLE_BF16   "Enable BF16"   ON)
option(SGL_KERNEL_ENABLE_FP8    "Enable FP8"    ON)

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)
    list(APPEND CUDA_NVCC_FLAGS
        "-gencode=arch=compute_100,code=sm_100"
        "-gencode=arch=compute_100a,code=sm_100a"
    )
else()
    list(APPEND CUDA_NVCC_FLAGS
        "-use_fast_math"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR SGL_KERNEL_ENABLE_SM90A)
    list(APPEND CUDA_NVCC_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()

if (SGL_KERNEL_ENABLE_BF16)
    list(APPEND CUDA_NVCC_FLAGS
        "-DFLASHINFER_ENABLE_BF16"
    )
endif()

if (SGL_KERNEL_ENABLE_FP8)
    list(APPEND CUDA_NVCC_FLAGS
        "-DFLASHINFER_ENABLE_FP8"
        "-DFLASHINFER_ENABLE_FP8_E4M3"
        "-DFLASHINFER_ENABLE_FP8_E5M2"
    )
endif()

string(REPLACE "-D__CUDA_NO_HALF_OPERATORS__"       "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF_CONVERSIONS__"     "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF2_OPERATORS__"      "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(SOURCES
    "csrc/allreduce/trt_reduce_internal.cu"
    "csrc/allreduce/trt_reduce_kernel.cu"
    "csrc/attention/lightning_attention_decode_kernel.cu"
    "csrc/elementwise/activation.cu"
    "csrc/elementwise/fused_add_rms_norm_kernel.cu"
    "csrc/elementwise/rope.cu"
    "csrc/gemm/awq_kernel.cu"
    "csrc/gemm/bmm_fp8.cu"
    "csrc/gemm/cublas_grouped_gemm.cu"
    "csrc/gemm/fp8_blockwise_gemm_kernel.cu"
    "csrc/gemm/fp8_gemm_kernel.cu"
    "csrc/gemm/int8_gemm_kernel.cu"
    "csrc/gemm/nvfp4_quant_entry.cu"
    "csrc/gemm/nvfp4_quant_kernels.cu"
    "csrc/gemm/nvfp4_scaled_mm_entry.cu"
    "csrc/gemm/nvfp4_scaled_mm_kernels.cu"
    "csrc/gemm/per_tensor_quant_fp8.cu"
    "csrc/gemm/per_token_group_quant_8bit.cu"
    "csrc/gemm/per_token_quant_fp8.cu"
    "csrc/moe/moe_align_kernel.cu"
    "csrc/moe/moe_topk_softmax_kernels.cu"
    "csrc/speculative/eagle_utils.cu"
    "csrc/speculative/speculative_sampling.cu"
    "csrc/speculative/packbit.cu"
    "csrc/torch_extension.cc"
    "${repo-flashinfer_SOURCE_DIR}/csrc/norm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/renorm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/sampling.cu"
)

Python_add_library(common_ops MODULE ${SOURCES} WITH_SOABI)

target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)

install(TARGETS common_ops LIBRARY DESTINATION "sgl_kernel")
