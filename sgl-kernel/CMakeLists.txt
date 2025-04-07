cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(sgl-kernel LANGUAGES CXX CUDA)

cmake_policy(SET CMP0169 OLD)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(BUILD_FA3, OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

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
# clean Torch Flag
clear_cuda_arches(CMAKE_FLAG)

include(FetchContent)

# cutlass
FetchContent_Declare(
    repo-cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass
    GIT_TAG        6f4921858b3bb0a82d7cbeb4e499690e9ae60d16
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-cutlass)
# DeepGEMM
FetchContent_Declare(
    repo-deepgemm
    GIT_REPOSITORY https://github.com/deepseek-ai/DeepGEMM
    GIT_TAG        c187c23ba8dcdbad91720737e8be9c43700cb9e9
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-deepgemm)
# flashinfer
FetchContent_Declare(
    repo-flashinfer
    GIT_REPOSITORY https://github.com/sgl-project/flashinfer
    GIT_TAG        sgl-kernel
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-flashinfer)
# flash-attention
FetchContent_Declare(
    repo-flash-attention
    GIT_REPOSITORY https://github.com/sgl-project/sgl-attn
    GIT_TAG        sgl-kernel
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-flash-attention)

# ccache option
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
    message(STATUS "Building with CCACHE enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
    ${repo-flashinfer_SOURCE_DIR}/include
    ${repo-flashinfer_SOURCE_DIR}/csrc
    ${repo-flash-attention_SOURCE_DIR}/hopper
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(SGL_KERNEL_CUDA_FLAGS
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
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "--expt-relaxed-constexpr"
    "-Xcompiler=-Wconversion"
    "-Xcompiler=-fno-strict-aliasing"
)

option(SGL_KERNEL_ENABLE_SM100A "Enable SM100A" OFF)
option(SGL_KERNEL_ENABLE_SM90A  "Enable SM90A"  OFF)
option(SGL_KERNEL_ENABLE_BF16   "Enable BF16"   ON)
option(SGL_KERNEL_ENABLE_FP8    "Enable FP8"    ON)
option(SGL_KERNEL_ENABLE_FP4    "Enable FP4"    OFF)

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_100,code=sm_100"
        "-gencode=arch=compute_100a,code=sm_100a"
    )
else()
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-use_fast_math"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR SGL_KERNEL_ENABLE_SM90A)
    set(BUILD_FA3 ON)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()

if (SGL_KERNEL_ENABLE_BF16)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-DFLASHINFER_ENABLE_BF16"
    )
endif()

if (SGL_KERNEL_ENABLE_FP8)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-DFLASHINFER_ENABLE_FP8"
        "-DFLASHINFER_ENABLE_FP8_E4M3"
        "-DFLASHINFER_ENABLE_FP8_E5M2"
    )
endif()

if (SGL_KERNEL_ENABLE_FP4)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-DENABLE_NVFP4=1"
    )
endif()

string(REPLACE "-D__CUDA_NO_HALF_OPERATORS__"       "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF_CONVERSIONS__"     "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF2_OPERATORS__"      "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(SOURCES
    "csrc/allreduce/custom_all_reduce.cu"
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
    "csrc/moe/moe_fused_gate.cu"
    "csrc/moe/moe_topk_softmax_kernels.cu"
    "csrc/speculative/eagle_utils.cu"
    "csrc/speculative/speculative_sampling.cu"
    "csrc/speculative/packbit.cu"
    "csrc/common_extension.cc"
    "${repo-flashinfer_SOURCE_DIR}/csrc/norm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/renorm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/sampling.cu"
)

# set flash-attention sources file
# BF16 source files
if (BUILD_FA3)
    set(SGL_FLASH_KERNEL_CUDA_FLAGS
        "-DNDEBUG"
        "-DOPERATOR_NAMESPACE=sgl-kernel"
        "-O3"
        "-Xcompiler"
        "-fPIC"
        "-gencode=arch=compute_90a,code=sm_90a"
        "-std=c++17"
        "-DCUTE_USE_PACKED_TUPLE=1"
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
        "-DCUTLASS_VERSIONS_GENERATED"
        "-DCUTLASS_TEST_LEVEL=0"
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
        "--expt-relaxed-constexpr"
        "--expt-extended-lambda"
        "--use_fast_math"
        "-Xcompiler=-Wconversion"
        "-Xcompiler=-fno-strict-aliasing"
    )

    file(GLOB FA3_BF16_GEN_SRCS
        "${repo-flash-attention_SOURCE_DIR}/hopper/instantiations/flash_fwd_hdimall_bf16*_sm90.cu")
    file(GLOB FA3_BF16_GEN_SRCS_
        "${repo-flash-attention_SOURCE_DIR}/hopper/instantiations/flash_fwd_hdimdiff_bf16*_sm90.cu")
    list(APPEND FA3_BF16_GEN_SRCS ${FA3_BF16_GEN_SRCS_})

    # FP16 source files
    file(GLOB FA3_FP16_GEN_SRCS
        "${repo-flash-attention_SOURCE_DIR}/hopper/instantiations/flash_fwd_hdimall_fp16*_sm90.cu")
    file(GLOB FA3_FP16_GEN_SRCS_
        "${repo-flash-attention_SOURCE_DIR}/hopper/instantiations/flash_fwd_hdimdiff_fp16*_sm90.cu")
    list(APPEND FA3_FP16_GEN_SRCS ${FA3_FP16_GEN_SRCS_})

    # FP8 source files
    file(GLOB FA3_FP8_GEN_SRCS
        "${repo-flash-attention_SOURCE_DIR}/hopper/instantiations/flash_fwd_hdimall_e4m3*_sm90.cu")
    file(GLOB FA3_FP8_GEN_SRCS_
        "${repo-flash-attention_SOURCE_DIR}/hopper/instantiations/flash_fwd_hdimdiff_e4m3*_sm90.cu")
    list(APPEND FA3_FP8_GEN_SRCS ${FA3_FP8_GEN_SRCS_})

    set(FA3_GEN_SRCS ${FA3_BF16_GEN_SRCS} ${FA3_FP16_GEN_SRCS} ${FA3_FP8_GEN_SRCS})

    set(FLASH_SOURCES
        "csrc/flash_extension.cc"
        "${repo-flash-attention_SOURCE_DIR}/hopper/flash_prepare_scheduler.cu"
        "${repo-flash-attention_SOURCE_DIR}/hopper/flash_api.cpp"
        "${repo-flash-attention_SOURCE_DIR}/hopper/flash_fwd_combine.cu"
        "${FA3_GEN_SRCS}"
    )

    Python_add_library(flash_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${FLASH_SOURCES})

    target_compile_options(flash_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SGL_FLASH_KERNEL_CUDA_FLAGS}>)
    target_include_directories(flash_ops PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(flash_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)

    install(TARGETS flash_ops LIBRARY DESTINATION "sgl_kernel")

    target_compile_definitions(flash_ops PRIVATE
        FLASHATTENTION_DISABLE_SM8x
        FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_DROPOUT
        # FLASHATTENTION_DISABLE_ALIBI
        # FLASHATTENTION_DISABLE_SOFTCAP
        FLASHATTENTION_DISABLE_UNEVEN_K
        # FLASHATTENTION_DISABLE_LOCAL
        FLASHATTENTION_VARLEN_ONLY
    )
endif()

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS}>)
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)

install(TARGETS common_ops LIBRARY DESTINATION "sgl_kernel")

# JIT Logic
# DeepGEMM

install(DIRECTORY "${repo-deepgemm_SOURCE_DIR}/deep_gemm/"
        DESTINATION "deep_gemm"
        PATTERN ".git*" EXCLUDE
        PATTERN "__pycache__" EXCLUDE)

install(DIRECTORY "${repo-cutlass_SOURCE_DIR}/include/cute/"
        DESTINATION "deep_gemm/include/cute")

install(DIRECTORY "${repo-cutlass_SOURCE_DIR}/include/cutlass/"
        DESTINATION "deep_gemm/include/cutlass")
