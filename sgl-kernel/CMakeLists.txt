cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(sgl-kernel LANGUAGES CXX CUDA)

# utils
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(FetchContent)

# CMake
cmake_policy(SET CMP0169 OLD)
cmake_policy(SET CMP0177 NEW)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Python
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "Detected CUDA_VERSION=${CUDA_VERSION}")
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
    message("CUDA_VERSION ${CUDA_VERSION} >= 13.0")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.8")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.4")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.1")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.1")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "11.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 11.8")
endif()

# Torch
find_package(Torch REQUIRED)
clear_cuda_arches(CMAKE_FLAG)

# Third Party repos
# cutlass
FetchContent_Declare(
    repo-cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass
    GIT_TAG        57e3cfb47a2d9e0d46eb6335c3dc411498efa198
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-cutlass)

# Triton kernel
FetchContent_Declare(
    repo-triton
    GIT_REPOSITORY "https://github.com/triton-lang/triton"
    GIT_TAG        v3.5.1
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-triton)

# flashinfer
FetchContent_Declare(
    repo-flashinfer
    GIT_REPOSITORY https://github.com/flashinfer-ai/flashinfer.git
    GIT_TAG        bc29697ba20b7e6bdb728ded98f04788e16ee021
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-flashinfer)

# mscclpp
FetchContent_Declare(
    repo-mscclpp
    GIT_REPOSITORY https://github.com/microsoft/mscclpp.git
    GIT_TAG        51eca89d20f0cfb3764ccd764338d7b22cd486a6
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-mscclpp)

# fast-hadamard-transform
FetchContent_Declare(
    repo-fast-hadamard-transform
    GIT_REPOSITORY https://github.com/sgl-project/fast-hadamard-transform.git
    GIT_TAG 48f3c13764dc2ec662ade842a4696a90a137f1bc
    GIT_SHALLOW OFF
)
FetchContent_Populate(repo-fast-hadamard-transform)

# ccache option
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
    message(STATUS "Building with CCACHE enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()

# Configure gencode below SM90
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(DEFAULT_ENABLE_BELOW_SM90 OFF)
    message(STATUS "For aarch64, disable gencode below SM90 by default")
else()
    set(DEFAULT_ENABLE_BELOW_SM90 ON)
endif()

option(ENABLE_BELOW_SM90 "Enable gencode below SM90" ${DEFAULT_ENABLE_BELOW_SM90})

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
)

set(SGL_KERNEL_CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=sgl-kernel"
    "-O3"
    "-Xcompiler"
    "-fPIC"
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
    "--expt-extended-lambda"

    # The following flag leads to the CMAKE_BUILD_PARALLEL_LEVEL breaking,
    # it triggers OOM with low memory host. Extract the threads number to
    # option named SGL_KERNEL_COMPILE_THREADS, default value 32.
    # "--threads=32"

    # Supress warnings
    "-Xcompiler=-Wno-clang-format-violations"
    "-Xcompiler=-Wno-conversion"
    "-Xcompiler=-Wno-deprecated-declarations"
    "-Xcompiler=-Wno-terminate"
    "-Xcompiler=-Wfatal-errors"
    "-Xcompiler=-ftemplate-backtrace-limit=1"
    "-Xcudafe=--diag_suppress=177"   # variable was declared but never referenced
    "-Xcudafe=--diag_suppress=2361"  # invalid narrowing conversion from "char" to "signed char"

    # uncomment to debug
    # "--ptxas-options=-v"
    # "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage"
)

set(SGL_KERNEL_COMPILE_THREADS 32 CACHE STRING "Set compilation threads, default 32")

# When SGL_KERNEL_COMPILE_THREADS value is less than 1, set it to 1
if (NOT SGL_KERNEL_COMPILE_THREADS MATCHES "^[0-9]+$")
    message(FATAL_ERROR "SGL_KERNEL_COMPILE_THREADS must be an integer, but was set to '${SGL_KERNEL_COMPILE_THREADS}'.")
elseif (SGL_KERNEL_COMPILE_THREADS LESS 1)
    message(STATUS "SGL_KERNEL_COMPILE_THREADS was set to a value less than 1. Using 1 instead.")
    set(SGL_KERNEL_COMPILE_THREADS 1)
endif()

list(APPEND SGL_KERNEL_CUDA_FLAGS
    "--threads=${SGL_KERNEL_COMPILE_THREADS}"
)

option(SGL_KERNEL_ENABLE_BF16             "Enable BF16"             ON)
option(SGL_KERNEL_ENABLE_FP8              "Enable FP8"              ON)
option(SGL_KERNEL_ENABLE_FP4              "Enable FP4"              OFF)
option(SGL_KERNEL_ENABLE_FA3              "Enable FA3"              OFF)
option(SGL_KERNEL_ENABLE_SM90A            "Enable SM90A"            OFF)
option(SGL_KERNEL_ENABLE_SM100A           "Enable SM100A"           OFF)

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

if (ENABLE_BELOW_SM90)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_80,code=sm_80"
        "-gencode=arch=compute_89,code=sm_89"
    )
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        list(APPEND SGL_KERNEL_CUDA_FLAGS
            "-gencode=arch=compute_87,code=sm_87"
        )
    endif()

endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_FP4)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-DENABLE_NVFP4=1"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_100a,code=sm_100a"
        "-gencode=arch=compute_120a,code=sm_120a"
    )
    # refer sm_121, sm_110 and sm_101 description  https://github.com/pytorch/pytorch/pull/156176
    if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
        list(APPEND SGL_KERNEL_CUDA_FLAGS
            "-gencode=arch=compute_103a,code=sm_103a"
            "--compress-mode=size"
        )
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            list(APPEND SGL_KERNEL_CUDA_FLAGS
                "-gencode=arch=compute_110a,code=sm_110a"
                "-gencode=arch=compute_121a,code=sm_121a"
            )
        endif()
    else()
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            list(APPEND SGL_KERNEL_CUDA_FLAGS
                "-gencode=arch=compute_101a,code=sm_101a"
            )
        endif()
    endif()
endif()


# All source files
# NOTE: Please sort the filenames alphabetically
set(SOURCES
    "csrc/allreduce/custom_all_reduce.cu"
    "csrc/allreduce/mscclpp_allreduce.cu"
    "csrc/attention/cascade.cu"
    "csrc/attention/cutlass_mla_kernel.cu"
    "csrc/attention/merge_attn_states.cu"
    "csrc/attention/vertical_slash_index.cu"
    "csrc/common_extension.cc"
    "csrc/elementwise/activation.cu"
    "csrc/elementwise/cast.cu"
    "csrc/elementwise/concat_mla.cu"
    "csrc/elementwise/copy.cu"
    "csrc/elementwise/fused_add_rms_norm_kernel.cu"
    "csrc/elementwise/rope.cu"
    "csrc/elementwise/pos_enc.cu"
    "csrc/elementwise/topk.cu"
    "csrc/sgl_diffusion/elementwise/timestep_embedding.cu"
    "csrc/expert_specialization/es_fp8_blockwise.cu"
    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu"
    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cu"

    "csrc/gemm/awq_kernel.cu"
    "csrc/gemm/bmm_fp8.cu"
    "csrc/gemm/dsv3_fused_a_gemm.cu"
    "csrc/gemm/dsv3_router_gemm_bf16_out.cu"
    "csrc/gemm/dsv3_router_gemm_entry.cu"
    "csrc/gemm/dsv3_router_gemm_float_out.cu"
    "csrc/gemm/fp8_blockwise_gemm_kernel.cu"
    "csrc/gemm/fp8_gemm_kernel.cu"
    "csrc/gemm/int8_gemm_kernel.cu"
    "csrc/gemm/nvfp4_expert_quant.cu"
    "csrc/gemm/nvfp4_quant_entry.cu"
    "csrc/gemm/nvfp4_quant_kernels.cu"
    "csrc/gemm/nvfp4_scaled_mm_entry.cu"
    "csrc/gemm/nvfp4_scaled_mm_kernels.cu"
    "csrc/gemm/per_tensor_quant_fp8.cu"
    "csrc/gemm/per_token_group_quant_8bit.cu"
    "csrc/gemm/per_token_group_quant_8bit_v2.cu"
    "csrc/gemm/per_token_quant_fp8.cu"
    "csrc/gemm/qserve_w4a8_per_chn_gemm.cu"
    "csrc/gemm/qserve_w4a8_per_group_gemm.cu"
    "csrc/gemm/marlin/gptq_marlin.cu"
    "csrc/gemm/marlin/gptq_marlin_repack.cu"
    "csrc/gemm/marlin/awq_marlin_repack.cu"
    "csrc/gemm/gptq/gptq_kernel.cu"
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu"

    "csrc/kvcacheio/transfer.cu"
    "csrc/mamba/causal_conv1d.cu"
    "csrc/memory/store.cu"
    "csrc/memory/weak_ref_tensor.cpp"

    "csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu"
    "csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu"
    "csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu"
    "csrc/moe/marlin_moe_wna16/ops.cu"
    "csrc/moe/moe_align_kernel.cu"
    "csrc/moe/moe_fused_gate.cu"
    "csrc/moe/fused_qknorm_rope_kernel.cu"
    "csrc/moe/kimi_k2_moe_fused_gate.cu"
    "csrc/moe/moe_sum.cu"
    "csrc/moe/moe_sum_reduce.cu"
    "csrc/moe/moe_topk_softmax_kernels.cu"
    "csrc/moe/moe_topk_sigmoid_kernels.cu"
    "csrc/moe/nvfp4_blockwise_moe.cu"
    "csrc/moe/fp8_blockwise_moe_kernel.cu"
    "csrc/moe/prepare_moe_input.cu"

    "csrc/quantization/gguf/gguf_kernel.cu"
    "csrc/speculative/eagle_utils.cu"
    "csrc/speculative/ngram_utils.cu"
    "csrc/speculative/packbit.cu"
    "csrc/speculative/speculative_sampling.cu"

    "${repo-flashinfer_SOURCE_DIR}/csrc/norm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/renorm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/sampling.cu"

    "${repo-fast-hadamard-transform_SOURCE_DIR}/csrc/fast_hadamard_transform_cuda.cu"
    "${repo-fast-hadamard-transform_SOURCE_DIR}/csrc/fast_hadamard_transform.cpp"

)

set(INCLUDES
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
    ${repo-flashinfer_SOURCE_DIR}/include
    ${repo-flashinfer_SOURCE_DIR}/csrc
    ${repo-mscclpp_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/examples/77_blackwell_fmha
    ${repo-cutlass_SOURCE_DIR}/examples/common
)

# =========================== Common SM90 Build ============================= #
# Build SM90 library with fast math optimization (same namespace, different directory)
Python_add_library(common_ops_sm90_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

target_compile_options(common_ops_sm90_build PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS} -use_fast_math>
)
target_include_directories(common_ops_sm90_build PRIVATE ${INCLUDES})
# Set output name and separate build directory to avoid conflicts
set_target_properties(common_ops_sm90_build PROPERTIES
    OUTPUT_NAME "common_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm90"
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "0")
    message(STATUS "Using old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
    message(STATUS "Using new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
endif()

# mscclpp option
set(MSCCLPP_USE_CUDA ON)
set(MSCCLPP_BYPASS_GPU_CHECK ON)
set(MSCCLPP_BUILD_TESTS OFF)
add_subdirectory(
    ${repo-mscclpp_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/mscclpp-build
)

target_link_libraries(common_ops_sm90_build PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt mscclpp_static)

install(TARGETS common_ops_sm90_build LIBRARY DESTINATION sgl_kernel/sm90)

# ============================ Extra Install: triton kernels ============================= #
install(DIRECTORY "${repo-triton_SOURCE_DIR}/python/triton_kernels/triton_kernels/"
        DESTINATION "triton_kernels"
        PATTERN ".git*" EXCLUDE
        PATTERN "__pycache__" EXCLUDE)
