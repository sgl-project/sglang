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

# DeepGEMM
FetchContent_Declare(
    repo-deepgemm
    GIT_REPOSITORY https://github.com/sgl-project/DeepGEMM
    GIT_TAG        ffe2b6b97420a9f8c58268ca55755168e6e2f360
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-deepgemm)

# fmt
FetchContent_Declare(
    repo-fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG        553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-fmt)

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

# flash-attention
FetchContent_Declare(
    repo-flash-attention
    GIT_REPOSITORY https://github.com/sgl-project/sgl-attn
    GIT_TAG        f20a52329482ddca4a627b2f028f88c2959ee299
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-flash-attention)

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

# Enable gencode below SM90
option(ENABLE_BELOW_SM90 "Enable below SM90" ON)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ENABLE_BELOW_SM90 OFF)
    message(STATUS "For aarch64, disable gencode below SM90 by default")
endif()

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

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    set(SGL_KERNEL_ENABLE_FA3 ON)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_FP4)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-DENABLE_NVFP4=1"
    )
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

    "${repo-flash-attention_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_sparse_hdim128_bf16_causal_sm80.cu"
    "${repo-flash-attention_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_sparse_hdim128_bf16_sm80.cu"
    "${repo-flash-attention_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_sparse_hdim128_fp16_causal_sm80.cu"
    "${repo-flash-attention_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_sparse_hdim128_fp16_sm80.cu"
    "${repo-flash-attention_SOURCE_DIR}/csrc/flash_attn/flash_sparse_api.cpp"
)

set(INCLUDES
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
    ${repo-flashinfer_SOURCE_DIR}/include
    ${repo-flashinfer_SOURCE_DIR}/csrc
    ${repo-mscclpp_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/examples/77_blackwell_fmha
    ${repo-cutlass_SOURCE_DIR}/examples/common
    ${repo-flash-attention_SOURCE_DIR}/csrc/flash_attn/src
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

# =========================== Common SM100+ Build ============================= #
# Build SM100+ library with precise math (same namespace, different directory)
Python_add_library(common_ops_sm100_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

target_compile_options(common_ops_sm100_build PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS}>
)
target_include_directories(common_ops_sm100_build PRIVATE ${INCLUDES})
# Set output name and separate build directory to avoid conflicts
set_target_properties(common_ops_sm100_build PROPERTIES
    OUTPUT_NAME "common_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm100"
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
target_link_libraries(common_ops_sm100_build PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt mscclpp_static)

# sparse flash attention
target_compile_definitions(common_ops_sm90_build PRIVATE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_UNEVEN_K
)
target_compile_definitions(common_ops_sm100_build PRIVATE
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_UNEVEN_K
)

# Install to different subdirectories
# CMake will find the built libraries in their respective LIBRARY_OUTPUT_DIRECTORY locations
# and install them to the specified destinations
install(TARGETS common_ops_sm90_build LIBRARY DESTINATION sgl_kernel/sm90)
install(TARGETS common_ops_sm100_build LIBRARY DESTINATION sgl_kernel/sm100)

# ============================ Optional Install: FA3 ============================= #
# set flash-attention sources file
# Now FA3 support sm80/sm86/sm90
if (SGL_KERNEL_ENABLE_FA3)
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

    if (ENABLE_BELOW_SM90)
        list(APPEND SGL_FLASH_KERNEL_CUDA_FLAGS
            "-gencode=arch=compute_80,code=sm_80"
            "-gencode=arch=compute_86,code=sm_86"
        )
        # SM8X Logic
        file(GLOB FA3_SM8X_GEN_SRCS
            "${repo-flash-attention_SOURCE_DIR}/hopper/instantiations/flash_fwd_hdim*_sm80.cu")
    endif()

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

    set(FA3_GEN_SRCS ${FA3_BF16_GEN_SRCS} ${FA3_FP16_GEN_SRCS} ${FA3_FP8_GEN_SRCS} ${FA3_SM8X_GEN_SRCS})

    set(FLASH_SOURCES
        "csrc/flash_extension.cc"
        "${repo-flash-attention_SOURCE_DIR}/hopper/flash_prepare_scheduler.cu"
        "${repo-flash-attention_SOURCE_DIR}/hopper/flash_api.cpp"
        "${repo-flash-attention_SOURCE_DIR}/hopper/flash_fwd_combine.cu"
        "${FA3_GEN_SRCS}"
    )

    Python_add_library(flash_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${FLASH_SOURCES})

    target_compile_options(flash_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SGL_FLASH_KERNEL_CUDA_FLAGS}>)
    target_include_directories(flash_ops PRIVATE
        ${repo-cutlass_SOURCE_DIR}/include
        ${repo-cutlass_SOURCE_DIR}/tools/util/include
        ${repo-flash-attention_SOURCE_DIR}/hopper
    )
    target_link_libraries(flash_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)

    install(TARGETS flash_ops LIBRARY DESTINATION "sgl_kernel")
    set(FLASH_OPS_COMPILE_DEFS
        FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_DROPOUT
        FLASHATTENTION_DISABLE_UNEVEN_K
        FLASHATTENTION_VARLEN_ONLY
    )

    if(NOT ENABLE_BELOW_SM90)
        list(APPEND FLASH_OPS_COMPILE_DEFS FLASHATTENTION_DISABLE_SM8x)
    endif()
    target_compile_definitions(flash_ops PRIVATE ${FLASH_OPS_COMPILE_DEFS})
endif()

# Build spatial_ops as a separate, optional extension for green contexts
set(SPATIAL_SOURCES
    "csrc/spatial/greenctx_stream.cu"
    "csrc/spatial_extension.cc"
)

Python_add_library(spatial_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SPATIAL_SOURCES})
target_compile_options(spatial_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS}>)
target_link_libraries(spatial_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)
install(TARGETS spatial_ops LIBRARY DESTINATION sgl_kernel)

# ============================ Extra Install: FLashMLA ============================= #
include(${CMAKE_CURRENT_LIST_DIR}/cmake/flashmla.cmake)

# ============================ Extra Install: DeepGEMM (JIT) ============================= #
# Create a separate library for DeepGEMM's Python API.
# This keeps its compilation isolated from the main common_ops.
set(DEEPGEMM_SOURCES
    "${repo-deepgemm_SOURCE_DIR}/csrc/python_api.cpp"
)

Python_add_library(deep_gemm_cpp MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${DEEPGEMM_SOURCES})

# Link against necessary libraries, including nvrtc for JIT compilation.
target_link_libraries(deep_gemm_cpp PRIVATE ${TORCH_LIBRARIES} c10 cuda nvrtc mscclpp_static)

# Add include directories needed by DeepGEMM.
target_include_directories(deep_gemm_cpp PRIVATE
    ${repo-deepgemm_SOURCE_DIR}/deep_gemm/include
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-fmt_SOURCE_DIR}/include
)

# Apply the same compile options as common_ops.
target_compile_options(deep_gemm_cpp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS}>)

# Create an empty __init__.py to make `deepgemm` a Python package.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/deepgemm_pkg_init.py "")
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/deepgemm_pkg_init.py
    DESTINATION deep_gemm
    RENAME __init__.py
)

# Install the compiled DeepGEMM API library.
install(TARGETS deep_gemm_cpp LIBRARY DESTINATION deep_gemm)

# Install the source files required by DeepGEMM for runtime JIT compilation.
install(
    DIRECTORY ${repo-deepgemm_SOURCE_DIR}/deep_gemm/
    DESTINATION deep_gemm
)

install(DIRECTORY "${repo-cutlass_SOURCE_DIR}/include/cute/"
        DESTINATION "deep_gemm/include/cute")

install(DIRECTORY "${repo-cutlass_SOURCE_DIR}/include/cutlass/"
        DESTINATION "deep_gemm/include/cutlass")

# ============================ Extra Install: triton kernels ============================= #
install(DIRECTORY "${repo-triton_SOURCE_DIR}/python/triton_kernels/triton_kernels/"
        DESTINATION "triton_kernels"
        PATTERN ".git*" EXCLUDE
        PATTERN "__pycache__" EXCLUDE)

# ============================ Extra Install: FA4 ============================= #
# TODO: find a better install condition.
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)

    set(FLASH_ATTN_CUTE_SRC "${repo-flash-attention_SOURCE_DIR}/flash_attn/cute")
    set(FLASH_ATTN_CUTE_DST "${CMAKE_CURRENT_BINARY_DIR}/flash_attn_origin/cute")

    file(MAKE_DIRECTORY "${FLASH_ATTN_CUTE_DST}")

    file(COPY "${FLASH_ATTN_CUTE_SRC}/"
         DESTINATION "${FLASH_ATTN_CUTE_DST}"
         PATTERN ".git*" EXCLUDE
         PATTERN "__pycache__" EXCLUDE)

    file(GLOB_RECURSE FLASH_ATTN_CUTE_DST_PY
         "${FLASH_ATTN_CUTE_DST}/*.py")

    foreach(FILE_PATH IN LISTS FLASH_ATTN_CUTE_DST_PY)
        file(READ "${FILE_PATH}" FILE_CONTENT)

        set(MODIFIED_CONTENT "${FILE_CONTENT}")

        # The main goal is to avoid using "flash_attn" so that other libraries (such as transformers) do not mistakenly assume that "flash_attn" is already installed.

        string(REPLACE "flash_attn.cute"
                       "flash_attn_origin.cute"
                       MODIFIED_CONTENT "${MODIFIED_CONTENT}")

        if (NOT FILE_CONTENT STREQUAL MODIFIED_CONTENT)
            file(WRITE "${FILE_PATH}" "${MODIFIED_CONTENT}")
            message(STATUS "  - [FA4 Patch] Patched: ${FILE_PATH}")
        endif()
    endforeach()

    install(DIRECTORY "${FLASH_ATTN_CUTE_DST}/"
            DESTINATION "flash_attn_origin/cute"
            PATTERN ".git*" EXCLUDE
            PATTERN "__pycache__" EXCLUDE)

endif()
