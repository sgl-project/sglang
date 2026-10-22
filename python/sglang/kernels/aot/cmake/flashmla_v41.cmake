if(NOT FLASHMLA_V41_SOURCE_DIR)
    FetchContent_Declare(
        repo-flashmla-v41
        URL https://${GITHUB_ARTIFACTORY}/deepseek-ai/FlashMLA/archive/2e5429fc5653bab6e081f09477126f731882a6a9.tar.gz
    )
    FetchContent_Populate(repo-flashmla-v41)
    FetchContent_Declare(
        repo-flashmla-v41-cutlass
        URL https://${GITHUB_ARTIFACTORY}/NVIDIA/cutlass/archive/4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b.tar.gz
        SOURCE_DIR ${repo-flashmla-v41_SOURCE_DIR}/csrc/3rdparty/cutlass
    )
    FetchContent_Populate(repo-flashmla-v41-cutlass)
    set(FLASHMLA_V41_SOURCE_DIR ${repo-flashmla-v41_SOURCE_DIR})
endif()

set(_FMLA41 ${FLASHMLA_V41_SOURCE_DIR}/csrc)
set(_FMLA41_K ${_FMLA41}/cuda_kernels/sm100)
set(_FMLA41_CORE ${_FMLA41_K}/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd)

set(FLASHMLA_V41_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/../csrc/flashmla_v41_mega_extension.cc
    ${CMAKE_CURRENT_LIST_DIR}/../csrc/flashmla_v41_decode_extension.cc

    ${_FMLA41}/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
    ${_FMLA41}/cuda_kernels/smxx/decode/combine/combine.cu

    ${_FMLA41_K}/decode/sparse/head64/instantiations/v41_h64.cu
    ${_FMLA41_K}/decode/sparse/head64/instantiations/v41_h64_no_split.cu
    ${_FMLA41_K}/decode/sparse/head64/instantiations/v41fp4_h64.cu
    ${_FMLA41_K}/decode/sparse/head64/instantiations/v41fp4_h64_no_split.cu
    ${_FMLA41_K}/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41.cu
    ${_FMLA41_K}/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41_splitkv.cu
    ${_FMLA41_K}/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4.cu
    ${_FMLA41_K}/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4_splitkv.cu

    ${_FMLA41_CORE}/permute_q_b_proj/kernel.cu
    ${_FMLA41_CORE}/permute_wv_proj/kernel.cu
)
foreach(HEADS 64 128)
    foreach(NORM norm nonorm)
        list(APPEND FLASHMLA_V41_SOURCES
            ${_FMLA41_CORE}/core_attn/instantiations/v41_h${HEADS}_prefill_${NORM}.cu
            ${_FMLA41_CORE}/core_attn/instantiations/v41_h${HEADS}_decode_${NORM}.cu
            ${_FMLA41_CORE}/core_attn/instantiations/v41fp4_h${HEADS}_decode_${NORM}.cu
        )
    endforeach()
endforeach()

set(FLASHMLA_V41_CUDA_FLAGS
    "-gencode=arch=compute_100a,code=sm_100a"
    "-gencode=arch=compute_103a,code=sm_103a"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--use_fast_math"
    "--ptxas-options=--register-usage-level=10"
    "-lineinfo"
    "--diag-suppress=445"
    "-Xcompiler=-fvisibility=hidden"
    "-U__CUDA_NO_HALF_OPERATORS__"
    "-U__CUDA_NO_HALF_CONVERSIONS__"
    "-U__CUDA_NO_HALF2_OPERATORS__"
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
)

if(NOT FLASHMLA_V41_TORCH_INCLUDE_DIRS)
    set(FLASHMLA_V41_TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
endif()
if(NOT FLASHMLA_V41_TORCH_LIBRARIES)
    set(FLASHMLA_V41_TORCH_LIBRARIES ${TORCH_LIBRARIES})
endif()
if(NOT FLASHMLA_V41_TORCH_PYTHON_LIBRARY)
    find_library(FLASHMLA_V41_TORCH_PYTHON_LIBRARY torch_python HINTS ${TORCH_INSTALL_PREFIX}/lib)
endif()
if(NOT FLASHMLA_V41_PYTHON_INCLUDE_DIRS)
    set(FLASHMLA_V41_PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
endif()

add_library(flashmla_v41_ops SHARED ${FLASHMLA_V41_SOURCES})
set_target_properties(flashmla_v41_ops PROPERTIES
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(flashmla_v41_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++20 -O3 -Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20 -O3 ${FLASHMLA_V41_CUDA_FLAGS}>
)
target_compile_definitions(flashmla_v41_ops PRIVATE
    FLASH_MLA_IS_BUILD_ON_CUDA KERUTILS_IS_BUILD_ON_CUDA NDEBUG
    TORCH_API_INCLUDE_EXTENSION_H TORCH_EXTENSION_NAME=flashmla_v41_ops
)
set(_FMLA41_CCCL_HINTS)
foreach(_dir IN LISTS CUDAToolkit_INCLUDE_DIRS CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    list(APPEND _FMLA41_CCCL_HINTS ${_dir} ${_dir}/cccl)
endforeach()
find_path(FLASHMLA_V41_CCCL_INCLUDE NAMES cuda/std/utility
    HINTS ${_FMLA41_CCCL_HINTS} ${CUDAToolkit_TARGET_DIR}/include/cccl ${CUDAToolkit_LIBRARY_ROOT}/include/cccl)
if(NOT FLASHMLA_V41_CCCL_INCLUDE)
    message(FATAL_ERROR "flashmla_v41_ops: CCCL headers (cuda/std/utility) not found under the CUDA toolkit include directories")
endif()
target_include_directories(flashmla_v41_ops PRIVATE
    ${_FMLA41}
    ${_FMLA41}/3rdparty/cutlass/include
    ${_FMLA41}/3rdparty/cutlass/tools/util/include
    ${_FMLA41}/3rdparty/kerutils/include
    ${FLASHMLA_V41_CCCL_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${FLASHMLA_V41_TORCH_INCLUDE_DIRS}
    ${FLASHMLA_V41_PYTHON_INCLUDE_DIRS}
)
target_link_options(flashmla_v41_ops PRIVATE "-Wl,-Bsymbolic" "-Wl,--exclude-libs,ALL")
target_link_libraries(flashmla_v41_ops PRIVATE ${FLASHMLA_V41_TORCH_LIBRARIES} ${FLASHMLA_V41_TORCH_PYTHON_LIBRARY} cuda)
