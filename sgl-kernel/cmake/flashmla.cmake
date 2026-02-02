include(FetchContent)

# flash_mla
FetchContent_Declare(
    repo-flashmla
    GIT_REPOSITORY https://github.com/sgl-project/FlashMLA
    GIT_TAG be055fb7df0090fde45f08e9cb5b8b4c0272da73
    GIT_SHALLOW OFF
)
FetchContent_Populate(repo-flashmla)

set(FLASHMLA_CUDA_FLAGS
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--use_fast_math"

    "-Xcudafe=--diag_suppress=177"   # variable was declared but never referenced
)

# The FlashMLA kernels only work on hopper and require CUDA 12.4 or later.
# Only build FlashMLA kernels if we are building for something compatible with
# sm90a
if(${CUDA_VERSION} VERSION_GREATER 12.4)
    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()
if(${CUDA_VERSION} VERSION_GREATER 12.8)
    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_100a,code=sm_100a"
    )
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "13.0")
    # Patch FlashMLA sources for SM103a support.
    # These patches are only needed (and only valid) with CUDA 13+.

    # Patch flashmla_utils.h: widen IS_SM100 to cover the full SM100 family
    set(FLASHMLA_UTILS_FILE "${repo-flashmla_SOURCE_DIR}/csrc/flashmla_utils.h")
    file(READ "${FLASHMLA_UTILS_FILE}" FLASHMLA_UTILS_CONTENT)
    string(REPLACE
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
#define IS_SM100 1"
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)
#define IS_SM100 1"
        FLASHMLA_UTILS_CONTENT "${FLASHMLA_UTILS_CONTENT}")
    file(WRITE "${FLASHMLA_UTILS_FILE}" "${FLASHMLA_UTILS_CONTENT}")
    message(STATUS "Patched flashmla_utils.h for SM103a support")

    # Patch cutlass/arch/config.h: add SM103 architecture defines.
    # The new block is inserted right before the existing "// SM101 and SM101a"
    # anchor in the upstream header.
    set(CUTLASS_CONFIG_FILE "${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include/cutlass/arch/config.h")
    file(READ "${CUTLASS_CONFIG_FILE}" CUTLASS_CONFIG_CONTENT)
    string(FIND "${CUTLASS_CONFIG_CONTENT}" "SM103" SM103_FOUND)
    if(SM103_FOUND EQUAL -1)
        string(REPLACE
"// SM101 and SM101a"
"// SM103 and SM103a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ >= 13)
  #define CUTLASS_ARCH_MMA_SM103_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM103_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030)
    #define CUTLASS_ARCH_MMA_SM103_ENABLED 1
    #if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
    #endif
    #if !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
    #endif
  #endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM101 and SM101a"
            CUTLASS_CONFIG_CONTENT "${CUTLASS_CONFIG_CONTENT}")
        file(WRITE "${CUTLASS_CONFIG_FILE}" "${CUTLASS_CONFIG_CONTENT}")
        message(STATUS "Patched cutlass/arch/config.h for SM103a support")
    else()
        message(STATUS "cutlass/arch/config.h already patched for SM103a")
    endif()

    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_103a,code=sm_103a"
    )
endif()


set(FlashMLA_SOURCES
    "csrc/flashmla_extension.cc"
    ${repo-flashmla_SOURCE_DIR}/csrc/python_api.cpp
    ${repo-flashmla_SOURCE_DIR}/csrc/smxx/get_mla_metadata.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/smxx/mla_combine.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/splitkv_mla.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/splitkv_mla.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd.cu

    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/dense_fp8_python_api.cpp
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu
)

Python_add_library(flashmla_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${FlashMLA_SOURCES})
target_compile_options(flashmla_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FLASHMLA_CUDA_FLAGS}>)
target_include_directories(flashmla_ops PRIVATE
    ${repo-flashmla_SOURCE_DIR}/csrc
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
)

target_link_libraries(flashmla_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)

install(TARGETS flashmla_ops LIBRARY DESTINATION "sgl_kernel")

target_compile_definitions(flashmla_ops PRIVATE)
