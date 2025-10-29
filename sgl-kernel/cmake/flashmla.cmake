include(FetchContent)

# flash_mla
FetchContent_Declare(
    repo-flashmla
    GIT_REPOSITORY https://github.com/sgl-project/FlashMLA
    GIT_TAG bc8576abc3e507425cf6498f3d3393df7733ce37
    GIT_SHALLOW OFF
)
FetchContent_Populate(repo-flashmla)

set(FLASHMLA_CUDA_FLAGS
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--use_fast_math"
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
)

Python_add_library(flashmla_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${FlashMLA_SOURCES})
target_compile_options(flashmla_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FLASHMLA_CUDA_FLAGS}>)
target_include_directories(flashmla_ops PRIVATE
    ${repo-flashmla_SOURCE_DIR}/csrc
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
)

target_link_libraries(flashmla_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)

install(TARGETS flashmla_ops LIBRARY DESTINATION "sgl_kernel")

target_compile_definitions(flashmla_ops PRIVATE)
