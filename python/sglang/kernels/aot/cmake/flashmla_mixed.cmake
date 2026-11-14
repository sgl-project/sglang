# SM90 mixed decode for the versioned packed DSV4.1 Main-KV layout.
FetchContent_Declare(
    repo-flashmla-mixed
    URL      https://${GITHUB_ARTIFACTORY}/yuyu5333/FlashMLA/archive/4960de014f73eb4475b5302b7c36ec2083a01d4a.tar.gz
    URL_HASH SHA256=c8186f3acc0fe1d8ff3ba4fb7c1003714dab084cf24e6ce5303af807e144f18f
)
FetchContent_Populate(repo-flashmla-mixed)

FetchContent_Declare(
    repo-flashmla-mixed-cutlass
    URL      https://${GITHUB_ARTIFACTORY}/NVIDIA/cutlass/archive/147f5673d0c1c3dcf66f78d677fd647e4a020219.tar.gz
    URL_HASH SHA256=9f6c53320a85b4a570975e557918cde65168cd311f081920446c238437347dc6
    SOURCE_DIR ${repo-flashmla-mixed_SOURCE_DIR}/csrc/cutlass
)
FetchContent_Populate(repo-flashmla-mixed-cutlass)

set(FLASHMLA_MIXED_SOURCES
    "csrc/flashmla_mixed_extension.cc"
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/api/sparse_decode.cpp
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/smxx/decode/combine/combine.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v4_dsv41_main_fp4_persistent_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v4_dsv41_main_fp4_persistent_h128.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v4_persistent_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v4_persistent_h128.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v32_persistent_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v32_persistent_h128.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v32_no_rope_persistent_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v32_no_rope_persistent_h128.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v32_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v32_h64_no_split.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v32_no_rope_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v32_no_rope_h64_no_split.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v4_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v4_h64_no_split.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41_h64_no_split.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64_no_split.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_splitkv.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41_splitkv.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4.cu
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4_splitkv.cu
)

Python_add_library(
    flashmla_mixed_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
    ${FLASHMLA_MIXED_SOURCES}
)
target_compile_options(flashmla_mixed_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>
)
target_include_directories(flashmla_mixed_ops PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/kerutils/include
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/cutlass/include
    ${repo-flashmla-mixed_SOURCE_DIR}/csrc/cutlass/tools/util/include
)
target_compile_definitions(flashmla_mixed_ops PRIVATE FLASH_MLA_LIBTORCH_ONLY)
target_link_libraries(flashmla_mixed_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)
install(TARGETS flashmla_mixed_ops LIBRARY DESTINATION "sgl_kernel")
