#pragma once

#include "params.h"

namespace sm90::decode::sparse_mxfp4_dsv4 {

// Launch the DeepSeek V4 SM90 sparse decode kernel. The primary and optional
// extra caches retain SparseAttnDecodeParams' independent runtime page sizes,
// capacities, indices, and dynamic top-k lengths.
void run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel(const SparseAttnDecodeParams& params);

// Generate split-K metadata with device-side length clamping. The upstream
// scheduler assumes every topk_length is already within the corresponding
// indices width; this private entry point makes that safety property explicit
// for graph-replayed DSV4 inputs.
void run_get_dsv4_mxfp4_decoding_sched_meta_kernel(const GetDecodeSchedMetaParams& params);

template <int NUM_HEADS>
void run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel_impl(const SparseAttnDecodeParams& params);

}  // namespace sm90::decode::sparse_mxfp4_dsv4
