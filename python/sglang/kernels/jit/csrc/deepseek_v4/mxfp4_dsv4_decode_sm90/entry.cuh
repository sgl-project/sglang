/*!
 * \brief TVM-FFI entry point for the fused MXFP4 DeepSeek-V4 sparse decode.
 *
 * Wraps the FlashMLA-style three-stage split-KV decode (scheduler metadata
 * kernel, persistent main kernel, combine kernel) so it can be JIT-compiled
 * as a single translation unit. All output and scratch tensors are allocated
 * on the Python side; this entry only parses shapes, fills the parameter
 * structs, and launches.
 *
 * The kernel sources under this directory are vendored from:
 *   - the SGLang reference PR #31269 (sparse_nvfp4_dsv4 + dequant/layout),
 *   - FlashMLA upstream @ 05e26647 (params.h, defines.h, utils.h, kerutils/,
 *     and the combine kernel),
 * with the MXFP4 (E8M0 block-32 scale) variant developed for DSV4 on Hopper.
 * See the PR body for the full provenance list.
 */

#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "combine.cuh"
#include "config.h"
#include "splitkv_mla.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sm90::decode::sparse_mxfp4_dsv4 {

// Split-K scheduler constants, shared by the metadata kernel and the
// host-side dispatch.
constexpr int kTopkBlockSize = 64;
constexpr int kFixedOverheadNumBlocks = 5;

// Explicit instantiations for the two supported query-head counts.
template void run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel_impl<64>(const SparseAttnDecodeParams& params);
template void run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel_impl<128>(const SparseAttnDecodeParams& params);

namespace {

__device__ __forceinline__ int clamp_length(const int* lengths, int request_idx, int width) {
  const int value = lengths == nullptr ? width : __ldg(lengths + request_idx);
  return max(0, min(value, width));
}

__device__ __forceinline__ int effective_length(const GetDecodeSchedMetaParams& params, int request_idx) {
  int primary = clamp_length(params.topk_length, request_idx, params.topk);
  // Keep one all-masked primary block for a zero-length request, matching the
  // FlashMLA consumer's progress invariant.
  primary = max(primary, 1);
  if (params.extra_topk > 0) {
    primary = ((primary + kTopkBlockSize - 1) / kTopkBlockSize) * kTopkBlockSize;
    primary += clamp_length(params.extra_topk_length, request_idx, params.extra_topk);
  }
  return primary;
}

__global__ void get_dsv4_mxfp4_decoding_sched_meta_kernel(__grid_constant__ const GetDecodeSchedMetaParams params) {
  if (threadIdx.x != 0) {
    return;
  }

  int total_num_blocks = 0;
  for (int request_idx = 0; request_idx < params.b; ++request_idx) {
    const int length = effective_length(params, request_idx);
    const int num_blocks = (length + kTopkBlockSize - 1) / kTopkBlockSize;
    total_num_blocks += num_blocks + kFixedOverheadNumBlocks;
  }

  const int payload = (total_num_blocks + params.num_sm_parts - 1) / params.num_sm_parts + kFixedOverheadNumBlocks;
  int request_idx = 0;
  int block_idx = 0;
  int request_split_idx = 0;
  int cumulative_num_splits = 0;
  params.num_splits_ptr[0] = 0;

  for (int part = 0; part < params.num_sm_parts; ++part) {
    if (request_idx >= params.b) {
      DecodingSchedMeta invalid = {};
      invalid.begin_req_idx = params.b;
      invalid.end_req_idx = params.b;
      params.tile_scheduler_metadata_ptr[part] = invalid;
      continue;
    }

    DecodingSchedMeta metadata = {};
    metadata.begin_req_idx = request_idx;
    metadata.begin_block_idx = block_idx;
    metadata.begin_split_idx = request_split_idx;
    metadata.is_first_req_splitted = block_idx != 0;

    int remaining_payload = payload;
    while (request_idx < params.b) {
      const int length = effective_length(params, request_idx);
      const int num_blocks = (length + kTopkBlockSize - 1) / kTopkBlockSize;
      const int remaining_blocks = num_blocks - block_idx;
      if (remaining_payload >= remaining_blocks + kFixedOverheadNumBlocks) {
        cumulative_num_splits += request_split_idx + 1;
        params.num_splits_ptr[request_idx + 1] = cumulative_num_splits;
        remaining_payload -= remaining_blocks + kFixedOverheadNumBlocks;
        ++request_idx;
        block_idx = 0;
        request_split_idx = 0;
      } else {
        if (remaining_payload > kFixedOverheadNumBlocks) {
          block_idx += remaining_payload - kFixedOverheadNumBlocks;
          ++request_split_idx;
        }
        break;
      }
    }

    metadata.end_req_idx = block_idx > 0 ? request_idx : request_idx - 1;
    if (block_idx > 0) {
      metadata.end_block_idx = block_idx;
      const int length = effective_length(params, metadata.end_req_idx);
      const int last_block_idx = (length + kTopkBlockSize - 1) / kTopkBlockSize - 1;
      metadata.is_last_req_splitted = metadata.end_block_idx != last_block_idx + 1;
    } else {
      const int length = effective_length(params, metadata.end_req_idx);
      metadata.end_block_idx = (length + kTopkBlockSize - 1) / kTopkBlockSize;
      metadata.is_last_req_splitted = false;
    }
    if (metadata.begin_req_idx == metadata.end_req_idx) {
      const int is_split = metadata.is_first_req_splitted || metadata.is_last_req_splitted;
      metadata.is_first_req_splitted = is_split;
      metadata.is_last_req_splitted = is_split;
    }
    params.tile_scheduler_metadata_ptr[part] = metadata;
  }

  // Requests the partition loop never reached (batch not evenly divisible by
  // the per-part payload) get zero splits so the combine kernel reads a
  // defined prefix sum instead of whatever the buffer held before.
  for (int i = request_idx + 1; i <= params.b; ++i) {
    params.num_splits_ptr[i] = cumulative_num_splits;
  }
}

}  // namespace

void run_get_dsv4_mxfp4_decoding_sched_meta_kernel(const GetDecodeSchedMetaParams& params) {
  get_dsv4_mxfp4_decoding_sched_meta_kernel<<<1, 1, 0, params.stream>>>(params);
  KU_CHECK_KERNEL_LAUNCH();
}

void run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel(const SparseAttnDecodeParams& params) {
  if (params.h_q == 64) {
    run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel_impl<64>(params);
  } else if (params.h_q == 128) {
    run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel_impl<128>(params);
  } else {
    KU_ASSERT(false, "DeepSeek V4 MXFP4 sparse decode supports 64 or 128 query heads");
  }
}

}  // namespace sm90::decode::sparse_mxfp4_dsv4

namespace sglang {

namespace {

constexpr int kHeadDimQk = 512;
constexpr int kHeadDimV = 512;
constexpr int kMxfp4BytesPerToken = 368;

// Optional tensors arrive as empty (0-element) tensors from the Python side.
static inline bool is_empty(tvm::ffi::TensorView t) {
  return t.numel() == 0;
}

// Raise a Python-visible ValueError instead of exiting: this entry runs
// inside a serving process, and a contract violation must surface as an
// exception the caller can handle, not kill the process.
[[noreturn]] static inline void fail(const char* msg) {
  TVM_FFI_THROW(ValueError) << "mxfp4_dsv4_decode: " << msg;
}

}  // namespace

//! \brief Fused DSV4 MXFP4 decode (scheduler + main + combine) for one step.
void mxfp4_dsv4_decode_dispatch(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView topk_length,
    tvm::ffi::TensorView attn_sink,
    tvm::ffi::TensorView tile_scheduler_metadata,
    tvm::ffi::TensorView num_splits,
    tvm::ffi::TensorView extra_k_cache,
    tvm::ffi::TensorView extra_indices,
    tvm::ffi::TensorView extra_topk_length,
    tvm::ffi::TensorView lse_accum,
    tvm::ffi::TensorView o_accum,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView lse,
    int64_t head_dim_v,
    double sm_scale,
    int64_t generate_sched_meta) {
  if (head_dim_v != kHeadDimV) {
    fail("d_v must be 512");
  }
  if (!(sm_scale > 0.0)) {
    fail("sm_scale must be finite and positive");
  }

  DLDevice dev = q.device();
  if (dev.device_type != kDLCUDA) {
    fail("q must be a CUDA tensor");
  }
  cudaSetDevice(dev.device_id);
  // Resolve the torch current stream (which is the capture stream during
  // CUDA-graph capture) on the C++ side; fetching it from Python costs
  // ~100us per call.
  cudaStream_t stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(kDLCUDA, dev.device_id));

  // One-time SM90 gate per device: cudaDeviceGetAttribute is a driver
  // round-trip and would cost microseconds on every decode step. A process
  // may span heterogeneous GPUs, so the cached verdict is keyed by the
  // device it was computed for rather than decided once for the process.
  static int sm90_checked_device = -1;
  static bool is_sm90 = false;
  if (sm90_checked_device != static_cast<int>(dev.device_id)) {
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev.device_id);
    is_sm90 = (major == 9);
    sm90_checked_device = static_cast<int>(dev.device_id);
  }
  if (!is_sm90) {
    fail("mxfp4_dsv4_decode only supports SM90 (Hopper)");
  }

  const int b = static_cast<int>(q.shape()[0]);
  const int s_q = static_cast<int>(q.shape()[1]);
  const int h_q = static_cast<int>(q.shape()[2]);
  const int topk = static_cast<int>(indices.shape()[2]);
  if (b <= 0 || s_q <= 0) {
    fail("q batch and sequence dimensions must be positive");
  }
  if (h_q != 64 && h_q != 128) {
    fail("q must contain 64 or 128 heads");
  }

  const bool have_extra = !is_empty(extra_k_cache);
  const int extra_num_blocks = have_extra ? static_cast<int>(extra_k_cache.shape()[0]) : 0;
  const int extra_page_block_size = have_extra ? static_cast<int>(extra_k_cache.shape()[1]) : 0;
  const int extra_topk = have_extra ? static_cast<int>(extra_indices.shape()[2]) : 0;

  const int num_sm_parts = static_cast<int>(tile_scheduler_metadata.shape()[0]);

  // The scheduler metadata kernel runs only when the caller signals a fresh
  // metadata buffer (first use, or a new buffer captured into a CUDA graph).
  // Replays and steady-state decode steps reuse the previously generated
  // metadata, matching the AOT behavior and avoiding a per-step serial
  // 1-CTA kernel. During graph capture the caller hands a fresh buffer, so
  // the generation kernel is captured into the graph and re-executes on
  // every replay with the replayed (clamped) top-k lengths.
  GetDecodeSchedMetaParams sched_params = {};
  if (generate_sched_meta) {
    sched_params.b = b;
    sched_params.s_q = s_q;
    sched_params.block_size_n = sm90::decode::sparse_mxfp4_dsv4::kTopkBlockSize;
    sched_params.fixed_overhead_num_blocks = sm90::decode::sparse_mxfp4_dsv4::kFixedOverheadNumBlocks;
    sched_params.topk = topk;
    sched_params.extra_topk = have_extra ? extra_topk : 0;
    sched_params.topk_length = is_empty(topk_length) ? nullptr : static_cast<int*>(topk_length.data_ptr());
    sched_params.extra_topk_length =
        is_empty(extra_topk_length) ? nullptr : static_cast<int*>(extra_topk_length.data_ptr());
    sched_params.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata.data_ptr());
    sched_params.num_splits_ptr = static_cast<int*>(num_splits.data_ptr());
    sched_params.num_sm_parts = num_sm_parts;
    sched_params.stream = stream;
    sm90::decode::sparse_mxfp4_dsv4::run_get_dsv4_mxfp4_decoding_sched_meta_kernel(sched_params);
  }

  const int num_blocks = static_cast<int>(k_cache.shape()[0]);
  const int page_block_size = static_cast<int>(k_cache.shape()[1]);

  SparseAttnDecodeParams params = {};
  params.b = b;
  params.s_q = s_q;
  params.h_q = h_q;
  params.h_kv = 1;
  params.d_qk = kHeadDimQk;
  params.d_v = kHeadDimV;
  params.sm_scale = static_cast<float>(sm_scale);
  params.sm_scale_div_log2 = static_cast<float>(sm_scale) * M_LOG2E;
  params.num_blocks = num_blocks;
  params.page_block_size = page_block_size;
  params.topk = topk;
  params.model_type = ModelType::MODEL1;

  params.q = reinterpret_cast<cutlass::bfloat16_t*>(q.data_ptr());
  params.kv = reinterpret_cast<cutlass::bfloat16_t*>(k_cache.data_ptr());
  params.indices = static_cast<int*>(indices.data_ptr());
  params.topk_length = is_empty(topk_length) ? nullptr : static_cast<int*>(topk_length.data_ptr());
  params.attn_sink = is_empty(attn_sink) ? nullptr : static_cast<float*>(attn_sink.data_ptr());
  params.lse = static_cast<float*>(lse.data_ptr());
  params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());

  params.extra_num_blocks = extra_num_blocks;
  params.extra_page_block_size = extra_page_block_size;
  params.extra_topk = extra_topk;
  params.extra_kv = have_extra ? reinterpret_cast<cutlass::bfloat16_t*>(extra_k_cache.data_ptr()) : nullptr;
  params.extra_indices = have_extra ? static_cast<int*>(extra_indices.data_ptr()) : nullptr;
  params.extra_topk_length = is_empty(extra_topk_length) ? nullptr : static_cast<int*>(extra_topk_length.data_ptr());

  params.stride_q_b = h_q * kHeadDimQk;
  params.stride_q_s_q = h_q * kHeadDimQk;
  params.stride_q_h_q = kHeadDimQk;
  params.stride_kv_block = page_block_size * kMxfp4BytesPerToken;
  params.stride_kv_row = kMxfp4BytesPerToken;
  params.stride_indices_b = s_q * topk;
  params.stride_indices_s_q = topk;
  params.stride_lse_b = s_q * h_q;
  params.stride_lse_s_q = h_q;
  params.stride_o_b = h_q * kHeadDimV;
  params.stride_o_s_q = h_q * kHeadDimV;
  params.stride_o_h_q = kHeadDimV;
  params.stride_extra_kv_block = have_extra ? extra_page_block_size * kMxfp4BytesPerToken : 0;
  params.stride_extra_kv_row = kMxfp4BytesPerToken;
  params.stride_extra_indices_b = have_extra ? s_q * extra_topk : 0;
  params.stride_extra_indices_s_q = have_extra ? extra_topk : 0;
  params.stream = stream;

  params.lse_accum = static_cast<float*>(lse_accum.data_ptr());
  params.o_accum = static_cast<float*>(o_accum.data_ptr());
  params.stride_lse_accum_split = s_q * h_q;
  params.stride_lse_accum_s_q = h_q;
  params.stride_o_accum_split = s_q * h_q * kHeadDimV;
  params.stride_o_accum_s_q = h_q * kHeadDimV;
  params.stride_o_accum_h_q = kHeadDimV;
  params.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata.data_ptr());
  params.num_splits_ptr = static_cast<int*>(num_splits.data_ptr());
  params.num_sm_parts = num_sm_parts;

  sm90::decode::sparse_mxfp4_dsv4::run_flash_splitkv_mla_mxfp4_dsv4_sparse_kernel(params);

  CombineParams combine_params = {};
  combine_params.b = b;
  combine_params.s_q = s_q;
  combine_params.h_q = h_q;
  combine_params.d_v = kHeadDimV;
  combine_params.lse = params.lse;
  combine_params.out = params.out;
  combine_params.stride_lse_b = params.stride_lse_b;
  combine_params.stride_lse_s_q = params.stride_lse_s_q;
  combine_params.stride_o_b = params.stride_o_b;
  combine_params.stride_o_s_q = params.stride_o_s_q;
  combine_params.stride_o_h_q = params.stride_o_h_q;
  combine_params.lse_accum = params.lse_accum;
  combine_params.o_accum = params.o_accum;
  combine_params.stride_lse_accum_split = params.stride_lse_accum_split;
  combine_params.stride_lse_accum_s_q = params.stride_lse_accum_s_q;
  combine_params.stride_o_accum_split = params.stride_o_accum_split;
  combine_params.stride_o_accum_s_q = params.stride_o_accum_s_q;
  combine_params.stride_o_accum_h_q = params.stride_o_accum_h_q;
  combine_params.tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
  combine_params.num_splits_ptr = params.num_splits_ptr;
  combine_params.num_sm_parts = params.num_sm_parts;
  combine_params.attn_sink = params.attn_sink;
  combine_params.stream = params.stream;
  smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(combine_params);
}

}  // namespace sglang
