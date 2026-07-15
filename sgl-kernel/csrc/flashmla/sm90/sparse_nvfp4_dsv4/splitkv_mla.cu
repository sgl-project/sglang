#include <kerutils/kerutils.cuh>

#include "splitkv_mla.h"

namespace sm90::decode::sparse_nvfp4_dsv4 {

namespace {

constexpr int kTopkBlockSize = 64;
constexpr int kFixedOverheadNumBlocks = 5;

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

__global__ void get_dsv4_nvfp4_decoding_sched_meta_kernel(__grid_constant__ const GetDecodeSchedMetaParams params) {
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
}

}  // namespace

void run_get_dsv4_nvfp4_decoding_sched_meta_kernel(const GetDecodeSchedMetaParams& params) {
  get_dsv4_nvfp4_decoding_sched_meta_kernel<<<1, 1, 0, params.stream>>>(params);
  KU_CHECK_KERNEL_LAUNCH();
}

void run_flash_splitkv_mla_nvfp4_dsv4_sparse_kernel(
    const SparseAttnDecodeParams& params, const float* kv_global_scale, const float* extra_kv_global_scale) {
  if (params.h_q == 64) {
    run_flash_splitkv_mla_nvfp4_dsv4_sparse_kernel_impl<64>(params, kv_global_scale, extra_kv_global_scale);
  } else if (params.h_q == 128) {
    run_flash_splitkv_mla_nvfp4_dsv4_sparse_kernel_impl<128>(params, kv_global_scale, extra_kv_global_scale);
  } else {
    KU_ASSERT(false, "DeepSeek V4 NVFP4 sparse decode supports 64 or 128 query heads");
  }
}

}  // namespace sm90::decode::sparse_nvfp4_dsv4
