/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// LoRA merged-virtual-expert routing align, fused with the virtual-expert id
// computation. Replaces the (_fused_virtual_topk_ids triton kernel + native
// moe_align_block_size) two-launch pair on the `--lora-use-virtual-experts`
// path: the align/scatter kernels read the RAW topk_ids + token_lora_mapping
// and compute the merged virtual id inline (mirrors _fused_virtual_topk_ids),
// so virtual_topk_ids is never materialized to global memory.
//
// Commit 1 scope: pure fusion (inline virtual id), NO EP skip. Output is
// bucket-for-bucket equivalent to the old path (dropped/-1 tokens still land in
// the sentinel bucket 0), so it can be asserted equal to the old kernels.
// Only the `64 < num_buckets <= 1024` branch is implemented here; other expert
// counts keep the old path (handled by the Python dispatcher).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define VEC_SIZE 4
using Vec = int4;

inline uint32_t next_pow2(uint32_t x) noexcept {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

namespace moe_lora_merged {

__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(mask, v, offset);
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;
}

// Inline mirror of _fused_virtual_topk_ids_kernel (virtual_experts.py). Returns
// the merged virtual expert id for flat slot `i` (range [-1, virtual_num_experts);
// -1 is the dropped/masked sentinel). The caller adds +1 to get the histogram
// bucket (sentinel -> bucket 0), matching the native +1 offset convention.
template <typename scalar_t>
__device__ __forceinline__ int compute_virtual_id(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    size_t i,
    int top_k,
    int num_experts_for_weight,
    int local_expert_offset,
    int local_num_experts,
    bool ep_local,
    bool shared_outer,
    bool compact) {
  int m = static_cast<int>(i) / top_k;
  int lora_id = token_lora_mapping[m];
  bool mask_val = lora_id >= 0;
  int safe_lora = lora_id > 0 ? lora_id : 0;

  int base = shared_outer ? 0 : static_cast<int>(topk_ids[i]);
  if (ep_local) {
    bool owned = base >= local_expert_offset && base < local_expert_offset + local_num_experts;
    base = owned ? base : -1;
  }
  if (!mask_val || base < 0) return -1;
  // compact: dense LOCAL expert id in [0, local_num_experts) so the histogram
  // spans only local_num_experts buckets instead of the full global virtual
  // space (337/385 empty under EP). Assumes max_loras==1 (safe_lora shift is 0;
  // the wrapper guards). expert_ids is converted back to global at write time.
  if (compact) return base - local_expert_offset;
  return base + safe_lora * num_experts_for_weight;
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel,
    int top_k,
    int num_experts_for_weight,
    int local_expert_offset,
    int local_num_experts,
    bool ep_local,
    bool shared_outer,
    bool do_skip,
    bool compact) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int vid = compute_virtual_id<scalar_t>(
        topk_ids,
        token_lora_mapping,
        i,
        top_k,
        num_experts_for_weight,
        local_expert_offset,
        local_num_experts,
        ep_local,
        shared_outer,
        compact);
    // EP skip: dropped/masked slots (vid < 0) produce no delta on this rank, so
    // they never need a slot in sorted_token_ids -> skip the global atomicAdd
    // (kills the sentinel-bucket-0 contention). When do_skip is off they fall
    // into bucket 0 (old behavior, kept for the bitwise-equivalence guardrail).
    if (do_skip && vid < 0) continue;
    int32_t expert_id = vid + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    bool* __restrict__ token_lora_mask,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size,
    int32_t max_num_tokens_padded,
    int top_k,
    int num_experts_for_weight,
    int local_expert_offset,
    int local_num_experts,
    bool ep_local,
    bool shared_outer,
    bool do_skip,
    bool compact) {
  // Use a separate thread block to populate sorted_token_ids
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      Vec fill_vec;
      fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
      int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        out_ptr[i] = fill_vec;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                  // [num_experts]
  int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int vid = compute_virtual_id<scalar_t>(
        topk_ids,
        token_lora_mapping,
        i,
        top_k,
        num_experts_for_weight,
        local_expert_offset,
        local_num_experts,
        ep_local,
        shared_outer,
        compact);
    // EP skip: dropped/masked slots don't increment any bucket (sentinel bucket
    // 0 stays empty), so they never get a block and never reach count_and_sort.
    if (!do_skip || vid >= 0) {
      atomicAdd(&shared_counts[vid + 1], 1);
    }
    // token_lora_mask[m] = token_lora_mapping[m] >= 0, written once per row.
    if (static_cast<int>(i) % top_k == 0) {
      int m = static_cast<int>(i) / top_k;
      token_lora_mask[m] = token_lora_mapping[m] >= 0;
    }
  }

  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

  // Intra warp prefix sum
  int32_t* warp_sums = scan_buf + scan_size;  // [<= 32]
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;
  const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;
  __syncthreads();

  // warp0 accumulate all the block's prefix sum
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();

  // Every thread obtains the whole block's sum
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  // Fill 0 to scan_buf extended area (tid >= num_expert)
  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();

  // Perform 2 level exclusive-prefix-sum to scan_buf
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();

  int offset = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + offset;
  __syncthreads();

  // Write prefix[0..num_experts - 1] and cumsum
  if (tid < num_experts) prefix[tid] = scan_buf[tid];

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }
  // fill expert_ids
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    // compact buckets hold LOCAL expert ids; restore the global id (+offset) so
    // the downstream GEMM still indexes the global contiguous LoRA weight.
    expert_ids[i] = left - 2 + (compact ? local_expert_offset : 0);
  }
}

// Single-block fused variant: does fill + histogram + scan + expert_ids + scatter
// in ONE threadblock (one launch), eliminating the separate count_and_sort kernel
// AND its redundant re-computation of the virtual id (cached in shared `svids`).
// Only valid for small numel (the scatter is single-block); the wrapper routes
// large numel (prefill) to the 2-kernel path. do_skip is implied (this is the
// decode hot path); dropped slots are simply never scattered.
template <typename scalar_t>
__global__ void fused_align_scatter_kernel(
    const scalar_t* __restrict__ topk_ids,
    const int32_t* __restrict__ token_lora_mapping,
    bool* __restrict__ token_lora_mask,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    const int32_t scan_size,
    int32_t max_num_tokens_padded,
    int top_k,
    int num_experts_for_weight,
    int local_expert_offset,
    int local_num_experts,
    bool ep_local,
    bool shared_outer,
    bool do_skip,
    bool compact) {
  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                  // [num_experts]
  int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
  int32_t* warp_sums = scan_buf + scan_size;      // [WARP_SIZE]
  int32_t* cursor = warp_sums + WARP_SIZE;        // [num_experts] scatter cursor
  int32_t* svids = cursor + num_experts;          // [numel] cached virtual ids
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;

  // Phase 1: fill sorted_token_ids with the `numel` padding sentinel.
  {
    Vec fill_vec;
    fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
    int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
      out_ptr[i] = fill_vec;
    }
  }
  if (tid < num_experts) shared_counts[tid] = 0;
  __syncthreads();

  // Phase 2: histogram + cache the virtual id per slot + token_lora_mask.
  for (size_t i = tid; i < numel; i += stride) {
    int vid = compute_virtual_id<scalar_t>(
        topk_ids,
        token_lora_mapping,
        i,
        top_k,
        num_experts_for_weight,
        local_expert_offset,
        local_num_experts,
        ep_local,
        shared_outer,
        compact);
    svids[i] = vid;
    if (!do_skip || vid >= 0) {
      atomicAdd(&shared_counts[vid + 1], 1);
    }
    if (static_cast<int>(i) % top_k == 0) {
      int m = static_cast<int>(i) / top_k;
      token_lora_mask[m] = token_lora_mapping[m] >= 0;
    }
  }
  __syncthreads();

  // Phase 3: padded counts + two-level warp exclusive prefix sum (verbatim).
  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }
  const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;
  __syncthreads();
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();
  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();
  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();
  int off = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + off;
  __syncthreads();
  if (tid < num_experts) prefix[tid] = scan_buf[tid];
  if (tid <= num_experts) cumsum[tid] = prefix[tid];
  __syncthreads();

  // Phase 4: expert_ids (binary search per block) + init the scatter cursor.
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2 + (compact ? local_expert_offset : 0);
  }
  if (tid < num_experts) cursor[tid] = prefix[tid];
  __syncthreads();

  // Phase 5: scatter owned tokens using the cached virtual ids + shared cursor.
  for (size_t i = tid; i < numel; i += stride) {
    int vid = svids[i];
    if (do_skip && vid < 0) continue;
    int bucket = vid + 1;
    int pos = atomicAdd(&cursor[bucket], 1);
    sorted_token_ids[pos] = i;
  }
}

}  // namespace moe_lora_merged

namespace {

template <typename scalar_t>
struct MoeLoraMergedAlignKernel {
  static void
  run(tvm::ffi::TensorView topk_ids,
      tvm::ffi::TensorView token_lora_mapping,
      tvm::ffi::TensorView token_lora_mask,
      int64_t num_experts,
      int64_t block_size,
      tvm::ffi::TensorView sorted_token_ids,
      tvm::ffi::TensorView expert_ids,
      tvm::ffi::TensorView num_tokens_post_pad,
      tvm::ffi::TensorView cumsum_buffer,
      bool pad_sorted_token_ids,
      int64_t top_k,
      int64_t num_experts_for_weight,
      int64_t local_expert_offset,
      int64_t local_num_experts,
      bool ep_local,
      bool shared_outer,
      bool do_skip,
      bool compact,
      bool fuse_scatter) {
    using namespace host;

    auto device = topk_ids.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);

    int threads = 1024;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    int64_t max_num_tokens_padded = sorted_token_ids.size(0);

    // num_experts here is the bucket count. Non-compact: virtual_num_experts+1
    // (typically 385). Compact: local_num_experts+1 (typically 49). Both use the
    // same single-block align path (valid for any bucket count <= 1024 that fits
    // shared memory); the v2 (>1024) regime keeps the old path via the wrapper.
    RuntimeCheck(
        num_experts <= 1024, "moe_lora_merged_align: num_experts (bucket count) must be <= 1024, got ", num_experts);
    // compact buckets hold LOCAL ids and restore global expert ids as
    // (left-2+offset). For the sentinel bucket 0 that yields (offset-1), NOT the
    // -1 the GEMM expects to skip -- only safe when do_skip empties bucket 0.
    RuntimeCheck(
        !compact || do_skip, "moe_lora_merged_align: compact requires do_skip (sentinel bucket must be empty)");

    const scalar_t* topk_ids_ptr = static_cast<const scalar_t*>(topk_ids.data_ptr());
    const int32_t* tlm_ptr = static_cast<const int32_t*>(token_lora_mapping.data_ptr());
    bool* token_lora_mask_ptr = static_cast<bool*>(token_lora_mask.data_ptr());
    int32_t* sorted_token_ids_ptr = static_cast<int32_t*>(sorted_token_ids.data_ptr());
    int32_t* expert_ids_ptr = static_cast<int32_t*>(expert_ids.data_ptr());
    int32_t* num_tokens_post_pad_ptr = static_cast<int32_t*>(num_tokens_post_pad.data_ptr());
    int32_t* cumsum_buffer_ptr = static_cast<int32_t*>(cumsum_buffer.data_ptr());
    size_t numel = topk_ids.numel();

    const size_t scan_size = next_pow2(num_experts);

    if (fuse_scatter) {
      // One block does fill + histogram + scan + expert_ids + scatter. Extra
      // shared for the scatter cursor [num_experts] and cached virtual ids [numel].
      const size_t shmem =
          (num_experts + (num_experts + 1) + scan_size + WARP_SIZE + num_experts + numel) * sizeof(int32_t);
      auto fused = moe_lora_merged::fused_align_scatter_kernel<scalar_t>;
      LaunchKernel(dim3(1), dim3(threads), stream, shmem)(
          fused,
          topk_ids_ptr,
          tlm_ptr,
          token_lora_mask_ptr,
          sorted_token_ids_ptr,
          expert_ids_ptr,
          num_tokens_post_pad_ptr,
          (int32_t)num_experts,
          (int32_t)block_size,
          numel,
          cumsum_buffer_ptr,
          (int32_t)scan_size,
          (int32_t)max_num_tokens_padded,
          (int)top_k,
          (int)num_experts_for_weight,
          (int)local_expert_offset,
          (int)local_num_experts,
          ep_local,
          shared_outer,
          do_skip,
          compact);
      return;
    }

    const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size + WARP_SIZE) * sizeof(int32_t);

    auto align_kernel = moe_lora_merged::moe_align_block_size_kernel<scalar_t>;
    LaunchKernel(dim3(2), dim3(threads), stream, shared_mem_size)(
        align_kernel,
        topk_ids_ptr,
        tlm_ptr,
        token_lora_mask_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_pad_ptr,
        (int32_t)num_experts,
        (int32_t)block_size,
        numel,
        cumsum_buffer_ptr,
        pad_sorted_token_ids,
        (int32_t)scan_size,
        (int32_t)max_num_tokens_padded,
        (int)top_k,
        (int)num_experts_for_weight,
        (int)local_expert_offset,
        (int)local_num_experts,
        ep_local,
        shared_outer,
        do_skip,
        compact);

    const int block_threads = std::min(256, threads);
    const int num_blocks = (numel + block_threads - 1) / block_threads;
    const int max_blocks = 65535;
    const int actual_blocks = std::min(num_blocks, max_blocks);

    auto sort_kernel = moe_lora_merged::count_and_sort_expert_tokens_kernel<scalar_t>;
    LaunchKernel(dim3(actual_blocks), dim3(block_threads), stream)(
        sort_kernel,
        topk_ids_ptr,
        tlm_ptr,
        sorted_token_ids_ptr,
        cumsum_buffer_ptr,
        numel,
        (int)top_k,
        (int)num_experts_for_weight,
        (int)local_expert_offset,
        (int)local_num_experts,
        ep_local,
        shared_outer,
        do_skip,
        compact);
  }
};

}  // namespace
