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

namespace moe {

__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(mask, v, offset);
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size,
    int32_t max_num_tokens_padded) {
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
    int expert_id = topk_ids[i] + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }

  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

#ifndef __CUDA_ARCH__  // HIP

  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }

  __syncthreads();

  // Blelloch scan
  int offset = 1;
#pragma unroll
  for (int d = scan_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      scan_buf[bi] += scan_buf[ai];
    }
    offset <<= 1;
    __syncthreads();
  }

  // down-sweep
  if (tid == 0) {
    prefix[num_experts] = scan_buf[scan_size - 1];
    scan_buf[scan_size - 1] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int d = 1; d < scan_size; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      if (bi < scan_size) {
        int temp = scan_buf[ai];
        scan_buf[ai] = scan_buf[bi];
        scan_buf[bi] += temp;
      }
    }
    __syncthreads();
  }

  if (tid < num_experts) {
    prefix[tid] = scan_buf[tid];
  }

  if (tid == 0) {
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

#else  // CUDA

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
#endif

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
    expert_ids[i] = left - 2;
  }
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
  // Adapted from
  // https://github.com/vllm-project/vllm/pull/29642/files#diff-5647b1413f4ae9aacba904eca8f8a8aee9079321eadff4c10101a2c6962dcc53R226
  // Use an additional group of threads to fill sorted_token_ids.
  // Since the kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    if (pad_sorted_token_ids) {
      for (int32_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
        sorted_token_ids[it] = numel;
      }
    }
    // Three __syncthreads() corresponding to the other threads
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i] + 1;
    ++tokens_cnts[(tid + 1) * num_experts + expert_id];
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid - 1;
    }
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[tid * num_experts + expert_id];
  }
}

// v2 kernel: supports >1024 experts via EXPERTS_PER_THREAD templating
// and a two-level warp scan (no cub dependency). Uses the same +1 offset
// convention as the original kernel (topk_ids shifted by +1 so -1 maps to 0).
// Launched with <<<2, 1024>>>: block 1 fills sorted_token_ids in parallel
// with block 0 doing the alignment compute.
//
// With 1024 threads and EXPERTS_PER_THREAD=4, covers at most 4096 expert
// indices. Since num_experts includes the +1 offset bucket, this supports
// up to 4095 real experts.
template <typename scalar_t, int EXPERTS_PER_THREAD>
__global__ void moe_align_block_size_kernel_v2(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t padded_num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
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
  // Layout: shared_counts[padded_num_experts] | warp_sums[WARP_SIZE]
  int32_t* shared_counts = smem;
  int32_t* warp_sums = smem + padded_num_experts;

  const size_t tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);

  // Phase 1: Zero shared counts and count tokens per expert
  const int my_start = tid * EXPERTS_PER_THREAD;
  for (size_t i = tid; i < padded_num_experts; i += blockDim.x) {
    shared_counts[i] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += blockDim.x) {
    int expert_id = topk_ids[i] + 1;  // +1 offset convention
    if (expert_id < num_experts) {
      atomicAdd(&shared_counts[expert_id], 1);
    }
  }

  __syncthreads();

  // Phase 2: Compute padded counts and two-level warp exclusive prefix sum
  int32_t local_padded[EXPERTS_PER_THREAD];
  int32_t thread_sum = 0;
  for (int i = 0; i < EXPERTS_PER_THREAD; ++i) {
    int eid = my_start + i;
    if (eid < num_experts) {
      local_padded[i] = CEILDIV(shared_counts[eid], block_size) * block_size;
    } else {
      local_padded[i] = 0;
    }
    thread_sum += local_padded[i];
  }

  // Level 1: intra-warp exclusive scan on thread_sum
  int32_t warp_prefix = warp_exclusive_scan(thread_sum);
  int32_t warp_total = warp_prefix + thread_sum;
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_total;
  __syncthreads();

  // Level 2: warp 0 scans the per-warp totals
  const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps) ? warp_sums[tid] : 0;
    warp_sums[tid] = warp_exclusive_scan(val);
  }
  __syncthreads();

  // Combine: thread_prefix = warp_sums[warp_id] + warp_prefix
  int32_t thread_prefix = warp_sums[warp_id] + warp_prefix;

  // Local sequential prefix sum within each thread's expert group
  int32_t running = 0;
  for (int i = 0; i < EXPERTS_PER_THREAD; ++i) {
    int eid = my_start + i;
    if (eid <= num_experts) {
      cumsum[eid] = thread_prefix + running;
    }
    running += local_padded[i];
  }

  // Last thread writes total
  if (tid == blockDim.x - 1) {
    cumsum[num_experts] = thread_prefix + thread_sum;
    *total_tokens_post_pad = thread_prefix + thread_sum;
  }

  __syncthreads();

  // Phase 3: Fill expert_ids (eid - 1 to match sgl-kernel convention)
  for (int i = 0; i < EXPERTS_PER_THREAD; ++i) {
    int eid = my_start + i;
    if (eid < num_experts) {
      for (int j = cumsum[eid]; j < cumsum[eid + 1]; j += block_size) {
        expert_ids[j / block_size] = eid - 1;
      }
    }
  }
}

}  // namespace moe

namespace {

template <typename scalar_t>
struct MoeAlignBlockSizeKernel {
  static void
  run(tvm::ffi::TensorView topk_ids,
      int64_t num_experts,
      int64_t block_size,
      tvm::ffi::TensorView sorted_token_ids,
      tvm::ffi::TensorView expert_ids,
      tvm::ffi::TensorView num_tokens_post_pad,
      tvm::ffi::TensorView cumsum_buffer,
      bool pad_sorted_token_ids) {
    using namespace host;

    auto device = topk_ids.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);

    int threads = 1024;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    int64_t max_num_tokens_padded = sorted_token_ids.size(0);

    // num_experts from Python is actual_num_experts + 1 (for EP offset convention).
    // The v2 kernel (>1024 experts) uses 1024 threads with EXPERTS_PER_THREAD up
    // to 8, covering at most 8192 expert indices. This supports up to 8191 real
    // experts, sufficient for LoRA virtual experts (num_moe_experts * max_loras).
    RuntimeCheck(num_experts <= 8192, "moe_align_block_size: num_experts must be <= 8192, got ", num_experts);

    const scalar_t* topk_ids_ptr = static_cast<const scalar_t*>(topk_ids.data_ptr());
    int32_t* sorted_token_ids_ptr = static_cast<int32_t*>(sorted_token_ids.data_ptr());
    int32_t* expert_ids_ptr = static_cast<int32_t*>(expert_ids.data_ptr());
    int32_t* num_tokens_post_pad_ptr = static_cast<int32_t*>(num_tokens_post_pad.data_ptr());
    int32_t* cumsum_buffer_ptr = static_cast<int32_t*>(cumsum_buffer.data_ptr());
    size_t numel = topk_ids.numel();

    bool small_batch_expert_mode = (numel < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode) {
      const int32_t num_thread = std::max((int32_t)num_experts, (int32_t)WARP_SIZE);
      constexpr int32_t fill_threads = 256;
      const int32_t shared_mem_size = ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

      auto kernel = moe::moe_align_block_size_small_batch_expert_kernel<scalar_t, fill_threads>;
      LaunchKernel(dim3(1), dim3(fill_threads + num_thread), stream, shared_mem_size)(
          kernel,
          topk_ids_ptr,
          sorted_token_ids_ptr,
          expert_ids_ptr,
          num_tokens_post_pad_ptr,
          (int32_t)num_experts,
          (int32_t)block_size,
          numel,
          pad_sorted_token_ids,
          (int32_t)max_num_tokens_padded);
    } else if (num_experts <= 1024) {
      const size_t scan_size = next_pow2(num_experts);
      const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size + WARP_SIZE) * sizeof(int32_t);

      auto align_kernel = moe::moe_align_block_size_kernel<scalar_t>;
      LaunchKernel(dim3(2), dim3(threads), stream, shared_mem_size)(
          align_kernel,
          topk_ids_ptr,
          sorted_token_ids_ptr,
          expert_ids_ptr,
          num_tokens_post_pad_ptr,
          (int32_t)num_experts,
          (int32_t)block_size,
          numel,
          cumsum_buffer_ptr,
          pad_sorted_token_ids,
          (int32_t)scan_size,
          (int32_t)max_num_tokens_padded);

      const int block_threads = std::min(256, threads);
      const int num_blocks = (numel + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      auto sort_kernel = moe::count_and_sort_expert_tokens_kernel<scalar_t>;
      LaunchKernel(dim3(actual_blocks), dim3(block_threads), stream)(
          sort_kernel, topk_ids_ptr, sorted_token_ids_ptr, cumsum_buffer_ptr, numel);
    } else {
      // v2 path for >1024 experts: two-level warp scan with EXPERTS_PER_THREAD
      int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
      size_t shared_mem_size = (padded_num_experts + WARP_SIZE) * sizeof(int32_t);

      auto launch_v2 = [&](auto ept_tag) {
        constexpr int EPT = decltype(ept_tag)::value;
        auto v2_kernel = moe::moe_align_block_size_kernel_v2<scalar_t, EPT>;
        LaunchKernel(dim3(2), dim3(threads), stream, shared_mem_size)(
            v2_kernel,
            topk_ids_ptr,
            sorted_token_ids_ptr,
            expert_ids_ptr,
            num_tokens_post_pad_ptr,
            (int32_t)num_experts,
            (int32_t)padded_num_experts,
            (int32_t)block_size,
            numel,
            cumsum_buffer_ptr,
            pad_sorted_token_ids,
            (int32_t)max_num_tokens_padded);
      };

      if (padded_num_experts <= 2048) {
        launch_v2(std::integral_constant<int, 2>{});
      } else if (padded_num_experts <= 4096) {
        launch_v2(std::integral_constant<int, 4>{});
      } else {
        launch_v2(std::integral_constant<int, 8>{});
      }

      const int block_threads = std::min(256, threads);
      const int num_blocks = (numel + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      auto sort_kernel = moe::count_and_sort_expert_tokens_kernel<scalar_t>;
      LaunchKernel(dim3(actual_blocks), dim3(block_threads), stream)(
          sort_kernel, topk_ids_ptr, sorted_token_ids_ptr, cumsum_buffer_ptr, numel);
    }
  }
};

}  // namespace
