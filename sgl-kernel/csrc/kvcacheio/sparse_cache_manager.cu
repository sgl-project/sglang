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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef USE_ROCM
#define WARP_SIZE 32
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#include "utils.h"
#endif

/**
 * Sparse KV Cache Manager Kernel
 *
 * This kernel manages KV cache for sparse attention, handling:
 * 1. Identifying cache hits (tokens already in GPU hot buffer)
 * 2. Identifying cache misses and eviction candidates
 * 3. Assigning GPU locations for top-k tokens
 * 4. Updating hot buffer metadata for future iterations
 *
 * Parallelism:
 * - Inter-query: Different requests are independent
 * - Intra-query: Tokens within a request processed in parallel with synchronization
 */

// Shared memory structure for coordination
struct SharedState {
  int32_t eviction_ptr;  // Atomic counter for claiming eviction slots
  int32_t num_hits;      // Count of cache hits
  int32_t num_misses;    // Count of cache misses
};

/**
 * Phase 1: Identify hits and mark eviction candidates
 *
 * For each token in top_k:
 *   - If found in hot_buffer → hit, record GPU location
 *   - If not found → miss, will need eviction
 *
 * Also marks which hot buffer slots can be evicted (not in top_k)
 */
__global__ void sparse_cache_identify_hits_kernel(
    const int64_t* __restrict__ top_k_indices,       // [K] tokens we need
    const int64_t* __restrict__ hot_buffer_token_indices,  // [H] tokens in GPU
    const int64_t* __restrict__ hot_buffer_device_locations,  // [H] GPU locations
    int64_t* __restrict__ top_k_device_locations,    // [K] output: GPU locations for top_k
    int32_t* __restrict__ top_k_hit_flags,           // [K] 1 if hit, 0 if miss
    int32_t* __restrict__ hot_buffer_evictable,      // [H] 1 if can be evicted
    int32_t top_k_size,
    int32_t hot_buffer_size) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize evictable flags for hot buffer entries
  // Each thread handles one hot buffer slot
  if (tid < hot_buffer_size) {
    // Assume evictable until proven otherwise
    hot_buffer_evictable[tid] = 1;

    int64_t hot_token = hot_buffer_token_indices[tid];
    // Check if this hot buffer token is in top_k
    for (int32_t i = 0; i < top_k_size; ++i) {
      if (top_k_indices[i] == hot_token) {
        // This token is needed, cannot evict
        hot_buffer_evictable[tid] = 0;
        break;
      }
    }
  }

  // Check hits for top_k tokens
  if (tid < top_k_size) {
    int64_t target_token = top_k_indices[tid];
    int32_t found = 0;
    int64_t device_loc = -1;

    // Search for target_token in hot buffer
    for (int32_t i = 0; i < hot_buffer_size; ++i) {
      if (hot_buffer_token_indices[i] == target_token) {
        found = 1;
        device_loc = hot_buffer_device_locations[i];
        break;
      }
    }

    top_k_hit_flags[tid] = found;
    top_k_device_locations[tid] = device_loc;
  }
}

/**
 * Phase 2: Build eviction candidate list
 *
 * Compacts the evictable hot buffer slots into a contiguous list
 */
__global__ void sparse_cache_build_eviction_list_kernel(
    const int32_t* __restrict__ hot_buffer_evictable,  // [H] evictable flags
    int32_t* __restrict__ eviction_list,               // [E] indices of evictable slots
    int32_t* __restrict__ eviction_count,              // [1] number of evictable slots
    int32_t hot_buffer_size) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < hot_buffer_size && hot_buffer_evictable[tid]) {
    // Atomically add to eviction list
    int32_t pos = atomicAdd(eviction_count, 1);
    eviction_list[pos] = tid;
  }
}

/**
 * Phase 3: Assign eviction slots to misses and prepare copy operations
 *
 * For each miss in top_k, claim an eviction slot and set up the copy
 */
__global__ void sparse_cache_assign_evictions_kernel(
    const int64_t* __restrict__ top_k_indices,           // [K] tokens we need
    const int64_t* __restrict__ cache_cpu_locations,     // [N] CPU locations for all tokens
    int64_t* __restrict__ hot_buffer_token_indices,      // [H] tokens in GPU (modified)
    int64_t* __restrict__ hot_buffer_device_locations,   // [H] GPU locations
    int64_t* __restrict__ top_k_device_locations,        // [K] output: GPU locations
    const int32_t* __restrict__ top_k_hit_flags,         // [K] hit flags
    const int32_t* __restrict__ eviction_list,           // [E] indices of evictable slots
    int64_t* __restrict__ copy_src_cpu_locations,        // [M] CPU locations to copy from
    int64_t* __restrict__ copy_dst_gpu_locations,        // [M] GPU locations to copy to
    int32_t* __restrict__ copy_count,                    // [1] number of copies needed
    int32_t* __restrict__ eviction_ptr,                  // [1] atomic counter for evictions
    int32_t top_k_size) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < top_k_size && !top_k_hit_flags[tid]) {
    // This is a miss, need to claim an eviction slot
    int32_t evict_pos = atomicAdd(eviction_ptr, 1);

    // Get the hot buffer slot index to evict
    int32_t hot_slot = eviction_list[evict_pos];

    // Get the GPU location from evicted slot
    int64_t gpu_loc = hot_buffer_device_locations[hot_slot];

    // Update top_k device location
    top_k_device_locations[tid] = gpu_loc;

    // Get CPU location for the token we need
    int64_t token_idx = top_k_indices[tid];
    int64_t cpu_loc = cache_cpu_locations[token_idx];

    // Add to copy list
    int32_t copy_pos = atomicAdd(copy_count, 1);
    copy_src_cpu_locations[copy_pos] = cpu_loc;
    copy_dst_gpu_locations[copy_pos] = gpu_loc;

    // Update hot buffer metadata
    hot_buffer_token_indices[hot_slot] = token_idx;
    // hot_buffer_device_locations[hot_slot] stays the same (reusing the location)
  }
}

/**
 * Fused single-kernel implementation for small top_k sizes
 *
 * Uses shared memory for coordination to avoid multiple kernel launches.
 * Best for top_k < 256 and hot_buffer_size < 1024
 */
template <int32_t MAX_TOP_K, int32_t MAX_HOT_BUFFER>
__global__ void sparse_cache_manager_fused_kernel(
    const int64_t* __restrict__ top_k_indices,           // [K] tokens we need
    int64_t* __restrict__ hot_buffer_token_indices,      // [H] tokens in GPU (modified)
    int64_t* __restrict__ hot_buffer_device_locations,   // [H] GPU locations
    const int64_t* __restrict__ cache_cpu_locations,     // [N] CPU locations for all tokens
    int64_t* __restrict__ top_k_device_locations,        // [K] output: GPU locations
    int64_t* __restrict__ copy_src_cpu_locations,        // [M] CPU locations to copy from
    int64_t* __restrict__ copy_dst_gpu_locations,        // [M] GPU locations to copy to
    int32_t* __restrict__ copy_count,                    // [1] number of copies needed
    int32_t top_k_size,
    int32_t hot_buffer_size) {
  // Shared memory for coordination
  __shared__ int32_t s_top_k_hit_flags[MAX_TOP_K];
  __shared__ int32_t s_hot_buffer_evictable[MAX_HOT_BUFFER];
  __shared__ int32_t s_eviction_list[MAX_HOT_BUFFER];
  __shared__ int32_t s_eviction_count;
  __shared__ int32_t s_eviction_ptr;
  __shared__ int32_t s_copy_count;

  int32_t tid = threadIdx.x;

  // Initialize shared counters
  if (tid == 0) {
    s_eviction_count = 0;
    s_eviction_ptr = 0;
    s_copy_count = 0;
  }
  __syncthreads();

  // Phase 1a: Initialize evictable flags (assume all evictable initially)
  for (int32_t i = tid; i < hot_buffer_size; i += blockDim.x) {
    s_hot_buffer_evictable[i] = 1;
  }
  __syncthreads();

  // Phase 1b: Check hits for top_k tokens and mark non-evictable hot buffer slots
  for (int32_t k = tid; k < top_k_size; k += blockDim.x) {
    int64_t target_token = top_k_indices[k];
    int32_t found = 0;
    int64_t device_loc = -1;

    // Search for target_token in hot buffer
    for (int32_t h = 0; h < hot_buffer_size; ++h) {
      if (hot_buffer_token_indices[h] == target_token) {
        found = 1;
        device_loc = hot_buffer_device_locations[h];
        // Mark as non-evictable (use atomicExch for safety)
        atomicExch(&s_hot_buffer_evictable[h], 0);
        break;
      }
    }

    s_top_k_hit_flags[k] = found;
    if (found) {
      top_k_device_locations[k] = device_loc;
    }
  }
  __syncthreads();

  // Phase 2: Build eviction list from evictable slots
  for (int32_t h = tid; h < hot_buffer_size; h += blockDim.x) {
    if (s_hot_buffer_evictable[h]) {
      int32_t pos = atomicAdd(&s_eviction_count, 1);
      s_eviction_list[pos] = h;
    }
  }
  __syncthreads();

  // Phase 3: Assign eviction slots to misses
  for (int32_t k = tid; k < top_k_size; k += blockDim.x) {
    if (!s_top_k_hit_flags[k]) {
      // This is a miss, claim an eviction slot
      int32_t evict_pos = atomicAdd(&s_eviction_ptr, 1);

      // Bounds check
      if (evict_pos < s_eviction_count) {
        int32_t hot_slot = s_eviction_list[evict_pos];
        int64_t gpu_loc = hot_buffer_device_locations[hot_slot];

        // Update output
        top_k_device_locations[k] = gpu_loc;

        // Get CPU location for the token
        int64_t token_idx = top_k_indices[k];
        int64_t cpu_loc = cache_cpu_locations[token_idx];

        // Add to copy list
        int32_t copy_pos = atomicAdd(&s_copy_count, 1);
        copy_src_cpu_locations[copy_pos] = cpu_loc;
        copy_dst_gpu_locations[copy_pos] = gpu_loc;

        // Update hot buffer metadata
        hot_buffer_token_indices[hot_slot] = token_idx;
      }
    }
  }
  __syncthreads();

  // Write back copy count
  if (tid == 0) {
    *copy_count = s_copy_count;
  }
}

/**
 * Transfer items from CPU (pinned) to GPU using warp-level parallelism
 */
__device__ __forceinline__ void transfer_item_warp_sparse(
    int32_t lane_id,
    const void* src_addr,
    void* dst_addr,
    int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
#ifndef USE_ROCM
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");
#else
    uint64_t tmp = __builtin_nontemporal_load(src + j);
    __builtin_nontemporal_store(tmp, dst + j);
#endif
  }
}

/**
 * Kernel to perform the actual CPU→GPU copies
 *
 * Each warp handles one copy operation
 */
__global__ void sparse_cache_copy_kernel(
    const void* __restrict__ cpu_cache,              // CPU pinned memory base
    void* __restrict__ gpu_cache,                    // GPU memory base
    const int64_t* __restrict__ copy_src_cpu_locs,   // [M] CPU locations
    const int64_t* __restrict__ copy_dst_gpu_locs,   // [M] GPU locations
    int32_t copy_count,
    int64_t item_size_bytes) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;

  if (warp_id < copy_count) {
    int64_t src_loc = copy_src_cpu_locs[warp_id];
    int64_t dst_loc = copy_dst_gpu_locs[warp_id];

    const char* src = static_cast<const char*>(cpu_cache) + src_loc * item_size_bytes;
    char* dst = static_cast<char*>(gpu_cache) + dst_loc * item_size_bytes;

    transfer_item_warp_sparse(lane_id, src, dst, item_size_bytes);
  }
}

// C++ launcher functions

void sparse_cache_manager_fused(
    at::Tensor top_k_indices,
    at::Tensor hot_buffer_token_indices,
    at::Tensor hot_buffer_device_locations,
    at::Tensor cache_cpu_locations,
    at::Tensor top_k_device_locations,
    at::Tensor copy_src_cpu_locations,
    at::Tensor copy_dst_gpu_locations,
    at::Tensor copy_count) {
  TORCH_CHECK(top_k_indices.is_cuda(), "top_k_indices must be CUDA tensor");
  TORCH_CHECK(hot_buffer_token_indices.is_cuda(), "hot_buffer_token_indices must be CUDA tensor");
  TORCH_CHECK(top_k_indices.scalar_type() == at::kLong, "top_k_indices must be int64");
  TORCH_CHECK(hot_buffer_token_indices.scalar_type() == at::kLong, "hot_buffer_token_indices must be int64");

  int32_t top_k_size = top_k_indices.numel();
  int32_t hot_buffer_size = hot_buffer_token_indices.numel();

  TORCH_CHECK(top_k_size <= 256, "Fused kernel supports top_k <= 256");
  TORCH_CHECK(hot_buffer_size <= 1024, "Fused kernel supports hot_buffer <= 1024");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Reset copy count
  cudaMemsetAsync(copy_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream);

  // Launch fused kernel
  int threads = 256;
  sparse_cache_manager_fused_kernel<256, 1024><<<1, threads, 0, stream>>>(
      top_k_indices.data_ptr<int64_t>(),
      hot_buffer_token_indices.data_ptr<int64_t>(),
      hot_buffer_device_locations.data_ptr<int64_t>(),
      cache_cpu_locations.data_ptr<int64_t>(),
      top_k_device_locations.data_ptr<int64_t>(),
      copy_src_cpu_locations.data_ptr<int64_t>(),
      copy_dst_gpu_locations.data_ptr<int64_t>(),
      copy_count.data_ptr<int32_t>(),
      top_k_size,
      hot_buffer_size);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_cache_manager(
    at::Tensor top_k_indices,
    at::Tensor hot_buffer_token_indices,
    at::Tensor hot_buffer_device_locations,
    at::Tensor cache_cpu_locations,
    at::Tensor top_k_device_locations,
    at::Tensor copy_src_cpu_locations,
    at::Tensor copy_dst_gpu_locations,
    at::Tensor copy_count) {
  TORCH_CHECK(top_k_indices.is_cuda(), "top_k_indices must be CUDA tensor");
  TORCH_CHECK(hot_buffer_token_indices.is_cuda(), "hot_buffer_token_indices must be CUDA tensor");
  TORCH_CHECK(top_k_indices.scalar_type() == at::kLong, "top_k_indices must be int64");
  TORCH_CHECK(hot_buffer_token_indices.scalar_type() == at::kLong, "hot_buffer_token_indices must be int64");

  int32_t top_k_size = top_k_indices.numel();
  int32_t hot_buffer_size = hot_buffer_token_indices.numel();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Allocate temporary buffers
  auto options_int32 = at::TensorOptions().dtype(at::kInt).device(top_k_indices.device());
  at::Tensor top_k_hit_flags = at::zeros({top_k_size}, options_int32);
  at::Tensor hot_buffer_evictable = at::zeros({hot_buffer_size}, options_int32);
  at::Tensor eviction_list = at::zeros({hot_buffer_size}, options_int32);
  at::Tensor eviction_count = at::zeros({1}, options_int32);
  at::Tensor eviction_ptr = at::zeros({1}, options_int32);

  // Reset copy count
  cudaMemsetAsync(copy_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream);

  int threads = 256;
  int blocks_topk = (top_k_size + threads - 1) / threads;
  int blocks_hot = (hot_buffer_size + threads - 1) / threads;
  int blocks_max = std::max(blocks_topk, blocks_hot);

  // Phase 1: Identify hits
  sparse_cache_identify_hits_kernel<<<blocks_max, threads, 0, stream>>>(
      top_k_indices.data_ptr<int64_t>(),
      hot_buffer_token_indices.data_ptr<int64_t>(),
      hot_buffer_device_locations.data_ptr<int64_t>(),
      top_k_device_locations.data_ptr<int64_t>(),
      top_k_hit_flags.data_ptr<int32_t>(),
      hot_buffer_evictable.data_ptr<int32_t>(),
      top_k_size,
      hot_buffer_size);

  // Phase 2: Build eviction list
  sparse_cache_build_eviction_list_kernel<<<blocks_hot, threads, 0, stream>>>(
      hot_buffer_evictable.data_ptr<int32_t>(),
      eviction_list.data_ptr<int32_t>(),
      eviction_count.data_ptr<int32_t>(),
      hot_buffer_size);

  // Phase 3: Assign evictions
  sparse_cache_assign_evictions_kernel<<<blocks_topk, threads, 0, stream>>>(
      top_k_indices.data_ptr<int64_t>(),
      cache_cpu_locations.data_ptr<int64_t>(),
      hot_buffer_token_indices.data_ptr<int64_t>(),
      hot_buffer_device_locations.data_ptr<int64_t>(),
      top_k_device_locations.data_ptr<int64_t>(),
      top_k_hit_flags.data_ptr<int32_t>(),
      eviction_list.data_ptr<int32_t>(),
      copy_src_cpu_locations.data_ptr<int64_t>(),
      copy_dst_gpu_locations.data_ptr<int64_t>(),
      copy_count.data_ptr<int32_t>(),
      eviction_ptr.data_ptr<int32_t>(),
      top_k_size);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_cache_copy(
    at::Tensor cpu_cache,
    at::Tensor gpu_cache,
    at::Tensor copy_src_cpu_locations,
    at::Tensor copy_dst_gpu_locations,
    int32_t copy_count,
    int64_t item_size_bytes) {
  if (copy_count == 0) {
    return;
  }

  TORCH_CHECK(item_size_bytes % 8 == 0, "item_size_bytes must be divisible by 8");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Each warp handles one copy
  int threads = 256;
  int warps_per_block = threads / WARP_SIZE;
  int blocks = (copy_count + warps_per_block - 1) / warps_per_block;

  sparse_cache_copy_kernel<<<blocks, threads, 0, stream>>>(
      cpu_cache.data_ptr(),
      gpu_cache.data_ptr(),
      copy_src_cpu_locations.data_ptr<int64_t>(),
      copy_dst_gpu_locations.data_ptr<int64_t>(),
      copy_count,
      item_size_bytes);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/**
 * Batch version: Process multiple requests in parallel
 *
 * Each request has its own top_k, hot_buffer, etc.
 * Uses one CUDA block per request.
 */
template <int32_t MAX_TOP_K, int32_t MAX_HOT_BUFFER>
__global__ void sparse_cache_manager_batch_kernel(
    const int64_t* __restrict__ top_k_indices,          // [B, K] tokens we need
    int64_t* __restrict__ hot_buffer_token_indices,     // [B, H] tokens in GPU
    int64_t* __restrict__ hot_buffer_device_locations,  // [B, H] GPU locations
    const int64_t* __restrict__ cache_cpu_locations,    // [B, N] CPU locations
    int64_t* __restrict__ top_k_device_locations,       // [B, K] output
    int64_t* __restrict__ copy_src_cpu_locations,       // [B, M] CPU locations to copy
    int64_t* __restrict__ copy_dst_gpu_locations,       // [B, M] GPU locations to copy
    int32_t* __restrict__ copy_counts,                  // [B] copies per request
    const int32_t* __restrict__ top_k_sizes,            // [B] actual top_k per request
    const int32_t* __restrict__ hot_buffer_sizes,       // [B] actual hot buffer per request
    const int32_t* __restrict__ token_pool_sizes,       // [B] total tokens per request
    int32_t max_top_k,
    int32_t max_hot_buffer,
    int32_t max_copies) {
  int32_t req_id = blockIdx.x;
  int32_t tid = threadIdx.x;

  int32_t top_k_size = top_k_sizes[req_id];
  int32_t hot_buffer_size = hot_buffer_sizes[req_id];

  // Calculate offsets for this request
  int64_t top_k_offset = req_id * max_top_k;
  int64_t hot_buffer_offset = req_id * max_hot_buffer;
  int64_t token_pool_offset = 0;
  for (int32_t i = 0; i < req_id; ++i) {
    token_pool_offset += token_pool_sizes[i];
  }
  int64_t copy_offset = req_id * max_copies;

  // Pointers for this request
  const int64_t* req_top_k = top_k_indices + top_k_offset;
  int64_t* req_hot_tokens = hot_buffer_token_indices + hot_buffer_offset;
  int64_t* req_hot_locs = hot_buffer_device_locations + hot_buffer_offset;
  const int64_t* req_cpu_locs = cache_cpu_locations + token_pool_offset;
  int64_t* req_out_locs = top_k_device_locations + top_k_offset;
  int64_t* req_copy_src = copy_src_cpu_locations + copy_offset;
  int64_t* req_copy_dst = copy_dst_gpu_locations + copy_offset;

  // Shared memory for this block
  __shared__ int32_t s_top_k_hit_flags[MAX_TOP_K];
  __shared__ int32_t s_hot_buffer_evictable[MAX_HOT_BUFFER];
  __shared__ int32_t s_eviction_list[MAX_HOT_BUFFER];
  __shared__ int32_t s_eviction_count;
  __shared__ int32_t s_eviction_ptr;
  __shared__ int32_t s_copy_count;

  if (tid == 0) {
    s_eviction_count = 0;
    s_eviction_ptr = 0;
    s_copy_count = 0;
  }
  __syncthreads();

  // Phase 1a: Initialize evictable flags
  for (int32_t i = tid; i < hot_buffer_size; i += blockDim.x) {
    s_hot_buffer_evictable[i] = 1;
  }
  __syncthreads();

  // Phase 1b: Check hits and mark non-evictable
  for (int32_t k = tid; k < top_k_size; k += blockDim.x) {
    int64_t target_token = req_top_k[k];
    int32_t found = 0;
    int64_t device_loc = -1;

    for (int32_t h = 0; h < hot_buffer_size; ++h) {
      if (req_hot_tokens[h] == target_token) {
        found = 1;
        device_loc = req_hot_locs[h];
        atomicExch(&s_hot_buffer_evictable[h], 0);
        break;
      }
    }

    s_top_k_hit_flags[k] = found;
    if (found) {
      req_out_locs[k] = device_loc;
    }
  }
  __syncthreads();

  // Phase 2: Build eviction list
  for (int32_t h = tid; h < hot_buffer_size; h += blockDim.x) {
    if (s_hot_buffer_evictable[h]) {
      int32_t pos = atomicAdd(&s_eviction_count, 1);
      s_eviction_list[pos] = h;
    }
  }
  __syncthreads();

  // Phase 3: Assign evictions to misses
  for (int32_t k = tid; k < top_k_size; k += blockDim.x) {
    if (!s_top_k_hit_flags[k]) {
      int32_t evict_pos = atomicAdd(&s_eviction_ptr, 1);

      if (evict_pos < s_eviction_count) {
        int32_t hot_slot = s_eviction_list[evict_pos];
        int64_t gpu_loc = req_hot_locs[hot_slot];

        req_out_locs[k] = gpu_loc;

        int64_t token_idx = req_top_k[k];
        int64_t cpu_loc = req_cpu_locs[token_idx];

        int32_t copy_pos = atomicAdd(&s_copy_count, 1);
        req_copy_src[copy_pos] = cpu_loc;
        req_copy_dst[copy_pos] = gpu_loc;

        req_hot_tokens[hot_slot] = token_idx;
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    copy_counts[req_id] = s_copy_count;
  }
}

void sparse_cache_manager_batch(
    at::Tensor top_k_indices,
    at::Tensor hot_buffer_token_indices,
    at::Tensor hot_buffer_device_locations,
    at::Tensor cache_cpu_locations,
    at::Tensor top_k_device_locations,
    at::Tensor copy_src_cpu_locations,
    at::Tensor copy_dst_gpu_locations,
    at::Tensor copy_counts,
    at::Tensor top_k_sizes,
    at::Tensor hot_buffer_sizes,
    at::Tensor token_pool_sizes,
    int32_t batch_size,
    int32_t max_top_k,
    int32_t max_hot_buffer,
    int32_t max_copies) {
  TORCH_CHECK(max_top_k <= 256, "Batch kernel supports max_top_k <= 256");
  TORCH_CHECK(max_hot_buffer <= 1024, "Batch kernel supports max_hot_buffer <= 1024");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Reset copy counts
  cudaMemsetAsync(copy_counts.data_ptr<int32_t>(), 0, batch_size * sizeof(int32_t), stream);

  int threads = 256;
  sparse_cache_manager_batch_kernel<256, 1024><<<batch_size, threads, 0, stream>>>(
      top_k_indices.data_ptr<int64_t>(),
      hot_buffer_token_indices.data_ptr<int64_t>(),
      hot_buffer_device_locations.data_ptr<int64_t>(),
      cache_cpu_locations.data_ptr<int64_t>(),
      top_k_device_locations.data_ptr<int64_t>(),
      copy_src_cpu_locations.data_ptr<int64_t>(),
      copy_dst_gpu_locations.data_ptr<int64_t>(),
      copy_counts.data_ptr<int32_t>(),
      top_k_sizes.data_ptr<int32_t>(),
      hot_buffer_sizes.data_ptr<int32_t>(),
      token_pool_sizes.data_ptr<int32_t>(),
      max_top_k,
      max_hot_buffer,
      max_copies);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
