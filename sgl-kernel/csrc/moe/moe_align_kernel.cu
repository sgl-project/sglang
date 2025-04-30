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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "utils.h"

#define WARP_SIZE 32

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
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
    int32_t padded_num_experts,
    int32_t experts_per_warp,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum) {
  extern __shared__ int32_t shared_counts[];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];

      cumsum[i] = cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[i * num_experts + threadIdx.x] += tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = tokens_cnts[threadIdx.x * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[threadIdx.x * num_experts + expert_id];
  }
}

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor token_cnts_buffer,
    torch::Tensor cumsum_buffer) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  int experts_per_warp = WARP_SIZE;
  int threads = 1024;

  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    bool small_batch_expert_mode = (topk_ids.numel() < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode) {
      const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
      const int32_t shared_mem_size = ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

      auto small_batch_expert_kernel = moe_align_block_size_small_batch_expert_kernel<scalar_t>;
      small_batch_expert_kernel<<<1, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel());
    } else {
      auto align_kernel = moe_align_block_size_kernel<scalar_t>;

      size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
      size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

      cumsum_buffer.zero_();

      align_kernel<<<1, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          padded_num_experts,
          experts_per_warp,
          block_size,
          topk_ids.numel(),
          cumsum_buffer.data_ptr<int32_t>());

      const int block_threads = std::min(256, (int)threads);
      const int num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
      sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          cumsum_buffer.data_ptr<int32_t>(),
          topk_ids.numel());
    }
  });
}
