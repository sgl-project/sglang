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
#include <torch/extension.h>

#include <THC/THCAtomics.cuh>

#include "utils.h"

#define WARP_SIZE 32

template <typename scalar_t>
__global__ void sort_token_ids_kernel(scalar_t* __restrict__ topk_ids, 
                                    int32_t* sorted_token_ids,
                                    int32_t* token_cnts_buffer,
                                    const int32_t* cumsum,
                                    int32_t num_experts,
                                    size_t numel,
                                    int32_t tokens_per_block) {
    const size_t start_idx = blockIdx.x * tokens_per_block;
    const size_t end_idx = min(start_idx + tokens_per_block, numel);
    
    const size_t off_t = blockIdx.x * num_experts;
    
    for (size_t i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        int expert_id = topk_ids[i];
        int token_cnt = atomicAdd(&token_cnts_buffer[off_t + expert_id], 1);
        int rank_post_pad = token_cnt + cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
    }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* cumsum) {
  __shared__ int32_t shared_counts[WARP_SIZE][8];
  __shared__ int32_t local_offsets[256];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int experts_per_warp = 8;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < num_experts) {
      shared_counts[warp_id][i] = 0;
    }
  }

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx][expert_offset];

      cumsum[i] = cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
    local_offsets[threadIdx.x] = cumsum[threadIdx.x];
  }

  __syncthreads();

  // Note: For the moe_align_kernel, the primary bottleneck lies in the atomic add and non-coalesced memory writes here.
  // If these operations can be performed using multiple blocks, similar to the Triton version, the performance of this
  // kernel can achieve state-of-the-art performance across all token cases. However, once multiple blocks are used,
  // illegal memory access occurs. Even replacing these lines of code with the stage 4 kernel from the Triton version
  // results in the same issue, and a correct solution has not yet been found.
  // for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
  //   int32_t expert_id = topk_ids[i];
  //   int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
  //   sorted_token_ids[rank_post_pad] = i;
  // }
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(num_experts == 256, "moe_align_block_size kernel only support deepseek v3 now.");

  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    auto align_kernel = moe_align_block_size_kernel<scalar_t>;
    align_kernel<<<1, 1024, 0, stream>>>(topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
                                         experts_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(),
                                         num_experts, block_size, topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());
    auto sort_kernel = sort_token_ids_kernel<scalar_t>;
    const int tokens_per_block = CEILDIV(topk_ids.numel(), num_experts);
    sort_kernel<<<num_experts, 256, 0, stream>>>(
        topk_ids.data_ptr<scalar_t>(),
        sorted_token_ids.data_ptr<int32_t>(),
        token_cnts_buffer.data_ptr<int32_t>(),
        cumsum_buffer.data_ptr<int32_t>(),
        num_experts,
        topk_ids.numel(),
        tokens_per_block);
  });
}
