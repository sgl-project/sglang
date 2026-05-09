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

#include "utils.h"

// Binary search: find first index where sorted_topk_ids[index] >= target
__device__ __forceinline__ int32_t lower_bound(
    const int32_t* __restrict__ data, int32_t n, int32_t target) {
  int32_t lo = 0, hi = n;
  while (lo < hi) {
    int32_t mid = lo + (hi - lo) / 2;
    if (data[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// All blocks cooperate on both expert_offsets (via binary search) and src2dst.
// No shared memory, no atomics.
__global__ void moe_permute_prepare_kernel(
    const int32_t* __restrict__ sorted_topk_ids,
    const int64_t* __restrict__ reorder_ids,
    void* __restrict__ expert_offsets,
    int32_t* __restrict__ src2dst,
    int32_t num_experts,
    int32_t numel,
    bool use_int64_offset,
    bool is_ep) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int32_t neg_count = 0;
  if (is_ep) neg_count = lower_bound(sorted_topk_ids, numel, 0);

  // Compute expert_offsets via binary search
  for (int e = tid; e <= num_experts; e += stride) {
    int32_t offset;
    if (e < num_experts) {
      offset = lower_bound(sorted_topk_ids, numel, e) - neg_count;
    } else {
      offset = numel - neg_count;
    }

    if (use_int64_offset) {
      reinterpret_cast<int64_t*>(expert_offsets)[e] = static_cast<int64_t>(offset);
    } else {
      reinterpret_cast<int32_t*>(expert_offsets)[e] = offset;
    }
  }

  // Compute src2dst, skipping negative entries when is_ep
  for (int i = tid; i < numel; i += stride) {
    src2dst[reorder_ids[i]] = i - neg_count;
  }
}

std::vector<torch::Tensor>
moe_permute_prepare(torch::Tensor topk_ids, int64_t num_experts, bool use_int64_offset, bool is_ep) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int, "topk_ids must be int32");

  const auto device = topk_ids.device();
  const at::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  int32_t numel = topk_ids.numel();

  // Sort using torch::sort
  auto [sorted_topk_ids, reorder_ids] = torch::sort(topk_ids.flatten());

  auto out_dtype = use_int64_offset ? at::ScalarType::Long : at::ScalarType::Int;
  auto expert_offsets = torch::empty({num_experts + 1}, topk_ids.options().dtype(out_dtype));
  auto src2dst = torch::empty({numel}, topk_ids.options());

  const int threads = 256;
  int num_blocks = std::max(1, (std::max((int)numel, (int)num_experts + 1) + threads - 1) / threads);

  moe_permute_prepare_kernel<<<num_blocks, threads, 0, stream>>>(
      sorted_topk_ids.data_ptr<int32_t>(),
      reorder_ids.data_ptr<int64_t>(),
      expert_offsets.data_ptr(),
      src2dst.data_ptr<int32_t>(),
      (int32_t)num_experts,
      numel,
      use_int64_offset,
      is_ep);

  return {expert_offsets, src2dst};
}
