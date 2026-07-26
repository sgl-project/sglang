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

#include "tvm_ffi_utils.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>

namespace sglang {

// Binary search: find first index where data[index] >= target.
__device__ __forceinline__ int32_t lower_bound(const int32_t* __restrict__ data, int32_t n, int32_t target) {
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

// All blocks cooperate on both expert_offsets and src2dst.
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

  for (int i = tid; i < numel; i += stride) {
    src2dst[reorder_ids[i]] = i - neg_count;
  }
}

}  // namespace sglang

void moe_permute_prepare(
    TensorView sorted_topk_ids,
    TensorView reorder_ids,
    TensorView expert_offsets,
    TensorView src2dst,
    int64_t num_experts,
    bool use_int64_offset,
    bool is_ep) {
  CHECK_INPUT_AND_TYPE(sorted_topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(reorder_ids, dl_int64);
  CHECK_INPUT_AND_TYPE(src2dst, dl_int32);
  CHECK_CUDA(expert_offsets);
  CHECK_CONTIGUOUS(expert_offsets);
  CHECK_DEVICE(sorted_topk_ids, reorder_ids);
  CHECK_DEVICE(sorted_topk_ids, expert_offsets);
  CHECK_DEVICE(sorted_topk_ids, src2dst);
  CHECK_DIM(1, sorted_topk_ids);
  CHECK_DIM(1, reorder_ids);
  CHECK_DIM(1, expert_offsets);
  CHECK_DIM(1, src2dst);
  TVM_FFI_ICHECK_EQ(reorder_ids.size(0), sorted_topk_ids.size(0));
  TVM_FFI_ICHECK_EQ(src2dst.size(0), sorted_topk_ids.size(0));
  TVM_FFI_ICHECK_GE(num_experts, 0);
  TVM_FFI_ICHECK_LT(num_experts, std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_LE(sorted_topk_ids.size(0), std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_EQ(expert_offsets.size(0), num_experts + 1);
  if (use_int64_offset) {
    CHECK_INPUT_TYPE(expert_offsets, dl_int64);
  } else {
    CHECK_INPUT_TYPE(expert_offsets, dl_int32);
  }

  cudaSetDevice(sorted_topk_ids.device().device_id);
  cudaStream_t stream = get_stream(sorted_topk_ids.device());
  int32_t numel = static_cast<int32_t>(sorted_topk_ids.size(0));
  int32_t num_experts_i32 = static_cast<int32_t>(num_experts);
  constexpr int threads = 256;
  int num_blocks = std::max(1, (std::max(numel, num_experts_i32 + 1) + threads - 1) / threads);

  sglang::moe_permute_prepare_kernel<<<num_blocks, threads, 0, stream>>>(
      static_cast<const int32_t*>(sorted_topk_ids.data_ptr()),
      static_cast<const int64_t*>(reorder_ids.data_ptr()),
      expert_offsets.data_ptr(),
      static_cast<int32_t*>(src2dst.data_ptr()),
      num_experts_i32,
      numel,
      use_int64_offset,
      is_ep);

  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << "moe_permute_prepare launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_permute_prepare, moe_permute_prepare);
