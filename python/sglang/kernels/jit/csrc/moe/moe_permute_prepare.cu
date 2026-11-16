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
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cstdint>
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
    bool use_int64_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  // Padding and nonlocal routes are both inactive, independently of EP.
  int32_t neg_count = lower_bound(sorted_topk_ids, numel, 0);

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

void moe_permute_prepare(
    tvm::ffi::TensorView sorted_topk_ids,
    tvm::ffi::TensorView reorder_ids,
    tvm::ffi::TensorView expert_offsets,
    tvm::ffi::TensorView src2dst,
    int64_t num_experts,
    bool use_int64_offset) {
  using namespace host;

  auto N = SymbolicSize{"numel"};
  auto device = SymbolicDevice{};
  device.set_options<kDLGPU>();
  TensorMatcher({N})  // sorted_topk_ids
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(sorted_topk_ids);
  TensorMatcher({N})  // reorder_ids
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(reorder_ids);
  TensorMatcher({N})  // src2dst
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(src2dst);
  CHECK_HOST(num_experts >= 0 && num_experts < std::numeric_limits<int32_t>::max());
  CHECK_HOST(N.unwrap() <= std::numeric_limits<int32_t>::max());
  if (use_int64_offset) {
    TensorMatcher({num_experts + 1})  // expert_offsets
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(expert_offsets);
  } else {
    TensorMatcher({num_experts + 1})  // expert_offsets
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(expert_offsets);
  }

  const auto numel = static_cast<int32_t>(N.unwrap());
  const auto num_experts_i32 = static_cast<int32_t>(num_experts);
  constexpr uint32_t kThreads = 256;
  const auto num_work = static_cast<uint32_t>(std::max(numel, num_experts_i32 + 1));
  const auto num_blocks = div_ceil(num_work, kThreads);

  LaunchKernel(num_blocks, kThreads, device.unwrap())(
      moe_permute_prepare_kernel,
      static_cast<const int32_t*>(sorted_topk_ids.data_ptr()),
      static_cast<const int64_t*>(reorder_ids.data_ptr()),
      expert_offsets.data_ptr(),
      static_cast<int32_t*>(src2dst.data_ptr()),
      num_experts_i32,
      numel,
      use_int64_offset);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_permute_prepare, moe_permute_prepare);

}  // namespace sglang
