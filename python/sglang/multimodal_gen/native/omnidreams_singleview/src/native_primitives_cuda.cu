/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "native_primitives.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>

namespace omnidreams_singleview {
namespace {

constexpr int64_t kMaxNativeTensorDims = 8;
constexpr int kThreadsPerBlock = 256;
constexpr int64_t kMaxBlocks = 4096;

struct TensorLayout {
  int64_t ndim;
  int64_t sizes[kMaxNativeTensorDims];
  int64_t strides[kMaxNativeTensorDims];
};

TensorLayout make_layout(const torch::Tensor& input) {
  TORCH_CHECK(
      input.dim() <= kMaxNativeTensorDims,
      "prepare_contiguous supports tensors up to ",
      kMaxNativeTensorDims,
      " dimensions, got ",
      input.dim());

  TensorLayout layout{};
  layout.ndim = input.dim();
  for (int64_t dim = 0; dim < layout.ndim; ++dim) {
    layout.sizes[dim] = input.size(dim);
    layout.strides[dim] = input.stride(dim);
  }
  return layout;
}

int64_t block_count(int64_t numel) {
  if (numel == 0) {
    return 1;
  }
  const int64_t blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return std::min(blocks, kMaxBlocks);
}

template <typename scalar_t>
__global__ void copy_strided_to_contiguous_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel,
    TensorLayout layout) {
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < numel;
       linear += blockDim.x * gridDim.x) {
    int64_t remaining = linear;
    int64_t input_offset = 0;
    for (int64_t dim = layout.ndim - 1; dim >= 0; --dim) {
      const int64_t index = remaining % layout.sizes[dim];
      remaining /= layout.sizes[dim];
      input_offset += index * layout.strides[dim];
    }
    output[linear] = input[input_offset];
  }
}

template <typename scalar_t>
__global__ void zero_workspace_kernel(scalar_t* __restrict__ data, int64_t numel) {
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < numel;
       linear += blockDim.x * gridDim.x) {
    data[linear] = scalar_t(0);
  }
}

}  // namespace

torch::Tensor prepare_contiguous_cuda(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "prepare_contiguous_cuda expects a CUDA tensor");
  if (input.is_contiguous()) {
    return input;
  }

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty(input.sizes(), input.options());
  if (input.numel() == 0) {
    return output;
  }

  const TensorLayout layout = make_layout(input);
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "prepare_contiguous_cuda",
      [&] {
        copy_strided_to_contiguous_kernel<scalar_t>
            <<<block_count(input.numel()), kThreadsPerBlock, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                input.numel(),
                layout);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void zero_workspace_cuda(torch::Tensor workspace) {
  TORCH_CHECK(workspace.is_cuda(), "zero_workspace_cuda expects a CUDA tensor");
  TORCH_CHECK(
      workspace.is_contiguous(),
      "zero_workspace_cuda expects a contiguous workspace tensor");
  if (workspace.numel() == 0) {
    return;
  }

  const c10::cuda::CUDAGuard device_guard(workspace.device());
  const auto stream = at::cuda::getCurrentCUDAStream(workspace.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      workspace.scalar_type(),
      "zero_workspace_cuda",
      [&] {
        zero_workspace_kernel<scalar_t>
            <<<block_count(workspace.numel()), kThreadsPerBlock, 0, stream>>>(
                workspace.data_ptr<scalar_t>(),
                workspace.numel());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace omnidreams_singleview
