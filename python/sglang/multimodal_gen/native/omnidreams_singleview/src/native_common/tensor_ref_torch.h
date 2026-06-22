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

#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "native_common/scalar_types.h"
#include "native_common/tensor_ref.h"

namespace omnidreams_native {

template <typename T, int Rank>
TensorRef<T, Rank> from_torch(const torch::Tensor& tensor) {
  TORCH_CHECK(
      tensor.dim() == Rank,
      "from_torch: expected ",
      Rank,
      "D tensor, got ",
      tensor.dim(),
      "D");
  TORCH_CHECK(tensor.is_cuda(), "from_torch: tensor must be on CUDA");

  TensorRef<T, Rank> ref{};
  ref.ptr = reinterpret_cast<T*>(tensor.data_ptr());
  for (int i = 0; i < Rank; ++i) {
    ref.shape[i] = tensor.size(i);
    ref.strides[i] = tensor.stride(i);
  }
  return ref;
}

template <typename T, int Rank>
TensorRef<T, Rank> from_torch_checked(
    const torch::Tensor& tensor,
    c10::ScalarType expected_dtype) {
  TORCH_CHECK(
      tensor.scalar_type() == expected_dtype,
      "from_torch_checked: expected dtype ",
      expected_dtype,
      ", got ",
      tensor.scalar_type());
  return from_torch<T, Rank>(tensor);
}

namespace detail {

template <typename T>
struct TorchDtype;

template <>
struct TorchDtype<float16_t> {
  static constexpr auto value = c10::kHalf;
};
template <>
struct TorchDtype<bfloat16_t> {
  static constexpr auto value = c10::kBFloat16;
};
template <>
struct TorchDtype<float> {
  static constexpr auto value = c10::kFloat;
};
template <>
struct TorchDtype<int8_t> {
  static constexpr auto value = c10::kChar;
};
template <>
struct TorchDtype<uint8_t> {
  static constexpr auto value = c10::kByte;
};
template <>
struct TorchDtype<int32_t> {
  static constexpr auto value = c10::kInt;
};
template <>
struct TorchDtype<int64_t> {
  static constexpr auto value = c10::kLong;
};

}  // namespace detail

template <typename T, int Rank>
torch::Tensor to_torch(
    const TensorRef<T, Rank>& ref,
    torch::Device device = torch::kCUDA) {
  std::vector<int64_t> sizes(ref.shape, ref.shape + Rank);
  std::vector<int64_t> strides(ref.strides, ref.strides + Rank);
  auto options = torch::TensorOptions()
                     .dtype(detail::TorchDtype<T>::value)
                     .device(device);
  return torch::from_blob(ref.ptr, sizes, strides, options);
}

}  // namespace omnidreams_native
