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

#ifndef OMNIDREAMS_SINGLEVIEW_WITH_CUDA
#error "OmniDreams native primitives require CUDA"
#endif

#include "native_common/scalar_types.h"
#include "native_common/tensor_ref_torch.h"
#include "native_common/workspace_allocator.h"

#include <c10/core/ScalarType.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace omnidreams_singleview {
namespace {

pybind11::tuple int_array_ref_to_tuple(c10::IntArrayRef values) {
  pybind11::tuple result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    result[i] = values[i];
  }
  return result;
}

void check_defined(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_tensor_ref_rank(const torch::Tensor& tensor) {
  TORCH_CHECK(
      tensor.dim() >= 1 && tensor.dim() <= 8,
      "native_tensor_ref_descriptor supports tensors with ranks 1..8, got ",
      tensor.dim());
}

template <typename T, int Rank>
pybind11::tuple shape_tuple(const omnidreams_native::TensorRef<T, Rank>& ref) {
  pybind11::tuple shape(Rank);
  for (int i = 0; i < Rank; ++i) {
    shape[i] = ref.shape[i];
  }
  return shape;
}

template <typename T, int Rank>
pybind11::tuple stride_tuple(const omnidreams_native::TensorRef<T, Rank>& ref) {
  pybind11::tuple stride(Rank);
  for (int i = 0; i < Rank; ++i) {
    stride[i] = ref.strides[i];
  }
  return stride;
}

template <typename T, int Rank>
pybind11::dict descriptor_from_ref(const torch::Tensor& tensor) {
  const auto ref = omnidreams_native::from_torch<T, Rank>(tensor);
  pybind11::dict descriptor;
  descriptor["shape"] = shape_tuple(ref);
  descriptor["stride"] = stride_tuple(ref);
  descriptor["rank"] = Rank;
  descriptor["dtype"] = std::string(c10::toString(tensor.scalar_type()));
  descriptor["device"] = tensor.device().str();
  descriptor["is_cuda"] = tensor.is_cuda();
  descriptor["is_contiguous"] = ref.is_contiguous();
  descriptor["nbytes"] = ref.nbytes();
  return descriptor;
}

template <typename T>
pybind11::dict dispatch_tensor_ref_rank(const torch::Tensor& tensor) {
  switch (tensor.dim()) {
    case 1:
      return descriptor_from_ref<T, 1>(tensor);
    case 2:
      return descriptor_from_ref<T, 2>(tensor);
    case 3:
      return descriptor_from_ref<T, 3>(tensor);
    case 4:
      return descriptor_from_ref<T, 4>(tensor);
    case 5:
      return descriptor_from_ref<T, 5>(tensor);
    case 6:
      return descriptor_from_ref<T, 6>(tensor);
    case 7:
      return descriptor_from_ref<T, 7>(tensor);
    case 8:
      return descriptor_from_ref<T, 8>(tensor);
    default:
      TORCH_CHECK(false, "unsupported tensor rank: ", tensor.dim());
  }
}

size_t checked_size(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0, name, " must be non-negative, got ", value);
  TORCH_CHECK(
      static_cast<uint64_t>(value) <=
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
      name,
      " is too large: ",
      value);
  return static_cast<size_t>(value);
}

}  // namespace

pybind11::dict native_tensor_descriptor(const torch::Tensor& tensor) {
  check_defined(tensor, "tensor");
  check_cuda_tensor(tensor, "tensor");

  pybind11::dict descriptor;
  descriptor["shape"] = int_array_ref_to_tuple(tensor.sizes());
  descriptor["stride"] = int_array_ref_to_tuple(tensor.strides());
  descriptor["dtype"] = std::string(c10::toString(tensor.scalar_type()));
  descriptor["device"] = tensor.device().str();
  descriptor["is_cuda"] = tensor.is_cuda();
  descriptor["is_contiguous"] = tensor.is_contiguous();
  descriptor["nbytes"] =
      static_cast<int64_t>(tensor.numel()) * static_cast<int64_t>(tensor.element_size());
  return descriptor;
}

pybind11::dict native_tensor_ref_descriptor(const torch::Tensor& tensor) {
  check_defined(tensor, "tensor");
  check_cuda_tensor(tensor, "tensor");
  check_tensor_ref_rank(tensor);

  switch (tensor.scalar_type()) {
    case c10::kHalf:
      return dispatch_tensor_ref_rank<omnidreams_native::float16_t>(tensor);
    case c10::kBFloat16:
      return dispatch_tensor_ref_rank<omnidreams_native::bfloat16_t>(tensor);
    case c10::kFloat:
      return dispatch_tensor_ref_rank<float>(tensor);
    case c10::kChar:
      return dispatch_tensor_ref_rank<int8_t>(tensor);
    case c10::kByte:
      return dispatch_tensor_ref_rank<uint8_t>(tensor);
    default:
      TORCH_CHECK(
          false,
          "native_tensor_ref_descriptor does not support dtype ",
          tensor.scalar_type());
  }
}

pybind11::dict workspace_allocation_plan(
    torch::Tensor workspace,
    const std::vector<int64_t>& byte_sizes,
    int64_t alignment) {
  check_defined(workspace, "workspace");
  check_cuda_tensor(workspace, "workspace");
  TORCH_CHECK(
      workspace.scalar_type() == c10::kByte,
      "workspace_allocation_plan expects a torch.uint8 workspace tensor");
  TORCH_CHECK(
      workspace.is_contiguous(),
      "workspace_allocation_plan expects a contiguous workspace tensor");
  const size_t requested_alignment = checked_size(alignment, "alignment");
  TORCH_CHECK(requested_alignment > 0, "alignment must be positive");

  const size_t element_size = workspace.element_size();
  const size_t numel = checked_size(workspace.numel(), "workspace element count");
  TORCH_CHECK(
      numel <= std::numeric_limits<size_t>::max() / element_size,
      "workspace byte size is too large");
  const size_t total_bytes = numel * element_size;
  omnidreams_native::WorkspaceAllocator allocator(
      workspace.data_ptr(), total_bytes, "omnidreams_workspace");

  pybind11::list offsets;
  pybind11::list sizes;
  for (size_t i = 0; i < byte_sizes.size(); ++i) {
    const size_t bytes = checked_size(byte_sizes[i], "requested workspace bytes");
    TORCH_CHECK(
        allocator.align_to(requested_alignment),
        "workspace alignment overflow before allocation ",
        i);
    const size_t offset = allocator.used();
    static_cast<void>(allocator.alloc<uint8_t>(bytes, 1));
    TORCH_CHECK(
        !allocator.overflowed(),
        "workspace allocation ",
        i,
        " requested ",
        allocator.failed_request_bytes(),
        " bytes at offset ",
        allocator.failed_offset(),
        " but only ",
        allocator.total(),
        " bytes are available");
    offsets.append(static_cast<int64_t>(offset));
    sizes.append(static_cast<int64_t>(bytes));
  }

  pybind11::dict plan;
  plan["offsets"] = offsets;
  plan["sizes"] = sizes;
  plan["alignment"] = static_cast<int64_t>(requested_alignment);
  plan["used_bytes"] = static_cast<int64_t>(allocator.used());
  plan["remaining_bytes"] = static_cast<int64_t>(allocator.remaining());
  plan["total_bytes"] = static_cast<int64_t>(allocator.total());
  return plan;
}

torch::Tensor prepare_contiguous(const torch::Tensor& input) {
  check_defined(input, "input");
  check_cuda_tensor(input, "input");

  if (input.is_contiguous()) {
    return input;
  }
  return prepare_contiguous_cuda(input);
}

torch::Tensor zero_workspace_(torch::Tensor workspace) {
  check_defined(workspace, "workspace");
  check_cuda_tensor(workspace, "workspace");
  TORCH_CHECK(
      workspace.is_contiguous(),
      "zero_workspace_ expects a contiguous workspace tensor");

  zero_workspace_cuda(workspace);
  return workspace;
}

void bind_native_primitives(pybind11::module_& module) {
  module.def("native_tensor_descriptor", &native_tensor_descriptor);
  module.def("native_tensor_ref_descriptor", &native_tensor_ref_descriptor);
  module.def("workspace_allocation_plan", &workspace_allocation_plan);
  module.def("prepare_contiguous", &prepare_contiguous);
  module.def("zero_workspace_", &zero_workspace_);
}

}  // namespace omnidreams_singleview
