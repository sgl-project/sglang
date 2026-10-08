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

// Adapted from: https://github.com/vllm-project/vllm/blob/main/csrc/ops.h

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

#include <vector>

at::Tensor weak_ref_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "weak_ref_tensor expects a CUDA tensor");

  void* data_ptr = tensor.data_ptr();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  auto options = tensor.options();

  auto new_tensor = at::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}
