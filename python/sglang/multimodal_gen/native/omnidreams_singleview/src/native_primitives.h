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

#include <torch/extension.h>

namespace omnidreams_singleview {

pybind11::dict native_tensor_descriptor(const torch::Tensor& tensor);
pybind11::dict native_tensor_ref_descriptor(const torch::Tensor& tensor);
pybind11::dict workspace_allocation_plan(
    torch::Tensor workspace,
    const std::vector<int64_t>& byte_sizes,
    int64_t alignment);
torch::Tensor prepare_contiguous(const torch::Tensor& input);
torch::Tensor zero_workspace_(torch::Tensor workspace);
void bind_native_primitives(pybind11::module_& module);
torch::Tensor prepare_contiguous_cuda(const torch::Tensor& input);
void zero_workspace_cuda(torch::Tensor workspace);

}  // namespace omnidreams_singleview
