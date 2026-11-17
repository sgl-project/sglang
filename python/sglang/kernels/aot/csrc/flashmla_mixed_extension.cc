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

#include <torch/all.h>
#include <torch/library.h>

#include <optional>
#include <string>
#include <tuple>

#include "sgl_kernel_ops.h"

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_mixed_decode_interface_v1(
    const at::Tensor& q,
    const at::Tensor& swa_cache,
    const at::Tensor& swa_indices,
    const std::optional<at::Tensor>& swa_topk_length,
    const at::Tensor& main_cache_bytes,
    const at::Tensor& main_indices,
    const std::optional<at::Tensor>& main_topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor>& tile_scheduler_metadata,
    std::optional<at::Tensor>& num_splits,
    int d_v,
    float sm_scale,
    const std::string& swa_layout,
    const std::string& main_layout,
    int main_page_slots,
    int main_page_bytes);

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sgl_sparse_decode_fwd_mixed_v1(
    const at::Tensor& q,
    const at::Tensor& swa_cache,
    const at::Tensor& swa_indices,
    const std::optional<at::Tensor>& swa_topk_length,
    const at::Tensor& main_cache_bytes,
    const at::Tensor& main_indices,
    const std::optional<at::Tensor>& main_topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    int64_t d_v,
    double sm_scale,
    const std::string& swa_layout,
    const std::string& main_layout,
    int64_t main_page_slots,
    int64_t main_page_bytes) {
  return sparse_attn_mixed_decode_interface_v1(
      q,
      swa_cache,
      swa_indices,
      swa_topk_length,
      main_cache_bytes,
      main_indices,
      main_topk_length,
      attn_sink,
      tile_scheduler_metadata,
      num_splits,
      static_cast<int>(d_v),
      static_cast<float>(sm_scale),
      swa_layout,
      main_layout,
      static_cast<int>(main_page_slots),
      static_cast<int>(main_page_bytes));
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "sparse_decode_fwd_mixed_v1(Tensor q, Tensor swa_cache, Tensor swa_indices, Tensor? swa_topk_length, "
      "Tensor main_cache_bytes, Tensor main_indices, Tensor? main_topk_length, Tensor? attn_sink, "
      "Tensor? tile_scheduler_metadata, Tensor? num_splits, int d_v, float sm_scale, str swa_layout, "
      "str main_layout, int main_page_slots, int main_page_bytes) -> (Tensor, Tensor, Tensor?, Tensor?)");
  m.impl("sparse_decode_fwd_mixed_v1", torch::kCUDA, &sgl_sparse_decode_fwd_mixed_v1);
  m.def("flashmla_mixed_kv_api_version() -> int", []() -> int64_t { return 1; });
}

REGISTER_EXTENSION(flashmla_mixed_ops)
