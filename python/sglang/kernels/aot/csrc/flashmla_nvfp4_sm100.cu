/* Copyright 2026 SGLang Team. All Rights Reserved.

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

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "api/sparse_decode.h"

namespace {

constexpr int kNvfp4RowBytes = 416;
constexpr int kTopK = 2048;
constexpr int kPageSize = 64;

}  // namespace

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_decode_fwd_nvfp4(
    const at::Tensor& q,
    const at::Tensor& packed_kv,
    const at::Tensor& kv_global_scale,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    int64_t d_v,
    double sm_scale) {
  TORCH_CHECK(q.is_cuda() && packed_kv.is_cuda() && indices.is_cuda());
  TORCH_CHECK(
      q.device() == packed_kv.device() && q.device() == indices.device() &&
          q.device() == kv_global_scale.device(),
      "q, packed_kv, indices, and kv_global_scale must be on one CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be BF16");
  TORCH_CHECK(packed_kv.scalar_type() == at::kByte, "packed_kv must be uint8");
  TORCH_CHECK(
      kv_global_scale.is_cuda() && kv_global_scale.scalar_type() == at::kFloat &&
      kv_global_scale.numel() == 1 && kv_global_scale.is_contiguous());
  TORCH_CHECK(indices.scalar_type() == at::kInt);
  TORCH_CHECK(q.dim() == 4 && q.size(2) == 64 && q.size(3) == 576);
  TORCH_CHECK(
      packed_kv.dim() == 4 && packed_kv.size(1) == kPageSize && packed_kv.size(2) == 1 &&
      packed_kv.size(3) == kNvfp4RowBytes);
  TORCH_CHECK(
      indices.sizes() == at::IntArrayRef({q.size(0), q.size(1), kTopK}),
      "indices must be [B, Sq, 2048]");
  TORCH_CHECK(d_v == 512);
  TORCH_CHECK(q.is_contiguous() && packed_kv.is_contiguous() && indices.is_contiguous());
  if (topk_length.has_value()) {
    TORCH_CHECK(topk_length->is_cuda() && topk_length->scalar_type() == at::kInt);
    TORCH_CHECK(topk_length->device() == q.device());
    TORCH_CHECK(topk_length->numel() == q.size(0));
    TORCH_CHECK(topk_length->is_contiguous());
  }
  c10::cuda::CUDAGuard device_guard(q.device());
  if (attn_sink.has_value()) {
    TORCH_CHECK(
        attn_sink->is_cuda() && attn_sink->device() == q.device() &&
        attn_sink->scalar_type() == at::kFloat && attn_sink->numel() == 64 &&
        attn_sink->is_contiguous());
  }

  auto [out, lse, new_metadata, new_splits] = sparse_attn_decode_interface(
      q,
      packed_kv,
      indices,
      topk_length,
      attn_sink,
      tile_scheduler_metadata,
      num_splits,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      static_cast<int>(d_v),
      static_cast<float>(sm_scale),
      kv_global_scale);

  return {out, lse, new_metadata, new_splits};
}
