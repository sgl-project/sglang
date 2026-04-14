// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/cuda/CUDAContext.h>
#include <cutlass/arch/arch.h>
#include <torch/extension.h>

#include <cute/numeric/numeric_types.hpp>

#include "kda/sm90/prefill_kernel.hpp"
#include "sgl_flash_kernel_ops.h"

using OptionalTensor = std::optional<torch::Tensor>;

std::tuple<torch::Tensor, torch::Tensor> kda_fwd_prefill(
    OptionalTensor output_,
    OptionalTensor output_state_,
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    OptionalTensor input_state_,
    OptionalTensor alpha_,
    OptionalTensor beta_,
    torch::Tensor const& cu_seqlens,
    torch::Tensor workspace_buffer,
    double scale,
    bool safe_gate) {
  // Q, K, V: [packed_seq, H, D] (already packed by Python layer)
  auto packed_seq = q.size(0);
  auto num_heads = q.size(1);
  auto head_size = q.size(2);
  auto num_seqs = cu_seqlens.size(0) - 1;

  // KDA constraint: all head counts must be the same
  TORCH_CHECK(num_heads == k.size(1), "KDA requires num_q_heads == num_k_heads, got ", num_heads, " vs ", k.size(1));
  TORCH_CHECK(num_heads == v.size(1), "KDA requires num_q_heads == num_v_heads, got ", num_heads, " vs ", v.size(1));
  TORCH_CHECK(head_size == v.size(2), "KDA requires Q and V head dim to match, got ", head_size, " vs ", v.size(2));

  // Allocate output if not provided
  torch::Tensor output = output_.has_value() ? output_.value()
                                             : torch::empty(
                                                   {packed_seq, num_heads, head_size},
                                                   torch::TensorOptions().dtype(q.dtype()).device(q.device()));

  // Allocate output state if not provided
  torch::Tensor output_state = output_state_.has_value()
                                   ? output_state_.value()
                                   : torch::zeros(
                                         {num_seqs, num_heads, head_size, head_size},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

  // Validate dtypes
  TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
  TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
  TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");

  // Validate contiguity
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output_state.is_contiguous(), "output_state must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  TORCH_CHECK(workspace_buffer.is_contiguous(), "workspace_buffer must be contiguous");

  // Extract optional pointers
  float const* alpha_ptr = nullptr;
  float const* beta_ptr = nullptr;
  float const* input_state_ptr = nullptr;

  if (alpha_.has_value()) {
    auto& alpha = alpha_.value();
    TORCH_CHECK(alpha.dtype() == torch::kFloat32, "alpha must be float32");
    TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
    TORCH_CHECK(
        alpha.size(0) == packed_seq && alpha.size(1) == num_heads && alpha.size(2) == head_size,
        "alpha shape must be [packed_seq, num_heads, head_size]");
    alpha_ptr = alpha.data_ptr<float>();
  }
  if (beta_.has_value()) {
    auto& beta = beta_.value();
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be float32");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(beta.size(0) == packed_seq && beta.size(1) == num_heads, "beta shape must be [packed_seq, num_heads]");
    beta_ptr = beta.data_ptr<float>();
  }
  // input_state is in SGLang VK layout [N, H, V, K], which matches the kernel's
  // native LayoutLeft (K, V, H, N) — K is the contiguous dimension in both.
  if (input_state_.has_value()) {
    auto& input_state = input_state_.value();
    TORCH_CHECK(input_state.dtype() == torch::kFloat32, "input_state must be float32");
    TORCH_CHECK(input_state.is_contiguous(), "input_state must be contiguous");
    input_state_ptr = input_state.data_ptr<float>();
  }

  // Auto-compute scale if 0
  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / std::sqrt(static_cast<float>(head_size));
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  auto sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  using bf16 = cute::bfloat16_t;
  using Sm90 = cutlass::arch::Sm90;

  kda::sm90::launch_kda_fwd_prefill_kernel<Sm90, bf16, bf16, float>(
      stream,
      reinterpret_cast<bf16*>(output.data_ptr()),
      output_state.data_ptr<float>(),
      reinterpret_cast<bf16 const*>(q.data_ptr()),
      reinterpret_cast<bf16 const*>(k.data_ptr()),
      reinterpret_cast<bf16 const*>(v.data_ptr()),
      input_state_ptr,
      alpha_ptr,
      beta_ptr,
      cu_seqlens.data_ptr<int32_t>(),
      workspace_buffer.data_ptr<uint8_t>(),
      static_cast<int32_t>(num_seqs),
      static_cast<int32_t>(num_heads),
      static_cast<int32_t>(head_size),
      static_cast<int64_t>(packed_seq),
      scale_f,
      safe_gate,
      static_cast<int32_t>(sm_count));

  // output_state uses CuTe LayoutLeft (K, V, H, N), which maps to
  // PyTorch contiguous [N, H, V, K] — already in SGLang VK layout.
  return {output, output_state};
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "kda_fwd_prefill("
      "Tensor? output_, "
      "Tensor? output_state_, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor? input_state_, "
      "Tensor? alpha_, "
      "Tensor? beta_, "
      "Tensor cu_seqlens, "
      "Tensor workspace_buffer, "
      "float scale, "
      "bool safe_gate"
      ") -> (Tensor, Tensor)");
  m.impl("kda_fwd_prefill", &kda_fwd_prefill);
}

REGISTER_EXTENSION(cula_kda_ops)
