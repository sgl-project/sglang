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

// Pybind entry for the InfLLM-V2 FlashAttention backend (vendored from
// 3rdparty/infllmv2_cuda_impl). This builds as a standalone extension module
// `infllm_ops` so its `flash::` symbols stay isolated from sgl-kernel's own
// flash attention (`flash_ops` / `common_ops`).

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <vector>

// Forward declarations of the FlashAttention entry points implemented in
// flash_attn/flash_api.cpp. Signatures must match exactly.
std::vector<at::Tensor> mha_varlen_fwd_stage1(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    c10::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_v,
    c10::optional<at::Tensor>& seqused_k,
    c10::optional<const at::Tensor>& leftpad_k_,
    c10::optional<at::Tensor>& block_table_,
    c10::optional<at::Tensor>& alibi_slopes_,
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    c10::optional<at::Generator> gen_);

PYBIND11_MODULE(infllm_ops, m) {
  m.doc() = "InfLLM V2 FlashAttention backend (vendored into sgl-kernel)";
  m.def("varlen_fwd_stage1", &mha_varlen_fwd_stage1, "Forward pass (variable length) NSA stage 1");
}
