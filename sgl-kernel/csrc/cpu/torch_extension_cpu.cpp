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

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "shm.h"

// silu_and_mul
at::Tensor silu_and_mul_cpu(at::Tensor& input);

// rmsnorm
at::Tensor rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps);

// fused_add_rmsnorm
void fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps);

// topk
std::tuple<at::Tensor, at::Tensor> grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group);

std::tuple<at::Tensor, at::Tensor> biased_grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    at::Tensor& correction_bias,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group);

// attention
void decode_attention_cpu(
    at::Tensor& query,
    at::Tensor& output,
    at::Tensor& k_cache,
    at::Tensor& v_cahce,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap);

void extend_attention_cpu(
    at::Tensor& q_extend,
    at::Tensor& k_extend,
    at::Tensor& v_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap);

// weight prepack
at::Tensor convert_weight_packed(at::Tensor& weight);

// quant
std::tuple<at::Tensor, at::Tensor> per_token_quant_int8_cpu(at::Tensor& A);

// gemm
at::Tensor weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, std::optional<at::Tensor>& bias, bool is_vnni);

// igemm
at::Tensor int8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales1,
    at::Tensor& scales2,
    std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// quant + igemm
at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// bmm
void bmm_cpu(at::Tensor& out, at::Tensor& mat1, at::Tensor& mat2, bool is_vnni, std::optional<at::Tensor>& scale);

// fused moe
at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool use_int8_w8a8,
    std::optional<at::Tensor>& w1_scale,
    std::optional<at::Tensor>& w2_scale,
    std::optional<at::Tensor>& a1_scale,
    std::optional<at::Tensor>& a2_scale,
    bool is_vnni);

at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    std::optional<at::Tensor>& w1_scale,
    std::optional<at::Tensor>& w2_scale,
    std::optional<at::Tensor>& a1_scale,
    std::optional<at::Tensor>& a2_scale,
    bool is_vnni);

// weight absorption
std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope(
    at::Tensor& hidden_states,
    at::Tensor& q_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& kv_a_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    std::optional<at::Tensor>& q_a_proj_scale,
    std::optional<at::Tensor>& q_b_proj_scale,
    std::optional<at::Tensor>& kv_a_proj_scale,
    bool is_vnni);

// shared memory init
void initialize(int size, int rank);

// shared mmeory all_reduce
void shm_allreduce(at::Tensor& data, c10::intrusive_ptr<c10d::ProcessGroup> process_group, py::object op);

// shared memory all_gather
at::Tensor shm_allgather(at::Tensor& data, c10::intrusive_ptr<c10d::ProcessGroup> process_group, int dim);

// rope
std::tuple<at::Tensor, at::Tensor>
rotary_position_embedding_cpu(at::Tensor& t_pos, at::Tensor& q_pe, at::Tensor& k_pe, at::Tensor& t_emb_pos);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // activation
  m.def("silu_and_mul_cpu", &silu_and_mul_cpu, "SiLU and mul for CPU");

  // norm
  m.def("rmsnorm_cpu", &rmsnorm_cpu, "Root mean square normalization for CPU");
  m.def("fused_add_rmsnorm_cpu", &fused_add_rmsnorm_cpu, "Fused add root mean square normalization for CPU");

  // topk
  m.def("grouped_topk_cpu", &grouped_topk_cpu, "Grouped TopK for CPU");

  // biased group topk
  m.def("biased_grouped_topk_cpu", &biased_grouped_topk_cpu, "Biased Grouped TopK for CPU");

  // decode
  m.def("decode_attention_cpu", &decode_attention_cpu, "Attention decoding for CPU");

  // extend
  m.def("extend_attention_cpu", &extend_attention_cpu, "Attention extend for CPU");

  // weight prepack
  m.def("convert_weight_packed", &convert_weight_packed, "prepack weight to vnni format for intel AMX");

  // quant
  m.def("per_token_quant_int8_cpu", &per_token_quant_int8_cpu, "dynamic quantization for CPU");

  // gemm
  m.def("weight_packed_linear", &weight_packed_linear, "weight packed linear for intel AMX");

  // igemm
  m.def("int8_scaled_mm_cpu", &int8_scaled_mm_cpu, "int8 weight packed linear for intel AMX");

  // quant + igemm
  m.def(
      "int8_scaled_mm_with_quant", &int8_scaled_mm_with_quant, "fused per row quant and int8 scaled mm for intel AMX");

  // bmm
  m.def("bmm_cpu", &bmm_cpu, "bmm kernel for intel AMX");

  // moe
  m.def("fused_experts_cpu", &fused_experts_cpu, "fused moe kernel for CPU");

  // weight absorption
  m.def("qkv_proj_with_rope", &qkv_proj_with_rope, "fused qkv projection kernel with weight absorption for intel AMX");

  // shared expert
  m.def("shared_expert_cpu", &shared_expert_cpu, "shared expert kernel for CPU");

  // all reduce
  m.def("initialize", &initialize, "shared memory initialization for CPU");
  m.def("shm_allreduce", &shm_allreduce, "low latency all_reduce implementation for CPU");
  m.def("shm_allgather", &shm_allgather, "low latency all_gather implementation for CPU");

  // rope
  m.def("rotary_position_embedding_cpu", &rotary_position_embedding_cpu, "rotary position embedding for CPU");
}
