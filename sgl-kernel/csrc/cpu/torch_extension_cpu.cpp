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
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"
#include "shm.h"

// silu_and_mul
at::Tensor silu_and_mul_cpu(at::Tensor& input);

// gelu_and_mul
at::Tensor gelu_tanh_and_mul_cpu(const at::Tensor& input);
at::Tensor gelu_and_mul_cpu(const at::Tensor& input);

// l2norm
at::Tensor l2norm_cpu(at::Tensor& input, double eps);

// rmsnorm
at::Tensor rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps);

// qwen3_next_rmsnorm_gated
at::Tensor fused_rmsnorm_gated_cpu(at::Tensor& input, at::Tensor& weight, at::Tensor& gate, double eps);

// fused_add_rmsnorm
void fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps);

// topk
std::tuple<at::Tensor, at::Tensor>
topk_sigmoid_cpu(at::Tensor& hidden_states, at::Tensor& gating_output, int64_t topk, bool renormalize);
std::tuple<at::Tensor, at::Tensor>
topk_softmax_cpu(at::Tensor& hidden_states, at::Tensor& gating_output, int64_t topk, bool renormalize);

std::tuple<at::Tensor, at::Tensor> grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t num_fused_shared_experts,
    std::optional<double> routed_scaling_factor,
    std::optional<at::Tensor> num_token_non_padded);

std::tuple<at::Tensor, at::Tensor> biased_grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    at::Tensor& correction_bias,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t num_fused_shared_experts,
    std::optional<double> routed_scaling_factor,
    std::optional<at::Tensor> num_token_non_padded);

// attention
void decode_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
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
at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_vnni);

// igemm
at::Tensor int8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales1,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// fp8 gemm
at::Tensor fp8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    std::vector<int64_t> block_size,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// quant + igemm
at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

// bmm
void bmm_cpu(at::Tensor& out, at::Tensor& mat1, at::Tensor& mat2, bool is_vnni, const std::optional<at::Tensor>& scale);

// fused moe
at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& a1_scale,
    const std::optional<at::Tensor>& a2_scale,
    bool is_vnni);

at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& a1_scale,
    const std::optional<at::Tensor>& a2_scale,
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
    bool use_fp8_w8a16,
    std::optional<at::Tensor> q_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    std::optional<at::Tensor> kv_a_proj_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope_fused_weight(
    at::Tensor& hidden_states,
    at::Tensor& qkv_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor> qkv_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size,
    int64_t q_lora_rank,
    int64_t kv_lora_rank,
    int64_t qk_rope_head_dim);

// shared memory init
void initialize(int64_t size, int64_t rank);

// shared mmeory all_reduce
void shm_allreduce(at::Tensor& data, int64_t op);

// shared memory all_gather
at::Tensor shm_allgather(at::Tensor& data, int64_t dim);

// rope
std::tuple<at::Tensor, at::Tensor> rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox);

// CPU and memory binding
std::string init_cpu_threads_env(const std::string& cpu_ids);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  // activation
  m.def("silu_and_mul_cpu(Tensor input) -> Tensor");
  m.impl("silu_and_mul_cpu", torch::kCPU, &silu_and_mul_cpu);
  m.def("gelu_tanh_and_mul_cpu(Tensor input) -> Tensor");
  m.impl("gelu_tanh_and_mul_cpu", torch::kCPU, &gelu_tanh_and_mul_cpu);
  m.def("gelu_and_mul_cpu(Tensor input) -> Tensor");
  m.impl("gelu_and_mul_cpu", torch::kCPU, &gelu_and_mul_cpu);

  // norm
  m.def("rmsnorm_cpu(Tensor input, Tensor weight, float eps) -> Tensor");
  m.impl("rmsnorm_cpu", torch::kCPU, &rmsnorm_cpu);
  m.def("l2norm_cpu(Tensor input, float eps) -> Tensor");
  m.impl("l2norm_cpu", torch::kCPU, &l2norm_cpu);
  m.def("fused_rmsnorm_gated_cpu(Tensor input, Tensor weight, Tensor gate, float eps) -> Tensor");
  m.impl("fused_rmsnorm_gated_cpu", torch::kCPU, &fused_rmsnorm_gated_cpu);
  m.def("fused_add_rmsnorm_cpu(Tensor(a!) input, Tensor residual, Tensor weight, float eps) -> ()");
  m.impl("fused_add_rmsnorm_cpu", torch::kCPU, &fused_add_rmsnorm_cpu);

  // topk
  m.def("topk_sigmoid_cpu(Tensor hidden_states, Tensor gating_output, int topk, bool renormalize) -> (Tensor, Tensor)");
  m.impl("topk_sigmoid_cpu", torch::kCPU, &topk_sigmoid_cpu);
  m.def("topk_softmax_cpu(Tensor hidden_states, Tensor gating_output, int topk, bool renormalize) -> (Tensor, Tensor)");
  m.impl("topk_softmax_cpu", torch::kCPU, &topk_softmax_cpu);
  m.def(
      "grouped_topk_cpu(Tensor hidden_states, Tensor gating_output, int topk, bool renormalize, int num_expert_group, "
      "int topk_group, int num_fused_shared_experts, float? routed_scaling_factor, Tensor? num_token_non_padded) -> "
      "(Tensor, Tensor)");
  m.impl("grouped_topk_cpu", torch::kCPU, &grouped_topk_cpu);

  // biased group topk
  m.def(
      "biased_grouped_topk_cpu(Tensor hidden_states, Tensor gating_output, Tensor correction_bias, int topk, bool "
      "renormalize, int num_expert_group, int topk_group, int num_fused_shared_experts, float? routed_scaling_factor, "
      "Tensor? num_token_non_padded) -> (Tensor, Tensor)");
  m.impl("biased_grouped_topk_cpu", torch::kCPU, &biased_grouped_topk_cpu);

  // decode
  m.def(
      "decode_attention_cpu(Tensor query, Tensor k_cache, Tensor v_cahce, Tensor(a!) output, Tensor key, Tensor value, "
      "Tensor loc, Tensor attn_logits, Tensor req_to_token, Tensor req_pool_indices, Tensor seq_lens, float sm_scale, "
      "float logit_cap) -> ()");
  m.impl("decode_attention_cpu", torch::kCPU, &decode_attention_cpu);

  // extend
  m.def(
      "extend_attention_cpu(Tensor q_extend, Tensor k_extend, Tensor v_extend, Tensor(a!) o_extend, Tensor k_buffer, "
      "Tensor v_buffer, Tensor req_to_token, Tensor req_pool_indices, Tensor seq_lens, Tensor extend_seq_lens, Tensor "
      "extend_start_loc, int max_len_extend, float sm_scale, float logit_cap) -> ()");
  m.impl("extend_attention_cpu", torch::kCPU, &extend_attention_cpu);

  // weight prepack
  m.def("convert_weight_packed(Tensor weight) -> Tensor");
  m.impl("convert_weight_packed", torch::kCPU, &convert_weight_packed);

  // quant
  m.def("per_token_quant_int8_cpu(Tensor A) -> (Tensor, Tensor)");
  m.impl("per_token_quant_int8_cpu", torch::kCPU, &per_token_quant_int8_cpu);

  // gemm
  m.def("weight_packed_linear(Tensor mat1, Tensor mat2, Tensor? bias, bool is_vnni) -> Tensor");
  m.impl("weight_packed_linear", torch::kCPU, &weight_packed_linear);

  // igemm
  m.def(
      "int8_scaled_mm_cpu(Tensor mat1, Tensor mat2, Tensor scales1, Tensor scales2, Tensor? bias, ScalarType "
      "out_dtype, bool is_vnni) -> Tensor");
  m.impl("int8_scaled_mm_cpu", torch::kCPU, &int8_scaled_mm_cpu);

  // fp8 gemm
  m.def(
      "fp8_scaled_mm_cpu(Tensor mat1, Tensor mat2, Tensor scales2, int[] block_size, Tensor? bias, ScalarType "
      "out_dtype, bool is_vnni) -> Tensor");
  m.impl("fp8_scaled_mm_cpu", torch::kCPU, &fp8_scaled_mm_cpu);

  // quant + igemm
  m.def(
      "int8_scaled_mm_with_quant(Tensor mat1, Tensor mat2, Tensor scales2, Tensor? bias, ScalarType out_dtype, bool "
      "is_vnni) -> Tensor");
  m.impl("int8_scaled_mm_with_quant", torch::kCPU, &int8_scaled_mm_with_quant);

  // bmm
  m.def("bmm_cpu(Tensor(a!) out, Tensor mat1, Tensor mat2, bool is_vnni, Tensor? scale) -> ()");
  m.impl("bmm_cpu", torch::kCPU, &bmm_cpu);

  // moe
  m.def(
      "fused_experts_cpu(Tensor hidden_states, Tensor w1, Tensor w2, Tensor topk_weights, Tensor topk_ids, bool "
      "inplace, bool use_int8_w8a8, bool use_fp8_w8a16, Tensor? w1_scale, Tensor? w2_scale, int[]? block_size, Tensor? "
      "a1_scale, Tensor? a2_scale, bool "
      "is_vnni) -> Tensor");
  m.impl("fused_experts_cpu", torch::kCPU, &fused_experts_cpu);

  // weight absorption
  m.def(
      "qkv_proj_with_rope(Tensor hidden_states, Tensor q_a_proj_weight, Tensor q_b_proj_weight, Tensor "
      "kv_a_proj_weight, Tensor w_kc, Tensor q_a_layernorm_weight, Tensor kv_a_layernorm_weight, Tensor positions, "
      "Tensor cos_sin_cache, float eps, bool use_int8_w8a8, bool use_fp8_w8a16, Tensor? q_a_proj_scale, Tensor? "
      "q_b_proj_scale, Tensor? "
      "kv_a_proj_scale, bool is_vnni, int[]? block_size) -> (Tensor, Tensor, Tensor)");
  m.impl("qkv_proj_with_rope", torch::kCPU, &qkv_proj_with_rope);
  m.def(
      "qkv_proj_with_rope_fused_weight(Tensor hidden_states, Tensor qkv_a_proj_weight, Tensor q_b_proj_weight, "
      "Tensor w_kc, Tensor q_a_layernorm_weight, Tensor kv_a_layernorm_weight, Tensor positions, "
      "Tensor cos_sin_cache, float eps, bool use_int8_w8a8, bool use_fp8_w8a16, Tensor? qkv_a_proj_scale, Tensor? "
      "q_b_proj_scale,"
      "bool is_vnni, int[]? block_size, int q_lora_rank, int kv_lora_rank,"
      "int qk_rope_head_dim) -> (Tensor, Tensor, Tensor)");
  m.impl("qkv_proj_with_rope_fused_weight", torch::kCPU, &qkv_proj_with_rope_fused_weight);

  // shared expert
  m.def(
      "shared_expert_cpu(Tensor hidden_states, Tensor w1, Tensor w2, Tensor fused_experts_out, float "
      "routed_scaling_factor, bool inplace, bool use_int8_w8a8, bool use_fp8_w8a16, Tensor? w1_scale, Tensor? "
      "w2_scale, int[]? block_size, Tensor? a1_scale, Tensor? a2_scale, bool is_vnni) -> Tensor");
  m.impl("shared_expert_cpu", torch::kCPU, &shared_expert_cpu);

  // all reduce
  m.def("initialize(int size, int rank) -> ()");
  m.def("shm_allreduce(Tensor(a!) data, int reduce_op) -> ()");
  m.impl("shm_allreduce", torch::kCPU, &shm_allreduce);
  m.def("shm_allgather(Tensor data, int dim) -> Tensor");
  m.impl("shm_allgather", torch::kCPU, &shm_allgather);

  // rope
  m.def(
      "rotary_embedding_cpu(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, "
      "bool is_neox) -> (Tensor, Tensor)");
  m.impl("rotary_embedding_cpu", torch::kCPU, &rotary_embedding_cpu);

  // CPU and memory binding
  m.def("init_cpu_threads_env(str cpu_ids) -> str");
}

TORCH_LIBRARY_IMPL(sgl_kernel, CatchAll, m) {
  m.impl("init_cpu_threads_env", init_cpu_threads_env);
  m.impl("initialize", &initialize);
}

REGISTER_EXTENSION(common_ops)
