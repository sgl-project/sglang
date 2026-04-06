#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#if defined(RVV_HAS_ZVFH)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvfh"))), apply_to = function)
#else
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#endif
#endif

#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

// silu_and_mul
at::Tensor silu_and_mul_cpu(at::Tensor& input);

// gelu_and_mul
at::Tensor gelu_tanh_and_mul_cpu(const at::Tensor& input);
at::Tensor gelu_and_mul_cpu(const at::Tensor& input);

// l2norm
at::Tensor l2norm_cpu(at::Tensor& input, double eps);

// rmsnorm
at::Tensor rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps);
at::Tensor gemma_rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps);
at::Tensor gemma3_rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps);

// layernorm
at::Tensor
layernorm_cpu(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias, double eps);

// fused_rmsnorm_gated
at::Tensor fused_rmsnorm_gated_cpu(at::Tensor& input, at::Tensor& weight, at::Tensor& gate, double eps);

// fused_add_rmsnorm
void fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps);
void gemma_fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps);

// fused_add_layernorm
at::Tensor fused_add_layernorm_cpu(
    const at::Tensor& input,
    at::Tensor& residual,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    double eps);

// decode attention
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

// extend attention
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
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_packed);

// igemm
at::Tensor int8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales1,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_packed);

// quant + igemm
at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_packed);

// rope
std::tuple<at::Tensor, at::Tensor> rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox);

// [NOTE] When registering kernels, we should accurately describe the in-place information.
// Taking fused_add_rmsnorm_cpu as an example, add `Tensor(a!)` modifier to all tensors that
// will be modified in-place to avoid incorrect fusing and execution order on graph mode.
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
  m.def("gemma_rmsnorm_cpu(Tensor input, Tensor weight, float eps) -> Tensor");
  m.impl("gemma_rmsnorm_cpu", torch::kCPU, &gemma_rmsnorm_cpu);
  m.def("gemma3_rmsnorm_cpu(Tensor input, Tensor weight, float eps) -> Tensor");
  m.impl("gemma3_rmsnorm_cpu", torch::kCPU, &gemma3_rmsnorm_cpu);
  m.def("layernorm_cpu(Tensor input, Tensor weight, Tensor? bias, float eps) -> Tensor");
  m.impl("layernorm_cpu", torch::kCPU, &layernorm_cpu);
  m.def("l2norm_cpu(Tensor input, float eps) -> Tensor");
  m.impl("l2norm_cpu", torch::kCPU, &l2norm_cpu);
  m.def("fused_rmsnorm_gated_cpu(Tensor input, Tensor weight, Tensor gate, float eps) -> Tensor");
  m.impl("fused_rmsnorm_gated_cpu", torch::kCPU, &fused_rmsnorm_gated_cpu);
  m.def("fused_add_rmsnorm_cpu(Tensor(a!) input, Tensor(a!) residual, Tensor weight, float eps) -> ()");
  m.impl("fused_add_rmsnorm_cpu", torch::kCPU, &fused_add_rmsnorm_cpu);
  m.def("gemma_fused_add_rmsnorm_cpu(Tensor(a!) input, Tensor(a!) residual, Tensor weight, float eps) -> ()");
  m.impl("gemma_fused_add_rmsnorm_cpu", torch::kCPU, &gemma_fused_add_rmsnorm_cpu);
  m.def(
      "fused_add_layernorm_cpu(Tensor input, Tensor residual, Tensor weight, Tensor? bias, float eps) -> "
      "Tensor");
  m.impl("fused_add_layernorm_cpu", torch::kCPU, &fused_add_layernorm_cpu);

  // decode
  m.def(
      "decode_attention_cpu(Tensor query, Tensor k_cache, Tensor v_cache, Tensor(a!) output, Tensor key, Tensor value, "
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
  m.def("weight_packed_linear(Tensor mat1, Tensor mat2, Tensor? bias, bool is_packed) -> Tensor");
  m.impl("weight_packed_linear", torch::kCPU, &weight_packed_linear);

  // igemm
  m.def(
      "int8_scaled_mm_cpu(Tensor mat1, Tensor mat2, Tensor scales1, Tensor scales2, Tensor? bias, ScalarType "
      "out_dtype, bool is_packed) -> Tensor");
  m.impl("int8_scaled_mm_cpu", torch::kCPU, &int8_scaled_mm_cpu);

  // quant + igemm
  m.def(
      "int8_scaled_mm_with_quant(Tensor mat1, Tensor mat2, Tensor scales2, Tensor? bias, ScalarType out_dtype, bool "
      "is_packed) -> Tensor");
  m.impl("int8_scaled_mm_with_quant", torch::kCPU, &int8_scaled_mm_with_quant);

  // rope
  m.def(
      "rotary_embedding_cpu(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, "
      "bool is_neox) -> (Tensor, Tensor)");
  m.impl("rotary_embedding_cpu", torch::kCPU, &rotary_embedding_cpu);
}

#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang attribute pop
#endif

REGISTER_EXTENSION(common_ops)
