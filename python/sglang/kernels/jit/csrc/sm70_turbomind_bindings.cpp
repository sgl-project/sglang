// SPDX-License-Identifier: Apache-2.0
// Private bindings for the LMDeploy TurboMind SM70 kernels carried by 1Cat-vLLM.

#include <torch/all.h>
#include <torch/library.h>

std::vector<torch::Tensor> awq_sm70_prepare(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t, bool);
std::vector<torch::Tensor> uint4_sm70_prepare(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t, bool);
std::vector<torch::Tensor> fp8_sm70_prepare(
    torch::Tensor, torch::Tensor, int64_t, bool);
torch::Tensor awq_gemm_sm70(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
void awq_gemm_sm70_out(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t, bool);
void awq_sm70_dequantize_out(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void fp8_sm70_dequantize_out(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
std::vector<torch::Tensor> sglang_sm70_f16_prepare(torch::Tensor);
void fp8_gemm_sm70_out(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t, bool);
void fp8_gemm_sm70_prefill_dispatch_out(
    torch::Tensor, int64_t, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t, bool, int64_t);
std::vector<torch::Tensor> awq_moe_build_strided_ptrs(
    torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
void awq_moe_gemm_sm70_out(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int64_t, int64_t, int64_t, int64_t, bool);
std::vector<torch::Tensor> sm70_f16_moe_build_strided_ptrs(
    torch::Tensor, int64_t, int64_t);
void sm70_f16_moe_gemm_sm70_out(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    int64_t, int64_t, bool);
int64_t moe_permute_sort_workspace_size(int64_t, int64_t);
void sm70_fp8_e5m2_cache_write(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    std::optional<torch::Tensor>, std::optional<torch::Tensor>);

TORCH_LIBRARY(sglang_sm70_turbomind, ops) {
  ops.def(
      "fp8_e5m2_cache_write(Tensor key, Tensor value, Tensor(a!) key_cache, "
      "Tensor(b!) value_cache, Tensor locations, Tensor? k_scale, "
      "Tensor? v_scale) -> ()");
  ops.impl(
      "fp8_e5m2_cache_write", torch::kCUDA,
      &sm70_fp8_e5m2_cache_write);
  ops.def(
      "awq_prepare(Tensor qweight, Tensor scales, Tensor qzeros, "
      "int group_size, bool interleave) -> Tensor[]");
  ops.impl("awq_prepare", torch::kCUDA, &awq_sm70_prepare);
  ops.def(
      "uint4_prepare(Tensor qweight, Tensor scales, Tensor qzeros, "
      "int group_size, bool interleave) -> Tensor[]");
  ops.impl("uint4_prepare", torch::kCUDA, &uint4_sm70_prepare);
  ops.def(
      "fp8_prepare(Tensor qweight, Tensor scales, int group_size, "
      "bool interleave) -> Tensor[]");
  ops.impl("fp8_prepare", torch::kCUDA, &fp8_sm70_prepare);
  ops.def("f16_prepare(Tensor weight) -> Tensor[]");
  ops.impl("f16_prepare", torch::kCUDA, &sglang_sm70_f16_prepare);
  ops.def(
      "fp8_gemm(Tensor(a!) out, Tensor input, Tensor qweight, Tensor scales, "
      "int group_size, int k_ld, int q_ld, bool gated_silu) -> ()");
  ops.impl("fp8_gemm", torch::kCUDA, &fp8_gemm_sm70_out);
  ops.def(
      "fp8_dequantize_out(Tensor(a!) out, Tensor qweight, Tensor scales, "
      "int group_size) -> ()");
  ops.impl(
      "fp8_dequantize_out", torch::kCUDA, &fp8_sm70_dequantize_out);
  ops.def(
      "fp8_prefill_dispatch(Tensor(a!) out, int dense_weight_ptr, "
      "Tensor input, Tensor qweight, Tensor scales, int group_size, "
      "int k_ld, int q_ld, bool gated_silu, int min_prefill_m) -> ()");
  ops.impl(
      "fp8_prefill_dispatch", torch::kCUDA,
      &fp8_gemm_sm70_prefill_dispatch_out);
  ops.def(
      "awq_gemm(Tensor input, Tensor qweight, Tensor scales, int group_size, "
      "int k_ld, int q_ld) -> Tensor");
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm_sm70);
  ops.def(
      "awq_gemm_out(Tensor(a!) out, Tensor input, Tensor qweight, "
      "Tensor scales, int group_size, int k_ld, int q_ld, "
      "bool gated_silu) -> ()");
  ops.impl("awq_gemm_out", torch::kCUDA, &awq_gemm_sm70_out);
  ops.def(
      "awq_dequantize_out(Tensor(a!) out, Tensor qweight, Tensor scales, "
      "int group_size) -> ()");
  ops.impl("awq_dequantize_out", torch::kCUDA, &awq_sm70_dequantize_out);
  ops.def(
      "build_ptrs(Tensor weights, Tensor scales, int k_ld, int q_ld, "
      "int num_experts) -> Tensor[]");
  ops.impl("build_ptrs", torch::kCUDA, &awq_moe_build_strided_ptrs);
  ops.def(
      "gemm(Tensor(a!) out, Tensor sorted_input, Tensor expert_offsets, "
      "Tensor strided_ptrs_w, Tensor strided_ptrs_s, int num_experts, "
      "int k, int n, int group_size, bool gated_silu) -> ()");
  ops.impl("gemm", torch::kCUDA, &awq_moe_gemm_sm70_out);
  ops.def(
      "f16_moe_build_ptrs(Tensor weights, int k_ld, int num_experts) "
      "-> Tensor[]");
  ops.impl(
      "f16_moe_build_ptrs", torch::kCUDA,
      &sm70_f16_moe_build_strided_ptrs);
  ops.def(
      "f16_moe_gemm(Tensor(a!) out, Tensor sorted_input, "
      "Tensor expert_offsets, Tensor strided_ptrs_w, int num_experts, "
      "int k, int n, bool gated_silu) -> ()");
  ops.impl("f16_moe_gemm", torch::kCUDA, &sm70_f16_moe_gemm_sm70_out);
  ops.def(
      "moe_permute_with_scratch(Tensor input, Tensor topk_ids, "
      "Tensor token_expert_indices, Tensor? expert_map, int n_expert, "
      "int n_local_expert, int topk, Tensor(a!) permuted_input, "
      "Tensor(b!) expert_first_token_offset, Tensor(c!) inv_permuted_idx, "
      "Tensor(d!) permuted_idx, Tensor(e!) sort_workspace, "
      "Tensor(f!) permuted_experts_id, Tensor(g!) sorted_row_idx, "
      "Tensor(h!) topk_ids_for_sort) -> ()");
  ops.def(
      "moe_unpermute(Tensor permuted_hidden_states, Tensor topk_weights, "
      "Tensor inv_permuted_idx, Tensor? expert_first_token_offset, "
      "int topk, Tensor(a!) hidden_states) -> ()");
  ops.def(
      "moe_permute_sort_workspace_size(int num_expanded_rows, "
      "int n_expert) -> int");
  ops.impl(
      "moe_permute_sort_workspace_size",
      &moe_permute_sort_workspace_size);
}
