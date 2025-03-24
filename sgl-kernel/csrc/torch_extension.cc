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

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  /*
   * From csrc/allreduce
   */
  m.def(
      "init_custom_ar(int rank_id, int world_size, Tensor rank_data, int[] buffers, int[] tmp_result_buffers, int[] "
      "barrier_in, int[] barrier_out) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  m.def("dispose", &dispose);

  m.def("all_reduce(int fa, Tensor inp, Tensor! out) -> ()");
  m.impl("all_reduce", torch::kCUDA, &all_reduce);

  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);

  /*
   * From csrc/attention
   */
  m.def(
      "lightning_attention_decode(Tensor q, Tensor k, Tensor v, Tensor past_kv, Tensor slope, Tensor! output, Tensor! "
      "new_kv) -> ()");
  m.impl("lightning_attention_decode", torch::kCUDA, &lightning_attention_decode);

  /*
   * From csrc/elementwise
   */
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, int cuda_stream) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);

  m.def("gemma_rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, int cuda_stream) -> ()");
  m.impl("gemma_rmsnorm", torch::kCUDA, &gemma_rmsnorm);

  m.def("gemma_fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, int cuda_stream) -> ()");
  m.impl("gemma_fused_add_rmsnorm", torch::kCUDA, &gemma_fused_add_rmsnorm);

  m.def("silu_and_mul(Tensor! out, Tensor input, int cuda_stream) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input, int cuda_stream) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input, int cuda_stream) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  m.def(
      "apply_rope_pos_ids_cos_sin_cache(Tensor q, Tensor k, Tensor! q_rope, Tensor! k_rope, Tensor cos_sin_cache, "
      "Tensor pos_ids, bool interleave, int cuda_stream) -> ()");
  m.impl("apply_rope_pos_ids_cos_sin_cache", torch::kCUDA, &apply_rope_pos_ids_cos_sin_cache);

  /*
   * From csrc/gemm
   */
  m.def("awq_dequantize(Tensor qweight, Tensor scales, Tensor qzeros) -> Tensor");
  m.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  m.def(
      "int8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");
  m.impl("int8_scaled_mm", torch::kCUDA, &int8_scaled_mm);

  m.def(
      "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");
  m.impl("fp8_scaled_mm", torch::kCUDA, &fp8_scaled_mm);

  m.def(
      "fp8_blockwise_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype) -> "
      "Tensor");
  m.impl("fp8_blockwise_scaled_mm", torch::kCUDA, &fp8_blockwise_scaled_mm);

  m.def(
      "sgl_per_token_group_quant_fp8(Tensor input, Tensor output_q, Tensor output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max) -> ()");
  m.impl("sgl_per_token_group_quant_fp8", torch::kCUDA, &sgl_per_token_group_quant_fp8);

  m.def(
      "sgl_per_token_group_quant_int8(Tensor input, Tensor output_q, Tensor output_s, int group_size,"
      " float eps, float int8_min, float int8_max) -> ()");
  m.impl("sgl_per_token_group_quant_int8", torch::kCUDA, &sgl_per_token_group_quant_int8);

  m.def("sgl_per_tensor_quant_fp8(Tensor input, Tensor output_q, Tensor output_s, bool is_static) -> ()");
  m.impl("sgl_per_tensor_quant_fp8", torch::kCUDA, &sgl_per_tensor_quant_fp8);

  m.def("sgl_per_token_quant_fp8(Tensor input, Tensor output_q, Tensor output_s) -> ()");
  m.impl("sgl_per_token_quant_fp8", torch::kCUDA, &sgl_per_token_quant_fp8);

  m.def(
      "cublas_grouped_gemm(Tensor[] inputs, Tensor[] weights, Tensor[] outputs,"
      " ScalarType out_dtype, int cublas_handle, int cuda_stream) -> ()");
  m.impl("cublas_grouped_gemm", torch::kCUDA, &cublas_grouped_gemm);

  /*
   * From csrc/moe
   */
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! token_cnts_buffer, Tensor! cumsum_buffer) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  /*
   * From csrc/speculative
   */
  m.def(
      "tree_speculative_sampling_target_only(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor uniform_samples, Tensor target_probs, Tensor draft_probs, "
      "float threshold_single, float threshold_acc, "
      "bool deterministic, int cuda_stream) -> ()");
  m.impl("tree_speculative_sampling_target_only", torch::kCUDA, &tree_speculative_sampling_target_only);

  m.def(
      "verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor target_predict, int cuda_stream) -> ()");
  m.impl("verify_tree_greedy", torch::kCUDA, &verify_tree_greedy);

  m.def(
      "build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
      "Tensor! tree_mask, Tensor! positions, Tensor! retrive_index, Tensor! retrive_next_token, "
      "Tensor! retrive_next_sibling, int topk, int depth, int draft_token_num) -> ()");
  m.impl("build_tree_kernel_efficient", torch::kCUDA, &build_tree_kernel_efficient);

  m.def("segment_packbits(Tensor x, Tensor input_indptr, Tensor output_indptr, Tensor! y, int cuda_stream) -> ()");
  m.impl("segment_packbits", torch::kCUDA, &segment_packbits);

  /*
   * From FlashInfer
   */
  m.def(
      "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, int "
      "cublas_handle, int cuda_stream) -> ()");
  m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);

  m.def(
      "min_p_sampling_from_probs(Tensor probs, Tensor uniform_samples, Tensor! samples, Tensor? maybe_min_p_arr, float "
      "min_p_val, bool deterministic, int cuda_stream) -> ()");
  m.impl("min_p_sampling_from_probs", torch::kCUDA, &min_p_sampling_from_probs);

  m.def(
      "top_k_renorm_probs_wrapper(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val, int "
      "cuda_stream) -> ()");
  m.impl("top_k_renorm_probs_wrapper", torch::kCUDA, &top_k_renorm_probs_wrapper);

  m.def(
      "top_p_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_p_arr, float top_p_val, int "
      "cuda_stream) -> ()");
  m.impl("top_p_renorm_probs", torch::kCUDA, &top_p_renorm_probs);

  m.def(
      "top_k_top_p_sampling_from_probs(Tensor probs, Tensor uniform_samples, Tensor! samples, Tensor! success, Tensor? "
      "maybe_top_k_arr, float top_k_val, Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, int "
      "cuda_stream) -> ()");
  m.impl("top_k_top_p_sampling_from_probs", torch::kCUDA, &top_k_top_p_sampling_from_probs);

  m.def(
      "top_p_sampling_from_probs(Tensor probs, Tensor uniform_samples, Tensor! samples, Tensor! success, Tensor? "
      "maybe_top_p_arr, float top_p_val, bool deterministic, int cuda_stream) -> ()");
  m.impl("top_p_sampling_from_probs", torch::kCUDA, &top_p_sampling_from_probs);
}

REGISTER_EXTENSION(common_ops)
