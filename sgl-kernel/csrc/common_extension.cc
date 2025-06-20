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
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From csrc/allreduce
   */

  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);

  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.impl("all_reduce", torch::kCUDA, &all_reduce);

  m.def("mscclpp_generate_unique_id", &mscclpp_generate_unique_id);
  m.def(
      "mscclpp_init_context(Tensor unique_id, int rank, int world_size, Tensor scratch, Tensor put_buffer, "
      "int nranks_per_node, int[] rank_to_node, int[] rank_to_ib, int context_selection) -> int");
  m.impl("mscclpp_init_context", torch::kCUDA, &mscclpp_init_context);

  m.def("mscclpp_allreduce(int context, Tensor inp, Tensor! out, int nthreads, int nblocks) -> ()");
  m.impl("mscclpp_allreduce", torch::kCUDA, &mscclpp_allreduce);
  /*
   * From csrc/attention
   */
  m.def(
      "lightning_attention_decode(Tensor q, Tensor k, Tensor v, Tensor past_kv, Tensor slope, Tensor! output, Tensor! "
      "new_kv) -> ()");
  m.impl("lightning_attention_decode", torch::kCUDA, &lightning_attention_decode);
  m.def("merge_state(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state", torch::kCUDA, &merge_state);
  m.def("merge_state_v2(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state_v2", torch::kCUDA, &merge_state_v2);
  m.def(
      "cutlass_mla_decode(Tensor! out, Tensor q_nope, Tensor q_pe, Tensor kv_c_and_k_pe_cache, Tensor seq_lens, Tensor "
      "page_table, Tensor! workspace, float sm_scale, int num_kv_splits) -> ()");
  m.impl("cutlass_mla_decode", torch::kCUDA, &cutlass_mla_decode);
  m.def("cutlass_mla_get_workspace_size", &cutlass_mla_get_workspace_size);

  /*
   * From csrc/elementwise
   */
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);

  m.def("gemma_rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("gemma_rmsnorm", torch::kCUDA, &gemma_rmsnorm);

  m.def("gemma_fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
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
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");
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
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b,"
      "                      Tensor block_scale_a, Tensor block_scale_b,"
      "                      Tensor alpha) -> ()");
  m.impl("cutlass_scaled_fp4_mm", torch::kCUDA, &cutlass_scaled_fp4_mm);

  m.def(
      "scaled_fp4_quant(Tensor! output, Tensor! input,"
      "                 Tensor! output_scale, Tensor! input_scale) -> ()");
  m.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

  // Compute NVFP4 experts quantization.
  m.def(
      "scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");
  m.impl("scaled_fp4_experts_quant", torch::kCUDA, &scaled_fp4_experts_quant);

  m.def(
      "cutlass_fp4_group_mm(Tensor! output, Tensor a, Tensor b,"
      "Tensor a_blockscale, Tensor b_blockscale, Tensor alphas,"
      "Tensor ab_strides, Tensor c_strides, Tensor problem_sizes,"
      " Tensor expert_offsets, Tensor sf_offsets) -> ()");
  m.impl("cutlass_fp4_group_mm", torch::kCUDA, &cutlass_fp4_group_mm);

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

  m.def(
      "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, int topk_group, int topk, int "
      "num_fused_shared_experts, float routed_scaling_factor) -> "
      "(Tensor[])");
  m.impl("moe_fused_gate", torch::kCUDA, &moe_fused_gate);
  m.def(
      "ep_moe_pre_reorder(Tensor input, Tensor gateup_input, Tensor src2dst, Tensor topk_ids, Tensor "
      "a1_scales, int start_expert_id, int end_expert_id, int topk, bool use_per_token_if_dynamic) -> ()");
  m.impl("ep_moe_pre_reorder", torch::kCUDA, &ep_moe_pre_reorder);
  m.def(
      "ep_moe_silu_and_mul(Tensor gateup_output, Tensor down_input, Tensor reorder_topk_ids, Tensor scales, int "
      "start_expert_id, int end_expert_id) -> ()");
  m.impl("ep_moe_silu_and_mul", torch::kCUDA, &ep_moe_silu_and_mul);
  m.def(
      "ep_moe_post_reorder(Tensor down_output, Tensor output, Tensor src2dst, Tensor topk_ids, Tensor "
      "topk_weights, int start_expert_id, int end_expert_id, int topk) -> ()");
  m.impl("ep_moe_post_reorder", torch::kCUDA, &ep_moe_post_reorder);
  m.def(
      "fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, Tensor "
      "a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
      "stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, Tensor problem_sizes, Tensor "
      "expert_offsets, Tensor workspace) -> ()");
  m.impl("fp8_blockwise_scaled_grouped_mm", torch::kCUDA, &fp8_blockwise_scaled_grouped_mm);
  m.def(
      "prepare_moe_input(Tensor topk_ids, Tensor expert_offsets, Tensor? blockscale_offsets, Tensor problem_sizes1,"
      " Tensor problem_sizes2, Tensor input_permutation, Tensor output_permutation, int num_experts, int n, int k) -> "
      "()");
  m.impl("prepare_moe_input", torch::kCUDA, &prepare_moe_input);

  m.def("shuffle_rows(Tensor input, Tensor dst2src_map, Tensor output) -> ()");
  m.impl("shuffle_rows", torch::kCUDA, &shuffle_rows);
  m.def("apply_shuffle_mul_sum(Tensor input, Tensor output, Tensor permutation, Tensor? factors) -> ()");
  m.impl("apply_shuffle_mul_sum", torch::kCUDA, &apply_shuffle_mul_sum);

  /*
   * From csrc/speculative
   */
  m.def(
      "tree_speculative_sampling_target_only(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor uniform_samples, Tensor uniform_samples_for_final_sampling, Tensor target_probs, Tensor draft_probs, "
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

  m.def(
      "segment_packbits(Tensor x, Tensor input_indptr, Tensor output_indptr, Tensor! y, int batch_size, "
      "int cuda_stream) -> ()");
  m.impl("segment_packbits", torch::kCUDA, &segment_packbits);

  /*
   * From FlashInfer
   */
  m.def(
      "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, int "
      "cublas_handle, int cuda_stream) -> ()",
      {at::Tag::needs_fixed_stride_order});
  m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);

  m.def(
      "min_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? maybe_min_p_arr, float "
      "min_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("min_p_sampling_from_probs", torch::kCUDA, &min_p_sampling_from_probs);

  m.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  m.impl("top_k_renorm_probs", torch::kCUDA, &top_k_renorm_probs);

  m.def("top_p_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_p_arr, float top_p_val) -> ()");
  m.impl("top_p_renorm_probs", torch::kCUDA, &top_p_renorm_probs);

  m.def(
      "top_k_top_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? maybe_top_k_arr, "
      "float top_k_val, Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("top_k_top_p_sampling_from_probs", torch::kCUDA, &top_k_top_p_sampling_from_probs);

  m.def(
      "top_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? "
      "maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("top_p_sampling_from_probs", torch::kCUDA, &top_p_sampling_from_probs);

  /*
   * From Sparse Flash Attention
   */
  m.def(
      "fwd_sparse(Tensor! q, Tensor k, Tensor v, "
      "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
      "Tensor!? out, Tensor? alibi_slopes, "
      "float p_dropout, float softmax_scale, bool is_causal, "
      "float softcap, bool return_softmax, Generator? gen)"
      "-> Tensor[]");
  m.impl("fwd_sparse", torch::kCUDA, &flash::mha_fwd_sparse);

  m.def(
      "varlen_fwd_sparse(Tensor! q, Tensor k, Tensor v, "
      "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
      "Tensor!? out, Tensor cu_seqlens_q, "
      "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? alibi_slopes, "
      "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
      "bool is_causal, float softcap, bool return_softmax, "
      "Generator? gen) -> Tensor[]");
  m.impl("varlen_fwd_sparse", torch::kCUDA, &flash::mha_varlen_fwd_sparse);

  // Sparse Attention utils
  m.def(
      "convert_vertical_slash_indexes("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  m.impl("convert_vertical_slash_indexes", torch::kCUDA, &convert_vertical_slash_indexes);

  m.def(
      "convert_vertical_slash_indexes_mergehead("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   Tensor vertical_indices_count, Tensor slash_indices_count, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  m.impl("convert_vertical_slash_indexes_mergehead", torch::kCUDA, &convert_vertical_slash_indexes_mergehead);

  /*
   * From XGrammar
   */
  m.def("apply_token_bitmask_inplace_cuda(Tensor logits, Tensor bitmask, Tensor? indices=None) -> ()");
  m.impl("apply_token_bitmask_inplace_cuda", &ApplyTokenBitmaskInplace);

  /*
   * From QServe
   */
  m.def(
      "qserve_w4a8_per_chn_gemm(Tensor _in_feats, Tensor _kernel, Tensor _wscales, Tensor _ascales, Tensor _w_szs, "
      "Tensor _a_ssums, Tensor! _out_feats) -> ()");
  m.impl("qserve_w4a8_per_chn_gemm", torch::kCUDA, &qserve_w4a8_per_chn_gemm);

  m.def(
      "qserve_w4a8_per_group_gemm(Tensor _in_feats, Tensor _kernel, Tensor _zeros, Tensor _scales_i8, Tensor _wscales, "
      "Tensor _ascales, Tensor! _out_feats) -> ()");
  m.impl("qserve_w4a8_per_group_gemm", torch::kCUDA, &qserve_w4a8_per_group_gemm);
}

REGISTER_EXTENSION(common_ops)
