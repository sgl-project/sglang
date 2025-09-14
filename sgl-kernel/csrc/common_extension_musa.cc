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
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  /*
   * From csrc/elementwise
   */
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("rmsnorm", torch::kMUSA, &rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("fused_add_rmsnorm", torch::kMUSA, &sgl_fused_add_rmsnorm);

  m.def("gemma_rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("gemma_rmsnorm", torch::kMUSA, &gemma_rmsnorm);

  m.def("gemma_fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("gemma_fused_add_rmsnorm", torch::kMUSA, &gemma_fused_add_rmsnorm);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kMUSA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kMUSA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kMUSA, &gelu_and_mul);

  /*
   * From csrc/gemm
   */
  m.def("awq_dequantize(Tensor qweight, Tensor scales, Tensor qzeros) -> Tensor");
  m.impl("awq_dequantize", torch::kMUSA, &awq_dequantize);

  m.def("dsv3_fused_a_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");
  m.impl("dsv3_fused_a_gemm", torch::kMUSA, &dsv3_fused_a_gemm);

  m.def("dsv3_router_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");
  m.impl("dsv3_router_gemm", torch::kMUSA, &dsv3_router_gemm);

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
  m.impl("init_custom_ar", torch::kMUSA, &init_custom_ar);

  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.impl("all_reduce", torch::kMUSA, &all_reduce);

  /*
   * From csrc/attention
   */
  m.def(
      "lightning_attention_decode(Tensor q, Tensor k, Tensor v, Tensor past_kv, Tensor slope, Tensor! output, Tensor! "
      "new_kv) -> ()");
  m.impl("lightning_attention_decode", torch::kMUSA, &lightning_attention_decode);

  m.def("merge_state_v2(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state_v2", torch::kMUSA, &merge_state_v2);

  /*
   * From csrc/moe
   */
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids) -> ()");
  m.impl("moe_align_block_size", torch::kMUSA, &moe_align_block_size);

  m.def("topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize) -> ()");
  m.impl("topk_softmax", torch::kMUSA, &topk_softmax);

  /*
   * From csrc/speculative
   */
  m.def(
      "tree_speculative_sampling_target_only(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor uniform_samples, Tensor uniform_samples_for_final_sampling, Tensor target_probs, Tensor draft_probs, "
      "float threshold_single, float threshold_acc, "
      "bool deterministic, int cuda_stream) -> ()");
  m.impl("tree_speculative_sampling_target_only", torch::kMUSA, &tree_speculative_sampling_target_only);

  m.def(
      "verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor target_predict, int cuda_stream) -> ()");
  m.impl("verify_tree_greedy", torch::kMUSA, &verify_tree_greedy);

  m.def(
      "build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
      "Tensor! tree_mask, Tensor! positions, Tensor! retrive_index, Tensor! retrive_next_token, "
      "Tensor! retrive_next_sibling, int topk, int depth, int draft_token_num, int tree_mask_mode) -> "
      "()");
  m.impl("build_tree_kernel_efficient", torch::kMUSA, &build_tree_kernel_efficient);

  /*
   * From XGrammar
   */
  m.def("apply_token_bitmask_inplace_cuda(Tensor logits, Tensor bitmask, Tensor? indices=None) -> ()");
  m.impl("apply_token_bitmask_inplace_cuda", &ApplyTokenBitmaskInplace);

  /*
   * From csrc/kvcacheio
   */
  m.def(
      "transfer_kv_per_layer(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int item_size, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer", torch::kMUSA, &transfer_kv_per_layer);
  m.def(
      "transfer_kv_per_layer_pf_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int layer_id, int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_pf_lf", torch::kMUSA, &transfer_kv_per_layer_pf_lf);
  m.def(
      "transfer_kv_all_layer(Tensor src_k_layers, Tensor dst_k_layers, Tensor src_v_layers, Tensor dst_v_layers, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer", torch::kMUSA, &transfer_kv_all_layer);
  m.def(
      "transfer_kv_all_layer_lf_pf(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_lf_pf", torch::kMUSA, &transfer_kv_all_layer_lf_pf);
  m.def(
      "transfer_kv_per_layer_mla(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int item_size, int "
      "block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla", torch::kMUSA, &transfer_kv_per_layer_mla);
  m.def(
      "transfer_kv_per_layer_mla_pf_lf(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int layer_id, "
      "int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla_pf_lf", torch::kMUSA, &transfer_kv_per_layer_mla_pf_lf);
  m.def(
      "transfer_kv_all_layer_mla(Tensor src_layers, Tensor dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "item_size, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla", torch::kMUSA, &transfer_kv_all_layer_mla);
  m.def(
      "transfer_kv_all_layer_mla_lf_pf(Tensor src_layers, Tensor dst, Tensor src_indices, Tensor dst_indices, "
      "int item_size, int dst_layout_dim, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla_lf_pf", torch::kMUSA, &transfer_kv_all_layer_mla_lf_pf);
  m.def(
      "transfer_kv_direct(Tensor[] src_layers, Tensor[] dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "page_size) -> ()");
  m.impl("transfer_kv_direct", torch::kMUSA, &transfer_kv_direct);

  /*
   * From csrc/memory
   */
  m.def("store_kv_cache(Tensor k_cache, Tensor v_cache, Tensor out_loc, Tensor k, Tensor v) -> ()");
  m.impl("store_kv_cache", &store_kv_cache);

  /*
   * From FlashInfer
   */
  m.def(
      "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, int "
      "cublas_handle, int cuda_stream) -> ()",
      {at::Tag::needs_fixed_stride_order});
  m.impl("bmm_fp8", torch::kMUSA, &bmm_fp8);

  m.def(
      "min_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? maybe_min_p_arr, float "
      "min_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("min_p_sampling_from_probs", torch::kMUSA, &min_p_sampling_from_probs);

  m.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  m.impl("top_k_renorm_probs", torch::kMUSA, &top_k_renorm_probs);

  m.def("top_p_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_p_arr, float top_p_val) -> ()");
  m.impl("top_p_renorm_probs", torch::kMUSA, &top_p_renorm_probs);

  m.def(
      "top_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? "
      "maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("top_p_sampling_from_probs", torch::kMUSA, &top_p_sampling_from_probs);

  m.def(
      "top_k_top_p_sampling_from_probs(Tensor probs, Tensor output, Tensor? maybe_indices, Tensor? maybe_top_k_arr, "
      "float top_k_val, Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, Generator? gen) -> ()");
  m.impl("top_k_top_p_sampling_from_probs", torch::kMUSA, &top_k_top_p_sampling_from_probs);

  m.def("top_k_mask_logits(Tensor logits, Tensor mask_logits, Tensor? maybe_top_k_arr, int top_k_val) -> ()");
  m.impl("top_k_mask_logits", torch::kMUSA, &top_k_mask_logits);
}

REGISTER_EXTENSION(common_ops)
