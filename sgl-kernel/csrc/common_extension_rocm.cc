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
   * From csrc/activation
   */
  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  m.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_quick", torch::kCUDA, &gelu_quick);

  /*
   * From csrc/allreduce
   */
  m.def(
      "init_custom_ar(Tensor meta, Tensor rank_data, "
      "str[] handles, int[] offsets, int rank, "
      "bool full_nvlink) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  m.def("all_reduce_reg(int fa, Tensor inp, Tensor! out) -> ()");
  m.impl("all_reduce_reg", torch::kCUDA, &all_reduce_reg);

  m.def(
      "all_reduce_unreg(int fa, Tensor inp, Tensor reg_buffer, Tensor! out) -> "
      "()");
  m.impl("all_reduce_unreg", torch::kCUDA, &all_reduce_unreg);

  m.def("dispose", &dispose);

  m.def("meta_size", &meta_size);

  m.def(
      "register_buffer(int fa, Tensor t, str[] handles, "
      "int[] offsets) -> ()");
  m.impl("register_buffer", torch::kCUDA, &register_buffer);

  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);

  m.def("allocate_meta_buffer", &allocate_meta_buffer);
  m.impl("allocate_meta_buffer", torch::kCUDA, &allocate_meta_buffer);

  m.def("get_meta_buffer_ipc_handle", &get_meta_buffer_ipc_handle);
  m.impl("get_meta_buffer_ipc_handle", torch::kCPU, &get_meta_buffer_ipc_handle);

  // quick allreduce
  m.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  m.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  m.def("init_custom_qr", &init_custom_qr);
  m.def("qr_destroy", &qr_destroy);

  m.def("qr_get_handle", &qr_get_handle);

  m.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  m.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  // Max input size in bytes
  m.def("qr_max_size", &qr_max_size);

  /*
   * From csrc/moe
   */
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, float "
      "moe_softcapping, Tensor? correction_bias) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  /*
   * From csrc/speculative
   */
  m.def(
      "verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor target_predict) -> ()");
  m.impl("verify_tree_greedy", torch::kCUDA, &verify_tree_greedy);

  m.def(
      "build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
      "Tensor! tree_mask, Tensor! positions, Tensor! retrive_index, Tensor! retrive_next_token, "
      "Tensor! retrive_next_sibling, int topk, int depth, int draft_token_num, int tree_mask_mode) -> "
      "()");
  m.impl("build_tree_kernel_efficient", torch::kCUDA, &build_tree_kernel_efficient);

  /*
   * From csrc/kvcacheio
   */
  m.def(
      "transfer_kv_per_layer(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int item_size, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer", torch::kCUDA, &transfer_kv_per_layer);
  m.def(
      "transfer_kv_per_layer_pf_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int layer_id, int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_pf_lf", torch::kCUDA, &transfer_kv_per_layer_pf_lf);
  m.def(
      "transfer_kv_all_layer(Tensor src_k_layers, Tensor dst_k_layers, Tensor src_v_layers, Tensor dst_v_layers, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer", torch::kCUDA, &transfer_kv_all_layer);
  m.def(
      "transfer_kv_all_layer_lf_pf(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_lf_pf", torch::kCUDA, &transfer_kv_all_layer_lf_pf);
  m.def(
      "transfer_kv_per_layer_mla(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int item_size, int "
      "block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla", torch::kCUDA, &transfer_kv_per_layer_mla);
  m.def(
      "transfer_kv_per_layer_mla_pf_lf(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int layer_id, "
      "int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla_pf_lf", torch::kCUDA, &transfer_kv_per_layer_mla_pf_lf);
  m.def(
      "transfer_kv_all_layer_mla(Tensor src_layers, Tensor dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "item_size, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla", torch::kCUDA, &transfer_kv_all_layer_mla);
  m.def(
      "transfer_kv_all_layer_mla_lf_pf(Tensor src_layers, Tensor dst, Tensor src_indices, Tensor dst_indices, "
      "int item_size, int dst_layout_dim, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla_lf_pf", torch::kCUDA, &transfer_kv_all_layer_mla_lf_pf);
  m.def(
      "transfer_kv_direct(Tensor[] src_layers, Tensor[] dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "page_size) -> ()");
  m.impl("transfer_kv_direct", torch::kCUDA, &transfer_kv_direct);
  m.def(
      "transfer_kv_per_layer_direct_pf_lf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, "
      "Tensor dst_indices, int layer_id, int page_size)->() ");
  m.impl("transfer_kv_per_layer_direct_pf_lf", torch::kCUDA, &transfer_kv_per_layer_direct_pf_lf);
  m.def(
      "transfer_kv_all_layer_direct_lf_pf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, "
      "Tensor dst_indices, int page_size) ->() ");
  m.impl("transfer_kv_all_layer_direct_lf_pf", torch::kCUDA, &transfer_kv_all_layer_direct_lf_pf);
  m.def(
      "transfer_kv_all_layer_lf_ph(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int page_size, int "
      "head_num, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_lf_ph", torch::kCUDA, &transfer_kv_all_layer_lf_ph);
  m.def(
      "transfer_kv_per_layer_ph_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int layer_id, int item_size, int src_layout_dim, int page_size, int head_num, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_ph_lf", torch::kCUDA, &transfer_kv_per_layer_ph_lf);

  /*
   * From csrc/grammar
   */
  m.def("apply_token_bitmask_inplace_cuda(Tensor logits, Tensor bitmask, Tensor? indices=None) -> ()");
  m.impl("apply_token_bitmask_inplace_cuda", &ApplyTokenBitmaskInplace);
}

REGISTER_EXTENSION(common_ops)
