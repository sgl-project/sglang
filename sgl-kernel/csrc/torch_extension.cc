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

TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  /*
   * From csrc/allreduce
   */
  m.def("init_custom_ar", init_custom_ar);
  m.def("dispose", dispose);
  m.def("all_reduce", all_reduce);
  m.def("get_graph_buffer_ipc_meta", get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", register_graph_buffers);

  /*
   * From csrc/attention
   */
  m.def("lightning_attention_decode", lightning_attention_decode);

  /*
   * From csrc/elementwise
   */
  m.def("rmsnorm", rmsnorm);
  m.def("fused_add_rmsnorm", sgl_fused_add_rmsnorm);
  m.def("gemma_rmsnorm", gemma_rmsnorm);
  m.def("gemma_fused_add_rmsnorm", gemma_fused_add_rmsnorm);
  m.def("silu_and_mul", silu_and_mul);
  m.def("gelu_tanh_and_mul", gelu_tanh_and_mul);
  m.def("gelu_and_mul", gelu_and_mul);
  m.def("apply_rope_pos_ids_cos_sin_cache", apply_rope_pos_ids_cos_sin_cache);

  /*
   * From csrc/gemm
   */
  m.def("awq_dequantize", awq_dequantize);
  m.def("int8_scaled_mm", int8_scaled_mm);
  m.def("fp8_scaled_mm", fp8_scaled_mm);
  m.def("fp8_blockwise_scaled_mm", fp8_blockwise_scaled_mm);
  m.def("sgl_per_token_group_quant_fp8", sgl_per_token_group_quant_fp8);
  m.def("sgl_per_token_group_quant_int8", sgl_per_token_group_quant_int8);
  m.def("sgl_per_tensor_quant_fp8", sgl_per_tensor_quant_fp8);
  m.def("sgl_per_token_quant_fp8", sgl_per_token_quant_fp8);
  m.def("cublas_grouped_gemm", cublas_grouped_gemm);
  m.def("cutlass_scaled_fp4_mm", cutlass_scaled_fp4_mm);
  m.def("scaled_fp4_quant", scaled_fp4_quant);

  /*
   * From csrc/moe
   */
  m.def("moe_align_block_size", moe_align_block_size);
  m.def("topk_softmax", topk_softmax);

  m.def(
      "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, int topk_group, int topk) -> "
      "(Tensor[])");
  m.impl("moe_fused_gate", torch::kCUDA, &moe_fused_gate);

  /*
   * From csrc/speculative
   */
  m.def("tree_speculative_sampling_target_only", tree_speculative_sampling_target_only);
  m.def("verify_tree_greedy", verify_tree_greedy);
  m.def("build_tree_kernel_efficient", build_tree_kernel_efficient);
  m.def("segment_packbits", segment_packbits);

  /*
   * From FlashInfer
   */
  m.def(
      "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, int "
      "cublas_handle, int cuda_stream) -> ()");
  m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
  m.def("min_p_sampling_from_probs", min_p_sampling_from_probs);
  m.def("top_k_renorm_probs", top_k_renorm_probs);
  m.def("top_p_renorm_probs", top_p_renorm_probs);
  m.def("top_k_top_p_sampling_from_probs", top_k_top_p_sampling_from_probs);
  m.def("top_p_sampling_from_probs", top_p_sampling_from_probs);

  /*
   * From flash-attention
   */
  m.def("fwd", make_pytorch_shim(mha_fwd));
}

REGISTER_EXTENSION(common_ops)
