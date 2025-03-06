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

#include "sgl_kernels_ops.h"

TORCH_LIBRARY_EXPAND(sgl_kernels, m) {
  /*
   * From csrc/activation
   */
  m.def("silu_and_mul(Tensor! out, Tensor input, int cuda_stream) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input, int cuda_stream) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input, int cuda_stream) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

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

  /*
   * From csrc/moe
   */
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! token_cnts_buffer, Tensor! cumsum_buffer) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

}

REGISTER_EXTENSION(_kernels)
