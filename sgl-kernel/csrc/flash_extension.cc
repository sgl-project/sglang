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

#include "sgl_flash_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From flash-attention
   */
  m.def(
      "fwd("
      "Tensor q,"
      "Tensor k,"
      "Tensor v,"
      "Tensor(k_new!)? k_new = None,"
      "Tensor(v_new!)? v_new = None,"
      "Tensor? q_v = None,"
      "Tensor(out!)? out = None,"
      "Tensor? cu_seqlens_q = None,"
      "Tensor? cu_seqlens_k = None,"
      "Tensor? cu_seqlens_k_new = None,"
      "Tensor? seqused_q = None,"
      "Tensor? seqused_k = None,"
      "int? max_seqlen_q = None,"
      "int? max_seqlen_k = None,"
      "Tensor? page_table = None,"
      "Tensor? kv_batch_idx = None,"
      "Tensor? leftpad_k = None,"
      "Tensor? rotary_cos = None,"
      "Tensor? rotary_sin = None,"
      "Tensor? seqlens_rotary = None,"
      "Tensor? q_descale = None,"
      "Tensor? k_descale = None,"
      "Tensor? v_descale = None,"
      "float? softmax_scale = None,"
      "bool is_causal = False,"
      "int window_size_left = -1,"
      "int window_size_right = -1,"
      "int attention_chunk = 0,"
      "float softcap = 0.0,"
      "bool is_rotary_interleaved = False,"
      "Tensor? scheduler_metadata = None,"
      "int num_splits = 0,"
      "bool? pack_gqa = None,"
      "int sm_margin = 0) -> (Tensor(out!), Tensor, Tensor, Tensor)");
  m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
}

REGISTER_EXTENSION(flash_ops)
