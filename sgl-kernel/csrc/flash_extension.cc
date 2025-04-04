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
      "fwd(Tensor!  q,"
      "    Tensor   k,"
      "    Tensor   v,"
      "    Tensor?  k_new,"
      "    Tensor?  v_new,"
      "    Tensor?  q_v,"
      "    Tensor!? out,"
      "    Tensor?  cu_seqlens_q,"
      "    Tensor?  cu_seqlens_k,"
      "    Tensor?  cu_seqlens_k_new,"
      "    Tensor?  seqused_q,"
      "    Tensor?  seqused_k,"
      "    int?     max_seqlen_q,"
      "    int?     max_seqlen_k,"
      "    Tensor?  page_table,"
      "    Tensor?  kv_batch_idx,"
      "    Tensor?  leftpad_k,"
      "    Tensor?  rotary_cos,"
      "    Tensor?  rotary_sin,"
      "    Tensor?  seqlens_rotary,"
      "    Tensor?  q_descale,"
      "    Tensor?  k_descale,"
      "    Tensor?  v_descale,"
      "    float    softmax_scale,"
      "    bool     is_causal,"
      "    int      window_size_left,"
      "    int      window_size_right,"
      "    float    softcap,"
      "    bool     is_rotary_interleaved,"
      "    Tensor?  scheduler_metadata,"
      "    int      num_splits,"
      "    bool?    pack_gqa,"
      "    int      sm_margin) -> Tensor[]");
  m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
}

REGISTER_EXTENSION(flash_ops)
