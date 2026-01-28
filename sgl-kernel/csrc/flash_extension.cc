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
      "fwd(Tensor   q,"                 // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
      "    Tensor   k,"                 // (b_k, s_k, h_k, d) or (total_k, h_k, d) or paged
      "    Tensor   v,"                 // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) or paged
      "    Tensor?  k_new,"             // (b, s_k_new, h_k, d) or (total_k_new, h_k, d)
      "    Tensor?  v_new,"             // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv)
      "    Tensor?  q_v,"               // (b, s_q, h, dv) or (total_q_new, h, dv)
      "    Tensor?  out,"               // (b, s_q, h, dv) or (total_q, h, dv)
      "    Tensor?  cu_seqlens_q,"      // b+1
      "    Tensor?  cu_seqlens_k,"      // b+1
      "    Tensor?  cu_seqlens_k_new,"  // b+1
      "    Tensor?  seqused_q,"         // b
      "    Tensor?  seqused_k,"         // b
      "    int?     max_seqlen_q,"
      "    int?     max_seqlen_k,"    // TODO: check if needed
      "    Tensor?  page_table,"      // (b_k, max_num_pages_per_seq)
      "    Tensor?  kv_batch_idx,"    // b
      "    Tensor?  leftpad_k,"       // b
      "    Tensor?  rotary_cos,"      // seqlen_ro x (rotary_dim / 2)
      "    Tensor?  rotary_sin,"      // seqlen_ro x (rotary_dim / 2)
      "    Tensor?  seqlens_rotary,"  // b
      "    Tensor?  q_descale,"       // (b, h_k)
      "    Tensor?  k_descale,"       // (b, h_k)
      "    Tensor?  v_descale,"       // (b, h_k)
      "    float?   softmax_scale,"   // now optional
      "    bool     is_causal,"
      "    int      window_size_left,"
      "    int      window_size_right,"
      "    int      attention_chunk,"  // NEW
      "    float    softcap,"          // promoted to double in C++; schema float is fine
      "    bool     is_rotary_interleaved,"
      "    Tensor?  scheduler_metadata,"  // (b + 1)
      "    int      num_splits,"
      "    bool?    pack_gqa,"
      "    int      sm_margin,"
      "    Tensor?  sinks"
      ") -> (Tensor, Tensor, Tensor, Tensor)");  // NEW return type: tuple of 4 tensors

  m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
}

REGISTER_EXTENSION(flash_ops)
