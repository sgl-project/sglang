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
#include <Python.h>
#include <torch/csrc/stable/library.h>

// Boxed entry points defined in sgl-attn's hopper/flash_api_stable.cpp. They
// follow the stable-ABI boxed calling convention: read inputs from the stack,
// call the torch::stable::Tensor implementation, write outputs back.
void boxed_mha_fwd(StableIValue* stack, uint64_t num_args, uint64_t num_outputs);
void boxed_mha_fwd_get_scheduler_metadata(StableIValue* stack, uint64_t num_args, uint64_t num_outputs);

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
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
      "    Tensor(a!)?  out,"           // (b, s_q, h, dv) or (total_q, h, dv)
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
      "    Tensor?  sinks,"                          // (h)
      "    Tensor?  sparse_mask_fine = None"         // (total_q, max_k_blocks, num_int32_per_block)
      ") -> (Tensor(a!), Tensor, Tensor, Tensor)");  // first return aliases out

  /*
   * From flash-attention: get_scheduler_metadata
   * Precomputes tile scheduling for FA3 to avoid per-layer prepare_varlen_num_blocks calls.
   */
  m.def(
      "get_scheduler_metadata("
      "    int      batch_size,"
      "    int      max_seqlen_q,"
      "    int      max_seqlen_k,"
      "    int      num_heads,"
      "    int      num_heads_k,"
      "    int      headdim,"
      "    int      headdim_v,"
      "    ScalarType qkv_dtype,"
      "    Tensor   seqused_k,"         // b
      "    Tensor?  cu_seqlens_q,"      // b+1
      "    Tensor?  cu_seqlens_k,"      // b+1
      "    Tensor?  cu_seqlens_k_new,"  // b+1
      "    Tensor?  seqused_q,"         // b
      "    Tensor?  leftpad_k,"         // b
      "    int?     page_size,"
      "    int      max_seqlen_k_new = 0,"
      "    bool     is_causal = False,"
      "    int      window_size_left = -1,"
      "    int      window_size_right = -1,"
      "    int      attention_chunk = 0,"
      "    bool     has_softcap = False,"
      "    int      num_splits = 0,"
      "    bool?    pack_gqa = None,"
      "    int      sm_margin = 0"
      ") -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("fwd", &boxed_mha_fwd);
  m.impl("get_scheduler_metadata", &boxed_mha_fwd_get_scheduler_metadata);
}

REGISTER_EXTENSION(flash_ops)
