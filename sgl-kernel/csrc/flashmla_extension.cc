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

#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From FlashMLA
   */
  m.def(
      "get_mla_decoding_metadata(Tensor seqlens_k, int num_q_tokens_per_head_k, int h_k, int? h_q, bool "
      "is_fp8_kvcache, int? topk) -> Tensor[]");
  m.impl("get_mla_decoding_metadata", torch::kCUDA, &get_mla_decoding_metadata);

  m.def(
      "fwd_kvcache_mla(Tensor q, Tensor kv_cache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, Tensor num_splits, bool is_fp8, Tensor? indices) "
      "-> Tensor[]");
  m.impl("fwd_kvcache_mla", torch::kCUDA, &fwd_kvcache_mla);

  m.def(
      "dense_prefill_fwd(Tensor workspace_buffer, Tensor q, Tensor k, Tensor v, Tensor cumulative_seqlen_q, Tensor "
      "cumulative_seqlen_kv, Tensor o, Tensor lse, int mask_mode_code, float softmax_scale, int max_seqlen_q, int "
      "max_seqlen_kv, bool is_varlen) -> ()");
  m.impl("dense_prefill_fwd", torch::kCUDA, &FMHACutlassSM100FwdRun);

  m.def("sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v) -> Tensor[]");
  m.impl("sparse_prefill_fwd", torch::kCUDA, &sparse_prefill_fwd);
}

REGISTER_EXTENSION(flashmla_ops)
