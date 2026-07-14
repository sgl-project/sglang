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

#include "api/dense_decode.h"
#include "api/sparse_decode.h"
#include "api/sparse_fwd.h"
#include "sgl_kernel_ops.h"

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>> sgl_sparse_decode_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int64_t d_v,
    double sm_scale) {
  return sparse_attn_decode_interface(
      q,
      kv,
      indices,
      topk_length,
      attn_sink,
      tile_scheduler_metadata,
      num_splits,
      extra_kv,
      extra_indices,
      extra_topk_length,
      static_cast<int>(d_v),
      static_cast<float>(sm_scale));
}

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>> sgl_dense_decode_fwd(
    at::Tensor q,
    const at::Tensor& kcache,
    int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits) {
  return dense_attn_decode_interface(
      q,
      kcache,
      static_cast<int>(head_size_v),
      seqlens_k,
      block_table,
      static_cast<float>(softmax_scale),
      is_causal,
      tile_scheduler_metadata,
      num_splits);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From FlashMLA
   */
  m.def(
      "get_mla_decoding_metadata(Tensor seqlens_k, int num_q_tokens_per_head_k, int h_k, int? h_q, bool "
      "is_fp8_kvcache, int? topk) -> Tensor[]");
  m.impl("get_mla_decoding_metadata", torch::kCUDA, &get_mla_decoding_metadata);

  m.def("get_mla_decoding_metadata_dense_fp8(Tensor seqlens_k, int num_heads_per_head_k, int num_heads_k) -> Tensor[]");
  m.impl("get_mla_decoding_metadata_dense_fp8", torch::kCUDA, &get_mla_decoding_metadata_dense_fp8);

  m.def(
      "fwd_kvcache_mla(Tensor q, Tensor kv_cache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, Tensor num_splits, bool is_fp8, Tensor? indices, "
      "Tensor? attn_sink, Tensor? extra_k_cache, Tensor? extra_indices_in_kvcache, Tensor? topk_length, Tensor? "
      "extra_topk_length) "
      "-> Tensor[]");
  m.impl("fwd_kvcache_mla", torch::kCUDA, &fwd_kvcache_mla);

#ifdef FLASHMLA_ENABLE_SM100
  m.def(
      "fwd_kvcache_mla_nvfp4(Tensor q, Tensor kcache, Tensor kv_global_scale, int head_size_v, Tensor seqlens_k, "
      "float softmax_scale, Tensor tile_scheduler_metadata, Tensor num_splits, Tensor indices) -> Tensor[]");
  m.impl("fwd_kvcache_mla_nvfp4", torch::kCUDA, &fwd_kvcache_mla_nvfp4);

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
  m.def(
      "_fwd_kvcache_mla_nvfp4_stage_timing(Tensor q, Tensor kcache, Tensor kv_global_scale, int head_size_v, Tensor "
      "seqlens_k, float softmax_scale, Tensor tile_scheduler_metadata, Tensor num_splits, Tensor indices) -> Tensor[]");
  m.impl(
      "_fwd_kvcache_mla_nvfp4_stage_timing", torch::kCUDA, &fwd_kvcache_mla_nvfp4_stage_timing);
#endif

  m.def(
      "dense_prefill_fwd(Tensor workspace_buffer, Tensor q, Tensor k, Tensor v, Tensor cumulative_seqlen_q, Tensor "
      "cumulative_seqlen_kv, Tensor o, Tensor lse, int mask_mode_code, float softmax_scale, int max_seqlen_q, int "
      "max_seqlen_kv, bool is_varlen) -> ()");
  m.impl("dense_prefill_fwd", torch::kCUDA, &FMHACutlassSM100FwdRun);
#endif

  m.def(
      "sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, "
      "Tensor? tile_scheduler_metadata, Tensor? num_splits, Tensor? extra_kv, Tensor? extra_indices, "
      "Tensor? extra_topk_length, int d_v, float sm_scale) -> (Tensor, Tensor, Tensor?, Tensor?)");
  m.impl("sparse_decode_fwd", torch::kCUDA, &sgl_sparse_decode_fwd);

  m.def(
      "dense_decode_fwd(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor? tile_scheduler_metadata, Tensor? num_splits) -> (Tensor, Tensor, "
      "Tensor?, "
      "Tensor?)");
  m.impl("dense_decode_fwd", torch::kCUDA, &sgl_dense_decode_fwd);

  m.def(
      "sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? attn_sink=None, "
      "Tensor? topk_length=None) -> Tensor[]");
  m.impl("sparse_prefill_fwd", torch::kCUDA, &sparse_prefill_fwd);

  m.def(
      "fwd_kvcache_mla_fp8(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, Tensor num_splits, Tensor? descale_q, Tensor? "
      "descale_k) -> Tensor[]");
  m.impl("fwd_kvcache_mla_fp8", torch::kCUDA, &fwd_kvcache_mla_fp8);
}

REGISTER_EXTENSION(flashmla_ops)
