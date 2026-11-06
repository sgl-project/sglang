/* Copyright 2026 SGLang Team. All Rights Reserved.

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

#pragma once

#include <cutlass/bfloat16.h>

#include <cstdint>
#include <cuda_runtime.h>

namespace sglang::q8kv8_sm100 {

/**
 * \brief Launch parameters of the SM100 Q8KV8 sparse MLA kernel.
 *
 * All 576 dims (512 NoPE + 64 RoPE) of Q and KV are e4m3 with per-tensor scales, as in TRT-LLM-gen's
 * FP8 sparse MLA: logits = bmm1_scale * (q8 . k8), out = bmm2_scale * softmax(logits) . v8.
 */
struct SparseMlaQ8Kv8Sm100Params {
  int32_t s_q, s_kv, h_q, topk;
  // The top-k of a query is cut into num_splits chunks of split_len (a multiple of 64) that run as
  // separate work items; with num_splits > 1, out / max_logits / lse are [num_splits, s_q, h_q, ...]
  // partials without the sink, merged by a second kernel.
  int32_t num_splits, split_len;
  float scale_log2;  // bmm1_scale * log2(e)
  float bmm2_scale;

  const uint8_t* __restrict__ q;            // [s_q, h_q, 576] e4m3, strides in bytes
  const uint8_t* __restrict__ kv;           // [s_kv, 576] e4m3, row stride in bytes
  const int32_t* __restrict__ indices;      // [s_q, topk]
  const float* __restrict__ attn_sink;      // [h_q] or nullptr
  const int32_t* __restrict__ topk_length;  // [s_q] or nullptr

  int64_t stride_q_s_q, stride_q_h_q;
  int64_t stride_kv_s_kv;
  int64_t stride_indices_s_q;

  cutlass::bfloat16_t* __restrict__ out;  // [(num_splits,) s_q, h_q, 512], contiguous
  float* __restrict__ max_logits;         // [(num_splits,) s_q, h_q]
  float* __restrict__ lse;                // [(num_splits,) s_q, h_q]

  // Final outputs of the split merge (num_splits > 1 only).
  cutlass::bfloat16_t* __restrict__ merged_out;
  float* __restrict__ merged_max_logits;
  float* __restrict__ merged_lse;

  cudaStream_t stream;
};

}  // namespace sglang::q8kv8_sm100
