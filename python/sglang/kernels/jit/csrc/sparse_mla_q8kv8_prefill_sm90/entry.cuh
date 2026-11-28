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

// JIT dispatch entry for the SM90 Q8KV8 sparse MLA prefill kernel.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "kernel.cuh"
#include <cmath>
#include <cstdint>
#include <limits>

namespace sglang {

namespace sparse_mla_q8kv8 {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

inline constexpr int64_t kHeadBlock = 64;   // B_H in kernel.cuh
inline constexpr int64_t kTopkBlock = 128;  // 2 * B_TOPK in kernel.cuh
inline constexpr int64_t kDimV = 512;

template <int D_QK>
void run_for_head_dim(
    const SparseMlaQ8Kv8PrefillParams& params, bool have_topk_length, bool have_attn_sink, cudaStream_t stream) {
  if (have_topk_length) {
    if (have_attn_sink) {
      sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<D_QK, true, true>(params, stream);
    } else {
      sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<D_QK, true, false>(params, stream);
    }
  } else {
    if (have_attn_sink) {
      sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<D_QK, false, true>(params, stream);
    } else {
      sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<D_QK, false, false>(params, stream);
    }
  }
}

inline void
run(TensorView q,
    TensorView kv,
    TensorView indices,
    TensorView q_scale,
    TensorView kv_scale,
    Optional<TensorView> attn_sink,
    Optional<TensorView> topk_length,
    TensorView out,
    TensorView max_logits,
    TensorView lse,
    double sm_scale) {
  using namespace host;

  auto device = SymbolicDevice{};
  auto s_q = SymbolicSize{"s_q"};
  auto h_q = SymbolicSize{"h_q"};
  auto d_qk = SymbolicSize{"d_qk"};
  device.set_options<kDLCUDA>();
  TensorMatcher({s_q, h_q, d_qk})  //
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(q);
  TensorMatcher({-1, 1, d_qk})  //
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(kv);
  TensorMatcher({s_q, 1, -1})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(indices);
  TensorMatcher({s_q, h_q, kDimV})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(out);
  TensorMatcher({s_q, h_q})  //
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(max_logits)
      .verify(lse);
  // Per-tensor scales may be 0-d or [1].
  for (const auto& scale : {q_scale, kv_scale}) {
    CHECK_HOST(scale.numel() == 1 && is_type<fp32_t>(scale.dtype())) << "q_scale / kv_scale must be one fp32 element";
    device.verify(scale.device());
  }
  if (topk_length.has_value()) {
    TensorMatcher({s_q})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(topk_length.value());
  }
  if (attn_sink.has_value()) {
    TensorMatcher({h_q})  //
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(attn_sink.value());
  }

  const int64_t num_q = s_q.unwrap();
  const int64_t num_heads = h_q.unwrap();
  const int64_t head_dim = d_qk.unwrap();
  const int64_t topk = indices.size(2);
  const int64_t num_kv = kv.size(0);
  CHECK_HOST(num_heads > 0 && num_heads % kHeadBlock == 0)
      << "h_q must be a positive multiple of 64, got " << num_heads;
  CHECK_HOST(head_dim == 512 || head_dim == 576) << "d_qk must be 512 or 576, got " << head_dim;
  CHECK_HOST(topk > 0 && topk % kTopkBlock == 0) << "topk must be a positive multiple of 128, got " << topk;
  // The kernel indexes indices and max_logits / lse with 32-bit row offsets (s_q_idx * topk, s_q_idx * h_q).
  constexpr int64_t kMaxInt32 = std::numeric_limits<int32_t>::max();
  CHECK_HOST(num_kv <= kMaxInt32 && num_q * topk <= kMaxInt32 && num_q * num_heads <= kMaxInt32)
      << "sparse_mla_q8kv8 prefill: shape too large for 32-bit offsets";
  if (num_q == 0) return;

  SparseMlaQ8Kv8PrefillParams params;
  params.s_q = static_cast<int>(num_q);
  params.s_kv = static_cast<int>(num_kv);
  params.h_q = static_cast<int>(num_heads);
  params.h_kv = 1;
  params.d_qk = static_cast<int>(head_dim);
  params.d_v = static_cast<int>(kDimV);
  params.topk = static_cast<int>(topk);
  params.sm_scale_div_log2 = static_cast<float>(sm_scale) * static_cast<float>(M_LOG2E);

  params.q = static_cast<const uint8_t*>(q.data_ptr());
  params.kv = static_cast<const uint8_t*>(kv.data_ptr());
  params.indices = static_cast<int*>(indices.data_ptr());
  params.attn_sink = attn_sink.has_value() ? static_cast<float*>(attn_sink.value().data_ptr()) : nullptr;
  params.topk_length = topk_length.has_value() ? static_cast<int*>(topk_length.value().data_ptr()) : nullptr;
  params.q_scale_ptr = static_cast<const float*>(q_scale.data_ptr());
  params.kv_scale_ptr = static_cast<const float*>(kv_scale.data_ptr());

  params.stride_q_s_q = static_cast<int>(num_heads * head_dim);
  params.stride_q_h_q = static_cast<int>(head_dim);
  params.stride_kv_s_kv = head_dim;
  params.stride_kv_h_kv = static_cast<int>(head_dim);
  params.stride_indices_s_q = static_cast<int>(topk);
  params.stride_indices_h_kv = static_cast<int>(topk);

  params.out = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
  params.max_logits = static_cast<float*>(max_logits.data_ptr());
  params.lse = static_cast<float*>(lse.data_ptr());

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  const bool have_topk_length = topk_length.has_value();
  const bool have_attn_sink = attn_sink.has_value();
  if (head_dim == 512) {
    run_for_head_dim<512>(params, have_topk_length, have_attn_sink, stream);
  } else {
    run_for_head_dim<576>(params, have_topk_length, have_attn_sink, stream);
  }
}

}  // namespace sparse_mla_q8kv8

void sparse_prefill_q8kv8_dispatch(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView q_scale,
    tvm::ffi::TensorView kv_scale,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView max_logits,
    tvm::ffi::TensorView lse,
    double sm_scale) {
  sparse_mla_q8kv8::run(q, kv, indices, q_scale, kv_scale, {}, {}, out, max_logits, lse, sm_scale);
}

void sparse_prefill_q8kv8_dispatch_full(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView q_scale,
    tvm::ffi::TensorView kv_scale,
    tvm::ffi::TensorView attn_sink,
    tvm::ffi::TensorView topk_length,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView max_logits,
    tvm::ffi::TensorView lse,
    double sm_scale) {
  sparse_mla_q8kv8::run(q, kv, indices, q_scale, kv_scale, attn_sink, topk_length, out, max_logits, lse, sm_scale);
}

void sparse_prefill_q8kv8_dispatch_topk_length(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView q_scale,
    tvm::ffi::TensorView kv_scale,
    tvm::ffi::TensorView topk_length,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView max_logits,
    tvm::ffi::TensorView lse,
    double sm_scale) {
  sparse_mla_q8kv8::run(q, kv, indices, q_scale, kv_scale, {}, topk_length, out, max_logits, lse, sm_scale);
}

}  // namespace sglang
