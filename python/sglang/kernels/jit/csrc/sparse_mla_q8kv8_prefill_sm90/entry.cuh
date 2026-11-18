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

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "etap.cuh"
#include "kvsplit.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang {

static inline void
_set_device_and_stream(SparseMlaQ8Kv8PrefillParams& params, tvm::ffi::TensorView q, int64_t cuda_stream) {
  DLDevice dev = q.device();
  cudaSetDevice(dev.device_id);
  params.stream = reinterpret_cast<cudaStream_t>(cuda_stream);
}

template <int D_QK, int H, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
static bool try_sparse_mla_q8kv8_compact(const SparseMlaQ8Kv8PrefillParams& params) {
  static thread_local int cached_device = -1, cached_sms = 0;
  int device;
  KU_CUDA_CHECK(cudaGetDevice(&device));
  if (cached_device != device) {
    KU_CUDA_CHECK(cudaDeviceGetAttribute(&cached_sms, cudaDevAttrMultiProcessorCount, device));
    cached_device = device;
  }
  if (cached_sms < 128 || (reinterpret_cast<uintptr_t>(params.out) & 15)) return false;
  if constexpr (H == 32) {
    const bool short_kv = params.topk <= 1024 && params.s_q > cached_sms / 4;
    if (params.s_q <= cached_sms / 2 && !short_kv) return false;
  } else if (params.s_q <= 64) {
    return false;
  }
  sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<D_QK, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK, H>(params);
  return true;
}

template <int D_QK, int H, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
static inline void _run_q8kv8_kernel(SparseMlaQ8Kv8PrefillParams& params) {
  if constexpr (H < 64) {
    if (try_sparse_mla_q8kv8_compact<D_QK, H, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>(params)) return;
    if (try_sparse_mla_q8kv8_kvsplit<D_QK, H, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK, 16>(params)) return;
    if constexpr (H == 32) {
      static thread_local int cached_device = -1, cached_sms = 0;
      int device;
      KU_CUDA_CHECK(cudaGetDevice(&device));
      if (device != cached_device) {
        KU_CUDA_CHECK(cudaDeviceGetAttribute(&cached_sms, cudaDevAttrMultiProcessorCount, device));
        cached_device = device;
      }
      if (cached_sms >= 128) {
        if (params.s_q <= 32) {
          SparseMlaQ8Kv8HeadSplitKernel<D_QK, 8, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>::run(params);
          return;
        }
        if (params.s_q <= cached_sms / 2) {
          SparseMlaQ8Kv8HeadSplitKernel<D_QK, 16, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>::run(params);
          return;
        }
      }
    }
    using Kernel = std::conditional_t<
        H == 32,
        SparseMlaQ8Kv8RowQKKernel<D_QK, H, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>,
        SparseMlaQ8Kv8EtapKernel<D_QK, H, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>>;
    Kernel::run(params);
  } else {
    sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<D_QK, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>(params);
  }
}

template <int D_QK, int H>
static inline void
_run_q8kv8_for_shape(SparseMlaQ8Kv8PrefillParams& params, bool have_topk_length, bool have_attn_sink) {
  if (have_topk_length) {
    if (have_attn_sink) {
      _run_q8kv8_kernel<D_QK, H, true, true>(params);
    } else {
      _run_q8kv8_kernel<D_QK, H, true, false>(params);
    }
  } else {
    if (have_attn_sink) {
      _run_q8kv8_kernel<D_QK, H, false, true>(params);
    } else {
      _run_q8kv8_kernel<D_QK, H, false, false>(params);
    }
  }
}

template <int D_QK>
static inline void
_run_q8kv8_for_head_dim(SparseMlaQ8Kv8PrefillParams& params, bool have_topk_length, bool have_attn_sink) {
  switch (params.h_q) {
    case 8:
      return _run_q8kv8_for_shape<D_QK, 8>(params, have_topk_length, have_attn_sink);
    case 16:
      return _run_q8kv8_for_shape<D_QK, 16>(params, have_topk_length, have_attn_sink);
    case 32:
      return _run_q8kv8_for_shape<D_QK, 32>(params, have_topk_length, have_attn_sink);
    default:
      return _run_q8kv8_for_shape<D_QK, 64>(params, have_topk_length, have_attn_sink);
  }
}

static inline void _run_q8kv8(SparseMlaQ8Kv8PrefillParams& params, bool have_topk_length, bool have_attn_sink) {
  switch (params.d_qk) {
    case 512:
      _run_q8kv8_for_head_dim<512>(params, have_topk_length, have_attn_sink);
      return;
    case 576:
      _run_q8kv8_for_head_dim<576>(params, have_topk_length, have_attn_sink);
      return;
    default:
      fprintf(stderr, "sparse_prefill_q8kv8: unsupported d_qk=%d (must be 512 or 576)\n", params.d_qk);
      exit(1);
  }
}

static inline SparseMlaQ8Kv8PrefillParams _make_common_params(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView q_scale,
    tvm::ffi::TensorView kv_scale,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView max_logits,
    tvm::ffi::TensorView lse,
    int64_t s_q_val,
    int64_t s_kv_val,
    int64_t h_q_val,
    int64_t h_kv_val,
    int64_t d_qk_val,
    int64_t d_v_val,
    int64_t topk_val,
    double sm_scale_val,
    int64_t cuda_stream) {
  SparseMlaQ8Kv8PrefillParams params;
  params.s_q = (int)s_q_val;
  params.s_kv = (int)s_kv_val;
  params.h_q = (int)h_q_val;
  params.h_kv = (int)h_kv_val;
  params.d_qk = (int)d_qk_val;
  params.d_v = (int)d_v_val;
  params.topk = (int)topk_val;
  params.sm_scale_div_log2 = (float)sm_scale_val * (float)M_LOG2E;

  params.q = reinterpret_cast<const uint8_t*>(q.data_ptr());
  params.kv = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  params.indices = static_cast<int*>(indices.data_ptr());
  params.attn_sink = nullptr;
  params.topk_length = nullptr;

  params.q_scale_ptr = static_cast<const float*>(q_scale.data_ptr());
  params.kv_scale_ptr = static_cast<const float*>(kv_scale.data_ptr());

  params.stride_q_s_q = (int)h_q_val * (int)d_qk_val;
  params.stride_q_h_q = (int)d_qk_val;
  params.stride_kv_s_kv = (int64_t)h_kv_val * (int64_t)d_qk_val;
  params.stride_kv_h_kv = (int)d_qk_val;
  params.stride_indices_s_q = (int)h_kv_val * (int)topk_val;
  params.stride_indices_h_kv = (int)topk_val;

  params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
  params.max_logits = static_cast<float*>(max_logits.data_ptr());
  params.lse = static_cast<float*>(lse.data_ptr());

  _set_device_and_stream(params, q, cuda_stream);
  return params;
}

void sparse_prefill_q8kv8_dispatch(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView q_scale,
    tvm::ffi::TensorView kv_scale,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView max_logits,
    tvm::ffi::TensorView lse,
    int64_t s_q_val,
    int64_t s_kv_val,
    int64_t h_q_val,
    int64_t h_kv_val,
    int64_t d_qk_val,
    int64_t d_v_val,
    int64_t topk_val,
    double sm_scale_val,
    int64_t cuda_stream) {
  SparseMlaQ8Kv8PrefillParams params = _make_common_params(
      q,
      kv,
      indices,
      q_scale,
      kv_scale,
      out,
      max_logits,
      lse,
      s_q_val,
      s_kv_val,
      h_q_val,
      h_kv_val,
      d_qk_val,
      d_v_val,
      topk_val,
      sm_scale_val,
      cuda_stream);
  _run_q8kv8(params, false, false);
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
    int64_t s_q_val,
    int64_t s_kv_val,
    int64_t h_q_val,
    int64_t h_kv_val,
    int64_t d_qk_val,
    int64_t d_v_val,
    int64_t topk_val,
    double sm_scale_val,
    int64_t cuda_stream) {
  SparseMlaQ8Kv8PrefillParams params = _make_common_params(
      q,
      kv,
      indices,
      q_scale,
      kv_scale,
      out,
      max_logits,
      lse,
      s_q_val,
      s_kv_val,
      h_q_val,
      h_kv_val,
      d_qk_val,
      d_v_val,
      topk_val,
      sm_scale_val,
      cuda_stream);
  params.attn_sink = static_cast<float*>(attn_sink.data_ptr());
  params.topk_length = static_cast<int*>(topk_length.data_ptr());
  _run_q8kv8(params, true, true);
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
    int64_t s_q_val,
    int64_t s_kv_val,
    int64_t h_q_val,
    int64_t h_kv_val,
    int64_t d_qk_val,
    int64_t d_v_val,
    int64_t topk_val,
    double sm_scale_val,
    int64_t cuda_stream) {
  SparseMlaQ8Kv8PrefillParams params = _make_common_params(
      q,
      kv,
      indices,
      q_scale,
      kv_scale,
      out,
      max_logits,
      lse,
      s_q_val,
      s_kv_val,
      h_q_val,
      h_kv_val,
      d_qk_val,
      d_v_val,
      topk_val,
      sm_scale_val,
      cuda_stream);
  params.topk_length = static_cast<int*>(topk_length.data_ptr());
  _run_q8kv8(params, true, false);
}

}  // namespace sglang
