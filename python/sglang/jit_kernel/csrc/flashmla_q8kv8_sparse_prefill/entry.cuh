// entry.cuh -- JIT dispatch entry for Q8KV8 sparse prefill attention kernel (SM90).
#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "phase1_new.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

static int _q8kv8_num_sm_cache[16] = {0};

static inline int _get_num_sm_cached(int device_id) {
  if (device_id < 0 || device_id >= 16) {
    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device_id);
    return num_sm;
  }
  int cached = _q8kv8_num_sm_cache[device_id];
  if (cached == 0) {
    cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, device_id);
    _q8kv8_num_sm_cache[device_id] = cached;
  }
  return cached;
}

static inline void
_set_device_stream_and_sm(SparseAttnFwdQ8SM90NewParams& params, tvm::ffi::TensorView q, int64_t cuda_stream) {
  DLDevice dev = q.device();
  cudaSetDevice(dev.device_id);
  params.stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  params.num_sm = _get_num_sm_cached(dev.device_id);
}

static inline void _run_q8kv8(SparseAttnFwdQ8SM90NewParams& params, bool have_topk_length, bool have_attn_sink) {
  if (params.d_qk == 512) {
    if (have_topk_length) {
      if (have_attn_sink) {
        sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<512, true, true>(params);
      } else {
        sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<512, true, false>(params);
      }
    } else {
      if (have_attn_sink) {
        sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<512, false, true>(params);
      } else {
        sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<512, false, false>(params);
      }
    }
  } else if (params.d_qk == 576) {
    if (have_topk_length) {
      if (have_attn_sink) {
        sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<576, true, true>(params);
      } else {
        sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<576, true, false>(params);
      }
    } else {
      if (have_attn_sink) {
        sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<576, false, true>(params);
      } else {
        sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<576, false, false>(params);
      }
    }
  } else {
    fprintf(stderr, "sparse_prefill_q8kv8: unsupported d_qk=%d (must be 512 or 576)\n", params.d_qk);
    exit(1);
  }
}

static inline SparseAttnFwdQ8SM90NewParams _make_common_params(
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
  SparseAttnFwdQ8SM90NewParams params;
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

  _set_device_stream_and_sm(params, q, cuda_stream);
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
  SparseAttnFwdQ8SM90NewParams params = _make_common_params(
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
  SparseAttnFwdQ8SM90NewParams params = _make_common_params(
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

}  // namespace
