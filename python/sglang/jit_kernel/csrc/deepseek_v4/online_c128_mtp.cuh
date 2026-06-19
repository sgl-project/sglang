#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

SGL_DEVICE int64_t clamp_accept_len(int64_t delta, int64_t max_accept) {
  if (delta < 0) return 0;
  return delta < max_accept ? delta : max_accept;
}

template <typename TSeq, typename TReq>
struct OnlineC128MTPWritePrefixParams {
  const float* __restrict__ kv_score_input;
  const TSeq* __restrict__ seq_lens;
  const TReq* __restrict__ req_pool_indices;
  const int32_t* __restrict__ req_to_token;
  const float* __restrict__ ape;
  float* __restrict__ state;
  int64_t kv_score_stride_b;
  int64_t req_to_token_stride_b;
  int64_t ape_stride_r;
  int64_t state_stride_b;
  int64_t layer_bs;
  int64_t num_verify_tokens;
  int64_t state_slot_stride;
};

template <typename TSeq, typename TReq>
struct OnlineC128MTPMarkPendingParams {
  const TSeq* __restrict__ seq_lens;
  const TReq* __restrict__ req_pool_indices;
  int64_t* __restrict__ pending_seq_lens;
  int64_t bs;
  int64_t max_num_reqs;
};

template <typename TSeq, typename TReq>
struct OnlineC128MTPCommitPendingParams {
  const TSeq* __restrict__ cur_seq_lens;
  const TReq* __restrict__ cur_req_pool_indices;
  const int32_t* __restrict__ req_to_token;
  const int64_t* __restrict__ pending_seq_lens;
  float* __restrict__ state;
  int64_t cur_bs;
  int64_t req_to_token_stride_b;
  int64_t state_stride_b;
  int64_t num_verify_tokens;
  int64_t state_slot_stride;
  int64_t max_num_reqs;
};

__global__ void online_c128_mtp_clear_all_pending_kernel(int64_t* pending_seq_lens, int64_t max_num_reqs) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < max_num_reqs) pending_seq_lens[idx] = -1;
}

template <typename TSeq, typename TReq>
__global__ void online_c128_mtp_mark_pending_kernel(const OnlineC128MTPMarkPendingParams<TSeq, TReq> params) {
  const int64_t bid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (bid >= params.bs) return;
  const int64_t req = static_cast<int64_t>(params.req_pool_indices[bid]);
  if (req >= 0 && req < params.max_num_reqs) {
    params.pending_seq_lens[req] = static_cast<int64_t>(params.seq_lens[bid]);
  }
}

template <int64_t kHeadDim, typename TSeq, typename TReq>
__global__ void online_c128_mtp_commit_pending_kernel(const OnlineC128MTPCommitPendingParams<TSeq, TReq> params) {
  const int64_t bid = static_cast<int64_t>(blockIdx.x);
  if (bid >= params.cur_bs) return;

  const int64_t req = static_cast<int64_t>(params.cur_req_pool_indices[bid]);
  if (req < 0 || req >= params.max_num_reqs) return;
  const int64_t old_seq = params.pending_seq_lens[req];
  if (old_seq < 0) return;

  const int64_t cur_seq = static_cast<int64_t>(params.cur_seq_lens[bid]);
  const int64_t accept = clamp_accept_len(cur_seq - old_seq, params.num_verify_tokens);
  if (accept <= 0) return;

  const int64_t final_seq = old_seq + accept;
  if ((final_seq & 127) == 0) return;

  const int64_t slot = req;
  const float* const src = params.state + (slot + accept * params.state_slot_stride) * params.state_stride_b;
  float* const dst = params.state + slot * params.state_stride_b;

  for (int64_t d = static_cast<int64_t>(threadIdx.x); d < kHeadDim * 3; d += blockDim.x) {
    dst[d] = src[d];
  }
}

template <int64_t kHeadDim, typename TSeq, typename TReq>
__global__ void online_c128_mtp_write_prefix_kernel(const OnlineC128MTPWritePrefixParams<TSeq, TReq> params) {
  const int64_t bid = static_cast<int64_t>(blockIdx.x);
  if (bid >= params.layer_bs) return;

  const int64_t seq_before = static_cast<int64_t>(params.seq_lens[bid]);
  const int64_t req_idx = static_cast<int64_t>(params.req_pool_indices[bid]);
  const int64_t start_pos = seq_before & 127;
  const bool has_partial = seq_before > 0 && start_pos != 0;

  int64_t init_slot = 0;
  if (has_partial) {
    init_slot = req_idx;
  }

  const int64_t d = static_cast<int64_t>(threadIdx.x);
  float run_max = 0.0f;
  float run_sum = 0.0f;
  float run_kv = 0.0f;
  if (has_partial) {
    const float* const init = params.state + init_slot * params.state_stride_b;
    run_max = init[d];
    run_sum = init[kHeadDim + d];
    run_kv = init[kHeadDim * 2 + d];
  }

  constexpr int kMaxVerifyTokens = 8;
  float kv_steps[kMaxVerifyTokens];
  float score_steps[kMaxVerifyTokens];

#pragma unroll
  for (int step = 0; step < kMaxVerifyTokens; ++step) {
    if (step >= params.num_verify_tokens) break;

    const int64_t pos = (start_pos + step) & 127;
    const float* const kv = params.kv_score_input + (bid * params.num_verify_tokens + step) * params.kv_score_stride_b;
    kv_steps[step] = kv[d];
    score_steps[step] = kv[kHeadDim + d] + params.ape[pos * params.ape_stride_r + d];
  }

#pragma unroll
  for (int step = 0; step < kMaxVerifyTokens; ++step) {
    if (step >= params.num_verify_tokens) break;

    const int64_t pos = (start_pos + step) & 127;
    const float kv_step = kv_steps[step];
    const float score_step = score_steps[step];
    if (pos == 0) {
      run_kv = kv_step;
      run_max = score_step;
      run_sum = 1.0f;
    } else {
      const float new_max = fmaxf(run_max, score_step);
      const float old_sum_scaled = run_sum * __expf(run_max - new_max);
      const float new_exp = __expf(score_step - new_max);
      const float new_sum = old_sum_scaled + new_exp;
      run_kv = (run_kv * old_sum_scaled + kv_step * new_exp) / new_sum;
      run_max = new_max;
      run_sum = new_sum;
    }

    const int64_t final_seq = seq_before + step + 1;
    if ((final_seq & 127) != 0) {
      const int64_t slot = req_idx + (step + 1) * params.state_slot_stride;
      float* const out = params.state + slot * params.state_stride_b;
      out[d] = run_max;
      out[kHeadDim + d] = run_sum;
      out[kHeadDim * 2 + d] = run_kv;
    }

    if (pos == 127) {
      run_kv = 0.0f;
      run_max = 0.0f;
      run_sum = 0.0f;
    }
  }
}

template <int64_t kHeadDim>
struct OnlineC128MTPWritePrefixKernel {
  template <typename TSeq, typename TReq>
  static void launch(
      tvm::ffi::TensorView kv_score_input,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView ape,
      tvm::ffi::TensorView state,
      int64_t layer_bs,
      int64_t num_verify_tokens,
      int64_t state_slot_stride,
      DLDevice device) {
    using namespace host;

    const auto params = OnlineC128MTPWritePrefixParams<TSeq, TReq>{
        .kv_score_input = static_cast<const float*>(kv_score_input.data_ptr()),
        .seq_lens = static_cast<const TSeq*>(seq_lens.data_ptr()),
        .req_pool_indices = static_cast<const TReq*>(req_pool_indices.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .ape = static_cast<const float*>(ape.data_ptr()),
        .state = static_cast<float*>(state.data_ptr()),
        .kv_score_stride_b = kv_score_input.stride(0),
        .req_to_token_stride_b = req_to_token.stride(0),
        .ape_stride_r = ape.stride(0),
        .state_stride_b = state.stride(0),
        .layer_bs = layer_bs,
        .num_verify_tokens = num_verify_tokens,
        .state_slot_stride = state_slot_stride,
    };

    static_assert(kHeadDim == 512, "online c128 MTP write-prefix only supports head_dim=512");
    constexpr uint32_t kThreads = static_cast<uint32_t>(kHeadDim);
    LaunchKernel(static_cast<uint32_t>(layer_bs), kThreads, device)(
        online_c128_mtp_write_prefix_kernel<kHeadDim, TSeq, TReq>, params);
  }

  static void
  run(tvm::ffi::TensorView kv_score_input,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView ape,
      tvm::ffi::TensorView state,
      int64_t layer_bs,
      int64_t num_verify_tokens,
      int64_t state_slot_stride) {
    using namespace host;

    auto seq_dtype = SymbolicDType{};
    auto req_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({-1, kHeadDim * 2}).with_dtype<float>().with_device(device).verify(kv_score_input);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(seq_dtype).with_device(device).verify(seq_lens);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(req_dtype).with_device(device).verify(req_pool_indices);
    TensorMatcher({-1, -1}).with_dtype<int32_t>().with_device(device).verify(req_to_token);
    TensorMatcher({128, kHeadDim}).with_dtype<float>().with_device(device).verify(ape);
    TensorMatcher({-1, kHeadDim * 3}).with_dtype<float>().with_device(device).verify(state);

    if (layer_bs <= 0) return;
    RuntimeCheck(num_verify_tokens > 0 && num_verify_tokens <= 8, "unsupported num_verify_tokens=", num_verify_tokens);
    RuntimeCheck(state_slot_stride > 0, "state_slot_stride must be positive");
    RuntimeCheck(layer_bs <= seq_lens.shape()[0], "layer_bs exceeds seq_lens rows");
    RuntimeCheck(layer_bs <= req_pool_indices.shape()[0], "layer_bs exceeds req_pool_indices rows");
    RuntimeCheck(layer_bs * num_verify_tokens <= kv_score_input.shape()[0], "kv_score_input is too small");

    if (seq_dtype.is_type<int32_t>()) {
      if (req_dtype.is_type<int32_t>()) {
        launch<int32_t, int32_t>(
            kv_score_input, seq_lens, req_pool_indices, req_to_token, ape, state,
            layer_bs, num_verify_tokens, state_slot_stride, device.unwrap());
      } else {
        launch<int32_t, int64_t>(
            kv_score_input, seq_lens, req_pool_indices, req_to_token, ape, state,
            layer_bs, num_verify_tokens, state_slot_stride, device.unwrap());
      }
    } else {
      if (req_dtype.is_type<int32_t>()) {
        launch<int64_t, int32_t>(
            kv_score_input, seq_lens, req_pool_indices, req_to_token, ape, state,
            layer_bs, num_verify_tokens, state_slot_stride, device.unwrap());
      } else {
        launch<int64_t, int64_t>(
            kv_score_input, seq_lens, req_pool_indices, req_to_token, ape, state,
            layer_bs, num_verify_tokens, state_slot_stride, device.unwrap());
      }
    }
  }
};

template <int64_t kHeadDim>
struct OnlineC128MTPMarkPendingKernel {
  template <typename TSeq, typename TReq>
  static void launch(
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView pending_seq_lens,
      int64_t bs,
      int64_t max_num_reqs,
      DLDevice device) {
    using namespace host;

    const auto params = OnlineC128MTPMarkPendingParams<TSeq, TReq>{
        .seq_lens = static_cast<const TSeq*>(seq_lens.data_ptr()),
        .req_pool_indices = static_cast<const TReq*>(req_pool_indices.data_ptr()),
        .pending_seq_lens = static_cast<int64_t*>(pending_seq_lens.data_ptr()),
        .bs = bs,
        .max_num_reqs = max_num_reqs,
    };

    constexpr uint32_t kThreads = 256;
    const uint32_t clear_blocks = host::div_ceil(static_cast<uint32_t>(max_num_reqs), kThreads);
    LaunchKernel(clear_blocks, kThreads, device)(
        online_c128_mtp_clear_all_pending_kernel, params.pending_seq_lens, max_num_reqs);
    const uint32_t mark_blocks = host::div_ceil(static_cast<uint32_t>(bs), kThreads);
    LaunchKernel(mark_blocks, kThreads, device)(online_c128_mtp_mark_pending_kernel<TSeq, TReq>, params);
  }

  static void
  run(tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView pending_seq_lens,
      int64_t bs,
      int64_t max_num_reqs) {
    using namespace host;

    auto seq_dtype = SymbolicDType{};
    auto req_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(seq_dtype).with_device(device).verify(seq_lens);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(req_dtype).with_device(device).verify(req_pool_indices);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(pending_seq_lens);

    if (bs <= 0) return;
    RuntimeCheck(bs <= seq_lens.shape()[0], "bs exceeds seq_lens rows");
    RuntimeCheck(bs <= req_pool_indices.shape()[0], "bs exceeds req_pool_indices rows");
    RuntimeCheck(max_num_reqs <= pending_seq_lens.shape()[0], "max_num_reqs exceeds pending rows");

    if (seq_dtype.is_type<int32_t>()) {
      if (req_dtype.is_type<int32_t>()) {
        launch<int32_t, int32_t>(seq_lens, req_pool_indices, pending_seq_lens, bs, max_num_reqs, device.unwrap());
      } else {
        launch<int32_t, int64_t>(seq_lens, req_pool_indices, pending_seq_lens, bs, max_num_reqs, device.unwrap());
      }
    } else {
      if (req_dtype.is_type<int32_t>()) {
        launch<int64_t, int32_t>(seq_lens, req_pool_indices, pending_seq_lens, bs, max_num_reqs, device.unwrap());
      } else {
        launch<int64_t, int64_t>(seq_lens, req_pool_indices, pending_seq_lens, bs, max_num_reqs, device.unwrap());
      }
    }
  }
};

template <int64_t kHeadDim>
struct OnlineC128MTPCommitPendingKernel {
  template <typename TSeq, typename TReq>
  static void launch(
      tvm::ffi::TensorView cur_seq_lens,
      tvm::ffi::TensorView cur_req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView pending_seq_lens,
      tvm::ffi::TensorView state,
      int64_t cur_bs,
      int64_t num_verify_tokens,
      int64_t state_slot_stride,
      int64_t max_num_reqs,
      DLDevice device) {
    using namespace host;

    const auto params = OnlineC128MTPCommitPendingParams<TSeq, TReq>{
        .cur_seq_lens = static_cast<const TSeq*>(cur_seq_lens.data_ptr()),
        .cur_req_pool_indices = static_cast<const TReq*>(cur_req_pool_indices.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .pending_seq_lens = static_cast<const int64_t*>(pending_seq_lens.data_ptr()),
        .state = static_cast<float*>(state.data_ptr()),
        .cur_bs = cur_bs,
        .req_to_token_stride_b = req_to_token.stride(0),
        .state_stride_b = state.stride(0),
        .num_verify_tokens = num_verify_tokens,
        .state_slot_stride = state_slot_stride,
        .max_num_reqs = max_num_reqs,
    };

    constexpr uint32_t kThreads = 256;
    LaunchKernel(static_cast<uint32_t>(cur_bs), kThreads, device)(
        online_c128_mtp_commit_pending_kernel<kHeadDim, TSeq, TReq>, params);
  }

  static void
  run(tvm::ffi::TensorView cur_seq_lens,
      tvm::ffi::TensorView cur_req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView pending_seq_lens,
      tvm::ffi::TensorView state,
      int64_t cur_bs,
      int64_t num_verify_tokens,
      int64_t state_slot_stride,
      int64_t max_num_reqs) {
    using namespace host;

    auto seq_dtype = SymbolicDType{};
    auto req_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(seq_dtype).with_device(device).verify(cur_seq_lens);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(req_dtype).with_device(device).verify(cur_req_pool_indices);
    TensorMatcher({-1, -1}).with_dtype<int32_t>().with_device(device).verify(req_to_token);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(pending_seq_lens);
    TensorMatcher({-1, kHeadDim * 3}).with_dtype<float>().with_device(device).verify(state);

    if (cur_bs <= 0) return;
    RuntimeCheck(num_verify_tokens > 0 && num_verify_tokens <= 8, "unsupported num_verify_tokens=", num_verify_tokens);
    RuntimeCheck(state_slot_stride > 0, "state_slot_stride must be positive");
    RuntimeCheck(cur_bs <= cur_seq_lens.shape()[0], "cur_bs exceeds seq_lens rows");
    RuntimeCheck(cur_bs <= cur_req_pool_indices.shape()[0], "cur_bs exceeds req rows");
    RuntimeCheck(max_num_reqs <= pending_seq_lens.shape()[0], "max_num_reqs exceeds pending rows");

    if (seq_dtype.is_type<int32_t>()) {
      if (req_dtype.is_type<int32_t>()) {
        launch<int32_t, int32_t>(
            cur_seq_lens, cur_req_pool_indices, req_to_token, pending_seq_lens,
            state, cur_bs, num_verify_tokens, state_slot_stride, max_num_reqs, device.unwrap());
      } else {
        launch<int32_t, int64_t>(
            cur_seq_lens, cur_req_pool_indices, req_to_token, pending_seq_lens,
            state, cur_bs, num_verify_tokens, state_slot_stride, max_num_reqs, device.unwrap());
      }
    } else {
      if (req_dtype.is_type<int32_t>()) {
        launch<int64_t, int32_t>(
            cur_seq_lens, cur_req_pool_indices, req_to_token, pending_seq_lens,
            state, cur_bs, num_verify_tokens, state_slot_stride, max_num_reqs, device.unwrap());
      } else {
        launch<int64_t, int64_t>(
            cur_seq_lens, cur_req_pool_indices, req_to_token, pending_seq_lens,
            state, cur_bs, num_verify_tokens, state_slot_stride, max_num_reqs, device.unwrap());
      }
    }
  }
};

}  // namespace
