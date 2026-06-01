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
  const int64_t* __restrict__ full_to_swa;
  const float* __restrict__ ape;
  float* __restrict__ state;
  int64_t kv_score_stride_b;
  int64_t req_to_token_stride_b;
  int64_t ape_stride_r;
  int64_t state_stride_b;
  int64_t layer_bs;
  int64_t swa_page_size;
  int64_t num_verify_tokens;
  int64_t state_slot_stride;
};

template <typename TOldSeq, typename TCurSeq, typename TOldReq, typename TCurReq>
struct OnlineC128MTPLazyCommitParams {
  const TOldSeq* __restrict__ old_seq_lens;
  const TOldReq* __restrict__ old_req_pool_indices;
  const TCurSeq* __restrict__ cur_seq_lens;
  const TCurReq* __restrict__ cur_req_pool_indices;
  const int32_t* __restrict__ req_to_token;
  const int64_t* __restrict__ full_to_swa;
  float* __restrict__ state;
  int64_t old_bs;
  int64_t cur_bs;
  int64_t req_to_token_stride_b;
  int64_t state_stride_b;
  int64_t swa_page_size;
  int64_t num_verify_tokens;
  int64_t state_slot_stride;
};

template <int64_t kHeadDim, typename TSeq, typename TReq>
__global__ void online_c128_mtp_write_prefix_kernel(
    const OnlineC128MTPWritePrefixParams<TSeq, TReq> params) {
  const int64_t bid = static_cast<int64_t>(blockIdx.x);
  if (bid >= params.layer_bs) return;

  const int64_t seq_before = static_cast<int64_t>(params.seq_lens[bid]);
  const int64_t req_idx = static_cast<int64_t>(params.req_pool_indices[bid]);
  const int64_t start_pos = seq_before & 127;
  const bool has_partial = seq_before > 0 && start_pos != 0;

  int64_t init_slot = 0;
  if (has_partial) {
    const int64_t chunk_start = ((seq_before - 1) / 128) * 128;
    const int64_t full_loc =
        static_cast<int64_t>(params.req_to_token[req_idx * params.req_to_token_stride_b + chunk_start]);
    const int64_t swa_loc = params.full_to_swa[full_loc];
    init_slot = swa_loc / params.swa_page_size;
  }

  for (int64_t d = static_cast<int64_t>(threadIdx.x); d < kHeadDim; d += blockDim.x) {
    float run_max = 0.0f;
    float run_sum = 0.0f;
    float run_kv = 0.0f;
    if (has_partial) {
      const float* const init = params.state + init_slot * params.state_stride_b;
      run_max = init[d];
      run_sum = init[kHeadDim + d];
      run_kv = init[kHeadDim * 2 + d];
    }

#pragma unroll
    for (int64_t step = 0; step < 8; ++step) {
      if (step >= params.num_verify_tokens) break;

      const int64_t pos = (start_pos + step) & 127;
      const float* const kv =
          params.kv_score_input + (bid * params.num_verify_tokens + step) * params.kv_score_stride_b;
      const float kv_step = kv[d];
      const float score_step = kv[kHeadDim + d] + params.ape[pos * params.ape_stride_r + d];

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
        const int64_t chunk_start = ((final_seq - 1) / 128) * 128;
        const int64_t full_loc =
            static_cast<int64_t>(params.req_to_token[req_idx * params.req_to_token_stride_b + chunk_start]);
        const int64_t swa_loc = params.full_to_swa[full_loc];
        const int64_t slot = swa_loc / params.swa_page_size + (step + 1) * params.state_slot_stride;
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
}

template <int64_t kHeadDim, typename TOldSeq, typename TCurSeq, typename TOldReq, typename TCurReq>
__global__ void online_c128_mtp_lazy_commit_kernel(
    const OnlineC128MTPLazyCommitParams<TOldSeq, TCurSeq, TOldReq, TCurReq> params) {
  const int64_t bid = static_cast<int64_t>(blockIdx.x);
  if (bid >= params.old_bs) return;

  const int64_t old_req = static_cast<int64_t>(params.old_req_pool_indices[bid]);
  const int64_t old_seq = static_cast<int64_t>(params.old_seq_lens[bid]);
  int64_t matched_seq = old_seq;
  for (int64_t i = 0; i < params.cur_bs; ++i) {
    if (static_cast<int64_t>(params.cur_req_pool_indices[i]) == old_req) {
      matched_seq = static_cast<int64_t>(params.cur_seq_lens[i]);
      break;
    }
  }

  const int64_t accept = clamp_accept_len(matched_seq - old_seq, params.num_verify_tokens);
  if (accept <= 0) return;

  const int64_t final_seq = old_seq + accept;
  if ((final_seq & 127) == 0) return;

  const int64_t chunk_start = ((final_seq - 1) / 128) * 128;
  const int64_t full_loc =
      static_cast<int64_t>(params.req_to_token[old_req * params.req_to_token_stride_b + chunk_start]);
  const int64_t swa_loc = params.full_to_swa[full_loc];
  const int64_t slot = swa_loc / params.swa_page_size;
  const float* const src = params.state + (slot + accept * params.state_slot_stride) * params.state_stride_b;
  float* const dst = params.state + slot * params.state_stride_b;

  for (int64_t d = static_cast<int64_t>(threadIdx.x); d < kHeadDim * 3; d += blockDim.x) {
    dst[d] = src[d];
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
      tvm::ffi::TensorView full_to_swa,
      tvm::ffi::TensorView ape,
      tvm::ffi::TensorView state,
      int64_t layer_bs,
      int64_t swa_page_size,
      int64_t num_verify_tokens,
      int64_t state_slot_stride,
      DLDevice device) {
    using namespace host;

    const auto params = OnlineC128MTPWritePrefixParams<TSeq, TReq>{
        .kv_score_input = static_cast<const float*>(kv_score_input.data_ptr()),
        .seq_lens = static_cast<const TSeq*>(seq_lens.data_ptr()),
        .req_pool_indices = static_cast<const TReq*>(req_pool_indices.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .full_to_swa = static_cast<const int64_t*>(full_to_swa.data_ptr()),
        .ape = static_cast<const float*>(ape.data_ptr()),
        .state = static_cast<float*>(state.data_ptr()),
        .kv_score_stride_b = kv_score_input.stride(0),
        .req_to_token_stride_b = req_to_token.stride(0),
        .ape_stride_r = ape.stride(0),
        .state_stride_b = state.stride(0),
        .layer_bs = layer_bs,
        .swa_page_size = swa_page_size,
        .num_verify_tokens = num_verify_tokens,
        .state_slot_stride = state_slot_stride,
    };

    constexpr uint32_t kThreads = 256;
    LaunchKernel(static_cast<uint32_t>(layer_bs), kThreads, device)
        (online_c128_mtp_write_prefix_kernel<kHeadDim, TSeq, TReq>, params);
  }

  static void run(
      tvm::ffi::TensorView kv_score_input,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView full_to_swa,
      tvm::ffi::TensorView ape,
      tvm::ffi::TensorView state,
      int64_t layer_bs,
      int64_t swa_page_size,
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
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(full_to_swa);
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
            kv_score_input, seq_lens, req_pool_indices, req_to_token, full_to_swa, ape, state,
            layer_bs, swa_page_size, num_verify_tokens, state_slot_stride, device.unwrap());
      } else {
        launch<int32_t, int64_t>(
            kv_score_input, seq_lens, req_pool_indices, req_to_token, full_to_swa, ape, state,
            layer_bs, swa_page_size, num_verify_tokens, state_slot_stride, device.unwrap());
      }
    } else {
      if (req_dtype.is_type<int32_t>()) {
        launch<int64_t, int32_t>(
            kv_score_input, seq_lens, req_pool_indices, req_to_token, full_to_swa, ape, state,
            layer_bs, swa_page_size, num_verify_tokens, state_slot_stride, device.unwrap());
      } else {
        launch<int64_t, int64_t>(
            kv_score_input, seq_lens, req_pool_indices, req_to_token, full_to_swa, ape, state,
            layer_bs, swa_page_size, num_verify_tokens, state_slot_stride, device.unwrap());
      }
    }
  }
};

template <int64_t kHeadDim>
struct OnlineC128MTPLazyCommitKernel {
  template <typename TOldSeq, typename TCurSeq, typename TOldReq, typename TCurReq>
  static void launch(
      tvm::ffi::TensorView old_seq_lens,
      tvm::ffi::TensorView old_req_pool_indices,
      tvm::ffi::TensorView cur_seq_lens,
      tvm::ffi::TensorView cur_req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView full_to_swa,
      tvm::ffi::TensorView state,
      int64_t old_bs,
      int64_t cur_bs,
      int64_t swa_page_size,
      int64_t num_verify_tokens,
      int64_t state_slot_stride,
      DLDevice device) {
    using namespace host;

    const auto params = OnlineC128MTPLazyCommitParams<TOldSeq, TCurSeq, TOldReq, TCurReq>{
        .old_seq_lens = static_cast<const TOldSeq*>(old_seq_lens.data_ptr()),
        .old_req_pool_indices = static_cast<const TOldReq*>(old_req_pool_indices.data_ptr()),
        .cur_seq_lens = static_cast<const TCurSeq*>(cur_seq_lens.data_ptr()),
        .cur_req_pool_indices = static_cast<const TCurReq*>(cur_req_pool_indices.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .full_to_swa = static_cast<const int64_t*>(full_to_swa.data_ptr()),
        .state = static_cast<float*>(state.data_ptr()),
        .old_bs = old_bs,
        .cur_bs = cur_bs,
        .req_to_token_stride_b = req_to_token.stride(0),
        .state_stride_b = state.stride(0),
        .swa_page_size = swa_page_size,
        .num_verify_tokens = num_verify_tokens,
        .state_slot_stride = state_slot_stride,
    };

    constexpr uint32_t kThreads = 256;
    LaunchKernel(static_cast<uint32_t>(old_bs), kThreads, device)
        (online_c128_mtp_lazy_commit_kernel<kHeadDim, TOldSeq, TCurSeq, TOldReq, TCurReq>, params);
  }

  static void run(
      tvm::ffi::TensorView old_seq_lens,
      tvm::ffi::TensorView old_req_pool_indices,
      tvm::ffi::TensorView cur_seq_lens,
      tvm::ffi::TensorView cur_req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView full_to_swa,
      tvm::ffi::TensorView state,
      int64_t old_bs,
      int64_t cur_bs,
      int64_t swa_page_size,
      int64_t num_verify_tokens,
      int64_t state_slot_stride) {
    using namespace host;

    auto old_seq_dtype = SymbolicDType{};
    auto cur_seq_dtype = SymbolicDType{};
    auto old_req_dtype = SymbolicDType{};
    auto cur_req_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(old_seq_dtype).with_device(device).verify(old_seq_lens);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(old_req_dtype).with_device(device).verify(old_req_pool_indices);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(cur_seq_dtype).with_device(device).verify(cur_seq_lens);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(cur_req_dtype).with_device(device).verify(cur_req_pool_indices);
    TensorMatcher({-1, -1}).with_dtype<int32_t>().with_device(device).verify(req_to_token);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(full_to_swa);
    TensorMatcher({-1, kHeadDim * 3}).with_dtype<float>().with_device(device).verify(state);

    if (old_bs <= 0 || cur_bs <= 0) return;
    RuntimeCheck(num_verify_tokens > 0 && num_verify_tokens <= 8, "unsupported num_verify_tokens=", num_verify_tokens);
    RuntimeCheck(state_slot_stride > 0, "state_slot_stride must be positive");
    RuntimeCheck(old_bs <= old_seq_lens.shape()[0], "old_bs exceeds old_seq_lens rows");
    RuntimeCheck(old_bs <= old_req_pool_indices.shape()[0], "old_bs exceeds old_req rows");
    RuntimeCheck(cur_bs <= cur_seq_lens.shape()[0], "cur_bs exceeds cur_seq_lens rows");
    RuntimeCheck(cur_bs <= cur_req_pool_indices.shape()[0], "cur_bs exceeds cur_req rows");

#define DISPATCH_CUR_SEQ(OLD_SEQ_T, OLD_REQ_T)                                                             \
  if (cur_seq_dtype.is_type<int32_t>()) {                                                                   \
    if (cur_req_dtype.is_type<int32_t>()) {                                                                 \
      launch<OLD_SEQ_T, int32_t, OLD_REQ_T, int32_t>(                                                       \
          old_seq_lens, old_req_pool_indices, cur_seq_lens, cur_req_pool_indices, req_to_token, full_to_swa, \
          state, old_bs, cur_bs, swa_page_size, num_verify_tokens, state_slot_stride, device.unwrap());      \
    } else {                                                                                                \
      launch<OLD_SEQ_T, int32_t, OLD_REQ_T, int64_t>(                                                       \
          old_seq_lens, old_req_pool_indices, cur_seq_lens, cur_req_pool_indices, req_to_token, full_to_swa, \
          state, old_bs, cur_bs, swa_page_size, num_verify_tokens, state_slot_stride, device.unwrap());      \
    }                                                                                                       \
  } else {                                                                                                  \
    if (cur_req_dtype.is_type<int32_t>()) {                                                                 \
      launch<OLD_SEQ_T, int64_t, OLD_REQ_T, int32_t>(                                                       \
          old_seq_lens, old_req_pool_indices, cur_seq_lens, cur_req_pool_indices, req_to_token, full_to_swa, \
          state, old_bs, cur_bs, swa_page_size, num_verify_tokens, state_slot_stride, device.unwrap());      \
    } else {                                                                                                \
      launch<OLD_SEQ_T, int64_t, OLD_REQ_T, int64_t>(                                                       \
          old_seq_lens, old_req_pool_indices, cur_seq_lens, cur_req_pool_indices, req_to_token, full_to_swa, \
          state, old_bs, cur_bs, swa_page_size, num_verify_tokens, state_slot_stride, device.unwrap());      \
    }                                                                                                       \
  }

    if (old_seq_dtype.is_type<int32_t>()) {
      if (old_req_dtype.is_type<int32_t>()) {
        DISPATCH_CUR_SEQ(int32_t, int32_t)
      } else {
        DISPATCH_CUR_SEQ(int32_t, int64_t)
      }
    } else {
      if (old_req_dtype.is_type<int32_t>()) {
        DISPATCH_CUR_SEQ(int64_t, int32_t)
      } else {
        DISPATCH_CUR_SEQ(int64_t, int64_t)
      }
    }
#undef DISPATCH_CUR_SEQ
  }
};

}  // namespace
