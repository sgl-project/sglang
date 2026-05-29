#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

template <typename TSeq, typename TReq>
struct OnlineC128MTPCommitParams {
  const float* __restrict__ kv_score_input;
  const float* __restrict__ pre_state;
  const int64_t* __restrict__ accept_lens;
  const TSeq* __restrict__ seq_lens;
  const TReq* __restrict__ req_pool_indices;
  const int32_t* __restrict__ req_to_token;
  const int64_t* __restrict__ full_to_swa;
  const float* __restrict__ ape;
  float* __restrict__ main_state;
  int64_t kv_score_stride_b;
  int64_t pre_state_stride_b;
  int64_t req_to_token_stride_b;
  int64_t ape_stride_r;
  int64_t main_state_stride_b;
  int64_t layer_bs;
  int64_t swa_page_size;
  int64_t num_verify_tokens;
};

template <int64_t kHeadDim, typename TSeq, typename TReq>
__global__ void online_c128_mtp_commit_kernel(
    const OnlineC128MTPCommitParams<TSeq, TReq> params) {
  const int64_t bid = static_cast<int64_t>(blockIdx.x);
  if (bid >= params.layer_bs) return;

  int64_t accept = params.accept_lens[bid];
  accept = accept < params.num_verify_tokens ? accept : params.num_verify_tokens;
  if (accept <= 0) return;

  const int64_t seq_before = static_cast<int64_t>(params.seq_lens[bid]);
  const int64_t start_pos = seq_before & 127;
  const int64_t final_seq = seq_before + accept;
  if ((final_seq & 127) == 0) return;

  const int64_t chunk_start = ((final_seq - 1) / 128) * 128;
  const int64_t req_idx = static_cast<int64_t>(params.req_pool_indices[bid]);
  const int64_t full_loc =
      static_cast<int64_t>(params.req_to_token[req_idx * params.req_to_token_stride_b + chunk_start]);
  const int64_t swa_loc = params.full_to_swa[full_loc];
  const int64_t slot = swa_loc / params.swa_page_size;

  const float* const pre = params.pre_state + bid * params.pre_state_stride_b;
  float* const out = params.main_state + slot * params.main_state_stride_b;

  for (int64_t d = static_cast<int64_t>(threadIdx.x); d < kHeadDim; d += blockDim.x) {
    float run_max = pre[d];
    float run_sum = pre[kHeadDim + d];
    float run_kv = pre[kHeadDim * 2 + d];

#pragma unroll
    for (int64_t step = 0; step < 8; ++step) {
      if (step >= params.num_verify_tokens) break;
      if (step >= accept) break;

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

      if (pos == 127) {
        run_kv = 0.0f;
        run_max = 0.0f;
        run_sum = 0.0f;
      }
    }

    out[d] = run_max;
    out[kHeadDim + d] = run_sum;
    out[kHeadDim * 2 + d] = run_kv;
  }
}

template <int64_t kHeadDim>
struct OnlineC128MTPCommitKernel {
  template <typename TSeq, typename TReq>
  static void launch(
      tvm::ffi::TensorView kv_score_input,
      tvm::ffi::TensorView pre_state,
      tvm::ffi::TensorView accept_lens,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView full_to_swa,
      tvm::ffi::TensorView ape,
      tvm::ffi::TensorView main_state,
      int64_t layer_bs,
      int64_t swa_page_size,
      int64_t num_verify_tokens,
      DLDevice device) {
    using namespace host;

    const auto params = OnlineC128MTPCommitParams<TSeq, TReq>{
        .kv_score_input = static_cast<const float*>(kv_score_input.data_ptr()),
        .pre_state = static_cast<const float*>(pre_state.data_ptr()),
        .accept_lens = static_cast<const int64_t*>(accept_lens.data_ptr()),
        .seq_lens = static_cast<const TSeq*>(seq_lens.data_ptr()),
        .req_pool_indices = static_cast<const TReq*>(req_pool_indices.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .full_to_swa = static_cast<const int64_t*>(full_to_swa.data_ptr()),
        .ape = static_cast<const float*>(ape.data_ptr()),
        .main_state = static_cast<float*>(main_state.data_ptr()),
        .kv_score_stride_b = kv_score_input.stride(0),
        .pre_state_stride_b = pre_state.stride(0),
        .req_to_token_stride_b = req_to_token.stride(0),
        .ape_stride_r = ape.stride(0),
        .main_state_stride_b = main_state.stride(0),
        .layer_bs = layer_bs,
        .swa_page_size = swa_page_size,
        .num_verify_tokens = num_verify_tokens,
    };

    constexpr uint32_t kThreads = 256;
    LaunchKernel(static_cast<uint32_t>(layer_bs), kThreads, device)
        (online_c128_mtp_commit_kernel<kHeadDim, TSeq, TReq>, params);
  }

  static void run(
      tvm::ffi::TensorView kv_score_input,
      tvm::ffi::TensorView pre_state,
      tvm::ffi::TensorView accept_lens,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView full_to_swa,
      tvm::ffi::TensorView ape,
      tvm::ffi::TensorView main_state,
      int64_t layer_bs,
      int64_t swa_page_size,
      int64_t num_verify_tokens) {
    using namespace host;

    auto seq_dtype = SymbolicDType{};
    auto req_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({-1, kHeadDim * 2}).with_dtype<float>().with_device(device).verify(kv_score_input);
    TensorMatcher({-1, kHeadDim * 3}).with_dtype<float>().with_device(device).verify(pre_state);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(accept_lens);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(seq_dtype).with_device(device).verify(seq_lens);
    TensorMatcher({-1}).with_dtype<int32_t, int64_t>(req_dtype).with_device(device).verify(req_pool_indices);
    TensorMatcher({-1, -1}).with_dtype<int32_t>().with_device(device).verify(req_to_token);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(full_to_swa);
    TensorMatcher({128, kHeadDim}).with_dtype<float>().with_device(device).verify(ape);
    TensorMatcher({-1, kHeadDim * 3}).with_dtype<float>().with_device(device).verify(main_state);

    if (layer_bs <= 0) return;
    RuntimeCheck(num_verify_tokens > 0 && num_verify_tokens <= 8, "unsupported num_verify_tokens=", num_verify_tokens);
    RuntimeCheck(layer_bs <= kv_score_input.shape()[0], "layer_bs exceeds kv_score_input rows");
    RuntimeCheck(layer_bs <= pre_state.shape()[0], "layer_bs exceeds pre_state rows");
    RuntimeCheck(layer_bs <= accept_lens.shape()[0], "layer_bs exceeds accept_lens rows");
    RuntimeCheck(layer_bs <= seq_lens.shape()[0], "layer_bs exceeds seq_lens rows");
    RuntimeCheck(layer_bs <= req_pool_indices.shape()[0], "layer_bs exceeds req_pool_indices rows");

    if (seq_dtype.is_type<int32_t>()) {
      if (req_dtype.is_type<int32_t>()) {
        launch<int32_t, int32_t>(
            kv_score_input, pre_state, accept_lens, seq_lens, req_pool_indices, req_to_token, full_to_swa, ape,
            main_state, layer_bs, swa_page_size, num_verify_tokens, device.unwrap());
      } else {
        launch<int32_t, int64_t>(
            kv_score_input, pre_state, accept_lens, seq_lens, req_pool_indices, req_to_token, full_to_swa, ape,
            main_state, layer_bs, swa_page_size, num_verify_tokens, device.unwrap());
      }
    } else {
      if (req_dtype.is_type<int32_t>()) {
        launch<int64_t, int32_t>(
            kv_score_input, pre_state, accept_lens, seq_lens, req_pool_indices, req_to_token, full_to_swa, ape,
            main_state, layer_bs, swa_page_size, num_verify_tokens, device.unwrap());
      } else {
        launch<int64_t, int64_t>(
            kv_score_input, pre_state, accept_lens, seq_lens, req_pool_indices, req_to_token, full_to_swa, ape,
            main_state, layer_bs, swa_page_size, num_verify_tokens, device.unwrap());
      }
    }
  }
};

}  // namespace
