// SPDX-License-Identifier: Apache-2.0
// Ungrouped sigmoid top-8 routing on Volta, one warp per token. Reproduces
// _router_triton_kernel's SM70 arithmetic (ex2.approx, div.full, its sum
// order), so ids and weights match it bitwise for 257..512 experts.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_router_topk {

inline constexpr int kTopK = 8;
inline constexpr int kWarp = 32;
inline constexpr int kPerLane = 16;
inline constexpr int kMinExperts = 257;
inline constexpr int kMaxExperts = kWarp * kPerLane;

SGL_DEVICE float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

// tl.sigmoid as Triton lowers it for SM70; 0x3FB8AA3B is its fp32 log2(e).
SGL_DEVICE float triton_sigmoid(float x) {
  const float t = __fmul_rn(__fsub_rn(0.0f, x), __int_as_float(0x3FB8AA3B));
  return div_full(1.0f, __fadd_rn(ex2_approx(t), 1.0f));
}

// Order-preserving float bits above the inverted expert id: the largest key is
// the highest value with the lowest id. 0 marks padding and taken experts,
// below every live key (even -inf), as Triton only ranks the remaining ones.
SGL_DEVICE uint64_t rank_key(float v, int e) {
  const uint32_t b = __float_as_uint(v);
  const uint32_t ord = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return (static_cast<uint64_t>(ord) << 32) | static_cast<uint32_t>(~e);
}

template <int kSlots>
__global__ __launch_bounds__(kWarp) void route_kernel(
    float* __restrict__ weights,
    int32_t* __restrict__ ids,
    const float* __restrict__ scores,
    int64_t score_stride,
    const float* __restrict__ bias,
    int32_t num_experts,
    float scale,
    bool renormalize,
    bool apply_scale) {
  const int64_t t = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const float* row = scores + t * score_stride;
  float act[kSlots];
  uint64_t key[kSlots];
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int e = lane + kWarp * j;
    const bool valid = e < num_experts;
    const float a = triton_sigmoid(valid ? row[e] : 0.0f);
    const float b = __fadd_rn(a, valid ? bias[e] : 0.0f);
    act[j] = a;
    key[j] = valid ? rank_key(b == b ? b : -1e30f, e) : 0;
  }

  float my_w = 0.0f;
  int32_t my_id = 0;
  // Rolled: one decode launch runs this once, so an unrolled body is fetched
  // cold from memory on every layer.
#pragma unroll 1
  for (int k = 0; k < kTopK; ++k) {
    uint64_t best = key[0];
#pragma unroll
    for (int j = 1; j < kSlots; ++j) {
      best = key[j] > best ? key[j] : best;
    }
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1) {
      const uint64_t other = __shfl_xor_sync(0xffffffffu, best, off);
      best = other > best ? other : best;
    }
    const int win = static_cast<int>(~static_cast<uint32_t>(best));
    float w = 0.0f;
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      const bool mine = key[j] == best;
      w = mine ? act[j] : w;
      key[j] = mine ? 0 : key[j];
    }
    w = __shfl_sync(0xffffffffu, w, win & (kWarp - 1));
    my_w = lane == k ? w : my_w;
    my_id = lane == k ? win : my_id;
  }

  float out = my_w;
  if (renormalize) {
    float slot[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      slot[k] = __shfl_sync(0xffffffffu, my_w, k);
    }
    const float lo = __fadd_rn(__fadd_rn(__fadd_rn(slot[0], slot[1]), slot[2]), slot[3]);
    const float hi = __fadd_rn(__fadd_rn(__fadd_rn(slot[4], slot[5]), slot[6]), slot[7]);
    const float sum = __fadd_rn(lo, hi);
    out = div_full(my_w, sum > 0.0f ? sum : 1.0f);
  }
  if (apply_scale) {
    out = __fmul_rn(out, scale);
  }
  if (lane < kTopK) {
    weights[t * kTopK + lane] = out;
    ids[t * kTopK + lane] = my_id;
  }
}

template <int kSlots>
void launch(uint32_t num_tokens,
            DLDevice device,
            float* weights,
            int32_t* ids,
            const float* scores,
            int64_t score_stride,
            const float* bias,
            int32_t num_experts,
            float scale,
            bool renormalize,
            bool apply_scale) {
  if constexpr (kSlots < kPerLane) {
    if (num_experts <= kSlots * kWarp) {
      host::LaunchKernel(num_tokens, kWarp, device)(
          route_kernel<kSlots>, weights, ids, scores, score_stride, bias, num_experts, scale, renormalize,
          apply_scale);
      return;
    }
    launch<kSlots + 1>(num_tokens, device, weights, ids, scores, score_stride, bias, num_experts, scale,
                       renormalize, apply_scale);
  } else {
    host::LaunchKernel(num_tokens, kWarp, device)(
        route_kernel<kSlots>, weights, ids, scores, score_stride, bias, num_experts, scale, renormalize,
        apply_scale);
  }
}

/// \brief weights/ids [M, 8] from raw logits scores [M, N] (row stride free),
///        fp32 bias [N]; 257 <= N <= 512.
void route(tvm::ffi::TensorView weights,
           tvm::ffi::TensorView ids,
           tvm::ffi::TensorView scores,
           tvm::ffi::TensorView bias,
           double scale,
           bool renormalize,
           bool apply_scale) {
  using namespace host;
  SymbolicSize m = {"num_tokens"};
  SymbolicSize n = {"num_experts"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({m, n}).with_dtype<fp32_t>().with_strides({-1, 1}).with_device<kDLCUDA>(device_).verify(scores);
  TensorMatcher({n}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({m, kTopK}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(weights);
  TensorMatcher({m, kTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(ids);
  const int64_t num_tokens = m.unwrap();
  const int64_t num_experts = n.unwrap();
  CHECK_HOST(num_experts >= kMinExperts && num_experts <= kMaxExperts);
  if (num_tokens == 0) {
    return;
  }
  launch<(kMinExperts + kWarp - 1) / kWarp>(
      static_cast<uint32_t>(num_tokens),
      device_.unwrap(),
      static_cast<float*>(weights.data_ptr()),
      static_cast<int32_t*>(ids.data_ptr()),
      static_cast<const float*>(scores.data_ptr()),
      scores.stride(0),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<int32_t>(num_experts),
      static_cast<float>(scale),
      renormalize,
      apply_scale);
}

}  // namespace sglang::sm70_router_topk
