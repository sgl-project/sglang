// Fused decode causal_conv1d, cache shift-update, and optional track copy.
//
// Decode: each token t is its own sequence (bos=t). Per token:
//   conv: acc = sum_{iw<W-1} cache[slot, iw, d]*cache_mask[t] * weight[d, iw]
//              + x[t, d] * weight[d, W-1]
//         y[t,d] = act(acc) (+ x[t,d] if residual).
//   update (valid lanes, ci != PAD): shift the state left, append current token --
//     new[iw] = cache[slot, iw+1]*cache_mask[t] (iw < W-2);  new[W-2] = x[t].
//   track (DO_TRACK): the same post-update window is also written to
//     cache[track_indices[t]] wherever track_mask[t] (prefix-cache ping-pong slot).
// Working slots and ping-pong track slots are pairwise-distinct, so writes never race.
// Cache history is loaded to registers BEFORE any write (RAW-safe). 2 channels/thread
// as bf16x2; conv accumulates in fp32 (matches the fp32 reference), update is a
// bit-exact bf16 move. Requires bf16 + even D + channel-contiguous cache/x/y.
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace {

struct DecodeUpdateParams {
  const void* __restrict__ x;              // [T, D], channel-contiguous
  void* __restrict__ cache;                // [pool, W-1, D], in-place update
  const void* __restrict__ cache_indices;  // int32 [T]  (PAD == -1)
  const void* __restrict__ cache_mask;     // bool  [T]
  const void* __restrict__ weight;         // [D, W]
  void* __restrict__ y;                    // [T, D] contiguous output
  const void* __restrict__ track_mask;     // bool  [T]   (DO_TRACK only)
  const void* __restrict__ track_indices;  // int64 [T]   (DO_TRACK only)
  int64_t x_stride_t;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t weight_stride_d;
  int64_t y_stride_t;
  int64_t track_idx_stride;
  uint32_t D;
};

constexpr uint32_t kDecThreads = 256;
constexpr int kPadSlot = -1;

template <int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_TRACK, typename DType>
__global__ void fused_decode_update_kernel(const __grid_constant__ DecodeUpdateParams p) {
  constexpr int W1 = W - 1;  // number of cached history taps / conv-state rows
  const int t = blockIdx.y;
  const int ci = static_cast<const int32_t*>(p.cache_indices)[t];
  const bool valid = ci != kPadSlot;
  const int slot = valid ? ci : 0;  // clamp: PAD lanes still emit y (discarded), no cache write

  const int c0 = (blockIdx.x * kDecThreads + threadIdx.x) * 2;
  if (c0 >= static_cast<int>(p.D)) return;

  const float cm = static_cast<const bool*>(p.cache_mask)[t] ? 1.0f : 0.0f;
  const auto* xp = static_cast<const __nv_bfloat16*>(p.x);
  const auto* wp = static_cast<const __nv_bfloat16*>(p.weight);
  auto* cp = static_cast<__nv_bfloat16*>(p.cache);
  auto* yp = static_cast<__nv_bfloat16*>(p.y);
  const int cw = static_cast<int>(p.cache_stride_w);
  const int swd = static_cast<int>(p.weight_stride_d);
  const int64_t cache_base = static_cast<int64_t>(slot) * p.cache_stride_slot + c0;

  // History taps -> registers (RAW-safe against the update writes below).
  __nv_bfloat162 hist[W1];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    hist[w] = *reinterpret_cast<const __nv_bfloat162*>(&cp[cache_base + static_cast<int64_t>(w) * cw]);
  }
  const __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(&xp[static_cast<int64_t>(t) * p.x_stride_t + c0]);
  const float2 xf = __bfloat1622float2(xv);

  float2 wv[W];
#pragma unroll
  for (int iw = 0; iw < W; ++iw) {
    wv[iw] = make_float2(__bfloat162float(wp[c0 * swd + iw]), __bfloat162float(wp[(c0 + 1) * swd + iw]));
  }

  // ---- conv (fp32 accum): W-1 cached taps (gated by cache_mask) + current token ----
  float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
  for (int iw = 0; iw < W1; ++iw) {
    const float2 h = __bfloat1622float2(hist[iw]);
    acc0 += h.x * cm * wv[iw].x;
    acc1 += h.y * cm * wv[iw].y;
  }
  acc0 += xf.x * wv[W1].x;
  acc1 += xf.y * wv[W1].y;
  if constexpr (USE_SILU) {
    acc0 = __fdividef(acc0, 1.0f + __expf(-acc0));
    acc1 = __fdividef(acc1, 1.0f + __expf(-acc1));
  }
  if constexpr (USE_RESIDUAL) {
    acc0 += xf.x;
    acc1 += xf.y;
  }
  *reinterpret_cast<__nv_bfloat162*>(&yp[static_cast<int64_t>(t) * p.y_stride_t + c0]) =
      __floats2bfloat162_rn(acc0, acc1);

  if (!valid) return;

  // ---- update: shift state left (gated by cache_mask), append current token ----
  const __nv_bfloat162 zero = __float2bfloat162_rn(0.0f);
  int64_t track_base = 0;
  bool do_tr = false;
  if constexpr (DO_TRACK) {
    do_tr = static_cast<const bool*>(p.track_mask)[t];
    if (do_tr) {
      const int64_t tslot = static_cast<const int64_t*>(p.track_indices)[static_cast<int64_t>(t) * p.track_idx_stride];
      track_base = tslot * p.cache_stride_slot + c0;
    }
  }
#pragma unroll
  for (int iw = 0; iw < W1; ++iw) {
    const __nv_bfloat162 nv = (iw < W1 - 1) ? ((cm != 0.0f) ? hist[iw + 1] : zero) : xv;
    *reinterpret_cast<__nv_bfloat162*>(&cp[cache_base + static_cast<int64_t>(iw) * cw]) = nv;
    if constexpr (DO_TRACK) {
      if (do_tr) {
        *reinterpret_cast<__nv_bfloat162*>(&cp[track_base + static_cast<int64_t>(iw) * cw]) = nv;
      }
    }
  }
}

template <int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_TRACK, typename DType>
struct FusedDecodeUpdateKernel {
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView y,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_indices) {
    using namespace host;
    auto T = SymbolicSize{"T"};
    auto D = SymbolicSize{"D"};
    auto Wd = SymbolicSize{"W"};
    auto W1s = SymbolicSize{"W_minus_1"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    Wd.set_value(W);
    W1s.set_value(W - 1);

    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(x);
    TensorMatcher({-1, W1s, D}).with_dtype<DType>().with_device(dev).verify(cache);
    TensorMatcher({T}).with_dtype<int32_t>().with_device(dev).verify(cache_indices);
    TensorMatcher({T}).with_device(dev).verify(cache_mask);
    TensorMatcher({D, Wd}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(weight);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(y);
    RuntimeCheck(sizeof(DType) == 2, "fused_decode: bf16x2 kernel requires 16-bit dtype");
    RuntimeCheck(D.unwrap() % 2 == 0, "fused_decode: D must be even for the bf16x2 kernel");
    RuntimeCheck(cache.stride(2) == 1, "fused_decode: cache must be channel-contiguous");

    const auto params = DecodeUpdateParams{
        .x = x.data_ptr(),
        .cache = cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .weight = weight.data_ptr(),
        .y = y.data_ptr(),
        .track_mask = DO_TRACK ? track_mask.data_ptr() : nullptr,
        .track_indices = DO_TRACK ? track_indices.data_ptr() : nullptr,
        .x_stride_t = x.stride(0),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .weight_stride_d = weight.stride(0),
        .y_stride_t = y.stride(0),
        .track_idx_stride = DO_TRACK ? track_indices.stride(0) : 0,
        .D = static_cast<uint32_t>(D.unwrap()),
    };

    const uint32_t d_pairs = params.D / 2;
    const dim3 grid{div_ceil(d_pairs, kDecThreads), static_cast<uint32_t>(T.unwrap())};
    const dim3 block{kDecThreads};
    constexpr auto kernel = fused_decode_update_kernel<W, USE_SILU, USE_RESIDUAL, DO_TRACK, DType>;
    LaunchKernel(grid, block, dev.unwrap())(kernel, params);
  }
};

}  // namespace
