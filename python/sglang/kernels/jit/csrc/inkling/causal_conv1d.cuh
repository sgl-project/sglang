// Depthwise causal conv1d (extend/prefill) with the W-1 prefix taps gathered
// directly from sconv_cache.
//
// Semantics:
// For packed token t in sequence s (bos = cu_seqlens[s], slot = safe_idx[s]) and
// tap iw in 0..W-1, shifted = t - (W-1) + iw:
//   shifted >= bos (in-seq history)          -> tap = x[shifted, d]
//   shifted <  bos, pp=shifted-bos+(W-1)>=0  -> tap = cache[slot, pp, d]
//                                               (* cache_mask[s] when !IS_DECODE)
//   else                                     -> tap = 0
//   out[t,d] = act(sum_iw tap*weight[d,iw]) (+ x[t,d] if residual), fp32 accum.
// in_x / in_prefix are mutually exclusive, so the fp32 tap sum is bit-identical to
// the Triton bf16 add (one operand is always 0).
//
// Channel-independent control is shared by two channels packed as bf16x2.
// Each thread keeps a token strip and its prefix window in registers across taps.
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/type.cuh>   // For fp32_t / bf16_t aliases
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace {

struct CausalConv1dParams {
  const void* __restrict__ x;           // [T, D]
  const void* __restrict__ cache;       // [max_slots, W-1, D]
  const void* __restrict__ safe_idx;    // int64 [nseq]  cache slot per sequence
  const void* __restrict__ cache_mask;  // bool  [nseq,1,1] raw metadata
  const void* __restrict__ weight;      // [D, W]
  const void* __restrict__ cu;          // int64 [nseq+1] packed sequence starts
  const void* __restrict__ seq_idx;     // int32 [T]     sequence id per token
  void* __restrict__ y;                 // [T, D] contiguous output
  int64_t x_stride_t;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t cache_mask_stride;
  int64_t weight_stride_d;
  int64_t y_stride_t;
  uint32_t T;
  uint32_t D;
};

constexpr int kConvBlockT = 4;          // tokens per thread strip
constexpr uint32_t kConvThreads = 256;  // threads per block (each owns 2 channels)

// blockIdx.x = token strip (BLOCK_T tokens); each thread owns channel pair (c0, c0+1).
// Requires bf16, D even, and unit channel/row-inner stride (host-checked).
template <int W, bool USE_SILU, bool USE_RESIDUAL, bool IS_DECODE, typename DType>
__global__ void causal_conv1d_kernel(const __grid_constant__ CausalConv1dParams p) {
  constexpr int BT = kConvBlockT;
  constexpr int WIN = BT + (W - 1);

  __shared__ int s_bos[BT];
  __shared__ int s_slot[BT];
  __shared__ float s_m[BT];

  const int T = static_cast<int>(p.T);
  const int t0 = static_cast<int>(blockIdx.x) * BT;

  if (threadIdx.x < static_cast<uint32_t>(BT)) {
    const int j = static_cast<int>(threadIdx.x);
    const int t = t0 + j;
    if (t < T) {
      const int seq = static_cast<const int32_t*>(p.seq_idx)[t];
      s_bos[j] = static_cast<int>(static_cast<const int64_t*>(p.cu)[seq]);
      s_slot[j] = static_cast<int>(static_cast<const int64_t*>(p.safe_idx)[seq]);
      if constexpr (!IS_DECODE) {
        s_m[j] = static_cast<const bool*>(p.cache_mask)[static_cast<int64_t>(seq) * p.cache_mask_stride] ? 1.0f : 0.0f;
      }
    }
  }
  __syncthreads();

  const int c0 = (blockIdx.y * kConvThreads + threadIdx.x) * 2;  // this thread's channel pair
  if (c0 >= static_cast<int>(p.D)) return;

  const int sxt = static_cast<int>(p.x_stride_t);
  const int syt = static_cast<int>(p.y_stride_t);
  const int swd = static_cast<int>(p.weight_stride_d);

  const auto* xp = static_cast<const __nv_bfloat16*>(p.x);
  const auto* cp = static_cast<const __nv_bfloat16*>(p.cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(p.weight);
  auto* yp = static_cast<__nv_bfloat16*>(p.y);

  // Window (bf16x2 per row), read once into registers.
  __nv_bfloat162 xr[WIN];
#pragma unroll
  for (int i = 0; i < WIN; ++i) {
    const int row = t0 - (W - 1) + i;
    xr[i] = (row >= 0 && row < T) ? *reinterpret_cast<const __nv_bfloat162*>(&xp[row * sxt + c0])
                                  : __float2bfloat162_rn(0.0f);
  }
  // Weight taps for the two channels (weight[c0, iw], weight[c0+1, iw]).
  float2 wv[W];
#pragma unroll
  for (int iw = 0; iw < W; ++iw) {
    wv[iw] = make_float2(__bfloat162float(wp[c0 * swd + iw]), __bfloat162float(wp[(c0 + 1) * swd + iw]));
  }

#pragma unroll
  for (int j = 0; j < BT; ++j) {
    const int t = t0 + j;
    if (t >= T) break;
    const int bos = s_bos[j];
    const float2 x_cur = __bfloat1622float2(xr[j + (W - 1)]);  // tap iw == W-1

    float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
    for (int iw = 0; iw < W; ++iw) {
      float2 tap;
      if (iw == W - 1) {
        tap = x_cur;
      } else {
        const int shifted = t - (W - 1) + iw;  // < T always
        tap = (shifted >= bos) ? __bfloat1622float2(xr[j + iw]) : make_float2(0.0f, 0.0f);
        const int prefix_pos = shifted - bos + (W - 1);
        if (shifted < bos && prefix_pos >= 0 && prefix_pos < (W - 1)) {  // rare: seq start
          const int64_t coff = static_cast<int64_t>(s_slot[j]) * p.cache_stride_slot +
                               static_cast<int64_t>(prefix_pos) * p.cache_stride_w + static_cast<int64_t>(c0);
          float2 pv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&cp[coff]));
          if constexpr (!IS_DECODE) {
            pv.x *= s_m[j];
            pv.y *= s_m[j];
          }
          tap.x += pv.x;
          tap.y += pv.y;
        }
      }
      acc0 += tap.x * wv[iw].x;
      acc1 += tap.y * wv[iw].y;
    }

    if constexpr (USE_SILU) {
      acc0 = __fdividef(acc0, 1.0f + __expf(-acc0));  // silu = x*sigmoid(x)
      acc1 = __fdividef(acc1, 1.0f + __expf(-acc1));
    }
    if constexpr (USE_RESIDUAL) {
      acc0 += x_cur.x;
      acc1 += x_cur.y;
    }
    *reinterpret_cast<__nv_bfloat162*>(&yp[t * syt + c0]) = __floats2bfloat162_rn(acc0, acc1);
  }
}

template <int W, bool USE_SILU, bool USE_RESIDUAL, bool IS_DECODE, typename DType>
struct CausalConv1dKernel {
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView safe_idx,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView cu,
      tvm::ffi::TensorView seq_idx,
      tvm::ffi::TensorView y) {
    using namespace host;
    auto T = SymbolicSize{"T"};
    auto D = SymbolicSize{"D"};
    auto Wd = SymbolicSize{"W"};
    auto Km1 = SymbolicSize{"W_minus_1"};
    auto NS = SymbolicSize{"nseq"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    Wd.set_value(W);
    Km1.set_value(W - 1);

    // x may be a non-contiguous row view (stride_t arbitrary) but must be
    // channel-contiguous. cache_mask is torch-bool (verify shape/device only).
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(x);
    TensorMatcher({-1, Km1, D}).with_dtype<DType>().with_device(dev).verify(cache);
    TensorMatcher({NS}).with_dtype<int64_t>().with_device(dev).verify(safe_idx);
    TensorMatcher({NS, 1, 1}).with_device(dev).verify(cache_mask);
    TensorMatcher({D, Wd}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(weight);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(dev).verify(cu);
    TensorMatcher({T}).with_dtype<int32_t>().with_device(dev).verify(seq_idx);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(y);
    RuntimeCheck(cu.size(0) == NS.unwrap() + 1, "cu must have length nseq+1");
    RuntimeCheck(sizeof(DType) == 2, "causal_conv1d: bf16x2 kernel requires a 16-bit dtype");
    RuntimeCheck(D.unwrap() % 2 == 0, "causal_conv1d: D must be even for the bf16x2 kernel");
    RuntimeCheck(cache.stride(2) == 1, "causal_conv1d: sconv_cache must be channel-contiguous");

    const auto params = CausalConv1dParams{
        .x = x.data_ptr(),
        .cache = cache.data_ptr(),
        .safe_idx = safe_idx.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .weight = weight.data_ptr(),
        .cu = cu.data_ptr(),
        .seq_idx = seq_idx.data_ptr(),
        .y = y.data_ptr(),
        .x_stride_t = x.stride(0),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .cache_mask_stride = cache_mask.stride(0),
        .weight_stride_d = weight.stride(0),
        .y_stride_t = y.stride(0),
        .T = static_cast<uint32_t>(T.unwrap()),
        .D = static_cast<uint32_t>(D.unwrap()),
    };

    const uint32_t d_pairs = params.D / 2;
    const dim3 grid{div_ceil(params.T, static_cast<uint32_t>(kConvBlockT)), div_ceil(d_pairs, kConvThreads)};
    const dim3 block{kConvThreads};
    constexpr auto kernel = causal_conv1d_kernel<W, USE_SILU, USE_RESIDUAL, IS_DECODE, DType>;
    LaunchKernel(grid, block, dev.unwrap())(kernel, params);
  }
};

}  // namespace
