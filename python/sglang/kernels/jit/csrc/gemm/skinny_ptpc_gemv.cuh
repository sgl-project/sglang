// Skinny (M == 1) fp8 per-token x per-channel GEMV over aiter
// (16,16)-preshuffled weights, gfx950 only.
//
//   out[n] = sum_k(a[k] * w[n, k]) * x_scale * w_scale[n], bf16 out.
//
// Adapted from vLLM's wvSplitKQ:
// https://github.com/vllm-project/vllm/blob/v0.11.0/csrc/rocm/skinny_gemms.cu
// (also vendored as aiter csrc/kernels/custom_kernels.cu): fp8 operands feed
// MFMA directly (no converts) and the weight stream uses nontemporal loads.
// Adapted to the preshuffled layout -- v_mfma_scale_f32_16x16x128_f8f6f4's
// B tile (16 n x 128 k) is exactly four consecutive 512B blocks of the
// (16,16) shuffle, so each lane supplies its 32 weight bytes with two
// full-width 16B loads -- and to per-channel weight scales, which wvSplitKQ
// (per-tensor scales, row-major weights) cannot express without a second
// weight copy.
//
// Grid: one CTA per 16-row n-tile; four waves split K and reduce through
// LDS. No cross-CTA traffic, so the kernel is trivially CUDA-graph safe.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

#ifdef USE_ROCM
#include <hip/hip_bf16.h>

namespace sglang {

namespace skinny_gemv {

constexpr int kThreads = 256;
constexpr int kNumWaves = kThreads / 64;         // 4 wave64
constexpr int kTileN = 16;                       // output rows per CTA / MFMA B columns
constexpr int kKPerMfma = 128;                   // K depth of one mfma_..._16x16x128 issue
constexpr int kUnroll = 4;                       // MFMA tiles in flight per wave iteration
constexpr int kGroupBytes = kKPerMfma * kTileN;  // one 16 n x 128 k B tile
// Each wave's K share must be an exact multiple of the unroll.
constexpr int kKGranularity = kKPerMfma * kNumWaves * kUnroll;

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8 = __attribute__((__vector_size__(8 * sizeof(int)))) int;

// 32 fp8 values as either two 16B load results or one MFMA operand.
union Operand {
  int32x4 quad[2];
  int32x8 full;
};

__global__ void __launch_bounds__(kThreads) skinny_ptpc_gemv_kernel(
    const uint8_t* __restrict__ a,      // [K] fp8
    const uint8_t* __restrict__ w,      // [N*K] shuffled fp8
    const float* __restrict__ x_scale,  // [1]
    const float* __restrict__ w_scale,  // [N]
    __hip_bfloat16* __restrict__ out,   // [N]
    int K) {
  const int t = threadIdx.x;
  const int wave = t >> 6;
  const int lane = t & 63;
  const int g = lane >> 4;   // 512B block within the 16x128 B tile
  const int nl = lane & 15;  // B column / C column

  const uint8_t* w_nt = w + static_cast<int64_t>(blockIdx.x) * K * kTileN;
  const int groups_per_wave = K / kKPerMfma / kNumWaves;
  const int g_begin = wave * groups_per_wave;
  const int g_end = g_begin + groups_per_wave;
  const bool row0 = (nl == 0);  // M == 1: only A row 0 carries data
  const int b_off = g * 512 + nl * 16;

  // Two accumulators so consecutive MFMAs are independent.
  floatx4 acc0 = {0.f, 0.f, 0.f, 0.f};
  floatx4 acc1 = {0.f, 0.f, 0.f, 0.f};
  for (int gr = g_begin; gr < g_end; gr += kUnroll) {
    Operand wv[kUnroll], av[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const uint8_t* wb = w_nt + static_cast<int64_t>(gr + j) * kGroupBytes + b_off;
      wv[j].quad[0] = __builtin_nontemporal_load(reinterpret_cast<const int32x4*>(wb));
      wv[j].quad[1] = __builtin_nontemporal_load(reinterpret_cast<const int32x4*>(wb + 256));
      if (row0) {
        const int32x4* ab = reinterpret_cast<const int32x4*>(a + (gr + j) * kKPerMfma + g * 32);
        av[j].quad[0] = ab[0];
        av[j].quad[1] = ab[1];
      } else {
        av[j].quad[0] = int32x4{0, 0, 0, 0};
        av[j].quad[1] = int32x4{0, 0, 0, 0};
      }
    }
#if defined(__gfx950__)
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      // cbsz/blgp 0 = fp8 e4m3; E8M0 scale 127 = x1.0 (unscaled).
      if (j & 1) {
        acc1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av[j].full, wv[j].full, acc1, 0, 0, 0, 127, 0, 127);
      } else {
        acc0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av[j].full, wv[j].full, acc0, 0, 0, 0, 127, 0, 127);
      }
    }
#elif defined(__HIP_DEVICE_COMPILE__)
#error "skinny_ptpc_gemv requires gfx950 (v_mfma_scale_f32_16x16x128_f8f6f4)"
#endif
  }
  acc0[0] += acc1[0];

  // C row 0 of the 16x16 tile lives in lanes 0..15, vgpr 0.
  __shared__ float red[kNumWaves][kTileN];
  if (lane < kTileN) {
    red[wave][lane] = acc0[0];
  }
  __syncthreads();
  if (t < kTileN) {
    float tot = 0.f;
#pragma unroll
    for (int wv_i = 0; wv_i < kNumWaves; ++wv_i) {
      tot += red[wv_i][t];
    }
    const int n = blockIdx.x * kTileN + t;
    out[n] = __float2bfloat16(tot * x_scale[0] * w_scale[n]);
  }
}

}  // namespace skinny_gemv

// -------------------------------------------------------------------------
// Launcher
// -------------------------------------------------------------------------
inline void skinny_ptpc_gemv(
    tvm::ffi::TensorView a,        // [K] fp8 bytes (uint8 view)
    tvm::ffi::TensorView w,        // [N*K] shuffled fp8 bytes (uint8 view)
    tvm::ffi::TensorView x_scale,  // [1] fp32
    tvm::ffi::TensorView w_scale,  // [N] fp32
    tvm::ffi::TensorView out) {    // [1, N] bf16
  using namespace host;

  SymbolicSize K = {"K"};
  SymbolicSize N = {"N"};
  SymbolicSize WBytes = {"w_bytes"};
  SymbolicSize M = {"M"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA, kDLROCM>();

  TensorMatcher({K}).with_dtype<uint8_t>().with_device(device_).verify(a);
  TensorMatcher({WBytes}).with_dtype<uint8_t>().with_device(device_).verify(w);
  TensorMatcher({M}).with_dtype<fp32_t>().with_device(device_).verify(x_scale);
  TensorMatcher({N}).with_dtype<fp32_t>().with_device(device_).verify(w_scale);
  TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device_).verify(out);

  const int64_t k = K.unwrap();
  const int64_t n = N.unwrap();
  RuntimeCheck(M.unwrap() == 1, "only M == 1 is supported");
  RuntimeCheck(WBytes.unwrap() == n * k, "weight must hold N*K bytes, got ", WBytes.unwrap());
  RuntimeCheck(
      k % skinny_gemv::kKGranularity == 0, "K must be a multiple of ", skinny_gemv::kKGranularity, ", got ", k);
  RuntimeCheck(n % skinny_gemv::kTileN == 0, "N must be a multiple of 16, got ", n);
  RuntimeCheck(k <= INT32_MAX, "K exceeds int32, got ", k);
  const DLDevice device = device_.unwrap();

  LaunchKernel(static_cast<uint32_t>(n / skinny_gemv::kTileN), skinny_gemv::kThreads, device, 0)(
      skinny_gemv::skinny_ptpc_gemv_kernel,
      static_cast<const uint8_t*>(a.data_ptr()),
      static_cast<const uint8_t*>(w.data_ptr()),
      static_cast<const float*>(x_scale.data_ptr()),
      static_cast<const float*>(w_scale.data_ptr()),
      static_cast<__hip_bfloat16*>(out.data_ptr()),
      static_cast<int>(k));
}

}  // namespace sglang

#endif  // USE_ROCM
