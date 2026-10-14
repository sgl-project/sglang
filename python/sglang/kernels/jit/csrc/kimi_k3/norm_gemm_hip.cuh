// K3 ROCm decode/verify (M = tokens <= 16): a row normalization fused into the
// prologue of the skinny bf16 GEMM that consumes it. One launch replaces
// "norm kernel -> bf16 GEMM".
//
//   y[M, N] = A[M, K] @ W[N, K]^T,  A = bf16(prologue(x))
//
//   kGatedHead (KDA o_norm + o_proj): per (row, head of kD = 128 columns)
//       A = bf16(((x * rstd) * nw) * sigmoid(g)),  rstd = 1 / sqrt(sum(x^2) / kD + eps)
//       (the op sequence of layer_norm_gated_fwd_kernel)
//
// Layout: a block owns kNT x 16 output columns and all M (<= 16) rows; its
// kWaves waves split K (kKB blocks of 32 per wave). Every lane (row r = lane %
// 16, k-group q = lane / 16) first issues ALL its weight loads (16 B per
// 32-wide K block per N tile), then loads its own A fragment straight from
// global memory (8 contiguous elements per K block -- exactly what the MFMA
// A operand needs, so A never goes through LDS) and normalizes it in
// registers while the weight stream is in flight:
//   kGatedHead: kKB = 4, one wave = one head (the head statistics reduce over
//               the 4 k-groups of a row with two lane shuffles).
// MFMA 16x16x16 bf16 (gfx90a+). Any bijection between MFMA k slots and actual
// columns is valid as long as A and W use the same one: each 16 B (8-column)
// chunk feeds two MFMAs. The per-wave 16 x 16 partials are summed over waves
// in LDS in wave order (deterministic) and stored as bf16.
#pragma once

#ifndef USE_ROCM
#error "norm_gemm_hip.cuh is ROCm only"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace k3_norm_gemm {

constexpr int kGatedHead = 0;
constexpr int kD = 128;
template <int kMode>
constexpr bool kGated = kMode == kGatedHead;

typedef short s4_t __attribute__((ext_vector_type(4)));
typedef float f4_t __attribute__((ext_vector_type(4)));

struct Params {
  const uint16_t* __restrict__ x;  // [M, K] bf16, row stride sx
  const uint16_t* __restrict__ g;  // [M, K] bf16 gate (kGatedHead), row stride sg
  const float* __restrict__ nwf;   // kGatedHead: [kD] fp32 o_norm weight
  const uint16_t* __restrict__ w;  // [N, K] bf16 contiguous
  uint16_t* __restrict__ y;        // [M, N] bf16, row stride sy
  int64_t sx, sg, sy;
  int32_t M, K;
  float eps;
};

__device__ __forceinline__ float bf2f(uint32_t h) {
  return __uint_as_float(h << 16);
}

__device__ __forceinline__ void unpack8(const uint4& v, float (&o)[8]) {
  const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    o[2 * i] = bf2f(a[i] & 0xFFFFu);
    o[2 * i + 1] = bf2f(a[i] >> 16);
  }
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  // RNE f32 -> bf16 pair; gfx950 lowers this to one v_cvt_pk_bf16_f32
  typedef __bf16 bf2_t __attribute__((ext_vector_type(2)));
  typedef float f2_t __attribute__((ext_vector_type(2)));
  const f2_t f = {a, b};
  const bf2_t h = __builtin_convertvector(f, bf2_t);
  return __builtin_bit_cast(uint32_t, h);
}

__device__ __forceinline__ uint4 pack8(const float (&o)[8]) {
  uint4 v;
  v.x = pack2(o[0], o[1]);
  v.y = pack2(o[2], o[3]);
  v.z = pack2(o[4], o[5]);
  v.w = pack2(o[6], o[7]);
  return v;
}

__device__ __forceinline__ s4_t lo4(const uint4& v) {
  s4_t s;
  s[0] = (short)(v.x & 0xFFFF);
  s[1] = (short)(v.x >> 16);
  s[2] = (short)(v.y & 0xFFFF);
  s[3] = (short)(v.y >> 16);
  return s;
}
__device__ __forceinline__ s4_t hi4(const uint4& v) {
  s4_t s;
  s[0] = (short)(v.z & 0xFFFF);
  s[1] = (short)(v.z >> 16);
  s[2] = (short)(v.w & 0xFFFF);
  s[3] = (short)(v.w >> 16);
  return s;
}

template <int kMode, int kWaves, int kNT, int kKB>
__global__ void __launch_bounds__(kWaves * 64) norm_gemm_kernel(const Params __grid_constant__ p) {
  __shared__ float red[kWaves * kNT * 256];

  const int lane = threadIdx.x & 63;
  const int wv = threadIdx.x >> 6;
  const int r = lane & 15;
  const int q = lane >> 4;
  const int K = p.K;
  const int kw0 = wv * (kKB * 32);
  const int nblk = blockIdx.x * kNT * 16;

  // 1) this lane's A fragment (row r, columns kw0 + 32 b + 8 q + [0, 8)) and
  // its norm weights first: s_waitcnt retires loads in issue order, so the
  // prologue only overlaps the weight stream if its own loads go out before it
  const bool live = r < p.M;
  const int rr = live ? r : 0;  // branch-free loads: dead rows read row 0, zeroed below
  uint4 xr[kKB];
  uint4 gr[kGated<kMode> ? kKB : 1];
  float nwb[kGated<kMode> ? kKB * 8 : 1];
#pragma unroll
  for (int b = 0; b < kKB; ++b) {
    xr[b] = *reinterpret_cast<const uint4*>(p.x + rr * p.sx + kw0 + 32 * b + 8 * q);
    if constexpr (kGated<kMode>) {
      gr[b] = *reinterpret_cast<const uint4*>(p.g + rr * p.sg + kw0 + 32 * b + 8 * q);
      // fp32 o_norm weight; kKB * 32 == kD so column % kD = 32 b + 8 q
      const float4 u0 = *reinterpret_cast<const float4*>(p.nwf + 32 * b + 8 * q);
      const float4 u1 = *reinterpret_cast<const float4*>(p.nwf + 32 * b + 8 * q + 4);
      nwb[b * 8 + 0] = u0.x;
      nwb[b * 8 + 1] = u0.y;
      nwb[b * 8 + 2] = u0.z;
      nwb[b * 8 + 3] = u0.w;
      nwb[b * 8 + 4] = u1.x;
      nwb[b * 8 + 5] = u1.y;
      nwb[b * 8 + 6] = u1.z;
      nwb[b * 8 + 7] = u1.w;
    }
  }

  // 2) weight stream: all of this lane's loads in flight before any math
  uint4 wreg[kNT][kKB];
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    const uint16_t* wrow = p.w + (int64_t)(nblk + t * 16 + r) * K + kw0 + 8 * q;
#pragma unroll
    for (int b = 0; b < kKB; ++b)
      wreg[t][b] = *reinterpret_cast<const uint4*>(wrow + 32 * b);
  }

  if (!live) {
#pragma unroll
    for (int b = 0; b < kKB; ++b)
      xr[b] = make_uint4(0, 0, 0, 0);
  }
  float ss = 0.f;
#pragma unroll
  for (int b = 0; b < kKB; ++b) {
    float xf[8];
    unpack8(xr[b], xf);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      ss = fmaf(xf[i], xf[i], ss);
  }
  ss += __shfl_xor(ss, 16);
  ss += __shfl_xor(ss, 32);

  static_assert(kGated<kMode>, "only the gated-head prologue is implemented");
  static_assert(kKB * 32 == kD, "one wave per head");
  const float rstd = 1.0f / sqrtf(ss / (float)kD + p.eps);

  uint4 ar[kKB];
#pragma unroll
  for (int b = 0; b < kKB; ++b) {
    float xf[8];
    unpack8(xr[b], xf);
    const int c0 = kw0 + 32 * b + 8 * q;
    if constexpr (kGated<kMode>) {
      float gf[8];
      unpack8(gr[b], gf);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float nwv = nwb[b * 8 + i];
        float v = xf[i] * rstd;
        v = v * nwv;
        v = v / (1.0f + __expf(-gf[i]));
        xf[i] = v;
      }
    }
    ar[b] = pack8(xf);  // dead rows: x zeroed above -> 0
  }

  // 3) MFMA
  f4_t acc[kNT];
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    acc[t] = f4_t{0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int b = 0; b < kKB; ++b) {
      acc[t] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(lo4(ar[b]), lo4(wreg[t][b]), acc[t], 0, 0, 0);
      acc[t] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(hi4(ar[b]), hi4(wreg[t][b]), acc[t], 0, 0, 0);
    }
  }

  // 4) sum the waves' partial tiles (wave order) and store
  // D layout: lane holds D[row = 4 q + i][col = r]
#pragma unroll
  for (int t = 0; t < kNT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      red[(wv * kNT + t) * 256 + (4 * q + i) * 16 + r] = acc[t][i];
  __syncthreads();
  for (int e = threadIdx.x; e < kNT * 256; e += kWaves * 64) {
    const int t = e >> 8;
    const int row = (e >> 4) & 15;
    const int col = e & 15;
    if (row < p.M) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < kWaves; ++i)
        s += red[(i * kNT + t) * 256 + (e & 255)];
      p.y[row * p.sy + nblk + t * 16 + col] = (uint16_t)(pack2(s, 0.f) & 0xFFFFu);
    }
  }
}

template <int kMode, int kWaves, int kNT, int kKB>
struct NormGemm {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView g,
      const tvm::ffi::TensorView nw,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView y,
      int64_t flags,  // bit0: g valid, bit2: nw valid (fp32)
      double eps) {
    using namespace host;
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = w.size(0);
    RuntimeCheck(M >= 1 && M <= 16, "norm_gemm: 1 <= M <= 16");
    RuntimeCheck(w.size(1) == K && K == (int64_t)kWaves * kKB * 32, "norm_gemm: K mismatch");
    RuntimeCheck(N % (16 * kNT) == 0, "norm_gemm: N tile");
    RuntimeCheck(x.stride(1) == 1 && w.stride(1) == 1 && w.stride(0) == K && y.stride(1) == 1, "norm_gemm: layout");
    RuntimeCheck(x.stride(0) % 8 == 0, "norm_gemm: x row alignment");
    Params p{};
    p.x = static_cast<const uint16_t*>(x.data_ptr());
    p.sx = x.stride(0);
    if (flags & 1) {
      RuntimeCheck(g.stride(1) == 1 && g.stride(0) % 8 == 0, "norm_gemm: gate layout");
      p.g = static_cast<const uint16_t*>(g.data_ptr());
      p.sg = g.stride(0);
    }
    if (flags & 4) p.nwf = static_cast<const float*>(nw.data_ptr());
    p.w = static_cast<const uint16_t*>(w.data_ptr());
    p.y = static_cast<uint16_t*>(y.data_ptr());
    p.sy = y.stride(0);
    p.M = static_cast<int32_t>(M);
    p.K = static_cast<int32_t>(K);
    p.eps = static_cast<float>(eps);
    LaunchKernel(static_cast<uint32_t>(N / (16 * kNT)), kWaves * 64, x.device())(
        norm_gemm_kernel<kMode, kWaves, kNT, kKB>, p);
  }
};

}  // namespace k3_norm_gemm
}  // namespace sglang
