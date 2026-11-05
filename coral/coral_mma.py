"""W8A16 tensor-core GEMV for M=1..4 tokens (MTP verify): hand-written CUDA, mma.sync m16n8k16 f16 -> f32.

One CTA = 16 weight rows over the full K (no cross-CTA split-K); its WARPS warps split K and reduce through shared
memory. Each lane loads 16 contiguous fp8 bytes of rows g and g+8 (g = lane/4) per 64-wide chunk with one 16-byte
load each, so the weights go global -> registers with no shared-memory staging. Inside a chunk the K order is
permuted to match those registers (step s, byte j -> mma column t*2 + (j&1) + 8*(j>>1)); the activations use the same
permutation, so the dot product is unchanged. fp8 e4m3 -> f16 is exact (cvt.rn.f16x2.e4m3x2). Activations are staged
once per CTA in shared memory as f16 (inputs to these linears are normalized, well inside f16 range). Block scales
(128x128) are applied per 128-wide K group in fp32. Tokens are the mma N dimension (8 wide, rows >= M are zero)."""
import os
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t v16) {
  uint32_t r; unsigned short h = (unsigned short)v16;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
  return r;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

template <int KW, int WARPS, int RT>
__global__ void __launch_bounds__(WARPS * 32) w8a16_mma_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                                                               const float* __restrict__ s, __nv_bfloat16* __restrict__ y, int M, int N) {
  // RT row tiles of 16 per CTA: the activation staging (M*K*2 bytes per CTA) is amortized over 16*RT rows.
  constexpr int K = KW * WARPS;
  constexpr int NC = KW / 64;
  constexpr int SROW = K / 128;
  constexpr int XS = K + 8;                       // padded f16 row stride (bank spread)
  constexpr int XB = 4 * XS * 2, RB = WARPS * RT * 32 * 4 * 4;
  __shared__ __align__(16) unsigned char smem[XB > RB ? XB : RB];   // x staging, reused for the K reduction
  __half* xs = reinterpret_cast<__half*>(smem);
  float* red = reinterpret_cast<float*>(smem);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16 * RT;
  uint4 wa[RT][NC], wb[RT][NC];
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const uint8_t* pa = w + (size_t)(r0 + r * 16 + g) * K + warp * KW + t * 16;
    const uint8_t* pb = pa + (size_t)8 * K;
#pragma unroll
    for (int c = 0; c < NC; ++c) { wa[r][c] = ld_stream(pa + c * 64); wb[r][c] = ld_stream(pb + c * 64); }
  }
  for (int i = threadIdx.x; i < M * (K / 8); i += WARPS * 32) {
    const int m = i / (K / 8), k8 = (i % (K / 8)) * 8;
    uint4 v = *reinterpret_cast<const uint4*>(x + (size_t)m * K + k8);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint4 o; __half2* h = reinterpret_cast<__half2*>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = __float22half2_rn(__bfloat1622float2(b[j]));
    *reinterpret_cast<uint4*>(xs + m * XS + k8) = o;
  }
  __syncthreads();
  float acc[RT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[r][i] = 0.f;
  const bool live = g < M;
#pragma unroll
  for (int grp = 0; grp < KW / 128; ++grp) {
    float c[RT][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[r][i] = 0.f;
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int ch = grp * 2 + cc;
      const int kl = warp * KW + ch * 64 + t * 16;
      uint4 xa = make_uint4(0, 0, 0, 0), xb = make_uint4(0, 0, 0, 0);
      if (live) { xa = *reinterpret_cast<const uint4*>(xs + g * XS + kl); xb = *reinterpret_cast<const uint4*>(xs + g * XS + kl + 8); }
      const uint32_t X[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const uint32_t A[4] = {wa[r][ch].x, wa[r][ch].y, wa[r][ch].z, wa[r][ch].w};
        const uint32_t B[4] = {wb[r][ch].x, wb[r][ch].y, wb[r][ch].z, wb[r][ch].w};
#pragma unroll
        for (int st = 0; st < 4; ++st)
          mma16816(c[r], e4m3x2_to_f16x2(A[st] & 0xffff), e4m3x2_to_f16x2(B[st] & 0xffff),
                   e4m3x2_to_f16x2(A[st] >> 16), e4m3x2_to_f16x2(B[st] >> 16), X[2 * st], X[2 * st + 1]);
      }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const float sc = __ldg(s + (size_t)((r0 + r * 16) / 128) * SROW + (warp * KW) / 128 + grp);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][i] += c[r][i] * sc;
    }
  }
  __syncthreads();                                // xs no longer needed: reuse it for the reduction
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[((warp * RT + r) * 32 + lane) * 4 + i] = acc[r][i];
  __syncthreads();
  // reduce: warp q (< RT) finalizes row tile q
  if (warp < RT) {
    const int r = warp;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < WARPS; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] += red[((q * RT + r) * 32 + lane) * 4 + i];
    const int n0 = t * 2, n1 = t * 2 + 1, row = r0 + r * 16 + g;
    if (n0 < M) { y[(size_t)n0 * N + row] = __float2bfloat16(v[0]); y[(size_t)n0 * N + row + 8] = __float2bfloat16(v[2]); }
    if (n1 < M) { y[(size_t)n1 * N + row] = __float2bfloat16(v[1]); y[(size_t)n1 * N + row + 8] = __float2bfloat16(v[3]); }
  }
}

// Direct variant: no shared-memory x staging (M <= 8). Each live lane (g < M) loads its 16 bf16 activations per
// 64-chunk straight from global (L1/L2-resident, 16 KB) and converts to f16 in registers.
template <int KW, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) w8a16_mma_direct(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                                                               const float* __restrict__ s, __nv_bfloat16* __restrict__ y, int M, int N) {
  constexpr int K = KW * WARPS;
  constexpr int NC = KW / 64;
  constexpr int SROW = K / 128;
  __shared__ float red[WARPS][32][4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const uint8_t* pa = w + (size_t)(r0 + g) * K + warp * KW + t * 16;
  uint4 wa[NC], wb[NC];
#pragma unroll
  for (int c = 0; c < NC; ++c) { wa[c] = ld_stream(pa + c * 64); wb[c] = ld_stream(pa + 8 * (size_t)K + c * 64); }
  const bool live = g < M;
  const __nv_bfloat16* xr = x + (size_t)(live ? g : 0) * K + warp * KW + t * 16;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const float* srow = s + (size_t)(r0 / 128) * SROW;
#pragma unroll
  for (int grp = 0; grp < KW / 128; ++grp) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int ch = grp * 2 + cc;
      uint32_t X[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (live) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(xr + ch * 64), v1 = *reinterpret_cast<const uint4*>(xr + ch * 64 + 8);
        const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          __half2 h0 = __float22half2_rn(__bfloat1622float2(b0[j])), h1 = __float22half2_rn(__bfloat1622float2(b1[j]));
          X[j] = *reinterpret_cast<uint32_t*>(&h0); X[4 + j] = *reinterpret_cast<uint32_t*>(&h1);
        }
      }
      const uint32_t A[4] = {wa[ch].x, wa[ch].y, wa[ch].z, wa[ch].w};
      const uint32_t B[4] = {wb[ch].x, wb[ch].y, wb[ch].z, wb[ch].w};
#pragma unroll
      for (int st = 0; st < 4; ++st)
        mma16816(c, e4m3x2_to_f16x2(A[st] & 0xffff), e4m3x2_to_f16x2(B[st] & 0xffff),
                 e4m3x2_to_f16x2(A[st] >> 16), e4m3x2_to_f16x2(B[st] >> 16), X[2 * st], X[2 * st + 1]);
    }
    const float sc = __ldg(srow + (warp * KW) / 128 + grp);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += c[i] * sc;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
  __syncthreads();
  if (warp == 0) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < WARPS; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] += red[q][lane][i];
    const int n0 = t * 2, n1 = t * 2 + 1;
    if (n0 < M) { y[(size_t)n0 * N + r0 + g] = __float2bfloat16(v[0]); y[(size_t)n0 * N + r0 + g + 8] = __float2bfloat16(v[2]); }
    if (n1 < M) { y[(size_t)n1 * N + r0 + g] = __float2bfloat16(v[1]); y[(size_t)n1 * N + r0 + g + 8] = __float2bfloat16(v[3]); }
  }
}

// Direct variant for 9..16 tokens (DFlash2 block-16 verify, or 2 x block-8): two mma N tiles (tokens 0-7, 8-15) share
// each weight fragment, so the weights are still read once. Same math per token as w8a16_mma_direct.
template <int KW, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) w8a16_mma_direct16(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                                                                 const float* __restrict__ s, __nv_bfloat16* __restrict__ y, int M, int N) {
  constexpr int K = KW * WARPS;
  constexpr int NC = KW / 64;
  constexpr int SROW = K / 128;
  __shared__ float red[WARPS][32][8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const uint8_t* pa = w + (size_t)(r0 + g) * K + warp * KW + t * 16;
  uint4 wa[NC], wb[NC];
#pragma unroll
  for (int c = 0; c < NC; ++c) { wa[c] = ld_stream(pa + c * 64); wb[c] = ld_stream(pa + 8 * (size_t)K + c * 64); }
  const bool live0 = g < M, live1 = g + 8 < M;
  const __nv_bfloat16* xr0 = x + (size_t)(live0 ? g : 0) * K + warp * KW + t * 16;
  const __nv_bfloat16* xr1 = x + (size_t)(live1 ? g + 8 : 0) * K + warp * KW + t * 16;
  float acc0[4] = {0.f, 0.f, 0.f, 0.f}, acc1[4] = {0.f, 0.f, 0.f, 0.f};
  const float* srow = s + (size_t)(r0 / 128) * SROW;
#pragma unroll
  for (int grp = 0; grp < KW / 128; ++grp) {
    float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int ch = grp * 2 + cc;
      uint32_t X0[8] = {0, 0, 0, 0, 0, 0, 0, 0}, X1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (live0) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(xr0 + ch * 64), v1 = *reinterpret_cast<const uint4*>(xr0 + ch * 64 + 8);
        const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          __half2 h0 = __float22half2_rn(__bfloat1622float2(b0[j])), h1 = __float22half2_rn(__bfloat1622float2(b1[j]));
          X0[j] = *reinterpret_cast<uint32_t*>(&h0); X0[4 + j] = *reinterpret_cast<uint32_t*>(&h1);
        }
      }
      if (live1) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(xr1 + ch * 64), v1 = *reinterpret_cast<const uint4*>(xr1 + ch * 64 + 8);
        const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          __half2 h0 = __float22half2_rn(__bfloat1622float2(b0[j])), h1 = __float22half2_rn(__bfloat1622float2(b1[j]));
          X1[j] = *reinterpret_cast<uint32_t*>(&h0); X1[4 + j] = *reinterpret_cast<uint32_t*>(&h1);
        }
      }
      const uint32_t A[4] = {wa[ch].x, wa[ch].y, wa[ch].z, wa[ch].w};
      const uint32_t B[4] = {wb[ch].x, wb[ch].y, wb[ch].z, wb[ch].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        const uint32_t a0 = e4m3x2_to_f16x2(A[st] & 0xffff), a1 = e4m3x2_to_f16x2(B[st] & 0xffff);
        const uint32_t a2 = e4m3x2_to_f16x2(A[st] >> 16), a3 = e4m3x2_to_f16x2(B[st] >> 16);
        mma16816(c0, a0, a1, a2, a3, X0[2 * st], X0[2 * st + 1]);
        mma16816(c1, a0, a1, a2, a3, X1[2 * st], X1[2 * st + 1]);
      }
    }
    const float sc = __ldg(srow + (warp * KW) / 128 + grp);
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc0[i] += c0[i] * sc; acc1[i] += c1[i] * sc; }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) { red[warp][lane][i] = acc0[i]; red[warp][lane][4 + i] = acc1[i]; }
  __syncthreads();
  if (warp == 0) {
    float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < WARPS; ++q)
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] += red[q][lane][i];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n0 = h * 8 + t * 2, n1 = n0 + 1;
      const float* u = v + 4 * h;
      if (n0 < M) { y[(size_t)n0 * N + r0 + g] = __float2bfloat16(u[0]); y[(size_t)n0 * N + r0 + g + 8] = __float2bfloat16(u[2]); }
      if (n1 < M) { y[(size_t)n1 * N + r0 + g] = __float2bfloat16(u[1]); y[(size_t)n1 * N + r0 + g + 8] = __float2bfloat16(u[3]); }
    }
  }
}

#define LAUNCH(KW_, W_, RT_) if (kw == KW_ && warps == W_ && rt == RT_) { \
  w8a16_mma_kernel<KW_, W_, RT_><<<N / (16 * RT_), W_ * 32, 0, stream>>>( \
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint8_t*>(w.data_ptr()), \
    s.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N); return y; }
#define LAUNCH_RT(KW_, W_) LAUNCH(KW_, W_, 1) LAUNCH(KW_, W_, 2) LAUNCH(KW_, W_, 4)

torch::Tensor w8a16_mma(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor y, int64_t warps, int64_t rt) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto stream0 = at::cuda::getCurrentCUDAStream();
  if (rt == 0) {   // direct variant, M <= 8
    TORCH_CHECK(M >= 1 && M <= 8 && N % 16 == 0 && K % (warps * 128) == 0, "bad shape");
    const int kwd = K / warps;
#define LAUNCHD(KW_, W_) if (kwd == KW_ && warps == W_) { w8a16_mma_direct<KW_, W_><<<N / 16, W_ * 32, 0, stream0>>>( \
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint8_t*>(w.data_ptr()), \
    s.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N); return y; }
    LAUNCHD(128, 4) LAUNCHD(256, 4) LAUNCHD(512, 4) LAUNCHD(1024, 4) LAUNCHD(128, 8) LAUNCHD(256, 8) LAUNCHD(512, 8)
    LAUNCHD(128, 16) LAUNCHD(256, 16) LAUNCHD(256, 2) LAUNCHD(512, 2) LAUNCHD(1024, 2)
    TORCH_CHECK(false, "no direct instantiation");
  }
  if (rt == -1) {  // direct16 variant, M <= 16
    TORCH_CHECK(M >= 1 && M <= 16 && N % 16 == 0 && K % (warps * 128) == 0, "bad shape");
    const int kwd = K / warps;
#define LAUNCHD16(KW_, W_) if (kwd == KW_ && warps == W_) { w8a16_mma_direct16<KW_, W_><<<N / 16, W_ * 32, 0, stream0>>>( \
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint8_t*>(w.data_ptr()), \
    s.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N); return y; }
    LAUNCHD16(128, 4) LAUNCHD16(256, 4) LAUNCHD16(512, 4) LAUNCHD16(128, 8) LAUNCHD16(256, 8) LAUNCHD16(512, 8)
    LAUNCHD16(128, 16) LAUNCHD16(256, 16)
    TORCH_CHECK(false, "no direct16 instantiation");
  }
  TORCH_CHECK(M >= 1 && M <= 4 && N % (16 * rt) == 0 && K % (warps * 128) == 0 && rt <= 8 && 128 % (16 * rt) == 0, "bad shape");
  const int kw = K / warps;
  auto stream = at::cuda::getCurrentCUDAStream();
  LAUNCH_RT(128, 4) LAUNCH_RT(256, 4) LAUNCH_RT(512, 4) LAUNCH_RT(1024, 4)
  LAUNCH_RT(128, 8) LAUNCH_RT(256, 8) LAUNCH_RT(512, 8)
  LAUNCH_RT(128, 16) LAUNCH_RT(256, 16)
  LAUNCH_RT(256, 2) LAUNCH_RT(512, 2) LAUNCH_RT(1024, 2)
  TORCH_CHECK(false, "no instantiation for K/warps=", kw, " warps=", warps, " rt=", rt);
  return y;
}
"""
_CPP = "torch::Tensor w8a16_mma(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor y, int64_t warps, int64_t rt);"
_ext = None


def _load():
    global _ext
    if _ext is None:
        d = os.environ.get("CORAL_EXT_DIR", "/data/ext_build/coral_mma16")
        os.makedirs(d, exist_ok=True)
        _ext = load_inline("coral_mma16", _CPP, cuda_sources=_SRC, functions=["w8a16_mma"], build_directory=d,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"],
                           verbose=False)
    return _ext


def w8a16_mma(x, w, s, warps=8, rt=1, out=None):
    """x bf16 [M, K], w fp8 e4m3 [N, K], s fp32 [N/128, K/128] -> bf16 [M, N]. rt >= 1: smem-staged x, M <= 4;
    rt == 0: direct-load variant, M <= 8; rt == -1: direct-load with two token tiles, M <= 16."""
    y = out if out is not None else torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
    return _load().w8a16_mma(x.contiguous(), w, s, y, warps, rt)
