"""Expert-grouped MoE for M=1..4 tokens with hand-written CUDA mma.sync (W8A16, fp8 -> f16 exact, f32 accumulate).

3 launches: route (Triton 1-warp top-8 table, coral_kernels._moe_route_tab_kernel) -> grouped up -> grouped down.
Grid = one CTA per (token slot, 16-row tile); a slot whose expert an earlier token already picked exits at once, and
the first picker applies the weight tile to every token that chose that expert: tokens are the mma N dimension
(8 wide; non-member tokens are zero columns), so each distinct expert's weights are read once per step regardless of
how many of the M tokens chose it. The shared expert (slot TOPK) is owned by token 0 for all tokens.
Same register-direct fragment layout as coral_mma.py (K permuted inside each 64-chunk to match 16-byte loads).
  up:   W13[e] rows (gate tile r, up tile I + r) x x[tokens] over K = H -> act f16 [M*NS, I] = silu(g) * u
  down: W2[e] rows (16 of H) x act[token slot] over K = I, * routing weight -> per-(token, row tile) workspace;
        the last of the token's NS slot CTAs to arrive sums the NS partials and writes y bf16."""
import os
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

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
// one 64-wide chunk: 4 mma steps for a 16-row tile (wa = row g, wb = row g+8) against B fragments X[8]
__device__ __forceinline__ void chunk_mma(float* c, const uint4& wa, const uint4& wb, const uint32_t* X) {
  const uint32_t A[4] = {wa.x, wa.y, wa.z, wa.w};
  const uint32_t B[4] = {wb.x, wb.y, wb.z, wb.w};
#pragma unroll
  for (int st = 0; st < 4; ++st)
    mma16816(c, e4m3x2_to_f16x2(A[st] & 0xffff), e4m3x2_to_f16x2(B[st] & 0xffff),
             e4m3x2_to_f16x2(A[st] >> 16), e4m3x2_to_f16x2(B[st] >> 16), X[2 * st], X[2 * st + 1]);
}

// Membership via one warp ballot: lane l holds table entries l and 32+l (M*NS <= 36). The owner of an expert is
// its lowest matching slot (token tables are duplicate-free, so that is the smallest token that picked it).
// Returns active; mem bit n = token n uses the expert; jpos[n] = its slot in token n's table; e = expert (-1 shared).
template <int TOPK>
__device__ __forceinline__ bool members(const int* __restrict__ eid, int p, int M, unsigned& mem, int* jpos, int& e) {
  constexpr int NS = TOPK + 1;
  const int lane = threadIdx.x & 31, tot = M * NS;
  const int m = p / NS, j = p % NS;
  mem = 0;
  if (j == TOPK) {
#pragma unroll
    for (int n = 0; n < 4; ++n) { jpos[n] = TOPK; if (n < M) mem |= 1u << n; }
    e = -1;
    return m == 0;
  }
  const int v0 = lane < tot ? __ldg(eid + lane) : -2;
  const int v1 = 32 + lane < tot ? __ldg(eid + 32 + lane) : -2;
  e = __shfl_sync(0xffffffffu, p < 32 ? v0 : v1, p & 31);
  const unsigned b0 = __ballot_sync(0xffffffffu, v0 == e && (lane % NS) != TOPK);
  const unsigned b1 = __ballot_sync(0xffffffffu, v1 == e && ((32 + lane) % NS) != TOPK);
  const int first = b0 ? __ffs(b0) - 1 : 32 + __ffs(b1) - 1;
  unsigned long long bits = (unsigned long long)b0 | ((unsigned long long)b1 << 32);
#pragma unroll
  for (int n = 0; n < 4; ++n) jpos[n] = 0;
  while (bits) {
    const int q = __ffsll(bits) - 1; bits &= bits - 1;
    mem |= 1u << (q / NS); jpos[q / NS] = q % NS;
  }
  return first == p;
}

// B fragments for one 64-chunk straight from a global f16 row (lane needs 16 halfs at kl..kl+15)
__device__ __forceinline__ void load_b(uint32_t* X, const __half* row, int kl, bool live) {
  uint4 xa = make_uint4(0, 0, 0, 0), xb = make_uint4(0, 0, 0, 0);
  if (live) { xa = *reinterpret_cast<const uint4*>(row + kl); xb = *reinterpret_cast<const uint4*>(row + kl + 8); }
  X[0] = xa.x; X[1] = xa.y; X[2] = xa.z; X[3] = xa.w; X[4] = xb.x; X[5] = xb.y; X[6] = xb.z; X[7] = xb.w;
}

template <int H, int I, int TOPK, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) moe_up_mma(const __half* __restrict__ x16, const int* __restrict__ eid,
    const uint8_t* __restrict__ w13, const float* __restrict__ s13, const uint8_t* __restrict__ sh13, const float* __restrict__ shs13,
    __half* __restrict__ act, int M) {
  constexpr int NS = TOPK + 1, KW = H / WARPS, NC = KW / 64, SROW = H / 128;
  __shared__ float red[WARPS][2][32][4];
  unsigned mem; int jpos[4]; int e;
  if (!members<TOPK>(eid, blockIdx.x, M, mem, jpos, e)) return;
  const uint8_t* wbase = e >= 0 ? w13 + (size_t)e * (2 * I * H) : sh13;
  const float* sbase = e >= 0 ? s13 + (size_t)e * ((2 * I / 128) * SROW) : shs13;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.y * 16;
  uint4 ga[NC], gb[NC], ua[NC], ub[NC];
  {
    const uint8_t* pg = wbase + (size_t)(r0 + g) * H + warp * KW + t * 16;
    const uint8_t* pu = pg + (size_t)I * H;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      ga[c] = ld_stream(pg + c * 64); gb[c] = ld_stream(pg + 8 * H + c * 64);
      ua[c] = ld_stream(pu + c * 64); ub[c] = ld_stream(pu + 8 * H + c * 64);
    }
  }
  const bool live = g < M && (mem >> g & 1);
  const __half* xrow = x16 + (size_t)(live ? g : 0) * H;
  float accg[4] = {0.f, 0.f, 0.f, 0.f}, accu[4] = {0.f, 0.f, 0.f, 0.f};
  const float* srg = sbase + (size_t)(r0 / 128) * SROW;
  const float* sru = sbase + (size_t)((I + r0) / 128) * SROW;
#pragma unroll
  for (int grp = 0; grp < KW / 128; ++grp) {
    float cg[4] = {0.f, 0.f, 0.f, 0.f}, cu[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int ch = grp * 2 + cc;
      uint32_t X[8];
      load_b(X, xrow, warp * KW + ch * 64 + t * 16, live);
      chunk_mma(cg, ga[ch], gb[ch], X);
      chunk_mma(cu, ua[ch], ub[ch], X);
    }
    const int kg = (warp * KW) / 128 + grp;
    const float sg = __ldg(srg + kg), su = __ldg(sru + kg);
#pragma unroll
    for (int i = 0; i < 4; ++i) { accg[i] += cg[i] * sg; accu[i] += cu[i] * su; }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) { red[warp][0][lane][i] = accg[i]; red[warp][1][lane][i] = accu[i]; }
  __syncthreads();
  if (warp == 0) {
    float vg[4] = {0.f, 0.f, 0.f, 0.f}, vu[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < WARPS; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) { vg[i] += red[q][0][lane][i]; vu[i] += red[q][1][lane][i]; }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = t * 2 + (i & 1), row = r0 + g + (i >> 1) * 8;
      if (n < M && (mem >> n & 1)) {
        const float a = vg[i] / (1.f + __expf(-vg[i])) * vu[i];
        act[(size_t)(n * NS + jpos[n]) * I + row] = __float2half_rn(a);
      }
    }
  }
}

template <int H, int I, int TOPK, int WARPS, int RT>
__global__ void __launch_bounds__(WARPS * 32) moe_down_mma(const __half* __restrict__ act, const int* __restrict__ eid,
    const float* __restrict__ wt, const uint8_t* __restrict__ w2, const float* __restrict__ s2, const uint8_t* __restrict__ sh2,
    const float* __restrict__ shs2, float* __restrict__ ws, int* __restrict__ cnt, __nv_bfloat16* __restrict__ y, int M) {
  constexpr int NS = TOPK + 1, KW = I / WARPS, NC = KW / 64, SROW = I / 128, RB = 16 * RT, NBLK = H / RB;
  __shared__ float red[WARPS][RT][32][4];
  __shared__ int last[4];
  unsigned mem; int jpos[4]; int e;
  if (!members<TOPK>(eid, blockIdx.x, M, mem, jpos, e)) return;
  const uint8_t* wbase = e >= 0 ? w2 + (size_t)e * (H * I) : sh2;
  const float* sbase = e >= 0 ? s2 + (size_t)e * ((H / 128) * SROW) : shs2;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int blk = blockIdx.y, r0 = blk * RB;
  uint4 wa[RT][NC], wb[RT][NC];
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const uint8_t* pa = wbase + (size_t)(r0 + r * 16 + g) * I + warp * KW + t * 16;
#pragma unroll
    for (int c = 0; c < NC; ++c) { wa[r][c] = ld_stream(pa + c * 64); wb[r][c] = ld_stream(pa + 8 * I + c * 64); }
  }
  const bool live = g < M && (mem >> g & 1);
  const __half* arow = act + (size_t)(live ? g * NS + jpos[g] : 0) * I;
  float acc[RT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[r][i] = 0.f;
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
      uint32_t X[8];
      load_b(X, arow, warp * KW + ch * 64 + t * 16, live);
#pragma unroll
      for (int r = 0; r < RT; ++r) chunk_mma(c[r], wa[r][ch], wb[r][ch], X);
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const float sc = __ldg(sbase + (size_t)((r0 + r * 16) / 128) * SROW + (warp * KW) / 128 + grp);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][i] += c[r][i] * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][r][lane][i] = acc[r][i];
  __syncthreads();
  for (int r = warp; r < RT; r += WARPS) {   // each row tile finalized by one warp (RT may exceed WARPS)
    float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < WARPS; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] += red[q][r][lane][i];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = t * 2 + (i & 1), row = r * 16 + g + (i >> 1) * 8;
      if (n < M && (mem >> n & 1))
        ws[((size_t)(n * NBLK + blk) * NS + jpos[n]) * RB + row] = v[i] * __ldg(wt + n * NS + jpos[n]);
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x < 4) {
    const int n = threadIdx.x;
    last[n] = (n < M && (mem >> n & 1)) ? (atomicAdd(cnt + n * NBLK + blk, 1) == NS - 1) : 0;
  }
  __syncthreads();
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    if (!last[n]) continue;
    __threadfence();
    for (int row = threadIdx.x; row < RB; row += WARPS * 32) {
      float tot = 0.f;
      const volatile float* p = ws + (size_t)(n * NBLK + blk) * NS * RB + row;
#pragma unroll
      for (int q = 0; q < NS; ++q) tot += p[q * RB];
      y[(size_t)n * H + r0 + row] = __float2bfloat16(tot);
    }
    if (threadIdx.x == 0) cnt[n * NBLK + blk] = 0;
  }
}

// M=1 down without any cross-CTA reduction: CTA = 16 output rows; warp w = expert slot w (NS warps, all loading
// their full-K weight tile up front), mma (token 0 = column 0), * routing weight, summed across warps in smem.
template <int H, int I, int TOPK, bool PDL>
__global__ void __launch_bounds__((TOPK + 1) * 32) moe_down_m1(const float* __restrict__ act, const int* __restrict__ eid,
    const float* __restrict__ wt, const uint8_t* __restrict__ w2, const float* __restrict__ s2, const uint8_t* __restrict__ sh2,
    const float* __restrict__ shs2, __nv_bfloat16* __restrict__ y) {
  constexpr int NS = TOPK + 1, NC = I / 64, SROW = I / 128;
  __shared__ float red[NS][16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const int e = warp < TOPK ? __ldg(eid + warp) : -1;
  const uint8_t* wbase = e >= 0 ? w2 + (size_t)e * (H * I) : sh2;
  const float* sbase = e >= 0 ? s2 + (size_t)e * ((H / 128) * SROW) : shs2;
  const uint8_t* pa = wbase + (size_t)(r0 + g) * I + t * 16;
  uint4 wa[NC], wb[NC];
#pragma unroll
  for (int c = 0; c < NC; ++c) { wa[c] = ld_stream(pa + c * 64); wb[c] = ld_stream(pa + 8 * I + c * 64); }
  // PDL: the weight tile (depends only on the routing table, finished before the up kernel started) is in flight;
  // now wait for the up kernel's act before reading it.
  if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  const float* arow = act + (size_t)warp * I + t * 16;
  const float* sr = sbase + (size_t)(r0 / 128) * SROW;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int grp = 0; grp < I / 128; ++grp) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int ch = grp * 2 + cc;
      uint32_t X[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (g == 0) {
        const float4* q = reinterpret_cast<const float4*>(arow + ch * 64);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float4 v = q[j];
          __half2 h0 = __floats2half2_rn(v.x, v.y), h1 = __floats2half2_rn(v.z, v.w);
          X[2 * j] = *reinterpret_cast<uint32_t*>(&h0); X[2 * j + 1] = *reinterpret_cast<uint32_t*>(&h1);
        }
      }
      chunk_mma(c, wa[ch], wb[ch], X);
    }
    const float sc = __ldg(sr + grp);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += c[i] * sc;
  }
  if (t == 0) { const float w = __ldg(wt + warp); red[warp][g] = acc[0] * w; red[warp][g + 8] = acc[2] * w; }
  __syncthreads();
  if (threadIdx.x < 16) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NS; ++q) v += red[q][threadIdx.x];
    y[r0 + threadIdx.x] = __float2bfloat16(v);
  }
}

void moe_down1(torch::Tensor act, torch::Tensor eid, torch::Tensor wt, torch::Tensor w2, torch::Tensor s2, torch::Tensor sh2,
               torch::Tensor shs2, torch::Tensor y, bool pdl) {
  constexpr int H = 2048, I = 512, TOPK = 8;
  auto st = at::cuda::getCurrentCUDAStream();
  const float* a = act.data_ptr<float>(); const int* e = eid.data_ptr<int>(); const float* w = wt.data_ptr<float>();
  const uint8_t* W2 = reinterpret_cast<const uint8_t*>(w2.data_ptr()); const float* S2 = s2.data_ptr<float>();
  const uint8_t* SH2 = reinterpret_cast<const uint8_t*>(sh2.data_ptr()); const float* SHS2 = shs2.data_ptr<float>();
  __nv_bfloat16* Y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  if (!pdl) {
    moe_down_m1<H, I, TOPK, false><<<H / 16, (TOPK + 1) * 32, 0, st>>>(a, e, w, W2, S2, SH2, SHS2, Y);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(H / 16); cfg.blockDim = dim3((TOPK + 1) * 32); cfg.dynamicSmemBytes = 0; cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_down_m1<H, I, TOPK, true>, a, e, w, W2, S2, SH2, SHS2, Y));
}

void moe_mma(torch::Tensor x16, torch::Tensor eid, torch::Tensor wt, torch::Tensor w13, torch::Tensor s13, torch::Tensor sh13,
             torch::Tensor shs13, torch::Tensor w2, torch::Tensor s2, torch::Tensor sh2, torch::Tensor shs2, torch::Tensor act,
             torch::Tensor ws, torch::Tensor cnt, torch::Tensor y, int64_t up_warps, int64_t dn_warps, int64_t dn_rt) {
  constexpr int H = 2048, I = 512, TOPK = 8, NS = TOPK + 1;
  const int M = x16.size(0);
  TORCH_CHECK(M >= 1 && M <= 4 && x16.size(1) == H && w13.size(1) == 2 * I && w2.size(1) == H, "bad shape");
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __half*>(x16.data_ptr());
  auto ap = reinterpret_cast<__half*>(act.data_ptr());
  auto u8 = [](torch::Tensor& t) { return reinterpret_cast<const uint8_t*>(t.data_ptr()); };
  const dim3 gu(M * NS, I / 16);
#define UP(W) moe_up_mma<H, I, TOPK, W><<<gu, W * 32, 0, st>>>(xp, eid.data_ptr<int>(), u8(w13), s13.data_ptr<float>(), u8(sh13), shs13.data_ptr<float>(), ap, M)
#define DN(W, R) if (dn_warps == W && dn_rt == R) { moe_down_mma<H, I, TOPK, W, R><<<dim3(M * NS, H / (16 * R)), W * 32, 0, st>>>(ap, eid.data_ptr<int>(), wt.data_ptr<float>(), u8(w2), s2.data_ptr<float>(), u8(sh2), shs2.data_ptr<float>(), ws.data_ptr<float>(), cnt.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M); return; }
  if (up_warps == 4) UP(4); else if (up_warps == 8) UP(8); else if (up_warps == 16) UP(16); else TORCH_CHECK(false, "up_warps");
  DN(1, 1) DN(1, 2) DN(1, 4) DN(2, 1) DN(2, 2) DN(2, 4) DN(4, 1) DN(4, 2) DN(4, 4)
  TORCH_CHECK(false, "dn cfg");
}
"""
_CPP = ("void moe_down1(torch::Tensor act, torch::Tensor eid, torch::Tensor wt, torch::Tensor w2, torch::Tensor s2, "
        "torch::Tensor sh2, torch::Tensor shs2, torch::Tensor y, bool pdl);"
        "void moe_mma(torch::Tensor x16, torch::Tensor eid, torch::Tensor wt, torch::Tensor w13, torch::Tensor s13, "
        "torch::Tensor sh13, torch::Tensor shs13, torch::Tensor w2, torch::Tensor s2, torch::Tensor sh2, torch::Tensor shs2, "
        "torch::Tensor act, torch::Tensor ws, torch::Tensor cnt, torch::Tensor y, int64_t up_warps, int64_t dn_warps, int64_t dn_rt);")
_ext = None
_STATE = {}


def _load():
    global _ext
    if _ext is None:
        d = os.environ.get("CORAL_EXT_DIR_MOE", "/data/ext_build/coral_moe_mma")
        os.makedirs(d, exist_ok=True)
        _ext = load_inline("coral_moe_mma", _CPP, cuda_sources=_SRC, functions=["moe_mma", "moe_down1"], build_directory=d,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"])
    return _ext


def coral_moe_decode_mma(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                         up_warps=4, dn_warps=4, dn_rt=2):
    """route (Triton, 1 launch) + grouped up (CUDA) + grouped down (CUDA) = 3 launches. M = 1..4."""
    from coral_kernels import _moe_route_tab16_kernel
    M, H = x.shape
    E = w13.shape[0]
    I = w13.shape[1] // 2
    NS = top_k + 1
    assert top_k == 8 and H == 2048 and I == 512 and 1 <= M <= 4
    dev = x.device
    cnt = _STATE.get(dev)
    if cnt is None:
        cnt = _STATE[dev] = torch.zeros(4 * (H // 16), dtype=torch.int32, device=dev)  # >= 4 * H / (16 * dn_rt)
    eid = torch.empty(M * NS, dtype=torch.int32, device=dev)
    wt = torch.empty(M * NS, dtype=torch.float32, device=dev)
    act = torch.empty(M * NS, I, dtype=torch.float16, device=dev)
    ws = torch.empty(M * (H // 16) * NS * 16, dtype=torch.float32, device=dev)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    x16 = torch.empty(M, H, dtype=torch.float16, device=dev)
    _moe_route_tab16_kernel[(M,)](x, logits, shared_gate_w, eid, wt, x16, H=H, E=E, TOPK=top_k, RENORM=renorm, num_warps=1)
    _load().moe_mma(x16, eid, wt, w13, s13, sh13, shs13, w2, s2, sh2, shs2, act, ws, cnt, y, up_warps, dn_warps, dn_rt)
    return y


def coral_moe_decode_v11(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k=8, renorm=True,
                         max_m=4, cfg=None, pdl=True):
    """M=1: route (Triton) + v9 no-split up (Triton) + CUDA expert-loop down (no cross-CTA reduction) = 3 launches.
    M>1 -> v9."""
    import coral_kernels as ck
    M, H = x.shape
    if M != 1:
        return ck.coral_moe_decode_v9(x, logits, shared_gate_w, w13, s13, sh13, shs13, w2, s2, sh2, shs2, top_k, renorm,
                                      max_m, cfg)
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    bu, kb, wu = (cfg or ck._V9_CFG[1])[:3]
    eid = torch.empty(NS, dtype=torch.int32, device=x.device)
    wt = torch.empty(NS, dtype=torch.float32, device=x.device)
    act = torch.empty(NS, I, dtype=torch.float32, device=x.device)
    y = torch.empty(1, H, dtype=torch.bfloat16, device=x.device)
    ck._moe_route_tab_kernel[(1,)](x, logits, shared_gate_w, eid, wt, H=H, E=E, TOPK=top_k, RENORM=renorm, num_warps=1)
    up = ck._moe_up_ns_pdl_kernel if pdl else ck._moe_up_ns_kernel
    up[(NS, I // bu)](x, eid, w13, s13, sh13, shs13, act, H=H, I=I, TOPK=top_k, BN=bu, KB=kb, num_warps=wu)
    _load().moe_down1(act, eid, wt, w2, s2, sh2, shs2, y, pdl)
    return y
