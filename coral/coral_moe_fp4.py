"""CORAL_MOE v9 for NVFP4 W4A16 weights (nvidia/Qwen3.6-35B-A3B-NVFP4): route + no-split up + split-K down, 3 launches.

Same structure as coral_kernels.coral_moe_decode_v9 (its route kernel is reused unchanged; slot TOPK is the shared
expert), with the weight reads changed from FP8 128x128-block to the ModelOpt NVFP4 checkpoint layout:
  w   uint8 [rows, K/2]   two e2m1 per byte, element 2c in the low nibble, 2c+1 in the high nibble
  s   fp8 e4m3 [rows, K/16]  one scale per 16 consecutive K
  g   fp32 per expert (weight_scale_2); W = e2m1 * s * g
w13 stacks gate rows then up rows ([E, 2I, H/2]); the checkpoint's gate and up weight_scale_2 are equal per expert,
but g13 keeps both ([E, 2]) so nothing depends on that.

Why: stock SGLang serves this checkpoint through Marlin W4A16, which reaches 35-47% of the HBM floor at verify M=8
(bench/nvfp4_marlin_moe_bench.py) and so saves almost nothing over FP8 v9 (~54 us at ~35 experts). The 4-bit floor at
~35 experts is ~19 us, so a kernel at v9's efficiency would save ~30 us per layer (HANDOFF.md, NVFP4 section).

Decode: e2m1 magnitude bits (e, m) shifted to fp16 bit 9 are an fp16 with exponent field e and top mantissa bit m, i.e.
the e2m1 value * 2^-14 (e=0 lands on fp16 subnormals, which convert to fp32 exactly); the sign goes to bit 15. The
2^14 is folded into g. The odd/even split means x is read as two strided halves instead of unpacking nibble pairs, and
the block scale multiplies a sum of 8 bytes (16 elements) rather than every element.
"""
import torch
import triton
import triton.language as tl

from coral_kernels import _moe_route_tab_kernel

E2M1_UNSCALE = 16384.0  # 2^14, see module docstring


@triton.jit
def _e2m1(code):
    """int32 nibble (0..15) -> fp32 e2m1 value * 2^-14."""
    h = ((code & 7) << 9) | ((code & 8) << 12)
    return h.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _fp4_dot(wrow_ptr, srow_ptr, xe, xo, rc, rs, KB2: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr,
             WSTRIDE: tl.constexpr, SSTRIDE: tl.constexpr):
    """sum over this K chunk of W[r, :] * x for BN rows; xe/xo = x at even/odd K of the chunk (fp32, [KB/2])."""
    rr = tl.arange(0, BN)
    b = tl.load(wrow_ptr + rr[:, None] * WSTRIDE + rc[None, :]).to(tl.int32)
    t = _e2m1(b & 15) * xe[None, :] + _e2m1(b >> 4) * xo[None, :]
    s = tl.load(srow_ptr + rr[:, None] * SSTRIDE + rs[None, :]).to(tl.float32)
    return tl.sum(tl.sum(tl.reshape(t, (BN, KS, 8)), axis=2) * s, axis=1)


@triton.jit
def _moe_up_fp4_kernel(x_ptr, eid_ptr, w13_ptr, s13_ptr, g13_ptr, sh13_ptr, shs13_ptr, shg13_ptr, act_ptr,
                       H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    NS: tl.constexpr = TOPK + 1
    HB: tl.constexpr = H // 2
    HS: tl.constexpr = H // 16
    KB2: tl.constexpr = KB // 2
    KS: tl.constexpr = KB // 16
    m = pid_s // NS
    j = pid_s % NS
    if j < TOPK:
        eid = tl.load(eid_ptr + pid_s).to(tl.int64)
        wbase = w13_ptr + eid * (2 * I * HB)
        sbase = s13_ptr + eid * (2 * I * HS)
        gg = tl.load(g13_ptr + eid * 2)
        gu = tl.load(g13_ptr + eid * 2 + 1)
    else:
        wbase = sh13_ptr
        sbase = shs13_ptr
        gg = tl.load(shg13_ptr)
        gu = tl.load(shg13_ptr + 1)
    r0 = pid_n * BN
    accg = tl.zeros((BN,), dtype=tl.float32)
    accu = tl.zeros((BN,), dtype=tl.float32)
    for k0 in tl.static_range(0, H, KB):
        rc = k0 // 2 + tl.arange(0, KB2)
        rs = k0 // 16 + tl.arange(0, KS)
        xe = tl.load(x_ptr + m * H + 2 * rc).to(tl.float32)
        xo = tl.load(x_ptr + m * H + 2 * rc + 1).to(tl.float32)
        accg += _fp4_dot(wbase + r0 * HB, sbase + r0 * HS, xe, xo, rc, rs, KB2, KS, BN, HB, HS)
        accu += _fp4_dot(wbase + (r0 + I) * HB, sbase + (r0 + I) * HS, xe, xo, rc, rs, KB2, KS, BN, HB, HS)
    accg = accg * (gg * 16384.0)
    accu = accu * (gu * 16384.0)
    tl.store(act_ptr + pid_s * I + r0 + tl.arange(0, BN), accg / (1.0 + tl.exp(-accg)) * accu)


@triton.jit
def _moe_down_fp4_kernel(act_ptr, eid_ptr, wt_ptr, w2_ptr, s2_ptr, g2_ptr, sh2_ptr, shs2_ptr, shg2_ptr, ws_ptr,
                         cnt_ptr, y_ptr, H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr,
                         SK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    NS: tl.constexpr = TOPK + 1
    NB: tl.constexpr = H // BN
    IB: tl.constexpr = I // 2
    IS: tl.constexpr = I // 16
    KC: tl.constexpr = I // SK
    m = pid_s // NS
    j = pid_s % NS
    if j < TOPK:
        eid = tl.load(eid_ptr + pid_s).to(tl.int64)
        wbase = w2_ptr + eid * (H * IB)
        sbase = s2_ptr + eid * (H * IS)
        g = tl.load(g2_ptr + eid)
    else:
        wbase = sh2_ptr
        sbase = shs2_ptr
        g = tl.load(shg2_ptr)
    r0 = pid_n * BN
    rc = pid_k * (KC // 2) + tl.arange(0, KC // 2)
    rs = pid_k * (KC // 16) + tl.arange(0, KC // 16)
    ae = tl.load(act_ptr + pid_s * I + 2 * rc)
    ao = tl.load(act_ptr + pid_s * I + 2 * rc + 1)
    acc = _fp4_dot(wbase + r0 * IB, sbase + r0 * IS, ae, ao, rc, rs, KC // 2, KC // 16, BN, IB, IS)
    acc = acc * (g * 16384.0 * tl.load(wt_ptr + pid_s))
    rn = r0 + tl.arange(0, BN)
    wsb = ws_ptr + ((m * NB + pid_n) * (NS * SK)) * BN
    tl.store(wsb + (j * SK + pid_k) * BN + tl.arange(0, BN), acc)
    prev = tl.atomic_add(cnt_ptr + m * NB + pid_n, 1, sem="acq_rel")
    if prev == NS * SK - 1:
        tot = tl.zeros((BN,), dtype=tl.float32)
        for q in tl.static_range(NS * SK):
            tot += tl.load(wsb + q * BN + tl.arange(0, BN), cache_modifier=".cg")
        tl.store(y_ptr + m * H + rn, tot.to(tl.bfloat16))
        tl.atomic_xchg(cnt_ptr + m * NB + pid_n, 0)


# per-M (up BN, up KB, up warps, down BN, down SK, down warps). Placeholders until bench/moe_fp4_test.py --sweep.
_V9F_CFG = {m: (16, 512, 4, 64, 1, 4) for m in range(1, 17)}

_STATE = {}


def coral_moe_decode_v9_fp4(x, logits, shared_gate_w, w13, s13, g13, sh13, shs13, shg13, w2, s2, g2, sh2, shs2, shg2,
                            top_k=8, renorm=True, max_m=16, cfg=None):
    """x [M, H] bf16, logits [M, E] bf16. w13 uint8 [E, 2I, H/2], s13 fp8 [E, 2I, H/16], g13 fp32 [E, 2] (gate, up);
    sh13/shs13 the shared expert's [2I, ...] and shg13 fp32 [2]; w2 uint8 [E, H, I/2], s2 fp8 [E, H, I/16], g2 fp32 [E],
    shg2 fp32 [1]. g* are the checkpoint's weight_scale_2 as is. Returns y [M, H] bf16 (routed + gated shared)."""
    M, H = x.shape
    E, I2, _ = w13.shape
    I = I2 // 2
    NS = top_k + 1
    assert M <= max_m and logits.shape == (M, E) and top_k <= 8
    bu, kb, wu, bd, sd, wd = cfg or _V9F_CFG[M]
    nb_dn = H // bd
    key = (x.device, bd, max_m)
    cnt_dn = _STATE.get(key)
    if cnt_dn is None:
        cnt_dn = _STATE[key] = torch.zeros(max_m * nb_dn, dtype=torch.int32, device=x.device)
    eid = torch.empty(M * NS, dtype=torch.int32, device=x.device)
    wt = torch.empty(M * NS, dtype=torch.float32, device=x.device)
    act = torch.empty(M * NS, I, dtype=torch.float32, device=x.device)
    ws_dn = torch.empty(M * nb_dn * NS * sd * bd, dtype=torch.float32, device=x.device)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _moe_route_tab_kernel[(M,)](x, logits, shared_gate_w, eid, wt, H=H, E=E, TOPK=top_k, RENORM=renorm, num_warps=1)
    _moe_up_fp4_kernel[(M * NS, I // bu)](x, eid, w13, s13, g13, sh13, shs13, shg13, act, H=H, I=I, TOPK=top_k,
                                          BN=bu, KB=kb, num_warps=wu)
    _moe_down_fp4_kernel[(M * NS, nb_dn, sd)](act, eid, wt, w2, s2, g2, sh2, shs2, shg2, ws_dn, cnt_dn, y, H=H, I=I,
                                              TOPK=top_k, BN=bd, SK=sd, num_warps=wd)
    return y


# ---------------------------------------------------------------------------------------------------------------
# Expert-grouped CUDA mma version (coral_moe_mma.py's structure, Hitesh's FP8 M<=4 kernel, ported to NVFP4 and M<=16).
# Why: the Triton per-slot port above is ALU-bound on the e2m1 decode (H100 2026-10-16, results/moe_fp4_v0.log: M=8
# 101 us flat in distinct experts, vs Marlin 45.7 routed-only): each of the M*NS slot CTAs decodes its expert again.
# Here one CTA per (distinct expert, 16-row tile) decodes the tile once and applies it to every member token with
# mma.sync m16n8k16 (tokens = N; 2 N tiles for M 9..16).
# Decode (Marlin's fe2m1f trick): nibble k of each 16-bit half shifted to bits 12-15, then
# (q & 0x80008000) | ((q & 0x70007000) >> 3) = f16x2 of (e_k, e_{4+k}) * 2^-14. Times the block scale as f16
# (s * 2^7, < 65504 for every e4m3), so A = e2m1 * s * 2^-7 stays in f16 normal range for s >= 2^-3; the epilogue
# multiplies by weight_scale_2 * 2^7. A word's 8 elements come out in the order (0,4),(1,5),(2,6),(3,7), so x and
# act are stored K-permuted inside every 8 (position p holds element (p % 2) * 4 + p // 2).
# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _moe_route_fp4_kernel(x_ptr, logits_ptr, sg_ptr, eid_ptr, wt_ptr, x16_ptr, H: tl.constexpr, E: tl.constexpr,
                          TOPK: tl.constexpr, RENORM: tl.constexpr):
    """_moe_route_tab_kernel + writes x as f16, K-permuted inside each 8 (the B operand of moe_up_fp4)."""
    _moe_route_tab_kernel(x_ptr, logits_ptr, sg_ptr, eid_ptr, wt_ptr, H, E, TOPK, RENORM)
    m = tl.program_id(0)
    rp = tl.arange(0, H)
    p = rp % 8
    src = (rp - p) + (p % 2) * 4 + p // 2
    tl.store(x16_ptr + m * H + rp, tl.load(x_ptr + m * H + src).to(tl.float16))


_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#ifndef CORAL_FP4_ABLATE
#define CORAL_FP4_ABLATE 0
#endif

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t v) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
  return r;
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if CORAL_FP4_ABLATE == 2  // timing ablation only: no mma (wrong numerics)
  c[0] += __uint_as_float((a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1) & 0x3fffffffu);
  return;
#endif
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
// Known gap (H100, M=8 / 35 experts, ncu): up runs at ~45% of DRAM bandwidth with long-scoreboard stalls; ptxas sinks
// the up-front weight loads next to their mma (1 of 8 chunks in flight per lane). Staging the tile through shared
// memory with cp.async (all loads issued at once, lane-private slots) was SLOWER (45-51 vs 41.3 us,
// results/moe_fp4_mma_sweep4_m8.log), so that is not the fix by itself.
// 8 e2m1 (low nibble first) -> o[k] = f16x2(e_k, e_{4+k}) * 2^-14
__device__ __forceinline__ void e2m1x8(uint32_t q, uint32_t* o) {
#if CORAL_FP4_ABLATE == 1  // timing ablation only: no decode (wrong numerics)
#pragma unroll
  for (int k = 0; k < 4; ++k) o[k] = q;
  return;
#endif
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint32_t s = q << (4 * (3 - k));
    o[k] = (s & 0x80008000u) | ((s & 0x70007000u) >> 3);
  }
}
// two fp8 block scales (groups A = bytes 0-7, B = 8-15 of a lane's 16-byte load) -> f16x2 (s, s) * 2^7 each
__device__ __forceinline__ void scale_pair(uint16_t v, uint32_t& sa, uint32_t& sb) {
  uint32_t h = e4m3x2_to_f16x2(v);
  __half2 x = __hmul2(*reinterpret_cast<__half2*>(&h), __float2half2_rn(128.f));
  __half2 a = __low2half2(x), b = __high2half2(x);
  sa = *reinterpret_cast<uint32_t*>(&a); sb = *reinterpret_cast<uint32_t*>(&b);
}
// one 128-K chunk of a 16-row tile: wa = row g's 16 bytes (32 K), wb = row g+8's; X[nt] = 16 B regs per N tile
template <int NT>
__device__ __forceinline__ void chunk_mma(float (*c)[4], const uint4& wa, const uint4& wb, uint16_t qa, uint16_t qb,
                                          const uint32_t (*X)[16]) {
  uint32_t saA, saB, sbA, sbB;
  scale_pair(qa, saA, saB);
  scale_pair(qb, sbA, sbB);
  const uint32_t A[4] = {wa.x, wa.y, wa.z, wa.w}, B[4] = {wb.x, wb.y, wb.z, wb.w};
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    uint32_t da[4], db[4];
    e2m1x8(A[w], da);
    e2m1x8(B[w], db);
    const uint32_t sa = w < 2 ? saA : saB, sb = w < 2 ? sbA : sbB;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const uint32_t a0 = hmul2u(da[2 * s], sa), a2 = hmul2u(da[2 * s + 1], sa);
      const uint32_t a1 = hmul2u(db[2 * s], sb), a3 = hmul2u(db[2 * s + 1], sb);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) mma16816(c[nt], a0, a1, a2, a3, X[nt][w * 4 + s * 2], X[nt][w * 4 + s * 2 + 1]);
    }
  }
}
__device__ __forceinline__ void load_x32(uint32_t* X, const __half* p, bool live) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 v = live ? *reinterpret_cast<const uint4*>(p + 8 * j) : make_uint4(0, 0, 0, 0);
    X[4 * j] = v.x; X[4 * j + 1] = v.y; X[4 * j + 2] = v.z; X[4 * j + 3] = v.w;
  }
}
// Is CTA slot p the owner of its expert (lowest table slot holding it)? mem bit n = token n uses it; sj[n] = its slot
// in token n's table (shared memory, written identically by every warp). The shared expert (slot TOPK) is owned by
// token 0 for all tokens. M*NS <= 160.
template <int TOPK>
__device__ __forceinline__ bool members(const int* __restrict__ eid, int p, int M, unsigned& mem, int* sj, int& e) {
  constexpr int NS = TOPK + 1;
  const int lane = threadIdx.x & 31, tot = M * NS;
  mem = 0;
  if (p % NS == TOPK) {
    if (threadIdx.x < 16) sj[threadIdx.x] = TOPK;
    mem = (1u << M) - 1;
    e = -1;
    __syncthreads();
    return p / NS == 0;
  }
  e = __ldg(eid + p);
  int first = -1;
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const int idx = lane + 32 * i;
    const bool hit = idx < tot && (idx % NS) != TOPK && __ldg(eid + idx) == e;
    const unsigned b = __ballot_sync(0xffffffffu, hit);
    if (b && first < 0) first = 32 * i + __ffs(b) - 1;
    if (hit) sj[idx / NS] = idx % NS;
    unsigned bb = b;
    while (bb) { const int q = __ffs(bb) - 1; bb &= bb - 1; mem |= 1u << ((32 * i + q) / NS); }
  }
  __syncthreads();
  return first == p;
}
__device__ __forceinline__ int kperm(int k) { return (k & ~7) | ((k & 3) << 1) | ((k >> 2) & 1); }  // element -> position

template <int H, int I, int TOPK, int WARPS, int NT>
__global__ void __launch_bounds__(WARPS * 32) moe_up_fp4(const __half* __restrict__ x16, const int* __restrict__ eid,
    const uint8_t* __restrict__ w13, const uint8_t* __restrict__ s13, const float* __restrict__ g13,
    const uint8_t* __restrict__ sh13, const uint8_t* __restrict__ shs13, const float* __restrict__ shg13,
    __half* __restrict__ act, int M) {
  constexpr int NS = TOPK + 1, KW = H / WARPS, NC = KW / 128, HB = H / 2, HS = H / 16;
  __shared__ float red[WARPS][2][32][4 * NT];
  __shared__ int sj[16];
  unsigned mem; int e;
  if (!members<TOPK>(eid, blockIdx.x, M, mem, sj, e)) return;
  const uint8_t* wbase = e >= 0 ? w13 + (size_t)e * (2 * I * HB) : sh13;
  const uint8_t* sbase = e >= 0 ? s13 + (size_t)e * (2 * I * HS) : shs13;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.y * 16;
  uint4 ga[NC], gb[NC], ua[NC], ub[NC];
  uint16_t qga[NC], qgb[NC], qua[NC], qub[NC];
  {
    const uint8_t* pg = wbase + (size_t)(r0 + g) * HB + warp * (KW / 2) + t * 16;
    const uint8_t* pu = pg + (size_t)I * HB;
    const uint16_t* sg = reinterpret_cast<const uint16_t*>(sbase + (size_t)(r0 + g) * HS + warp * (KW / 16) + t * 2);
    const uint16_t* su = sg + (size_t)I * HS / 2;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      ga[c] = ld_stream(pg + c * 64); gb[c] = ld_stream(pg + 8 * HB + c * 64);
      ua[c] = ld_stream(pu + c * 64); ub[c] = ld_stream(pu + 8 * HB + c * 64);
      qga[c] = __ldg(sg + c * 4); qgb[c] = __ldg(sg + 4 * HS + c * 4);
      qua[c] = __ldg(su + c * 4); qub[c] = __ldg(su + 4 * HS + c * 4);
    }
  }
  bool live[NT];
  const __half* xr[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int n = g + 8 * nt;
    live[nt] = n < M && (mem >> n & 1);
    xr[nt] = x16 + (size_t)(live[nt] ? n : 0) * H + warp * KW + t * 32;
  }
  float cg[NT][4], cu[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) { cg[nt][i] = 0.f; cu[nt][i] = 0.f; }
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    uint32_t X[NT][16];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) load_x32(X[nt], xr[nt] + c * 128, live[nt]);
    chunk_mma<NT>(cg, ga[c], gb[c], qga[c], qgb[c], X);
    chunk_mma<NT>(cu, ua[c], ub[c], qua[c], qub[c], X);
  }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) { red[warp][0][lane][nt * 4 + i] = cg[nt][i]; red[warp][1][lane][nt * 4 + i] = cu[nt][i]; }
  __syncthreads();
  if (warp == 0) {
    const float fg = (e >= 0 ? __ldg(g13 + 2 * e) : __ldg(shg13)) * 128.f;
    const float fu = (e >= 0 ? __ldg(g13 + 2 * e + 1) : __ldg(shg13 + 1)) * 128.f;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float vg = 0.f, vu = 0.f;
#pragma unroll
        for (int q = 0; q < WARPS; ++q) { vg += red[q][0][lane][nt * 4 + i]; vu += red[q][1][lane][nt * 4 + i]; }
        const int n = nt * 8 + t * 2 + (i & 1), row = r0 + g + (i >> 1) * 8;
        if (n < M && (mem >> n & 1)) {
          vg *= fg; vu *= fu;
          act[(size_t)(n * NS + sj[n]) * I + kperm(row)] = __float2half_rn(vg / (1.f + __expf(-vg)) * vu);
        }
      }
  }
}

template <int H, int I, int TOPK, int WARPS, int RT, int NT>
__global__ void __launch_bounds__(WARPS * 32) moe_down_fp4(const __half* __restrict__ act, const int* __restrict__ eid,
    const float* __restrict__ wt, const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2, const float* __restrict__ g2,
    const uint8_t* __restrict__ sh2, const uint8_t* __restrict__ shs2, const float* __restrict__ shg2,
    float* __restrict__ ws, int* __restrict__ cnt, __nv_bfloat16* __restrict__ y, int M) {
  constexpr int NS = TOPK + 1, KW = I / WARPS, NC = KW / 128, IB = I / 2, IS = I / 16, RB = 16 * RT, NBLK = H / RB;
  __shared__ float red[WARPS][RT][32][4 * NT];
  __shared__ int sj[16];
  __shared__ int last[16];
  unsigned mem; int e;
  if (!members<TOPK>(eid, blockIdx.x, M, mem, sj, e)) return;
  const uint8_t* wbase = e >= 0 ? w2 + (size_t)e * (H * IB) : sh2;
  const uint8_t* sbase = e >= 0 ? s2 + (size_t)e * (H * IS) : shs2;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int blk = blockIdx.y, r0 = blk * RB;
  uint4 wa[RT][NC], wb[RT][NC];
  uint16_t qa[RT][NC], qb[RT][NC];
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const uint8_t* pa = wbase + (size_t)(r0 + r * 16 + g) * IB + warp * (KW / 2) + t * 16;
    const uint16_t* sa = reinterpret_cast<const uint16_t*>(sbase + (size_t)(r0 + r * 16 + g) * IS + warp * (KW / 16) + t * 2);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      wa[r][c] = ld_stream(pa + c * 64); wb[r][c] = ld_stream(pa + 8 * IB + c * 64);
      qa[r][c] = __ldg(sa + c * 4); qb[r][c] = __ldg(sa + 4 * IS + c * 4);
    }
  }
  bool live[NT];
  const __half* ar[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int n = g + 8 * nt;
    live[nt] = n < M && (mem >> n & 1);
    ar[nt] = act + (size_t)(live[nt] ? n * NS + sj[n] : 0) * I + warp * KW + t * 32;
  }
  float acc[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][nt][i] = 0.f;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    uint32_t X[NT][16];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) load_x32(X[nt], ar[nt] + c * 128, live[nt]);
#pragma unroll
    for (int r = 0; r < RT; ++r) chunk_mma<NT>(acc[r], wa[r][c], wb[r][c], qa[r][c], qb[r][c], X);
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][r][lane][nt * 4 + i] = acc[r][nt][i];
  __syncthreads();
  const float f = (e >= 0 ? __ldg(g2 + e) : __ldg(shg2)) * 128.f;
  for (int r = warp; r < RT; r += WARPS) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = nt * 8 + t * 2 + (i & 1), row = r * 16 + g + (i >> 1) * 8;
        if (n < M && (mem >> n & 1)) {
          float v = 0.f;
#pragma unroll
          for (int q = 0; q < WARPS; ++q) v += red[q][r][lane][nt * 4 + i];
          ws[((size_t)(n * NBLK + blk) * NS + sj[n]) * RB + row] = v * f * __ldg(wt + n * NS + sj[n]);
        }
      }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x < 16) {
    const int n = threadIdx.x;
    last[n] = (n < M && (mem >> n & 1)) ? (atomicAdd(cnt + n * NBLK + blk, 1) == NS - 1) : 0;
  }
  __syncthreads();
  for (int n = 0; n < M; ++n) {
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

// Down without cross-CTA reduction: CTA = 16 output rows (H / 16 CTAs); its warps loop over the distinct experts (+ the
// shared one), each loading a whole expert tile at once (memory parallelism comes from the warp count), and scale each
// token column by that token's routing weight. Warps are summed in shared memory in a fixed order, so the result is deterministic.
// slot[n][e] = position of expert e in token n's table (-1 if absent); ex = distinct experts in id order.
template <int H, int I, int TOPK, int WARPS, int NT>
__global__ void __launch_bounds__(WARPS * 32) moe_down_fp4_loop(const __half* __restrict__ act, const int* __restrict__ eid,
    const float* __restrict__ wt, const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2, const float* __restrict__ g2,
    const uint8_t* __restrict__ sh2, const uint8_t* __restrict__ shs2, const float* __restrict__ shg2,
    __nv_bfloat16* __restrict__ y, int M) {
  constexpr int NS = TOPK + 1, IB = I / 2, IS = I / 16, NC = I / 128, E = 256;
  __shared__ signed char slot[16][E];
  __shared__ int ex[16 * TOPK];
  __shared__ int nex;
  __shared__ float red[WARPS][32][4 * NT];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 16 * E / 4; i += WARPS * 32) reinterpret_cast<int*>(slot)[i] = -1;
  __syncthreads();
  for (int i = tid; i < M * NS; i += WARPS * 32)
    if (i % NS != TOPK) slot[i / NS][__ldg(eid + i)] = (signed char)(i % NS);
  __syncthreads();
  if (warp == 0) {
    int c = 0;
#pragma unroll 1
    for (int base = 0; base < E; base += 32) {
      bool f = false;
      for (int n = 0; n < M; ++n) f |= slot[n][base + lane] >= 0;
      const unsigned b = __ballot_sync(0xffffffffu, f);
      if (f) ex[c + __popc(b & ((1u << lane) - 1))] = base + lane;
      c += __popc(b);
    }
    if (lane == 0) nex = c;
  }
  __syncthreads();
  const int D = nex + 1;  // index nex = shared expert
  const int r0 = blockIdx.x * 16;
  auto wptr = [&](int q) { return q < nex ? w2 + (size_t)ex[q] * (H * IB) : sh2; };
  auto sptr = [&](int q) { return q < nex ? s2 + (size_t)ex[q] * (H * IS) : shs2; };
  uint4 wa[NC], wb[NC];
  uint16_t qa[NC], qb[NC];
  auto load = [&](int q, uint4* a, uint4* b, uint16_t* sa, uint16_t* sb) {
    const uint8_t* pa = wptr(q) + (size_t)(r0 + g) * IB + t * 16;
    const uint16_t* ps = reinterpret_cast<const uint16_t*>(sptr(q) + (size_t)(r0 + g) * IS + t * 2);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      a[c] = ld_stream(pa + c * 64); b[c] = ld_stream(pa + 8 * IB + c * 64);
      sa[c] = __ldg(ps + c * 4); sb[c] = __ldg(ps + 4 * IS + c * 4);
    }
  };
  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;
#pragma unroll 1
  for (int q = warp; q < D; q += WARPS) {
    load(q, wa, wb, qa, qb);

    const int e = q < nex ? ex[q] : -1;
    bool live[NT];
    const __half* ar[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int n = g + 8 * nt;
      const int j = n < M ? (e >= 0 ? slot[n][e] : TOPK) : -1;
      live[nt] = j >= 0;
      ar[nt] = act + (size_t)(live[nt] ? n * NS + j : 0) * I + t * 32;
    }
    float c4[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) c4[nt][i] = 0.f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      uint32_t X[NT][16];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) load_x32(X[nt], ar[nt] + c * 128, live[nt]);
      chunk_mma<NT>(c4, wa[c], wb[c], qa[c], qb[c], X);
    }
    const float f = (e >= 0 ? __ldg(g2 + e) : __ldg(shg2)) * 128.f;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int n = nt * 8 + t * 2 + k;
        const int j = n < M ? (e >= 0 ? slot[n][e] : TOPK) : -1;
        const float w = j >= 0 ? __ldg(wt + n * NS + j) * f : 0.f;
        acc[nt][k] += c4[nt][k] * w;
        acc[nt][k + 2] += c4[nt][k + 2] * w;
      }
  }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][nt * 4 + i] = acc[nt][i];
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = nt * 8 + t * 2 + (i & 1), row = g + (i >> 1) * 8;
        if (n < M) {
          float v = 0.f;
#pragma unroll
          for (int q = 0; q < WARPS; ++q) v += red[q][lane][nt * 4 + i];
          y[(size_t)n * H + r0 + row] = __float2bfloat16(v);
        }
      }
  }
}

void moe_fp4(torch::Tensor x16, torch::Tensor eid, torch::Tensor wt, torch::Tensor w13, torch::Tensor s13, torch::Tensor g13,
             torch::Tensor sh13, torch::Tensor shs13, torch::Tensor shg13, torch::Tensor w2, torch::Tensor s2, torch::Tensor g2,
             torch::Tensor sh2, torch::Tensor shs2, torch::Tensor shg2, torch::Tensor act, torch::Tensor ws, torch::Tensor cnt,
             torch::Tensor y, int64_t up_warps, int64_t dn_warps, int64_t dn_rt) {
  constexpr int H = 2048, I = 512, TOPK = 8, NS = TOPK + 1;
  const int M = x16.size(0);
  TORCH_CHECK(M >= 1 && M <= 16 && x16.size(1) == H && w13.size(1) == 2 * I && w13.size(2) == H / 2 && w2.size(1) == H, "bad shape");
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __half*>(x16.data_ptr());
  auto ap = reinterpret_cast<__half*>(act.data_ptr());
  auto u8 = [](torch::Tensor& t) { return reinterpret_cast<const uint8_t*>(t.data_ptr()); };
  auto yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  const int* ep = eid.data_ptr<int>();
  const dim3 gu(M * NS, I / 16);
  bool ok = false;
#define UP(W, N) if (!ok && up_warps == W && (M > 8) == (N == 2)) { ok = true; moe_up_fp4<H, I, TOPK, W, N><<<gu, W * 32, 0, st>>>(xp, ep, u8(w13), u8(s13), g13.data_ptr<float>(), u8(sh13), u8(shs13), shg13.data_ptr<float>(), ap, M); }
  UP(2, 1) UP(4, 1) UP(8, 1) UP(16, 1) UP(2, 2) UP(4, 2) UP(8, 2) UP(16, 2)
  TORCH_CHECK(ok, "up_warps");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ok = false;
#define DN(W, R, N) if (!ok && dn_warps == W && dn_rt == R && (M > 8) == (N == 2)) { ok = true; moe_down_fp4<H, I, TOPK, W, R, N><<<dim3(M * NS, H / (16 * R)), W * 32, 0, st>>>(ap, ep, wt.data_ptr<float>(), u8(w2), u8(s2), g2.data_ptr<float>(), u8(sh2), u8(shs2), shg2.data_ptr<float>(), ws.data_ptr<float>(), cnt.data_ptr<int>(), yp, M); }
  DN(1, 1, 1) DN(1, 2, 1) DN(1, 4, 1) DN(2, 1, 1) DN(2, 2, 1) DN(2, 4, 1) DN(4, 1, 1) DN(4, 2, 1) DN(4, 4, 1)
  DN(1, 1, 2) DN(1, 2, 2) DN(1, 4, 2) DN(2, 1, 2) DN(2, 2, 2) DN(2, 4, 2) DN(4, 1, 2) DN(4, 2, 2) DN(4, 4, 2)
#define DL(W, N) if (!ok && dn_rt == 0 && dn_warps == W && (M > 8) == (N == 2)) { ok = true; moe_down_fp4_loop<H, I, TOPK, W, N><<<H / 16, W * 32, 0, st>>>(ap, ep, wt.data_ptr<float>(), u8(w2), u8(s2), g2.data_ptr<float>(), u8(sh2), u8(shs2), shg2.data_ptr<float>(), yp, M); }
  DL(4, 1) DL(8, 1) DL(12, 1) DL(16, 1) DL(24, 1) DL(32, 1) DL(4, 2) DL(8, 2) DL(12, 2) DL(16, 2) DL(24, 2) DL(32, 2)
  TORCH_CHECK(ok, "dn cfg");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
_CPP = ("void moe_fp4(torch::Tensor x16, torch::Tensor eid, torch::Tensor wt, torch::Tensor w13, torch::Tensor s13, "
        "torch::Tensor g13, torch::Tensor sh13, torch::Tensor shs13, torch::Tensor shg13, torch::Tensor w2, torch::Tensor s2, "
        "torch::Tensor g2, torch::Tensor sh2, torch::Tensor shs2, torch::Tensor shg2, torch::Tensor act, torch::Tensor ws, "
        "torch::Tensor cnt, torch::Tensor y, int64_t up_warps, int64_t dn_warps, int64_t dn_rt);")
_ext = None
# per-M (up warps, down warps, down row tiles; 0 = moe_down_fp4_loop, the expert-loop down). Tuned at M=8, 35 distinct
# experts (bench/moe_fp4_test.py --sweep 8:35, results/moe_fp4_mma_sweep3_m8.log): (2, 12, 0) 41.35 us; split-K down
# best (2, 4, 4) 44.38. Other M reuse it until swept.
_MMA_CFG = {m: (2, 12, 0) for m in range(1, 17)}


def _load():
    global _ext
    if _ext is None:
        import os
        from torch.utils.cpp_extension import load_inline
        ab = int(os.environ.get("CORAL_FP4_ABLATE", "0"))  # 1 = skip e2m1 decode, 2 = skip mma; timing only
        d = os.environ.get("CORAL_EXT_DIR_MOE_FP4", "/data/ext_build/coral_moe_fp4") + (f"_ab{ab}" if ab else "")
        os.makedirs(d, exist_ok=True)
        _ext = load_inline("coral_moe_fp4" + (f"_ab{ab}" if ab else ""), _CPP, cuda_sources=_SRC, functions=["moe_fp4"],
                           build_directory=d, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a",
                                                                 "--use_fast_math", f"-DCORAL_FP4_ABLATE={ab}"])
    return _ext


def coral_moe_decode_fp4_mma(x, logits, shared_gate_w, w13, s13, g13, sh13, shs13, shg13, w2, s2, g2, sh2, shs2, shg2,
                             top_k=8, renorm=True, cfg=None):
    """Same arguments and result as coral_moe_decode_v9_fp4; route (Triton) + grouped up + grouped down (CUDA). M 1..16."""
    M, H = x.shape
    E = w13.shape[0]
    I = w13.shape[1] // 2
    NS = top_k + 1
    assert top_k == 8 and H == 2048 and I == 512 and 1 <= M <= 16
    up_w, dn_w, dn_rt = cfg or _MMA_CFG[M]
    dev = x.device
    cnt = _STATE.get(("mma", dev))
    if cnt is None:
        cnt = _STATE[("mma", dev)] = torch.zeros(16 * (H // 16), dtype=torch.int32, device=dev)
    eid = torch.empty(M * NS, dtype=torch.int32, device=dev)
    wt = torch.empty(M * NS, dtype=torch.float32, device=dev)
    act = torch.empty(M * NS, I, dtype=torch.float16, device=dev)
    ws = torch.empty(M * (H // 16) * NS * 16, dtype=torch.float32, device=dev)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    x16 = torch.empty(M, H, dtype=torch.float16, device=dev)
    _moe_route_fp4_kernel[(M,)](x, logits, shared_gate_w, eid, wt, x16, H=H, E=E, TOPK=top_k, RENORM=renorm,
                                num_warps=1)
    _load().moe_fp4(x16, eid, wt, w13, s13, g13, sh13, shs13, shg13, w2, s2, g2, sh2, shs2, shg2, act, ws, cnt, y,
                    up_w, dn_w, dn_rt)
    return y
