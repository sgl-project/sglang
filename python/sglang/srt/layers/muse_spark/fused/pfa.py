"""SM90 extend / mixed attention (fused QK-norm, RoPE, KV write, FP8 stream-K, norm gate) for glue.py under
hopper_kernels_enabled(); SGLang numerics except rare 1-ulp E4M3 differences in new K rows."""

import functools
import struct

import torch

from sglang.kernels.jit import muse_spark_cuda as cu

_SRC = r"""
#include "ptx.cuh"
#define CW 3
#define QB 24
#define QP (2 * QB)
#define NS 8
#define STG 16384
#define BETA 4
struct P {
  const u16* qkv; u16* out; u8* q8; u8* kc; u8* vc; u8* kv; float* ps;
  const i64* loc; const int* pt; const int* cu; const int* kvl; int* tb;
  i64 ldpt; int B, T, G, um; float sc, eps;
  const i64* pos; const float* cst; int W;
  const i64* rdst; i64 rld; int rrows, rcol, rbase, Tz;
};
DEVI u16* orow(const P& p, int t) {
  if (!p.rdst) return p.out + (size_t)t * 1024;
  const int g = p.rbase + t, o = g / p.rrows;
  return AS(u16, p.rdst[o]) + (size_t)(g - o * p.rrows) * p.rld + p.rcol;
}
// tb plan (int32): [U, key tiles, C, -] | RT[B] {t0, new, past, tile0} | UP[B + 1] | UT[um] {b, h << 16 | qb, tiles,
// cost0} | CT[G / 2 + 1] per cluster {unit, tile} | NC | CR
#define RT(p) AS(int4, (p).tb + 4)
#define UP(p) ((p).tb + 4 + 4 * (p).B)
#define UT(p) AS(int4, (p).tb + 4 + ((5 * (p).B + 4) & ~3))
#define CT(p) AS(int2, UT(p) + (p).um)
// warps that stored their partial, per (cut unit, CTA rank)
#define NC(p) AS(int, CT(p) + ((p).G >> 1) + 1)
// boundary k owning a cut unit: {unit, UT.yzw}, {t0, e, last chunk, 1}; else .w = 0
#define CR(p) AS(int4, (p).tb + ((4 + ((5 * (p).B + 4) & ~3) + 4 * (p).um + 2 * ((p).G + 1) + 2 * (p).um + 3) & ~3))
DEVI u32 ldc1(const u32* a) { u32 v; asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
DEVI void mb_arrive_el(u32 b) { asm volatile("{ .reg .pred P1; elect.sync _|P1, 0xffffffff; @P1 mbarrier.arrive.shared::cta.b64 _, [%0]; }" :: "r"(b) : "memory"); }
DEVI void pair_sync(u32 b, u32 ph, u32 rk) {
  mbar_arrive(b);
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" :: "r"(mapa(b, rk ^ 1)) : "memory");
  mbar_wait(b, ph);
}
#define OPS64 OPS32 ",%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63"
DEVI void wg_qks(float (&s)[64], u64 da, u64 d, int acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %64, 0; wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {" OPS64 "},"
      " %65, %66, p, 1, 1; }" : R16("+f", s, 0), R16("+f", s, 16), R16("+f", s, 32), R16("+f", s, 48) : "r"(acc), "l"(da), "l"(d));
}
DEVI int cdiv(i64 a, i64 b) { return (int)((a + b - 1) / b); }
DEVI int2 bscan(int2 v, int2& tot) {
  __shared__ int2 ws[8];
  const int l = threadIdx.x & 31, w = threadIdx.x >> 5;
  int2 s = v;
  for (int o = 1; o < 32; o <<= 1) {
    const int a = shfl_up(s.x, o), b = shfl_up(s.y, o);
    if (l >= o) { s.x += a; s.y += b; }
  }
  if (l == 31) ws[w] = s;
  __syncthreads();
  int2 pre = {}; tot = pre;
  for (int i = 0; i < 8; ++i) {
    if (i < w) { pre.x += ws[i].x; pre.y += ws[i].y; }
    tot.x += ws[i].x; tot.y += ws[i].y;
  }
  __syncthreads();
  return make_int2(pre.x + s.x - v.x, pre.y + s.y - v.y);
}
DEVI int tlo(const P& p, int q) { return p.W ? max(0, q - p.W + 1) >> 7 : 0; }
// unit = 48 rows x 1 KV head or 24 x 2 (a CTA each), cost = key tiles + BETA; cluster k: [C k / G2, C (k + 1) / G2)
// lengths clamped: capture dummies stay in bounds
DEVI void plan(const P& p) {
  int4* rt = RT(p); int* up = UP(p); int4* ut = UT(p); int2* ct = CT(p);
  const int tid = threadIdx.x, B = p.B, G = p.G >> 1;
  auto req = [&](int b, int& e, int& L0) {
    const int t0 = min(max(p.cu[b], 0), p.T);
    e = max(min(p.cu[b + 1], p.T) - t0, 0);
    const int kv = min(max(p.kvl[b], e), (int)p.ldpt);
    e = min(e, kv); L0 = kv - e;
    return t0;
  };
  int ch = (B + 255) >> 8, b0 = min(B, tid * ch), b1 = min(B, b0 + ch), e, L0;
  int2 s = {}, tot;
  auto add = [&]() { s.x += (e + QB - 1) / QB; s.y += ((L0 + e + 127) >> 7) - tlo(p, L0); };
  for (int b = b0; b < b1; ++b) { req(b, e, L0); add(); }
  s = bscan(s, tot);
  for (int b = b0; b < b1; ++b) {
    const int t0 = req(b, e, L0);
    rt[b] = make_int4(t0, e, L0, s.y); up[b] = s.x;
    add();
  }
  const int U = tot.x;
  if (tid == 0) { up[B] = U; p.tb[0] = U; p.tb[1] = tot.y; ct[G] = make_int2(U, 0); }
  __syncthreads();
  ch = (U + 255) >> 8;
  const int u0 = min(U, tid * ch), u1 = min(U, u0 + ch);
  int bs = 0;
  for (int lo = 0, hi = B; hi - lo > 1;) { const int m = (lo + hi) >> 1; if (up[m] <= u0) bs = lo = m; else hi = m; }
  auto unit = [&](int u, int& b, int& h, int& qb) {
    while (u >= up[b + 1]) ++b;
    const int4 r = rt[b];
    const int n2 = p.W ? 0 : (up[b + 1] - up[b]) >> 1, x = u - up[b];
    h = x < 2 * n2 ? x & 1 : 2; qb = h < 2 ? x >> 1 : x;
    const int q0 = qb * (h < 2 ? QP : QB);
    return ((r.z + min(q0 + (h < 2 ? QP : QB), r.y) + 127) >> 7) - tlo(p, r.z + q0);
  };
  int b = bs, h, qb, c = 0;
  for (int u = u0; u < u1; ++u) c += unit(u, b, h, qb) + BETA;
  int P0 = bscan(make_int2(c, 0), tot).x;
  const int C = tot.x;
  b = bs;
  for (int u = u0; u < u1; ++u) {
    const int nt = unit(u, b, h, qb), P1 = P0 + nt + BETA;
    ut[u] = make_int4(b, h << 16 | qb, nt, P0);
    const int k1 = min(G - 1, cdiv((i64)P1 * G, C) - 1);
    int k = cdiv((i64)P0 * G, C);
    if (k > k1) { P0 = P1; continue; }
    int q = (int)((i64)C * k / G), r = (int)((i64)C * k % G);
    for (; k <= k1; ++k) {
      ct[k] = make_int2(u, max(0, q - P0 - BETA));
      q += C / G; r += C % G;
      if (r >= G) { r -= G; ++q; }
    }
    P0 = P1;
  }
  if (U == 0) for (int k = tid; k < G; k += 256) ct[k] = {};
  for (int k = tid; k < 2 * U; k += 256) NC(p)[k] = 0;
  if (tid == 0) p.tb[2] = C;
  __syncthreads();
  if (tid <= G) {
    const int2 a = ct[tid], a0 = ct[max(tid - 1, 0)];
    int4 w = {};
    if (tid > 0 && a.y > 0 && a.x < U && !(a0.x == a.x && a0.y > 0)) {
      const int4 x = ut[a.x], r = rt[x.x];
      CR(p)[2 * tid] = make_int4(a.x, x.y, x.z, x.w);
      w = make_int4(r.x, r.y, cdiv((i64)(x.w + BETA + x.z) * G, C) - 1, 1);
    }
    CR(p)[2 * tid + 1] = w;
  }
}
// block 0 plans; else q, k: fp32 RMSNorm -> bf16 [-> fp32 RoPE -> bf16] -> E4M3
extern "C" __global__ void __launch_bounds__(256, 5) pa_post(const P p) {
  pdl_wait();
  if (blockIdx.x == 0) { plan(p); return; }
  const int t = (blockIdx.x - 1) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, sub = lane & 7;
  if (t >= p.Tz) return;
  if (t >= p.T || t >= p.cu[p.B]) {
    for (int i = 0; i < 4; ++i) AS(uint4, orow(p, t))[lane + 32 * i] = {};
    return;
  }
  const long long row = p.loc[t];
  float4 cs = make_float4(1.f, 1.f, 1.f, 1.f), sn = {};
  if (p.cst) { const float4* cr = AS(const float4, p.cst + p.pos[t] * 64 + 4 * sub); cs = cr[0]; sn = cr[8]; }
  uint4 raw[5];
  for (int ps = 0; ps < 5; ++ps) raw[ps] = *AS(const uint4, p.qkv + (size_t)t * 2304 + (4 * ps + (lane >> 3)) * 64 + 8 * sub);
#pragma unroll
  for (int ps = 0; ps < 5; ++ps) {
    const int hv = 4 * ps + (lane >> 3);
    const u32* w = AS(const u32, &raw[ps]);
    float x[8], ss = 0.f;
    for (int k = 0; k < 4; ++k) { x[2 * k] = bf_lo(w[k]); x[2 * k + 1] = bf_hi(w[k]); }
    for (int k = 0; k < 8; ++k) ss += x[k] * x[k];
    ss = quad_sum(ss); ss += shfl_xor(ss, 4);
    if (hv < 18) {
      const float r = rsqrtf(ss * (1.f / 64.f) + p.eps);
      for (int k = 0; k < 8; ++k) x[k] = bfr(x[k] * r);
      if (p.cst) {
        const float c4[4] = {cs.x, cs.y, cs.z, cs.w}, s4[4] = {sn.x, sn.y, sn.z, sn.w};
        for (int k = 0; k < 4; ++k) {
          const float a = x[2 * k], b = x[2 * k + 1];
          x[2 * k] = bfr(a * c4[k] - b * s4[k]); x[2 * k + 1] = bfr(a * s4[k] + b * c4[k]);
        }
      }
    }
    const uint2 v = make_uint2(e4m3x4(x[0], x[1], x[2], x[3]), e4m3x4(x[4], x[5], x[6], x[7]));
    u8* dst = hv < 16 ? p.q8 + (size_t)t * 1024 + hv * 64 : (hv < 18 ? p.kc : p.vc) + row * 128 + (hv & 1) * 64;
    *AS(uint2, dst + 8 * sub) = v;
  }
}
// keys past the request's end are zero
extern "C" __global__ void __launch_bounds__(256) pa_pack(const P p) {
  pdl_trigger();
  pdl_wait();
  __shared__ u32 vs[128 * 17];
  const int tid = threadIdx.x, nit = 2 * p.tb[1];
  for (int item = blockIdx.x; item < nit; item += gridDim.x) {
  const int gt = item >> 1, h = item & 1;
  const int4* rt = RT(p);
  int b = 0;
  for (int lo = 0, hi = p.B; hi - lo > 1;) { const int m = (lo + hi) >> 1; if (rt[m].w <= gt) b = lo = m; else hi = m; }
  const int4 r = rt[b];
  const int kv = r.y + r.z, k0 = (gt - r.w + tlo(p, r.z)) * 128, kmin = p.W ? max(0, r.z - p.W + 1) : 0;
  u8* kd = p.kv + ((size_t)gt * 2 + h) * 16384;
  int rows[2];
  for (int i = 0; i < 2; ++i) { const int pos = k0 + ((tid + 256 * i) >> 2); rows[i] = pos < kv && pos >= kmin ? __ldg(p.pt + b * p.ldpt + pos) : -1; }
  uint4 kx[2], vx[2];
  for (int i = 0; i < 2; ++i) {
    const size_t o = (size_t)rows[i] * 128 + h * 64 + (tid & 3) * 16;
    kx[i] = vx[i] = {};
    if (rows[i] >= 0) { kx[i] = ldq_u4(p.kc + o); vx[i] = ldq_u4(p.vc + o); }
  }
  for (int i = 0; i < 2; ++i) {
    const int kk = (tid + 256 * i) >> 2, c = tid & 3;
    *AS(uint4, kd + kk * 64 + ((c ^ ((kk >> 1) & 3)) << 4)) = kx[i];
    u32* vr = vs + kk * 17 + c * 4;
    vr[0] = vx[i].x; vr[1] = vx[i].y; vr[2] = vx[i].z; vr[3] = vx[i].w;
  }
  __syncthreads();
  if (tid < 128) {
    const int w = tid & 15, un = tid >> 4;
    u32 x[16], o[4][4];
    for (int e = 0; e < 16; ++e) x[e] = vs[(16 * un + e) * 17 + w];
    for (int q = 0; q < 4; ++q) {
      const u32 t0 = prmt(x[2 * q], x[2 * q + 1], 0x5140), t1 = prmt(x[2 * q], x[2 * q + 1], 0x7362);
      const u32 t2 = prmt(x[2 * q + 8], x[2 * q + 9], 0x5140), t3 = prmt(x[2 * q + 8], x[2 * q + 9], 0x7362);
      o[0][q] = prmt(t0, t2, 0x5410); o[1][q] = prmt(t0, t2, 0x7632); o[2][q] = prmt(t1, t3, 0x5410); o[3][q] = prmt(t1, t3, 0x7632);
    }
    for (int j = 0; j < 4; ++j) {
      const int d = 4 * w + j;
      *AS(uint4, kd + 8192 + d * 128 + ((un ^ (d & 7)) << 4)) = make_uint4(o[j][0], o[j][1], o[j][2], o[j][3]);
    }
  }
  __syncthreads();
  }
}
template <class F> DEVI void segs(const P& p, F&& f) {
  const int2 a = CT(p)[blockIdx.x >> 1], z = CT(p)[(blockIdx.x >> 1) + 1];
  const int U = p.tb[0];
  for (int u = a.x; u <= z.x && u < U; ++u) {
    const int4 x = UT(p)[u];
    const int jb = u == a.x ? a.y : 0, je = u == z.x ? z.y : x.z;
    if (je > jb) f(x, jb, je);
  }
}
// bf16(o / l), then SGLang's norm_gate
DEVI const u32* grow(const P& p, int t, int hg) { return AS(const u32, p.qkv + (size_t)t * 2304 + 1280 + hg * 64) + (threadIdx.x & 3); }
template <bool CG = false> DEVI void gates(const P& p, u32 (&g)[16], int4 r, int i, int hg) {
  const bool va = i < r.y, vb = i + 1 < r.y;
  const u32* ga = grow(p, r.x + i, hg);
  const u32* gb = grow(p, r.x + i + 1, hg);
  auto ld = [](const u32* a) { return CG ? __ldcg(a) : ldc1(a); };
  for (int j = 0; j < 8; ++j) { g[j] = va ? ld(ga + 4 * j) : 0u; g[8 + j] = vb ? ld(gb + 4 * j) : 0u; }
}
// ST: rows staged in shared memory, stored as whole lines (NVLink)
template <bool ST = false> DEVI void epi(const P& p, float (&o)[32], float la, float lb, const u32 (&gv)[16], int4 r, int i, int hg, u32 stg = 0) {
  const int tq = threadIdx.x & 3, ta = r.x + i, tb = ta + 1;
  const bool va = i < r.y, vb = i + 1 < r.y;
  const float ia = rcp0(la), ib = rcp0(lb);
  float sa = 0.f, sb = 0.f;
  for (int j = 0; j < 8; ++j) {
    for (int e = 0; e < 4; ++e) o[4 * j + e] = bfr(o[4 * j + e] * (e < 2 ? ia : ib));
    sa += o[4 * j] * o[4 * j] + o[4 * j + 1] * o[4 * j + 1]; sb += o[4 * j + 2] * o[4 * j + 2] + o[4 * j + 3] * o[4 * j + 3];
  }
  sa = quad_sum(sa); sb = quad_sum(sb);
  const float na = rsqrtf(sa * (1.f / 64.f) + p.eps), nb = rsqrtf(sb * (1.f / 64.f) + p.eps);
  if (ST) {
    const int g = (threadIdx.x >> 2) & 7, lane = threadIdx.x & 31;
    for (int j = 0; j < 8; ++j) {
      sts32(stg + g * 144 + 16 * j + 4 * tq, gate2(o[4 * j], o[4 * j + 1], na, gv[j]));
      sts32(stg + 1152 + g * 144 + 16 * j + 4 * tq, gate2(o[4 * j + 2], o[4 * j + 3], nb, gv[8 + j]));
    }
    __syncwarp();
    for (int rr = 0; rr < 2; ++rr)
      if (rr ? vb : va)
        for (int k = 0; k < 2; ++k) {
          const int x = lane + 32 * k;
          AS(uint4, orow(p, rr ? tb : ta) + (hg - g) * 64)[x] = lds128(stg + 1152 * rr + (x >> 3) * 144 + (x & 7) * 16);
        }
    return;
  }
  u32* oa = va ? AS(u32, orow(p, ta) + hg * 64 + 2 * tq) : nullptr;
  u32* ob = vb ? AS(u32, orow(p, tb) + hg * 64 + 2 * tq) : nullptr;
  for (int j = 0; j < 8; ++j) {
    if (va) oa[4 * j] = gate2(o[4 * j], o[4 * j + 1], na, gv[j]);
    if (vb) ob[4 * j] = gate2(o[4 * j + 2], o[4 * j + 3], nb, gv[8 + j]);
  }
}
#define SLOT(p, k, kind, c) AS(float4, (p).ps + ((size_t)((k) * 2 + (kind)) * CW + (c)) * (128 * 36))
DEVI int i3(bool mx, int a, int b, int c) { return mx ? max(max(a, b), c) : min(min(a, b), c); }
template <bool MX> DEVI int rmx(const float (&s)[64], int r, int e = 0) {
#define V_(k) __float_as_int(s[4 * ((k) >> 1) + 2 * r + ((k) & 1)])
  int m0 = i3(MX, V_(0), V_(1), V_(2)), m1 = i3(MX, V_(3), V_(4), V_(5)), m2 = i3(MX, V_(6), V_(7), V_(8)), m3 = i3(MX, V_(9), V_(10), V_(11));
  for (int k = 12; k < 28; k += 8) {
    m0 = i3(MX, m0, V_(k), V_(k + 1)); m1 = i3(MX, m1, V_(k + 2), V_(k + 3));
    m2 = i3(MX, m2, V_(k + 4), V_(k + 5)); m3 = i3(MX, m3, V_(k + 6), V_(k + 7));
  }
  m0 = i3(MX, m0, V_(28), V_(29)); m1 = i3(MX, m1, V_(30), V_(31));
#undef V_
  m0 = i3(MX, m0, m1, m2);
  return MX ? max(max(m0, m3), e) : min(m0, m3);
}
extern "C" __global__ void __launch_bounds__(128 * (CW + 1), 1) pa_fwd(const P p) {
  extern __shared__ __align__(1024) u8 sm[];
  const u32 sb = smem_u32(sm), bars = sb + NS * STG;
  const int tid = threadIdx.x, wg = tid >> 7, lt = tid & 127;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(bars + 16 * s, 1); mbar_init(bars + 16 * s + 8, 4 * CW); mbar_init(bars + 16 * NS + 8 * s, 2); }
    mbar_init_fence();
  }
  cl_sync();
  pdl_wait();
  pdl_trigger();
  const int4* rt = RT(p);
  if (wg == 0) {
    nreg_dec<24>();
    if (tid != 0) return;
    const u32 rk = cl_rank();
    int it = 0;
    auto empty = [&](int st) {
      mbar_wait(bars + 16 * st + 8, ((it / NS) & 1) ^ 1);
      if (it >= NS) pair_sync(bars + 16 * NS + 8 * st, ((it / NS) & 1) ^ 1, rk);
    };
    segs(p, [&](const int4& x, int jb, int je) {
      const int4 r = rt[x.x];
      const int k2 = x.y >> 16;
      const u8* src = p.kv + ((size_t)(r.w + tlo(p, r.z + (x.y & 0xFFFF) * (k2 < 2 ? QP : QB)) - tlo(p, r.z) + je - 1) * 2 + (k2 < 2 ? k2 : rk)) * 16384;
      for (int j = jb; j < je; ++j, ++it, src -= 32768) {
        const int st = it % NS;
        empty(st);
        mbar_expect(bars + 16 * st, 16384);
        if (k2 < 2) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], 8192, [%2], %3;"
            :: "r"(sb + st * STG + rk * 8192), "l"(src + rk * 8192), "r"(bars + 16 * st), "h"((u16)3) : "memory");
        else bulk_g2s(sb + st * STG, src, 16384, bars + 16 * st);
      }
    });
    for (int k = 0; k < NS; ++k, ++it) empty(it % NS);
    return;
  }
  nreg_inc<160>();
  const int c = shfl(wg, 0) - 1, warp = shfl(lt >> 5, 0), lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const u64 dk0 = wg_desc(sb, 16, 512, 2), dv0 = wg_desc(sb + 8192, 16, 1024, 1);
  const float sc = p.sc;
  int it = 0, nseg = 0;
  const int rk = cl_rank();
  auto body = [&](const int4& x, int jb, int je, bool fu) {
    const int h = x.y >> 16, qb = x.y & 0xFFFF;
    const int4 r = rt[x.x];
    const int ia = qb * QB + c * 8 + 2 * warp, pa = r.z + ia, hg = h * 8 + g;
    const u32 qwb = sb + NS * STG + 1024 + 4096 * c;
    const u64 qd0 = wg_desc(qwb, 16, 512, 2);
    {
      const int R = lt >> 1, hb = lt & 1, tk = qb * QB + c * 8 + (R >> 3);
      uint4 x0 = {}, x1 = {};
      if (tk < r.y) {
        const u8* src = p.q8 + ((size_t)(r.x + tk) * 16 + h * 8 + (R & 7)) * 64 + 32 * hb;
        x0 = ldq_u4(src); x1 = ldq_u4(src + 16);
      }
      const u32 ra = qwb + R * 64, sw = (R >> 1) & 3;
      sts4(ra + (((2 * hb) ^ sw) << 4), x0);
      sts4(ra + (((2 * hb + 1) ^ sw) << 4), x1);
      fence_async();
      bar_sync(1 + c, 128);
    }
    float o[32] = {};
    float ma = NEG_INF, mb = NEG_INF, la = 0.f, lb = 0.f, xa = 0.f, xb = 0.f, sca = 1.f, scb = 1.f;
    bool resc = false, mpos = false;
    const int it0 = it, n = je - jb;
    it += n;
    const int q0 = r.z + qb * QB, tl = tlo(p, q0) + jb, jm = ((q0 + 1) >> 7) - tl, kw = q0 + QB - p.W, jw = p.W && kw >= 0 ? (kw >> 7) - tl : -1;
    const u32 nu = max(jm - jw - 1, 0);
    float s[64];
    u32 pfa[4][4], pfb[4][4];
    auto wfull = [&](int jt) { const u32 i = it0 + jt; mbar_wait(bars + 16 * (i % NS), (i / NS) & 1); };
    auto issue_qk = [&](int jt, bool w = true) {
      const u32 i = it0 + jt, st = i % NS;
      if (w) wfull(jt);
      const u64 dk = dadd(dk0, st * (STG >> 4));
      wg_fence();
      wg_qks(s, qd0, dk, 0);
      wg_qks(s, qd0 + 2, dk + 2, 1);
      wg_commit();
    };
    auto issue_pv = [&](int jt, const u32 (&pf)[4][4]) {
      const u64 dv = dadd(dv0, ((u32)(it0 + jt) % NS) * (STG >> 4));
      wg_fence();
      for (int kc = 0; kc < 4; ++kc) wg_pv(o, pf[kc], dv + 2 * kc);
      wg_commit();
    };
    auto prep = [&](int jt) {
      jt = n - 1 - jt;
      const int k0 = (tl + jt) * 128, da = pa - k0 - 2 * tq;
      if ((u32)(jt - jw - 1) >= nu) {
      auto msk = [&](int d, auto f) {
        for (int n8 = 0; n8 < 16; ++n8)
          for (int c2 = 0; c2 < 2; ++c2) {
            if (f(d, 8 * n8 + c2)) s[4 * n8 + c2] = NEG_INF;
            if (f(d + 1, 8 * n8 + c2)) s[4 * n8 + 2 + c2] = NEG_INF;
          }
      };
      if (jt >= jm) msk(da, [](int d, int k) { return d < k; });
      else msk(da - p.W + 1, [](int d, int k) { return d > k; });
      }
      // m as 3rd input: no fallback once all row maxes >= +0
      int ia_ = rmx<true>(s, 0, __float_as_int(ma)), ib_ = rmx<true>(s, 1, __float_as_int(mb));
      if (!mpos && __any_sync(~0u, (ia_ | ib_) < 0)) {
        const int na_ = rmx<false>(s, 0), nb_ = rmx<false>(s, 1);
        ia_ = ia_ < 0 ? na_ : ia_; ib_ = ib_ < 0 ? nb_ : ib_;
      }
      const float na = fmaxf(ma, quad_max(__int_as_float(ia_))), nb = fmaxf(mb, quad_max(__int_as_float(ib_)));
      resc = __any_sync(~0u, na != ma || nb != mb);
      if (resc) {
        const float ca = na == NEG_INF ? 0.f : na, cb = nb == NEG_INF ? 0.f : nb;
        sca = ex2((ma - ca) * sc); scb = ex2((mb - cb) * sc);
        la *= sca; lb *= scb;
        ma = na; mb = nb;
        mpos = __all_sync(~0u, (__float_as_int(ma) | __float_as_int(mb)) >= 0);
        xa = na == NEG_INF ? 0.f : -fmaf(na, sc, -8.f); xb = nb == NEG_INF ? 0.f : -fmaf(nb, sc, -8.f);
      }
    };
    auto expo = [&](u32 (&pf)[4][4]) {
      float ua[4] = {}, ub[4] = {};
      for (int kc = 0; kc < 4; ++kc) {
        float q[16];
        for (int e = 0; e < 16; ++e) {
          const int i = 16 * kc + e;
          const bool rb = (i >> 1) & 1;
          q[e] = ex2(fmaf(s[i], sc, rb ? xb : xa));
          if (rb) ub[(i >> 2) & 3] += q[e]; else ua[(i >> 2) & 3] += q[e];
        }
        p_frag(pf[kc], q);
      }
      la += (ua[0] + ua[1]) + (ua[2] + ua[3]); lb += (ub[0] + ub[1]) + (ub[2] + ub[3]);
    };
    auto fin = [&](int jt, u32 (&pf)[4][4]) {
      acc_pin(o);
      for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) asm volatile("" : "+r"(pf[i][k]) :: "memory");
      mb_arrive_el(bars + 16 * ((u32)(it0 + jt) % NS) + 8);
    };
    issue_qk(0);
    bool wok = true;
    if (n > 1) wok = mbar_test(bars + 16 * ((it0 + 1) % NS), ((it0 + 1) / NS) & 1);
    wg_wait<0>();
    acc_pin(s);
    prep(0);
    expo(pfa);
    resc = false;
    auto tile = [&](int j, u32 (&pp)[4][4], u32 (&pc)[4][4]) {
      issue_qk(j, !wok);
      issue_pv(j - 1, pp);
      if (j + 1 < n) { const u32 i1 = it0 + j + 1; wok = mbar_test(bars + 16 * (i1 % NS), (i1 / NS) & 1); }
      wg_wait<1>();
      acc_pin(s);
      prep(j);
      acc_pin(s);
      expo(pc);
      wg_wait<0>();
      fin(j - 1, pp);
      if (resc) {
        for (int i = 0; i < 32; ++i) o[i] *= i & 2 ? scb : sca;
        resc = false;
      }
    };
    int j = 1;
    for (; j + 1 < n; j += 2) { tile(j, pfa, pfb); tile(j + 1, pfb, pfa); }
    if (j < n) tile(j, pfa, pfb);
    u32 gv[16];
    if (fu) gates(p, gv, r, ia, hg);
    if ((n - 1) & 1) issue_pv(n - 1, pfb); else issue_pv(n - 1, pfa);
    wg_wait<0>();
    fin(n - 1, pfa);
    la = quad_sum(la); lb = quad_sum(lb);
    if (fu) {
      epi(p, o, la, lb, gv, r, ia, hg);
    } else {
      float4* sl = SLOT(p, blockIdx.x, nseg ? 1 : 0, c);
      for (int j = 0; j < 8; ++j) __stcg(sl + j * 128 + lt, make_float4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]));
      __stcg(sl + 8 * 128 + lt, make_float4(la, lb, ma == NEG_INF ? NEG_INF : -xa, mb == NEG_INF ? NEG_INF : -xb));
    }
  };
  auto skip = [&](int n) {
    for (; n > 0; --n, ++it) { mbar_wait(bars + 16 * (it % NS), (it / NS) & 1); mb_arrive_el(bars + 16 * (it % NS) + 8); }
  };
  segs(p, [&](const int4& x, int jb, int je) {
    const int4 r = rt[x.x];
    const int k2 = x.y >> 16, q = x.y & 0xFFFF, qb = k2 < 2 ? 2 * q + rk : q, lo = k2 < 2 ? tlo(p, r.z + qb * QB) - tlo(p, r.z + q * QP) : 0;
    const int nt = qb * QB < r.y ? ((r.z + min(qb * QB + QB, r.y) + 127) >> 7) - tlo(p, r.z + qb * QB) : 0;
    const int j0 = max(jb, lo), j1 = max(j0, min(je, lo + nt));
    const bool fu = jb == 0 && je == x.z;
    skip(je - j1);
    if (j1 > j0) body(make_int4(x.x, (k2 < 2 ? k2 : rk) << 16 | qb, nt, x.w), j0 - lo, j1 - lo, fu);
    else if (!fu) __stcg(SLOT(p, blockIdx.x, nseg ? 1 : 0, c) + 8 * 128 + lt, make_float4(0.f, 0.f, NEG_INF, NEG_INF));
    skip(j0 - jb);
    ++nseg;
  });
  // cut pieces are counted after the unit loop: the tile loop compiles as without counting
  const int2 ra = CT(p)[blockIdx.x >> 1], rz = CT(p)[(blockIdx.x >> 1) + 1];
  auto cut = [&](int u) {
    if (u > rz.x || u >= p.tb[0]) return false;
    const int xz = UT(p)[u].z, jb = u == ra.x ? ra.y : 0, je = u == rz.x ? rz.y : xz;
    return je > jb && !(jb == 0 && je == xz);
  };
  const bool ka = cut(ra.x), kz = rz.x != ra.x && cut(rz.x);
  if (ka || kz) {
    __threadfence();
    __syncwarp();
    if (lane == 0) {
      if (ka) red_release(NC(p) + 2 * ra.x + rk, 1);
      if (kz) red_release(NC(p) + 2 * rz.x + rk, 1);
    }
  }
}
DEVI void mrg(float (&o)[32], float& l, float& M, const float* v, float vl, float vm, int rr) {
  if (vm == NEG_INF) return;
  const float nm = fmaxf(M, vm), f0 = ex2(M - nm), f1 = ex2(vm - nm);
  l = fmaf(l, f0, __fmul_rn(vl, f1));
  for (int j = 0; j < 8; ++j)
    for (int e = 0; e < 2; ++e) o[4 * j + 2 * rr + e] = fmaf(o[4 * j + 2 * rr + e], f0, __fmul_rn(v[4 * j + 2 * rr + e], f1));
  M = nm;
}
// fixed merge tree
struct PS { float o[32], l[2], m[2]; };
DEVI void ps_init(PS& a) {
  for (int i = 0; i < 32; ++i) a.o[i] = 0.f;
  a.l[0] = a.l[1] = 0.f; a.m[0] = a.m[1] = NEG_INF;
}
DEVI void ps_mrg(PS& a, const float (&v)[32], float l0, float l1, float m0, float m1) {
  mrg(a.o, a.l[0], a.m[0], v, l0, m0, 0);
  mrg(a.o, a.l[1], a.m[1], v, l1, m1, 1);
}
DEVI void ps_add(PS& a, const PS& b) { ps_mrg(a, b.o, b.l[0], b.l[1], b.m[0], b.m[1]); }
// Merges a cut unit (same tree) once pa_fwd's warps counted its pieces in NC.
extern "C" __global__ void __launch_bounds__(128, 3) pa_cmb(const P p) {
  pdl_trigger();
  extern __shared__ __align__(16) float4 pb[];
  __shared__ int lst[68], cnt[4], wc[4][4];
  const int k = blockIdx.x / (2 * CW) + 1, rk = blockIdx.x / CW & 1, c = blockIdx.x % CW, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, lt = tid;
  const int4 x = __ldcg(CR(p) + 2 * k), r = __ldcg(CR(p) + 2 * k + 1);
  const int G2 = p.G >> 1, C = __ldcg(p.tb + 2);
  // the last block waits for pa_fwd before it exits: pa_cmb completes after pa_fwd
  const bool wl = blockIdx.x == gridDim.x - 1;
  if (!r.w) { if (wl) pdl_wait(); return; }
  const int u = x.x, k2 = x.y >> 16, qb = k2 < 2 ? 2 * (x.y & 0xFFFF) + rk : x.y & 0xFFFF, ia = qb * QB + c * 8 + 2 * w, hg = (k2 < 2 ? k2 : rk) * 8 + (lane >> 2);
  if (qb * QB >= r.y) { if (wl) pdl_wait(); return; }
  const int nf = C >= G2 ? r.z - k + 2 : 0;
  // no empty chunk (C >= G2): the count is known, thread 32 waits while the list is built
  if (tid == 32 && nf) while (ld_acquire(NC(p) + 2 * u + rk) < 12 * nf) __nanosleep(32);
  // pieces in tree order: thread i takes chunk k - 1 + i of slice i & 3
  {
    const int x0 = x.w + BETA, x1 = x0 + x.z, kk = k - 1 + tid, sl = tid & 3;
    bool in = false;
    int s0 = 0;
    if (kk <= r.z) {
      const long long a = (long long)C * kk;
      s0 = (int)(a / G2);
      in = max(s0, x0) < min((int)((a + C) / G2), x1);
    }
    const u32 m = __ballot_sync(~0u, in), ms = m & (0x11111111u << sl);
    if (lane < 4) wc[w][lane] = __popc(m & (0x11111111u << lane));
    __syncthreads();
    const int n0 = wc[0][0] + wc[1][0] + wc[2][0] + wc[3][0], n1 = wc[0][1] + wc[1][1] + wc[2][1] + wc[3][1];
    const int n2 = wc[0][2] + wc[1][2] + wc[2][2] + wc[3][2], n3 = wc[0][3] + wc[1][3] + wc[2][3] + wc[3][3];
    if (in) {
      int at = sl == 0 ? 0 : sl == 1 ? n0 + n2 : sl == 2 ? n0 : n0 + n2 + n1;
      for (int v = 0; v < w; ++v) at += wc[v][sl];
      lst[at + __popc(ms & ((1u << lane) - 1))] = (2 * kk + rk) * 2 + (s0 >= x.w ? 0 : 1);
    }
    if (tid == 0) { cnt[0] = n0; cnt[1] = n2; cnt[2] = n1; cnt[3] = n3; }
  }
  __syncthreads();
  const int n = cnt[0] + cnt[1] + cnt[2] + cnt[3];
  if (tid == 0 && !nf) while (ld_acquire(NC(p) + 2 * u + rk) < 12 * n) __nanosleep(32);
  // one acquire; the barrier orders all partial reads after it
  if (!nf) __syncthreads();
  u32 gv[16];  // gate rows, loaded after the acquire
  int pos = 0;
  auto get = [&](float (&v)[32]) {
    if (pos % 2 == 0) {
      for (int i = 0; i < 2 && pos + i < n; ++i) {
        const float4* src = AS(const float4, p.ps + ((size_t)lst[pos + i] * CW + c) * (128 * 36)) + lt;
        for (int j = 0; j < 9; ++j) cpa16(smem_u32(pb + (i * 9 + j) * 128 + tid), src + j * 128);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      if (pos == 0) gates<true>(p, gv, r, ia, hg);
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    const float4* b = pb + (pos++ % 2) * 9 * 128 + tid;
    for (int j = 0; j < 8; ++j) { const float4 t = b[j * 128]; v[4 * j] = t.x; v[4 * j + 1] = t.y; v[4 * j + 2] = t.z; v[4 * j + 3] = t.w; }
    return b[8 * 128];
  };
  auto slice = [&](PS& acc, int m) {
    ps_init(acc);
#pragma unroll 1
    for (int e = 0; e < m; ++e) {
      float v[32];
      const float4 lm = get(v);
      ps_mrg(acc, v, lm.x, lm.y, lm.z, lm.w);
    }
  };
  PS A, B;
  slice(A, cnt[0]);
  slice(B, cnt[1]);
  ps_add(A, B);
  {
    PS D;
    slice(B, cnt[2]);
    slice(D, cnt[3]);
    ps_add(B, D);
  }
  ps_add(A, B);
  if (pos == 0) gates<true>(p, gv, r, ia, hg);  // no pieces
  __syncthreads();
  epi<true>(p, A.o, A.l[0], A.l[1], gv, r, ia, hg, smem_u32(pb) + w * 2304);
  if (wl) pdl_wait();
} """

_NS, _STG, _CW = 8, 16384, 3
_SMEM = {"pa_fwd": _NS * _STG + 1024 + 4096 * _CW, "pa_cmb": 2 * 9 * 128 * 16}
_SLOT = 128 * 36 * 4


class _Mod:
    def __init__(self, dev):
        self.dev = dev
        mod, self.G = cu.device_module(dev, "pfe.cu", _SRC)
        self.ps = torch.empty(self.G * 2 * _CW * _SLOT, dtype=torch.uint8, device=dev)
        self.f = {
            n: cu.function(mod, n, _SMEM.get(n, 0))
            for n in ("pa_post", "pa_pack", "pa_fwd", "pa_cmb")
        }

    def launch(self, name, grid, block, args, cl=0):
        # PDL: pa_cmb reads the plan early (pa_fwd triggers after its wait); others wait first
        cu.launch(
            self.f[name],
            grid,
            block,
            _SMEM.get(name, 0),
            torch.cuda.current_stream(self.dev).cuda_stream,
            args,
            [cu.pdl()] + ([cu.cluster_dim(cl)] if cl else []),
        )


_mod = functools.cache(_Mod)


def forward(
    qkv,
    cache_loc,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    kv_bound,
    sm_scale,
    qk_scale_factor,
    eps,
    out,
    window,
    cos_sin,
    positions,
    push=None,
    last_only=False,
):
    """out [T, 1024] = norm-gated attention, new K / V -> cache_loc; kv_bound >= packed keys; window includes the query."""
    T, B, dev = qkv.shape[0], cache_seqlens.shape[0], qkv.device
    m = _mod(dev)
    G = m.G
    um = 2 * (T // 24 + B) + 2
    nkt = (kv_bound + T) // 128 + B
    tb = qkv.new_empty(
        4 + ((5 * B + 4) & ~3) + 4 * um + 2 * (G + 1) + 2 * um + 4 * G + 12,
        dtype=torch.int32,
    )
    q8 = qkv.new_empty(T * 1024, dtype=torch.uint8)
    # packed K | V^T: 32 KB per 128 keys read (~256 B per KV token)
    kv = qkv.new_empty(0 if last_only else nkt * 32768, dtype=torch.uint8)
    args = b"".join(
        struct.pack("Q", t.data_ptr())
        for t in (
            qkv,
            out,
            q8,
            k_cache,
            v_cache,
            kv,
            m.ps,
            cache_loc,
            page_table,
            cu_seqlens_q,
            cache_seqlens,
            tb,
        )
    )
    rope = cos_sin is not None
    args += struct.pack(
        "qiiiiffQQi",
        page_table.stride(0),
        B,
        T,
        G,
        um,
        cu.scale_log2(sm_scale, qk_scale_factor),
        eps,
        positions.data_ptr() if rope else 0,
        cos_sin.data_ptr() if rope else 0,
        window,
    )
    table, rows, ld, col, base, tz = push or (None, 1, 0, 0, 0, T)
    tz = max(tz, T)
    args += struct.pack(
        "=4xQqiiii", table.data_ptr() if push else 0, ld, rows, col, base, tz
    )
    m.launch("pa_post", (tz + 7) // 8 + 1, 256, args)
    if last_only:
        last = cu_seqlens_q[1:].long() - 1
        return forward(
            qkv[last],
            cache_loc[last],
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            torch.arange(B + 1, dtype=cu_seqlens_q.dtype, device=dev),
            kv_bound,
            sm_scale,
            qk_scale_factor,
            eps,
            out,
            window,
            cos_sin,
            positions[last] if rope else None,
        )
    m.launch("pa_pack", min(2 * nkt, 8 * G), 256, args)
    m.launch("pa_fwd", G, 128 * (_CW + 1), args, 2)
    m.launch("pa_cmb", (G - 2) * _CW, 128, args)
    return out
