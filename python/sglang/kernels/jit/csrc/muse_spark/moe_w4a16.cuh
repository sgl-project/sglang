// Adapted from https://github.com/gauravjain14/muse-okc-submissions (muse_moe_w4a16, Apache-2.0; its NOTICE credits
// https://github.com/vjeux/okc-2026 and https://github.com/alygg77/okc-muse-spark-kernels), modified.
// SM90 Muse Spark MoE, <= 2048 tokens: NVFP4 experts x BF16 rows, SwiGLU, post-expert RMSNorm and top-k sum. One CTA
// per SM: warps 0-7 MMA, warp 8 fills the ring, 9-11 gather x / signal / epilogue; CTAs only wait on claimed work.
#include "ptx.cuh"
struct __align__(16) StageInfo {
  int item, phase, tile, rb, ks, st, last, n, p0, e, pad0, pad1;
};
DEVI const u8* wc6_w(const void* wc, int e, int nrg, int kc, int rg) {
  return AS(const u8, wc) + ((size_t)(e * 16 + kc) * (nrg >> 4) + (rg >> 4)) * 35840 + (rg & 15) * 2048;
}
DEVI const u8* wc6_s(const void* wc, int e, int nrg, int kc, int rg) {
  return wc6_w(wc, e, nrg, kc, rg & ~15) + 32768 + (rg & 15) * 192;
}
// Tile order of a lane's expert: count `key` descending, ties by lane (key < 2^(nb + 1)).
DEVI int key_rank(int key, int nb) {
  u32 gtm = 0u, eqm = ~0u;
  for (int b = nb; b >= 0; --b) {
    const bool one = (key >> b) & 1;
    const u32 B = ballot(one);
    if (one)
      eqm &= B;
    else {
      gtm |= eqm & B;
      eqm &= ~B;
    }
  }
  return __popc(gtm) + __popc(eqm & ((1u << (threadIdx.x & 31)) - 1u));
}
// Alternate heavy and light experts, except the 5 lightest (c = this lane's route count).
DEVI int mix_rank(int rank, int c) {
  const int nz = __popc(ballot(c > 0)), m = nz - min(5, nz);
  return rank >= m ? rank : rank < (m + 1) / 2 ? 2 * rank : 2 * (m - 1 - rank) + 1;
}
DEVI int pwt(u32 n, int tt) {
  return n ? 64 * (int)((n + tt - 1) / tt) + (int)n : 0;
}
#define LD_IDS(ids, t, v0, v1)             \
  {                                        \
    v0 = AS(const int4, ids)[(t) * 2];     \
    v1 = AS(const int4, ids)[(t) * 2 + 1]; \
  }
#define EV8(v0, v1) {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w}
DEVI void mbar_tx(u32 b, u32 n) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared.b64 [%0], %1;" ::"r"(b), "r"(n) : "memory");
}
DEVI void mbar_arrive_n(u32 bar, u32 n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(n) : "memory");
}
DEVI void mbar_poll(u32 bar, u32 parity) {
  u32 ok = 0;
  while (!ok)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok)
                 : "r"(bar), "r"(parity)
                 : "memory");
}
template <int HINT = 1000000>
DEVI void mbar_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  while (!ok)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok)
                 : "r"(bar), "r"(parity), "n"(HINT)
                 : "memory");
}
DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar, u64 pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
          "r"(dst),
      "l"(src),
      "r"(bytes),
      "r"(bar),
      "l"(pol)
      : "memory");
}

DEVI int gix(u32 gsl, u32 s) {
  return (gsl & 0xff) == s ? 0 : (gsl >> 8 & 0xff) == s ? 1 : -1;
}
DEVI int ld_relaxed(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
DEVI bool elect_one() {
  u32 p;
  asm volatile("{ .reg .pred e; elect.sync _|e, 0xffffffff; selp.u32 %0, 1, 0, e; }" : "=r"(p));
  return p;
}
DEVI void consumer_sync() {
  bar_sync(1, 256);
}

#define M_BF 0x81C081C0u

template <int N>
DEVI void wgmma_rs(float* d, const u32* a, u64 desc);
// @WGMMA_RS@

__constant__ u32 c_s6o[3] = {16u, 256u, 64u};
__constant__ u32 c_s6oc[1] = {0x78007800u};
DEVI void ld_stage(const u8* wbase, const u8* sclb, uint4 (&qw)[2][4], u32 (&R)[8]) {
  for (int kk = 0; kk < 4; ++kk) {
    qw[0][kk] = *AS(const uint4, wbase + kk * 512);
    qw[1][kk] = *AS(const uint4, wbase + 2048 + kk * 512);
  }
  const u32* sw = AS(const u32, sclb);
  const u32 w0 = sw[0], w1 = sw[1], w2 = sw[2];
  constexpr u32 M6 = 0x03F003F0u, NB = 0x00F000F0u;
  const u32 C = c_s6oc[0];
  R[0] = (w0 & M6) | C;
  R[4] = ((w0 >> 6) & M6) | C;
  R[1] = (w1 & M6) | C;
  R[5] = ((w1 >> 6) & M6) | C;
  R[2] = (w2 & M6) | C;
  R[6] = ((w2 >> 6) & M6) | C;
  const u32 vg = ((w0 * c_s6o[0]) & NB) | ((w2 * c_s6o[1]) & ~NB);
  R[3] = (vg & M6) | C;
  const u32 vh = ((w1 * c_s6o[0]) & NB) | ((w2 * c_s6o[2]) & ~NB);
  R[7] = (vh & M6) | C;
}

__constant__ u32 c_x8[3] = {4u, 64u, 16u};
DEVI void dq4(u32 q, u32 qp, int j, u32 R, u32* a) {
  const u32 F = j == 0 ? __byte_perm(q, qp, 0x6341) : __byte_perm(qp, q, 0x6341);
  u32 o[4];
  o[0] = q & M_BF;
  if (j == 0) {
    o[1] = ((q * c_x8[0]) ^ F) & M_BF;
    o[2] = (q * c_x8[1]) & M_BF;
    o[3] = rotl(q, 12) & M_BF;
  } else {
    o[1] = (q * c_x8[2]) & M_BF;
    o[2] = rotl(q, 10) & M_BF;
    o[3] = (rotl(q, 14) ^ F) & M_BF;
  }
  a[0] = bmul2s<0>(o[0], R);
  a[1] = bmul2s<1>(o[1], R);
  a[2] = bmul2s<0>(o[2], R);
  a[3] = bmul2s<1>(o[3], R);
}
// EWR: weights back to the producer once loaded (bar id depends on every load; c_z0 = 0)
__constant__ u32 c_z0;
DEVI void ew_arv(const uint4 (&q)[2][4], const u32* R, int ew) {
  u32 z = R[0] ^ R[1] ^ R[2];
  for (int i = 0; i < 8; ++i)
    z ^= q[i / 4][i % 4].x;
  if (ew >= 0) bar_arrive(ew + (z & c_z0), 288);
}
DEVI void ld_x(uint4 (&b)[2], const u8* p) {
  b[0] = *AS(const uint4, p);
  b[1] = *AS(const uint4, p + 16);
}
DEVI void mma_x(float* c, const u32 (&a)[4], const uint4 (&b)[2], int jj) {
  const u32 B[8] = {b[0].x, b[0].y, b[0].z, b[0].w, b[1].x, b[1].y, b[1].z, b[1].w};
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(B[2 * jj]), "r"(B[2 * jj + 1]));
}
DEVI void dec_kr(const uint4 (&qw)[2][4], const u32 (&R6)[8], int kk, int r, u32 (&A)[4][4]) {
  const uint4 q = qw[r][kk];
  const u32 R = R6[4 * r + kk];
  dq4(q.x, q.y, 0, R, A[0]);
  dq4(q.y, q.x, 1, R, A[1]);
  dq4(q.z, q.w, 0, R, A[2]);
  dq4(q.w, q.z, 1, R, A[3]);
}
DEVI void dec_k(const uint4 (&qw)[2][4], const u32 (&R6)[8], int kk, u32 (&A)[2][4][4]) {
  for (int r = 0; r < 2; ++r)
    dec_kr(qw, R6, kk, r, A[r]);
}

DEVI u32 tr32(u32 x) {
  const int lane = threadIdx.x & 31;
  u32 m = 0x0000FFFFu;
  for (int j = 16; j != 0; j >>= 1, m ^= m << j) {
    const u32 y = shfl_xor(x, j);
    x ^= (lane & j) ? (((y >> j) ^ x) & m) : ((((x >> j) ^ y) & m) << j);
  }
  return x;
}
#define FOR_ACC(B0)                                                                               \
  _Pragma("unroll") for (int a = 0; a < 2; ++a) _Pragma("unroll") for (int b = (B0); b < NT; ++b) \
      _Pragma("unroll") for (int c = 0; c < 4; ++c)
#define ZERO_ACC(B0)     \
  FOR_ACC(B0) {          \
    acc[a][b][c] = 0.f;  \
    acc2[a][b][c] = 0.f; \
  }
template <int NT>
DEVI void pin_acc(float (&acc)[2][NT][4]) {
  FOR_ACC(0) asm volatile("" : "+f"(acc[a][b][c])::"memory");
}

DEVI u64 sw128_dlo(u32 lo) {
  return (u64)lo | ((u64)64 << 32) | ((u64)1 << 62);
}
template <int NT, int NTL, bool RGC>
DEVI void stage_compute_wg(float (&acc)[2][NT][4], const u8* wbase, const u8* sclb, u32 xlo, int T, int ew) {
  uint4 qw[2][4];
  u32 R6[8];
  ld_stage(wbase, sclb, qw, R6);
  if constexpr (RGC) {
    u32 A[2][4][4];
    for (int kk = 0; kk < 4; ++kk) {
      for (int r = 0; r < 2; ++r) {
        dec_kr(qw, R6, kk, r, A[r]);
        wg_fence();
        for (int jj = 0; jj < 4; ++jj)
          wgmma_rs<8 * NTL>(acc[r][0], A[r][jj], sw128_dlo(xlo + kk * T * 8 + 2 * jj));
        wg_commit();
        if (kk + r == 1 && r) ew_arv(qw, R6, ew);
        wg_wait<1>();
      }
    }
    wg_wait<0>();
    pin_acc<NT>(acc);
    return;
  }
  const u64 xd = sw128_dlo(xlo);
  u32 A[2][2][4][4];
  for (int kk = 0; kk < 4; ++kk) {
    u32(&Ab)[2][4][4] = A[kk & 1];
    dec_k(qw, R6, kk, Ab);
    wg_fence();
    for (int jj = 0; jj < 4; ++jj)
      for (int r = 0; r < 2; ++r)
        wgmma_rs<8 * NTL>(acc[r][0], Ab[r][jj], xd + (u32)(kk * T * 8) + 2 * jj);
    wg_commit();
    wg_wait<1>();
  }
  wg_wait<0>();
  pin_acc<NT>(acc);
}

template <int NT, int NTL>
DEVI void stage_compute(float (&acc)[2][NT][4], const u8* wbase, const u8* sclb, const u8* xbase, int XP) {
  uint4 qw[2][4];
  u32 R6[8];
  ld_stage(wbase, sclb, qw, R6);
  uint4 bx[2][NTL][2];
  for (int nt = 0; nt < NTL; ++nt)
    ld_x(bx[0][nt], xbase + nt * 8 * XP);
  for (int kk = 0; kk < 4; ++kk) {
    if (kk < 3)
      for (int nt = 0; nt < NTL; ++nt)
        ld_x(bx[(kk + 1) & 1][nt], xbase + nt * 8 * XP + (kk + 1) * 128);
    u32 A[2][4][4];
    dec_k(qw, R6, kk, A);
    for (int nt = 0; nt < NTL; ++nt)
      for (int jj = 0; jj < 4; ++jj)
        for (int r = 0; r < 2; ++r)
          mma_x(acc[r][nt], A[r][jj], bx[kk & 1][nt], jj);
  }
}

template <int NTL>
struct DsCarry {
  u32 A[2][4][4];
  uint4 b[NTL][2];
};
template <int SPL, int NT, int NTL>
DEVI void ds_cmma(float (&acc)[2][NT][4], float (&acc2)[2][NT][4], const DsCarry<NTL>& c) {
  for (int nt = 0; nt < NTL; ++nt)
    for (int jj = 2; jj < 4; ++jj)
      for (int r = 0; r < 2; ++r)
        mma_x(SPL && (jj & 1) ? acc2[r][nt] : acc[r][nt], c.A[r][jj], c.b[nt], jj);
}
// last chunk's jj 2..3 MMAs are deferred (cr) to the next stage / ds_flush
template <int SPL, int NT, int NTL>
DEVI void stage_ds(
    float (&acc)[2][NT][4],
    float (&acc2)[2][NT][4],
    const u8* wbase,
    const u8* sclb,
    const u8* xbase,
    int XP,
    DsCarry<NTL>& cr,
    int ew) {
  uint4 qw[2][4];
  u32 R6[8];
  ld_stage(wbase, sclb, qw, R6);
  uint4 bx[4][NTL][2];
  for (int nt = 0; nt < NTL; ++nt)
    ld_x(bx[0][nt], xbase + nt * 8 * XP);
  ds_cmma<SPL, NT, NTL>(acc, acc2, cr);
  u32 A[4][2][4][4];
  for (int kk = 0; kk < 4; ++kk) {
    if (kk < 3)
      for (int nt = 0; nt < NTL; ++nt)
        ld_x(bx[kk + 1][nt], xbase + nt * 8 * XP + (kk + 1) * 128);
    dec_k(qw, R6, kk, A[kk]);
    for (int nt = 0; nt < NTL; ++nt)
      for (int jj = 0; jj < (kk == 3 ? 2 : 4); ++jj)
        for (int r = 0; r < 2; ++r)
          mma_x(SPL && (jj & 1) ? acc2[r][nt] : acc[r][nt], A[kk][r][jj], bx[kk][nt], jj);
    if (kk == 0) ew_arv(qw, R6, ew);
  }
  for (int r = 0; r < 2; ++r)
    for (int j = 0; j < 4; ++j)
      for (int e = 0; e < 4; ++e)
        cr.A[r][j][e] = A[3][r][j][e];
  for (int n = 0; n < NTL; ++n) {
    cr.b[n][0] = bx[3][n][0];
    cr.b[n][1] = bx[3][n][1];
  }
}
__host__ __device__ constexpr int ds_code(int rw, int k, int mw) {
  return rw == 8 && !(mw == 1 && k == 1) ? k : 0;
}
template <int C, int NT, int NTL>
DEVI void ds_flush(float (&acc)[2][NT][4], float (&acc2)[2][NT][4], const DsCarry<NTL>& cr) {
  ds_cmma<C == 2, NT, NTL>(acc, acc2, cr);
  if constexpr (C == 2)
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < NTL; ++b)
        for (int c = 0; c < 4; ++c) {
          acc[a][b][c] += acc2[a][b][c];
          acc2[a][b][c] = 0.f;
        }
}

#define PUT_INFO(f, a, b, c) \
  {                          \
    int4* fi = AS(int4, f);  \
    fi[0] = a;               \
    fi[1] = b;               \
    fi[2] = c;               \
  }
#define TILE_SIG(f) make_int2(f.phase, f.phase == 1 ? f.tile * 16 + f.rb * RB / 512 : f.tile)
DEVI void decode_item2(
    int item, int ntiles, int tsplit, int NRB1, int NRB2, int KSX, int& phase, int& tile, int& rb, int& ks, int& kss) {
  const int n1 = ntiles * NRB1;
  if (item < n1) {
    phase = 1;
    tile = item / NRB1;
    rb = item - tile * NRB1;
    ks = 0;
    kss = 1;
    return;
  }
  phase = 2;
  int j = item - n1;
  const int na = tsplit * NRB2;
  if (j < na) {
    kss = 1;
    tile = j / NRB2;
    rb = j - tile * NRB2;
    ks = 0;
  } else {
    kss = KSX;
    j -= na;
    const int t2 = j / (NRB2 * KSX);
    j -= t2 * NRB2 * KSX;
    tile = tsplit + t2;
    rb = j / KSX;
    ks = j - rb * KSX;
  }
}
struct Params {
  const u16* x;
  const int* ids;
  const float* topk_w;
  int M;
  const u32* w13;
  const float* gs13;
  const u32* w2;
  const float* gs2;
  const float* nw1;
  u16* inter;
  float* ypart;
  u16* out;
  int* ctr;
  int pmax;
  u16* xs;
  u16* yh;
  float* scr;
  const u16* xsp[8];
  u16* osp[8];
  int R;
  int* ptj;
  u8* sw13;
  u8* sw2;
  const int* gsrc;
  const u64* peer;
};
template <class T>
DEVI T sp_pick(T const (&a)[8], int r) {
  T v = a[0];
#pragma unroll
  for (int i = 1; i < 8; ++i)
    v = r == i ? a[i] : v;
  return v;
}

DEVI u32 xpack(u32 v0, u32 v1, int g) {
  const u32 rv = shfl_xor((g & 1) ? v0 : v1, 4);
  return (g & 1) ? (rv | (v1 << 16)) : (v0 | (rv << 16));
}
DEVI void st_bf4(u16* p, float4 v) {
  *AS(uint2, p) = make_uint2(bf2(v.x, v.y), bf2(v.z, v.w));
}
struct SpRows {
  u64 a, b;
};
DEVI SpRows sp_rows(const Params& P, int p0, int n) {
  const int lane = threadIdx.x & 31;
  u64 r[2] = {};
  for (int h = 0; h < 2; ++h)
    if (lane + 32 * h < n) {
      const int tj = __ldcg(P.ptj + p0 + lane + 32 * h), t = tj >> 3, j = tj & 7, o = t / P.R;
      r[h] = reinterpret_cast<u64>(sp_pick(P.osp, o) + ((size_t)j * P.R + (t - o * P.R)) * 4096);
    }
  return SpRows{r[0], r[1]};
}
DEVI u16* sp_row(const SpRows& r, int tok) {
  return AS(u16, shfl(tok < 32 ? r.a : r.b, tok & 31));
}
DEVI float4 sp_mul(const Params& P, int c, float gsc) {
  const float4 w = __ldg(AS(const float4, P.nw1 + c));
  return make_float4(w.x * gsc, w.y * gsc, w.z * gsc, w.w * gsc);
}
DEVI void sp_st4(u16* p, float4 v, float4 m) {
  v.x *= m.x;
  v.y *= m.y;
  v.z *= m.z;
  v.w *= m.w;
  st_bf4(p, v);
}
DEVI float4 ld_bf4(const u16* p) {
  const uint2 u = __ldcg(AS(const uint2, p));
  return make_float4(bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y));
}
DEVI float rcp_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
DEVI u32 swiglu_bf2(const float* rr, float gke, float gs2) {
  const float2 gv = *AS(const float2, rr), uv = *AS(const float2, rr + 8);
  const float g0 = gv.x, g1 = gv.y, u0 = uv.x, u1 = uv.y;
  const float e0 = ex2(g0 * gke), e1 = ex2(g1 * gke), r0 = rcp_ftz(1.f + e0), r1 = rcp_ftz(1.f + e1);
  return bf2((g0 * u0) * (gs2 * r0), (g1 * u1) * (gs2 * r1));
}
DEVI u32 swiglu1(float g0, float u0, float gke, float gs2) {
  return (u32)f2bf((g0 * u0) * (gs2 * rcp_ftz(1.f + ex2(g0 * gke))));
}
DEVI u32 swiglu_x(const float (&c)[4], float gke, float gs2, int g) {
  return xpack(swiglu1(c[0], c[2], gke, gs2), swiglu1(c[1], c[3], gke, gs2), g);
}

template <int KC>
__device__ __noinline__ void wp32_fst(
    const u32* bm,
    const u32* w13,
    const u16* x,
    int M,
    u8* info,
    u32 st0,
    u32 b0,
    int item,
    int nrb1,
    int ns,
    int stage,
    int wb,
    int sb,
    int srg,
    int rgs,
    int xp,
    int cmax) {
  const int lane = threadIdx.x & 31;
  const u32 bml = bm[lane];
  const int c = __popc(bml);
  const int tile0 = item / nrb1, rb0 = item - tile0 * nrb1;
  int key = c > 0 ? (c << 5) | (31 - lane) : -1;
  int p0 = 0, mx = -1;
#pragma unroll 1
  for (int k = 0; k <= tile0; ++k) {
    mx = __reduce_max_sync(~0u, key);
    if (mx < 0) break;
    if (k < tile0) {
      p0 += mx >> 5;
      if (lane == 31 - (mx & 31)) key = -1;
    }
  }
  if (mx < 0 || (mx >> 5) > cmax) return;
  const int e0 = 31 - (mx & 31);
  const u32 bme = shfl(bml, e0);
  const int n = __popc(bme);
  const int r = (lane < M && ((bme >> lane) & 1u)) ? __popc(bme & ((1u << lane) - 1u)) : -1;
  const u64 polw = policy_evict_first(), polx = policy_evict_last();
  const u8* wsrc = wc6_w(w13, e0, 512, 0, rb0 * rgs);
  const u8* ssrc = wc6_s(w13, e0, 512, 0, rb0 * rgs);
#pragma unroll 1
  for (int st = 0; st < ns; ++st) {
    const u32 full = b0 + st * 8;
    if (lane == 0) {
      PUT_INFO(info + st * 48, make_int4(item, 1, tile0, rb0), make_int4(0, st, 0, n), make_int4(p0, e0, 0, 0))
      mbar_expect(full, wb + sb + (u32)(n * 512 * KC));
      for (int j = 0; j < KC; ++j) {
        bulk_g2s(st0 + st * stage + j * rgs * 2048, wsrc + (size_t)(st * KC + j) * 32 * 35840, rgs * 2048, full, polw);
        bulk_g2s(
            st0 + st * stage + wb + j * rgs * srg, ssrc + (size_t)(st * KC + j) * 32 * 35840, rgs * srg, full, polw);
      }
    }
    __syncwarp();
    if (r >= 0)
      bulk_g2s(st0 + st * stage + wb + sb + r * xp, x + (size_t)lane * 4096 + st * 256 * KC, 512 * KC, full, polx);
  }
  __syncwarp();
}
template <int RW, int NT, int NS, int MW, bool SPX = false>
DEVI void moe_body(const Params& P) {
  static_assert(!SPX || (RW == 8 && NT >= 8), "SP: NT >= 8 only");
  // Item: RB = 32 RW rows x T = 8 NT tokens of one expert; M <= 32 MW. KW warps split K; HB: 128-row items; WSC /
  // LEANC: one copy per stage, back to back; IT2K: <= 32 tokens; GNEW: warps 9-11 gather x; BMT: bm word-major; CEPI:
  // producer-reserved signal slots, SP down (CE1: all) items store from registers; SPX: rows to token owners.
  constexpr int KW = 8 / RW;
  constexpr bool HB = RW == 4;
  constexpr bool WSC = KW == 1 && !(NT == 4 && MW == 1);
  constexpr bool LEANC = WSC;
  constexpr bool IT2K = NT == 4 && MW == 1;
  constexpr int RB = 32 * RW, RGS = RB / 16, KST = 256 * KW, T = 8 * NT, XP = KST * 2 + 16, WB = RGS * 2048 * KW;
  constexpr int WGMIN = KW == 1 ? 3 : 99;
  constexpr bool HASWG = WGMIN < 99;
  constexpr int SRG = 192, SB = RGS * SRG * KW;
  constexpr int XB = HB ? 16 * XP : T * XP;
  constexpr bool GNEW = !HB && NT >= 8, BMT = GNEW && MW >= 16, CEPI = BMT;
#define BMI(e, w) (BMT ? (w) * 32 + (e) : (e) * MW + (w))
  constexpr int GHC = SPX || MW > 8 ? 4 : 1, GUPT = (8 * NT + GHC - 1) / GHC, GTC_OFF = 4624, DG_CTR = 6000;
  constexpr int STAGE = HASWG ? (WB + SB + XB + 1023) / 1024 * 1024 : (WB + SB + XB + 127) / 128 * 128;
  constexpr int IT2XOFF = WB + SB + 8 * XP;
  constexpr int MOE_WPA_BUFB = NT == 4 ? (MW == 1 ? 16384 : 15360) : (MW <= 8 ? 13312 : 0);
  static_assert(!IT2K || (IT2XOFF % 128 == 0 && IT2XOFF + 8 * (RB + 4) * 4 <= STAGE));
  constexpr int NRB1 = 8192 / RB, NRB2 = 4096 / RB;
  constexpr int ST1 = 16 / KW, MAXP = MW * 32 * 8, MAXT = 32 + MAXP / T;
  constexpr int KSX = HB ? 2 : 4, STX = 16 / (KW * KSX);
  constexpr bool CE1 = MW >= 64 || T * (RB + 4) * 4 > STAGE;
  constexpr bool EWR = SPX && NS == 2 && RW == 8 && MW >= 16;
  constexpr bool MOE_L2_CLDV = (LEANC && MW >= 4) || HB;
  constexpr int NSIG = 4;
  extern __shared__ __align__(128) u8 smem[];
  static_assert(NS <= 4);
  // bars: stage full; epi / sfree: signal posted / taken; aq_full / aq_slot: epilogue queue; one-shot rdy, rtd, plr,
  // eis: x gathered, routing / pair list done, producer start; sinfo: (phase, counter); gsm: scales x 256.
  // scal: 0 tiles, 2 items, 3 pairs, 4 / 5 first k-split tile / pair, 6 gather units,
  // 8-10 and 12-13 claim broadcasts, 14 counter bank, 15 guests' phase-2 tiles.
  // bm[BMI(e, w)] bit t: token 32 w + t -> slot e; tiles: (slot, first pair, tokens, first unit)
  u64* bars = AS(u64, smem);
  u64* epi = AS(u64, smem + 64);
  u64* sfree = AS(u64, smem + 96);
  StageInfo* info = AS(StageInfo, smem + 128);
  int2* sinfo = AS(int2, smem + 320);
  int* scal = AS(int, smem + 352);
  int* ftile = AS(int, smem + 416);
  int* off = AS(int, smem + 544);
  const u32 rdy = smem_u32(smem + 672);
  const u32 rtd = smem_u32(smem + 688);
  const u32 plr = smem_u32(smem + 752);
  const u32 eis = smem_u32(smem + 696);
  float* gsm = AS(float, smem + 768);
  u64* aq_full = AS(u64, smem + 704);
  int* aq_slot = AS(int, smem + 736);
  u32* bm = AS(u32, smem + 1024);
  int* wpre = AS(int, bm + 32 * MW);
  int4* tiles = AS(int4, wpre + 32 * MW);
  constexpr bool PAIRS = HB || NT < 8;
  u16* pair_tok = AS(u16, tiles + MAXT);
  signed char* lid = AS(signed char, pair_tok + (PAIRS ? MAXP : 0));
  constexpr int ROUTE_BYTES = (32 * MW * 8 + MAXT * 16 + MAXP * (PAIRS ? 3 : 1) + 127) / 128 * 128;
  constexpr int STB = (1024 + ROUTE_BYTES + 1023) / 1024 * 1024;
  // 4 KB of SP epilogue scratch behind the ring when it fits
  constexpr bool EPI2 = SPX && CEPI && STB + NS * STAGE + 4096 <= 232448;
  u8* stages = smem + STB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int M = P.M;
  constexpr bool FT_GSDV = IT2K, MOE_FSTV = IT2K;
  constexpr int PAR_OFF = 65536, PAR_WORD = PAR_OFF + 16384;
  int fi0 = 0, gc0 = 0;
  pdl_wait();
  if (tid == 256) fi0 = atomicAdd(P.ctr + PAR_WORD + 32, 1);
  int par_r = 0;
  if (tid == 0) par_r = __ldcg(P.ctr + PAR_WORD);
  int* CB = P.ctr;
  int4 hv0 = {}, hv1 = {};
  if (warp * 32 + lane < M) LD_IDS(P.ids, warp * 32 + lane, hv0, hv1)
  constexpr int NIT = BMT ? (MW * 32 + 383) / 384 : 1;
  int4 hvx[NIT][2];
  if constexpr (BMT) {
    for (int k = 1; k < NIT; ++k) {
      const int t = warp * 32 + k * 384 + lane;
      hvx[k][0] = hvx[k][1] = {};
      if (t < M) LD_IDS(P.ids, t, hvx[k][0], hvx[k][1])
    }
  }
  const float hgs = tid < 64 ? (tid < 32 ? P.gs13[tid] : P.gs2[tid - 32]) : 0.f;  // scaled at use
  for (int i = tid; i < 32 * MW; i += blockDim.x)
    bm[i] = 0u;
  if (tid == 352) {
    for (int s = 0; s < NS; ++s)
      mbar_init(smem_u32(&bars[s]), 1);
    for (int s = 0; s < NSIG; ++s) {
      mbar_init(smem_u32(&epi[s]), 8);
      mbar_init(smem_u32(&sfree[s]), 1);
    }
    for (int s = 0; s < NS; ++s)
      mbar_init(smem_u32(&aq_full[s]), 8);
    if constexpr (MOE_WPA_BUFB > 0) mbar_init(smem_u32(stages + NS * STAGE + MOE_WPA_BUFB - 16), 1);
    mbar_init(rdy, 1);
    mbar_init(rtd, 1);
    mbar_init(eis, 1);
    mbar_init(plr, 1);
    mbar_init_fence();
    fence_async();
  }
  __syncthreads();
  constexpr bool DGS = SPX && CE1;
  const bool dg = DGS && P.gsrc;
  u32 gsl = ~0u;
  if (dg) {
    u32 occ = 0;
    for (int k = 0; k < NIT; ++k) {
      const int t = warp * 32 + k * 384 + lane;
      if (warp * 32 + k * 384 >= MW * 32) break;
      if (t < M) {
        const int4 v0 = k == 0 ? hv0 : hvx[k][0], v1 = k == 0 ? hv1 : hvx[k][1];
        const int ev[8] = EV8(v0, v1);
        for (int j = 0; j < 8; ++j)
          occ |= (unsigned)ev[j] < 32u ? 1u << ev[j] : 0u;
      }
    }
    if (tid == 0) scal[13] = 0;
    __syncthreads();
    occ = __reduce_or_sync(~0u, occ);
    if (lane == 0 && occ) atomicOr(scal + 13, (int)occ);
    __syncthreads();
    const u32 fr = ~scal[13], f0 = __ffs(fr) - 1, f1 = __ffs(fr & (fr - 1)) - 1;
    gsl = (__ldg(P.gsrc) >= 0 ? f0 & 0xffu : 0xffu) | (__ldg(P.gsrc + 1) >= 0 ? f1 & 0xffu : 0xffu) << 8 | 0xffff0000u;
  }
  for (int k = 0; k < NIT; ++k) {
    const int t0 = warp * 32 + k * 384;
    if (t0 >= MW * 32) break;
    const int t = t0 + lane;
    u32 em = 0;
    if (t < M) {
      const int4 v0 = k == 0 ? hv0 : hvx[k][0], v1 = k == 0 ? hv1 : hvx[k][1];
      const int ev[8] = EV8(v0, v1);
      for (int j = 0; j < 8; ++j) {
        int e = ev[j];
        if (dg && (unsigned)(e - 32) < 2u) e = (gsl >> 8 * (e - 32)) & 0xff;
        const bool loc = (unsigned)e < 32u;
        lid[t * 8 + j] = loc ? (signed char)e : (signed char)-1;
        em |= loc ? 1u << e : 0u;
      }
    }
    bm[BMI(lane, t0 >> 5)] = tr32(em);
  }
  if (tid < 64) {
    float g = hgs * (FT_GSDV ? 1.f : 256.f);
    const int gj = dg ? gix(gsl, tid & 31) : -1;
    if (gj >= 0) {
      const int x = __ldg(P.gsrc + gj);
      g = AS(const float, __ldg(P.peer + 4 * (x >> 5) + 2 + (tid >> 5)))[x & 31] * 256.f;
    }
    gsm[tid] = FT_GSDV ? g * 256.f : g;
  }
  if (tid == 0) scal[14] = par_r;
  __syncthreads();
  // P.ctr: two counter banks for alternate launches; each zeroes the other and its last CTA at CB + 6 flips the parity.
  {
    const int par = scal[14];
    CB = P.ctr + par * PAR_OFF;
    if (NT >= 8 ? tid == 288 : tid == 0) gc0 = atomicAdd(CB + 7, 1);
    int* OB = P.ctr + (1 - par) * PAR_OFF;
    for (int i = blockIdx.x * (int)blockDim.x + tid; i < 8192 + 256; i += (int)gridDim.x * (int)blockDim.x)
      OB[i] = 0;
    if (warp == 9 && lane == 0) {
      if (atomicAdd(CB + 6, 1) == (int)gridDim.x - 1) atomicExch(P.ctr + PAR_WORD, 1 - par);
    }
  }
  if constexpr (MOE_FSTV) {
    static_assert(sizeof(StageInfo) == 48 && ST1 > NS, "wp32_fst layout");
    if (warp == 8) {
      bar_arrive(2, 384);
      wp32_fst<KW>(
          bm,
          P.w13,
          P.x,
          M,
          AS(u8, info),
          smem_u32(stages),
          smem_u32(bars),
          shfl(fi0, 0),
          NRB1,
          NS,
          STAGE,
          WB,
          SB,
          SRG,
          RGS,
          XP,
          HASWG ? 8 * (WGMIN - 1) : 32);
    }
  }
  if (warp == 0) {
    int c = 0;
    u32 bw[MW];
    for (int w = 0; w < MW; ++w)
      bw[w] = bm[BMI(lane, w)];
    for (int w = 0; w < MW; ++w) {
      wpre[BMI(lane, w)] = c;
      c += __popc(bw[w]);
    }
    const int nt = (c + T - 1) / T;
    int rank = key_rank(DGS ? c | (dg && gix(gsl, lane) >= 0 ? 0 : 1 << 12) : c, DGS ? 12 : 9);
    if constexpr (NT >= 8 && (MW <= 8 || SPX)) rank = mix_rank(rank, c);
    const int nu = GNEW && c > 0 ? (nt - 1) * GUPT + (c - (nt - 1) * T + GHC - 1) / GHC : 0;
    int4* rk = AS(int4, tiles);
    rk[rank] = make_int4(c, nt, nu, 0);
    __syncwarp();
    const int4 mine = rk[lane];
    int sc = mine.x, st = mine.y, su = mine.z;
    for (int d = 1; d < 32; d <<= 1) {
      const int a = shfl_up(sc, d), b = shfl_up(st, d), u = GNEW ? shfl_up(su, d) : 0;
      if (lane >= d) {
        sc += a;
        st += b;
        su += u;
      }
    }
    const int tu = shfl(su, 31), tot_t = shfl(st, 31), tot_p = shfl(sc, 31);
    __syncwarp();
    rk[lane] = make_int4(sc - mine.x, st - mine.y, su - mine.z, 0);
    __syncwarp();
    const int4 ex = rk[rank];
    const int o = ex.x, to = ex.y, uo = ex.z;
    __syncwarp();
    ftile[lane] = to;
    off[lane] = o;
    for (int k = 0; k < nt; ++k)
      tiles[to + k] = make_int4(lane, o + k * T, min(T, c - k * T), GNEW ? uo + k * GUPT : 0);
    if (GNEW && lane == 0) scal[6] = tu;
    // phase 2 takes the guests' tiles (last in phase 1, behind the guest weight pull) after two waves
    if constexpr (DGS) {
      const int ngt = __reduce_add_sync(~0u, dg && gix(gsl, lane) >= 0 ? nt : 0);
      if (lane == 0) scal[15] = min(tot_t - ngt, (2 * (int)gridDim.x + NRB2 - 1) / NRB2) | ngt << 16;
    }
    __syncwarp();
    if (lane == 0) {
      const int nspl = (!SPX && MW < 32) ? min(tot_t,
                                               (int)ceilf(
                                                   (HB        ? 0.75f
                                                    : NT >= 8 ? 0.6f
                                                              : 1.0f) *
                                                   gridDim.x / NRB2))
                                         : 0;
      const int tsplit = tot_t - nspl;
      scal[0] = tot_t;
      scal[2] = NRB1 * tot_t + NRB2 * tsplit + NRB2 * KSX * nspl;
      scal[3] = tot_p;
      scal[4] = tsplit;
      scal[5] = nspl > 0 ? tiles[tsplit].y : tot_p;
    }
  }
  if constexpr (MOE_FSTV) {
    if (warp != 8) bar_sync(2, 384);
  } else
    __syncthreads();
  if (tid == 0) mbar_arrive(rtd);
  constexpr bool MOE_XDIRV = NT == 4, MOE_XRELV = NT == 4, MOE_RT_WRELV = NT == 4 && MW > 1;
  if (warp < 8) {
#pragma unroll 1
    for (int i0 = tid; PAIRS && i0 < M * 8; i0 += 1024) {
      int ev[4], pv[4];
      for (int u = 0; u < 4; ++u)
        ev[u] = i0 + 256 * u < M * 8 ? lid[i0 + 256 * u] : -1;
      for (int u = 0; u < 4; ++u) {
        const int t = (i0 + 256 * u) >> 3;
        pv[u] = 0;
        if (ev[u] >= 0) {
          const u32 word = bm[ev[u] * MW + (t >> 5)];
          pv[u] = off[ev[u]] + wpre[ev[u] * MW + (t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));
        }
      }
      for (int u = 0; u < 4; ++u)
        if (ev[u] >= 0) pair_tok[pv[u]] = (i0 + 256 * u) >> 3;
    }
    consumer_sync();
    if (tid == 0) mbar_arrive(plr);
    if constexpr (!HB && !GNEW) {
      const int np_all = scal[3], gch = (np_all + (int)gridDim.x - 1) / (int)gridDim.x;
      const int gnc = gch > 0 ? (np_all + gch - 1) / gch : 0;
      int tile = 0;
      if (tid == 0) scal[9] = gc0;
      consumer_sync();
      int gc = scal[9];
      for (;;) {
        consumer_sync();
        if (gc >= gnc) break;
        if (tid == 0) scal[9] = atomicAdd(CB + 7, 1);
        const int pb = gc * gch, pe = min(np_all, pb + gch);
        for (int pg = pb; pg < pe; pg += 4) {
          uint4 val[4][2];
          int tl[4], ri[4];
          bool wg[4];
          for (int r = 0; r < 4; ++r) {
            const int p = pg + r;
            tl[r] = 0;
            ri[r] = 0;
            wg[r] = false;
            if (p < pe) {
              while (tile + 1 < scal[0] && tiles[tile + 1].y <= p)
                ++tile;
              tl[r] = tile;
              ri[r] = p - tiles[tile].y;
              wg[r] = (tiles[tile].z + 7) / 8 >= WGMIN;
              const u16* srow = P.x + (size_t)pair_tok[p] * 4096;
              for (int h = 0; h < 2; ++h) {
                const int ch = tid + 256 * h;
                if (wg[r]) {
                  const int c64 = ch >> 3, pc = ch & 7, lc = pc ^ (ri[r] & 7);
                  u32 v[4];
                  for (int e2 = 0; e2 < 4; ++e2) {
                    const int qp = lc * 8 + e2 * 2, jj = qp >> 4, pp = qp & 15;
                    const int k = c64 * 64 + 16 * ((pp & 7) >> 1) + 4 * jj + ((pp >> 3) << 1);
                    v[e2] = __ldg(AS(const u32, srow + k));
                  }
                  val[r][h] = make_uint4(v[0], v[1], v[2], v[3]);
                } else
                  val[r][h] = __ldg(AS(const uint4, srow) + ch);
              }
            }
          }
          for (int r = 0; r < 4; ++r)
            if (pg + r < pe)
              for (int h = 0; h < 2; ++h) {
                const int ch = tid + 256 * h;
                if (wg[r]) {
                  const int c64 = ch >> 3, pc = ch & 7;
                  u16* dtile = P.xs + (size_t)tl[r] * ST1 * T * XP / 2;
                  *AS(uint4, dtile + ((size_t)c64 * T + ri[r]) * 64 + pc * 8) = val[r][h];
                } else {
                  u8* dst = AS(u8, P.xs) + ((size_t)tl[r] * ST1 * T + ri[r]) * XP;
                  const int st = ch / (KST / 8), w = ch - st * (KST / 8);
                  *AS(uint4, dst + (size_t)st * T * XP + w * 16) = val[r][h];
                }
              }
        }
        consumer_sync();
        if (tid == 0) red_release(CB + 3, 1);
        gc = scal[9];
      }
    }
    if (tid == 0) mbar_arrive(rdy);
  }
  if (MOE_FSTV && warp == 8) {
    if (lane == 0) mbar_sleep(rtd, 0);
    __syncwarp();
  }
  int ntiles = scal[0], total = scal[2], tsplit = scal[4], psplit = scal[5];
  int* head = P.ctr + PAR_WORD + 32;
  int* cnt1 = CB + 8;     // [16 tile + c]: gate/up items done for intermediate chunk c
  int* cnt2 = CB + 8192;  // [tile]: down items done
  if (warp >= 8) {
    nreg_dec<72>();
    if constexpr (GNEW) {
      if (warp >= 9) {
        const int gt = tid - 288;
        if (lane == 0) mbar_poll(rtd, 0);
        __syncwarp();
        const int NU = scal[6], ntl = scal[0];
        int gtl = 0;
        const int g0 = gt >> 3, pcp = gt & 7;
        if (gt == 0) scal[10] = gc0;
        bar_sync(3, 96);
        int u = scal[10];
        while (u < NU) {
          int nxt = 0;
          if (gt == 0) nxt = atomicAdd(CB + 7, 1);
          while (gtl + 1 < ntl && tiles[gtl + 1].w <= u)
            ++gtl;
          const int tile = gtl;
          const int4 tv = tiles[tile];
          const int r0 = (u - tv.w) * GHC, nr = min(GHC, tv.z - r0);
          int tok = 0, jr = 0;
          if (lane < nr) {
            const int e = tv.x, pr = tv.y - off[e] + r0 + lane;
            int w0 = 0;
            for (int s = MW / 2; s > 0; s >>= 1)
              if (wpre[BMI(e, w0 + s)] <= pr) w0 += s;
            u32 word = bm[BMI(e, w0)];
            int nb = pr - wpre[BMI(e, w0)], pos = 0;
            for (int s = 16; s > 0; s >>= 1) {
              const int cl = __popc(word & ((1u << s) - 1u));
              if (nb >= cl) {
                nb -= cl;
                word >>= s;
                pos += s;
              }
            }
            tok = w0 * 32 + pos;
            if constexpr (SPX)
              for (int j = 0; j < 8; ++j)
                if (lid[tok * 8 + j] == e) jr = j;
          }
          const bool wgt = (tv.z + 7) / 8 >= WGMIN;
          for (int r = 0; r < nr; ++r) {
            const int ri = r0 + r, tq = shfl(tok, r);
            const u16* rp;
            if constexpr (SPX) {
              const int oq = tq / P.R;
              rp = sp_pick(P.xsp, oq) + (size_t)(tq - oq * P.R) * 4096;
              const int jq = shfl(jr, r);
              if (gt == 0) P.ptj[tv.y + ri] = tq * 8 + jq;
            } else
              rp = P.x + (size_t)tq * 4096;
            uint4 v[6];
            if (wgt) {
              const u32* sw = AS(const u32, rp) + g0 * 32 + (pcp ^ (ri & 7));
              uint4* dst = AS(uint4, P.xs + (size_t)tile * ST1 * T * XP / 2 + ((size_t)g0 * T + ri) * 64) + pcp;
              for (int i = 0; i < 6; ++i)
                if (g0 + 12 * i < 64) {
                  const u32* q = sw + i * 12 * 32;
                  v[i] = make_uint4(__ldg(q), __ldg(q + 8), __ldg(q + 16), __ldg(q + 24));
                }
              for (int i = 0; i < 6; ++i)
                if (g0 + 12 * i < 64) dst[(size_t)i * 96 * T] = v[i];
            } else {
              u8* dst = AS(u8, P.xs) + ((size_t)tile * ST1 * T + ri) * XP + (size_t)(gt >> 5) * T * XP + (gt & 31) * 16;
              for (int i = 0; i < 6; ++i)
                if (gt + 96 * i < 512) v[i] = __ldg(AS(const uint4, rp) + gt + 96 * i);
              for (int i = 0; i < 6; ++i)
                if (gt + 96 * i < 512) *AS(uint4, dst + (size_t)i * 3 * T * XP) = v[i];
            }
          }
          bar_sync(3, 96);
          if (gt == 0) {
            red_release(CB + GTC_OFF + tile, 1);
            scal[10] = nxt;
          }
          bar_sync(3, 96);
          u = scal[10];
        }
      }
    }
    if (dg && warp >= 10) {
      const int nwb = ((__ldg(P.gsrc) >= 0) + (__ldg(P.gsrc + 1) >= 0)) * (512 + 256);
      for (;;) {
        int k = 0;
        if (lane == 0) k = atomicAdd(CB + DG_CTR + 4, 1);
        k = shfl(k, 0);
        if (k >= nwb) break;
        const int j = k / 768, w = k - j * 768, k2 = w >= 512, b = k2 ? w - 512 : w, nb = k2 ? 256 : 512,
                  x = __ldg(P.gsrc + j);
        const uint4* src =
            AS(const uint4, AS(const u8, __ldg(P.peer + 4 * (x >> 5) + k2)) + ((size_t)(x & 31) * nb + b) * 35840) +
            lane;
        uint4* dst = AS(uint4, (k2 ? P.sw2 : P.sw13) + ((size_t)j * nb + b) * 35840) + lane;
#pragma unroll 1
        for (int i0 = 0; i0 < 70; i0 += 7) {
          uint4 v[7];
          for (int i = 0; i < 7; ++i)
            v[i] = __ldcs(src + 32 * (i0 + i));
          for (int i = 0; i < 7; ++i)
            dst[32 * (i0 + i)] = v[i];
        }
        __syncwarp();
        if (lane == 0) red_release(CB + DG_CTR + 2 * j + k2, 1);
      }
    }
    if (warp >= 10) {
      const int et = tid - 320;
      int q_sig = 0;
      u32 tmc_ph = 0u;
      for (int qe = 0;; ++qe) {
        mbar_sleep(smem_u32(&aq_full[qe % NS]), (qe / NS) & 1);
        const int aqr = aq_slot[qe % NS];
        const int sl = CEPI ? (aqr & 0xFF) - 1 : aqr;
        const int qsg = CEPI ? (aqr >> 8) : q_sig;
        const int j = qsg % NSIG;
        if (sl >= 0) {
          const StageInfo f = info[sl];
          constexpr int SP = RB + 4;
          auto wpa_g1 = [&](const StageInfo& f, const float* src, int tlo, int thi) {
            const int nn = f.n, ntl = (nn + 7) >> 3;
            const float gsc = gsm[f.e] * (IT2K ? 64.f : 1.f);
            const float gke = -1.4426950408889634f * gsc, gs2 = gsc * gsc;
            auto ld2 = [&](int tok, int row) { return swiglu_bf2(src + tok * SP + row, gke, gs2); };
            if (HASWG && ntl >= WGMIN) {
              u16* dt = P.inter + (size_t)f.tile * ST1 * T * XP / 2;
              const int len = min(thi, nn) - tlo, pc = et & 7;
              for (int o = et; o < len * (RB / 16); o += 64) {
                const int rest = o >> 3, c64l = rest >= len ? 1 : 0, tok = tlo + rest - (c64l ? len : 0);
                const int lc = pc ^ (tok & 7);
                u32 h[4];
                for (int e2 = 0; e2 < 4; ++e2) {
                  const int il = 64 * c64l + 16 * e2 + 4 * (lc >> 1) + 2 * (lc & 1);
                  h[e2] = ld2(tok, 16 * (il >> 3) + (il & 7));
                }
                *AS(uint4, dt + ((size_t)((RB / 128) * f.rb + c64l) * T + tok) * 64 + pc * 8) =
                    make_uint4(h[0], h[1], h[2], h[3]);
              }
            } else {
              const int te = min(thi, nn);
              for (int o = et + tlo * (RB / 16); o < te * (RB / 16); o += 64) {
                const int tok = o / (RB / 16), j8 = o - tok * (RB / 16);
                u32 h[4];
                for (int q = 0; q < 4; ++q)
                  h[q] = ld2(tok, 16 * j8 + 2 * q);
                const int ig = (RB / 2) * f.rb + 8 * j8;
                *AS(uint4, AS(u8, P.inter) + (((size_t)f.tile * ST1 + ig / KST) * T + tok) * XP + (ig % KST) * 2) =
                    make_uint4(h[0], h[1], h[2], h[3]);
              }
            }
          };
          auto wpa_g2 = [&](const StageInfo& f, const float* src, int tlo, int thi) {
            const float gsc = gsm[32 + f.e] * (IT2K ? 64.f : 1.f);
            const bool full_k = f.tile < tsplit;
            float* yb = P.ypart + (size_t)f.ks * P.pmax * 4096 + (size_t)f.p0 * 4096 + f.rb * RB;
            const int te = min(thi, f.n);
            if (SPX) {
              const SpRows rows = sp_rows(P, f.p0, f.n);
              const float4 mul = sp_mul(P, f.rb * RB + 4 * et, gsc);
              for (int tok = tlo; tok < te; ++tok)
                sp_st4(sp_row(rows, tok) + f.rb * RB + 4 * et, *AS(const float4, src + tok * SP + 4 * et), mul);
            } else if (full_k) {
              for (int o = et + tlo * (RB / 4); o < te * (RB / 4); o += 64) {
                const int tok = o / (RB / 4), r4 = o - tok * (RB / 4);
                float4 v = *AS(const float4, src + tok * SP + 4 * r4);
                v.x *= gsc;
                v.y *= gsc;
                v.z *= gsc;
                v.w *= gsc;
                st_bf4(P.yh + (size_t)(f.p0 + tok) * 4096 + f.rb * RB + 4 * r4, v);
              }
            } else
              for (int o = et + tlo * (RB / 4); o < te * (RB / 4); o += 64) {
                const int tok = o / (RB / 4), r4 = o - tok * (RB / 4);
                float4 v = *AS(const float4, src + tok * SP + 4 * r4);
                v.x *= gsc;
                v.y *= gsc;
                v.z *= gsc;
                v.w *= gsc;
                *AS(float4, yb + (size_t)tok * 4096 + 4 * r4) = v;
              }
          };
#define MOE_WPA(...)                   \
  {                                    \
    if (f.phase == 1 || (CEPI && SPX)) \
      wpa_g1(f, __VA_ARGS__);          \
    else                               \
      wpa_g2(f, __VA_ARGS__);          \
  }
          if constexpr (MOE_WPA_BUFB > 0) {
            u8* const wbuf = stages + NS * STAGE;
            const float* const stg = AS(const float, stages + sl * STAGE + (IT2K && f.n <= 8 ? IT2XOFF : 0));
            int nn = f.n;
            const int nb = min(nn, (MOE_WPA_BUFB - 16) / (SP * 4));
            const u32 bytes = nb * (SP * 4), cb = smem_u32(wbuf + MOE_WPA_BUFB - 16);
            bar_sync(15, 64);
            float* gscr = P.scr + (size_t)blockIdx.x * (64 * 1040 / 4);
            // no same-CTA smem->smem bulk copy
            for (int v = et; v < (int)(bytes / 16); v += 64)
              AS(uint4, wbuf)[v] = AS(const uint4, stg)[v];
            bar_sync(15, 64);
            if (et == 0) {
              if (MW >= 4 && nn > nb) {
                for (int t0 = nb; t0 < nn; t0 += nb) {
                  const u32 cbytes = min(nb, nn - t0) * (SP * 4);
                  bulk_s2g(gscr + (size_t)(t0 - nb) * SP, smem_u32(stg + t0 * SP), cbytes);
                }
                bulk_commit_wait_read();
              }
            }
            if constexpr (MW >= 4) {
              if (nn > nb) bar_sync(15, 64);
            } else if (nn > nb)
              MOE_WPA(stg, nb, 1 << 20)
            __syncwarp();
            bar_arrive(4 + sl, 96);
            const float* cpy = AS(const float, wbuf);
            MOE_WPA(cpy, 0, nb)
            if constexpr (MW >= 4)
              if (nn > nb) {
                if (et == 0) bulk_wait();
                for (int t0 = nb; t0 < nn; t0 += nb) {
                  const u32 cbytes = min(nb, nn - t0) * (SP * 4);
                  bar_sync(15, 64);
                  if (et == 0) {
                    mbar_expect(cb, cbytes);
                    bulk_g2s(smem_u32(wbuf), gscr + (size_t)(t0 - nb) * SP, cbytes, cb);
                  }
                  mbar_sleep(cb, tmc_ph);
                  tmc_ph ^= 1u;
                  const float* ck = cpy - (size_t)t0 * SP;
                  MOE_WPA(ck, t0, t0 + nb)
                }
              }
          } else {
            MOE_WPA(AS(const float, stages + sl * STAGE), 0, 1 << 20) __syncwarp();
            bar_arrive(4 + sl, 96);
          }
          if (et == 0) {
            mbar_sleep(smem_u32(&sfree[j]), ((qsg / NSIG) & 1) ^ 1);
            sinfo[j] = TILE_SIG(f);
          }
        } else if (et == 0) {
          mbar_sleep(smem_u32(&sfree[j]), ((qsg / NSIG) & 1) ^ 1);
          sinfo[j] = make_int2(-1, 0);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive_n(smem_u32(&epi[j]), 4);
        ++q_sig;
        if (sl < 0) break;
      }
      return;
    }
    // Slot s frees at bar 4 + s: consumers + producer (288), or epilogue + producer (96) after a queued last stage.
    if (warp == 8) {
      const u64 pol_w = policy_evict_first(), pol_x = policy_evict_last();
      const u32 stage0 = smem_u32(stages), bars0 = smem_u32(bars);
      int it = 0;
      bool gathered = false;
      int gtile = -1, pq = 0;
      fi0 = shfl(fi0, 0);
      int item = fi0;
      const int gch8 = (scal[3] + (int)gridDim.x - 1) / (int)gridDim.x;
      const int gnch = gch8 > 0 ? (scal[3] + gch8 - 1) / gch8 : 0;
      int nb_le = -1000;
      for (;;) {
        if (item >= total) {
          const int s = it % NS;
          if (it >= NS) bar_sync(4 + s, it - NS == nb_le ? 96 : 288);
          if (lane == 0) {
            info[s].item = -1;
            mbar_arrive(smem_u32(&bars[s]));
            if (atomicAdd(head + 1, 1) == (int)gridDim.x - 1) {
              atomicExch(head, 0);
              atomicExch(head + 1, 0);
            }
          }
          break;
        }
        int next;
#define MOE_L2_CLAIM()                                                                                \
  asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; @p atom.global.inc.u32 %0, [%2], 0x7fffffff; }" \
               : "=r"(next)                                                                           \
               : "r"(lane), "l"(head)                                                                 \
               : "memory")
        if constexpr (!MOE_L2_CLDV)
          asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; @p atom.global.add.u32 %0, [%2], 1; }"
                       : "=r"(next)
                       : "r"(lane), "l"(head)
                       : "memory");
        int phase, tile, rb, ks, kss;
        decode_item2(item, ntiles, tsplit, NRB1, NRB2, KSX, phase, tile, rb, ks, kss);
        if (DGS && phase == 2) {
          const int q = scal[15] & 0xffff, g = scal[15] >> 16;
          if (tile >= q) tile = tile < q + g ? ntiles - g + tile - q : tile - g;
        }
        if constexpr (CEPI) {
          if (lane == 0) mbar_sleep(smem_u32(&sfree[pq % NSIG]), ((pq / NSIG) & 1) ^ 1);
          __syncwarp();
          ++pq;
        }
        const int nst = phase == 2 && kss == KSX ? STX : ST1;
        int ccl = MOE_L2_CLDV && nst >= 8 ? nst - 4 : 0;
        if (MOE_L2_CLDV && ccl <= 0) {
          MOE_L2_CLAIM();
          ccl = 1 << 20;
        }
        const int4 tl = tiles[tile];
        const int e = tl.x, p0 = tl.y, n = tl.z;
        const bool wgl = HASWG && (n + 7) / 8 >= WGMIN;
        const bool epi_item = !(CEPI && SPX) || !(CE1 || phase == 2);
        const int nrg_tot = phase == 1 ? 512 : 256;
        const int kc_item = phase == 1 ? 0 : ks * (kss == KSX ? 16 / KSX : 16);
        const u16* xrow0 = nullptr;
        u32 rmask = 0;
        const int pkw = lane >> 1;
        const bool pscl = lane & 1;
        const void* wcb = phase == 1 ? P.w13 : P.w2;
        const u8* csrc =
            pscl ? wc6_s(wcb, e, nrg_tot, kc_item + pkw, rb * RGS) : wc6_w(wcb, e, nrg_tot, kc_item + pkw, rb * RGS);
        const size_t cstep = (size_t)KW * (nrg_tot >> 4) * 35840;
        u32 cdst = pscl ? WB + pkw * RGS * SRG : pkw * RGS * 2048;
        u32 cbytes = pscl ? RGS * SRG : RGS * 2048;
        u32 lrdy = 0;
        const bool xdir = MOE_XDIRV && (HB || item == fi0) && phase == 1 && !wgl;
        if (xdir) {
          if (lane == 0) mbar_sleep(plr, 0);
          __syncwarp();
          if (lane < n) xrow0 = P.x + (size_t)pair_tok[p0 + lane] * 4096;
        }
        if constexpr (GNEW) {
          bool rd = gtile == tile;
          if (phase == 1 && !rd) {
            int ok = 0;
            if (lane == 0) ok = ld_acquire(CB + GTC_OFF + tile) >= (n + GHC - 1) / GHC;
            ok = shfl(ok, 0);
            if (ok) {
              __syncwarp();
              fence_async_global();
              gtile = tile;
              rd = true;
            }
          }
          lrdy = phase == 1 && rd ? ~0u : 0u;
        } else if constexpr (LEANC)
          lrdy = phase == 1 && gathered && !xdir ? ~0u : 0u;
        const int nrx = phase == 1 ? 32 : 16;
        const int gj = dg ? gix(gsl, e) : -1, ew = gj >= 0 ? gj : e;
        const u8* wsrc = gj >= 0 ? (phase == 1 ? P.sw13 : P.sw2) : AS(const u8, wcb);
        if (gj >= 0) {
          if (lane == 0)
            while (ld_acquire(CB + DG_CTR + 2 * gj + phase - 1) < (phase == 1 ? 512 : 256))
              __nanosleep(256);
          __syncwarp();
          fence_async_global();
        }
        auto wblock = [&](int kc) { return wsrc + ((size_t)(ew * 16 + kc) * nrx + rb) * (WB + SB); };
        int st_first = 0;
        if (MOE_FSTV && item == fi0 && phase == 1) {
          if (xdir) {
            st_first = NS;
            it += NS;
            csrc += NS * cstep;
          }
        }
        for (int st = st_first; st < nst; ++st, ++it) {
          if (MOE_L2_CLDV && st >= ccl) {
            MOE_L2_CLAIM();
            ccl = 1 << 20;
          }
          static_assert(ST1 >= NS && STX >= NS);
          u32 nb_was = it - NS == nb_le;
          if (st == nst - 1 && epi_item) nb_le = it;
          const int s = it % NS;
          u8* sbase = stages + s * STAGE;
          const u32 full = bars0 + s * 8;
          const int kc0 = kc_item + st * KW;
          if constexpr (LEANC) {
            const u32 lrest3 = ((1u << (nst - st)) - 1u) << kc0;
            if ((lrdy & lrest3) == lrest3) {
              const u8* wq = wblock(kc0);
              const u32 wqs = KW * nrx * (WB + SB);
              const u8* xbq = AS(const u8, phase == 2 ? P.inter : P.xs) + (size_t)tile * ST1 * T * XP;
              const u8* xq = wgl ? xbq + ((size_t)(kc0 * 4) * T) * 128 : xbq + (size_t)(kc0 / KW) * T * XP;
              const u32 xqs = wgl ? KW * 4 * T * 128 : T * XP;
              const u32 xqb = wgl ? ((n + 7) & ~7) * 128 : n * XP;
              const u32 ltx3 = WB + SB + (wgl ? 4 * xqb : xqb);
              int sl = s, stl = st;
              u32 fl = full;
              auto issue = [&](bool w) {
                const u32 sd = stage0 + sl * STAGE;
                if (EWR && w) {
                  mbar_tx(fl, WB + SB);
                  bulk_g2s(sd, wq, WB + SB, fl, pol_w);
                  return;
                }
                mbar_expect(fl, EWR ? ltx3 - WB - SB : ltx3);
                if (!EWR) bulk_g2s(sd, wq, WB + SB, fl, pol_w);
                if (wgl) {
                  for (int c = 0; c < 4; ++c)
                    bulk_g2s(sd + WB + SB + c * T * 128, xq + c * T * 128, xqb, fl, pol_x);
                } else
                  bulk_g2s(sd + WB + SB, xq, xqb, fl, pol_x);
              };
              auto ew = [&](bool r) {
                if (EWR) {
                  if (r) bar_sync(12 + sl, 288);
                  if (elect_one()) issue(1);
                  __syncwarp();
                }
              };
              for (;;) {
                const int l4e = MOE_L2_CLDV && ccl < nst - 1 ? ccl : nst - 1;
                if (NT >= 8 && stl >= NS && stl < l4e) {
                  for (;;) {
                    ew(1);
                    bar_sync(4 + sl, 288);
                    if (elect_one()) issue(0);
                    __syncwarp();
                    wq += wqs;
                    xq += xqs;
                    ++it;
                    ++stl;
                    if (++sl == NS) sl = 0;
                    fl = bars0 + sl * 8;
                    if (stl == l4e) break;
                  }
                  nb_was = 0;
                  nb_le = epi_item ? it : -1000;
                }
                if (MOE_L2_CLDV && stl >= ccl) {
                  MOE_L2_CLAIM();
                  ccl = 1 << 20;
                }
                ew(it >= NS);
                if (it >= NS) bar_sync(4 + sl, nb_was ? 96 : 288);
                if (elect_one()) {
                  if (stl == 0 || stl == nst - 1)
                    PUT_INFO(
                        &info[sl],
                        make_int4(item, phase, tile, rb),
                        make_int4(ks, stl, stl == nst - 1, n),
                        make_int4(p0, e, 0, 0))
                  issue(0);
                }
                __syncwarp();
                wq += wqs;
                xq += xqs;
                ++it;
                if (++stl == nst) break;
                if (++sl == NS) sl = 0;
                fl = bars0 + sl * 8;
                nb_was = it - NS == nb_le;
                if (stl == nst - 1 && epi_item) nb_le = it;
              }
              st = nst - 1;
              --it;
              continue;
            }
          }
          if (EWR && it >= NS) bar_sync(12 + s, 288);
          if (it >= NS) bar_sync(4 + s, nb_was ? 96 : 288);
          if (lane == 0) {
            if (MOE_RT_WRELV && it == 0) mbar_sleep(eis, 0);
            PUT_INFO(
                &info[s], make_int4(item, phase, tile, rb), make_int4(ks, st, st == nst - 1, n), make_int4(p0, e, 0, 0))
            mbar_expect(
                full, WB + SB + (wgl ? 4 * ((n + 7) & ~7) * 128 : (phase == 2 || !xdir ? n * XP : n * KST * 2)));
          }
          __syncwarp();
          if constexpr (WSC) {
            static_assert(WB == 32768 && (SB == 4096 || SB == 3072) && RGS == 16);
            if (lane == 0) bulk_g2s(stage0 + s * STAGE, wblock(kc0), WB + SB, full, pol_w);
          } else if (lane < KW * 2)
            bulk_g2s(stage0 + s * STAGE + cdst, csrc, cbytes, full, pol_w);
          csrc += cstep;
          if (phase == 2) {
            u32 need = ((1u << KW) - 1u) << kc0;
            const bool acq = (rmask & need) != need;
            if (acq) {
              rmask = ballot(lane < 16 && ld_acquire(&cnt1[tile * 16 + lane]) >= 512 / RB);
              if constexpr (LEANC) lrdy = rmask;
            }
            if (lane == 0 && (rmask & need) != need)
              for (int kw = 0; kw < KW; ++kw)
                while (ld_acquire(&cnt1[tile * 16 + kc0 + kw]) < 512 / RB) {
                }
            __syncwarp();
            // fence.proxy.async only orders the issuing thread's accesses, and several lanes
            // issue bulk copies: every lane fences, after the barrier that publishes the acquire.
            if (acq) fence_async_global();
          }
          if (xdir) {
            if (xrow0) bulk_g2s(smem_u32(sbase + WB + SB + lane * XP), xrow0 + kc0 * 256, KST * 2, full, pol_x);
          } else {
            if constexpr (GNEW) {
              if (phase == 1 && gtile != tile) {
                if (lane == 0)
                  while (ld_acquire(CB + GTC_OFF + tile) < (n + GHC - 1) / GHC) {
                  }
                __syncwarp();
                fence_async_global();
                gtile = tile;
                lrdy = ~0u;
              }
            } else if (phase == 1 && !gathered) {
              if (lane == 0)
                while (ld_acquire(CB + 3) < gnch) {
                }
              __syncwarp();
              fence_async_global();
              gathered = true;
              if constexpr (LEANC) lrdy = ~0u;
            }
            const u8* xreg = AS(const u8, phase == 2 ? P.inter : P.xs) + (size_t)tile * ST1 * T * XP;
            if (wgl) {
              if (lane < 4)
                bulk_g2s(
                    smem_u32(sbase + WB + SB + lane * T * 128),
                    xreg + ((size_t)(kc0 * 4 + lane) * T) * 128,
                    ((n + 7) & ~7) * 128,
                    full,
                    pol_x);
            } else if (lane == 0)
              bulk_g2s(smem_u32(sbase + WB + SB), xreg + (size_t)(kc0 / KW) * T * XP, n * XP, full, pol_x);
          }
        }
        if (MOE_L2_CLDV && ccl < (1 << 20)) MOE_L2_CLAIM();
        item = shfl(next, 0);
      }
      return;
    }
    if (warp == 9 && lane == 0) {
      if (MOE_XRELV) {
        mbar_sleep(rdy, 0);
        if (MOE_RT_WRELV) mbar_arrive(eis);
      }
      for (int q = 0;; ++q) {
        const int j = q % NSIG;
        mbar_sleep<200>(smem_u32(&epi[j]), (q / NSIG) & 1);
        const int2 si = sinfo[j];
        mbar_arrive(smem_u32(&sfree[j]));
        if (si.x < 0) break;
        red_release(si.x == 1 ? &cnt1[si.y] : &cnt2[si.y], 1);
      }
    }
    return;
  }
  nreg_inc<216>();
  const int kw = warp % KW, rw = warp / KW;
  int q_epi = 0, q_sig_c = 0;
  const int g = lane >> 2, tig = lane & 3;
#define MOE_WOFF ((kw * RGS + rw * 2) * 2048 + lane * 16)
#define MOE_SOFF (WB + (kw * RGS + rw * 2) * SRG + lane * 12)
#define MOE_XOFF (WB + SB + g * XP + kw * 512 + tig * 32)
#define MOE_STG(a, b, c) stg[(kw * T + (b) * 8 + 2 * tig + ((c) & 1)) * SP + 16 * (2 * rw + (a)) + g + 8 * ((c) >> 1)]
  float acc[2][NT][4], acc2[2][NT][4];
  ZERO_ACC(0)
  int it = 0;
  [[maybe_unused]] const u32 invR = SPX ? 0xffffffffu / (u32)max(P.R, 2) + 1u : 0u;
  [[maybe_unused]] const u32 xlo0 = sw128_desc(smem_u32(stages + WB + SB));
#define MOE_DSKV(K) ((K) < 100 && ds_code(RW, (K), MW) != 0)
#define MOE_STAGES(K)                                                                                                 \
  {                                                                                                                   \
    DsCarry<((K) >= 1 && (K) <= 4) ? (K) : 1> ds_cr{};                                                                \
    for (int st = 0; st < nst; ++st) {                                                                                \
      if (st > 0) {                                                                                                   \
        bar_arrive(4 + s, 288);                                                                                       \
        advance<NS>(s, ph);                                                                                           \
        mbar_poll(smem_u32(&bars[s]), ph);                                                                            \
      }                                                                                                               \
      const u8* sbase = stages + s * STAGE;                                                                           \
      const u8* wbase = sbase + MOE_WOFF;                                                                             \
      const u8* sclb = sbase + MOE_SOFF;                                                                              \
      if constexpr ((K) >= 100)                                                                                       \
        stage_compute_wg<NT, (K) - 100, (MW >= 16)>(acc, wbase, sclb, xlo0 + s * (STAGE >> 4), T, EWR ? 12 + s : -1); \
      else if constexpr (MOE_DSKV(K))                                                                                 \
        stage_ds<ds_code(RW, (K), MW) == 2, NT, (K)>(                                                                 \
            acc, acc2, wbase, sclb, sbase + MOE_XOFF, XP, ds_cr, EWR ? 12 + s : -1);                                  \
      else                                                                                                            \
        stage_compute<NT, (K)>(acc, wbase, sclb, sbase + MOE_XOFF, XP);                                               \
      __syncwarp();                                                                                                   \
    }                                                                                                                 \
    if constexpr (MOE_DSKV(K)) ds_flush<ds_code(RW, (K), MW), NT, (K)>(acc, acc2, ds_cr);                             \
  }
  for (;;) {
    int s, ntl;
    u32 ph;
    StageInfo f;
    s = it % NS;
    ph = (it / NS) & 1;
    mbar_poll(smem_u32(&bars[s]), ph);
    f = info[s];
    if (f.item < 0) {
      __syncwarp();
      if (warp == 0 && lane == 0) aq_slot[q_epi % NS] = CEPI ? (q_sig_c << 8) : -1;
      if (lane == 0) mbar_arrive(smem_u32(&aq_full[q_epi % NS]));
      break;
    }
    ntl = (f.n + 7) >> 3;
    [[maybe_unused]] int ptjv[(T + 31) / 32] = {};
    if (CEPI && SPX && f.item >= 0 && f.phase == 2)
      for (int h = 0; h < (T + 31) / 32; ++h)
        if (lane + 32 * h < f.n) ptjv[h] = __ldcg(P.ptj + f.p0 + lane + 32 * h);
    const int nst = f.phase == 2 && f.tile >= tsplit ? STX : ST1;
    if (HASWG && ntl >= WGMIN) {
#define MOE_WGC(N) \
  case N:          \
    if constexpr (NT >= N) MOE_STAGES(100 + N) break;
      // clang-format off
      switch (ntl) {
        MOE_WGC(3)
        MOE_WGC(4) MOE_WGC(5) MOE_WGC(6) MOE_WGC(7) MOE_WGC(8) MOE_WGC(9) MOE_WGC(10) MOE_WGC(11) MOE_WGC(12)
            MOE_WGC(13) MOE_WGC(14) MOE_WGC(15) default : if constexpr (NT >= 16) MOE_STAGES(116) break;
      }
      // clang-format on
    } else {
      switch (ntl) {
        case 1:
          MOE_STAGES(1) {
            ZERO_ACC(1)
          }
          break;
        case 2:
          MOE_STAGES(2) {
            ZERO_ACC(2)
          }
          break;
      }
    }
    it += nst;
    if constexpr (!IT2K) {
      FOR_ACC(0) acc[a][b][c] *= 64.f;
    }
    if constexpr (CEPI && SPX) {
      if (CE1 || f.phase == 2) {
        bar_arrive(4 + s, 288);
        const int j = q_sig_c % NSIG, t8 = 2 * tig + (g & 1);
        if (f.phase == 1) {
          const float gsc = gsm[f.e];
          const float gke = -1.4426950408889634f * gsc, gs2 = gsc * gsc;
          u8* const hb = AS(u8, P.inter) + (size_t)f.tile * ST1 * T * XP;
          const bool wgt = ntl >= WGMIN;
          if (EPI2 && wgt) {
            const int wg = warp >> 2, wl = tid & 127;
            const u32 st0 = smem_u32(stages + NS * STAGE) + wg * 2048 + (rw & 3) * 4;
            u8* const dst0 = hb + (size_t)((RB / 128) * f.rb + wg) * T * 128;
#pragma unroll
            for (int b = 0; b < NT; ++b)
              if (b < ntl) {
                const u32 sb = st0 + (b & 1) * 1024 + t8 * 128;
                for (int a = 0; a < 2; ++a)
                  sts32(sb + (((4 * a + (g >> 1)) ^ t8) << 4), swiglu_x(acc[a][b], gke, gs2, g));
                bar_sync(8 + wg, 128);
                if (wl < 64) {
                  const int tok = 8 * b + (wl >> 3);
                  const uint4 v = lds128(smem_u32(stages + NS * STAGE) + wg * 2048 + (b & 1) * 1024 + wl * 16);
                  if (tok < f.n) *AS(uint4, dst0 + (size_t)tok * 128 + (wl & 7) * 16) = v;
                }
              }
          } else
#pragma unroll
            for (int a = 0; a < 2; ++a) {
              const int il = 8 * (2 * rw + a) + (g & 6), c64 = (RB / 128) * f.rb + (il >> 6), e2 = (il >> 4) & 3,
                        lc = (il & 15) >> 1;
              const int ig = (RB / 2) * f.rb + 8 * (2 * rw + a);
              u8* const hms = hb + (size_t)(ig / KST) * T * XP + (ig % KST) * 2 + (g >> 1) * 4;
              u8* const hwg = hb + (size_t)c64 * T * 128 + e2 * 4;
#pragma unroll
              for (int b = 0; b < NT; ++b)
                if (b < ntl) {
                  const u32 pr = swiglu_x(acc[a][b], gke, gs2, g);
                  const int tok = b * 8 + t8;
                  if (tok < f.n)
                    *AS(u32, wgt ? hwg + tok * 128 + ((lc ^ (tok & 7)) << 4) : hms + (size_t)tok * XP) = pr;
                }
            }
        } else {
          const float gsc = gsm[32 + f.e];
          float mr[2][2];
          for (int a = 0; a < 2; ++a)
            for (int hh = 0; hh < 2; ++hh)
              mr[a][hh] = __ldg(P.nw1 + f.rb * RB + 16 * (2 * rw + a) + g + 8 * hh) * gsc;
          if constexpr (EPI2) {
            // stmatrix.trans: 16 B of one token row per lane
            const u32 scr = smem_u32(stages + NS * STAGE) + warp * 512, i8 = lane & 7, i4 = lane >> 2;
            const u32 sw = scr + i8 * 64 + ((((lane >> 3) ^ (i8 >> 1)) & 3) << 4),
                      sr = scr + i4 * 64 + ((((lane & 3) ^ (i4 >> 1)) & 3) << 4);
#pragma unroll
            for (int b = 0; b < NT; ++b)
              if (b < ntl) {
                u32 v[4];
                for (int a = 0; a < 2; ++a)
                  for (int hh = 0; hh < 2; ++hh)
                    v[2 * a + hh] = bf2(acc[a][b][2 * hh] * mr[a][hh], acc[a][b][2 * hh + 1] * mr[a][hh]);
                __syncwarp();
                stmatrix_t(sw, v);
                __syncwarp();
                const uint4 o4 = lds128(sr);
                const int tok = b * 8 + (int)i4;
                const int tj = shfl(ptjv[b >> 2], tok & 31), tt = tj >> 3, jj = tj & 7,
                          o = (int)__umulhi((u32)tt, invR);
                u16* const row = sp_pick(P.osp, o) + ((size_t)jj * P.R + (tt - o * P.R)) * 4096 + f.rb * RB + 32 * rw +
                                 8 * (lane & 3);
                if (tok < f.n) *AS(uint4, row) = o4;
              }
          } else {
#pragma unroll
            for (int b = 0; b < NT; ++b)
              if (b < ntl) {
                const int tok = b * 8 + t8;
                const int tj = shfl(ptjv[b >> 2], tok & 31), tt = tj >> 3, jj = tj & 7, o = tt / P.R;
                u16* const row = sp_pick(P.osp, o) + ((size_t)jj * P.R + (tt - o * P.R)) * 4096 + f.rb * RB + (g & 6);
                for (int a = 0; a < 2; ++a)
                  for (int hh = 0; hh < 2; ++hh) {
                    const u32 b0 = f2bf(acc[a][b][2 * hh] * mr[a][hh]), b1 = f2bf(acc[a][b][2 * hh + 1] * mr[a][hh]);
                    const u32 pr = xpack(b0, b1, g);
                    if (tok < f.n) *AS(u32, row + 16 * (2 * rw + a) + 8 * hh) = pr;
                  }
              }
          }
        }
        __syncwarp();
        // producer already waited on sfree[j]
        if (warp == 0 && lane == 0) sinfo[j] = TILE_SIG(f);
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(&epi[j]));
        ++q_sig_c;
        FOR_ACC(0) acc[a][b][c] = 0.f;
        continue;
      }
    }
    constexpr int SP = RB + 4;
    static_assert((CEPI && SPX && CE1) || T * SP * 4 <= STAGE);
    const bool it2xs = IT2K && ntl == 1;
    float* stg = AS(float, stages + s * STAGE + (it2xs ? IT2XOFF : 0));
    if (!it2xs) consumer_sync();
    if constexpr (HB) {
      float* red = AS(float, stages + s * STAGE + (RGS + rw * 2) * 2048) + lane;
      if (kw) {
        FOR_ACC(0) if (b < ntl) red[((a * NT + b) * 4 + c) * 32] = acc[a][b][c];
        bar_arrive(8 + rw, 64);
      } else {
        bar_sync(8 + rw, 64);
        FOR_ACC(0) if (b < ntl) acc[a][b][c] += red[((a * NT + b) * 4 + c) * 32];
      }
    }
    if constexpr (IT2K) {
      if (ntl > 0)
        for (int a = 0; a < ((HB && kw) ? 0 : 2); ++a)
          for (int c = 0; c < 4; ++c)
            MOE_STG(a, 0, c) = acc[a][0][c];
      if (!it2xs)
        for (int b = 1; b < NT; ++b)
          if (b < ntl)
            for (int a = 0; a < ((HB && kw) ? 0 : 2); ++a)
              for (int c = 0; c < 4; ++c)
                MOE_STG(a, b, c) = acc[a][b][c];
    } else
      for (int a = 0; a < ((HB && kw) ? 0 : 2); ++a)
        for (int b = 0; b < NT; ++b)
          if (b < ntl)
            for (int c = 0; c < 4; ++c)
              MOE_STG(a, b, c) = acc[a][b][c];
    if constexpr (MOE_WPA_BUFB > 0) fence_async();
    __syncwarp();
    if (warp == 0 && lane == 0) aq_slot[q_epi % NS] = CEPI ? (s + 1) | (q_sig_c << 8) : s;
    if (lane == 0) mbar_arrive(smem_u32(&aq_full[q_epi % NS]));
    ++q_epi;
    if constexpr (CEPI) ++q_sig_c;
    FOR_ACC(0) acc[a][b][c] = 0.f;
  }
  if constexpr (SPX) return;
  float* rbuf = AS(float, stages + ((NS - 1) * STAGE + (T * (RB + 4) * 4 + 127) / 128 * 128));
  static_assert(SPX || (T * (RB + 4) * 4 + 127) / 128 * 128 + 256 <= STAGE);
  constexpr bool PW = MW <= 1;  // <= 32 tokens: each route pair is combined as soon as its two tiles are done
  float4 w4s[4];
  for (int k = 0; k < 4; ++k)
    w4s[k] = *AS(const float4, P.nw1 + (tid + 256 * k) * 4);
  auto claim = [&](int* ctr_word) -> int {
    if (tid == 0) scal[8] = atomicAdd(ctr_word, 1);
    consumer_sync();
    const int v = scal[8];
    consumer_sync();
    return v;
  };
  int ci = claim(CB + 4);
  for (int epar = 0; ci < M; epar ^= 1) {
    const int t = ci;
    const int e = lane < 8 ? lid[t * 8 + lane] : -1;
    const u32 vm = ballot(e >= 0);
    const int np = __popc(vm);
    int pv = 0, tv = -1;
    float wv = 0.f;
    if (e >= 0) {
      const u32 word = bm[BMI(e, t >> 5)];
      const int pr = wpre[BMI(e, t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));
      pv = off[e] + pr;
      wv = P.topk_w[t * 8 + lane];
      tv = ftile[e] + pr / T;
    }
    int src = lane < np ? (int)__fns(vm, 0, lane + 1) : 0;
    if constexpr (PW) {  // tiles finish about in index order: pair routes by tile
      int rk = 0;
      for (int i = 0; i < 8; ++i) {
        const int ti = shfl(tv, i);
        rk += ti >= 0 && ti < tv;
      }
      for (int i = 0; i < 8; ++i) {
        const int ri = shfl(rk, i);
        if (((vm >> i) & 1u) && ri == lane) src = i;
      }
    }
    const int pl = shfl(pv, src);
    const float wq = shfl(wv, src);
    int tlq = shfl(tv, src);
    if (lane >= np) tlq = -1;
    auto wait_tile = [&](bool w) {
      if (w && warp == 0 && tlq >= 0) {
        const int want = NRB2 * (tlq >= tsplit ? KSX : 1);
        while (ld_relaxed(&cnt2[tlq]) < want) {
        }
        (void)ld_acquire(&cnt2[tlq]);
      }
      consumer_sync();
    };
    if constexpr (!PW) wait_tile(true);
    int enxt = M;
    if ((!PW || np == 0) && tid == 0) enxt = atomicAdd(CB + 4, 1);
    float accv[16] = {};
    int rnd = 0;
    for (int q0 = 0; q0 < np; q0 += 2, ++rnd) {
      if constexpr (PW) {
        wait_tile(lane >> 1 == q0 >> 1);
        if (q0 + 2 >= np && tid == 0) enxt = atomicAdd(CB + 4, 1);
      }
      float v[2][16];
      float ss[2];
      for (int gq = 0; gq < 2; ++gq) {
        ss[gq] = 0.f;
        const int p = shfl(pl, (q0 + gq) & 31);
        const bool live = q0 + gq < np;
        const int kp = p >= psplit ? KSX : 1;
        for (int k = 0; k < 4; ++k) {
          const int n = (tid + 256 * k) * 4;
          float4 y = {};
          if (live && kp == 1) y = ld_bf4(P.yh + (size_t)p * 4096 + n);
          for (int k2 = 0; k2 < KSX; ++k2)
            if (live && kp > 1 && k2 < kp) {
              const float4 u = __ldcg(AS(const float4, P.ypart + ((size_t)k2 * P.pmax + p) * 4096 + n));
              y.x += u.x;
              y.y += u.y;
              y.z += u.z;
              y.w += u.w;
            }
          const float4 w = w4s[k];
          v[gq][4 * k] = y.x * w.x;
          v[gq][4 * k + 1] = y.y * w.y;
          v[gq][4 * k + 2] = y.z * w.z;
          v[gq][4 * k + 3] = y.w * w.w;
          ss[gq] += v[gq][4 * k] * v[gq][4 * k] + v[gq][4 * k + 1] * v[gq][4 * k + 1] +
                    v[gq][4 * k + 2] * v[gq][4 * k + 2] + v[gq][4 * k + 3] * v[gq][4 * k + 3];
        }
      }
      for (int s2 = 16; s2 > 0; s2 >>= 1)
        for (int gq = 0; gq < 2; ++gq)
          ss[gq] += shfl_xor(ss[gq], s2);
      if (lane == 0)
        for (int gq = 0; gq < 2; ++gq)
          rbuf[(rnd & 1) * 32 + gq * 8 + warp] = ss[gq];
      consumer_sync();
      for (int gq = 0; gq < 2; ++gq)
        if (q0 + gq < np) {
          float tot = 0.f;
          for (int w = 0; w < 8; ++w)
            tot += rbuf[(rnd & 1) * 32 + gq * 8 + w];
          const float gw = shfl(wq, (q0 + gq) & 31);
          float r = rsqrtf(tot * (1.f / 4096.f) + 1e-8f) * gw;
          for (int k = 0; k < 16; ++k)
            accv[k] += r * v[gq][k];
        }
    }
    for (int k = 0; k < 4; ++k) {
      const int n = (tid + 256 * k) * 4;
      *AS(uint2, P.out + (size_t)t * 4096 + n) =
          make_uint2(bf2(accv[4 * k], accv[4 * k + 1]), bf2(accv[4 * k + 2], accv[4 * k + 3]));
    }
    if (tid == 0) scal[12 + epar] = enxt;
    consumer_sync();
    ci = scal[12 + epar];
  }
}
#undef EV8
#undef TILE_SIG
#undef MOE_WPA
#define MOE_INST(PRE, SPX, RW, NT, KS2, NS, MW)                                  \
  extern "C" __global__ void __launch_bounds__(384, 1) __cluster_dims__(1, 1, 1) \
      PRE##_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) {                        \
    moe_body<RW, NT, NS, MW, SPX>(p);                                            \
  }
#define INST(...) MOE_INST(moe, false, __VA_ARGS__)
#define INSTSP(...) MOE_INST(moesp, true, __VA_ARGS__)
