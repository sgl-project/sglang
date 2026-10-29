// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from TensorRT-LLM (Apache-2.0, NVIDIA): tensorrt_llm/_torch/modules/fla/q35_gdn_core.py
// (q35_gdn_core: work split, MMA fragments and PDL placement), on SGLang's ReplaySSM circular ring
// (python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py,
// gdn_replayssm_spec_circular_kernel) and gated RMSNorm (kernels/ops/attention/fla/layernorm_gated.py).
//
// FN adaptation of Q35 source @5474b2cdbc61; sigmoid gate / BF16 output.
// The ReplaySSM verify of a GDN target-verify step (the circular ring's non-flush rows) and its gated
// RMSNorm writing out_proj's BF16 input, in one launch: one CTA per (request, value head), 4
// warps, warp w owns the value columns 32w .. 32w + 31. Products run as mma.m16n8k16 (BF16 in, FP32
// accumulate) on the normalized q / k rows and the checkpoint rows; the forward substitution, the
// intra-step output and the norm run in FP32. The BF16 roundings of SGLang's kernels are kept
// (normalized q, key high / low parts, update high / low parts, the replay output before the norm,
// the final BF16 norm/gate output); FP32 sums run in other orders (numeric).
//
// PDL: before griddepcontrol.wait the kernel reads only data of earlier steps or static data (indices,
// ring cursors, the checkpoint rows, A_log / dt_bias / norm weight); every global store
// and every read of the step's activations (q / k / v, z, a, b) follows the wait. The commit that writes the
// ring cursors and checkpoint rows must stay out of this launch's PDL window (it runs in the previous round).
//
// Defines: TT (rows a request), HK / HV (key / value heads), NORM_F32 (FP32 norm weight), TRIGGER (2: release dependents at entry, 0: at exit),
// STORE_O (also store the BF16 replay output, checks only), EARLY_HIST (1: the committed history -- keys, log-decays,
// update vectors and their decays -- before the wait, held there; the same arithmetic, the same bits), FLAT_STORES
// (1: the ring entries and BF16 outputs computed first and only their stores predicated, one low-part test, the update
// vectors stored after the output; the same bits).

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;

#define KD 128
#define VD 128
#define RING 16
#ifndef MINB
#define MINB 1  // resident CTAs an SM the registers are capped for
#endif
#ifndef EARLY_HIST
#define EARLY_HIST 0
#endif
#ifndef FLAT_STORES
#define FLAT_STORES 0
#endif
#ifndef NW
#define NW 4  // warps; each owns VD / NW value columns as MT m16 tiles
#endif
#ifndef VROWS
#define VROWS 128
#endif
#define MT (VROWS / (16 * NW))
#if FUSED_NORM && VROWS != 128
#error full-head RMS requires VROWS=128
#endif
#define FULL 0xffffffffu

struct alignas(16) U4 { u32 x, y, z, w; };

struct Params {
  const u16* x;          // convolved q | k | v rows
  long long x_stride;
  const u16* z;          // gate rows: z + row * z_stride + hv * VD
  long long z_stride;
  const u16* a;
  const u16* b;
  long long ab_stride;
  const float* A_log;
  const void* dt_bias;
  const u16* state;      // [slots, HV, V, K]
  long long state_slot;
  u16* ring_d;           // [slots, HV, RING, V]
  long long d_slot;
  u16* ring_k;           // [slots, HK, RING, K]
  long long k_slot;
  float* ring_g;         // [slots, HV, RING]
  long long g_slot;
  u16* ring_dlo;         // update low parts (layout of ring_d), or null
  long long dlo_slot;
  u16* ring_klo;         // key low parts (layout of ring_k), or null
  long long klo_slot;
  const void* state_indices;
  const void* replay_indices;
  int state_index64;     // int64 (else int32) state / replay indices
  int replay_index64;
  const int* write_pos;
  const int* cache_base;
  const signed char* is_flush;
  const void* norm_w;
  const float* scale;
  float eps;
  float qscale;
  u16* y;                 // BF16 out_proj input [rows, HV * V]
  long long y_stride;
  u16* o_out;            // [rows, HV, V] replay output (STORE_O)
  long long* prof;       // PROF: clock64 stamps [cta][warp][8]
};

__device__ __forceinline__ float bflo(u32 x) { return __int_as_float(x << 16); }
__device__ __forceinline__ float bfhi(u32 x) { return __int_as_float(x & 0xffff0000u); }
__device__ __forceinline__ float bf2f(u16 x) { return __int_as_float(((u32)x) << 16); }
__device__ __forceinline__ u16 f2bf(float x) {
  u16 r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float bfround(float x) { return bf2f(f2bf(x)); }
__device__ __forceinline__ u32 pack2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ u32 word(const U4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ U4 ld16(const void* p) { return *reinterpret_cast<const U4*>(p); }
__device__ __forceinline__ void st16(void* p, U4 v) { *reinterpret_cast<U4*>(p) = v; }
__device__ __forceinline__ U4 zero4() { U4 z; z.x = z.y = z.z = z.w = 0u; return z; }
__device__ __forceinline__ float rcp_rn(float x) {
  float r;
  asm("div.rn.f32 %0, 0f3F800000, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ u8 f2e4m3(float x) {
  u16 r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(r) : "f"(x));
  return (u8)(r & 0xff);
}
__device__ __forceinline__ float pick8(const float (&a)[8], int i) {
  float r = a[0];
#pragma unroll
  for (int k = 1; k < 8; ++k) r = (i == k) ? a[k] : r;
  return r;
}
__device__ __forceinline__ void mma16816(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// K-reduction of 128-wide rows: A = rows r (rows r + 8 zero), B = 8 rows as columns. Lane (r, q) holds the
// elements [32 c + 8 q, + 8) of its row in chunk c; k-step 2c + e takes words (2e, 2e + 1) of both operands.
__device__ __forceinline__ void kred(float (&acc)[4], const U4 (&A)[4], const U4 (&B)[4]) {
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int e = 0; e < 2; ++e)
      mma16816(acc, word(A[c], 2 * e), 0u, word(A[c], 2 * e + 1), 0u, word(B[c], 2 * e), word(B[c], 2 * e + 1));
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Holds the loads that `dep` reads (one word of each) ahead of griddepcontrol.wait, which ptxas may otherwise sink
// them below; `never` is a launch constant that is never 0.
__device__ __forceinline__ void pin_before_wait(u32 dep, unsigned long long never) {
  asm volatile("{ .reg .pred a, b; setp.eq.u64 a, %1, 0; setp.eq.u32 b, %0, 1; and.pred a, a, b; @a trap; }"
               :: "r"(dep), "l"(never) : "memory");
}


__device__ __forceinline__ long long load_index(const void* p, int i, bool i64) {
  return i64 ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
}

#ifndef PROF
#define PROF 0
#endif
#if PROF
// PROF_GT (micro builds): %globaltimer (ns, comparable across SMs) instead of clock64
__device__ __forceinline__ long long prof_time() {
#ifdef PROF_GT
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return (long long)t;
#else
  return clock64();
#endif
}
#ifdef PROF_GT
// the timer read predicated on a test of `dep`: stamped once dep is available
__device__ __forceinline__ long long prof_time_after(float dep) {
  unsigned long long t;
  asm volatile("{ .reg .pred p; setp.ne.f32 p, %1, 0f7F7FFFFF; @p mov.u64 %0, %%globaltimer; "
               "@!p mov.u64 %0, %%globaltimer; }" : "=l"(t) : "f"(dep));
  return (long long)t;
}
#define STAMP(i, dep) do { if (lane == 0) st_[i] = prof_time_after((float)(dep)); } while (0)
#else
#define STAMP(i, dep) do { if (lane == 0) { asm volatile("" ::"f"((float)(dep))); st_[i] = prof_time(); } } while (0)
#endif
#else
#define STAMP(i, dep) do { } while (0)
#endif

// MINB 3: three CTAs an SM (168 registers, no spill), for grids past two waves of two an SM
extern "C" __global__ void __launch_bounds__(NW * 32, MINB) q38_mkgdn_core(const Params p) {
  constexpr int GRP = HV / HK;
  const int cta = blockIdx.x;
  const int slice = cta % (VD / VROWS);
  const int head_cta = cta / (VD / VROWS);
  const int n = head_cta / HV;
  const int hv = head_cta - n * HV;
  const int hk = hv / GRP;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int r = lane >> 2;
  const int q = lane & 3;
  const int vb = slice * VROWS + warp * 16 * MT;
  const int t0 = 2 * q;
  const int t1 = 2 * q + 1;
  const long long row0 = (long long)n * TT;

  __shared__ float s_A[8][8];    // A[s][t] = beta_s e^{G_s - G_t} k_s.k_t, s > t
  __shared__ float s_F[8][8];    // F[s][t] = e^{G_t - G_s} k_s.q_t, s <= t
  __shared__ u32 s_sc[2][2][32]; // history scores (q / k, history tile) as BF16 B fragments
  __shared__ float s_norm[NW][8];

#if PROF
  long long st_[8];
  st_[0] = prof_time();
#endif
#if TRIGGER == 2
  pdl_launch();
#endif
  // ---------------- before the wait: earlier-step and static data ----------------
  const long long slot = load_index(p.state_indices, n, p.state_index64 != 0);
  bool live = slot >= 0;
  // the checkpoint rows first: they depend on the state index only
  const u16* sp = p.state + (live ? slot : 0) * p.state_slot + (long long)hv * VD * KD;
  U4 H[MT][2][4];  // checkpoint rows v = vb + 16 m + 8 h + r: [m-tile][half][chunk]
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int c = 0; c < 4; ++c)
        H[m][h][c] = live ? ld16(sp + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q) : zero4();
  long long rid = 0;
  int wp = 0, base = 0;
  if (live) {
    rid = load_index(p.replay_indices, n, p.replay_index64 != 0);
    wp = p.write_pos[rid];
    base = p.cache_base[rid];
    live = p.is_flush[rid] == 0;
  }
  if (!live) {
    // padding (or a flush row, which the verify never sees): zeros for out_proj
    pdl_wait();
    for (int t = 0; t < TT && tid < VROWS; ++t) {
      p.y[(row0 + t) * p.y_stride + hv * VD + slice * VROWS + tid] = 0;
#if STORE_O
      p.o_out[((row0 + t) * HV + hv) * VD + tid] = 0;
#endif
    }
    return;
  }
  const float negAexp = -__expf(p.A_log[hv]);
#if DT_BF16
  const float dtb = bf2f(reinterpret_cast<const u16*>(p.dt_bias)[hv]);
#else
  const float dtb = reinterpret_cast<const float*>(p.dt_bias)[hv];
#endif
#if FUSED_NORM
  float nw[MT][2];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int v = vb + 16 * m + 8 * h + r;
#if NORM_F32
      nw[m][h] = reinterpret_cast<const float*>(p.norm_w)[v];
#else
      nw[m][h] = bf2f(reinterpret_cast<const u16*>(p.norm_w)[v]);
#endif
    }
#endif
  const float rscale = 1.f;
#if EARLY_HIST
  // the committed history (ring entries j < wp, written by earlier steps; this step writes j >= wp): the history keys
  // of warps 2 / 3, and every lane's update-vector A fragments with their decays (the post-sync block's arithmetic)
  U4 KHe[2][4];
  u32 hist_a[MT][4];
  float hist_td = 0.f;
  {
    u32 dep = 0;
#pragma unroll
    for (int jt = 0; jt < 2; ++jt)
#pragma unroll
      for (int c = 0; c < 4; ++c) KHe[jt][c] = zero4();
    if ((warp == 2 || warp == 3) && wp > 0) {
#pragma unroll
      for (int jt = 0; jt < 2; ++jt) {
        const int j = 8 * jt + r;
        const u16* kh = p.ring_k + rid * p.k_slot + ((long long)hk * RING + ((base + j) & (RING - 1))) * KD;
#pragma unroll
        for (int c = 0; c < 4; ++c) KHe[jt][c] = j < wp ? ld16(kh + 32 * c + 8 * q) : zero4();
      }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) hist_a[m][i] = 0u;
    if (wp > 0) {
      float gj = 0.f;
      if (lane < RING && lane < wp) gj = p.ring_g[rid * p.g_slot + hv * RING + ((base + lane) & (RING - 1))];
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float o = __shfl_up_sync(FULL, gj, d);
        if (lane >= d) gj += o;
      }
      const float gtot = __shfl_sync(FULL, gj, 31);
      const int jj[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
      float dec[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float pj = __shfl_sync(FULL, gj, jj[i]);
        dec[i] = jj[i] < wp ? __expf(gtot - pj) : 0.f;
      }
      hist_td = __expf(gtot);
      const u16* dp = p.ring_d + rid * p.d_slot + (long long)hv * RING * VD;
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        float e[2][4];  // [half][j index i]
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            e[h][i] = jj[i] < wp
                          ? bf2f(dp[((base + jj[i]) & (RING - 1)) * VD + vb + 16 * m + 8 * h + r]) * dec[i]
                          : 0.f;
        hist_a[m][0] = pack2(e[0][0], e[0][1]);
        hist_a[m][1] = pack2(e[1][0], e[1][1]);
        hist_a[m][2] = pack2(e[0][2], e[0][3]);
        hist_a[m][3] = pack2(e[1][2], e[1][3]);
        dep ^= hist_a[m][0] ^ hist_a[m][3];
      }
    }
#pragma unroll
    for (int jt = 0; jt < 2; ++jt)
#pragma unroll
      for (int c = 0; c < 4; ++c) dep ^= KHe[jt][c].x;
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) dep ^= H[m][h][0].x;
    pin_before_wait(dep, (unsigned long long)p.write_pos);
  }
#endif

  STAMP(1, bflo(H[1][1][3].w) + bflo(H[0][0][0].x) + rscale);
  // ---------------- the wait: the step's activations from here on ----------------
  pdl_wait();
  STAMP(2, 0.f);

  const u16* xr = p.x + (row0 + (r < TT ? r : 0)) * p.x_stride;
  const bool rowok = r < TT;
  const bool ok0 = t0 < TT, ok1 = t1 < TT;
  // every activation load is issued here, right after the wait
  U4 QN[4], KR[4];  // normalized q (BF16) and raw k rows; rows r >= TT zero
  U4 QR[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    QR[c] = rowok ? ld16(xr + hk * KD + 32 * c + 8 * q) : zero4();
    KR[c] = rowok ? ld16(xr + HK * KD + hk * KD + 32 * c + 8 * q) : zero4();
  }
  // v / z of this lane's (v, t): [m][(h, t)] = (h0, t0), (h0, t1), (h1, t0), (h1, t1)
  u16 vr[MT][4], zr[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int v = vb + 16 * m + 8 * h + r;
      const u16* xv = p.x + 2 * HK * KD + hv * VD + v;

      vr[m][2 * h] = ok0 ? xv[(row0 + t0) * p.x_stride] : (u16)0;
      vr[m][2 * h + 1] = ok1 ? xv[(row0 + t1) * p.x_stride] : (u16)0;
#if FUSED_NORM
      const u16* zv = p.z + hv * VD + v;
      zr[m][2 * h] = ok0 ? zv[(row0 + t0) * p.z_stride] : (u16)0;
      zr[m][2 * h + 1] = ok1 ? zv[(row0 + t1) * p.z_stride] : (u16)0;
#endif
    }
  u16 ar = 0, br_ = 0;
  if (lane < TT) {
    ar = p.a[(row0 + lane) * p.ab_stride + hv];
    br_ = p.b[(row0 + lane) * p.ab_stride + hv];
  }
  float kmul;       // 1 / |k_r|
  {
    float sq[4] = {0.f, 0.f, 0.f, 0.f}, sk[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const u32 wq = word(QR[c], i), wk = word(KR[c], i);
        sq[i] += bflo(wq) * bflo(wq) + bfhi(wq) * bfhi(wq);
        sk[i] += bflo(wk) * bflo(wk) + bfhi(wk) * bfhi(wk);
      }
    float sqs = (sq[0] + sq[1]) + (sq[2] + sq[3]), sks = (sk[0] + sk[1]) + (sk[2] + sk[3]);
    sqs += __shfl_xor_sync(FULL, sqs, 1);
    sks += __shfl_xor_sync(FULL, sks, 1);
    sqs += __shfl_xor_sync(FULL, sqs, 2);
    sks += __shfl_xor_sync(FULL, sks, 2);
    const float qmul = rowok ? (1.0f / sqrtf(sqs + 1e-6f)) * p.qscale : 0.f;
    kmul = rowok ? 1.0f / sqrtf(sks + 1e-6f) : 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      QN[c].x = pack2(bflo(QR[c].x) * qmul, bfhi(QR[c].x) * qmul);
      QN[c].y = pack2(bflo(QR[c].y) * qmul, bfhi(QR[c].y) * qmul);
      QN[c].z = pack2(bflo(QR[c].z) * qmul, bfhi(QR[c].z) * qmul);
      QN[c].w = pack2(bflo(QR[c].w) * qmul, bfhi(QR[c].w) * qmul);
    }
  }
  const float km0 = __shfl_sync(FULL, kmul, 4 * t0), km1 = __shfl_sync(FULL, kmul, 4 * t1);
  STAMP(3, km0 + bflo(QN[3].w));
  // gates of token = lane, cumulative log-decay by an 8-lane scan
  float gl = 0.f, beta = 0.f;
  if (lane < TT) {
    const float xg = bf2f(ar) + dtb;
    const float spl = xg <= 20.f ? __logf(1.f + __expf(xg)) : xg;
    gl = negAexp * spl;
    beta = __fdividef(1.f, 1.f + __expf(-bf2f(br_)));
  }
  // the norm's sigmoid(z) gate, here so that z arrives while the products run
#if FUSED_NORM
  float zg[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float zv = bf2f(zr[m][i]);
      zg[m][i] = __fdividef(1.f, 1.f + __expf(-zv));
    }
#endif
  float Gv = gl;
#pragma unroll
  for (int d = 1; d < 8; d <<= 1) {
    const float o = __shfl_up_sync(FULL, Gv, d);
    if (lane >= d) Gv += o;
  }
  const float Gt0 = __shfl_sync(FULL, Gv, t0), Gt1 = __shfl_sync(FULL, Gv, t1);
  const float bt0 = __shfl_sync(FULL, beta, t0), bt1 = __shfl_sync(FULL, beta, t1);
  const float Gr = __shfl_sync(FULL, Gv, r), br = __shfl_sync(FULL, beta, r);
  const float eG0 = __expf(Gt0), eG1 = __expf(Gt1);

  // checkpoint products (all warps): rows v, columns = tokens; the key product on raw rows, scaled by 1/|k_t|
  float HQ[MT][4], HKs[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int i = 0; i < 4; ++i) HQ[m][i] = HKs[m][i] = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const u32 a0 = word(H[m][0][c], 2 * e), a1 = word(H[m][1][c], 2 * e);
        const u32 a2 = word(H[m][0][c], 2 * e + 1), a3 = word(H[m][1][c], 2 * e + 1);
        mma16816(HKs[m], a0, a1, a2, a3, word(KR[c], 2 * e), word(KR[c], 2 * e + 1));
        mma16816(HQ[m], a0, a1, a2, a3, word(QN[c], 2 * e), word(QN[c], 2 * e + 1));
      }
    HKs[m][0] *= km0;
    HKs[m][1] *= km1;
    HKs[m][2] *= km0;
    HKs[m][3] *= km1;
  }
  // this step's ring entries at (base + wp + t) mod RING, t < TT, wp + t < RING: keys (warp 2) and
  // log-decays (warp 3) now, while warps 0 / 1 finish the head-common products; updates at the end
  if (slice == 0 && warp == 3 && lane < TT && wp + lane < RING)
    p.ring_g[rid * p.g_slot + hv * RING + ((base + wp + lane) & (RING - 1))] = gl;
  if (slice == 0 && warp == 2 && hv % GRP == 0 && r < TT && wp + r < RING) {
    const long long off = ((long long)hk * RING + ((base + wp + r) & (RING - 1))) * KD;
    u16* kp = p.ring_k + rid * p.k_slot + off;
    u16* klp = p.ring_klo ? p.ring_klo + rid * p.klo_slot + off : nullptr;
#if FLAT_STORES
    U4 vh[4], vl[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      u32 kh[4], kl[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const u32 wk = word(KR[c], i);
        const float k0 = bflo(wk) * kmul, k1 = bfhi(wk) * kmul;
        kh[i] = pack2(k0, k1);
        kl[i] = pack2(k0 - bflo(kh[i]), k1 - bfhi(kh[i]));
      }
      vh[c].x = kh[0]; vh[c].y = kh[1]; vh[c].z = kh[2]; vh[c].w = kh[3];
      vl[c].x = kl[0]; vl[c].y = kl[1]; vl[c].z = kl[2]; vl[c].w = kl[3];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) st16(kp + 32 * c + 8 * q, vh[c]);
    if (klp) {
#pragma unroll
      for (int c = 0; c < 4; ++c) st16(klp + 32 * c + 8 * q, vl[c]);
    }
#else
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      u32 kh[4], kl[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const u32 wk = word(KR[c], i);
        const float k0 = bflo(wk) * kmul, k1 = bfhi(wk) * kmul;
        kh[i] = pack2(k0, k1);
        kl[i] = pack2(k0 - bflo(kh[i]), k1 - bfhi(kh[i]));
      }
      U4 vh, vl;
      vh.x = kh[0]; vh.y = kh[1]; vh.z = kh[2]; vh.w = kh[3];
      vl.x = kl[0]; vl.y = kl[1]; vl.z = kl[2]; vl.w = kl[3];
      st16(kp + 32 * c + 8 * q, vh);
      if (klp) st16(klp + 32 * c + 8 * q, vl);
    }
#endif
  }

  // head-common products: warp 0 k k^T (A), warp 1 k q^T (F), warps 2 / 3 the history scores
  if (warp == 0) {
    float kk[4] = {0.f, 0.f, 0.f, 0.f};
    kred(kk, KR, KR);
    const float ks = kmul;  // row s = r
    s_A[r][t0] = (r > t0 && r < TT) ? br * __expf(Gr - Gt0) * (kk[0] * ks * km0) : 0.f;
    s_A[r][t1] = (r > t1 && r < TT) ? br * __expf(Gr - Gt1) * (kk[1] * ks * km1) : 0.f;
  } else if (warp == 1) {
    float kq[4] = {0.f, 0.f, 0.f, 0.f};
    kred(kq, KR, QN);
    s_F[r][t0] = (r <= t0 && t0 < TT) ? __expf(Gt0 - Gr) * (kq[0] * kmul) : 0.f;
    s_F[r][t1] = (r <= t1 && t1 < TT) ? __expf(Gt1 - Gr) * (kq[1] * kmul) : 0.f;
  } else if ((warp == 2 || warp == 3) && wp > 0) {
    // scores[j][t] = khist_j . (q_t | k_t), j = 8 jt + 2q + i (rows j >= wp zero), rounded to BF16
    U4 KHh[4];
#pragma unroll
    for (int jt = 0; jt < 2; ++jt) {
#if EARLY_HIST
#pragma unroll
      for (int c = 0; c < 4; ++c) KHh[c] = KHe[jt][c];
#else
      const int j = 8 * jt + r;
      const u16* kh = p.ring_k + rid * p.k_slot + ((long long)hk * RING + ((base + j) & (RING - 1))) * KD;
#pragma unroll
      for (int c = 0; c < 4; ++c) KHh[c] = j < wp ? ld16(kh + 32 * c + 8 * q) : zero4();
#endif
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      if (warp == 2) {
        kred(acc, QN, KHh);
      } else {
        U4 KH[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          KH[c].x = pack2(bflo(KR[c].x) * kmul, bfhi(KR[c].x) * kmul);
          KH[c].y = pack2(bflo(KR[c].y) * kmul, bfhi(KR[c].y) * kmul);
          KH[c].z = pack2(bflo(KR[c].z) * kmul, bfhi(KR[c].z) * kmul);
          KH[c].w = pack2(bflo(KR[c].w) * kmul, bfhi(KR[c].w) * kmul);
        }
        kred(acc, KH, KHh);
      }
      s_sc[warp - 2][jt][lane] = pack2(acc[0], acc[1]);
    }
  }
  STAMP(4, HQ[MT - 1][3] + HKs[MT - 1][3]);
  __syncthreads();
  STAMP(5, 0.f);

  if (wp > 0) {
#if EARLY_HIST
    const u32 bq0 = s_sc[0][0][lane], bq1 = s_sc[0][1][lane];
    const u32 bk0 = s_sc[1][0][lane], bk1 = s_sc[1][1][lane];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        HQ[m][i] *= hist_td;
        HKs[m][i] *= hist_td;
      }
      mma16816(HQ[m], hist_a[m][0], hist_a[m][1], hist_a[m][2], hist_a[m][3], bq0, bq1);
      mma16816(HKs[m], hist_a[m][0], hist_a[m][1], hist_a[m][2], hist_a[m][3], bk0, bk1);
    }
#else
    // committed history: hw = e^{g_total} hw + bf16(d_j e^{g_total - G_j}) . bf16(scores)
    float gj = 0.f;
    if (lane < RING && lane < wp) gj = p.ring_g[rid * p.g_slot + hv * RING + ((base + lane) & (RING - 1))];
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float o = __shfl_up_sync(FULL, gj, d);
      if (lane >= d) gj += o;
    }
    const float gtot = __shfl_sync(FULL, gj, 31);
    const int jj[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
    float dec[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float pj = __shfl_sync(FULL, gj, jj[i]);
      dec[i] = jj[i] < wp ? __expf(gtot - pj) : 0.f;
    }
    const float td = __expf(gtot);
    const u16* dp = p.ring_d + rid * p.d_slot + (long long)hv * RING * VD;
    const u32 bq0 = s_sc[0][0][lane], bq1 = s_sc[0][1][lane];
    const u32 bk0 = s_sc[1][0][lane], bk1 = s_sc[1][1][lane];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float e[2][4];  // [half][j index i]
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          e[h][i] = jj[i] < wp
                        ? bf2f(dp[((base + jj[i]) & (RING - 1)) * VD + vb + 16 * m + 8 * h + r]) * dec[i]
                        : 0.f;
      const u32 a0 = pack2(e[0][0], e[0][1]), a1 = pack2(e[1][0], e[1][1]);
      const u32 a2 = pack2(e[0][2], e[0][3]), a3 = pack2(e[1][2], e[1][3]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        HQ[m][i] *= td;
        HKs[m][i] *= td;
      }
      mma16816(HQ[m], a0, a1, a2, a3, bq0, bq1);
      mma16816(HKs[m], a0, a1, a2, a3, bk0, bk1);
    }
#endif
  }

  // rhs = beta (v - e^G hw_k); forward substitution (I + A) U = rhs per value column, every lane of a quad
  // solving the quad's four columns after an all-gather
  float U[2 * MT][8];
  {
    float rhs[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      rhs[m][0] = bt0 * (bf2f(vr[m][0]) - eG0 * HKs[m][0]);
      rhs[m][1] = bt1 * (bf2f(vr[m][1]) - eG1 * HKs[m][1]);
      rhs[m][2] = bt0 * (bf2f(vr[m][2]) - eG0 * HKs[m][2]);
      rhs[m][3] = bt1 * (bf2f(vr[m][3]) - eG1 * HKs[m][3]);
    }
    float R[2 * MT][8];
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
      const int src = (lane & ~3) | qq;
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          R[2 * m + h][2 * qq] = __shfl_sync(FULL, rhs[m][2 * h], src);
          R[2 * m + h][2 * qq + 1] = __shfl_sync(FULL, rhs[m][2 * h + 1], src);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float u[2 * MT] = {};
      if (i < TT) {
#pragma unroll
        for (int col = 0; col < 2 * MT; ++col) u[col] = R[col][i];
#pragma unroll
        for (int s = 0; s < i; ++s) {
          const float a = s_A[i][s];
#pragma unroll
          for (int col = 0; col < 2 * MT; ++col) u[col] -= a * U[col][s];
        }
      }
#pragma unroll
      for (int col = 0; col < 2 * MT; ++col) U[col][i] = u[col];
    }
  }
  STAMP(6, U[2 * MT - 1][TT - 1] + U[0][TT - 1]);
#if !FLAT_STORES
  // this step's update vectors (high, low parts) into the ring
  {
    u16* dp = p.ring_d + rid * p.d_slot + (long long)hv * RING * VD;
    u16* dlp = p.ring_dlo ? p.ring_dlo + rid * p.dlo_slot + (long long)hv * RING * VD : nullptr;
    const bool w0 = ok0 && wp + t0 < RING, w1 = ok1 && wp + t1 < RING;
    const int ph0 = (base + wp + t0) & (RING - 1), ph1 = (base + wp + t1) & (RING - 1);
#pragma unroll
    for (int col = 0; col < 2 * MT; ++col) {
      const int v = vb + 16 * (col >> 1) + 8 * (col & 1) + r;
      if (w0) {
        const float d = pick8(U[col], t0);
        const u16 hi = f2bf(d);
        dp[ph0 * VD + v] = hi;
        if (dlp) dlp[ph0 * VD + v] = f2bf(d - bf2f(hi));
      }
      if (w1) {
        const float d = pick8(U[col], t1);
        const u16 hi = f2bf(d);
        dp[ph1 * VD + v] = hi;
        if (dlp) dlp[ph1 * VD + v] = f2bf(d - bf2f(hi));
      }
    }
  }
#endif
  // output O[v, t] = e^{G_t} hw_q + sum_{s <= t} U[v, s] F[s][t]
  float O[MT][4];
  {
    float f0[8], f1[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      f0[s] = s_F[s][t0];
      f1[s] = s_F[s][t1];
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int s = 0; s < TT; ++s) {
          a0 += U[2 * m + h][s] * f0[s];
          a1 += U[2 * m + h][s] * f1[s];
        }
        O[m][2 * h] = eG0 * HQ[m][2 * h] + a0;
        O[m][2 * h + 1] = eG1 * HQ[m][2 * h + 1] + a1;
      }
  }
#if FLAT_STORES
  // this step's update vectors (high, low parts) into the ring, off the output's path
  {
    u16* dp = p.ring_d + rid * p.d_slot + (long long)hv * RING * VD;
    const bool w0 = ok0 && wp + t0 < RING, w1 = ok1 && wp + t1 < RING;
    const int ph0 = (base + wp + t0) & (RING - 1), ph1 = (base + wp + t1) & (RING - 1);
    if (p.ring_dlo) {
      u16* dlp = p.ring_dlo + rid * p.dlo_slot + (long long)hv * RING * VD;
#pragma unroll
      for (int col = 0; col < 2 * MT; ++col) {
        const int v = vb + 16 * (col >> 1) + 8 * (col & 1) + r;
        const float d0 = pick8(U[col], t0), d1 = pick8(U[col], t1);
        const u16 h0 = f2bf(d0), h1 = f2bf(d1);
        const u16 l0 = f2bf(d0 - bf2f(h0)), l1 = f2bf(d1 - bf2f(h1));
        if (w0) dp[ph0 * VD + v] = h0;
        if (w0) dlp[ph0 * VD + v] = l0;
        if (w1) dp[ph1 * VD + v] = h1;
        if (w1) dlp[ph1 * VD + v] = l1;
      }
    } else {
#pragma unroll
      for (int col = 0; col < 2 * MT; ++col) {
        const int v = vb + 16 * (col >> 1) + 8 * (col & 1) + r;
        const u16 h0 = f2bf(pick8(U[col], t0)), h1 = f2bf(pick8(U[col], t1));
        if (w0) dp[ph0 * VD + v] = h0;
        if (w1) dp[ph1 * VD + v] = h1;
      }
    }
  }
#endif

  // gated RMSNorm over the head's 128 columns of the BF16-rounded replay output, then sigmoid gate
  float xb[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) xb[m][i] = bfround(O[m][i]);
#if STORE_O
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = (i & 1) ? t1 : t0;
      const int v = vb + 16 * m + 8 * (i >> 1) + r;
      if (t < TT) p.o_out[((row0 + t) * HV + hv) * VD + v] = f2bf(O[m][i]);
    }
#endif
#if !FUSED_NORM
  return;
#else
  float p0 = 0.f, p1 = 0.f;
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    p0 += xb[m][0] * xb[m][0] + xb[m][2] * xb[m][2];
    p1 += xb[m][1] * xb[m][1] + xb[m][3] * xb[m][3];
  }
#pragma unroll
  for (int msk = 4; msk < 32; msk <<= 1) {
    p0 += __shfl_xor_sync(FULL, p0, msk);
    p1 += __shfl_xor_sync(FULL, p1, msk);
  }
  if (r == 0) {
    s_norm[warp][t0] = p0;
    s_norm[warp][t1] = p1;
  }
  __syncthreads();
  STAMP(7, 0.f);
  float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
  for (int w = 0; w < NW; ++w) {
    ss0 += s_norm[w][t0];
    ss1 += s_norm[w][t1];
  }
  const float rstd0 = rsqrtf(ss0 / (float)VD + p.eps);
  const float rstd1 = rsqrtf(ss1 / (float)VD + p.eps);
#if FLAT_STORES
  {
    u16* y0 = p.y + (row0 + t0) * p.y_stride + hv * VD + vb + r;
    u16* y1 = p.y + (row0 + t1) * p.y_stride + hv * VD + vb + r;
    u16 yq[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float yv = xb[m][i] * ((i & 1) ? rstd1 : rstd0) * nw[m][i >> 1];
        yv *= zg[m][i];
        yq[m][i] = f2bf(yv);
      }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i & 1) {
          if (ok1) y1[16 * m + 8 * (i >> 1)] = yq[m][i];
        } else {
          if (ok0) y0[16 * m + 8 * (i >> 1)] = yq[m][i];
        }
      }
  }
#else
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = i >> 1;
      const bool ok = (i & 1) ? ok1 : ok0;
      if (ok) {
        const int t = (i & 1) ? t1 : t0;
        const int v = vb + 16 * m + 8 * h + r;
        float yv = xb[m][i] * ((i & 1) ? rstd1 : rstd0) * nw[m][h];
        yv *= zg[m][i];
        p.y[(row0 + t) * p.y_stride + hv * VD + v] = f2bf(yv);
      }
    }
#endif
#endif  // FUSED_NORM
#if PROF
  if (lane == 0) {
    const long long end = prof_time();
    long long* pw = p.prof + ((long long)cta * NW + warp) * 9;
#pragma unroll
    for (int i = 0; i < 8; ++i) pw[i] = st_[i];
    pw[8] = end;
  }
#endif
}
