// SPDX-License-Identifier: Apache-2.0
// KDA (Kimi Delta Attention) chunked prefill for gfx950 (MI350X / MI355X).
//
// Inputs are bf16 q, k, v, raw gates and raw beta of shape [1, T, H, 128] with cu_seqlens; the gate (A_log,
// dt_bias, lower bound -5), the beta sigmoid and the q / k L2 norms are applied here.  Each sequence's fp32
// [128, 128] state per head is read from its slot of the state pool and overwritten with the final state;
// optionally the state after one chunk is copied to track_state.
//
// Two phases.  kda_intra_kernel computes the intra-chunk terms of every (64-token chunk, head) into a workspace
// block (normalized q and k, decayed keys, the UT transform of the chunk's delta rule, intra-chunk attention).  A
// recurrence kernel then walks each sequence's chunks in order, applying the blocks to the state and writing the
// outputs.  The recurrence paths below compute the same result and differ only in how the (sequence, head) chains
// are spread over the CUs; kda_recur_path picks one from the batch shape, the longest sequence and the CU count:
//
//   fused  kda_fusedp_kernel      both phases pipelined in one workgroup per (sequence, head), block kept in LDS
//   grp    kda_recur_grp_*        one long sequence as G groups walked in parallel, joined by an fp32 scan
//   ws4    kda_recur_kernel_ws4   value quarters, warp-specialized math and loader waves
//   ws2    kda_recur_kernel_ws2   value halves, warp-specialized math and loader waves
//   vs2    kda_recur_kernel_vs2   value halves
//   vs1    kda_recur_kernel_vs1   one 512-thread workgroup per (sequence, head)
//
// Only grp, vs2 and vs1 store track_state.
//
// Sections:
//    1. common helpers: vector types, bf16 packing, MFMA wrappers, LDS index maps
//    2. workspace block and LDS layouts
//    3. intra-chunk phase (kda_intra_kernel)
//    4. recurrence step: chunk fetch, LDS staging and one chunk of the state update
//    5. whole-head and value-half walks (vs1, vs2)
//    6. warp-specialized walks (ws2, ws4)
//    7. sequence groups (grp)
//    8. pipelined fused kernel (fused)
//    9. kernel entry points
//   10. host: path selection and launch
//   11. torch binding

#include <hip/hip_runtime.h>

#include <stddef.h>
#include <stdint.h>

// =====================================================================================================================
// 1. Common helpers
// =====================================================================================================================

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef float f32x2 __attribute__((ext_vector_type(2)));

#define DEVI static __device__ __forceinline__
#define DEVM __device__ __forceinline__  // member functions (staging policies)

// two fp32 -> packed bf16 pair (round to nearest even), one packed-convert instruction on gfx950
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
DEVI uint32_t pk_bf16(float lo, float hi) {
  f32x2 f;
  f[0] = lo;
  f[1] = hi;
  return __builtin_bit_cast(uint32_t, __builtin_convertvector(f, bf16x2_t));
}
DEVI uint32_t pk2(f32x2 f) {
  return __builtin_bit_cast(uint32_t, __builtin_convertvector(f, bf16x2_t));
}
DEVI float bf_lo(uint32_t u) {
  return __uint_as_float(u << 16);
}
DEVI float bf_hi(uint32_t u) {
  return __uint_as_float(u & 0xffff0000u);
}

#define EXP2(x) __builtin_amdgcn_exp2f(x)
#define RCP(x) __builtin_amdgcn_rcpf(x)
#define RSQ(x) __builtin_amdgcn_rsqf(x)
#define TID ((int)threadIdx.x)
#define BID ((int)blockIdx.x)
#define SYNC() __syncthreads()
// scheduling barrier only: the LDS operations of one wave complete in issue order
#define WAVE_SYNC() __builtin_amdgcn_wave_barrier()
// scheduling fence: no instruction is moved across it by the instruction schedulers (stages of the
// software-pipelined recurrence chunk)
#define SCHED_FENCE() __builtin_amdgcn_sched_barrier(0)
// loader waves of the warp-specialized recurrence: ~256 cycles of sleep at the top of each chunk
#define LOADER_SLEEP() __builtin_amdgcn_s_sleep(4)
// wave index as a wave-uniform (scalar) value: wave-dependent branches become scalar branches
#define WAVE_U() __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 6)
// DPP lane permutation within 16-lane rows (all source lanes valid for the controls used here:
// quad_perm 0x00-0xff, row_mirror 0x140, row_half_mirror 0x141)
#define DPP_F(x, ctrl) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), (ctrl), 0xF, 0xF, true))

DEVI float lane_shfl(float x, int src_lane) {
  return __int_as_float(__builtin_amdgcn_ds_bpermute(src_lane << 2, __float_as_int(x)));
}
DEVI float SHFL_UP64(float x, int d) {
  int lane = (int)threadIdx.x & 63;
  return lane_shfl(x, lane >= d ? lane - d : lane);
}

// bf16 16x16x32 and fp32 16x16x4 MFMA through the clang builtins (compiler-managed hazards)
DEVI f32x4 MFMA16(u32x4 a, u32x4 b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(
      __builtin_bit_cast(bf16x8_t, a), __builtin_bit_cast(bf16x8_t, b), c, 0, 0, 0);
}
DEVI f32x4 MFMA4(float a, float b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
}

#define KC 64
#define KD 128
#define TR_LD 72
#define SZ_TR (128 * TR_LD * 2)
#define SZ_A16 (64 * TR_LD * 2)
#define LOG2E_F 1.4426950408889634f
DEVI int tr_idx(int c, int j) {
  int h = j >> 2;
  int s = ((c >> 4) & 7) << 1;
  return c * TR_LD + ((h ^ s) << 2) + (j & 3);
}
DEVI int a16_idx(int i, int j) {
  return i * TR_LD + j;
}
DEVI u32x4 lds_ld16(const uint16_t* base, int idx) {
  return *(const u32x4*)(base + idx);
}
DEVI void lds_st16(uint16_t* base, int idx, u32x4 v) {
  *(u32x4*)(base + idx) = v;
}
DEVI void lds_st8(uint16_t* base, int idx, u32x2 v) {
  *(u32x2*)(base + idx) = v;
}
DEVI u32x2 pack4(float a, float b, float c, float d) {
  u32x2 r;
  r[0] = pk_bf16(a, b);
  r[1] = pk_bf16(c, d);
  return r;
}
DEVI u32x4 acc_pair_to_bf16(f32x4 lo, f32x4 hi) {
  u32x4 r;
  r[0] = pk_bf16(lo[0], lo[1]);
  r[1] = pk_bf16(lo[2], lo[3]);
  r[2] = pk_bf16(hi[0], hi[1]);
  r[3] = pk_bf16(hi[2], hi[3]);
  return r;
}
DEVI f32x4 zero4() {
  f32x4 z;
  z[0] = z[1] = z[2] = z[3] = 0.f;
  return z;
}
DEVI f32x2 mk2(float a, float b) {
  f32x2 r;
  r[0] = a;
  r[1] = b;
  return r;
}
DEVI f32x2 sp2(float a) {
  return mk2(a, a);
}
DEVI f32x2 bf2(uint32_t u) {
  return mk2(bf_lo(u), bf_hi(u));
}
DEVI f32x2 exp2v(f32x2 v) {
  return mk2(EXP2(v[0]), EXP2(v[1]));
}
DEVI f32x2 rcpv(f32x2 v) {
  return mk2(RCP(v[0]), RCP(v[1]));
}
DEVI float row_sum16(float x) {
  x += DPP_F(x, 0xB1);
  x += DPP_F(x, 0x4E);
  x += DPP_F(x, 0x141);
  x += DPP_F(x, 0x140);
  return x;
}
DEVI void gst16(char* p, u32x4 v) {
  *(u32x4*)p = v;
}
DEVI void gst8(char* p, u32x2 v) {
  *(u32x2*)p = v;
}
DEVI void MFMA_SA(f32x4& c0, f32x4& c1, const u32x4* a, const u32x4* b0, const u32x4* b1, int n) {
  for (int s = 0; s < n; ++s) {
    c0 = MFMA16(a[s], b0[s], c0);
    c1 = MFMA16(a[s], b1[s], c1);
  }
}
DEVI void MFMA_SB(f32x4& c0, f32x4& c1, const u32x4* a0, const u32x4* a1, const u32x4* b, int n) {
  for (int s = 0; s < n; ++s) {
    c0 = MFMA16(a0[s], b[s], c0);
    c1 = MFMA16(a1[s], b[s], c1);
  }
}

// =====================================================================================================================
// 2. Workspace block and LDS layouts
// =====================================================================================================================

// GB_*: byte offsets in the GB_BYTES workspace block of one (chunk, head); LA_*: LDS of kda_intra_kernel;
// SMEM_B*: LDS of the recurrence walks (double-buffered block plus transposed values)
#define GB_W 0
#define GB_Q 16384
#define GB_K 32768
#define GB_A 49152
#define GB_T 55296
#define GB_G 61440
#define GB_SH 61952
#define GB_B 61952
#define GB_BYTES 62208
#define SZ_RX (64 * 128 * 2)
#define LA_QR 0
#define LA_KR (SZ_RX)
#define LA_KE (2 * SZ_RX)
#define LA_KGT 0
#define LA_T16 (SZ_TR)
#define LA_AKK (LA_T16 + SZ_A16)
#define LA_TFD (LA_AKK + 12800)
#define LA_YEND (LA_TFD + 5376)
#define LA_BETA (3 * SZ_RX)
#define LA_TAU (LA_BETA + 256)
#define LA_FACD (LA_TAU + 2048)
#define SMEM_A (LA_FACD + 2048)
static_assert(LA_YEND <= 3 * SZ_RX, "phase-Y arrays must fit in the dead tile bytes");
static_assert(SMEM_A <= 42 * 1280, "kernel A must fit three workgroups per CU");
DEVI int rx_idx(int row, int col) {
  return row * 128 + ((((col >> 3) ^ row) & 15) << 3) + (col & 7);
}
#define LB_VT GB_SH
#define SMEM_B1 (2 * (GB_SH + 128 * TR_LD * 2))
#define SMEM_B2 (2 * (GB_SH + 64 * TR_LD * 2))
DEVI int akk_idx(int I, int J, int ii, int jj) {
  return ((I * (I + 1) / 2 + J) * 16 + ii) * 20 + jj;
}
DEVI int tfd_idx(int I, int ii, int jj) {
  return I * 336 + ii * 20 + jj;
}
DEVI int fa_idx(int mt, int ks) {
  return mt < 2 ? mt : 2 + 2 * (mt - 2) + ks;
}

// =====================================================================================================================
// 3. Intra-chunk phase
// =====================================================================================================================

struct IntraItem {
  long t0;
  int L, h, cs, valid;
};
struct IntraRaw {
  u32x4 q[4], k[4], g[4];
  uint32_t b[4];
  float a_log;
  f32x4 dt0, dt1;
};
DEVI IntraItem intra_decode(int item, const int* __restrict__ cu, int T, int H, int N) {
  IntraItem it;
  it.cs = item / H;
  it.h = item - it.cs * H;
  it.valid = 0;
  it.t0 = 0;
  it.L = 0;
  long bos = 0, eos = T;
  int ci = it.cs;
  long acc = 0;
  int found = 0;
  for (int m = 0; m < N; ++m) {
    long b0 = (long)cu[m], b1 = (long)cu[m + 1];
    long cnt = (b1 - b0 + KC - 1) / KC;
    if (!found && it.cs < acc + cnt) {
      bos = b0;
      eos = b1;
      ci = (int)(it.cs - acc);
      found = 1;
    }
    acc += cnt;
  }
  if (!found) return it;
  it.t0 = bos + (long)ci * KC;
  it.L = (eos - it.t0) < KC ? (int)(eos - it.t0) : KC;
  it.valid = 1;
  return it;
}
DEVI void intra_load_qkg(
    IntraRaw& r,
    const IntraItem& it,
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ g,
    const uint16_t* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    int H,
    int w,
    int lg,
    int lr) {
  const int h = it.h;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int R = 16 * w + 4 * lg + i;
    long t = (R < it.L) ? (it.t0 + R) : it.t0;
    size_t off = ((size_t)t * H + h) * KD + 8 * lr;
    r.q[i] = *(const u32x4*)(q + off);
    r.k[i] = *(const u32x4*)(k + off);
    r.g[i] = *(const u32x4*)(g + off);
  }
  r.a_log = A_log[h];
  r.dt0 = *(const f32x4*)(dt_bias + h * KD + 8 * lr);
  r.dt1 = *(const f32x4*)(dt_bias + h * KD + 8 * lr + 4);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int R = 16 * w + 4 * lg + i;
    long t = (R < it.L) ? (it.t0 + R) : it.t0;
    r.b[i] = (uint32_t)beta[(size_t)t * H + h];
  }
}
DEVI int intra_nreal(const int* __restrict__ cu, int T, int H, int N) {
  long tot = 0;
  for (int m = 0; m < N; ++m)
    tot += ((long)cu[m + 1] - (long)cu[m] + KC - 1) / KC;
  return (int)(tot * H);
}
DEVI void p3_col(u32x4* col, const uint16_t* KE, int J, int lg, int lr) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
    col[ks] = lds_ld16(KE, rx_idx(16 * J + lr, 32 * ks + 8 * lg));
}
DEVI void p3_scale(u32x4* dst, const u32x4* src, const float* fac, int lg) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int cb = 32 * ks + 8 * lg;
    const f32x4 fa = *(const f32x4*)(fac + cb);
    const f32x4 fb = *(const f32x4*)(fac + cb + 4);
    u32x4 r;
    r[0] = pk2(bf2(src[ks][0]) * mk2(fa[0], fa[1]));
    r[1] = pk2(bf2(src[ks][1]) * mk2(fa[2], fa[3]));
    r[2] = pk2(bf2(src[ks][2]) * mk2(fb[0], fb[1]));
    r[3] = pk2(bf2(src[ks][3]) * mk2(fb[2], fb[3]));
    dst[ks] = r;
  }
}
template <int NB>
DEVI void p3_scale_tau(u32x4* dst, const u32x4* src, const float* ta, const float* tb, int lg) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int cb = 32 * ks + 8 * lg;
    f32x4 fa = *(const f32x4*)(ta + cb);
    f32x4 fb = *(const f32x4*)(ta + cb + 4);
    if (NB == 2) {
      fa += *(const f32x4*)(tb + cb);
      fb += *(const f32x4*)(tb + cb + 4);
    }
    u32x4 r;
    r[0] = pk2(bf2(src[ks][0]) * exp2v(mk2(fa[0], fa[1])));
    r[1] = pk2(bf2(src[ks][1]) * exp2v(mk2(fa[2], fa[3])));
    r[2] = pk2(bf2(src[ks][2]) * exp2v(mk2(fb[0], fb[1])));
    r[3] = pk2(bf2(src[ks][3]) * exp2v(mk2(fb[2], fb[3])));
    dst[ks] = r;
  }
}
DEVI f32x4 p3_pair(
    const u32x4* col,
    int I,
    int J,
    bool diag,
    const uint16_t* QR,
    const uint16_t* KR,
    const float* BETA,
    char* gb,
    float scale,
    int lg,
    int lr) {
  f32x4 accq = zero4(), acck = zero4();
  u32x4 bq[4], bk[4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int cb = 32 * ks + 8 * lg;
    bq[ks] = lds_ld16(QR, rx_idx(16 * I + lr, cb));
    bk[ks] = lds_ld16(KR, rx_idx(16 * I + lr, cb));
  }
  MFMA_SA(accq, acck, col, bq, bk, 4);
  const float bi = BETA[16 * I + lr];
  const f32x2 q01 = mk2(accq[0], accq[1]) * sp2(scale), q23 = mk2(accq[2], accq[3]) * sp2(scale);
  const f32x2 k01 = mk2(acck[0], acck[1]) * sp2(bi), k23 = mk2(acck[2], acck[3]) * sp2(bi);
  float aq[4] = {q01[0], q01[1], q23[0], q23[1]};
  float ak[4] = {k01[0], k01[1], k23[0], k23[1]};
  if (diag) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      aq[jj] = (4 * lg + jj <= lr) ? aq[jj] : 0.f;
      ak[jj] = (4 * lg + jj < lr) ? ak[jj] : 0.f;
    }
  }
  gst8(
      gb + GB_A + (size_t)(fa_idx(I, J >> 1) * 64 + lg * 16 + lr) * 16 + (J & 1) * 8,
      pack4(aq[0], aq[1], aq[2], aq[3]));
  f32x4 akv;
  akv[0] = ak[0];
  akv[1] = ak[1];
  akv[2] = ak[2];
  akv[3] = ak[3];
  return akv;
}
DEVI void st_akk(float* AKK, int I, int J, f32x4 akv, int lg, int lr) {
  *(f32x4*)(AKK + akk_idx(I, J, lr, 4 * lg)) = akv;
}
template <int J>
DEVI void p5_col(const float* AKK, const float* TFD, uint16_t* T16, int lg, int lr) {
  f32x4 Tc[4];
#pragma unroll
  for (int e = 0; e < 4; ++e)
    Tc[J][e] = TFD[tfd_idx(J, 4 * lg + e, lr)];
#pragma unroll
  for (int I = J + 1; I < 4; ++I) {
    f32x4 X = zero4();
#pragma unroll
    for (int K0 = J; K0 < I; K0 += 2) {
      const f32x4 a_lo = *(const f32x4*)(AKK + akk_idx(I, K0, lr, 4 * lg));
      const f32x4 a_hi = (K0 + 1 < I) ? *(const f32x4*)(AKK + akk_idx(I, K0 + 1, lr, 4 * lg)) : zero4();
      const f32x4 b_hi = (K0 + 1 < I) ? Tc[K0 + 1] : zero4();
      X = MFMA16(acc_pair_to_bf16(a_lo, a_hi), acc_pair_to_bf16(Tc[K0], b_hi), X);
    }
    const f32x4 td = *(const f32x4*)(TFD + tfd_idx(I, lr, 4 * lg));
    const f32x4 Y = MFMA16(acc_pair_to_bf16(td, zero4()), acc_pair_to_bf16(X, zero4()), zero4());
    f32x4 nt;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      nt[e] = -Y[e];
      T16[a16_idx(16 * I + 4 * lg + e, 16 * J + lr)] = (uint16_t)(pk_bf16(nt[e], 0.f) & 0xffffu);
    }
    Tc[I] = nt;
  }
}
DEVI void
intra_process(char* smem, const IntraItem& cit, const IntraRaw& cur, char* __restrict__ gbuf, int H, float scale) {
  uint16_t* L16 = (uint16_t*)smem;
  float* LF = (float*)smem;
  const int tid = TID;
  const int wu = WAVE_U();
  const int lane = tid & 63;
  const int lg = lane >> 4;
  const int lr = lane & 15;
  uint16_t* const QR = L16 + LA_QR / 2;
  uint16_t* const KR = L16 + LA_KR / 2;
  uint16_t* const KE = L16 + LA_KE / 2;
  uint16_t* const KGT = L16 + LA_KGT / 2;
  uint16_t* const T16 = L16 + LA_T16 / 2;
  float* const AKK = LF + LA_AKK / 4;
  float* const TFD = LF + LA_TFD / 4;
  float* const BETA = LF + LA_BETA / 4;
  float* const TAU = LF + LA_TAU / 4;
  float* const FACD = LF + LA_FACD / 4;
  const int L = cit.L;
  const int h = cit.h;
  char* const gb = gbuf + ((size_t)cit.cs * H + h) * GB_BYTES;
  const float a_h = EXP2(cur.a_log * LOG2E_F);
  const float a2 = -a_h * LOG2E_F;
  const float k1 = 1.0f / (-5.0f * LOG2E_F);
  f32x2 b2v[4];
  b2v[0] = sp2(a2) * mk2(cur.dt0[0], cur.dt0[1]);
  b2v[1] = sp2(a2) * mk2(cur.dt0[2], cur.dt0[3]);
  b2v[2] = sp2(a2) * mk2(cur.dt1[0], cur.dt1[1]);
  b2v[3] = sp2(a2) * mk2(cur.dt1[2], cur.dt1[3]);
  f32x2 lam2[4][4];
  float rqn[4], rkn[4], bsig[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f32x2 sq = sp2(0.f), sk = sp2(0.f);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const f32x2 qv = bf2(cur.q[i][p]);
      const f32x2 kv = bf2(cur.k[i][p]);
      sq += qv * qv;
      sk += kv * kv;
      const f32x2 x = bf2(cur.g[i][p]) * sp2(a2) + b2v[p];
      const f32x2 e = exp2v(x);
      lam2[i][p] = rcpv(e * sp2(k1) + sp2(k1));
    }
    rqn[i] = sq[0] + sq[1];
    rkn[i] = sk[0] + sk[1];
    bsig[i] = RCP(1.0f + EXP2(-bf_lo(cur.b[i]) * LOG2E_F));
  }
  if (L < KC) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bool ok = 16 * wu + 4 * lg + i < L;
#pragma unroll
      for (int p = 0; p < 4; ++p)
        lam2[i][p] = ok ? lam2[i][p] : sp2(0.f);
      bsig[i] = ok ? bsig[i] : 0.f;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    rqn[i] = RSQ(row_sum16(rqn[i]) + 1e-6f);
    rkn[i] = RSQ(row_sum16(rkn[i]) + 1e-6f);
  }
#pragma unroll
  for (int i = 1; i < 4; ++i)
#pragma unroll
    for (int p = 0; p < 4; ++p)
      lam2[i][p] += lam2[i - 1][p];
  {
    const float m1 = (lg >= 1) ? 1.f : 0.f, m2 = (lg >= 2) ? 1.f : 0.f;
    f32x2 totw[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const f32x2 tot = lam2[3][p];
      f32x2 inc = mk2(SHFL_UP64(tot[0], 16), SHFL_UP64(tot[1], 16)) * sp2(m1) + tot;
      inc = mk2(SHFL_UP64(inc[0], 32), SHFL_UP64(inc[1], 32)) * sp2(m2) + inc;
      const f32x2 excl = inc - tot;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        lam2[i][p] += excl;
      totw[p] = inc;
    }
    if (lg == 3) {
      f32x4 t0v, t1v;
      t0v[0] = totw[0][0];
      t0v[1] = totw[0][1];
      t0v[2] = totw[1][0];
      t0v[3] = totw[1][1];
      t1v[0] = totw[2][0];
      t1v[1] = totw[2][1];
      t1v[2] = totw[3][0];
      t1v[3] = totw[3][1];
      *(f32x4*)(TAU + wu * KD + 8 * lr) = t0v;
      *(f32x4*)(TAU + wu * KD + 8 * lr + 4) = t1v;
    }
  }
  if (lr == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      BETA[16 * wu + 4 * lg + i] = bsig[i];
    f32x4 bv;
    bv[0] = bsig[0];
    bv[1] = bsig[1];
    bv[2] = bsig[2];
    bv[3] = bsig[3];
    *(f32x4*)(gb + GB_B + (16 * wu + 4 * lg) * 4) = bv;
  }
  SYNC();
  u32x2 kgt[4][2];
  {
    f32x2 tw[4], e2c[4], qgf[4], kdf[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int c = 8 * lr + 2 * p;
      const f32x2 t0 = *(const f32x2*)(TAU + c);
      const f32x2 t1 = *(const f32x2*)(TAU + KD + c);
      const f32x2 t2 = *(const f32x2*)(TAU + 2 * KD + c);
      const f32x2 t3 = *(const f32x2*)(TAU + 3 * KD + c);
      f32x2 carry = sp2(0.f);
      if (wu > 0) carry = t0;
      if (wu > 1) carry += t1;
      if (wu > 2) carry += t2;
      tw[p] = (wu == 0) ? t0 : (wu == 1) ? t1 : (wu == 2) ? t2 : t3;
      const f32x2 glt = (t0 + t1) + (t2 + t3);
      e2c[p] = exp2v(carry);
      qgf[p] = e2c[p] * sp2(scale);
      kdf[p] = exp2v((glt - carry) - tw[p]);
    }
    f32x2 kdv[4], kgv[4];
    u32x2 kdp[4][2];
    const int q4 = lr & 3;
    const int ksq = lr >> 2;
    const int h1 = q4 >> 1;
    const int ga = 2 * (q4 & 1);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const f32x2 rq = sp2(rqn[i]), rk = sp2(rkn[i]), bs = sp2(bsig[i]);
      u32x4 qrp, krp, kep;
      uint32_t qgp[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const f32x2 lm = lam2[i][p];
        const f32x2 E = exp2v(lm);
        const f32x2 G = exp2v(tw[p] - lm);
        const f32x2 qh = bf2(cur.q[i][p]) * rq;
        const f32x2 kh = bf2(cur.k[i][p]) * rk;
        const f32x2 qr = qh * E;
        const f32x2 kr = kh * E;
        const f32x2 ke = kh * G;
        qrp[p] = pk2(qr);
        krp[p] = pk2(kr);
        kep[p] = pk2(ke);
        qgp[p] = pk2(qr * qgf[p]);
        const f32x2 kd = ke * kdf[p];
        const f32x2 kg = kr * e2c[p] * bs;
        if (i & 1) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            kdp[p][e][i >> 1] = pk_bf16(kdv[p][e], kd[e]);
            kgt[p][e][i >> 1] = pk_bf16(kgv[p][e], kg[e]);
          }
        } else {
          kdv[p] = kd;
          kgv[p] = kg;
        }
      }
      const int R = 16 * wu + 4 * lg + i;
      lds_st16(QR, rx_idx(R, 8 * lr), qrp);
      lds_st16(KR, rx_idx(R, 8 * lr), krp);
      lds_st16(KE, rx_idx(R, 8 * lr), kep);
      const int n = 4 * lg + i;
      char* qb = gb + GB_Q + (size_t)((wu * 4 + ksq) * 64) * 16 + h1 * 8;
      u32x2 lo, hi;
      lo[0] = qgp[0];
      lo[1] = qgp[1];
      hi[0] = qgp[2];
      hi[1] = qgp[3];
      gst8(qb + (n + 16 * ga) * 16, lo);
      gst8(qb + (n + 16 * (ga + 1)) * 16, hi);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int c = 8 * lr + 2 * p + e;
        char* kb = gb + GB_K + (size_t)(((c >> 4) * 2 + (wu >> 1)) * 64 + lg * 16 + (c & 15)) * 16 + (wu & 1) * 8;
        gst8(kb, kdp[p][e]);
      }
  }
  {
    const int c0 = 8 * lr + 2 * lg;
    const f32x2 tov = *(const f32x2*)(TAU + wu * KD + c0);
    *(f32x2*)(FACD + wu * KD + c0) = exp2v(-tov);
    if (wu == 0) {
      const f32x2 t0v = *(const f32x2*)(TAU + c0);
      const f32x2 t1v = *(const f32x2*)(TAU + KD + c0);
      const f32x2 t2v = *(const f32x2*)(TAU + 2 * KD + c0);
      const f32x2 t3v = *(const f32x2*)(TAU + 3 * KD + c0);
      *(f32x2*)(gb + GB_G + c0 * 4) = exp2v((t0v + t1v) + (t2v + t3v));
    }
  }
  SYNC();
  f32x4 akd, ako0, ako1 = zero4();
  {
    u32x4 cr[4], cs[4];
    if (wu == 0) {
      p3_col(cr, KE, 0, lg, lr);
      ako0 = p3_pair(cr, 1, 0, false, QR, KR, BETA, gb, scale, lg, lr);
      p3_scale(cs, cr, FACD + 0 * KD, lg);
      akd = p3_pair(cs, 0, 0, true, QR, KR, BETA, gb, scale, lg, lr);
      p3_scale_tau<1>(cs, cr, TAU + 1 * KD, TAU, lg);
      ako1 = p3_pair(cs, 2, 0, false, QR, KR, BETA, gb, scale, lg, lr);
    } else if (wu == 1) {
      p3_col(cr, KE, 1, lg, lr);
      ako0 = p3_pair(cr, 2, 1, false, QR, KR, BETA, gb, scale, lg, lr);
      p3_scale(cs, cr, FACD + 1 * KD, lg);
      akd = p3_pair(cs, 1, 1, true, QR, KR, BETA, gb, scale, lg, lr);
      p3_col(cr, KE, 0, lg, lr);
      p3_scale_tau<2>(cs, cr, TAU + 1 * KD, TAU + 2 * KD, lg);
      ako1 = p3_pair(cs, 3, 0, false, QR, KR, BETA, gb, scale, lg, lr);
    } else if (wu == 2) {
      p3_col(cr, KE, 2, lg, lr);
      p3_scale(cs, cr, FACD + 2 * KD, lg);
      akd = p3_pair(cs, 2, 2, true, QR, KR, BETA, gb, scale, lg, lr);
      p3_col(cr, KE, 1, lg, lr);
      p3_scale_tau<1>(cs, cr, TAU + 2 * KD, TAU, lg);
      ako0 = p3_pair(cs, 3, 1, false, QR, KR, BETA, gb, scale, lg, lr);
    } else {
      p3_col(cr, KE, 2, lg, lr);
      ako0 = p3_pair(cr, 3, 2, false, QR, KR, BETA, gb, scale, lg, lr);
      p3_col(cr, KE, 3, lg, lr);
      p3_scale(cs, cr, FACD + 3 * KD, lg);
      akd = p3_pair(cs, 3, 3, true, QR, KR, BETA, gb, scale, lg, lr);
    }
    if (wu >= 2) {
      u32x2 z;
      z[0] = 0u;
      z[1] = 0u;
      gst8(gb + GB_A + (size_t)(fa_idx(wu == 2 ? 0 : 2, wu == 2 ? 0 : 1) * 64 + lane) * 16 + 8, z);
    }
  }
  SYNC();
  {
    if (wu == 0) {
      st_akk(AKK, 1, 0, ako0, lg, lr);
      st_akk(AKK, 2, 0, ako1, lg, lr);
    } else if (wu == 1) {
      st_akk(AKK, 2, 1, ako0, lg, lr);
      st_akk(AKK, 3, 0, ako1, lg, lr);
    } else if (wu == 2) {
      st_akk(AKK, 3, 1, ako0, lg, lr);
    } else {
      st_akk(AKK, 3, 2, ako0, lg, lr);
    }
    st_akk(AKK, wu, wu, akd, lg, lr);
    const int jrow = 16 * wu + 4 * lg;
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        lds_st8(KGT, tr_idx(8 * lr + 2 * p + e, jrow), kgt[p][e]);
    {
      int blk = tid >> 7, rr = (tid >> 3) & 15, cc = (tid & 7) * 2;
      int I = blk ? 2 : 0;
      T16[a16_idx(16 * I + rr, 16 * (I + 1) + cc)] = 0;
      T16[a16_idx(16 * I + rr, 16 * (I + 1) + cc + 1)] = 0;
    }
    WAVE_SYNC();
    {
      const int j = lr;
      const float* Ab = AKK + akk_idx(wu, wu, 0, 0);
      f32x2 t2[8];
#pragma unroll
      for (int ii = 0; ii < 16; ++ii) {
        f32x2 acc = sp2(0.f);
#pragma unroll
        for (int kk = 0; kk < (ii >> 1); ++kk)
          acc += *(const f32x2*)(Ab + ii * 20 + 2 * kk) * t2[kk];
        float s = acc[0] + acc[1];
        if (ii & 1) s += Ab[ii * 20 + ii - 1] * t2[ii >> 1][0];
        t2[ii >> 1][ii & 1] = ((ii == j) ? 1.f : 0.f) - s;
      }
      if (lg == 0) {
#pragma unroll
        for (int ii = 0; ii < 16; ++ii) {
          const float tv = t2[ii >> 1][ii & 1];
          TFD[tfd_idx(wu, ii, j)] = tv;
          T16[a16_idx(16 * wu + ii, 16 * wu + j)] = (uint16_t)(pk_bf16(tv, 0.f) & 0xffffu);
        }
      }
    }
  }
  SYNC();
  if (wu == 0)
    p5_col<0>(AKK, TFD, T16, lg, lr);
  else if (wu == 1)
    p5_col<1>(AKK, TFD, T16, lg, lr);
  else if (wu == 2)
    p5_col<2>(AKK, TFD, T16, lg, lr);
  SYNC();
  {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      f32x4 acc[2];
      u32x4 a0[2], a1[2], b[2];
      const int nks = (nt < 2) ? 1 : 2;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        a0[ks] = lds_ld16(KGT, tr_idx(16 * (2 * wu) + lr, 32 * ks + 8 * lg));
        a1[ks] = lds_ld16(KGT, tr_idx(16 * (2 * wu + 1) + lr, 32 * ks + 8 * lg));
        b[ks] = lds_ld16(T16, a16_idx(16 * nt + lr, 32 * ks + 8 * lg));
      }
      acc[0] = zero4();
      acc[1] = zero4();
      MFMA_SB(acc[0], acc[1], a0, a1, b, nks);
      u32x4 fr;
      fr[0] = pk_bf16(-acc[0][0], -acc[0][1]);
      fr[1] = pk_bf16(-acc[0][2], -acc[0][3]);
      fr[2] = pk_bf16(-acc[1][0], -acc[1][1]);
      fr[3] = pk_bf16(-acc[1][2], -acc[1][3]);
      gst16(gb + GB_W + (size_t)((nt * 4 + wu) * 64 + lane) * 16, fr);
    }
    for (int f = wu; f < 6; f += 4) {
      const int mt = f < 2 ? f : 2 + ((f - 2) >> 1);
      const int ks = f < 2 ? 0 : ((f - 2) & 1);
      gst16(gb + GB_T + (size_t)(f * 64 + lane) * 16, lds_ld16(T16, a16_idx(16 * mt + lr, 32 * ks + 8 * lg)));
    }
  }
}
DEVI void kda_intra_body_np(
    char* smem,
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ g,
    const uint16_t* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const int* __restrict__ cu,
    char* __restrict__ gbuf,
    int T,
    int H,
    int N,
    float scale,
    int nwork) {
  const int tid = TID;
  const int w = tid >> 6;
  const int lane = tid & 63;
  const int lg = lane >> 4;
  const int lr = lane & 15;
  int nreal = intra_nreal(cu, T, H, N);
  nreal = nreal < nwork ? nreal : nwork;
  const int b = BID;
  if (b >= nreal) return;
  IntraRaw raw;
  const IntraItem cit = intra_decode(nreal - 1 - b, cu, T, H, N);
  intra_load_qkg(raw, cit, q, k, g, beta, A_log, dt_bias, H, w, lg, lr);
  intra_process(smem, cit, raw, gbuf, H, scale);
}

// =====================================================================================================================
// 4. Recurrence step: chunk fetch, LDS staging and one chunk of the state update
// =====================================================================================================================

template <int VS>
struct RecurPF {
  static constexpr int NT = 512 / VS;
  static constexpr int NU = GB_SH / 16;
  static constexpr int NL = (NU + NT - 1) / NT;
  static constexpr int NCOL = 128 / VS;
  static constexpr int NCG = NCOL / 8;
  static constexpr int STRIDE = GB_SH + NCOL * TR_LD * 2;
};
template <int VS, int K0, int K1>
DEVI void recur_fetch_units(u32x4* P, const char* __restrict__ nb, int tid) {
  constexpr int NT = RecurPF<VS>::NT;
  constexpr int NL = RecurPF<VS>::NL;
  constexpr int NU = RecurPF<VS>::NU;
#pragma unroll
  for (int kk = K0; kk < K1; ++kk) {
    int u = tid + NT * kk;
    if (kk == NL - 1) u = u < NU ? u : NU - 1;
    P[kk] = *(const u32x4*)(nb + (size_t)u * 16);
  }
}
template <int VS>
DEVI void recur_fetch_vb(
    u32x4* VP,
    u32x2& BP,
    const char* __restrict__ nb,
    const uint16_t* __restrict__ v,
    long t0,
    int L,
    int H,
    int h,
    int hf,
    int tid) {
  constexpr int NCG = RecurPF<VS>::NCG;
  const int cg = tid % NCG;
  const int r0 = 2 * (tid / NCG);
  const long ta = (r0 < L) ? t0 + r0 : t0;
  const long tb = (r0 + 1 < L) ? t0 + r0 + 1 : t0;
  const int col = RecurPF<VS>::NCOL * hf + 8 * cg;
  VP[0] = *(const u32x4*)(v + ((size_t)ta * H + h) * KD + col);
  VP[1] = *(const u32x4*)(v + ((size_t)tb * H + h) * KD + col);
  BP = *(const u32x2*)(nb + GB_B + r0 * 4);
}
template <int VS>
DEVI void recur_fetch(
    u32x4* P,
    u32x4* VP,
    u32x2& BP,
    const char* __restrict__ nb,
    const uint16_t* __restrict__ v,
    long t0,
    int L,
    int H,
    int h,
    int hf,
    int tid) {
  recur_fetch_units<VS, 0, RecurPF<VS>::NL>(P, nb, tid);
  recur_fetch_vb<VS>(VP, BP, nb, v, t0, L, H, h, hf, tid);
}
template <int VS, int K0, int K1>
DEVI void recur_stage_units(char* LB, const u32x4* P, int tid) {
  constexpr int NT = RecurPF<VS>::NT;
  constexpr int NL = RecurPF<VS>::NL;
  constexpr int NU = RecurPF<VS>::NU;
#pragma unroll
  for (int kk = K0; kk < K1; ++kk) {
    int u = tid + NT * kk;
    if (kk == NL - 1) u = u < NU ? u : NU - 1;
    *(u32x4*)(LB + u * 16) = P[kk];
  }
}
template <int VS>
DEVI void recur_stage_vt(char* LB, const u32x4* VP, u32x2 BP, int tid) {
  constexpr int NCG = RecurPF<VS>::NCG;
  uint16_t* VT = (uint16_t*)(LB + LB_VT);
  const int cg = tid % NCG;
  const int r0 = 2 * (tid / NCG);
  const float b0 = __uint_as_float(BP[0]), b1 = __uint_as_float(BP[1]);
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const uint32_t a = VP[0][p], c = VP[1][p];
    *(uint32_t*)(VT + tr_idx(8 * cg + 2 * p, r0)) = pk_bf16(bf_lo(a) * b0, bf_lo(c) * b1);
    *(uint32_t*)(VT + tr_idx(8 * cg + 2 * p + 1, r0)) = pk_bf16(bf_hi(a) * b0, bf_hi(c) * b1);
  }
}
template <int VS>
DEVI void recur_stage(char* LB, const u32x4* P, const u32x4* VP, u32x2 BP, int tid) {
  recur_stage_units<VS, 0, RecurPF<VS>::NL>(LB, P, tid);
  recur_stage_vt<VS>(LB, VP, BP, tid);
}
DEVI void recur_chunk(
    f32x4* S,
    const char* LB,
    uint16_t* __restrict__ o,
    char* __restrict__ trash,
    long t0,
    int L,
    int H,
    int h,
    int sw,
    int wl,
    int lane,
    int lg,
    int lr) {
  f32x4 gv[8];
#pragma unroll
  for (int mt = 0; mt < 8; ++mt)
    gv[mt] = *(const f32x4*)(LB + GB_G + (16 * mt + 4 * lg) * 4);
  f32x4 U[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    U[mt] = zero4();
  {
    const uint16_t* VT = (const uint16_t*)(LB + LB_VT);
    const int cl = 16 * wl + lr;
    u32x4 vb[2], ta0[1], ta1[1], ta2[2], ta3[2];
    vb[0] = lds_ld16(VT, tr_idx(cl, 8 * lg));
    vb[1] = lds_ld16(VT, tr_idx(cl, 32 + 8 * lg));
    ta0[0] = *(const u32x4*)(LB + GB_T + (fa_idx(0, 0) * 64 + lane) * 16);
    ta1[0] = *(const u32x4*)(LB + GB_T + (fa_idx(1, 0) * 64 + lane) * 16);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      ta2[ks] = *(const u32x4*)(LB + GB_T + (fa_idx(2, ks) * 64 + lane) * 16);
      ta3[ks] = *(const u32x4*)(LB + GB_T + (fa_idx(3, ks) * 64 + lane) * 16);
    }
    MFMA_SB(U[0], U[1], ta0, ta1, vb, 1);
    MFMA_SB(U[2], U[3], ta2, ta3, vb, 2);
  }
  u32x4 Sb[4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
    Sb[ks] = acc_pair_to_bf16(S[2 * ks], S[2 * ks + 1]);
#pragma unroll
  for (int mt = 0; mt < 8; ++mt) {
    S[mt][0] *= gv[mt][0];
    S[mt][1] *= gv[mt][1];
    S[mt][2] *= gv[mt][2];
    S[mt][3] *= gv[mt][3];
  }
#pragma unroll
  for (int mp = 0; mp < 4; mp += 2) {
    u32x4 a0[4], a1[4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      a0[ks] = *(const u32x4*)(LB + GB_W + ((mp * 4 + ks) * 64 + lane) * 16);
      a1[ks] = *(const u32x4*)(LB + GB_W + (((mp + 1) * 4 + ks) * 64 + lane) * 16);
    }
    MFMA_SB(U[mp], U[mp + 1], a0, a1, Sb, 4);
  }
  f32x4 o_acc[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    o_acc[mt] = zero4();
#pragma unroll
  for (int mp = 0; mp < 4; mp += 2) {
    u32x4 b0[4], b1[4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      b0[ks] = *(const u32x4*)(LB + GB_Q + ((mp * 4 + ks) * 64 + lane) * 16);
      b1[ks] = *(const u32x4*)(LB + GB_Q + (((mp + 1) * 4 + ks) * 64 + lane) * 16);
    }
    MFMA_SA(o_acc[mp], o_acc[mp + 1], Sb, b0, b1, 4);
  }
  u32x4 Ub[2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks)
    Ub[ks] = acc_pair_to_bf16(U[2 * ks], U[2 * ks + 1]);
  {
    u32x4 q0[1], q1[1], q2[2], q3[2];
    q0[0] = *(const u32x4*)(LB + GB_A + (fa_idx(0, 0) * 64 + lane) * 16);
    q1[0] = *(const u32x4*)(LB + GB_A + (fa_idx(1, 0) * 64 + lane) * 16);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      q2[ks] = *(const u32x4*)(LB + GB_A + (fa_idx(2, ks) * 64 + lane) * 16);
      q3[ks] = *(const u32x4*)(LB + GB_A + (fa_idx(3, ks) * 64 + lane) * 16);
    }
    MFMA_SA(o_acc[0], o_acc[1], Ub, q0, q1, 1);
    MFMA_SA(o_acc[2], o_acc[3], Ub, q2, q3, 2);
  }
  char* const tr = trash + lane * 8;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    const int i = 16 * mt + lr;
    const bool ok = i < L;
    const long row = ok ? t0 + i : t0;
    char* const real = (char*)(o + ((size_t)row * H + h) * KD + 16 * sw + 4 * lg);
    char* dst = ok ? real : tr;
    *(u32x2*)dst = pack4(o_acc[mt][0], o_acc[mt][1], o_acc[mt][2], o_acc[mt][3]);
  }
#pragma unroll
  for (int mp = 0; mp < 8; mp += 2) {
    u32x4 a0[2], a1[2];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      a0[ks] = *(const u32x4*)(LB + GB_K + ((mp * 2 + ks) * 64 + lane) * 16);
      a1[ks] = *(const u32x4*)(LB + GB_K + (((mp + 1) * 2 + ks) * 64 + lane) * 16);
    }
    MFMA_SB(S[mp], S[mp + 1], a0, a1, Ub, 2);
  }
}
#define SMEM_B4 (2 * (GB_SH + 32 * TR_LD * 2))
template <int VS>
struct StageP1 {
  char* LBn;
  const u32x4* P;
  const u32x4* VP;
  u32x2 BP;
  int tid;
  DEVM void half0() {
    recur_stage_units<VS, 0, RecurPF<VS>::NL / 2>(LBn, P, tid);
  }
  DEVM void half1() {
    recur_stage_units<VS, RecurPF<VS>::NL / 2, RecurPF<VS>::NL>(LBn, P, tid);
  }
  DEVM void vt() {
    recur_stage_vt<VS>(LBn, VP, BP, tid);
  }
};
template <int VS>
struct StageRoll {
  char* LBn;
  u32x4* P;
  u32x4* VP;
  u32x2& BP;
  int tid;
  const char* nb2;
  const uint16_t* v;
  long t02;
  int L2, H, h, hf;
  DEVM void half0() {
    recur_stage_units<VS, 0, RecurPF<VS>::NL / 2>(LBn, P, tid);
    recur_fetch_units<VS, 0, RecurPF<VS>::NL / 2>(P, nb2, tid);
  }
  DEVM void half1() {
    recur_stage_units<VS, RecurPF<VS>::NL / 2, RecurPF<VS>::NL>(LBn, P, tid);
    recur_fetch_units<VS, RecurPF<VS>::NL / 2, RecurPF<VS>::NL>(P, nb2, tid);
  }
  DEVM void vt() {
    recur_stage_vt<VS>(LBn, VP, BP, tid);
    recur_fetch_vb<VS>(VP, BP, nb2, v, t02, L2, H, h, hf, tid);
  }
};
struct StageNone {
  DEVM void half0() {}
  DEVM void half1() {}
  DEVM void vt() {}
};
// OUT: compute and store the outputs; UV: the step has values (U starts from T (beta v), else from zero)
template <int VS, class ST, bool OUT = true, bool UV = true>
DEVI void recur_chunk_p(
    f32x4* S,
    const char* LB,
    ST& st,
    uint16_t* __restrict__ o,
    char* __restrict__ trash,
    long t0,
    int L,
    int H,
    int h,
    int sw,
    int wl,
    int lane,
    int lg,
    int lr) {
  const char* const LW = LB + GB_W + lane * 16;
  const char* const LQ = LB + GB_Q + lane * 16;
  const char* const LK = LB + GB_K + lane * 16;
  const char* const LA = LB + GB_A + lane * 16;
  const char* const LT = LB + GB_T + lane * 16;
  const char* const LG = LB + GB_G + lg * 16;
  u32x4 vb[2], ta0[1], ta1[1], ta2[2], ta3[2];
  if constexpr (UV) {
    const uint16_t* VT = (const uint16_t*)(LB + LB_VT);
    const int cl = 16 * wl + lr;
    vb[0] = lds_ld16(VT, tr_idx(cl, 8 * lg));
    vb[1] = lds_ld16(VT, tr_idx(cl, 32 + 8 * lg));
    ta0[0] = *(const u32x4*)(LT + fa_idx(0, 0) * 1024);
    ta1[0] = *(const u32x4*)(LT + fa_idx(1, 0) * 1024);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      ta2[ks] = *(const u32x4*)(LT + fa_idx(2, ks) * 1024);
      ta3[ks] = *(const u32x4*)(LT + fa_idx(3, ks) * 1024);
    }
  }
  u32x4 Sb[4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
    Sb[ks] = acc_pair_to_bf16(S[2 * ks], S[2 * ks + 1]);
  u32x4 wa[4], wb[4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    wa[ks] = *(const u32x4*)(LW + (0 * 4 + ks) * 1024);
    wb[ks] = *(const u32x4*)(LW + (1 * 4 + ks) * 1024);
  }
  SCHED_FENCE();
  f32x4 U[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    U[mt] = zero4();
  if constexpr (UV) {
    MFMA_SB(U[0], U[1], ta0, ta1, vb, 1);
    MFMA_SB(U[2], U[3], ta2, ta3, vb, 2);
  }
  SCHED_FENCE();
  u32x4 wc[4], wd[4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    wc[ks] = *(const u32x4*)(LW + (2 * 4 + ks) * 1024);
    wd[ks] = *(const u32x4*)(LW + (3 * 4 + ks) * 1024);
  }
  SCHED_FENCE();
  MFMA_SB(U[0], U[1], wa, wb, Sb, 4);
  SCHED_FENCE();
  u32x4 qa[4], qb[4];
  if constexpr (OUT) {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      qa[ks] = *(const u32x4*)(LQ + (0 * 4 + ks) * 1024);
      qb[ks] = *(const u32x4*)(LQ + (1 * 4 + ks) * 1024);
    }
  }
  SCHED_FENCE();
  MFMA_SB(U[2], U[3], wc, wd, Sb, 4);
  SCHED_FENCE();
  u32x4 qc[4], qd[4];
  if constexpr (OUT) {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      qc[ks] = *(const u32x4*)(LQ + (2 * 4 + ks) * 1024);
      qd[ks] = *(const u32x4*)(LQ + (3 * 4 + ks) * 1024);
    }
  }
  SCHED_FENCE();
  f32x4 o_acc[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    o_acc[mt] = zero4();
  if constexpr (OUT) MFMA_SA(o_acc[0], o_acc[1], Sb, qa, qb, 4);
  u32x4 Ub[2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks)
    Ub[ks] = acc_pair_to_bf16(U[2 * ks], U[2 * ks + 1]);
  SCHED_FENCE();
  u32x4 aq0[1], aq1[1], aq2[2], aq3[2];
  if constexpr (OUT) {
    aq0[0] = *(const u32x4*)(LA + fa_idx(0, 0) * 1024);
    aq1[0] = *(const u32x4*)(LA + fa_idx(1, 0) * 1024);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      aq2[ks] = *(const u32x4*)(LA + fa_idx(2, ks) * 1024);
      aq3[ks] = *(const u32x4*)(LA + fa_idx(3, ks) * 1024);
    }
  }
  SCHED_FENCE();
  if constexpr (OUT) MFMA_SA(o_acc[2], o_acc[3], Sb, qc, qd, 4);
  SCHED_FENCE();
  f32x4 gv[8];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
    gv[mt] = *(const f32x4*)(LG + mt * 64);
  u32x4 ka[2], kb[2], kc[2], kd[2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    ka[ks] = *(const u32x4*)(LK + (0 * 2 + ks) * 1024);
    kb[ks] = *(const u32x4*)(LK + (1 * 2 + ks) * 1024);
    kc[ks] = *(const u32x4*)(LK + (2 * 2 + ks) * 1024);
    kd[ks] = *(const u32x4*)(LK + (3 * 2 + ks) * 1024);
  }
  SCHED_FENCE();
  if constexpr (OUT) {
    MFMA_SA(o_acc[0], o_acc[1], Ub, aq0, aq1, 1);
    MFMA_SA(o_acc[2], o_acc[3], Ub, aq2, aq3, 2);
    char* const tr = trash + lane * 8;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const int i = 16 * mt + lr;
      const bool ok = i < L;
      const long row = ok ? t0 + i : t0;
      char* const real = (char*)(o + ((size_t)row * H + h) * KD + 16 * sw + 4 * lg);
      char* dst = ok ? real : tr;
      *(u32x2*)dst = pack4(o_acc[mt][0], o_acc[mt][1], o_acc[mt][2], o_acc[mt][3]);
    }
  }
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    S[mt][0] *= gv[mt][0];
    S[mt][1] *= gv[mt][1];
    S[mt][2] *= gv[mt][2];
    S[mt][3] *= gv[mt][3];
  }
  st.half0();
  SCHED_FENCE();
#pragma unroll
  for (int mt = 4; mt < 8; ++mt)
    gv[mt] = *(const f32x4*)(LG + mt * 64);
  u32x4 ke[2], kf[2], kg[2], kh[2];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    ke[ks] = *(const u32x4*)(LK + (4 * 2 + ks) * 1024);
    kf[ks] = *(const u32x4*)(LK + (5 * 2 + ks) * 1024);
    kg[ks] = *(const u32x4*)(LK + (6 * 2 + ks) * 1024);
    kh[ks] = *(const u32x4*)(LK + (7 * 2 + ks) * 1024);
  }
  SCHED_FENCE();
  MFMA_SB(S[0], S[1], ka, kb, Ub, 2);
  MFMA_SB(S[2], S[3], kc, kd, Ub, 2);
#pragma unroll
  for (int mt = 4; mt < 8; ++mt) {
    S[mt][0] *= gv[mt][0];
    S[mt][1] *= gv[mt][1];
    S[mt][2] *= gv[mt][2];
    S[mt][3] *= gv[mt][3];
  }
  st.half1();
  SCHED_FENCE();
  MFMA_SB(S[4], S[5], ke, kf, Ub, 2);
  MFMA_SB(S[6], S[7], kg, kh, Ub, 2);
  st.vt();
}
DEVI void store_recur_state(f32x4* S, float* state, int n, int H, int h, int sw, int lg, int lr) {
  float* dst = state + (size_t)(n * H + h) * (KD * KD);
#pragma unroll
  for (int mt = 0; mt < 8; ++mt)
    *(f32x4*)(dst + (16 * sw + lr) * KD + 16 * mt + 4 * lg) = S[mt];
}

// The sequence of (sequence, head) pair p: with N <= 64 the walks take the sequences longest first (lane m of the
// wave ranks sequence m), otherwise in input order.  pre counts the chunks of the sequences before it.  Every lane
// of the wave must call it.
struct RecurSeq {
  int n, nch;
  long bos, len, pre;
};
DEVI RecurSeq recur_seq(int p, const int* __restrict__ cu, int N) {
  RecurSeq s;
  s.n = p;
  s.pre = 0;
  if (N <= 64) {
    const int lane = TID & 63;
    const int len = lane < N ? cu[lane + 1] - cu[lane] : -1;
    int rank = 0;
    for (int m = 0; m < N; ++m) {
      const int lm = __builtin_amdgcn_readlane(len, m);
      rank += (lm > len || (lm == len && m < lane)) ? 1 : 0;
    }
    const unsigned long long hit = __ballot(lane < N && rank == p);
    if (hit) s.n = __ffsll(hit) - 1;
    const int nch = lane < N ? (len + KC - 1) / KC : 0;
    for (int m = 0; m < s.n; ++m)
      s.pre += __builtin_amdgcn_readlane(nch, m);
  } else {
    for (int m = 0; m < s.n; ++m)
      s.pre += ((long)cu[m + 1] - (long)cu[m] + KC - 1) / KC;
  }
  s.bos = (long)cu[s.n];
  s.len = (long)cu[s.n + 1] - s.bos;
  s.nch = (int)((s.len + KC - 1) / KC);
  return s;
}

// =====================================================================================================================
// 5. Whole-head and value-half walks (vs1, vs2)
// =====================================================================================================================

template <int VS, int PFD>
DEVI void kda_recur_body_t(
    char* smem,
    const char* __restrict__ gbuf,
    const uint16_t* __restrict__ v,
    const float* h0,
    const int* __restrict__ cu,
    const int* __restrict__ slots,
    uint16_t* __restrict__ o,
    float* ht,
    float* __restrict__ track_state,
    const long long* __restrict__ track_chunk_idx,
    char* __restrict__ trash,
    int T,
    int H,
    int N,
    int bid) {
  constexpr int NL = RecurPF<VS>::NL;
  constexpr int LBS = RecurPF<VS>::STRIDE;
  const int tid = TID;
  const int w = tid >> 6;
  const int lane = tid & 63;
  const int lg = lane >> 4;
  const int lr = lane & 15;
  int hf = 0, ph = bid;
  if (VS == 2) {
    const int full = (N * H * 2) & ~15;
    if (bid < full) {
      hf = (bid >> 3) & 1;
      ph = ((bid >> 4) << 3) | (bid & 7);
    } else {
      hf = (bid - full) & 1;
      ph = (full >> 1) + ((bid - full) >> 1);
    }
  }
  const int p = ph / H;
  const int h = ph - p * H;
  const int sw = w + (8 / VS) * hf;
  const RecurSeq sq = recur_seq(p, cu, N);
  const int n = sq.n;
  const long bos = sq.bos;
  const long pre = sq.pre;
  const int len = (int)sq.len;
  const int nch = sq.nch;
  f32x4 S[8];
  const float* h0p = h0 + ((size_t)slots[n] * H + h) * (KD * KD);
#pragma unroll
  for (int mt = 0; mt < 8; ++mt)
    S[mt] = *(const f32x4*)(h0p + (16 * sw + lr) * KD + 16 * mt + 4 * lg);
  const long long track_c = track_chunk_idx ? track_chunk_idx[n] : -1;
  if (nch > 0) {
    const size_t bstride = (size_t)H * GB_BYTES;
    const char* gbase = gbuf + ((size_t)pre * H + h) * GB_BYTES;
#define CH_L(c) ((len - (c) * KC) < KC ? (len - (c) * KC) : KC)
#define CH_FETCH(Pd, VPd, BPd, c) \
  recur_fetch<VS>(Pd, VPd, BPd, gbase + (size_t)(c) * bstride, v, bos + (long)(c) * KC, CH_L(c), H, h, hf, tid)
    if (PFD == 2) {
      u32x4 P0[NL], P1[NL], VP0[2], VP1[2];
      u32x2 BP0, BP1;
      CH_FETCH(P0, VP0, BP0, 0);
      CH_FETCH(P1, VP1, BP1, (nch > 1 ? 1 : 0));
      recur_stage<VS>(smem, P0, VP0, BP0, tid);
      SYNC();
      int c = 0;
      for (; c + 2 <= nch; c += 2) {
        {
          const int cn = (c + 2 < nch) ? c + 2 : nch - 1;
          CH_FETCH(P0, VP0, BP0, cn);
          recur_chunk(S, smem, o, trash, bos + (long)c * KC, CH_L(c), H, h, sw, w, lane, lg, lr);
          recur_stage<VS>(smem + LBS, P1, VP1, BP1, tid);
          SYNC();
        }
        {
          const int cn = (c + 3 < nch) ? c + 3 : nch - 1;
          CH_FETCH(P1, VP1, BP1, cn);
          const int c1 = c + 1;
          recur_chunk(S, smem + LBS, o, trash, bos + (long)c1 * KC, CH_L(c1), H, h, sw, w, lane, lg, lr);
          recur_stage<VS>(smem, P0, VP0, BP0, tid);
          SYNC();
        }
      }
      if (c < nch) recur_chunk(S, smem, o, trash, bos + (long)c * KC, CH_L(c), H, h, sw, w, lane, lg, lr);
    } else {
      u32x4 P0[NL], VP0[2];
      u32x2 BP0;
      CH_FETCH(P0, VP0, BP0, 0);
      recur_stage<VS>(smem, P0, VP0, BP0, tid);
      if constexpr (VS == 2) {
        CH_FETCH(P0, VP0, BP0, (nch > 1 ? 1 : 0));
        SYNC();
        for (int c = 0; c < nch; ++c) {
          if (track_state && c == track_c) store_recur_state(S, track_state, n, H, h, sw, lg, lr);
          const int c2 = (c + 2 < nch) ? c + 2 : nch - 1;
          StageRoll<VS> st{
              smem + ((c + 1) & 1) * LBS,
              P0,
              VP0,
              BP0,
              tid,
              gbase + (size_t)c2 * bstride,
              v,
              bos + (long)c2 * KC,
              CH_L(c2),
              H,
              h,
              hf};
          recur_chunk_p<VS>(
              S, smem + (c & 1) * LBS, st, o, trash, bos + (long)c * KC, CH_L(c), H, h, sw, w, lane, lg, lr);
          SYNC();
        }
      } else {
        SYNC();
        for (int c = 0; c < nch; ++c) {
          if (track_state && c == track_c) store_recur_state(S, track_state, n, H, h, sw, lg, lr);
          const int cn = (c + 1 < nch) ? c + 1 : nch - 1;
          CH_FETCH(P0, VP0, BP0, cn);
          StageP1<VS> st{smem + ((c + 1) & 1) * LBS, P0, VP0, BP0, tid};
          recur_chunk_p<VS>(
              S, smem + (c & 1) * LBS, st, o, trash, bos + (long)c * KC, CH_L(c), H, h, sw, w, lane, lg, lr);
          SYNC();
        }
      }
    }
#undef CH_FETCH
#undef CH_L
  }
  float* htp = ht + ((size_t)slots[n] * H + h) * (KD * KD);
#pragma unroll
  for (int mt = 0; mt < 8; ++mt)
    *(f32x4*)(htp + (16 * sw + lr) * KD + 16 * mt + 4 * lg) = S[mt];
}

// =====================================================================================================================
// 6. Warp-specialized walks (ws2, ws4)
// =====================================================================================================================

template <int VS>
DEVI void recur_wg_map(int bid, int N, int H, int& ph, int& pt) {
  const int full = (N * H * VS) & ~(8 * VS - 1);
  if (bid < full) {
    pt = (bid >> 3) & (VS - 1);
    ph = ((bid / (8 * VS)) << 3) | (bid & 7);
  } else {
    pt = (bid - full) % VS;
    ph = full / VS + (bid - full) / VS;
  }
}
template <int VS>
DEVI void kda_recur_body_ws(
    char* smem,
    const char* __restrict__ gbuf,
    const uint16_t* __restrict__ v,
    const float* h0,
    const int* __restrict__ cu,
    const int* __restrict__ slots,
    uint16_t* __restrict__ o,
    float* ht,
    char* __restrict__ trash,
    int T,
    int H,
    int N,
    int bid,
    int nmap) {
  constexpr int NM = 8 / VS;
  constexpr int NL = RecurPF<VS>::NL;
  constexpr int LBS = RecurPF<VS>::STRIDE;
  const int tid = TID;
  const int wu = WAVE_U();
  const int lane = tid & 63;
  const int lg = lane >> 4;
  const int lr = lane & 15;
  int ph, pt;
  recur_wg_map<VS>(bid, nmap, H, ph, pt);
  const int p = ph / H;
  const int h = ph - p * H;
  const RecurSeq sq = recur_seq(p, cu, N);
  const int n = sq.n;
  const long bos = sq.bos;
  const int len = (int)sq.len;
  const int nch = sq.nch;
  const size_t bstride = (size_t)H * GB_BYTES;
  const char* gbase = gbuf + ((size_t)sq.pre * H + h) * GB_BYTES;
#define CH_L(c) ((len - (c) * KC) < KC ? (len - (c) * KC) : KC)
  if (wu < NM) {
    const int sw = wu + NM * pt;
    f32x4 S[8];
    const float* h0p = h0 + ((size_t)slots[n] * H + h) * (KD * KD);
#pragma unroll
    for (int mt = 0; mt < 8; ++mt)
      S[mt] = *(const f32x4*)(h0p + (16 * sw + lr) * KD + 16 * mt + 4 * lg);
    if (nch > 0) {
      StageNone st;
      SYNC();
      for (int c = 0; c < nch; ++c) {
        recur_chunk_p<VS>(
            S, smem + (c & 1) * LBS, st, o, trash, bos + (long)c * KC, CH_L(c), H, h, sw, wu, lane, lg, lr);
        SYNC();
      }
    }
    float* htp = ht + ((size_t)slots[n] * H + h) * (KD * KD);
#pragma unroll
    for (int mt = 0; mt < 8; ++mt)
      *(f32x4*)(htp + (16 * sw + lr) * KD + 16 * mt + 4 * lg) = S[mt];
  } else if (nch > 0) {
    const int lt = tid - 64 * NM;
#define CH_FETCH_L(Pd, VPd, BPd, c) \
  recur_fetch<VS>(Pd, VPd, BPd, gbase + (size_t)(c) * bstride, v, bos + (long)(c) * KC, CH_L(c), H, h, pt, lt)
    u32x4 P0[NL], VP0[2];
    u32x2 BP0;
    CH_FETCH_L(P0, VP0, BP0, 0);
    recur_stage<VS>(smem, P0, VP0, BP0, lt);
    CH_FETCH_L(P0, VP0, BP0, (nch > 1 ? 1 : 0));
    SYNC();
    for (int c = 0; c < nch; ++c) {
      LOADER_SLEEP();
      recur_stage<VS>(smem + ((c + 1) & 1) * LBS, P0, VP0, BP0, lt);
      const int c2 = (c + 2 < nch) ? c + 2 : nch - 1;
      CH_FETCH_L(P0, VP0, BP0, c2);
      SYNC();
    }
#undef CH_FETCH_L
  }
#undef CH_L
}
// =====================================================================================================================
// 7. Sequence groups (grp)
// =====================================================================================================================

// block units (16 bytes) a step without outputs reads, LM 1: W, Kg, T and the gates; LM 2 (zero values): W, Kg
// and the gates.  Unit u of that list is unit grp_unit(u) of the block.
template <int VS, int LM>
struct GrpPF {
  static constexpr int NU = LM == 1 ? 2464 : 2080;
  static constexpr int NL = (NU + RecurPF<VS>::NT - 1) / RecurPF<VS>::NT;
};
template <int LM>
DEVI int grp_unit(int u) {
  return u < GB_Q / 16 ? u : (u < 2 * GB_Q / 16 ? u + GB_Q / 16 : u + (LM == 1 ? GB_T - GB_K : GB_G - GB_K) / 16);
}
template <int VS, int LM, int K0, int K1>
DEVI void grp_fetch_units(u32x4* P, const char* __restrict__ nb, int tid) {
#pragma unroll
  for (int kk = K0; kk < K1; ++kk) {
    const int u = tid + RecurPF<VS>::NT * kk;
    P[kk] = *(const u32x4*)(nb + (size_t)grp_unit<LM>(u < GrpPF<VS, LM>::NU ? u : GrpPF<VS, LM>::NU - 1) * 16);
  }
}
template <int VS, int LM, int K0, int K1>
DEVI void grp_stage_units(char* LB, const u32x4* P, int tid) {
#pragma unroll
  for (int kk = K0; kk < K1; ++kk) {
    const int u = tid + RecurPF<VS>::NT * kk;
    *(u32x4*)(LB + grp_unit<LM>(u < GrpPF<VS, LM>::NU ? u : GrpPF<VS, LM>::NU - 1) * 16) = P[kk];
  }
}
// math waves of a sequence-group walk in value parts (OUT, UV as in recur_chunk_p); the state at the top of
// chunk tc is stored to track_state
template <int VS, bool OUT, bool UV>
DEVI void grp_walk(
    f32x4* S,
    char* smem,
    uint16_t* __restrict__ o,
    char* __restrict__ trash,
    float* __restrict__ track_state,
    long tc,
    long bos,
    long len,
    int nch,
    int H,
    int h,
    int sw,
    int wu,
    int lane,
    int lg,
    int lr) {
  StageNone st;
  SYNC();
  for (int c = 0; c < nch; ++c) {
    if (OUT && c == tc) store_recur_state(S, track_state, 0, H, h, sw, lg, lr);
    const int L = (len - (long)c * KC) < KC ? (int)(len - (long)c * KC) : KC;
    recur_chunk_p<VS, StageNone, OUT, UV>(
        S, smem + (c & 1) * RecurPF<VS>::STRIDE, st, o, trash, bos + (long)c * KC, L, H, h, sw, wu, lane, lg, lr);
    SYNC();
  }
}
// global loads and LDS staging of one chunk for a sequence-group walk: LM 0 the whole block and the values,
// 1 the units of GrpPF and the values, 2 the units of GrpPF only
template <int VS, int LM>
struct GrpChunk {
  static constexpr int NL = LM ? GrpPF<VS, LM>::NL : RecurPF<VS>::NL;
  u32x4 P[NL], VP[2];
  u32x2 BP;
  DEVM void fetch(const char* nb, const uint16_t* __restrict__ v, long t0, int L, int H, int h, int hf, int tid) {
    if constexpr (LM == 0)
      recur_fetch_units<VS, 0, NL>(P, nb, tid);
    else
      grp_fetch_units<VS, LM, 0, NL>(P, nb, tid);
    if constexpr (LM != 2) recur_fetch_vb<VS>(VP, BP, nb, v, t0, L, H, h, hf, tid);
  }
  template <int K0, int K1>
  DEVM void stage_units(char* LB, int tid) const {
    if constexpr (LM == 0)
      recur_stage_units<VS, K0, K1>(LB, P, tid);
    else
      grp_stage_units<VS, LM, K0, K1>(LB, P, tid);
  }
  DEVM void stage_vt(char* LB, int tid) const {
    if constexpr (LM != 2) recur_stage_vt<VS>(LB, VP, BP, tid);
  }
};
// staging hooks of recur_chunk_p for the whole-head walk: the next chunk is staged during the step
template <int LM>
struct StageGrp {
  char* LBn;
  const GrpChunk<1, LM>& ch;
  int tid;
  DEVM void half0() {
    ch.template stage_units<0, GrpChunk<1, LM>::NL / 2>(LBn, tid);
  }
  DEVM void half1() {
    ch.template stage_units<GrpChunk<1, LM>::NL / 2, GrpChunk<1, LM>::NL>(LBn, tid);
  }
  DEVM void vt() {
    ch.stage_vt(LBn, tid);
  }
};
// whole-head sequence-group walk (VS 1, all eight waves do math and loads, one chunk of loads in flight)
template <int LM, bool OUT, bool UV>
DEVI void grp_walk1(
    f32x4* S,
    char* smem,
    const char* gbase,
    size_t bstride,
    const uint16_t* __restrict__ v,
    uint16_t* __restrict__ o,
    char* __restrict__ trash,
    float* __restrict__ track_state,
    long tc,
    long bos,
    long len,
    int nch,
    int H,
    int h,
    int tid,
    int lane,
    int lg,
    int lr) {
  const int w = tid >> 6;
  GrpChunk<1, LM> ch;
  auto fetch = [&](int c) {
    c = c < nch ? c : nch - 1;
    const int L = (len - (long)c * KC) < KC ? (int)(len - (long)c * KC) : KC;
    ch.fetch(gbase + (size_t)c * bstride, v, bos + (long)c * KC, L, H, h, 0, tid);
  };
  fetch(0);
  ch.template stage_units<0, GrpChunk<1, LM>::NL>(smem, tid);
  ch.stage_vt(smem, tid);
  SYNC();
  for (int c = 0; c < nch; ++c) {
    if (OUT && c == tc) store_recur_state(S, track_state, 0, H, h, w, lg, lr);
    fetch(c + 1);
    StageGrp<LM> st{smem + ((c + 1) & 1) * RecurPF<1>::STRIDE, ch, tid};
    const int L = (len - (long)c * KC) < KC ? (int)(len - (long)c * KC) : KC;
    recur_chunk_p<1, StageGrp<LM>, OUT, UV>(
        S, smem + (c & 1) * RecurPF<1>::STRIDE, st, o, trash, bos + (long)c * KC, L, H, h, w, w, lane, lg, lr);
    SYNC();
  }
}
// loader waves of a sequence-group walk in value parts: chunk c + 1 is staged while the math waves run chunk
// c, the loads of chunk c + 2 in flight
template <int VS, int LM>
DEVI void grp_load(
    char* smem,
    const char* gbase,
    size_t bstride,
    const uint16_t* __restrict__ v,
    long bos,
    long len,
    int nch,
    int H,
    int h,
    int pt,
    int lt) {
  GrpChunk<VS, LM> ch;
  auto fetch = [&](int c) {
    c = c < nch ? c : nch - 1;
    const int L = (len - (long)c * KC) < KC ? (int)(len - (long)c * KC) : KC;
    ch.fetch(gbase + (size_t)c * bstride, v, bos + (long)c * KC, L, H, h, pt, lt);
  };
  auto stage = [&](int c) {
    char* LB = smem + (c & 1) * RecurPF<VS>::STRIDE;
    ch.template stage_units<0, GrpChunk<VS, LM>::NL>(LB, lt);
    ch.stage_vt(LB, lt);
  };
  fetch(0);
  stage(0);
  fetch(1);
  SYNC();
  for (int c = 0; c < nch; ++c) {
    LOADER_SLEEP();
    stage(c + 1);
    fetch(c + 2);
    SYNC();
  }
}
// one sequence (N == 1) as G groups of gc chunks whose chunk chains are walked in parallel.  Pass 0: group 0
// from the initial state (outputs; its end state is the entry state E_1 of group 1) and, for groups 1..G-2,
// the end state B_g from a zero state and the transition M_g = prod_c (diag(decay_c) - W_c^T Kg_c) (the same
// step from the identity with zero values).  kda_recur_grp_scan chains E_{g+1} = E_g M_g + B_g, then pass 1
// walks groups 1..G-1 from E_g.  gst holds E, B and M as three [G, H, 128, 128] fp32 arrays.
template <int VS>
DEVI void kda_recur_body_grp(
    char* smem,
    const char* __restrict__ gbuf,
    const uint16_t* __restrict__ v,
    const float* h0,
    const int* __restrict__ cu,
    const int* __restrict__ slots,
    uint16_t* __restrict__ o,
    float* ht,
    float* __restrict__ track_state,
    const long long* __restrict__ track_chunk_idx,
    float* __restrict__ gst,
    char* __restrict__ trash,
    int H,
    int G,
    int gc,
    int pass,
    int bid) {
  constexpr int NM = 8 / VS;
  const int tid = TID;
  const int wu = WAVE_U();
  const int lane = tid & 63;
  const int lg = lane >> 4;
  const int lr = lane & 15;
  int ph, pt;
  recur_wg_map<VS>(bid, pass ? G - 1 : 2 * G - 3, H, ph, pt);
  const int it = ph / H;
  const int h = ph - it * H;
  // mode 0: true state (outputs, tracking), 1: zero state (B_g), 2: identity state and zero values (M_g)
  const int grp = pass ? it + 1 : (it < G - 1 ? it : it - G + 2);
  const int mode = pass ? 0 : (it < G - 1 ? (it ? 1 : 0) : 2);
  const long bos0 = (long)cu[0];
  const long tlen = (long)cu[1] - bos0;
  const int nch_all = (int)((tlen + KC - 1) / KC);
  const int c0 = grp * gc < nch_all ? grp * gc : nch_all;
  const int nch = nch_all - c0 < gc ? nch_all - c0 : gc;
  const long bos = bos0 + (long)c0 * KC;
  const long len = tlen - (long)c0 * KC;
  const size_t hs = (size_t)KD * KD;
  const size_t gs = (size_t)G * H * hs;
  const size_t bstride = (size_t)H * GB_BYTES;
  const char* gbase = gbuf + ((size_t)c0 * H + h) * GB_BYTES;
  if (wu < NM) {
    const int sw = wu + NM * pt;
    f32x4 S[8];
    const float* src =
        pass ? gst + ((size_t)grp * H + h) * hs : (mode == 0 ? h0 + ((size_t)slots[0] * H + h) * hs : nullptr);
    if (src) {
#pragma unroll
      for (int mt = 0; mt < 8; ++mt)
        S[mt] = *(const f32x4*)(src + (16 * sw + lr) * KD + 16 * mt + 4 * lg);
    } else {
#pragma unroll
      for (int mt = 0; mt < 8; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          S[mt][i] = (mode == 2 && 16 * sw + lr == 16 * mt + 4 * lg + i) ? 1.f : 0.f;
    }
    const long tc = (track_state && mode == 0) ? (long)track_chunk_idx[0] - c0 : -1;
    if (nch > 0) {
      if constexpr (VS == 1) {
        if (mode == 0)
          grp_walk1<0, true, true>(
              S, smem, gbase, bstride, v, o, trash, track_state, tc, bos, len, nch, H, h, tid, lane, lg, lr);
        else if (mode == 1)
          grp_walk1<1, false, true>(
              S, smem, gbase, bstride, v, o, trash, nullptr, -1, bos, len, nch, H, h, tid, lane, lg, lr);
        else
          grp_walk1<2, false, false>(
              S, smem, gbase, bstride, v, o, trash, nullptr, -1, bos, len, nch, H, h, tid, lane, lg, lr);
      } else {
        if (mode == 0)
          grp_walk<VS, true, true>(S, smem, o, trash, track_state, tc, bos, len, nch, H, h, sw, wu, lane, lg, lr);
        else if (mode == 1)
          grp_walk<VS, false, true>(S, smem, o, trash, nullptr, -1, bos, len, nch, H, h, sw, wu, lane, lg, lr);
        else
          grp_walk<VS, false, false>(S, smem, o, trash, nullptr, -1, bos, len, nch, H, h, sw, wu, lane, lg, lr);
      }
    }
    float* dst = pass ? (grp == G - 1 ? ht + ((size_t)slots[0] * H + h) * hs : nullptr)
                      : gst + (size_t)mode * gs + ((size_t)(mode ? grp : 1) * H + h) * hs;
    if (dst) {
#pragma unroll
      for (int mt = 0; mt < 8; ++mt)
        *(f32x4*)(dst + (16 * sw + lr) * KD + 16 * mt + 4 * lg) = S[mt];
    }
  } else if constexpr (VS > 1) {
    if (nch > 0) {
      const int lt = tid - 64 * NM;
      if (mode == 0)
        grp_load<VS, 0>(smem, gbase, bstride, v, bos, len, nch, H, h, pt, lt);
      else if (mode == 1)
        grp_load<VS, 1>(smem, gbase, bstride, v, bos, len, nch, H, h, pt, lt);
      else
        grp_load<VS, 2>(smem, gbase, bstride, v, bos, len, nch, H, h, pt, lt);
    }
  }
}

// =====================================================================================================================
// 8. Pipelined fused kernel (fused)
// =====================================================================================================================

#define F_BLK SMEM_A
#define SMEM_F (F_BLK + GB_SH + 128 * TR_LD * 2)
#define F_ZERO SMEM_F
#define SMEM_F2 (SMEM_F + 32)
#define OPQ_LOAD(p) __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)
#define OPQ_STORE(p, x) __hip_atomic_store((p), (x), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)
DEVI void opq_init(char* smem, int wv) {
  if ((TID & 63) == 0) OPQ_STORE((int*)(smem + F_ZERO + 4 * wv), 0);
  WAVE_SYNC();
}
DEVI int opq_pre(const char* smem, int wv) {
  return OPQ_LOAD((int*)(smem + F_ZERO + 4 * wv));
}
DEVI int opq_lane(int z) {
  return (TID + z) & 63;
}
DEVI float shfl_up64l(float x, int d, int lane) {
  return lane_shfl(x, lane >= d ? lane - d : lane);
}
DEVI void fused_load_rows(
    IntraRaw& r,
    long t0,
    int L,
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ g,
    const uint16_t* __restrict__ beta,
    int H,
    int h,
    int w,
    int lg,
    int lr) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int R = 16 * w + 4 * lg + i;
    long t = (R < L) ? (t0 + R) : t0;
    size_t off = ((size_t)t * H + h) * KD + 8 * lr;
    r.q[i] = *(const u32x4*)(q + off);
    r.k[i] = *(const u32x4*)(k + off);
    r.g[i] = *(const u32x4*)(g + off);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int R = 16 * w + 4 * lg + i;
    long t = (R < L) ? (t0 + R) : t0;
    r.b[i] = (uint32_t)beta[(size_t)t * H + h];
  }
}
DEVI void
fused_load_consts(IntraRaw& r, const float* __restrict__ A_log, const float* __restrict__ dt_bias, int h, int lr) {
  r.a_log = A_log[h];
  r.dt0 = *(const f32x4*)(dt_bias + h * KD + 8 * lr);
  r.dt1 = *(const f32x4*)(dt_bias + h * KD + 8 * lr + 4);
}
#define F_SYNC_RELANE()               \
  do {                                \
    const int z_ = opq_pre(smem, wu); \
    SYNC();                           \
    lane = opq_lane(z_);              \
    lg = lane >> 4;                   \
    lr = lane & 15;                   \
  } while (0)
DEVI void fused_fetch_v(u32x4* VP, const uint16_t* __restrict__ v, long t0, int L, int H, int h, int lt) {
  const int cg = lt & 15;
  const int r0 = 4 * (lt >> 4);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const long t = (r0 + i < L) ? t0 + r0 + i : t0;
    VP[i] = *(const u32x4*)(v + ((size_t)t * H + h) * KD + 8 * cg);
  }
}
DEVI void fused_stage_vt(char* VTB, const u32x4* VP, const float* BETA, int lt) {
  uint16_t* VT = (uint16_t*)VTB;
  const int cg = lt & 15;
  const int r0 = 4 * (lt >> 4);
  const f32x4 bb = *(const f32x4*)(BETA + r0);
#pragma unroll
  for (int hp = 0; hp < 2; ++hp) {
    const float b0 = bb[2 * hp], b1 = bb[2 * hp + 1];
    const u32x4 va = VP[2 * hp], vc = VP[2 * hp + 1];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint32_t a = va[p], c = vc[p];
      *(uint32_t*)(VT + tr_idx(8 * cg + 2 * p, r0 + 2 * hp)) = pk_bf16(bf_lo(a) * b0, bf_lo(c) * b1);
      *(uint32_t*)(VT + tr_idx(8 * cg + 2 * p + 1, r0 + 2 * hp)) = pk_bf16(bf_hi(a) * b0, bf_hi(c) * b1);
    }
  }
}
template <class PF>
DEVI void intra_process_fp(char* smem, int L, IntraRaw& cur, char* gb, float scale, const PF& pf, int z1) {
  uint16_t* L16 = (uint16_t*)smem;
  float* LF = (float*)smem;
  const int wu = WAVE_U();
  int lane = opq_lane(z1);
  int lg = lane >> 4;
  int lr = lane & 15;
  uint16_t* const QR = L16 + LA_QR / 2;
  uint16_t* const KR = L16 + LA_KR / 2;
  uint16_t* const KE = L16 + LA_KE / 2;
  uint16_t* const KGT = L16 + LA_KGT / 2;
  uint16_t* const T16 = L16 + LA_T16 / 2;
  float* const AKK = LF + LA_AKK / 4;
  float* const TFD = LF + LA_TFD / 4;
  float* const BETA = LF + LA_BETA / 4;
  float* const TAU = LF + LA_TAU / 4;
  float* const FACD = LF + LA_FACD / 4;
  const float a_h = EXP2(cur.a_log * LOG2E_F);
  const float a2 = -a_h * LOG2E_F;
  const float k1 = 1.0f / (-5.0f * LOG2E_F);
  f32x2 b2v[4];
  b2v[0] = sp2(a2) * mk2(cur.dt0[0], cur.dt0[1]);
  b2v[1] = sp2(a2) * mk2(cur.dt0[2], cur.dt0[3]);
  b2v[2] = sp2(a2) * mk2(cur.dt1[0], cur.dt1[1]);
  b2v[3] = sp2(a2) * mk2(cur.dt1[2], cur.dt1[3]);
  f32x2 lam2[4][4];
  float rqn[4], rkn[4], bsig[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const f32x2 x = bf2(cur.g[i][p]) * sp2(a2) + b2v[p];
      const f32x2 e = exp2v(x);
      lam2[i][p] = rcpv(e * sp2(k1) + sp2(k1));
    }
    bsig[i] = RCP(1.0f + EXP2(-bf_lo(cur.b[i]) * LOG2E_F));
  }
  pf.gate(cur, lg, lr);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f32x2 sq = sp2(0.f), sk = sp2(0.f);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const f32x2 qv = bf2(cur.q[i][p]);
      const f32x2 kv = bf2(cur.k[i][p]);
      sq += qv * qv;
      sk += kv * kv;
    }
    rqn[i] = sq[0] + sq[1];
    rkn[i] = sk[0] + sk[1];
  }
  if (L < KC) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bool ok = 16 * wu + 4 * lg + i < L;
#pragma unroll
      for (int p = 0; p < 4; ++p)
        lam2[i][p] = ok ? lam2[i][p] : sp2(0.f);
      bsig[i] = ok ? bsig[i] : 0.f;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    rqn[i] = RSQ(row_sum16(rqn[i]) + 1e-6f);
    rkn[i] = RSQ(row_sum16(rkn[i]) + 1e-6f);
  }
#pragma unroll
  for (int i = 1; i < 4; ++i)
#pragma unroll
    for (int p = 0; p < 4; ++p)
      lam2[i][p] += lam2[i - 1][p];
  {
    const float m1 = (lg >= 1) ? 1.f : 0.f, m2 = (lg >= 2) ? 1.f : 0.f;
    f32x2 totw[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const f32x2 tot = lam2[3][p];
      f32x2 inc = mk2(shfl_up64l(tot[0], 16, lane), shfl_up64l(tot[1], 16, lane)) * sp2(m1) + tot;
      inc = mk2(shfl_up64l(inc[0], 32, lane), shfl_up64l(inc[1], 32, lane)) * sp2(m2) + inc;
      const f32x2 excl = inc - tot;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        lam2[i][p] += excl;
      totw[p] = inc;
    }
    if (lg == 3) {
      f32x4 t0v, t1v;
      t0v[0] = totw[0][0];
      t0v[1] = totw[0][1];
      t0v[2] = totw[1][0];
      t0v[3] = totw[1][1];
      t1v[0] = totw[2][0];
      t1v[1] = totw[2][1];
      t1v[2] = totw[3][0];
      t1v[3] = totw[3][1];
      *(f32x4*)(TAU + wu * KD + 8 * lr) = t0v;
      *(f32x4*)(TAU + wu * KD + 8 * lr + 4) = t1v;
    }
  }
  if (lr == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      BETA[16 * wu + 4 * lg + i] = bsig[i];
  }
  F_SYNC_RELANE();
  u32x2 kgt[4][2];
  u32x2 qgh[4][2];
  u32x2 kdp[4][2];
  {
    f32x2 tw[4], e2c[4], qgf[4], kdf[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int c = 8 * lr + 2 * p;
      const f32x2 t0 = *(const f32x2*)(TAU + c);
      const f32x2 t1 = *(const f32x2*)(TAU + KD + c);
      const f32x2 t2 = *(const f32x2*)(TAU + 2 * KD + c);
      const f32x2 t3 = *(const f32x2*)(TAU + 3 * KD + c);
      f32x2 carry = sp2(0.f);
      if (wu > 0) carry = t0;
      if (wu > 1) carry += t1;
      if (wu > 2) carry += t2;
      tw[p] = (wu == 0) ? t0 : (wu == 1) ? t1 : (wu == 2) ? t2 : t3;
      const f32x2 glt = (t0 + t1) + (t2 + t3);
      e2c[p] = exp2v(carry);
      qgf[p] = e2c[p] * sp2(scale);
      kdf[p] = exp2v((glt - carry) - tw[p]);
    }
    f32x2 kdv[4], kgv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const f32x2 rq = sp2(rqn[i]), rk = sp2(rkn[i]), bs = sp2(bsig[i]);
      u32x4 qrp, krp, kep;
      uint32_t qgp[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const f32x2 lm = lam2[i][p];
        const f32x2 E = exp2v(lm);
        const f32x2 G = exp2v(tw[p] - lm);
        const f32x2 qh = bf2(cur.q[i][p]) * rq;
        const f32x2 kh = bf2(cur.k[i][p]) * rk;
        const f32x2 qr = qh * E;
        const f32x2 kr = kh * E;
        const f32x2 ke = kh * G;
        qrp[p] = pk2(qr);
        krp[p] = pk2(kr);
        kep[p] = pk2(ke);
        qgp[p] = pk2(qr * qgf[p]);
        const f32x2 kd = ke * kdf[p];
        const f32x2 kg = kr * e2c[p] * bs;
        if (i & 1) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            kdp[p][e][i >> 1] = pk_bf16(kdv[p][e], kd[e]);
            kgt[p][e][i >> 1] = pk_bf16(kgv[p][e], kg[e]);
          }
        } else {
          kdv[p] = kd;
          kgv[p] = kg;
        }
      }
      const int R = 16 * wu + 4 * lg + i;
      lds_st16(QR, rx_idx(R, 8 * lr), qrp);
      lds_st16(KR, rx_idx(R, 8 * lr), krp);
      lds_st16(KE, rx_idx(R, 8 * lr), kep);
      qgh[i][0][0] = qgp[0];
      qgh[i][0][1] = qgp[1];
      qgh[i][1][0] = qgp[2];
      qgh[i][1][1] = qgp[3];
    }
  }
  pf.qk(cur, lg, lr);
  {
    const int c0 = 8 * lr + 2 * lg;
    const f32x2 tov = *(const f32x2*)(TAU + wu * KD + c0);
    *(f32x2*)(FACD + wu * KD + c0) = exp2v(-tov);
  }
  F_SYNC_RELANE();
  {
    const int q4 = lr & 3;
    const int ksq = lr >> 2;
    const int h1 = q4 >> 1;
    const int ga = 2 * (q4 & 1);
    char* qb = gb + GB_Q + (size_t)((wu * 4 + ksq) * 64) * 16 + h1 * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = 4 * lg + i;
      gst8(qb + (n + 16 * ga) * 16, qgh[i][0]);
      gst8(qb + (n + 16 * (ga + 1)) * 16, qgh[i][1]);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int c = 8 * lr + 2 * p + e;
        char* kb = gb + GB_K + (size_t)(((c >> 4) * 2 + (wu >> 1)) * 64 + lg * 16 + (c & 15)) * 16 + (wu & 1) * 8;
        gst8(kb, kdp[p][e]);
      }
    if (wu == 0) {
      const int c0 = 8 * lr + 2 * lg;
      const f32x2 t0v = *(const f32x2*)(TAU + c0);
      const f32x2 t1v = *(const f32x2*)(TAU + KD + c0);
      const f32x2 t2v = *(const f32x2*)(TAU + 2 * KD + c0);
      const f32x2 t3v = *(const f32x2*)(TAU + 3 * KD + c0);
      *(f32x2*)(gb + GB_G + c0 * 4) = exp2v((t0v + t1v) + (t2v + t3v));
    }
  }
  f32x4 akd, ako0, ako1 = zero4();
  {
    u32x4 cr[4], cs[4];
    if (wu == 0) {
      p3_col(cr, KE, 0, lg, lr);
      ako0 = p3_pair(cr, 1, 0, false, QR, KR, BETA, gb, scale, lg, lr);
      p3_scale(cs, cr, FACD + 0 * KD, lg);
      akd = p3_pair(cs, 0, 0, true, QR, KR, BETA, gb, scale, lg, lr);
      p3_scale_tau<1>(cs, cr, TAU + 1 * KD, TAU, lg);
      ako1 = p3_pair(cs, 2, 0, false, QR, KR, BETA, gb, scale, lg, lr);
    } else if (wu == 1) {
      p3_col(cr, KE, 1, lg, lr);
      ako0 = p3_pair(cr, 2, 1, false, QR, KR, BETA, gb, scale, lg, lr);
      p3_scale(cs, cr, FACD + 1 * KD, lg);
      akd = p3_pair(cs, 1, 1, true, QR, KR, BETA, gb, scale, lg, lr);
      p3_col(cr, KE, 0, lg, lr);
      p3_scale_tau<2>(cs, cr, TAU + 1 * KD, TAU + 2 * KD, lg);
      ako1 = p3_pair(cs, 3, 0, false, QR, KR, BETA, gb, scale, lg, lr);
    } else if (wu == 2) {
      p3_col(cr, KE, 2, lg, lr);
      p3_scale(cs, cr, FACD + 2 * KD, lg);
      akd = p3_pair(cs, 2, 2, true, QR, KR, BETA, gb, scale, lg, lr);
      p3_col(cr, KE, 1, lg, lr);
      p3_scale_tau<1>(cs, cr, TAU + 2 * KD, TAU, lg);
      ako0 = p3_pair(cs, 3, 1, false, QR, KR, BETA, gb, scale, lg, lr);
    } else {
      p3_col(cr, KE, 2, lg, lr);
      ako0 = p3_pair(cr, 3, 2, false, QR, KR, BETA, gb, scale, lg, lr);
      p3_col(cr, KE, 3, lg, lr);
      p3_scale(cs, cr, FACD + 3 * KD, lg);
      akd = p3_pair(cs, 3, 3, true, QR, KR, BETA, gb, scale, lg, lr);
    }
    if (wu >= 2) {
      u32x2 z;
      z[0] = 0u;
      z[1] = 0u;
      gst8(gb + GB_A + (size_t)(fa_idx(wu == 2 ? 0 : 2, wu == 2 ? 0 : 1) * 64 + lane) * 16 + 8, z);
    }
  }
  F_SYNC_RELANE();
  {
    if (wu == 0) {
      st_akk(AKK, 1, 0, ako0, lg, lr);
      st_akk(AKK, 2, 0, ako1, lg, lr);
    } else if (wu == 1) {
      st_akk(AKK, 2, 1, ako0, lg, lr);
      st_akk(AKK, 3, 0, ako1, lg, lr);
    } else if (wu == 2) {
      st_akk(AKK, 3, 1, ako0, lg, lr);
    } else {
      st_akk(AKK, 3, 2, ako0, lg, lr);
    }
    st_akk(AKK, wu, wu, akd, lg, lr);
    const int jrow = 16 * wu + 4 * lg;
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        lds_st8(KGT, tr_idx(8 * lr + 2 * p + e, jrow), kgt[p][e]);
    {
      const int tidl = 64 * wu + lane;
      int blk = tidl >> 7, rr = (tidl >> 3) & 15, cc = (tidl & 7) * 2;
      int I = blk ? 2 : 0;
      T16[a16_idx(16 * I + rr, 16 * (I + 1) + cc)] = 0;
      T16[a16_idx(16 * I + rr, 16 * (I + 1) + cc + 1)] = 0;
    }
    WAVE_SYNC();
    {
      const int j = lr;
      const float* Ab = AKK + akk_idx(wu, wu, 0, 0);
      f32x2 t2[8];
#pragma unroll
      for (int ii = 0; ii < 16; ++ii) {
        f32x2 acc = sp2(0.f);
#pragma unroll
        for (int kk = 0; kk < (ii >> 1); ++kk)
          acc += *(const f32x2*)(Ab + ii * 20 + 2 * kk) * t2[kk];
        float s = acc[0] + acc[1];
        if (ii & 1) s += Ab[ii * 20 + ii - 1] * t2[ii >> 1][0];
        t2[ii >> 1][ii & 1] = ((ii == j) ? 1.f : 0.f) - s;
      }
      if (lg == 0) {
#pragma unroll
        for (int ii = 0; ii < 16; ++ii) {
          const float tv = t2[ii >> 1][ii & 1];
          TFD[tfd_idx(wu, ii, j)] = tv;
          T16[a16_idx(16 * wu + ii, 16 * wu + j)] = (uint16_t)(pk_bf16(tv, 0.f) & 0xffffu);
        }
      }
    }
  }
  F_SYNC_RELANE();
  if (wu == 0)
    p5_col<0>(AKK, TFD, T16, lg, lr);
  else if (wu == 1)
    p5_col<1>(AKK, TFD, T16, lg, lr);
  else if (wu == 2)
    p5_col<2>(AKK, TFD, T16, lg, lr);
  F_SYNC_RELANE();
  {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      f32x4 acc[2];
      u32x4 a0[2], a1[2], b[2];
      const int nks = (nt < 2) ? 1 : 2;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        a0[ks] = lds_ld16(KGT, tr_idx(16 * (2 * wu) + lr, 32 * ks + 8 * lg));
        a1[ks] = lds_ld16(KGT, tr_idx(16 * (2 * wu + 1) + lr, 32 * ks + 8 * lg));
        b[ks] = lds_ld16(T16, a16_idx(16 * nt + lr, 32 * ks + 8 * lg));
      }
      acc[0] = zero4();
      acc[1] = zero4();
      MFMA_SB(acc[0], acc[1], a0, a1, b, nks);
      u32x4 fr;
      fr[0] = pk_bf16(-acc[0][0], -acc[0][1]);
      fr[1] = pk_bf16(-acc[0][2], -acc[0][3]);
      fr[2] = pk_bf16(-acc[1][0], -acc[1][1]);
      fr[3] = pk_bf16(-acc[1][2], -acc[1][3]);
      gst16(gb + GB_W + (size_t)((nt * 4 + wu) * 64 + lane) * 16, fr);
    }
    for (int f = wu; f < 6; f += 4) {
      const int mt = f < 2 ? f : 2 + ((f - 2) >> 1);
      const int ks = f < 2 ? 0 : ((f - 2) & 1);
      gst16(gb + GB_T + (size_t)(f * 64 + lane) * 16, lds_ld16(T16, a16_idx(16 * mt + lr, 32 * ks + 8 * lg)));
    }
  }
}
#define SMEM_FP SMEM_F2
static_assert(SMEM_FP <= 163840, "pipelined fused kernel must fit the 160 KB LDS");
DEVI void fused_load_gb(
    IntraRaw& r,
    long t0,
    int L,
    const uint16_t* __restrict__ g,
    const uint16_t* __restrict__ beta,
    int H,
    int h,
    int w,
    int lg,
    int lr) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int R = 16 * w + 4 * lg + i;
    long t = (R < L) ? (t0 + R) : t0;
    size_t off = ((size_t)t * H + h) * KD + 8 * lr;
    r.g[i] = *(const u32x4*)(g + off);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int R = 16 * w + 4 * lg + i;
    long t = (R < L) ? (t0 + R) : t0;
    r.b[i] = (uint32_t)beta[(size_t)t * H + h];
  }
}
DEVI void fused_load_qk(
    IntraRaw& r,
    long t0,
    int L,
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    int H,
    int h,
    int w,
    int lg,
    int lr) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int R = 16 * w + 4 * lg + i;
    long t = (R < L) ? (t0 + R) : t0;
    size_t off = ((size_t)t * H + h) * KD + 8 * lr;
    r.q[i] = *(const u32x4*)(q + off);
    r.k[i] = *(const u32x4*)(k + off);
  }
}
struct FusedPF2 {
  const uint16_t* q;
  const uint16_t* k;
  const uint16_t* g;
  const uint16_t* beta;
  char* smem;
  long t0n;
  int Ln, H, h, w;
  DEVM void gate(IntraRaw& r, int lg, int lr) const {
    fused_load_gb(r, t0n, Ln, g, beta, H, h, w, lg, lr);
  }
  DEVM void qk(IntraRaw& r, int lg, int lr) const {
    fused_load_qk(r, t0n, Ln, q, k, H, h, w, lg, lr);
  }
};
DEVI void kda_fused_body_pipe(
    char* smem,
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v,
    const uint16_t* __restrict__ g,
    const uint16_t* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* h0,
    const int* __restrict__ cu,
    const int* __restrict__ slots,
    uint16_t* __restrict__ o,
    float* ht,
    char* __restrict__ trash,
    int T,
    int H,
    int N,
    float scale,
    int bid) {
  const int wu = WAVE_U();
  const int p = bid / H;
  const int h = bid - p * H;
  if (p >= N) return;
  opq_init(smem, wu);
  const RecurSeq sq = recur_seq(p, cu, N);
  const int n = sq.n;
  const long bos = sq.bos;
  const int len = (int)sq.len;
  const int nch = sq.nch;
  char* const BLK = smem + F_BLK;
#define CH_L(c) ((len - (c) * KC) < KC ? (len - (c) * KC) : KC)
  if (wu < 4) {
    if (nch > 0) {
      IntraRaw raw;
      {
        const int l0 = opq_lane(opq_pre(smem, wu));
        fused_load_consts(raw, A_log, dt_bias, h, l0 & 15);
        fused_load_rows(raw, bos, CH_L(0), q, k, g, beta, H, h, wu, l0 >> 4, l0 & 15);
      }
      for (int a = 0; a < nch; ++a) {
        const int L = CH_L(a);
        const int an = (a + 1 < nch) ? a + 1 : a;
        const int z1 = opq_pre(smem, wu);
        FusedPF2 pf{q, k, g, beta, smem, bos + (long)an * KC, CH_L(an), H, h, wu};
        intra_process_fp(smem, L, raw, BLK, scale, pf, z1);
        SYNC();
      }
    }
  } else {
    const float* BETA = (const float*)(smem + LA_BETA);
    const int sw0 = 2 * (wu - 4);
    const int sw1 = sw0 + 1;
    f32x4 S0[8], S1[8];
    {
      const int l0 = opq_lane(opq_pre(smem, wu));
      const float* h0p = h0 + ((size_t)slots[n] * H + h) * (KD * KD);
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        S0[mt] = *(const f32x4*)(h0p + (16 * sw0 + (l0 & 15)) * KD + 16 * mt + 4 * (l0 >> 4));
        S1[mt] = *(const f32x4*)(h0p + (16 * sw1 + (l0 & 15)) * KD + 16 * mt + 4 * (l0 >> 4));
      }
    }
    if (nch > 0) {
      StageNone st;
      u32x4 VP[4];
      fused_fetch_v(VP, v, bos, CH_L(0), H, h, 64 * (wu - 4) + opq_lane(opq_pre(smem, wu)));
      SYNC();
      {
        const int zv = opq_pre(smem, wu);
        SYNC();
        const int lt = 64 * (wu - 4) + opq_lane(zv);
        fused_stage_vt(BLK + LB_VT, VP, BETA, lt);
        const int c1 = (1 < nch) ? 1 : 0;
        fused_fetch_v(VP, v, bos + (long)c1 * KC, CH_L(c1), H, h, lt);
      }
      SYNC();
      SYNC();
      SYNC();
      int zs = opq_pre(smem, wu);
      SYNC();
      for (int c = 0; c < nch; ++c) {
        const long t0 = bos + (long)c * KC;
        const int L = CH_L(c);
        const bool more = c + 1 < nch;
        {
          const int ls = opq_lane(zs);
          recur_chunk_p<1>(S0, BLK, st, o, trash, t0, L, H, h, sw0, sw0, ls, ls >> 4, ls & 15);
        }
        zs = opq_pre(smem, wu);
        if (more) SYNC();
        {
          const int ls = opq_lane(zs);
          recur_chunk_p<1>(S1, BLK, st, o, trash, t0, L, H, h, sw1, sw1, ls, ls >> 4, ls & 15);
        }
        if (more) {
          const int zv = opq_pre(smem, wu);
          SYNC();
          const int lt = 64 * (wu - 4) + opq_lane(zv);
          fused_stage_vt(BLK + LB_VT, VP, BETA, lt);
          const int c2 = (c + 2 < nch) ? c + 2 : c + 1;
          fused_fetch_v(VP, v, bos + (long)c2 * KC, CH_L(c2), H, h, lt);
          SYNC();
          SYNC();
          SYNC();
          zs = opq_pre(smem, wu);
          SYNC();
        }
      }
    }
    {
      const int l0 = opq_lane(opq_pre(smem, wu));
      float* htp = ht + ((size_t)slots[n] * H + h) * (KD * KD);
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        *(f32x4*)(htp + (16 * sw0 + (l0 & 15)) * KD + 16 * mt + 4 * (l0 >> 4)) = S0[mt];
        *(f32x4*)(htp + (16 * sw1 + (l0 & 15)) * KD + 16 * mt + 4 * (l0 >> 4)) = S1[mt];
      }
    }
  }
#undef CH_L
}

// =====================================================================================================================
// 9. Kernel entry points
// =====================================================================================================================

// kernel A: one (chunk, head) item per 256-thread workgroup, three workgroups per CU (53504 B of LDS)
extern "C" __global__ void __launch_bounds__(256, 3) kda_intra_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ g,
    const uint16_t* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const int* __restrict__ cu,
    char* __restrict__ gbuf,
    int T,
    int H,
    int N,
    float scale,
    int nwork) {
  __shared__ __attribute__((aligned(16))) char smem[SMEM_A];
  kda_intra_body_np(smem, q, k, g, beta, A_log, dt_bias, cu, gbuf, T, H, N, scale, nwork);
}

// one (sequence, head) chain per workgroup (VS = 1) or per value half (VS = 2).  The value-half build has
// a two-wave register budget so that its MFMAs keep the state in VGPRs.
#define KDA_RECUR_KERNEL(NAME, BOUND, VS, SMEM)                                                                      \
  extern "C" __global__ void BOUND NAME(                                                                             \
      const char* __restrict__ gbuf,                                                                                 \
      const uint16_t* __restrict__ v,                                                                                \
      const float* h0,                                                                                               \
      const int* __restrict__ cu,                                                                                    \
      const int* __restrict__ slots,                                                                                 \
      uint16_t* __restrict__ o,                                                                                      \
      float* ht,                                                                                                     \
      float* __restrict__ track_state,                                                                               \
      const long long* __restrict__ track_chunk_idx,                                                                 \
      char* __restrict__ trash,                                                                                      \
      int T,                                                                                                         \
      int H,                                                                                                         \
      int N) {                                                                                                       \
    __shared__ __attribute__((aligned(16))) char smem[SMEM];                                                         \
    kda_recur_body_t<VS, 1>(smem, gbuf, v, h0, cu, slots, o, ht, track_state, track_chunk_idx, trash, T, H, N, BID); \
  }
KDA_RECUR_KERNEL(kda_recur_kernel_vs2, __launch_bounds__(256, 2), 2, SMEM_B2)
KDA_RECUR_KERNEL(kda_recur_kernel_vs1, __launch_bounds__(512), 1, SMEM_B1)

// warp-specialized value-part workgroups: NM math waves + NM loader waves (NM = 8 / VS).
// ws2: value halves, 512 threads (two waves per SIMD, so at most 256 registers each, VGPR-form
// MFMAs); ws4: value quarters (4 N H <= #CU), 256 threads, two-wave budget.
#define KDA_RECUR_WS_KERNEL(NAME, BOUND, VS, SMEM)                                      \
  extern "C" __global__ void BOUND NAME(                                                \
      const char* __restrict__ gbuf,                                                    \
      const uint16_t* __restrict__ v,                                                   \
      const float* h0,                                                                  \
      const int* __restrict__ cu,                                                       \
      const int* __restrict__ slots,                                                    \
      uint16_t* __restrict__ o,                                                         \
      float* ht,                                                                        \
      char* __restrict__ trash,                                                         \
      int T,                                                                            \
      int H,                                                                            \
      int N) {                                                                          \
    __shared__ __attribute__((aligned(16))) char smem[SMEM];                            \
    kda_recur_body_ws<VS>(smem, gbuf, v, h0, cu, slots, o, ht, trash, T, H, N, BID, N); \
  }
KDA_RECUR_WS_KERNEL(kda_recur_kernel_ws2, __launch_bounds__(512), 2, SMEM_B2)
KDA_RECUR_WS_KERNEL(kda_recur_kernel_ws4, __launch_bounds__(256, 2), 4, SMEM_B4)

// sequence-group passes of one long sequence (kda_recur_body_grp), same workgroup shapes as ws2 / ws4 / vs1
#define KDA_RECUR_GRP_KERNEL(NAME, BOUND, VS, SMEM)                                                          \
  extern "C" __global__ void BOUND NAME(                                                                     \
      const char* __restrict__ gbuf,                                                                         \
      const uint16_t* __restrict__ v,                                                                        \
      const float* h0,                                                                                       \
      const int* __restrict__ cu,                                                                            \
      const int* __restrict__ slots,                                                                         \
      uint16_t* __restrict__ o,                                                                              \
      float* ht,                                                                                             \
      float* __restrict__ track_state,                                                                       \
      const long long* __restrict__ track_chunk_idx,                                                         \
      float* __restrict__ gst,                                                                               \
      char* __restrict__ trash,                                                                              \
      int H,                                                                                                 \
      int G,                                                                                                 \
      int gc,                                                                                                \
      int pass) {                                                                                            \
    __shared__ __attribute__((aligned(16))) char smem[SMEM];                                                 \
    kda_recur_body_grp<VS>(                                                                                  \
        smem, gbuf, v, h0, cu, slots, o, ht, track_state, track_chunk_idx, gst, trash, H, G, gc, pass, BID); \
  }
KDA_RECUR_GRP_KERNEL(kda_recur_grp_ws2, __launch_bounds__(512), 2, SMEM_B2)
KDA_RECUR_GRP_KERNEL(kda_recur_grp_ws4, __launch_bounds__(256, 2), 4, SMEM_B4)
KDA_RECUR_GRP_KERNEL(kda_recur_grp_vs1, __launch_bounds__(512), 1, SMEM_B1)

// entry states of the sequence groups, E_{g+1} = E_g M_g + B_g for g = 1..G-2 in fp32: one 512-thread
// workgroup per (head, 16 value rows), wave w computes the key columns 16 w .. 16 w + 15
extern "C" __global__ void __launch_bounds__(512) kda_recur_grp_scan(float* __restrict__ gst, int H, int G) {
  __shared__ float es[16 * 132];
  const int h = BID >> 3;
  const int r0 = 16 * (BID & 7);
  const int w = TID >> 6;
  const int lane = TID & 63;
  const int li = lane & 15;
  const int lk = lane >> 4;
  const size_t hs = (size_t)KD * KD;
  const size_t gs = (size_t)G * H * hs;
  // this lane's operands of step g: M_g[4 s + lk][16 w + li] and B_g[r0 + 4 lk + r][16 w + li]
  auto load = [&](float* mv, float* bv, int g) {
    const float* m = gst + 2 * gs + ((size_t)g * H + h) * hs + 16 * w + li;
    const float* b = gst + gs + ((size_t)g * H + h) * hs + (size_t)r0 * KD + 16 * w + li;
#pragma unroll
    for (int s = 0; s < 32; ++s)
      mv[s] = m[(4 * s + lk) * KD];
#pragma unroll
    for (int r = 0; r < 4; ++r)
      bv[r] = b[(4 * lk + r) * KD];
  };
  const float* e1 = gst + ((size_t)H + h) * hs + (size_t)r0 * KD;
  for (int i = TID; i < 16 * KD; i += 512)
    es[(i >> 7) * 132 + (i & 127)] = e1[i];
  SYNC();
  for (int g = 1; g + 1 < G; ++g) {
    float mv[32], bv[4];
    load(mv, bv, g);
    f32x4 acc[4] = {zero4(), zero4(), zero4(), zero4()};
#pragma unroll
    for (int s = 0; s < 32; ++s)
      acc[s & 3] = MFMA4(es[li * 132 + 4 * s + lk], mv[s], acc[s & 3]);
    const f32x4 d = acc[0] + acc[1] + acc[2] + acc[3];
    float* e = gst + ((size_t)(g + 1) * H + h) * hs + (size_t)r0 * KD + 16 * w + li;
    SYNC();
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = 4 * lk + r;
      const float x = d[r] + bv[r];
      es[i * 132 + 16 * w + li] = x;
      e[i * KD] = x;
    }
    SYNC();
  }
}

// pipelined fused kernel: one 512-thread workgroup per (sequence, head) keeps the
// chunk's block in LDS (one workgroup per CU); waves 0-3 run the intra phase of chunk c + 1 while waves 4-7 run
// the recurrence step of chunk c (two 16-column state tiles each)
extern "C" __global__ void __launch_bounds__(512) kda_fusedp_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v,
    const uint16_t* __restrict__ g,
    const uint16_t* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* h0,
    const int* __restrict__ cu,
    const int* __restrict__ slots,
    uint16_t* __restrict__ o,
    float* ht,
    char* __restrict__ trash,
    int T,
    int H,
    int N,
    float scale) {
  __shared__ __attribute__((aligned(16))) char smem[SMEM_FP];
  kda_fused_body_pipe(smem, q, k, v, g, beta, A_log, dt_bias, h0, cu, slots, o, ht, trash, T, H, N, scale, BID);
}

// =====================================================================================================================
// 10. Host: path selection and launch
// =====================================================================================================================

// value split of a sequence-group pass over items (group, head) pairs: the widest that fits one workgroup
// per CU (0: none)
static int kda_grp_vs(int64_t items, int64_t H, int ncu) {
  if (items * H * 4 <= ncu) return 4;
  if (items * H * 2 <= ncu) return 2;
  return items * H <= ncu ? 1 : 0;
}
// sequence groups for one sequence: the most whose pass 0 (2 G - 3 items) fits, at least 8 chunks per group;
// 0: the single chunk chain
static int kda_grp_groups(int64_t T, int64_t H, int64_t N, int ncu) {
  int G = 0;
  for (int64_t g = 3; N == 1 && kda_grp_vs(2 * g - 3, H, ncu) && g * 8 * KC <= T; ++g)
    G = (int)g;
  return G;
}
static int64_t kda_grp_state_bytes(int64_t T, int64_t H, int64_t N, int ncu) {
  return 3 * (int64_t)kda_grp_groups(T, H, N, ncu) * H * KD * KD * 4;
}

enum KdaPath { KDA_FUSED, KDA_GRP, KDA_WS4, KDA_WS2, KDA_VS2, KDA_VS1 };

// The thresholds are tuned on MI355X (256 CUs) for 1..128 sequences of 64..64K tokens with 12 and 24 heads.
static KdaPath kda_recur_path(int64_t T, int64_t H, int64_t N, int64_t max_seqlen, bool tracking, int ncu) {
  const int64_t pairs = N * H;
  const int64_t chain = (max_seqlen + KC - 1) / KC;
  // the longest chunk chain is at most twice the chunk work per CU
  const int64_t work = (T / KC + N) * H;
  const bool balanced = chain <= 2 || chain * ncu <= 2 * work;
  // pipelined fused kernel: balanced batches with at least 3/4 pair per CU, or short sequences
  if (!tracking && balanced && (4 * pairs >= 3 * ncu || chain <= 1 || (chain <= 2 && 2 * pairs >= ncu)))
    return KDA_FUSED;
  if (kda_grp_groups(T, H, N, ncu)) return KDA_GRP;
  if (tracking) return 2 * pairs <= ncu ? KDA_VS2 : KDA_VS1;
  // a long chain dominates: split its value columns over two or four workgroups
  if (!balanced) return chain * ncu > 4 * work ? KDA_WS4 : KDA_WS2;
  if (4 * pairs <= ncu) return KDA_WS4;
  if (2 * pairs <= ncu) return KDA_WS2;
  return KDA_VS1;
}

// state[slots[n]] holds the initial state of sequence n and receives its final state
static void kda_fwd_launch(
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    const uint16_t* g,
    const uint16_t* beta,
    const float* A_log,
    const float* dt_bias,
    float* state,
    const int* cu,
    const int* slots,
    uint16_t* o,
    float* track_state,
    const long long* track_chunk_idx,
    char* gbuf,
    int T,
    int H,
    int N,
    int max_seqlen,
    float scale,
    int ncu,
    hipStream_t stream) {
  if (N <= 0 || H <= 0) return;
  const KdaPath path = kda_recur_path(T, H, N, max_seqlen, track_state != nullptr, ncu);
  const int nwork = (T / KC + N) * H;
  // rows past the end of a sequence are stored to this scratch slot, right after the per-(chunk, head) blocks
  char* trash = gbuf + (size_t)nwork * GB_BYTES;
  if (path == KDA_FUSED) {
    hipLaunchKernelGGL(
        kda_fusedp_kernel,
        dim3(N * H),
        dim3(512),
        0,
        stream,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        state,
        cu,
        slots,
        o,
        state,
        trash,
        T,
        H,
        N,
        scale);
    return;
  }
  hipLaunchKernelGGL(
      kda_intra_kernel,
      dim3(nwork),
      dim3(256),
      0,
      stream,
      q,
      k,
      g,
      beta,
      A_log,
      dt_bias,
      cu,
      gbuf,
      T,
      H,
      N,
      scale,
      nwork);
  switch (path) {
    case KDA_GRP: {
      const int G = kda_grp_groups(T, H, N, ncu);
      float* gst = (float*)(trash + 4096);
      const int gc = ((T + KC - 1) / KC + G - 1) / G;
      for (int pass = 0; pass < 2; ++pass) {
        const int items = pass ? G - 1 : 2 * G - 3;
        if (pass) hipLaunchKernelGGL(kda_recur_grp_scan, dim3(H * 8), dim3(512), 0, stream, gst, H, G);
#define KDA_LAUNCH_GRP(KN, VS)         \
  hipLaunchKernelGGL(                  \
      KN,                              \
      dim3(items* H* VS),              \
      dim3(VS == 1 ? 512 : 1024 / VS), \
      0,                               \
      stream,                          \
      gbuf,                            \
      v,                               \
      state,                           \
      cu,                              \
      slots,                           \
      o,                               \
      state,                           \
      track_state,                     \
      track_chunk_idx,                 \
      gst,                             \
      trash,                           \
      H,                               \
      G,                               \
      gc,                              \
      pass)
        const int vs = kda_grp_vs(items, H, ncu);
        if (vs == 4)
          KDA_LAUNCH_GRP(kda_recur_grp_ws4, 4);
        else if (vs == 2)
          KDA_LAUNCH_GRP(kda_recur_grp_ws2, 2);
        else
          KDA_LAUNCH_GRP(kda_recur_grp_vs1, 1);
#undef KDA_LAUNCH_GRP
      }
      break;
    }
    case KDA_WS4:
      // 2 math + 2 loader waves per workgroup, the four parts of a head on one XCD
      hipLaunchKernelGGL(
          kda_recur_kernel_ws4,
          dim3(N * H * 4),
          dim3(256),
          0,
          stream,
          gbuf,
          v,
          state,
          cu,
          slots,
          o,
          state,
          trash,
          T,
          H,
          N);
      break;
    case KDA_WS2:
      // 4 math + 4 loader waves per workgroup
      hipLaunchKernelGGL(
          kda_recur_kernel_ws2,
          dim3(N * H * 2),
          dim3(512),
          0,
          stream,
          gbuf,
          v,
          state,
          cu,
          slots,
          o,
          state,
          trash,
          T,
          H,
          N);
      break;
    case KDA_VS2:
      hipLaunchKernelGGL(
          kda_recur_kernel_vs2,
          dim3(N * H * 2),
          dim3(256),
          0,
          stream,
          gbuf,
          v,
          state,
          cu,
          slots,
          o,
          state,
          track_state,
          track_chunk_idx,
          trash,
          T,
          H,
          N);
      break;
    default:
      hipLaunchKernelGGL(
          kda_recur_kernel_vs1,
          dim3(N * H),
          dim3(512),
          0,
          stream,
          gbuf,
          v,
          state,
          cu,
          slots,
          o,
          state,
          track_state,
          track_chunk_idx,
          trash,
          T,
          H,
          N);
      break;
  }
}

// =====================================================================================================================
// 11. Torch binding
// =====================================================================================================================

#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>

// ssm_states[cache_indices[n]] holds the initial state of sequence n and is overwritten with its final state
torch::Tensor kda_prefill_gfx950(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor ssm_states,
    torch::Tensor cache_indices,
    torch::Tensor cu_seqlens,
    int64_t max_seqlen,
    torch::Tensor track_state,
    torch::Tensor track_chunk_idx,
    double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on a GPU");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
  TORCH_CHECK(
      k.sizes() == q.sizes() && v.sizes() == q.sizes() && g.sizes() == q.sizes(),
      "q, k, v, and g must have the same shape");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == 128, "q must have shape [1, T, H, 128]");
  TORCH_CHECK(
      beta.dim() == 3 && beta.size(0) == 1 && beta.size(1) == q.size(1) && beta.size(2) == q.size(2),
      "beta must have shape [1, T, H]");
  TORCH_CHECK(ssm_states.scalar_type() == at::ScalarType::Float, "ssm_states must be fp32");
  TORCH_CHECK(
      ssm_states.dim() == 4 && ssm_states.size(1) == q.size(2) && ssm_states.size(2) == 128 &&
          ssm_states.size(3) == 128,
      "ssm_states must have shape [S, H, 128, 128]");
  TORCH_CHECK(
      cache_indices.scalar_type() == at::ScalarType::Int && cache_indices.dim() == 1,
      "cache_indices must be int32 and one-dimensional");
  TORCH_CHECK(
      cu_seqlens.scalar_type() == at::ScalarType::Int && cu_seqlens.dim() == 1,
      "cu_seqlens must be int32 and one-dimensional");
  TORCH_CHECK(
      A_log.scalar_type() == at::ScalarType::Float && A_log.numel() == q.size(2), "A_log must be fp32 with H elements");
  TORCH_CHECK(
      dt_bias.scalar_type() == at::ScalarType::Float && dt_bias.numel() == q.size(2) * 128,
      "dt_bias must be fp32 with H*128 elements");
  TORCH_CHECK(
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() && beta.is_contiguous() &&
          A_log.is_contiguous() && dt_bias.is_contiguous() && ssm_states.is_contiguous() &&
          cache_indices.is_contiguous() && cu_seqlens.is_contiguous(),
      "all inputs must be contiguous");

  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t N = cache_indices.numel();
  TORCH_CHECK(cu_seqlens.numel() == N + 1, "cu_seqlens must contain N+1 offsets");
  const bool tracking = track_state.numel() != 0;
  TORCH_CHECK(tracking == (track_chunk_idx.numel() != 0), "track_state and track_chunk_idx must be passed together");
  if (tracking) {
    TORCH_CHECK(
        track_state.scalar_type() == at::ScalarType::Float &&
            track_state.sizes() == at::IntArrayRef({N, H, 128, 128}) && track_state.is_contiguous(),
        "track_state must be contiguous fp32 [N, H, 128, 128]");
    TORCH_CHECK(
        track_chunk_idx.scalar_type() == at::ScalarType::Long && track_chunk_idx.numel() == N &&
            track_chunk_idx.is_contiguous(),
        "track_chunk_idx must be contiguous int64 [N]");
  }

  auto output = torch::empty_like(q);
  const int ncu = at::cuda::getDeviceProperties(q.get_device())->multiProcessorCount;
  const int64_t gbuf_bytes = (T / KC + N) * H * GB_BYTES;
  auto workspace =
      torch::empty({gbuf_bytes + 4096 + kda_grp_state_bytes(T, H, N, ncu)}, q.options().dtype(at::ScalarType::Byte));
  const hipStream_t stream = at::hip::getCurrentHIPStream();

  kda_fwd_launch(
      reinterpret_cast<const uint16_t*>(q.data_ptr()),
      reinterpret_cast<const uint16_t*>(k.data_ptr()),
      reinterpret_cast<const uint16_t*>(v.data_ptr()),
      reinterpret_cast<const uint16_t*>(g.data_ptr()),
      reinterpret_cast<const uint16_t*>(beta.data_ptr()),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      ssm_states.data_ptr<float>(),
      cu_seqlens.data_ptr<int>(),
      cache_indices.data_ptr<int>(),
      reinterpret_cast<uint16_t*>(output.data_ptr()),
      tracking ? track_state.data_ptr<float>() : nullptr,
      tracking ? reinterpret_cast<const long long*>(track_chunk_idx.data_ptr<int64_t>()) : nullptr,
      reinterpret_cast<char*>(workspace.data_ptr()),
      (int)T,
      (int)H,
      (int)N,
      (int)max_seqlen,
      (float)scale,
      ncu,
      stream);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kda_prefill_gfx950", &kda_prefill_gfx950, "gfx950 KDA prefill forward");
}
