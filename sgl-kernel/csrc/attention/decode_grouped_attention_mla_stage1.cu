// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
// SPDX-License-Identifier: Apache-2.0
//
// HIP / CDNA4 (MI350, gfx950) MFMA port of the Triton
// `_fwd_grouped_kernel_stage1` (MLA decode, stage 1) from
// python/sglang/kernels/ops/attention/decode_attention.py.
//
// Tailored to the MLA decode config only:
//     PAGE_SIZE   = 1        (contiguous KV addressing; no paging math)
//     BLOCK_N     = 32       (KV tokens per chunk == N_TILES=2 MFMA N-tiles)
//     BLOCK_H     = 16       (query heads per program == MFMA M tile)
//     BLOCK_DMODEL= 512      (nope portion of the head dim)
//     BLOCK_DPE   = 64       (rope portion of the head dim)
//     BLOCK_DV    = 512      (value dim; == BLOCK_DMODEL for MLA)
//     HAS_MLA     = true     (V is the nope part of K; single shared buffer)
//     Q / K       = bf16, math in fp32, KV cache = bf16 for now.
//
// MFMA usage:
//   - qk  : v_mfma_f32_16x16x32_bf16 (K=32) reducing D=512 + DPE=64 -> 18 steps;
//           N_TILES [16,16] output tiles/chunk, one per wave (waves 0..N_TILES-1).
//   - p@v : v_mfma_f32_16x16x16_bf16 (K=16), N_TILES K-substeps accumulated per
//           DV tile, 32 DV tiles/chunk.
//
// PERFORMANCE (MI355X gfx950, bench_stage1_triton_vs_cuda.py) vs Triton
// _fwd_grouped_kernel_stage1, when the launch has enough blocks to fill the 256
// CUs (block count = batch * ceil(head/BLOCK_H) * num_kv_splits):
//   long-context DECODE (seq=80k, the tuning target):
//     h-q=12 (Kimi TP8), batch=32 splits=16: 1.59x    batch=8 splits=64: 1.27x
//   short-context (seq=4k): h-q=96: 1.21x   h-q=48: ~1.4x   h-q=128: ~1.4x
// BLOCK_N=32 is tuned for the long-context regime (many chunks/block): it halves
// the per-chunk barrier count and adds memory-level parallelism, at the cost of
// LDS (2 wg/CU). BLOCK_N=16 is ~1.57x at short-context h-q=96 but only ~1.46x on
// the long-context target — the opposite trade. See NOTES §7.
// At h-q=12 with too few num_kv_splits (block count < 2*CU) the grid is
// occupancy-starved (<1 wg/CU) -> <1x; more splits fixes it (and long context
// makes many splits free). Our latency IMPROVES with splits, Triton's degrades.
//
// Three optimizations drive this:
//  1. LDS layout (the big one): K is staged in *natural* [token][dim] order (see
//     below), NOT transposed. rocprof on the transposed version showed it was
//     entirely LDS-bank-stall bound (SQ_WAIT_INST_LDS ~72k cyc/wave); 3x.
//  2. Software pipelining: chunk i+1's K is prefetched global->registers while
//     chunk i computes, then stored into k_sh after p@v reads it — hiding the
//     K-load latency.
//  3. BLOCK_N=32: fewer barriers + more MLP per chunk for long streams.
//
// The kernel is latency-bound (not BW/MFMA-bound): waves stall ~65-70% of their
// life, MFMA busy ~7%, and halving KV traffic (bigger BLOCK_H) does NOT help.
// Parallel-qk, LDS double-buffering, larger BLOCK_H, BLOCK_N=64, and freeing LDS
// for occupancy were all tried and regressed (NOTES §7).
//
// LDS budget: Q IS staged in LDS (q_sh, 18 KB) — the block is VGPR-bound with Q
// in registers, and MI355X has 160 KB LDS/CU. At BLOCK_N=32 resident tiles are
// ~72 KB -> ~2 workgroups/CU.
//
// fp8 KV cache lives in the sibling kernel decode_grouped_attention_mla_stage1_fp8.cu
// (native fp8 MFMA; 1.81x at the long-context target). Not ported (inactive for
// this config): logit_cap / xai_temperature / SCORE_MOD / USE_PDL, PAGE_SIZE > 1.

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#error "This MFMA kernel is CDNA-only (gfx950). Compile with hipcc for AMD."
#endif

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace {

// ----- Compile-time tile shape ---------------------------------------------
constexpr int BLOCK_H = 16;       // query heads per program == MFMA M
constexpr int BLOCK_N = 32;       // kv tokens per inner step (N_TILES MFMA N-tiles)
constexpr int D = 512;            // BLOCK_DMODEL (nope)
constexpr int DPE = 64;           // BLOCK_DPE   (rope)
constexpr int DV = 512;           // BLOCK_DV
constexpr int MIN_BLOCK_KV = 32;  // _MIN_BLOCK_KV
constexpr int MMA_N = 16;         // MFMA output-N tile
constexpr int N_TILES = BLOCK_N / MMA_N;  // qk N-tiles / p@v K-substeps per chunk

// qk contraction: wide K=32 bf16 MFMA over the full latent (nope + rope).
constexpr int QK_MMA_K = 32;
constexpr int QK_K_PER_LANE = 8;             // bf16/lane on a K=32 operand
constexpr int QK_STEPS_NOPE = D / QK_MMA_K;  // 16
constexpr int QK_STEPS_ROPE = DPE / QK_MMA_K;  // 2
constexpr int QK_STEPS = QK_STEPS_NOPE + QK_STEPS_ROPE;  // 18 K=32 steps over Lk

// p@v contraction: K=16 bf16 MFMA (each token-16 subtile).
constexpr int PV_K_PER_LANE = 4;  // bf16/lane on a K=16 operand

constexpr int NUM_WAVES = 4;                 // 256 threads / 64-lane wavefront
constexpr int THREADS = NUM_WAVES * 64;      // 256
constexpr int DV_PER_WAVE = DV / NUM_WAVES;  // 128 => 8 output n-tiles per wave
constexpr int NTILES_PER_WAVE = DV_PER_WAVE / MMA_N;  // 8

constexpr float NEG_INF = -__builtin_huge_valf();

// Vectorized global loads: one 128-bit (int4) transaction = 8 bf16 along the
// contiguous head-dim. Assumes Lk (== D + DPE) is a multiple of VEC and the
// head-dim is the innermost (stride-1) axis of Q / K_Buffer — both hold here.
constexpr int VEC = 8;
constexpr int LK = D + DPE;        // 576, full latent width per row
constexpr int LK_VECS = LK / VEC;  // 72 vec-loads per row
constexpr int NOPE_VECS = D / VEC; // 64 => vec < 64 is nope, >= 64 is rope

// MFMA operand vectors. qk uses native 8-wide bf16 (gfx950 K=32); p@v uses the
// K=16 _1k op whose operands are <4 x i16> (bf16 bits reinterpreted).
using bf16x8 = __attribute__((ext_vector_type(8))) __bf16;
using bf16x4i = __attribute__((ext_vector_type(4))) short;
using f32x4 = __attribute__((ext_vector_type(4))) float;

static __device__ __forceinline__ int ceil_div_dev(int a, int b) {
  return (a + b - 1) / b;
}

// float -> bf16 with RNE rounding (matches Triton's .to(bf16)).
static __device__ __forceinline__ __bf16 f2bf(float f) {
  __hip_bfloat16 b = __float2bfloat16(f);
  return __builtin_bit_cast(__bf16, b);
}
static __device__ __forceinline__ short bf2s(__bf16 b) {
  return __builtin_bit_cast(short, b);
}

// qk: D[16x16] += A[16x32] * B[32x16].  p@v: D[16x16] += A[16x16] * B[16x16].
static __device__ __forceinline__ f32x4 mma_16x16x32(bf16x8 a, bf16x8 b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}
static __device__ __forceinline__ f32x4 mma_16x16x16(bf16x4i a, bf16x4i b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// Vectorized 128-bit KV load: 8 contiguous bf16 (one head-dim vec). bf16 only
// for now; the fp8 cache path needs its own overload (8x fp8 = 64-bit).
static __device__ __forceinline__ void load8(const __hip_bfloat16* p, __bf16 out[VEC]) {
  const int4 v = *reinterpret_cast<const int4*>(p);
#pragma unroll
  for (int e = 0; e < VEC; ++e) out[e] = reinterpret_cast<const __bf16*>(&v)[e];
}

// MFMA register<->matrix layout (gfx950), lane L in [0,64): blk = L/16, mn = L%16.
//   K=32 (qk):  A[m][k]: m=mn, k=blk*8+v (v=0..7);  B[k][n]: k=blk*8+v, n=mn
//   K=16 (p@v): A[m][k]: m=mn, k=blk*4+v (v=0..3);  B[k][n]: k=blk*4+v, n=mn
//   D[m][n]  (both): m=blk*4+v (v=0..3), n=mn                       -> f32x4
static __device__ __forceinline__ float get(const f32x4& r, int v) { return r[v]; }
static __device__ __forceinline__ void set(f32x4& r, int v, float x) { r[v] = x; }
static __device__ __forceinline__ void set(bf16x8& r, int v, __bf16 x) { r[v] = x; }
static __device__ __forceinline__ void set(bf16x4i& r, int v, __bf16 x) { r[v] = bf2s(x); }

}  // namespace

// -----------------------------------------------------------------------------
// One block == (cur_batch, cur_head_block, split_kv_id); 256 threads / 4 waves.
// Q and K are staged in LDS (K in natural [token][dim] layout). Wave 0 computes
// qk + streaming softmax into LDS; every wave owns a DV slice of the accumulator
// and consumes the shared p / rescale to build p@v (V is k_sh reused).
// -----------------------------------------------------------------------------
template <typename KV_T>
__global__ void __launch_bounds__(THREADS) fwd_grouped_kernel_stage1_mla(
    const __hip_bfloat16* __restrict__ Q,   // [batch, head_num, Lk]
    const KV_T* __restrict__ K_Buffer,      // [max_slots, kv_head_num, Lk]
    float sm_scale_withk,
    const int* __restrict__ kv_indptr,      // [batch + 1]
    const int* __restrict__ kv_indices,     // [total_kv]
    float* __restrict__ Att_Out,            // [batch, head_num, max_kv_splits, Lv]
    float* __restrict__ Att_Lse,            // [batch, head_num, max_kv_splits]
    const int* __restrict__ num_kv_splits,  // [batch]
    long stride_qbs, long stride_qh,
    long stride_buf_kbs, long stride_buf_kh,
    long stride_mid_ob, long stride_mid_oh, long stride_mid_os,
    int kv_group_num, int q_head_num, int Lv) {
  const long cur_batch = blockIdx.x;
  const int cur_head_id = blockIdx.y;
  const int split_kv_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;    // 0..3
  const int lane = tid & 63;    // 0..63
  const int blk = lane >> 4;    // 0..3
  const int mn = lane & 15;     // 0..15

  // Head-block bookkeeping (VALID_BLOCK_H / mask_h in Triton).
  const int valid_block_h = BLOCK_H < kv_group_num ? BLOCK_H : kv_group_num;
  const int cur_kv_head = cur_head_id / ceil_div_dev(kv_group_num, BLOCK_H);
  const int head_lo = cur_head_id * valid_block_h;
  const int head_hi = min(head_lo + valid_block_h, q_head_num);

  const int cur_batch_kv_start = kv_indptr[cur_batch];
  const int cur_batch_seq_len = kv_indptr[cur_batch + 1] - cur_batch_kv_start;
  const int kv_splits = num_kv_splits[cur_batch];

  const int kv_len_per_split =
      ceil_div_dev(ceil_div_dev(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV;
  const int split_kv_start = kv_len_per_split * split_kv_id;
  const int split_kv_end = min(split_kv_start + kv_len_per_split, cur_batch_seq_len);
  if (split_kv_end <= split_kv_start) return;  // no store, as in Triton

  // ----- Shared tiles (native bf16, natural [token][dim] layout). -------------
  // K is staged as [token][dim] (dim innermost), NOT the transpose. This makes
  // BOTH the global->LDS store and the qk B-operand read contiguous 128-bit LDS
  // ops (the old [dim][token] layout forced a scalar-strided store + 8 scalar
  // reads/step, which rocprof showed dominated runtime via LDS bank stalls).
  // Rows are padded (KPAD) so the row stride in 4B-words isn't a multiple of 32,
  // breaking the all-lanes-same-bank aliasing on the strided p@v V-read.
  // Q is likewise staged in LDS: the block is VGPR-bound on gfx950 and LDS is
  // 160 KB/CU, so q_sh/k_sh are nearly free for occupancy.
  constexpr int KPAD = 8;                       // row pad (bf16); (D+KPAD)/2 %32 != 0
  __shared__ __bf16 q_sh[BLOCK_H][LK];          // 18 KB (Q latent; qk A-operand)
  __shared__ __bf16 k_sh[BLOCK_N][D + KPAD];    // nope K, also V for p@v
  __shared__ __bf16 kpe_sh[BLOCK_N][DPE + KPAD];// rope K (qk only)
  __shared__ __bf16 p_sh[BLOCK_H][BLOCK_N];
  __shared__ float qk_sh[BLOCK_H][BLOCK_N];
  __shared__ float e_max_sh[BLOCK_H];
  __shared__ float e_sum_sh[BLOCK_H];
  __shared__ float rescale_sh[BLOCK_H];

  // Per-thread staging registers for one K chunk. The loop is software-pipelined:
  // chunk i+1 is prefetched global->registers while chunk i computes, then stored
  // into the (single) k_sh after p@v has finished reading it. This hides the
  // K-load latency, which is fully exposed at low head counts (few head-blocks =>
  // ~1 workgroup/CU, so occupancy can't overlap it). LDS is unchanged from the
  // non-pipelined version, so occupancy at high head counts is preserved.
  constexpr int KREG = (BLOCK_N * LK_VECS + THREADS - 1) / THREADS;  // 5
  int4 rk[KREG];

  // Per-wave accumulator: NTILES_PER_WAVE tiles of [16 x 16], D[v] -> (m,n).
  f32x4 acc[NTILES_PER_WAVE];
#pragma unroll
  for (int t = 0; t < NTILES_PER_WAVE; ++t) acc[t] = f32x4{0, 0, 0, 0};

  if (tid < BLOCK_H) {
    e_max_sh[tid] = NEG_INF;
    e_sum_sh[tid] = 0.0f;
  }

  // ----- Load Q into LDS (loop-invariant) ------------------------------------
  // All waves cooperate on a coalesced + vectorized (128-bit) copy of the Q rows;
  // q_sh[h][:] is head head_lo+h's full latent (nope then rope), contiguous, so
  // the qk A-operand is one bf16x8 per step. Staged once and reused every chunk
  // (Q-from-global regressed at long context — many chunks re-read it).
  for (int j = tid; j < BLOCK_H * LK_VECS; j += THREADS) {
    const int h = j / LK_VECS, vec = j % LK_VECS, d = vec * VEC;
    const int head = head_lo + h;
    if (head < head_hi) {
      const long qoff = cur_batch * stride_qbs + (long)head * stride_qh + d;
      *reinterpret_cast<int4*>(&q_sh[h][d]) = *reinterpret_cast<const int4*>(Q + qoff);
    } else {
      *reinterpret_cast<int4*>(&q_sh[h][d]) = int4{0, 0, 0, 0};
    }
  }
  __syncthreads();

  // ----- Stream over KV in BLOCK_N chunks (software-pipelined) ----------------
  // load_chunk: gather one KV chunk global->registers (rk), inlining the kv_loc
  //   lookup. store_chunk: commit rk->k_sh/kpe_sh (natural [token][dim]).
  const int nchunks = ceil_div_dev(split_kv_end - split_kv_start, BLOCK_N);

  auto load_chunk = [&](int sn) {
#pragma unroll
    for (int it = 0; it < KREG; ++it) {
      const int j = tid + it * THREADS;
      if (j < BLOCK_N * LK_VECS) {
        const int n = j / LK_VECS, vec = j % LK_VECS, d = vec * VEC;
        if ((sn + n) < split_kv_end) {
          const int kvloc = kv_indices[cur_batch_kv_start + sn + n];
          rk[it] = *reinterpret_cast<const int4*>(
              K_Buffer + (long)kvloc * stride_buf_kbs +
              (long)cur_kv_head * stride_buf_kh + d);
        } else {
          rk[it] = int4{0, 0, 0, 0};
        }
      }
    }
  };
  auto store_chunk = [&]() {
#pragma unroll
    for (int it = 0; it < KREG; ++it) {
      const int j = tid + it * THREADS;
      if (j < BLOCK_N * LK_VECS) {
        const int n = j / LK_VECS, vec = j % LK_VECS, d = vec * VEC;
        if (vec < NOPE_VECS) {
          *reinterpret_cast<int4*>(&k_sh[n][d]) = rk[it];
        } else {
          *reinterpret_cast<int4*>(&kpe_sh[n][d - D]) = rk[it];
        }
      }
    }
  };

  // Prologue: prefetch + stage chunk 0.
  load_chunk(split_kv_start);
  store_chunk();
  __syncthreads();

  for (int i = 0; i < nchunks; ++i) {
    const int start_n = split_kv_start + i * BLOCK_N;
    const bool has_next = (i + 1) < nchunks;

    // Prefetch chunk i+1's K into registers. These global loads stay in flight
    // (independent of the compute below) and are consumed only at store_chunk.
    if (has_next) load_chunk(start_n + BLOCK_N);

    // (c) qk = q@k + qpe@kpe -> qk_sh, with scale + masking. BLOCK_N tokens =
    //     N_TILES MFMA N-tiles, distributed one-per-wave (wave w owns n-tile w).
    //     Each is one [16 head x 16 token] tile; K=32 steps over nope then rope.
    if (wave < N_TILES) {
      const int nt = wave;
      const int tok = nt * MMA_N + mn;  // token within the chunk (n == mn per tile)
      f32x4 d = f32x4{0, 0, 0, 0};
      // A[m=head=mn][k=dim], B[k=dim][n=token] both read one bf16x8 contiguously.
#pragma unroll
      for (int kk = 0; kk < QK_STEPS_NOPE; ++kk) {
        const int kbase = kk * QK_MMA_K + blk * QK_K_PER_LANE;
        const bf16x8 a = *reinterpret_cast<const bf16x8*>(&q_sh[mn][kbase]);
        const bf16x8 b = *reinterpret_cast<const bf16x8*>(&k_sh[tok][kbase]);
        d = mma_16x16x32(a, b, d);
      }
#pragma unroll
      for (int kk = 0; kk < QK_STEPS_ROPE; ++kk) {
        const int kbase = kk * QK_MMA_K + blk * QK_K_PER_LANE;  // rope-local
        // rope lives at q_sh columns [D, D+DPE); kpe_sh is rope-only.
        const bf16x8 a = *reinterpret_cast<const bf16x8*>(&q_sh[mn][D + kbase]);
        const bf16x8 b = *reinterpret_cast<const bf16x8*>(&kpe_sh[tok][kbase]);
        d = mma_16x16x32(a, b, d);
      }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int m = blk * 4 + v;  // D[v] -> (m, token)
        const bool live = (start_n + tok) < split_kv_end && (head_lo + m) < head_hi;
        qk_sh[m][tok] = live ? get(d, v) * sm_scale_withk : NEG_INF;
      }
    }
    __syncthreads();

    // (d) Streaming-softmax row update (wave 0, one lane per head).
    if (tid < BLOCK_H) {
      const int m = tid;
      float row_max = NEG_INF;
#pragma unroll
      for (int n = 0; n < BLOCK_N; ++n) row_max = fmaxf(row_max, qk_sh[m][n]);
      const float new_max = fmaxf(row_max, e_max_sh[m]);
      const float rescale = __expf(e_max_sh[m] - new_max);
      float row_sum = 0.0f;
#pragma unroll
      for (int n = 0; n < BLOCK_N; ++n) {
        const float p = __expf(qk_sh[m][n] - new_max);
        p_sh[m][n] = f2bf(p);
        row_sum += p;
      }
      e_sum_sh[m] = e_sum_sh[m] * rescale + row_sum;
      e_max_sh[m] = new_max;
      rescale_sh[m] = rescale;
    }
    __syncthreads();

    // (e) acc[m][dv] = acc[m][dv]*rescale[m] + (p @ v)[m][dv]; each wave owns dv
    //     in [wave*128, +128). BLOCK_N tokens => N_TILES K=16 MFMA per DV tile,
    //     accumulated (V[token][dv] == k_sh[token][dv]).
#pragma unroll
    for (int t = 0; t < NTILES_PER_WAVE; ++t) {
      const int nn = wave * NTILES_PER_WAVE + t;  // dv-tile index (0..31)
      f32x4 pv = f32x4{0, 0, 0, 0};
#pragma unroll
      for (int ksub = 0; ksub < N_TILES; ++ksub) {
        bf16x4i a, b;
#pragma unroll
        for (int v = 0; v < PV_K_PER_LANE; ++v) {
          const int k = ksub * MMA_N + blk * PV_K_PER_LANE + v;  // token in chunk
          set(a, v, p_sh[mn][k]);                   // A[m=head=mn][k=token]
          set(b, v, k_sh[k][nn * MMA_N + mn]);      // B[k=token][n=dv]==V[token][dv]
        }
        pv = mma_16x16x16(a, b, pv);
      }
#pragma unroll
      for (int v = 0; v < 4; ++v)
        set(acc[t], v, get(acc[t], v) * rescale_sh[blk * 4 + v] + get(pv, v));
    }
    __syncthreads();  // p@v done reading k_sh; safe to overwrite it

    // Commit the prefetched next chunk into the (single) k_sh buffer. store_chunk
    // waits on the in-flight loads (vmcnt) here — after they overlapped compute.
    if (has_next) {
      store_chunk();
      __syncthreads();
    }
  }

  // ----- Epilogue: normalize + store -----------------------------------------
#pragma unroll
  for (int t = 0; t < NTILES_PER_WAVE; ++t) {
    const int nn = wave * NTILES_PER_WAVE + t;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int m = blk * 4 + v, dv = nn * MMA_N + mn;
      const int head = head_lo + m;
      if (head < head_hi && dv < Lv) {
        const long off = cur_batch * stride_mid_ob + (long)head * stride_mid_oh +
                         (long)split_kv_id * stride_mid_os + dv;
        Att_Out[off] = get(acc[t], v) / e_sum_sh[m];
      }
    }
  }

  if (tid < BLOCK_H) {
    const int head = head_lo + tid;
    if (head < head_hi) {
      // Triton: offs = (b*ob + head*oh + split*os) // Lv  (Att_Lse has no dv dim)
      const long off = (cur_batch * stride_mid_ob + (long)head * stride_mid_oh +
                        (long)split_kv_id * stride_mid_os) /
                       Lv;
      Att_Lse[off] = e_max_sh[tid] + logf(e_sum_sh[tid]);
    }
  }
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

// -----------------------------------------------------------------------------
// Host launcher (bf16 KV). Grid/block mirror _decode_grouped_att_m_fwd.
// -----------------------------------------------------------------------------
void launch_fwd_grouped_stage1_mla(
    const __hip_bfloat16* Q, const __hip_bfloat16* K_Buffer, float sm_scale_withk,
    const int* kv_indptr, const int* kv_indices, float* Att_Out, float* Att_Lse,
    const int* num_kv_splits, long stride_qbs, long stride_qh, long stride_buf_kbs,
    long stride_buf_kh, long stride_mid_ob, long stride_mid_oh, long stride_mid_os,
    int batch, int q_head_num, int kv_group_num, int max_kv_splits, int Lv,
    hipStream_t stream) {
  const int valid_block_h = BLOCK_H < kv_group_num ? BLOCK_H : kv_group_num;
  dim3 grid(batch, ceil_div(q_head_num, valid_block_h), max_kv_splits);
  dim3 block(THREADS);
  fwd_grouped_kernel_stage1_mla<__hip_bfloat16><<<grid, block, 0, stream>>>(
      Q, K_Buffer, sm_scale_withk, kv_indptr, kv_indices, Att_Out, Att_Lse,
      num_kv_splits, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh,
      stride_mid_ob, stride_mid_oh, stride_mid_os, kv_group_num, q_head_num, Lv);
}
