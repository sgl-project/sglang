// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
// SPDX-License-Identifier: Apache-2.0
//
// HIP / CDNA4 (MI350, gfx950) MFMA port of the Triton _fwd_grouped_kernel_stage1
// (MLA decode, stage 1) — **fp8 (e4m3) KV cache** variant.
//
// This is the sibling of decode_grouped_attention_mla_stage1.cu (bf16 KV). It is
// a SEPARATE kernel because the fp8 path differs structurally, not just in a load
// helper: the KV cache, Q, and the softmax probabilities all live as fp8 in LDS
// and feed *native fp8 MFMA* (v_mfma_f32_16x16x32_fp8_fp8), which is exactly the
// recipe the Triton kernel uses when K_Buffer is fp8 (q.to(fp8), fp8 dot, p.to(fp8)).
//
// Why a native-fp8 path (vs dequant-to-bf16 in the bf16 kernel): the bf16 kernel
// is latency/barrier-bound at ~2 wg/CU, and simply halving HBM traffic does not
// help (memory latency is already hidden). Keeping everything fp8 IN LDS halves
// q_sh/k_sh, which ~doubles occupancy (→ ~4-5 wg/CU) and *hides the barrier
// waits* — that is where the speed comes from. It also avoids the per-element
// dequant that made bf16-kernel+fp8-load slower.
//
// Layout: the gfx950 fp8 K=32 MFMA uses the SAME lane<->matrix mapping as the
// bf16 K=32 MFMA (verified with a standalone reference), so the tiling mirrors
// the bf16 kernel one-to-one; only the operand type (8 fp8 packed in a `long`)
// and the LDS element type change.
//
// Config: PAGE_SIZE=1, BLOCK_H=16, BLOCK_N=32, D=512, DPE=64, DV=512, HAS_MLA.
// The caller folds k_scale into sm_scale_withk (as Triton does); v_scale is
// applied in stage 2. Q is quantized to fp8 with no separate scale (matches
// Triton's q.to(fp8)); callers whose Q exceeds e4m3 range must pre-scale.

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#error "This MFMA kernel is CDNA-only (gfx950). Compile with hipcc for AMD."
#endif

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace {

// ----- Compile-time tile shape (mirrors the bf16 kernel) -------------------
constexpr int BLOCK_H = 16;
constexpr int BLOCK_N = 32;
constexpr int D = 512;
constexpr int DPE = 64;
constexpr int DV = 512;
constexpr int MIN_BLOCK_KV = 32;
constexpr int MMA_N = 16;
constexpr int N_TILES = BLOCK_N / MMA_N;  // 2 qk N-tiles / chunk

// qk & p@v both use the wide K=32 fp8 MFMA (8 fp8/lane = one `long`).
constexpr int MMA_K = 32;
constexpr int K_PER_LANE = 8;                 // fp8/lane on a K=32 operand
constexpr int QK_STEPS_NOPE = D / MMA_K;      // 16
constexpr int QK_STEPS_ROPE = DPE / MMA_K;    // 2
constexpr int PV_KSUB = BLOCK_N / MMA_K;      // 1 (BLOCK_N == MMA_K)

constexpr int NUM_WAVES = 4;
constexpr int THREADS = NUM_WAVES * 64;
constexpr int DV_PER_WAVE = DV / NUM_WAVES;   // 128
constexpr int NTILES_PER_WAVE = DV_PER_WAVE / MMA_N;  // 8

constexpr float NEG_INF = -__builtin_huge_valf();

// fp8 element + LDS row padding. k_sh rows are padded so the byte stride isn't a
// multiple of 128 (32 banks x 4 B), breaking bank aliasing on the strided p@v
// V-read. (D=512 B is 4*128 -> aliases; +KPAD fixes it.)
using fp8_t = __hip_fp8_e4m3;
constexpr int KPAD = 16;
constexpr int LK = D + DPE;         // 576
constexpr int VEC = 16;             // 16 fp8 = 16 B (int4) per global/LDS vec
constexpr int LK_VECS = LK / VEC;   // 36
constexpr int NOPE_VECS = D / VEC;  // 32 => vec < 32 is nope, >= 32 is rope

using f32x4 = __attribute__((ext_vector_type(4))) float;

static __device__ __forceinline__ int ceil_div_dev(int a, int b) {
  return (a + b - 1) / b;
}

// float -> e4m3 (matches torch float8_e4m3fn / Triton .to(fp8)).
static __device__ __forceinline__ fp8_t f2fp8(float f) { return fp8_t(f); }

// D[16x16] += A[16x32] * B[32x16], fp8 x fp8, 8 fp8/lane packed in a `long`.
static __device__ __forceinline__ f32x4 mma_fp8(long a, long b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, 0);
}

static __device__ __forceinline__ float get(const f32x4& r, int v) { return r[v]; }
static __device__ __forceinline__ void set(f32x4& r, int v, float x) { r[v] = x; }

// Read 8 contiguous fp8 (one K=32 lane operand) from an fp8 LDS row as a `long`.
static __device__ __forceinline__ long op8(const fp8_t* p) {
  return *reinterpret_cast<const long*>(p);
}

}  // namespace

// -----------------------------------------------------------------------------
// One block == (cur_batch, cur_head_block, split_kv_id); 256 threads / 4 waves.
// Q, K, and softmax p are staged in LDS as fp8 (half the bf16 footprint). Waves
// 0..N_TILES-1 compute qk; softmax runs on wave 0; every wave owns a DV slice of
// the accumulator and builds p@v from the shared fp8 p / k_sh (V).
// -----------------------------------------------------------------------------
// No min-blocks hint: the fp8 LDS (~31 KB) would allow ~5 wg/CU, but forcing the
// VGPR down to realize that (__launch_bounds__(THREADS,4) -> 128 VGPR, 4 wg/CU)
// measured SLOWER than letting the compiler keep ~184 VGPR at 2 wg/CU (better
// ILP). The fp8 win is native fp8 MFMA + fp8 LDS + half HBM, not occupancy.
__global__ void __launch_bounds__(THREADS) fwd_grouped_kernel_stage1_mla_fp8(
    const __hip_bfloat16* __restrict__ Q,   // [batch, head_num, Lk] bf16
    const fp8_t* __restrict__ K_Buffer,     // [max_slots, kv_head_num, Lk] fp8
    float sm_scale_withk,                   // caller folds k_scale in
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    float* __restrict__ Att_Out,            // [batch, head_num, max_kv_splits, Lv]
    float* __restrict__ Att_Lse,            // [batch, head_num, max_kv_splits]
    const int* __restrict__ num_kv_splits,
    long stride_qbs, long stride_qh,
    long stride_buf_kbs, long stride_buf_kh,
    long stride_mid_ob, long stride_mid_oh, long stride_mid_os,
    int kv_group_num, int q_head_num, int Lv) {
  const long cur_batch = blockIdx.x;
  const int cur_head_id = blockIdx.y;
  const int split_kv_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int blk = lane >> 4;
  const int mn = lane & 15;

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
  if (split_kv_end <= split_kv_start) return;

  // ----- Shared tiles: Q / K / p in fp8 (half the bf16 LDS -> ~2x occupancy) --
  __shared__ fp8_t q_sh[BLOCK_H][LK];             // 9 KB
  __shared__ fp8_t k_sh[BLOCK_N][D + KPAD];       // nope K, also V for p@v
  __shared__ fp8_t kpe_sh[BLOCK_N][DPE + KPAD];   // rope K (qk only)
  __shared__ fp8_t p_sh[BLOCK_H][BLOCK_N];        // softmax probs, fp8 for p@v
  __shared__ float qk_sh[BLOCK_H][BLOCK_N];
  __shared__ float e_max_sh[BLOCK_H];
  __shared__ float e_sum_sh[BLOCK_H];
  __shared__ float rescale_sh[BLOCK_H];

  // Per-thread K-chunk staging (fp8, half the bf16 staging). Software-pipelined.
  constexpr int KREG = (BLOCK_N * LK_VECS + THREADS - 1) / THREADS;  // 5
  int4 rk[KREG];  // 16 fp8 per int4

  f32x4 acc[NTILES_PER_WAVE];
#pragma unroll
  for (int t = 0; t < NTILES_PER_WAVE; ++t) acc[t] = f32x4{0, 0, 0, 0};

  if (tid < BLOCK_H) {
    e_max_sh[tid] = NEG_INF;
    e_sum_sh[tid] = 0.0f;
  }

  // ----- Load Q, quantize bf16 -> fp8 into q_sh (loop-invariant) --------------
  // Each thread converts VEC=16 contiguous Q elements per iteration.
  for (int j = tid; j < BLOCK_H * LK_VECS; j += THREADS) {
    const int h = j / LK_VECS, vec = j % LK_VECS, d = vec * VEC;
    const int head = head_lo + h;
    fp8_t out[VEC];
    if (head < head_hi) {
      const __hip_bfloat16* qp = Q + cur_batch * stride_qbs + (long)head * stride_qh + d;
#pragma unroll
      for (int e = 0; e < VEC; ++e) out[e] = f2fp8((float)qp[e]);
    } else {
#pragma unroll
      for (int e = 0; e < VEC; ++e) out[e] = f2fp8(0.0f);
    }
    *reinterpret_cast<int4*>(&q_sh[h][d]) = *reinterpret_cast<const int4*>(out);
  }
  __syncthreads();

  // ----- Stream over KV in BLOCK_N chunks (software-pipelined) ----------------
  const int nchunks = ceil_div_dev(split_kv_end - split_kv_start, BLOCK_N);

  auto load_chunk = [&](int sn) {
#pragma unroll
    for (int it = 0; it < KREG; ++it) {
      const int j = tid + it * THREADS;
      if (j < BLOCK_N * LK_VECS) {
        const int n = j / LK_VECS, vec = j % LK_VECS, d = vec * VEC;
        if ((sn + n) < split_kv_end) {
          const int kvloc = kv_indices[cur_batch_kv_start + sn + n];
          // fp8 K is copied byte-for-byte HBM -> LDS (no dequant): one int4 = 16
          // fp8. This is HALF the HBM bytes of the bf16 kernel.
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

  load_chunk(split_kv_start);
  store_chunk();
  __syncthreads();

  for (int i = 0; i < nchunks; ++i) {
    const int start_n = split_kv_start + i * BLOCK_N;
    const bool has_next = (i + 1) < nchunks;
    if (has_next) load_chunk(start_n + BLOCK_N);

    // (c) qk = q@k + qpe@kpe -> qk_sh, scale + mask. N_TILES tiles, one per wave.
    if (wave < N_TILES) {
      const int nt = wave;
      const int tok = nt * MMA_N + mn;  // token in chunk (n == mn per tile)
      f32x4 d = f32x4{0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < QK_STEPS_NOPE; ++kk) {
        const int kbase = kk * MMA_K + blk * K_PER_LANE;
        d = mma_fp8(op8(&q_sh[mn][kbase]), op8(&k_sh[tok][kbase]), d);
      }
#pragma unroll
      for (int kk = 0; kk < QK_STEPS_ROPE; ++kk) {
        const int kbase = kk * MMA_K + blk * K_PER_LANE;  // rope-local
        d = mma_fp8(op8(&q_sh[mn][D + kbase]), op8(&kpe_sh[tok][kbase]), d);
      }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int m = blk * 4 + v;  // head-within-block (D-layout row); n-tiles
                                    // share the 16 heads, differ only in tokens
        const bool live = (start_n + tok) < split_kv_end && (head_lo + m) < head_hi;
        qk_sh[m][tok] = live ? get(d, v) * sm_scale_withk : NEG_INF;
      }
    }
    __syncthreads();

    // (d) Streaming softmax (wave 0, one lane per head). p -> fp8 for p@v.
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
        p_sh[m][n] = f2fp8(p);
        row_sum += p;
      }
      e_sum_sh[m] = e_sum_sh[m] * rescale + row_sum;
      e_max_sh[m] = new_max;
      rescale_sh[m] = rescale;
    }
    __syncthreads();

    // (e) acc += p @ V, V == k_sh (fp8). BLOCK_N==MMA_K -> one K=32 fp8 MFMA/DV.
    //     A[m=head][k=token] from p_sh; B[k=token][n=dv] gathered from k_sh.
    float rs[4];
#pragma unroll
    for (int v = 0; v < 4; ++v) rs[v] = rescale_sh[blk * 4 + v];
#pragma unroll
    for (int t = 0; t < NTILES_PER_WAVE; ++t) {
      const int nn = wave * NTILES_PER_WAVE + t;  // dv-tile (0..31)
      const long a = op8(&p_sh[mn][blk * K_PER_LANE]);  // 8 tokens, contiguous
      // B: 8 tokens (blk*8..+7) at fixed dv=nn*16+mn, strided by k_sh row.
      long b = 0;
#pragma unroll
      for (int v = 0; v < K_PER_LANE; ++v) {
        const int k = blk * K_PER_LANE + v;  // token
        const unsigned char bv =
            *reinterpret_cast<const unsigned char*>(&k_sh[k][nn * MMA_N + mn]);
        b |= (long)bv << (v * 8);
      }
      const f32x4 pv = mma_fp8(a, b, f32x4{0, 0, 0, 0});
#pragma unroll
      for (int v = 0; v < 4; ++v)
        set(acc[t], v, get(acc[t], v) * rs[v] + get(pv, v));
    }
    __syncthreads();

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
      const long off = (cur_batch * stride_mid_ob + (long)head * stride_mid_oh +
                        (long)split_kv_id * stride_mid_os) / Lv;
      Att_Lse[off] = e_max_sh[tid] + logf(e_sum_sh[tid]);
    }
  }
}

static int ceil_div_h(int a, int b) { return (a + b - 1) / b; }

void launch_fwd_grouped_stage1_mla_fp8(
    const __hip_bfloat16* Q, const fp8_t* K_Buffer, float sm_scale_withk,
    const int* kv_indptr, const int* kv_indices, float* Att_Out, float* Att_Lse,
    const int* num_kv_splits, long stride_qbs, long stride_qh, long stride_buf_kbs,
    long stride_buf_kh, long stride_mid_ob, long stride_mid_oh, long stride_mid_os,
    int batch, int q_head_num, int kv_group_num, int max_kv_splits, int Lv,
    hipStream_t stream) {
  const int valid_block_h = BLOCK_H < kv_group_num ? BLOCK_H : kv_group_num;
  dim3 grid(batch, ceil_div_h(q_head_num, valid_block_h), max_kv_splits);
  dim3 block(THREADS);
  fwd_grouped_kernel_stage1_mla_fp8<<<grid, block, 0, stream>>>(
      Q, K_Buffer, sm_scale_withk, kv_indptr, kv_indices, Att_Out, Att_Lse,
      num_kv_splits, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh,
      stride_mid_ob, stride_mid_oh, stride_mid_os, kv_group_num, q_head_num, Lv);
}
