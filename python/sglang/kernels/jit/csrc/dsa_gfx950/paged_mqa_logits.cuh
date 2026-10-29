/// GLM-5.2 DSA indexer, the logits kernel: paged MQA FP8 logits plus the fine
/// histogram and coarse summary topk_transform.cuh selects from.  Three kernels
/// ship: the per-row logits_hist_m<8, 12>, the per-row logits_hist_s<8, 12> for
/// short batches, and the sibling-group logits_hist_grp<16, 12, 6>; the host
/// picks one per launch (``variant``).

#pragma once

#ifndef USE_ROCM
#error "paged_mqa_logits.cuh targets gfx950; it is raw CDNA4 MFMA and wave64"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace dsa_gfx950::logits {

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int i32x8;
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int i32x4;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float f32x4;

#define PAGE_TOK 64
#define HD 128
#define HEADS 32
// topk_transform.cuh's TOPK: both kernels must agree on which rows get binned.
#define LG_TOPK 2048
#define TOK_STRIDE 132
#define PAGE_BYTES (PAGE_TOK * TOK_STRIDE)  // 8448
#define K_BYTES (PAGE_TOK * HD)             // 8192

union V8 {
  i32x8 v;
  i32x4 h[2];
};

__device__ __forceinline__ f32x4 mfma128(i32x8 a, i32x8 b, f32x4 c) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127, 0, 127);
}

// wave64 lane exchanges on the VALU instead of through LDS:
//   xor32 <- v_permlane64_b32 (lane l reads lane l^32)
//   xor16 <- v_permlanex16_b32 with identity selectors (lane l reads lane l^16)
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int i32x2;
__device__ __forceinline__ float xor32(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane32_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 32) ? r[0] : r[1]);
}
__device__ __forceinline__ float xor16(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane16_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 16) ? r[0] : r[1]);
}

// The top-k stage's histogram, built here while this kernel waits on KV loads.
// It shares three invariants with topk_transform.cuh: the key (order_bin_fast),
// the npages clamp, and do_hist's TOPK.
#define LG_CBITS 6
#define LG_CBINS (1 << LG_CBITS)

// Fused paged-MQA FP8 logits + in-loop fine histogram + coarse summary.  The
// histogram is what lets the top-k stage skip a separate scoring pass.

// order_key16(x) >> LOWB in 4 VALU ops after the cvt, with no branch and no
// v_cndmask.  Bit-identical to topk_transform.cuh's order_key16(x) >> LOWB for
// every finite x and for +/-0.
template <int LOWB>
__device__ __forceinline__ uint32_t order_bin_fast(float x) {
  const uint32_t h = (uint32_t)__half_as_ushort(__float2half_rn(x));
  const uint32_t m = (uint32_t)((int32_t)__float_as_int(x) >> 31);
  return (h >> LOWB) ^ (0x8000u >> LOWB) ^ (m & (0x7fffu >> LOWB));
}

// ---- gated fine flush ----
// topk_transform.cuh reads, per binned row, the LG_CBINS coarse bins (whose total must equal the row length) and the
// fine bins of ONE coarse bin: j*, the bin where the coarse suffix sum (from the top) first reaches LG_TOPK.  Its flat
// fallback needs only the bins at or above its threshold.  So fine bins below coarse bin j* never affect any output,
// and a block only has to flush its fine bins from some gate <= j* upward; the coarse flush stays complete.
// The gate is the bin where the BLOCK's own coarse suffix sum reaches LG_TOPK: the block's elements are a subset of
// the row's, so its suffix sums are <= the row's and that bin cannot lie above j*.  (0, i.e. flush everything, when the
// block holds fewer than LG_TOPK elements of the row.)  Bins below it are skipped by every block, so their global
// counts stay partial: only n4's hierarchical threshold reads ghist, and it never looks there.
__device__ __forceinline__ uint32_t lg_wave_suffix_sum(uint32_t v, int lane) {
#pragma unroll
  for (int o = 1; o < 64; o <<= 1) {
    const uint32_t t = __shfl_down(v, o, 64);
    v += (lane + o < 64) ? t : 0u;
  }
  return v;
}

// The gate: n4's j predicate on this block's coarse counts, held one bin per lane; LG_CBINS (flush nothing) when the
// block holds no element of the row.  Wave-uniform.
__device__ __forceinline__ int lg_gate(uint32_t v, int lane) {
  const uint32_t s = lg_wave_suffix_sum(v, lane);
  if (__shfl(s, 0, 64) == 0u) return LG_CBINS;
  uint32_t sn = __shfl_down(s, 1, 64);
  if (lane == 63) sn = 0u;
  const unsigned long long b = __ballot(s >= LG_TOPK && sn < LG_TOPK);
  return b ? (int)__ffsll(b) - 1 : 0;
}

template <int WARPS, int HB>
__global__ __launch_bounds__(WARPS * 64) void logits_hist_m(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ wgt,
    const int* __restrict__ seqlens,
    const int* __restrict__ ptable,
    float* __restrict__ out,
    unsigned int* __restrict__ ghist,
    int max_pages,
    int out_stride) {
  constexpr int NBIN = 1 << HB;
  constexpr int LOWB = 16 - HB;
  constexpr int GHS = NBIN + LG_CBINS;
  constexpr int NTH = WARPS * 64;
  constexpr int PERT = NBIN / NTH;  // GLOBAL bins per thread (coarse map)
  constexpr int CSH = HB - LG_CBITS;
  constexpr int GRP = NTH / LG_CBINS;
  static_assert(NBIN >= 2 && (NBIN % 2) == 0, "paired flush needs even bins");
  static_assert(NBIN % NTH == 0 && NTH >= LG_CBINS, "");
  static_assert(PERT <= LG_CBINS && (LG_CBINS % PERT) == 0, "a thread's contiguous run must sit inside one coarse bin");
  __shared__ unsigned int s_hist[NBIN];

  const int row = blockIdx.y;
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int seqlen = seqlens[row];
  // seqlen is device data, so no host check can bound it.  Without the clamp
  // an oversized row walks into the NEXT row's page table -- valid memory,
  // plausible page ids, confidently wrong logits.
  const int npages_raw = (seqlen + PAGE_TOK - 1) / PAGE_TOK;
  // Bound the walk by BOTH the page table and `out`.  The host only requires
  // that `out` cover whole pages -- a captured table is as wide as the graph
  // while the row is only as long as seqlens says -- so a table wider than
  // `out` is legal, and without the second term an oversized seqlen writes
  // past the end of the row.
  const int pages_out = out_stride / PAGE_TOK;
  const int npages_cap = max_pages < pages_out ? max_pages : pages_out;
  const int npages = npages_raw < npages_cap ? npages_raw : npages_cap;
  const int g = lane >> 4, c = lane & 15;
  // Nothing to rank when the row is shorter than topk: everything is selected,
  // so the histogram, its two barriers and the epilogue all compile out.
  //
  // This predicate MUST use the same clamped length the paired top-k kernel
  // applies (it clamps row_ends by the page-table width before its own
  // row_len <= TOPK early return).  A row binned here but skipped there is
  // never zeroed, and the histogram stays dirty for the life of the process.
  const int seqlen_eff = seqlen < npages_cap * PAGE_TOK ? seqlen : npages_cap * PAGE_TOK;
  const bool do_hist = (seqlen_eff > LG_TOPK);

  if (do_hist) {
    for (int i = threadIdx.x; i < NBIN; i += NTH)
      s_hist[i] = 0u;
  }

  const uint8_t* qp = q + (size_t)row * (size_t)(HEADS * HD) + (size_t)c * HD + g * 32;
  V8 qa0, qa1;
  qa0.h[0] = *(const i32x4*)(qp);
  qa0.h[1] = *(const i32x4*)(qp + 16);
  qa1.h[0] = *(const i32x4*)(qp + 16 * HD);
  qa1.h[1] = *(const i32x4*)(qp + 16 * HD + 16);
  const float* wp = wgt + (size_t)row * HEADS + g * 4;
  f32x4 w0 = *(const f32x4*)(wp);
  f32x4 w1 = *(const f32x4*)(wp + 16);

  const int nwaves = gridDim.x * WARPS;
  const int wid = blockIdx.x * WARPS + warp;
  const int koff = g * 512 + c * 16;

  if (do_hist) __syncthreads();

  unsigned int* __restrict__ gh = ghist + (size_t)row * GHS;

  const int* __restrict__ pt = ptable + (size_t)row * max_pages;

  for (int p = wid; p < npages; p += nwaves) {
    const int phys = pt[p];
    const uint8_t* base = kv + (size_t)phys * PAGE_BYTES;
    const uint8_t* kb = base + koff;
    V8 b[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      b[tt].h[0] = *(const i32x4*)(kb + tt * 2048);
      b[tt].h[1] = *(const i32x4*)(kb + tt * 2048 + 256);
    }
    const float ks = ((const float*)(base + K_BYTES))[lane];
    float s[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      const f32x4 z = {0.f, 0.f, 0.f, 0.f};
      f32x4 a0 = mfma128(qa0.v, b[tt].v, z);
      f32x4 a1 = mfma128(qa1.v, b[tt].v, z);
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a0[i], 0.f), w0[i], acc);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a1[i], 0.f), w1[i], acc);
      s[tt] = acc;
    }
    const bool hi = (g & 2) != 0;
    float A0 = (hi ? s[2] : s[0]) + xor32(hi ? s[0] : s[2], lane);
    float A1 = (hi ? s[3] : s[1]) + xor32(hi ? s[1] : s[3], lane);
    const bool odd = (g & 1) != 0;
    float T = (odd ? A1 : A0) + xor16(odd ? A0 : A1, lane);
    const int pos = p * PAGE_TOK + lane;
    const bool live = pos < seqlen;
    const float v = T * ks;
    out[(size_t)row * out_stride + pos] = live ? v : -INFINITY;
    if (do_hist && live) {
      const uint32_t bb = order_bin_fast<LOWB>(v);
      atomicAdd(&s_hist[bb], 1u);
    }
  }

  if (do_hist) {
    __syncthreads();

    // ---- coarse summary, mapped over the GLOBAL bin index ----
    // Thread tx owns global bins [tx*PERT, tx*PERT+PERT), which lies inside
    // coarse bin (tx*PERT)>>CSH.  Needs no barrier of its own.
    const int g0 = (int)threadIdx.x * PERT;
    unsigned int acc = 0u;
#pragma unroll
    for (int j = 0; j < PERT; ++j)
      acc += s_hist[g0 + j];
#pragma unroll
    for (int o = GRP >> 1; o >= 1; o >>= 1)
      acc += __shfl_down(acc, o, 64);
    if ((threadIdx.x & (GRP - 1)) == 0 && acc) atomicAdd(&gh[NBIN + (g0 >> CSH)], acc);

    for (int i = threadIdx.x; i < NBIN / 2; i += NTH) {
      const unsigned long long vv = (unsigned long long)s_hist[2 * i] | ((unsigned long long)s_hist[2 * i + 1] << 32);
      if (vv) atomicAdd((unsigned long long*)(gh + 2 * i), vv);
    }
  }
}

// Per-row form tuned for short batches (rows <= 24 here): page ids preloaded one per lane, a per-page buffer
// resource, the gated fine flush, and longest-row-first dispatch when the grid needs more than one round of slots.
// Same per-page math as logits_hist_m, so its logits are bit-identical; at 36..48 rows it is slower than
// logits_hist_m (measured), which is why the host keeps both.
template <int WARPS, int HB>
__global__ __launch_bounds__(WARPS * 64) void logits_hist_s(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ wgt,
    const int* __restrict__ seqlens,
    const int* __restrict__ ptable,
    float* __restrict__ out,
    unsigned int* __restrict__ ghist,
    int max_pages,
    int out_stride,
    int rank_rows) {
  constexpr int NBIN = 1 << HB;
  constexpr int LOWB = 16 - HB;
  constexpr int GHS = NBIN + LG_CBINS;
  constexpr int NTH = WARPS * 64;
  constexpr int PERT = NBIN / NTH;  // GLOBAL bins per thread (coarse map)
  constexpr int CSH = HB - LG_CBITS;
  constexpr int GRP = NTH / LG_CBINS;
  static_assert(NBIN >= 2 && (NBIN % 2) == 0, "paired flush needs even bins");
  static_assert(NBIN % NTH == 0 && NTH >= LG_CBINS, "");
  static_assert(PERT <= LG_CBINS && (LG_CBINS % PERT) == 0, "a thread's contiguous run must sit inside one coarse bin");
  __shared__ unsigned int s_hist[NBIN];

  const int lane = threadIdx.x & 63;
  // Wave-uniform by construction; readfirstlane lets the compiler keep the page walk in SGPRs.
  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  int row = blockIdx.y;
  int seqlen;
  if (rank_rows) {
    // Longest rows first: blockIdx.y is a RANK, mapped to the row with that rank in descending length (ties by
    // index).  A bijection for any seqlens contents, so every row still gets exactly the same blocks and work; it
    // only changes which rows the hardware dispatches first when the grid needs more than one round of slots.
    // The host only sets rank_rows when gridDim.y <= 64.
    const int R = (int)gridDim.y;
    const int sl = lane < R ? seqlens[lane] : 0;
    int rank = 0;
    for (int j = 0; j < R; ++j) {
      const int sj = __builtin_amdgcn_readlane(sl, j);
      rank += (sj > sl || (sj == sl && j < lane)) ? 1 : 0;
    }
    const unsigned long long m = __ballot(lane < R && rank == (int)blockIdx.y);
    row = __builtin_amdgcn_readfirstlane(m ? (int)__ffsll(m) - 1 : (int)blockIdx.y);
    seqlen = __builtin_amdgcn_readlane(sl, row);
  } else {
    seqlen = seqlens[row];
  }
  // seqlen is device data, so no host check can bound it.  Without the clamp
  // an oversized row walks into the NEXT row's page table -- valid memory,
  // plausible page ids, confidently wrong logits.
  const int npages_raw = (seqlen + PAGE_TOK - 1) / PAGE_TOK;
  // Bound the walk by BOTH the page table and `out`.  The host only requires
  // that `out` cover whole pages -- a captured table is as wide as the graph
  // while the row is only as long as seqlens says -- so a table wider than
  // `out` is legal, and without the second term an oversized seqlen writes
  // past the end of the row.
  const int pages_out = out_stride / PAGE_TOK;
  const int npages_cap = max_pages < pages_out ? max_pages : pages_out;
  const int npages = npages_raw < npages_cap ? npages_raw : npages_cap;
  const int g = lane >> 4, c = lane & 15;
  // Nothing to rank when the row is shorter than topk: everything is selected,
  // so the histogram, its two barriers and the epilogue all compile out.
  //
  // This predicate MUST use the same clamped length the paired top-k kernel
  // applies (it clamps row_ends by the page-table width before its own
  // row_len <= TOPK early return).  A row binned here but skipped there is
  // never zeroed, and the histogram stays dirty for the life of the process.
  const int seqlen_eff = seqlen < npages_cap * PAGE_TOK ? seqlen : npages_cap * PAGE_TOK;
  const bool do_hist = (seqlen_eff > LG_TOPK);

  if (do_hist) {
    for (int i = threadIdx.x; i < NBIN; i += NTH)
      s_hist[i] = 0u;
  }

  const uint8_t* qp = q + (size_t)row * (size_t)(HEADS * HD) + (size_t)c * HD + g * 32;
  V8 qa0, qa1;
  qa0.h[0] = *(const i32x4*)(qp);
  qa0.h[1] = *(const i32x4*)(qp + 16);
  qa1.h[0] = *(const i32x4*)(qp + 16 * HD);
  qa1.h[1] = *(const i32x4*)(qp + 16 * HD + 16);
  const float* wp = wgt + (size_t)row * HEADS + g * 4;
  f32x4 w0 = *(const f32x4*)(wp);
  f32x4 w1 = *(const f32x4*)(wp + 16);

  const int nwaves = gridDim.x * WARPS;
  const int wid = blockIdx.x * WARPS + warp;
  const int koff = g * 512 + c * 16;
  // Pages of this wave: p = wid + it * nwaves, it in [0, cnt).  Uniform.
  const int cnt = wid < npages ? (npages - 1 - wid) / nwaves + 1 : 0;

  unsigned int* __restrict__ gh = ghist + (size_t)row * GHS;

  const int* __restrict__ pt = ptable + (size_t)row * max_pages;
  float* __restrict__ orow = out + (size_t)row * out_stride;

  // Page ids of the first 64 iterations, one per lane, read back with readlane (uniform lane index).
  const int pid = (lane < cnt) ? pt[wid + lane * nwaves] : 0;

  // K of one page through a per-page buffer resource: the page base lives in SGPRs and every lane offset is
  // koff + an immediate, so a page costs 9 loads and no per-lane address arithmetic.
  auto load_page = [&](V8(&b)[4], float& ks, int phys) {
    const __amdgpu_buffer_rsrc_t rs =
        __builtin_amdgcn_make_buffer_rsrc((void*)(kv + (size_t)phys * PAGE_BYTES), (short)0, PAGE_BYTES, 0x00020000);
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      b[tt].h[0] = __builtin_bit_cast(
          i32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, koff + (tt & 1) * 2048, (tt >> 1) * 4096, 0));
      b[tt].h[1] = __builtin_bit_cast(
          i32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, koff + (tt & 1) * 2048 + 256, (tt >> 1) * 4096, 0));
    }
    ks = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rs, lane * 4, K_BYTES, 0));
  };

  // One page: MFMAs, head-weighted reduction, logits store and histogram.  Operands, scale and summation order
  // are exactly the original kernel's, so the logits are bit-identical.
  auto compute = [&](const V8(&b)[4], const float ks, const int it) {
    float s[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      const f32x4 z = {0.f, 0.f, 0.f, 0.f};
      f32x4 a0 = mfma128(qa0.v, b[tt].v, z);
      f32x4 a1 = mfma128(qa1.v, b[tt].v, z);
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a0[i], 0.f), w0[i], acc);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a1[i], 0.f), w1[i], acc);
      s[tt] = acc;
    }
    // Cross-lane head reduction, one swap per level: permlane32_swap(x, y) yields [x_lo | y_lo] and
    // [x_hi | y_hi], whose sum is exactly the original "own + xor32 partner" of the selected operand (the upper
    // half adds the same two values in the other order; fp add is commutative, so the result is bit-identical).
    // permlane16_swap does the same on 16-lane rows.
    // NOTE: this LLVM folds fadd(swap[0], swap[1]) into swap[0] + swap[0] (miscompile seen on gfx950 / ROCm 7.2),
    // so the second result is laundered through an empty asm before the add.
    const i32x2 p02 =
        __builtin_amdgcn_permlane32_swap(__builtin_bit_cast(int, s[0]), __builtin_bit_cast(int, s[2]), false, false);
    const i32x2 p13 =
        __builtin_amdgcn_permlane32_swap(__builtin_bit_cast(int, s[1]), __builtin_bit_cast(int, s[3]), false, false);
    int h02 = p02[1], h13 = p13[1];
    asm volatile("" : "+v"(h02), "+v"(h13));
    const float A0 = __builtin_bit_cast(float, p02[0]) + __builtin_bit_cast(float, h02);
    const float A1 = __builtin_bit_cast(float, p13[0]) + __builtin_bit_cast(float, h13);
    const i32x2 pa =
        __builtin_amdgcn_permlane16_swap(__builtin_bit_cast(int, A0), __builtin_bit_cast(int, A1), false, false);
    int ha = pa[1];
    asm volatile("" : "+v"(ha));
    const float T = __builtin_bit_cast(float, pa[0]) + __builtin_bit_cast(float, ha);
    const int p = wid + it * nwaves;
    const bool live = p * PAGE_TOK + lane < seqlen;
    const float v = T * ks;
    (orow + (size_t)p * PAGE_TOK)[lane] = live ? v : -INFINITY;
    // Unconditional (adds 0 when not counted) so the page body stays one basic block.
    const uint32_t bb = order_bin_fast<LOWB>(v);
    atomicAdd(&s_hist[bb], (do_hist && live) ? 1u : 0u);
  };

  // Two K buffers, one page ahead: page it+1's loads are in flight under page it's MFMAs and reduction.
  V8 bA[4], bB[4];
  float ksA = 0.f, ksB = 0.f;
  // readfirstlane outside the branch: every path into the loop has then waited on pid, so the loop's readlanes
  // need no vmcnt(0) of their own.
  const int phys0 = __builtin_amdgcn_readfirstlane(pid);
  if (cnt > 0) load_page(bA, ksA, phys0);

  if (do_hist) __syncthreads();

  // The scheduler would otherwise sink the next page's loads below the current page's MFMAs (register pressure),
  // which serialises load -> use again.  An empty volatile asm that "rewrites" the current buffer is ordered after
  // the loads just issued, and the MFMAs depend on its outputs.
  auto fence = [&](V8(&b)[4], float& ks) {
#pragma unroll
    for (int tt = 0; tt < 4; ++tt)
      asm volatile("" : "+v"(b[tt].v));
    asm volatile("" : "+v"(ks));
  };

  // The host sizes the grid so that a wave has at most 64 pages; the generic tail below keeps longer walks exact.
  const int cnt_p = cnt < 64 ? cnt : 64;
  int it = 0;
  for (; it + 2 < cnt_p; it += 2) {
    load_page(bB, ksB, __builtin_amdgcn_readlane(pid, it + 1));
    fence(bA, ksA);
    compute(bA, ksA, it);
    load_page(bA, ksA, __builtin_amdgcn_readlane(pid, it + 2));
    fence(bB, ksB);
    compute(bB, ksB, it + 1);
  }
  if (it + 1 < cnt_p) {
    load_page(bB, ksB, __builtin_amdgcn_readlane(pid, it + 1));
    fence(bA, ksA);
    compute(bA, ksA, it);
    compute(bB, ksB, it + 1);
    it += 2;
  } else if (it < cnt_p) {
    compute(bA, ksA, it);
    it += 1;
  }
  for (; it < cnt; ++it) {
    load_page(bA, ksA, __builtin_amdgcn_readfirstlane(pt[wid + it * nwaves]));
    compute(bA, ksA, it);
  }

  if (do_hist) {
    __syncthreads();

    // ---- coarse summary, mapped over the GLOBAL bin index ----
    // Thread tx owns global bins [tx*PERT, tx*PERT+PERT), which lies inside
    // coarse bin (tx*PERT)>>CSH.  Needs no barrier of its own.
    const int g0 = (int)threadIdx.x * PERT;
    unsigned int acc = 0u;
#pragma unroll
    for (int j = 0; j < PERT; ++j)
      acc += s_hist[g0 + j];
#pragma unroll
    for (int o = GRP >> 1; o >= 1; o >>= 1)
      acc += __shfl_down(acc, o, 64);
    if ((threadIdx.x & (GRP - 1)) == 0 && acc) atomicAdd(&gh[NBIN + (g0 >> CSH)], acc);

    for (int i = threadIdx.x; i < NBIN / 2; i += NTH) {
      const unsigned long long vv = (unsigned long long)s_hist[2 * i] | ((unsigned long long)s_hist[2 * i + 1] << 32);
      if (vv) atomicAdd((unsigned long long*)(gh + 2 * i), vv);
    }
  }
}

// Sibling-group form of logits_hist_m.  A TARGET_VERIFY batch holds GR consecutive rows per request (its draft
// tokens), and those rows share one page table, so each K page is loaded ONCE and scored against all GR rows' queries
// instead of once per row.  The grid is persistent (about one block per CU) and is split over the groups in proportion
// to their page counts, so a long request no longer bounds the kernel with a fixed blocks-per-row share.  Inside a
// group a block owns a contiguous page range and its waves interleave over it.  The GR histograms hold the CU's LDS,
// so the block is the CU's whole occupancy: the queries live in LDS too, which keeps a wave at <= 128 VGPRs and lets
// the block run WARPS = 16 (4 waves per SIMD) instead of 8.
//
// Every row-page goes through exactly logits_hist_m's per-row body (same MFMA operands, fma chain, cross-lane
// reduction, scale, store and histogram key), so logits and histograms are bit-identical to it.  Whether siblings
// really share their pages is device data, so it is checked per wave: a group whose rows disagree on any page id (or
// a wave with more than 64 pages) takes the per-row general path, which is exact for any page tables.
template <int WARPS, int HB, int GR>
__global__ __launch_bounds__(WARPS * 64) void logits_hist_grp(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ wgt,
    const int* __restrict__ seqlens,
    const int* __restrict__ ptable,
    float* __restrict__ out,
    unsigned int* __restrict__ ghist,
    int rows,
    int max_pages,
    int out_stride) {
  constexpr int NBIN = 1 << HB;
  constexpr int LOWB = 16 - HB;
  constexpr int GHS = NBIN + LG_CBINS;
  constexpr int NTH = WARPS * 64;
  static_assert(NBIN >= 2 && (NBIN % 2) == 0, "paired flush needs even bins");
  static_assert(GR * HEADS <= NTH, "one thread per staged head weight");
  static_assert((GR * NBIN) % 4 == 0, "vectorised LDS clear");
  static_assert(NBIN % (2 * LG_CBINS) == 0, "a coarse bin is a whole number of fine pairs");
  static_assert(WARPS >= GR, "one wave per row flushes that row");
  __shared__ unsigned int s_hist[GR][NBIN];
  __shared__ float s_w[GR][HEADS];
  __shared__ i32x4 s_q[GR][4][64];

  const int lane = threadIdx.x & 63;
  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int g = lane >> 4, c = lane & 15;

  // Per-row lengths, row r on lane r (the host guarantees rows <= 64), with logits_hist_m's clamps.
  const int pages_out = out_stride / PAGE_TOK;
  const int npages_cap = max_pages < pages_out ? max_pages : pages_out;
  const int sl = lane < rows ? seqlens[lane] : 0;
  const int np_raw = (sl + PAGE_TOK - 1) / PAGE_TOK;
  const int np_c = np_raw < npages_cap ? np_raw : npages_cap;
  const int np = np_c > 0 ? np_c : 0;
  // The clamped length (logits_hist_m's seqlen_eff), which decides whether a row is binned.
  const int se_l = sl < npages_cap * PAGE_TOK ? sl : npages_cap * PAGE_TOK;

  // Partition (uniform; every wave computes the same integers): group gi has P_gi = max sibling page count and gets
  // blocks [gi + E*cum_gi/U, gi + 1 + E*(cum_gi + P_gi)/U), E = NB - ngroups, U = sum P.  Every group gets at least
  // one block and the ranges tile [0, NB) exactly.
  const int ngroups = rows / GR;
  int U = 0;
  for (int gi = 0; gi < ngroups; ++gi) {
    int Pg = 0;
#pragma unroll
    for (int k = 0; k < GR; ++k)
      Pg = max(Pg, __builtin_amdgcn_readlane(np, gi * GR + k));
    U += Pg;
  }
  if (U == 0) return;  // block-uniform, before any barrier

  const long long E = (long long)gridDim.x - ngroups;
  const int bx = (int)blockIdx.x;
  int r0 = 0, lead = 0, P = 0, bj = 0, B = 1;
  {
    long long cum = 0;
    for (int gi = 0; gi < ngroups; ++gi) {
      int Pg = 0, Lg = 0;
#pragma unroll
      for (int k = 0; k < GR; ++k) {
        const int v = __builtin_amdgcn_readlane(np, gi * GR + k);
        if (v > Pg) {
          Pg = v;
          Lg = k;
        }
      }
      const int s0 = gi + (int)((E * cum) / U);
      const int s1 = gi + 1 + (int)((E * (cum + Pg)) / U);
      if (bx >= s0 && bx < s1) {
        r0 = gi * GR;
        lead = Lg;
        P = Pg;
        bj = bx - s0;
        B = s1 - s0;
      }
      cum += Pg;
    }
  }
  r0 = __builtin_amdgcn_readfirstlane(r0);
  lead = __builtin_amdgcn_readfirstlane(lead);
  const int lo = __builtin_amdgcn_readfirstlane((int)(((long long)P * bj) / B));
  const int hi = __builtin_amdgcn_readfirstlane((int)(((long long)P * (bj + 1)) / B));
  const int span = hi - lo;
  // Pages of this wave: p = lo + warp + it * WARPS, it in [0, cnt).  Uniform.
  const int cnt = warp < span ? (span - 1 - warp) / WARPS + 1 : 0;

  int seq[GR], npk[GR];
  bool dh[GR];
  bool any_hist = false;
#pragma unroll
  for (int k = 0; k < GR; ++k) {
    seq[k] = __builtin_amdgcn_readlane(sl, r0 + k);
    npk[k] = __builtin_amdgcn_readlane(np, r0 + k);
    // The same clamped length the top-k kernel uses (see logits_hist_m).
    const int se = seq[k] < npages_cap * PAGE_TOK ? seq[k] : npages_cap * PAGE_TOK;
    dh[k] = se > LG_TOPK;
    any_hist = any_hist || dh[k];
  }

  // Page ids of the leader row (the one with the most pages), one per lane, for the first 64 iterations.
  const int cnt_p = cnt < 64 ? cnt : 64;
  const int pl = lo + warp + lane * WARPS;
  const int* __restrict__ ptl = ptable + (size_t)(r0 + lead) * max_pages;
  const int pid = (lane < cnt_p) ? ptl[pl] : 0;

  // The GR rows' queries go to LDS in MFMA lane order, s_q[k][part][lane] (parts: qa0 lo/hi, qa1 lo/hi), so a row's
  // operands are four conflict-free ds_read_b128 instead of 16 VGPRs held for the whole kernel.
  for (int i = (int)threadIdx.x; i < GR * 4 * 64; i += NTH) {
    const int k = i >> 8, part = (i >> 6) & 3, l = i & 63;
    const uint8_t* qp = q + (size_t)(r0 + k) * (size_t)(HEADS * HD) + (size_t)(l & 15) * HD + (l >> 4) * 32 +
                        (part & 1) * 16 + (part >> 1) * 16 * HD;
    s_q[k][part][l] = *(const i32x4*)qp;
  }

  // Sibling check: every row must name the leader's page for each of its own pages this wave walks.
  bool bad = false;
#pragma unroll
  for (int k = 0; k < GR; ++k) {
    const int v = (lane < cnt_p && pl < npk[k]) ? ptable[(size_t)(r0 + k) * max_pages + pl] : pid;
    bad = bad || (v != pid);
  }
  const bool fast = __ballot(bad) == 0ull;

  if (any_hist) {
    uint4* sh4 = reinterpret_cast<uint4*>(&s_hist[0][0]);
    for (int i = threadIdx.x; i < GR * NBIN / 4; i += NTH)
      sh4[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  if (threadIdx.x < GR * HEADS) (&s_w[0][0])[threadIdx.x] = wgt[(size_t)r0 * HEADS + threadIdx.x];

  const int koff = g * 512 + c * 16;

  auto load_page = [&](V8(&b)[4], float& ks, int phys) {
    const __amdgpu_buffer_rsrc_t rs =
        __builtin_amdgcn_make_buffer_rsrc((void*)(kv + (size_t)phys * PAGE_BYTES), (short)0, PAGE_BYTES, 0x00020000);
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      b[tt].h[0] = __builtin_bit_cast(
          i32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, koff + (tt & 1) * 2048, (tt >> 1) * 4096, 0));
      b[tt].h[1] = __builtin_bit_cast(
          i32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, koff + (tt & 1) * 2048 + 256, (tt >> 1) * 4096, 0));
    }
    ks = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rs, lane * 4, K_BYTES, 0));
  };

  // One row-page: logits_hist_m's compute body for row r0 + k, page p (see there for the reduction notes).
  auto compute_row = [&](const int k, const V8(&b)[4], const float ks, const int p) {
    V8 qa0k, qa1k;
    qa0k.h[0] = s_q[k][0][lane];
    qa0k.h[1] = s_q[k][1][lane];
    qa1k.h[0] = s_q[k][2][lane];
    qa1k.h[1] = s_q[k][3][lane];
    const f32x4 w0 = *(const f32x4*)(&s_w[k][g * 4]);
    const f32x4 w1 = *(const f32x4*)(&s_w[k][16 + g * 4]);
    float s[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      const f32x4 z = {0.f, 0.f, 0.f, 0.f};
      f32x4 a0 = mfma128(qa0k.v, b[tt].v, z);
      f32x4 a1 = mfma128(qa1k.v, b[tt].v, z);
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a0[i], 0.f), w0[i], acc);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a1[i], 0.f), w1[i], acc);
      s[tt] = acc;
    }
    const i32x2 p02 =
        __builtin_amdgcn_permlane32_swap(__builtin_bit_cast(int, s[0]), __builtin_bit_cast(int, s[2]), false, false);
    const i32x2 p13 =
        __builtin_amdgcn_permlane32_swap(__builtin_bit_cast(int, s[1]), __builtin_bit_cast(int, s[3]), false, false);
    int h02 = p02[1], h13 = p13[1];
    asm volatile("" : "+v"(h02), "+v"(h13));
    const float A0 = __builtin_bit_cast(float, p02[0]) + __builtin_bit_cast(float, h02);
    const float A1 = __builtin_bit_cast(float, p13[0]) + __builtin_bit_cast(float, h13);
    const i32x2 pa =
        __builtin_amdgcn_permlane16_swap(__builtin_bit_cast(int, A0), __builtin_bit_cast(int, A1), false, false);
    int ha = pa[1];
    asm volatile("" : "+v"(ha));
    const float T = __builtin_bit_cast(float, pa[0]) + __builtin_bit_cast(float, ha);
    const bool live = p * PAGE_TOK + lane < seq[k];
    const float v = T * ks;
    // Through a per-row buffer resource: the row and page offsets stay scalar, so the store needs no 64-bit
    // per-lane address per row (those would not fit the 128-VGPR budget of the 16-wave block).
    const __amdgpu_buffer_rsrc_t ors = __builtin_amdgcn_make_buffer_rsrc(
        (void*)(out + (size_t)(r0 + k) * out_stride), (short)0, out_stride * (int)sizeof(float), 0x00020000);
    __builtin_amdgcn_raw_buffer_store_b32(
        __builtin_bit_cast(int, live ? v : -INFINITY), ors, lane * 4, p * PAGE_TOK * (int)sizeof(float), 0);
    const uint32_t bb = order_bin_fast<LOWB>(v);
    atomicAdd(&s_hist[k][bb], (dh[k] && live) ? 1u : 0u);
  };

  // All GR rows against one loaded page; rows shorter than the leader skip their missing tail pages.
  auto compute_grp = [&](const V8(&b)[4], const float ks, const int it) {
    const int p = lo + warp + it * WARPS;
#pragma unroll
    for (int k = 0; k < GR; ++k)
      if (p < npk[k]) compute_row(k, b, ks, p);
  };

  auto fence = [&](V8(&b)[4], float& ks) {
#pragma unroll
    for (int tt = 0; tt < 4; ++tt)
      asm volatile("" : "+v"(b[tt].v));
    asm volatile("" : "+v"(ks));
  };

  V8 bA[4], bB[4];
  float ksA = 0.f, ksB = 0.f;
  // Issued before the sibling verdict is known (it is the leader's page either way); the general path discards it.
  const int phys0 = __builtin_amdgcn_readfirstlane(pid);
  if (cnt_p > 0) load_page(bA, ksA, phys0);

  __syncthreads();  // s_hist clear and s_w

  int it = 0;
  if (fast) {
    for (; it + 2 < cnt_p; it += 2) {
      load_page(bB, ksB, __builtin_amdgcn_readlane(pid, it + 1));
      fence(bA, ksA);
      compute_grp(bA, ksA, it);
      load_page(bA, ksA, __builtin_amdgcn_readlane(pid, it + 2));
      fence(bB, ksB);
      compute_grp(bB, ksB, it + 1);
    }
    if (it + 1 < cnt_p) {
      load_page(bB, ksB, __builtin_amdgcn_readlane(pid, it + 1));
      fence(bA, ksA);
      compute_grp(bA, ksA, it);
      compute_grp(bB, ksB, it + 1);
      it += 2;
    } else if (it < cnt_p) {
      compute_grp(bA, ksA, it);
      it += 1;
    }
  }
  // General path: every row reads its own page id (a group whose siblings disagree, or pages past the 64th).
  for (; it < cnt; ++it) {
    const int p = lo + warp + it * WARPS;
#pragma unroll
    for (int k = 0; k < GR; ++k) {
      if (p < npk[k]) {
        load_page(bA, ksA, __builtin_amdgcn_readfirstlane(ptable[(size_t)(r0 + k) * max_pages + p]));
        compute_row(k, bA, ksA, p);
      }
    }
  }

  // logits_hist_m's flush, once per binned row of the group: the complete coarse flush, then the fine flush from the
  // block's gate up (see lg_gate).  Every block of the group adds into the same rows, so the ungated fine flush was
  // mostly same-address atomic traffic on bins nobody reads.  After the one barrier that completes s_hist, wave k
  // flushes row k on its own (lane = coarse bin), so the block needs no further barrier.
  if (any_hist) {
    __syncthreads();
    if (warp < GR) {
      const int k = warp;  // wave-uniform
      const int se = __builtin_amdgcn_readlane(se_l, r0 + k);
      if (se > LG_TOPK) {
        unsigned int* __restrict__ gh = ghist + (size_t)(r0 + k) * GHS;
        // Coarse bin `lane` = fine bins [64 lane, 64 lane + 64): 16 uint4 reads, each lane starting at chunk lane % 16
        // so that 16 consecutive lanes hit 16 different bank quads.
        static_assert(NBIN == 64 * LG_CBINS && LG_CBINS == 64, "one lane per coarse bin");
        const uint4* src = reinterpret_cast<const uint4*>(&s_hist[k][lane * (NBIN / LG_CBINS)]);
        unsigned int acc = 0u;
#pragma unroll
        for (int j = 0; j < NBIN / LG_CBINS / 4; ++j) {
          const uint4 v = src[(j + lane) & (NBIN / LG_CBINS / 4 - 1)];
          acc += (v.x + v.y) + (v.z + v.w);
        }
        if (acc) atomicAdd(&gh[NBIN + lane], acc);
        for (int i = lg_gate(acc, lane) * (NBIN / LG_CBINS / 2) + lane; i < NBIN / 2; i += 64) {
          const unsigned long long vv =
              (unsigned long long)s_hist[k][2 * i] | ((unsigned long long)s_hist[k][2 * i + 1] << 32);
          if (vv) atomicAdd((unsigned long long*)(gh + 2 * i), vv);
        }
      }
    }
  }
}

// CUs of the current device (per-device constant, cached); 0 if unknown.
inline int device_cus() {
  int dev = 0;
  static constexpr int kMaxDev = 16;
  static int s_cu[kMaxDev] = {0};
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) {
    return 0;
  }
  if (s_cu[dev] == 0) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
      return 0;
    }
    s_cu[dev] = prop.multiProcessorCount;
  }
  return s_cu[dev];
}

// Resident blocks of `fn` on the current device (per-device constant, cached); 0 if unknown.
inline long long resident_slots(const void* fn, int threads) {
  int dev = 0;
  static constexpr int kMaxDev = 16;
  static long long s_slots[kMaxDev] = {0};
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) {
    return 0;
  }
  if (s_slots[dev] == 0) {
    hipDeviceProp_t prop;
    int per_cu = 0;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, fn, threads, 0) != hipSuccess) {
      return 0;
    }
    s_slots[dev] = (long long)per_cu * (long long)prop.multiProcessorCount;
  }
  return s_slots[dev];
}

/// The fp8 dtype the MFMA reads as raw OCP e4m3 bytes.  Checked by hand rather
/// than through ``with_dtype<fp8_e4m3_t>``: on ROCm ``sglang::fp8_e4m3_t`` is
/// ``uint8_t``, which ``DLDataTypeTrait`` maps to kDLUInt -- that would reject a
/// genuine fp8 tensor and accept a uint8 one, exactly backwards.
inline auto is_fp8_e4m3fn(DLDataType dtype) -> bool {
  return dtype.lanes == 1 && dtype.bits == 8 && dtype.code == kDLFloat8_e4m3fn;
}

}  // namespace dsa_gfx950::logits

struct PagedMqaLogitsKernel {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView seqlens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView ghist,
      int64_t blocks_per_row,
      int64_t variant) {
    using namespace host;
    namespace impl = dsa_gfx950::logits;

    // The launch below is logits_hist_m<8, HOST_HB>, so the histogram row
    // stride the kernel writes is fixed here too.  topk_transform.cuh checks
    // the same width from its side; without this one a mismatch overruns into
    // the next row's counters.
    constexpr int HOST_HB = 12;
    constexpr int64_t HOST_GHS = (int64_t{1} << HOST_HB) + LG_CBINS;

    auto rows_ = SymbolicSize{"rows"};
    auto pages_ = SymbolicSize{"table_pages"};
    auto cols_ = SymbolicSize{"logits_cols"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    // Rebinding one symbolic dtype across q and kv is the "kv must have the
    // same fp8 dtype as q" check; the options stay open because e4m3fn has no
    // DLDataTypeTrait on ROCm.
    auto fp8_ = SymbolicDType{};

    // No with_strides means TensorMatcher demands full contiguity, which is
    // what each of these wants.
    TensorMatcher({rows_, int64_t{HEADS}, int64_t{HD}}).with_dtype(fp8_).with_device(device_).verify(q);
    // PAGE_TOK and TOK_STRIDE are compiled in; a cache laid out differently
    // would be read as garbage rather than refused, hence the row width here.
    TensorMatcher({-1, int64_t{PAGE_TOK} * TOK_STRIDE}).with_dtype(fp8_).with_device(device_).verify(kv);
    RuntimeCheck(
        impl::is_fp8_e4m3fn(fp8_.unwrap()), "q and the kv cache must be fp8 e4m3fn; they are read as raw bytes");

    TensorMatcher({rows_, int64_t{HEADS}}).with_dtype<fp32_t>().with_device(device_).verify(weights);
    TensorMatcher({rows_, pages_}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({rows_, HOST_GHS})
        .with_dtype<int32_t>()
        .with_device(device_)
        .with_strides({HOST_GHS, 1})
        .verify(ghist);
    // `out` is a column slice of the workspace, so only the row is packed.
    TensorMatcher({rows_, cols_}).with_dtype<fp32_t>().with_device(device_).with_strides({-1, 1}).verify(out);

    // One entry per row, but the caller may hand over a longer buffer.
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_).verify(seqlens);
    RuntimeCheck(
        seqlens.size(0) >= rows_.unwrap(),
        "seqlens needs one entry per row: got ",
        seqlens.size(0),
        " for ",
        rows_.unwrap(),
        " rows");

    // A captured page table is as wide as the graph while the row is as long
    // as seqlens says, so it may exceed `out`; what must hold is that `out`
    // covers whole pages.
    RuntimeCheck(cols_.unwrap() % PAGE_TOK == 0, "out row must be a whole number of pages, got ", cols_.unwrap());

    // Rank rows by length (longest dispatched first) only when the grid needs more than one round of resident
    // slots: with a single round the order is moot and the ranking would only add a round trip.  rows and
    // blocks_per_row are host constants and the slot count is a per-device constant, so this is graph-safe.
    const int rows_i = static_cast<int>(rows_.unwrap());

    // variant: 0 = logits_hist_m, 1 = logits_hist_s (both blocks_per_row blocks per row), 2 = logits_hist_grp
    // (blocks_per_row unused).
    RuntimeCheck(variant >= 0 && variant <= 2, "logits variant must be 0, 1 or 2, got ", variant);
    RuntimeCheck(variant == 2 || blocks_per_row > 0, "blocks_per_row must be positive, got ", blocks_per_row);
    // variant 2 is the sibling-group kernel: rows come in groups of HOST_GR consecutive rows that
    // (normally) share a page table.  Its grid is about one block per CU (it holds HOST_GR histograms in LDS, so one
    // block fits per CU anyway), never fewer blocks than groups.  rows and the CU count are host constants, so this
    // is graph-safe.
    constexpr int HOST_GR = 6;
    // Waves per sibling-group block: one block per CU (LDS), so this is the CU's whole occupancy.
    constexpr int HOST_GW = 16;
    if (variant == 2) {
      RuntimeCheck(
          rows_i % HOST_GR == 0 && rows_i >= HOST_GR && rows_i <= 64,
          "the sibling-group logits kernel needs rows to be a multiple of ",
          HOST_GR,
          " in [",
          HOST_GR,
          ", 64], got ",
          rows_i);
      int nb = impl::device_cus();
      if (nb <= 0) nb = 256;
      if (nb < rows_i / HOST_GR) nb = rows_i / HOST_GR;
      LaunchKernel(dim3(nb), dim3(HOST_GW * 64), device_.unwrap())(
          impl::logits_hist_grp<HOST_GW, HOST_HB, HOST_GR>,
          reinterpret_cast<const uint8_t*>(q.data_ptr()),
          reinterpret_cast<const uint8_t*>(kv.data_ptr()),
          static_cast<fp32_t*>(weights.data_ptr()),
          static_cast<int*>(seqlens.data_ptr()),
          static_cast<int*>(page_table.data_ptr()),
          static_cast<fp32_t*>(out.data_ptr()),
          reinterpret_cast<unsigned int*>(ghist.data_ptr()),
          rows_i,
          static_cast<int>(pages_.unwrap()),
          static_cast<int>(out.stride(0)));
      return;
    }

    if (variant == 0) {
      LaunchKernel(dim3(static_cast<int>(blocks_per_row), rows_i), dim3(8 * 64), device_.unwrap())(
          impl::logits_hist_m<8, HOST_HB>,
          reinterpret_cast<const uint8_t*>(q.data_ptr()),
          reinterpret_cast<const uint8_t*>(kv.data_ptr()),
          static_cast<fp32_t*>(weights.data_ptr()),
          static_cast<int*>(seqlens.data_ptr()),
          static_cast<int*>(page_table.data_ptr()),
          static_cast<fp32_t*>(out.data_ptr()),
          reinterpret_cast<unsigned int*>(ghist.data_ptr()),
          static_cast<int>(pages_.unwrap()),
          static_cast<int>(out.stride(0)));
      return;
    }

    const long long slots =
        impl::resident_slots(reinterpret_cast<const void*>(impl::logits_hist_s<8, HOST_HB>), 8 * 64);
    const int rank_rows = (rows_i <= 64 && blocks_per_row * rows_i > slots) ? 1 : 0;

    LaunchKernel(dim3(static_cast<int>(blocks_per_row), rows_i), dim3(8 * 64), device_.unwrap())(
        impl::logits_hist_s<8, HOST_HB>,
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(kv.data_ptr()),
        static_cast<fp32_t*>(weights.data_ptr()),
        static_cast<int*>(seqlens.data_ptr()),
        static_cast<int*>(page_table.data_ptr()),
        static_cast<fp32_t*>(out.data_ptr()),
        reinterpret_cast<unsigned int*>(ghist.data_ptr()),
        static_cast<int>(pages_.unwrap()),
        static_cast<int>(out.stride(0)),
        rank_rows);
  }
};

}  // namespace sglang

// PAGE_TOK / HD / TOK_STRIDE are generic enough to collide with another header
// compiled into the same translation unit; they exist only for the kernel above.
#undef PAGE_TOK
#undef HD
#undef HEADS
#undef LG_TOPK
#undef TOK_STRIDE
#undef PAGE_BYTES
#undef K_BYTES
#undef LG_CBITS
#undef LG_CBINS
