/// GLM-5.2 DSA indexer, the logits kernel: paged MQA FP8 logits plus the fine
/// histogram and coarse summary topk_transform.cuh selects from.  One kernel
/// ships, logits_hist_m<8, 12>; WARPS and HB are template arguments.

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

// ---- logits_hist_slot: the small-batch variant (rows <= kSlotMaxRows) ----
//
// MTP verify launches 6 rows per request whose page tables are the same pages,
// so logits_hist_m loads every KV page once per row.  logits_hist_slot takes
// the rows in slots of kSlotRows, splits a slot's pages between all the blocks
// of the slot, and runs every row of the slot against each page it loads, so a
// shared page is read once per slot instead of once per row.  The per-logit math
// is logits_hist_m's, so `out` and `ghist` are bit-identical.
//
// PagedMqaLogitsKernel::run launches it only when the caller passes
// share_pages, which the host sets for long MTP verify contexts, where it is
// faster; every other launch is logits_hist_m.
constexpr int kSlotRows = 3;      // rows per slot
constexpr int kSlotMaxRows = 18;  // larger grids keep logits_hist_m
constexpr int kSlotPrefetch = 2;  // KV pages in flight per wave
constexpr int kKvLoadAux = 3;     // buffer-load cache policy for KV (glc|slc: streamed)
constexpr int kOutStoreAux = 18;  // buffer-store cache policy for the logits row
constexpr int kSkewLo = 8;        // pair-skew pages (x/64 of a queue) for <= 6 rows
constexpr int kSkew = 8;          // ... for <= 12 rows
constexpr int kSkewHi = 6;        // ... for <= 18 rows

template <int LOWB>
__device__ __forceinline__ uint32_t order_bin_fast_nofma(float x) {
  // Opaque copy: keeps the backend from fusing a producer fmul into the
  // f32->f16 convert (v_fma_mix rounds once; the key must round twice).
  asm("" : "+v"(x));
  const uint32_t h = (uint32_t)__half_as_ushort(__float2half_rn(x));
  const uint32_t m = (uint32_t)((int32_t)__float_as_int(x) >> 31);
  return (h >> LOWB) ^ (0x8000u >> LOWB) ^ (m & (0x7fffu >> LOWB));
}

template <int WARPS, int HB>
__global__ __launch_bounds__(WARPS * 64) __attribute__((amdgpu_waves_per_eu(2, 2))) void logits_hist_slot(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ wgt,
    const int* __restrict__ seqlens,
    const int* __restrict__ ptable,
    float* __restrict__ out,
    unsigned int* __restrict__ ghist,
    int max_pages,
    int out_stride,
    int wide_ctx,
    int narrow_bpr) {
  constexpr int NBIN = 1 << HB;
  constexpr int LOWB = 16 - HB;
  constexpr int GHS = NBIN + LG_CBINS;
  constexpr int NTH = WARPS * 64;
  constexpr int PERT = NBIN / NTH;  // GLOBAL bins per thread (coarse map)
  constexpr int CSH = HB - LG_CBITS;
  constexpr int GRP = NTH / LG_CBINS;
  constexpr int G = kSlotRows;  // rows per slot: the most rows that share one page stream
  static_assert(NBIN >= 2 && (NBIN % 2) == 0, "paired flush needs even bins");
  static_assert(NBIN % NTH == 0 && NTH >= LG_CBINS, "");
  static_assert(PERT <= LG_CBINS && (LG_CBINS % PERT) == 0, "a thread's contiguous run must sit inside one coarse bin");
  static_assert(G >= 1 && G <= 32, "");
  __shared__ __attribute__((aligned(16))) unsigned int s_hist[G][NBIN];
  __shared__ i32x4 s_q[G][4][64];
  __shared__ f32x4 s_w[G][8];
  __shared__ unsigned int s_brk[WARPS];
  // Block-shared dynamic page queue: block page index i (i = k*WARPS + warp in
  // the static split) -> physical page id, and one pull counter per run start.
  __shared__ int s_pt[64 * WARPS];
  __shared__ unsigned int s_ctr[G];

  // Mid contexts: fewer blocks per row, so each wave streams more pages.  A
  // uniform branch: every block reads the same <= kSlotMaxRows seqlens.
  int slot_bpr;
  {
    int longest = 0;
    for (int r = 0; r < (int)gridDim.y; ++r) {
      const int sl = seqlens[r];
      longest = sl > longest ? sl : longest;
    }
    slot_bpr = longest < wide_ctx ? narrow_bpr : (int)gridDim.x;
  }
  if ((int)blockIdx.x >= slot_bpr) return;

  // ---- cross-row page sharing --------------------------------------------
  // Rows are taken in aligned slots of G rows.  All blocks of a slot (grid
  // rows slot0..slot_end-1, blockIdx.x 0..bpr-1) split the slot's pages
  // between them as S*bpr "virtual blocks", and every block runs the math of
  // EVERY row of the slot against the pages it owns -- so a page shared by
  // the slot's rows is loaded once.  Rows a and a+1 share a stream only if,
  // over this wave's pages, they walk the same page set and read the same
  // physical page ids (decided at runtime from page_table and seqlens; any
  // mismatch splits the slot into separately streamed runs).  Each row keeps
  // its own seqlen mask, out row and histogram, and its per-logit math is
  // exactly the unshared kernel's.
  const int rows = gridDim.y;
  // Dispatch-order remap (a bijection on the grid).  Rows come in groups of
  // 2G (two G-row slots, which read the same pages when the group is one
  // request).  Block (lr, bx) of the first slot and block (lr, bx) of the
  // second form a pair; pairs are numbered P and laid out in chunks of 16
  // linear block ids: 8 first-slot members, then their 8 partners.  Linear
  // ids 8 apart sit on the same XCD and are dispatched back to back, so the
  // partner's page reads hit L2 despite the nt K loads -- for ANY bpr (only
  // the last partial chunk's pairs are split across XCDs).  Rows past the
  // last full group keep the identity mapping.
  int bx = blockIdx.x, row = blockIdx.y;
  {
    const int bpr = slot_bpr;
    const int L = row * bpr + bx;
    const int gsz = G * bpr;                // pairs per group
    const int NP = (rows / (2 * G)) * gsz;  // pairs in full groups
    if (L < 2 * NP) {
      const int c16 = L & ~15;
      const int t = (2 * NP - c16) < 16 ? (2 * NP - c16) : 16;
      const int w = L & 15;
      // Last partial chunk (t < 16 blocks): pair members w and w + 8 (same
      // XCD) for w < t - 8; the rest pair up consecutively.
      int h, P;
      {
        const int n1 = t > 8 ? t - 8 : 0;
        const int hl = t == 16 ? 8 : n1;
        if (w < hl) {
          h = 0;
          P = (c16 >> 1) + w;
        } else if (w >= 8) {
          h = 1;
          P = (c16 >> 1) + w - 8;
        } else {
          const int u = w - n1;
          h = u & 1;
          P = (c16 >> 1) + n1 + (u >> 1);
        }
      }
      // Pair P runs on XCD P % 8, and XCD x owns the contiguous pair range
      // [x*NPf/8, (x+1)*NPf/8) (page-slice-major: one request's groups land on
      // few XCDs).  Pairs are numbered by virtual block (rem = vb = bx*G + lr),
      // so contiguous pairs read adjacent 8-page chunks.
      int Q = P;
      // The partial chunk's same-XCD pairs are folded in: XCD x < n1 owns
      // NJ + 1 contiguous pairs, the rest NJ; the cross-XCD tail pairs keep
      // Q = P.
      {
        const int NPf = NP & ~7, NJ = NPf >> 3;
        const int n1 = (NP & 7) > 4 ? 2 * (NP & 7) - 8 : 0;
        const int x = P & 7;
        if (P < NPf + n1) Q = x * NJ + (x < n1 ? x : n1) + (P < NPf ? (P >> 3) : NJ);
      }
      const int gi = Q / gsz;
      const int rem = Q - gi * gsz;
      bx = rem / G;
      const int lr = rem - bx * G;
      row = gi * 2 * G + h * G + lr;
    }
  }
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int g = lane >> 4, c = lane & 15;
  const int slot0 = (row / G) * G;
  const int slot_end = (slot0 + G) < rows ? (slot0 + G) : rows;
  const int S = slot_end - slot0;
  // seqlen is device data, so no host check can bound it.  Without the clamp
  // an oversized row walks into the NEXT row's page table.  Bound the walk by
  // BOTH the page table and `out` (a table wider than `out` is legal).
  const int pages_out = out_stride / PAGE_TOK;
  const int npages_cap = max_pages < pages_out ? max_pages : pages_out;

  // Wave-uniform wave id: lets the page index, the page-table read and the
  // KV page base live in SGPRs (s_load + SGPR-based global loads).
  const int warp_u = __builtin_amdgcn_readfirstlane(warp);
  const int nwaves = slot_bpr * S * WARPS;
  const int wid = (bx * S + (row - slot0)) * WARPS + warp_u;
  const int koff = g * 512 + c * 16;

  auto npages_of = [&](int sl) __attribute__((always_inline)) {
    const int np = (sl + PAGE_TOK - 1) / PAGE_TOK;
    return np < npages_cap ? np : npages_cap;
  };
  auto nk_of = [&](int np) __attribute__((always_inline)) { return wid < np ? (np - wid + nwaves - 1) / nwaves : 0; };
  // Nothing to rank when the row is shorter than topk: everything is selected.
  // This predicate MUST use the same clamped length the paired top-k kernel
  // applies (it clamps row_ends by the page-table width before its own
  // row_len <= TOPK early return).
  auto dohist_of = [&](int sl) __attribute__((always_inline)) {
    const int sle = sl < npages_cap * PAGE_TOK ? sl : npages_cap * PAGE_TOK;
    return sle > LG_TOPK;
  };

  // Page-table prefetch: lane i holds pt[p_{k0+i}] for a chunk of 64 pages of
  // this wave (predicated, never reads past npages); broadcast with readlane.
  int ptv = 0;

  // Always called with 0 <= k < cnt, so the page id is in range.
  auto issue_p = [&](V8(&bb)[4], float& ksv, int phys, bool real) __attribute__((always_inline)) {
    // Wave-uniform page base in an SGPR V#: the per-lane offsets are the
    // loop-invariant koff / lane*4, so no per-page 64-bit VALU address math.
    // A ring slot past the wave's last page still issues (the vmcnt
    // bookkeeping stays a compile-time constant) but with num_records = 0:
    // every lane is out of range, so no memory request leaves the CU.
    const int nrec = real ? PAGE_BYTES : 0;
    const __amdgpu_buffer_rsrc_t rs =
        __builtin_amdgcn_make_buffer_rsrc((void*)(kv + (size_t)phys * PAGE_BYTES), (short)0, nrec, 0x00020000);
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      bb[tt].h[0] = __builtin_amdgcn_raw_buffer_load_b128(rs, koff + (tt & 1) * 2048, (tt >> 1) * 4096, kKvLoadAux);
      bb[tt].h[1] =
          __builtin_amdgcn_raw_buffer_load_b128(rs, koff + (tt & 1) * 2048 + 256, (tt >> 1) * 4096, kKvLoadAux);
    }
    ksv = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rs, lane * 4, K_BYTES, kKvLoadAux));
  };
  auto issue = [&](V8(&bb)[4], float& ksv, int k, bool real)
                   __attribute__((always_inline)) { issue_p(bb, ksv, __builtin_amdgcn_readlane(ptv, k), real); };

  // ---- prologue: every global read the block needs is issued up front ----
  // All prologue loads are unconditional (row/page indices are clamped into
  // valid memory, the value is then masked).  Order: the slot rows' seqlens
  // (scalar), the first page-table chunk of every slot row, q and w -- all
  // independent, so they are in flight together -- then the first pages of
  // the first run, and only THEN the LDS staging / histogram zeroing / link
  // check, which therefore run under the page loads' latency.
  int slr[G];
#pragma unroll
  for (int j = 0; j < G; ++j) {
    const int jc = j < S ? j : S - 1;
    slr[j] = seqlens[slot0 + jc];
  }
  int ptr_[G];
  const int pidx = wid + lane * nwaves;
  const int pidx_c = pidx < max_pages ? pidx : max_pages - 1;
#pragma unroll
  for (int j = 0; j < G; ++j) {
    const int jc = j < S ? j : S - 1;
    ptr_[j] = ptable[(size_t)(slot0 + jc) * max_pages + pidx_c];
  }
  // Pair skew: virtual blocks (2u, 2u+1) of the slot pair up.  The early
  // member (odd) leaves its queue indices [2*WARPS, 2*WARPS + k) to the late
  // member (even), whose pages are exactly the early member's pages shifted by
  // WARPS: so the late member loads the early member's page ids now (address
  // independent of seqlens), and decides after the seqlens arrive which of
  // them it takes.
  const int sk_vb = wid / WARPS;
  const int sk_baseL = (sk_vb & ~1) * WARPS;
  const bool sk_isL = (sk_vb & 1) == 0;
  const bool sk_pair = sk_baseL + WARPS < nwaves;
  int ptE_[G];
#pragma unroll
  for (int j = 0; j < G; ++j)
    ptE_[j] = 0;
  {  // unconditional (clamped, in-bounds): no branch for a waitcnt to land in
    const int pe = pidx + WARPS;
    const int pe_c = pe < max_pages ? pe : max_pages - 1;
#pragma unroll
    for (int j = 0; j < G; ++j) {
      const int jc = j < S ? j : S - 1;
      ptE_[j] = ptable[(size_t)(slot0 + jc) * max_pages + pe_c];
    }
  }
  // q fragments + w of every slot row (staged to LDS below).  Slots past the
  // last row replay row S-1, so an out-of-range thread can store its (equal)
  // value onto slot G-1 instead of branching around the store.
  constexpr int QPT = (G * 256 + NTH - 1) / NTH;
  i32x4 qreg[QPT];
#pragma unroll
  for (int t = 0; t < QPT; ++t) {
    const int i = threadIdx.x + t * NTH;
    const int j = i >> 8, h = (i >> 6) & 3, l = i & 63;
    const int jc = j < S ? j : S - 1;
    const int gg = l >> 4, cc = l & 15;
    const uint8_t* qp =
        q + (size_t)(slot0 + jc) * (size_t)(HEADS * HD) + (size_t)cc * HD + gg * 32 + (h >> 1) * 16 * HD + (h & 1) * 16;
    qreg[t] = *(const i32x4*)qp;
  }
  const int wj = (int)(threadIdx.x >> 3) < S ? (int)(threadIdx.x >> 3) : S - 1;
  const f32x4 wreg = *(const f32x4*)(wgt + (size_t)(slot0 + wj) * HEADS + (threadIdx.x & 7) * 4);
  V8 buf[kSlotPrefetch][4];
  float ksb[kSlotPrefetch];
  // Speculative first KV pages: the first run always starts at slot0 and its
  // first kSlotPrefetch pages are wid + d*nwaves, whose ids sit in lanes d of ptr_[0].
  // Issued as soon as that page-table load returns -- while q/w are still in
  // flight and before the LDS staging -- so the KV latency overlaps the rest
  // of the prologue.  Liveness test wid + d*nwaves < np0 == d < nk0 (no
  // division); a dead slot issues with num_records = 0 (no memory request).
  ptv = ptr_[0];
  {
    const int np0 = npages_of(slr[0]);
#pragma unroll
    for (int d = 0; d < 1; ++d) {
      const bool live = wid + d * nwaves < np0;
      issue(buf[d], ksb[d], live ? d : 0, live);
    }
  }

  // ---- pair skew parameters (wave-uniform) ----
  int sk_add = 0;      // own queue length delta: +k (late member) / 0 (early)
  int sk_d = 0;        // late: queue index i >= NBo is the early member's index i + sk_d
  int sk_skip = 0;     // early: queue indices [2*WARPS, 2*WARPS + k) belong to the late member
  bool sk_own = true;  // this thread's s_pt slot holds its own page id
  bool sk_in = false;  // this thread's early-member id goes to s_pt[nbL + i - 2*WARPS]
  int sk_slot = 0;
  {
    const int np0 = npages_of(slr[0]);
    bool npeq = true;
#pragma unroll
    for (int j = 1; j < G; ++j)
      npeq &= (j >= S) || (npages_of(slr[j]) == np0);
    // nbof(baseL) and nbof(baseL + WARPS) from ONE division: with
    // R = np0 - baseL = q*nwaves + r, R - WARPS = q*nwaves + (r - WARPS) when
    // r >= WARPS, else (q-1)*nwaves + (r - WARPS + nwaves) whose min(., WARPS)
    // is WARPS (nwaves is a multiple of WARPS and > WARPS when sk_pair).
    const int R = np0 - sk_baseL;
    const unsigned q = (unsigned)R / (unsigned)nwaves;
    const int r = R - (int)q * nwaves;
    const int nbL = R <= 0 ? 0 : (int)q * WARPS + (r < WARPS ? r : WARPS);
    const int rE = r - WARPS;
    const int nbE = R <= WARPS ? 0 : (rE >= 0 ? (int)q * WARPS + (rE < WARPS ? rE : WARPS) : (int)q * WARPS);
    const int k = (nbE * (rows <= 6 ? kSkewLo : rows <= 12 ? kSkew : kSkewHi)) >> 6;
    const bool ok = npeq && sk_pair && k > 0 && nbE >= 2 * WARPS + k && nbL + k <= 64 * WARPS;
    if (ok) {
      if (sk_isL) {
        sk_add = k;
        sk_d = 2 * WARPS - nbL;
      } else {
        sk_skip = k;
      }
    }
    const int i = lane * WARPS + warp_u;
    sk_own = !(ok && sk_isL) || i < nbL;
    sk_in = ok && sk_isL && i >= 2 * WARPS && i < 2 * WARPS + k;
    sk_slot = nbL + i - 2 * WARPS;
  }
  sk_add = __builtin_amdgcn_readfirstlane(sk_add);
  sk_d = __builtin_amdgcn_readfirstlane(sk_d);
  sk_skip = __builtin_amdgcn_readfirstlane(sk_skip);
  int nkr[G];
#pragma unroll
  for (int j = 0; j < G; ++j) {
    // Branch-free nk_of(): identical value, but no control flow, so the
    // compiler cannot sink the seqlens loads into a serial chain.
    const int np = npages_of(slr[j]);
    const int rem = np - wid > 0 ? np - wid : 0;
    const int nk = (int)(((unsigned)rem + (unsigned)nwaves - 1u) / (unsigned)nwaves);
    nkr[j] = j < S ? nk : 0;
  }
#pragma unroll
  for (int t = 0; t < QPT; ++t) {
    const int i = threadIdx.x + t * NTH;
    const int ii = i < G * 256 ? i : (G - 1) * 256 + (i & 255);
    (&s_q[0][0][0])[ii] = qreg[t];
  }
  {
    const int wi = (int)threadIdx.x < G * 8 ? (int)threadIdx.x : (G - 1) * 8 + (int)(threadIdx.x & 7);
    (&s_w[0][0])[wi] = wreg;
  }
  // Everything the prologue loaded is consumed: say so explicitly, so the
  // page ring below starts from an empty vmcnt queue (otherwise the waitcnt
  // pass merges the store branch's pending loads into the ring's loop state
  // and falls back to vmcnt(0) inside the main loop).
  // Only the kSlotPrefetch ring pages (9 loads each) may still be in flight.
  __builtin_amdgcn_s_waitcnt(0x0F79);  // vmcnt(9)

  bool any_hist = false;
#pragma unroll
  for (int j = 0; j < G; ++j)
    any_hist |= (j < S) && dohist_of(slr[j]);
  if (any_hist) {
    // 16-byte stores: S * NBIN is a multiple of 4 and s_hist is 16-aligned.
    static_assert(NBIN % 4 == 0, "");
    i32x4* __restrict__ h4 = (i32x4*)&s_hist[0][0];
    const i32x4 z4 = {0, 0, 0, 0};
    for (int i = threadIdx.x; i < S * (NBIN / 4); i += NTH)
      h4[i] = z4;
  }
  // Run-link check of this wave's pages (wave-uniform): the first 64-page
  // chunk comes from the prefetched ptr_, later chunks (only when a wave
  // owns > 64 pages) are read here.
  unsigned int brk = 0u;
#pragma unroll
  for (int j = 0; j + 1 < G; ++j) {
    if (j + 1 < S) {
      const int cnt = nkr[j] < 64 ? nkr[j] : 64;
      bool bad = nkr[j] != nkr[j + 1];
      if (!bad) bad = __any((lane < cnt) && (ptr_[j] != ptr_[j + 1]));
      if (!bad) bad = __any(sk_in && (ptE_[j] != ptE_[j + 1]));
      if (!bad) {
        const int nkp = nkr[j];
        const int* __restrict__ pa = ptable + (size_t)(slot0 + j) * max_pages;
        const int* __restrict__ pb = pa + max_pages;
        for (int k0 = 64; k0 < nkp; k0 += 64) {
          const int cnt2 = (nkp - k0) < 64 ? (nkp - k0) : 64;
          const int pp = wid + (k0 + lane) * nwaves;
          const bool diff = (lane < cnt2) && (pa[pp] != pb[pp]);
          if (__any(diff)) {
            bad = true;
            break;
          }
        }
      }
      if (bad) brk |= 1u << j;
    }
  }
  brk = __builtin_amdgcn_readfirstlane(brk);
  {
    const bool live = wid + nwaves < npages_of(slr[0]);
    issue(buf[1], ksb[1], live ? 1 : 0, live);
  }
  s_brk[warp_u] = brk;
  if (sk_own) s_pt[lane * WARPS + warp_u] = ptr_[0];
  if (sk_in) s_pt[sk_slot] = ptE_[0];
  if (threadIdx.x < G) s_ctr[threadIdx.x] = (WARPS * kSlotPrefetch + sk_skip) * 64;
  __syncthreads();
  {
    unsigned int bo = 0u;
#pragma unroll
    for (int w = 0; w < WARPS; ++w)
      bo |= s_brk[w];
    brk = __builtin_amdgcn_readfirstlane(bo);
  }
  bool primed = true;

  // Runs of linked rows inside the slot; normally one run of S rows.
  for (int rs = slot0; rs < slot_end;) {
    int gcnt = 1;
    while (rs + gcnt < slot_end && !((brk >> (rs + gcnt - 1 - slot0)) & 1u))
      ++gcnt;
    const int js = rs - slot0;

    // Rows beyond gcnt replay the run's first row (same q/w, same out row,
    // identical value) with the histogram off: the store count per page is
    // then a compile-time constant, which keeps the vmcnt bookkeeping of the
    // prefetch ring exact.
    // The slot rows' q fragments and head weights, loaded from LDS once per
    // run instead of once per page and row.
    V8 qr[G][2];
    f32x4 wr[G][2];
#pragma unroll
    for (int j = 0; j < G; ++j) {
      const int jr = js + (j < gcnt ? j : 0);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        qr[j][hh].h[0] = s_q[jr][2 * hh][lane];
        qr[j][hh].h[1] = s_q[jr][2 * hh + 1][lane];
        wr[j][hh] = s_w[jr][4 * hh + g];
      }
    }
    // Cross-row pipelined consume: the 8 MFMAs of row j+1 are issued before
    // the VALU work (relu-weighted head sum, lane reduction, store, histogram)
    // of row j, so the MFMA pipe runs under the previous row's VALU tail.
    // Per-element math and its order are unchanged (bitwise identical).
    auto consume = [&](const V8(&b)[4], const float ks, int pk) __attribute__((always_inline)) {
      const int p = pk;
      const int pos = p * PAGE_TOK + lane;
      f32x4 am[2][2][4];
      auto mf = [&](int j, f32x4(&o)[2][4]) __attribute__((always_inline)) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int tt = 0; tt < 4; ++tt) {
            const f32x4 z = {0.f, 0.f, 0.f, 0.f};
            o[hh][tt] = mfma128(qr[j][hh].v, b[tt].v, z);
          }
      };
      auto va = [&](int j, const f32x4(&o)[2][4]) __attribute__((always_inline)) {
        const int jr = js + (j < gcnt ? j : 0);
        const int seq_r = seqlens[slot0 + jr];
        const bool dh_r = (j < gcnt) & dohist_of(seq_r);
        float* __restrict__ orow_r = out + (size_t)(slot0 + jr) * out_stride;
        float s[4];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const f32x4 w = wr[j][hh];
#pragma unroll
          for (int tt = 0; tt < 4; ++tt) {
            float acc = hh ? s[tt] : 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i)
              acc = fmaf(fmaxf(o[hh][tt][i], 0.f), w[i], acc);
            s[tt] = acc;
          }
        }
        float A0 = s[0], B0 = s[2], A1 = s[1], B1 = s[3];
        asm("s_nop 1\n\tv_permlane32_swap_b32 %0, %1" : "+v"(A0), "+v"(B0));
        asm("s_nop 1\n\tv_permlane32_swap_b32 %0, %1" : "+v"(A1), "+v"(B1));
        float T0 = A0 + B0, T1 = A1 + B1;
        asm("s_nop 1\n\tv_permlane16_swap_b32 %0, %1" : "+v"(T0), "+v"(T1));
        const float T = T0 + T1;
        const bool live = pos < seq_r;
        const float v = T * ks;
        __builtin_amdgcn_raw_buffer_store_b32(
            __builtin_bit_cast(int, live ? v : -INFINITY),
            __builtin_amdgcn_make_buffer_rsrc((void*)orow_r, (short)0, 0x7fffffff, 0x00020000),
            pos * 4,
            0,
            kOutStoreAux);
        // Branch-free 0/1 add: no divergent branch in the page body.
        const uint32_t bin = order_bin_fast_nofma<LOWB>(v);
        atomicAdd(&s_hist[jr][bin], (dh_r && live) ? 1u : 0u);
      };
      mf(0, am[0]);
#pragma unroll
      for (int j = 0; j < G; ++j) {
        if (j + 1 < G) mf(j + 1, am[(j + 1) & 1]);
        __builtin_amdgcn_sched_barrier(0);
        va(j, am[j & 1]);
        __builtin_amdgcn_sched_barrier(0);
      }
    };

    const int* __restrict__ pt = ptable + (size_t)rs * max_pages;
    // Dynamic intra-block page queue.  The block owns pages
    // p(i) = base + (i % WARPS) + (i / WARPS) * nwaves, i in [0, NB) (exactly
    // the union of its waves' static pages).  Each wave starts with its static
    // first kSlotPrefetch pages (i = warp + d*WARPS) and then pulls the next i from an
    // LDS counter, so waves that the SIMD arbiter favours take more pages and
    // the block's waves finish together.  Per-page math is unchanged.
    {
      const int base = wid - warp_u;
      const int np = npages_of(seqlens[rs]);
      const int R = np - base;
      // Own queue length, then the pair skew (both members saw the same np:
      // the skew is off unless every slot row has the same page count).
      const int NBo = R <= 0 ? 0 : (R / nwaves) * WARPS + ((R % nwaves) < WARPS ? (R % nwaves) : WARPS);
      const int NB = NBo + sk_add;
      auto pof = [&](int i) __attribute__((always_inline)) {
        const bool own = i < NBo;
        const int ii = own ? i : i + sk_d;
        const int bb = own ? base : base + WARPS;
        return bb + (ii & (WARPS - 1)) + (ii / WARPS) * nwaves;
      };
      auto phys_of = [&](int i) __attribute__((always_inline)) {
        int v;
        if (i < 64 * WARPS)
          v = __builtin_amdgcn_readfirstlane(s_pt[i]);
        else
          v = pt[pof(i)];
        return v;
      };
      auto grab = [&]() __attribute__((always_inline)) {
        // Every lane adds 1, so one pull advances the counter by 64 and any
        // lane's return value >> 6 is the pulled index (lane order is moot).
        const unsigned int o = __hip_atomic_fetch_add(&s_ctr[js], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        return (int)(__builtin_amdgcn_readfirstlane(o) >> 6);
      };
      if (rs != slot0) {
        __syncthreads();
        for (int t = threadIdx.x; t < (NB < 64 * WARPS ? NB : 64 * WARPS); t += NTH)
          s_pt[t] = pt[pof(t)];
        __syncthreads();
#pragma unroll
        for (int d = 0; d < kSlotPrefetch; ++d) {
          const int i = warp_u + d * WARPS;
          // Unconditional issue (num_records = 0 when dead): both ring slots
          // are always in flight, so the vmcnt bookkeeping stays exact.
          issue_p(buf[d], ksb[d], i < NB ? phys_of(i) : 0, i < NB);
        }
      }
      int ia = warp_u, ib = warp_u + WARPS;
      static_assert(kSlotPrefetch == 2, "dynamic queue ring is written for kSlotPrefetch == 2");
      // ia < ib always (pulled in that order), so ib < NB means both are live.
      while (ib < NB) {
        consume(buf[0], ksb[0], pof(ia));
        ia = grab();
        issue_p(buf[0], ksb[0], ia < NB ? phys_of(ia) : 0, ia < NB);
        consume(buf[1], ksb[1], pof(ib));
        ib = grab();
        issue_p(buf[1], ksb[1], ib < NB ? phys_of(ib) : 0, ib < NB);
      }
      if (ia < NB) consume(buf[0], ksb[0], pof(ia));
    }
    rs += gcnt;
  }

  if (any_hist) {
    __syncthreads();
    // Flush, one batch: every LDS read of every slot row is issued before the
    // first wait, the 8-lane coarse sums use DPP (no ds_bpermute round trips),
    // the dohist test uses the prologue's seqlens (no s_load per row), and two
    // adjacent coarse bins go out as one 64-bit atomic.  Integer sums only, so
    // ghist is bit-identical whatever the order.
    constexpr int PAIRS_PER_THREAD = NBIN / 2 / NTH;  // fine pairs per thread per row
    static_assert(PERT == 8 && GRP == 8, "DPP coarse tree assumes 8 bins/thread, 8 threads/coarse bin");
    unsigned long long fv[G][PAIRS_PER_THREAD];
    i32x4 cv[G][2];
#pragma unroll
    for (int j = 0; j < G; ++j) {
      const int jj = j;  // rows >= S: LDS read only, the atomics below are skipped
      const i32x4* c4 = (const i32x4*)&s_hist[jj][threadIdx.x * PERT];
      cv[j][0] = c4[0];
      cv[j][1] = c4[1];
#pragma unroll
      for (int t = 0; t < PAIRS_PER_THREAD; ++t)
        fv[j][t] = ((const unsigned long long*)&s_hist[jj][0])[threadIdx.x + t * NTH];
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
      if (!((j < S) && dohist_of(slr[j]))) continue;
      unsigned int* __restrict__ gh = ghist + (size_t)(slot0 + j) * GHS;
      int acc =
          cv[j][0][0] + cv[j][0][1] + cv[j][0][2] + cv[j][0][3] + cv[j][1][0] + cv[j][1][1] + cv[j][1][2] + cv[j][1][3];
      acc += __builtin_amdgcn_update_dpp(0, acc, 0x104, 0xf, 0xf, true);              // row_shl:4
      acc += __builtin_amdgcn_update_dpp(0, acc, 0x102, 0xf, 0xf, true);              // row_shl:2
      acc += __builtin_amdgcn_update_dpp(0, acc, 0x101, 0xf, 0xf, true);              // row_shl:1
      const int acc_hi = __builtin_amdgcn_update_dpp(0, acc, 0x108, 0xf, 0xf, true);  // row_shl:8
      const unsigned long long cc = (unsigned long long)(unsigned)acc | ((unsigned long long)(unsigned)acc_hi << 32);
      if ((threadIdx.x & 15) == 0 && cc) atomicAdd((unsigned long long*)(gh + NBIN + (int)(threadIdx.x >> 3)), cc);
#pragma unroll
      for (int t = 0; t < PAIRS_PER_THREAD; ++t) {
        const unsigned long long vv = fv[j][t];
        if (vv) atomicAdd((unsigned long long*)(gh + 2 * ((int)threadIdx.x + t * NTH)), vv);
      }
    }
  }
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
      int64_t share_pages) {
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

    const int rows = static_cast<int>(rows_.unwrap());
    const int table_pages = static_cast<int>(pages_.unwrap());
    // The caller asks for the shared-page kernel only where it measured faster
    // (long MTP verify contexts, chosen on the host per CUDA-graph variant).
    if (share_pages && rows <= impl::kSlotMaxRows) {
      // Per grid size (MTP verify: 6 rows per request), measured on MI355X:
      // blocks per row, and the narrower width used below `wide_ctx` tokens.
      // A row's 8 waves never get more blocks than the table has 8-page groups.
      const int bpr_want = rows <= 6 ? 40 : rows <= 12 ? 20 : 14;
      const int wide_ctx = rows <= 6 ? 196608 : rows <= 12 ? 49152 : 0;
      const int narrow_want = rows <= 6 ? 24 : rows <= 12 ? 16 : 14;
      const int bpr_cap = (table_pages + 7) / 8 > 0 ? (table_pages + 7) / 8 : 1;
      const int bpr = bpr_want < bpr_cap ? bpr_want : bpr_cap;
      const int narrow_bpr = narrow_want < bpr ? narrow_want : bpr;
      LaunchKernel(dim3(bpr, rows), dim3(8 * 64), device_.unwrap())(
          impl::logits_hist_slot<8, HOST_HB>,
          reinterpret_cast<const uint8_t*>(q.data_ptr()),
          reinterpret_cast<const uint8_t*>(kv.data_ptr()),
          static_cast<fp32_t*>(weights.data_ptr()),
          static_cast<int*>(seqlens.data_ptr()),
          static_cast<int*>(page_table.data_ptr()),
          static_cast<fp32_t*>(out.data_ptr()),
          reinterpret_cast<unsigned int*>(ghist.data_ptr()),
          table_pages,
          static_cast<int>(out.stride(0)),
          wide_ctx,
          narrow_bpr);
      return;
    }

    LaunchKernel(
        dim3(static_cast<int>(blocks_per_row), static_cast<int>(rows_.unwrap())), dim3(8 * 64), device_.unwrap())(
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
