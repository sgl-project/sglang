// Top-k(2048) + page transform, adapted from aiter's coop_topk.cuh (MIT).
// aiter runs one block per row; this splits a row across G blocks that agree
// without communicating, via the histogram the logits kernel already built.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace dsa_topk {

constexpr uint32_t TOPK = 2048u;
// Coarse bin width, in bits of the fp16 ordered key: trades candidate-set size
// against global-histogram traffic.
#ifndef DSA_TOPK_HIST_BITS
#define DSA_TOPK_HIST_BITS 12
#endif
constexpr uint32_t HIST_BITS = (uint32_t)DSA_TOPK_HIST_BITS;
constexpr uint32_t HIST_BINS = 1u << HIST_BITS;
constexpr uint32_t LOW_BITS = 16u - HIST_BITS;
// Hierarchical (two-level) threshold.
constexpr uint32_t CBITS = 6u;
constexpr uint32_t CBINS = 1u << CBITS;               // 64, one wave wide
constexpr uint32_t FINE_PER_CRS = HIST_BINS >> CBITS; // 64 at HIST_BITS=12
static_assert(HIST_BITS >= CBITS,
              "coarse must be no wider than the fine histogram");
static_assert(FINE_PER_CRS <= 64u, "the fine group must fit in one wave");
// Row stride of the ghist workspace: the fine histogram followed by the coarse
// summary.  Both halves are zeroed by whoever owns the reset.
constexpr uint32_t GH_STRIDE = HIST_BINS + CBINS;
constexpr uint32_t GH_CRS_OFF = HIST_BINS;
#ifndef DSA_TOPK_BS
#define DSA_TOPK_BS 256
#endif
constexpr uint32_t BS = (uint32_t)DSA_TOPK_BS; // threads per block
constexpr uint32_t RADIX = 256u;               // refinement radix
static_assert(BS % 64u == 0u, "BS must be a whole number of wave64 waves");
static_assert(BS >= 64u && BS <= 1024u, "HIP workgroup bound");
#ifndef DSA_TOPK_STAGE
#define DSA_TOPK_STAGE 512
#endif
constexpr uint32_t STAGE = (uint32_t)DSA_TOPK_STAGE;
// Candidates held in LDS by the refinement.  Above this it streams from global,
// which only happens on inputs where one coarse bin holds > REF_CAP elements.
constexpr uint32_t REF_CAP = 2048u;
// Candidate count up to which the refinement uses the O(n^2) rank method
// instead of the radix rounds.  See refine_row.
#ifndef DSA_TOPK_RANK_CAP
#define DSA_TOPK_RANK_CAP 256
#endif
constexpr uint32_t RANK_CAP = (uint32_t)DSA_TOPK_RANK_CAP;
// The parallel rank puts one thread on one j, so it needs n <= BS as well as
// n <= RANK_CAP; the serial rank stays the fallback above the min of the two.
constexpr uint32_t PRANK_CAP = RANK_CAP < BS ? RANK_CAP : BS;
// Per-row counters are padded to their own cache line, and the two counters a
// block updates live in one 8-byte word, so a block reserves output space with
// ONE returning global atomic instead of two and rows do not false-share.
constexpr uint32_t RC_STRIDE = 32u;
constexpr uint32_t RC_WIN = 0u;  // winners emitted so far
constexpr uint32_t RC_CAND = 1u; // candidates appended so far
constexpr uint32_t RC_ARR = 4u;  // arrival counter (own dword)
// Departure half of the two-phase row barrier.  The arrival counter alone is
// not a barrier more than one block may pass: whoever reset it would race the
// blocks still spinning on it.
constexpr uint32_t RC_DEP = 5u;

// PTMODE 2 stages a window of the page table in LDS.  A block's slice is
// contiguous, so every position it can emit lies in one span of the table.
constexpr uint32_t PT_WIN = 256u;

__device__ __forceinline__ uint32_t order_key32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Ordered 16-bit key of x rounded to fp16.  __float2half_rn is monotonic, so
// key16(a) <= key16(b) whenever a <= b, which is what makes the three-way
// split in K2 sound.
__device__ __forceinline__ uint32_t order_key16(float x) {
  __half h = __float2half_rn(x);
  unsigned short bits = __half_as_ushort(h);
  unsigned short key = (bits & 0x8000) ? (unsigned short)(~bits)
                                       : (unsigned short)(bits | 0x8000);
  return (uint32_t)key;
}

// Physical KV slot of a row-relative position: pt64[row][p >> 6] * 64 + (p &
// 63) for page_size 64, which is the definition of page_table_1.  PB/PM carry
// the shift and mask so page_size 1 collapses to the identity.
__device__ __forceinline__ int32_t slot_of(const int32_t *__restrict__ pt,
                                           uint32_t pos, uint32_t page_bits,
                                           uint32_t page_mask) {
  return (pt[pos >> page_bits] << page_bits) | (int32_t)(pos & page_mask);
}

struct Slice {
  uint32_t start, len;
};

// This block's slice, cut on float4 boundaries so every load stays 16B aligned.
__device__ __forceinline__ Slice slice_of(uint32_t row_len, uint32_t g,
                                          uint32_t G) {
  const uint32_t units = (row_len + 3u) / 4u;
  const uint32_t base = units / G;
  const uint32_t extra = units % G;
  const uint32_t my_u = base + (g < extra ? 1u : 0u);
  const uint32_t off_u = g * base + (g < extra ? g : extra);
  Slice s{};
  s.start = off_u * 4u;
  s.len = s.start >= row_len ? 0u : min(my_u * 4u, row_len - s.start);
  return s;
}

template <typename Op>
__device__ __forceinline__ void scan_slice(const float *__restrict__ in,
                                           Slice sl, Op op) {
  const uint32_t tx = threadIdx.x;
  const uint32_t vec_len = sl.len & ~3u;
  const float4 *in4 = reinterpret_cast<const float4 *>(in + sl.start);
  const uint32_t n4 = vec_len >> 2;

  uint32_t i = tx;
  for (; i + 3u * BS < n4; i += 4u * BS) {
    const float4 v0 = in4[i];
    const float4 v1 = in4[i + BS];
    const float4 v2 = in4[i + 2u * BS];
    const float4 v3 = in4[i + 3u * BS];
    const float4 vv[4] = {v0, v1, v2, v3};
#pragma unroll
    for (uint32_t u = 0; u < 4u; ++u) {
      const float4 v = vv[u];
      const float vals[4] = {v.x, v.y, v.z, v.w};
      const uint32_t bpos = sl.start + ((i + u * BS) << 2);
#pragma unroll
      for (uint32_t j = 0; j < 4u; ++j) {
        op(vals[j], bpos + j);
      }
    }
  }
  for (; i < n4; i += BS) {
    const float4 v = in4[i];
    const float vals[4] = {v.x, v.y, v.z, v.w};
    const uint32_t bpos = sl.start + (i << 2);
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
      op(vals[j], bpos + j);
    }
  }
  for (uint32_t t = vec_len + tx; t < sl.len; t += BS) {
    op(in[sl.start + t], sl.start + t);
  }
}

constexpr uint32_t WAVE = 64u; // gfx950
constexpr uint32_t NWAVE = BS / WAVE;

__device__ __forceinline__ void hist_add_agg(uint32_t *hist, uint32_t bin) {
  const uint64_t active = __ballot(1);
  const int leader = __ffsll((unsigned long long)active) - 1;
  const uint32_t lead_bin = __shfl(bin, leader, WAVE);
  if (__all(bin == lead_bin)) {
    if ((int)(threadIdx.x % WAVE) == leader) {
      atomicAdd(&hist[lead_bin], (uint32_t)__popcll(active));
    }
  } else {
    atomicAdd(&hist[bin], 1u);
  }
}

template <uint32_t NB>
__device__ __forceinline__ void
find_thr(const uint32_t *__restrict__ hist, uint32_t *__restrict__ wtot,
         uint32_t want, uint32_t *out_thr, uint32_t *out_above) {
  // NB < BS is legal: PER is then 1, threads past NB hold a zero count, and a
  // zero bin cannot satisfy the bracket predicate below.
  constexpr uint32_t PER = (NB >= BS) ? (NB / BS) : 1u;
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % WAVE;
  const uint32_t wv = tx / WAVE;

  uint32_t local[PER];
  uint32_t mine = 0;
#pragma unroll
  for (uint32_t j = 0; j < PER; ++j) {
    if constexpr (NB >= BS) {
      local[j] = hist[tx * PER + j];
    } else {
      const uint32_t idx = tx * PER + j;
      local[j] = (idx < NB) ? hist[idx] : 0u;
    }
    mine += local[j];
  }

  uint32_t incl = mine;
#pragma unroll
  for (uint32_t o = 1; o < WAVE; o <<= 1) {
    const uint32_t nv = __shfl_up(incl, o, WAVE);
    if (lane >= o) {
      incl += nv;
    }
  }
  if (lane == WAVE - 1u) {
    wtot[wv] = incl;
  }
  __syncthreads();

  uint32_t base = 0, total = 0;
#pragma unroll
  for (uint32_t w = 0; w < NWAVE; ++w) {
    const uint32_t t = wtot[w];
    if (w < wv) {
      base += t;
    }
    total += t;
  }
  uint32_t acc = total - (base + incl); // strictly above this thread's group

#pragma unroll
  for (int j = (int)PER - 1; j >= 0; --j) {
    const uint32_t c = local[j];
    if (acc < want && acc + c >= want) {
      *out_thr = tx * PER + (uint32_t)j;
      *out_above = acc;
    }
    acc += c;
  }
  __syncthreads(); // wtot[] is scratch and out_thr/out_above must be visible
}

__device__ __forceinline__ uint32_t wave_suffix_sum(uint32_t v) {
  const uint32_t lane = threadIdx.x & (WAVE - 1u);
#pragma unroll
  for (uint32_t o = 1; o < WAVE; o <<= 1) {
    const uint32_t t = __shfl_down(v, o, WAVE);
    v += (lane + o < WAVE) ? t : 0u;
  }
  return v;
}

// The same threshold as find_thr<HIST_BINS>, from two levels, bit-identical.
// A histogram written without its coarse summary fails silently, so lane 0's
// suffix sum must equal row_len; on mismatch return HIER_BAD.
constexpr uint32_t HIER_BAD = 0xFFFFFFFFu;

__device__ __forceinline__ uint32_t
find_thr_hier_impl(const uint32_t *__restrict__ gh, uint32_t want, uint32_t crs,
                   uint32_t total_must_be) {
  const uint32_t lane = threadIdx.x & (WAVE - 1u);

  const uint32_t c = crs;
  const uint32_t s = wave_suffix_sum(c);
  {
    if (__shfl(s, 0, WAVE) != total_must_be) {
      return HIER_BAD;
    }
  }
  uint32_t sn = __shfl_down(s, 1, WAVE);
  if (lane == WAVE - 1u) {
    sn = 0u;
  }
  const uint64_t b1 = __ballot(s >= want && sn < want);
  // Cannot be empty (total == row_len > want and lane 63 sees sn == 0), but a
  // zero ballot would index gh at -1, so it degrades to bin 0 -- which is also
  // what find_thr leaves in *out_thr when it finds nothing.
  const uint32_t j = b1 ? (uint32_t)(__ffsll((unsigned long long)b1) - 1) : 0u;
  const uint32_t abv = b1 ? __shfl(sn, (int)j, WAVE) : 0u;

  // level 2 -- the FINE_PER_CRS fine bins under coarse bin j.  Lanes past the
  // group contribute 0, so they can never satisfy the predicate (want2 >= 1).
  const uint32_t f = (lane < FINE_PER_CRS) ? gh[j * FINE_PER_CRS + lane] : 0u;
  const uint32_t sf = wave_suffix_sum(f);
  uint32_t sfn = __shfl_down(sf, 1, WAVE);
  if (lane == WAVE - 1u) {
    sfn = 0u;
  }
  const uint32_t want2 = want - abv;
  const uint64_t b2 = __ballot(sf >= want2 && sfn < want2);
  const uint32_t l = b2 ? (uint32_t)(__ffsll((unsigned long long)b2) - 1) : 0u;
  return j * FINE_PER_CRS + l;
}

// Exact refinement of the threshold bin + padding.

__device__ __forceinline__ float ld_val(const float *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ int32_t ld_idx(const int32_t *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <bool IN_LDS>
__device__ __forceinline__ void
refine_core(uint32_t n, uint32_t above, uint32_t remain,
            const uint32_t *__restrict__ s_key,
            const int32_t *__restrict__ s_slot, const float *__restrict__ cv,
            const int32_t *__restrict__ ci, int32_t *__restrict__ o,
            uint32_t *__restrict__ s_hist, uint32_t *__restrict__ s_grp,
            uint32_t *s_thr, uint32_t *s_above, uint32_t *s_emit) {
  const uint32_t tx = threadIdx.x;

  if (remain == 0 || n <= remain) {
    // Defensive: the threshold guarantees n >= remain, so this is the
    // degenerate path only.  Emit everything we have, then pad.
    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t p = above + i;
      if (p < TOPK) {
        o[p] = IN_LDS ? s_slot[i] : ld_idx(&ci[i]);
      }
    }
    __syncthreads();
    for (uint32_t i = above + n + tx; i < TOPK; i += BS) {
      o[i] = -1;
    }
    return;
  }

  uint32_t prefix = 0;
  for (int r = 0; r < 4; ++r) {
    const uint32_t sh = 24u - (uint32_t)r * 8u;

    for (uint32_t i = tx; i < RADIX; i += BS) {
      s_hist[i] = 0u;
    }
    if (tx == 0) {
      *s_thr = 0;
      *s_above = 0;
    }
    __syncthreads();

    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t key = IN_LDS ? s_key[i] : order_key32(ld_val(&cv[i]));
      const bool in_play =
          (r == 0) || (((key >> (sh + 8u)) << (sh + 8u)) == prefix);
      if (in_play) {
        hist_add_agg(s_hist, (key >> sh) & 0xFFu);
      }
    }
    __syncthreads();

    find_thr<RADIX>(s_hist, s_grp, remain, s_thr, s_above);

    const uint32_t thr = *s_thr;
    const uint32_t abv = *s_above;
    // Both reads must complete before any thread can reach the next round's
    // reset of *s_thr / *s_above at the top of this loop.  The `continue` below
    // has no barrier after it, so without this a thread that took it would zero
    // *s_thr while a lagging wave was still reading it, and that wave would
    // carry `prefix |= 0` into every later round.
    __syncthreads();

    // Nothing above the threshold bin settles no winner, so the emit scan
    // would find nothing.  Skipping it matters on clustered rows, where the
    // early rounds are exactly this case.
    if (abv == 0 && r != 3) {
      prefix |= (thr << sh);
      continue;
    }

    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t key = IN_LDS ? s_key[i] : order_key32(ld_val(&cv[i]));
      const bool in_play =
          (r == 0) || (((key >> (sh + 8u)) << (sh + 8u)) == prefix);
      if (!in_play) {
        continue;
      }
      const uint32_t bin = (key >> sh) & 0xFFu;
      // Last round: survivors in the threshold bin are numerically equal,
      // so any of them is a correct answer.  The TopK bound on the cursor
      // enforces the quota.
      if (bin > thr || (bin == thr && r == 3)) {
        const uint32_t p = atomicAdd(s_emit, 1u);
        if (p < TOPK) {
          o[p] = IN_LDS ? s_slot[i] : ld_idx(&ci[i]);
        }
      }
    }
    __syncthreads();

    prefix |= (thr << sh);
    remain -= abv;
    if (remain == 0) {
      break;
    }
  }

  __syncthreads();
  // The emit loop above already wrote o[above .. *s_emit) directly, so all
  // that is left is the -1 padding.
  const uint32_t filled = *s_emit < TOPK ? *s_emit : TOPK;
  for (uint32_t i = filled + tx; i < TOPK; i += BS) {
    o[i] = -1;
  }
}

// Exact rank of the boundary candidates, bit-identical to the serial rank.
// Only the i axis is partitioned -- rank_i sums over ALL j in EVERY block --
// so a candidate on a partition boundary is still ranked against the full set.
template <bool PRANK = false>
__device__ __forceinline__ void
refine_row(uint32_t row, uint32_t above, uint32_t n_raw, uint32_t cap,
           const int32_t *__restrict__ cand_idx,
           const float *__restrict__ cand_val, int32_t *__restrict__ o,
           uint32_t *__restrict__ s_hist, uint32_t *__restrict__ s_grp,
           uint32_t *__restrict__ s_key, int32_t *__restrict__ s_slot,
           uint32_t *s_thr, uint32_t *s_above, uint32_t *s_emit,
           bool have_pre = false, int32_t pre_slot = 0, float pre_val = 0.f,
           uint32_t pb = 0u, uint32_t PB = 1u) {
  const uint32_t tx = threadIdx.x;
  const int32_t *__restrict__ ci = cand_idx + (size_t)row * cap;
  const float *__restrict__ cv = cand_val + (size_t)row * cap;

  const uint32_t n = n_raw > cap ? cap : n_raw;
  const uint32_t remain = above < TOPK ? TOPK - above : 0u;

  if (tx == 0) {
    *s_emit = above;
  }

  const bool in_lds = (n <= REF_CAP);
  if (in_lds) {
    // The caller may already have issued cand_idx[tx] / cand_val[tx] without
    // waiting for `n`, collapsing row_ends -> cursor -> candidates into one
    // round trip.
    const uint32_t i0 = have_pre ? tx + BS : tx;
    if (have_pre && tx < n) {
      s_key[tx] = order_key32(pre_val);
      s_slot[tx] = pre_slot;
    }
    for (uint32_t i = i0; i < n; i += BS) {
      s_key[i] = order_key32(ld_val(&cv[i]));
      s_slot[i] = ld_idx(&ci[i]); // k_scatter already translated it
    }
  }
  __syncthreads();

  if (in_lds && n <= RANK_CAP && remain != 0 && n > remain) {
    if constexpr (PRANK) {
      if (n <= PRANK_CAP) {
        const uint32_t lane = tx & (WAVE - 1u);
        const uint32_t wv = tx / WAVE;
        const uint32_t kj = (tx < n) ? s_key[tx] : 0u;
        for (uint32_t i = pb; i < n; i += PB) {
          const uint32_t ki = s_key[i]; // block-uniform read
          const bool p = (tx < n) && ((kj > ki) || (kj == ki && tx < i));
          const uint64_t m = __ballot(p);
          if (lane == 0) {
            s_grp[wv] = (uint32_t)__popcll((unsigned long long)m);
          }
          __syncthreads();
          uint32_t rank = 0;
#pragma unroll
          for (uint32_t q = 0; q < NWAVE; ++q) {
            rank += s_grp[q];
          }
          if (tx == 0 && rank < remain) {
            o[above + rank] = s_slot[i];
          }
          __syncthreads(); // s_grp is reused by the next i
        }
        return;
      }
      // n > PRANK_CAP: one thread per j is not available.  Fall through
      // to the serial rank, in ONE block only -- never silently skipped.
      if (pb != 0u) {
        return;
      }
    }
    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t ki = s_key[i];
      uint32_t rank = 0;
      uint32_t j = 0;
      // Unrolled by 8: the naive loop waits on each LDS read, so it runs at
      // LDS latency rather than throughput.
      for (; j + 8u <= n; j += 8u) {
        const uint32_t k0 = s_key[j + 0], k1 = s_key[j + 1];
        const uint32_t k2 = s_key[j + 2], k3 = s_key[j + 3];
        const uint32_t k4 = s_key[j + 4], k5 = s_key[j + 5];
        const uint32_t k6 = s_key[j + 6], k7 = s_key[j + 7];
        rank += (k0 > ki) || (k0 == ki && (j + 0u) < i);
        rank += (k1 > ki) || (k1 == ki && (j + 1u) < i);
        rank += (k2 > ki) || (k2 == ki && (j + 2u) < i);
        rank += (k3 > ki) || (k3 == ki && (j + 3u) < i);
        rank += (k4 > ki) || (k4 == ki && (j + 4u) < i);
        rank += (k5 > ki) || (k5 == ki && (j + 5u) < i);
        rank += (k6 > ki) || (k6 == ki && (j + 6u) < i);
        rank += (k7 > ki) || (k7 == ki && (j + 7u) < i);
      }
      for (; j < n; ++j) {
        const uint32_t kj = s_key[j];
        rank += (kj > ki) || (kj == ki && j < i);
      }
      if (rank < remain) {
        o[above + rank] = s_slot[i];
      }
    }
  } else if (PRANK && pb != 0u) {
    // The radix fallback emits through a shared cursor and pads o[..TOPK):
    // one block's work by construction, so the rest of the row returns here.
  } else if (in_lds) {
    refine_core<true>(n, above, remain, s_key, s_slot, cv, ci, o, s_hist, s_grp,
                      s_thr, s_above, s_emit);
  } else {
    refine_core<false>(n, above, remain, s_key, s_slot, cv, ci, o, s_hist,
                       s_grp, s_thr, s_above, s_emit);
  }
}

// K2: threshold + scatter (page gather folded into the emit)

// Stage this block's span of the row's page table into LDS (PTMODE 2 only).
// A block's slice is contiguous, so the window is [pt_base, pt_base + npt).
__device__ __forceinline__ void
stage_pt_window(int32_t *__restrict__ s_pt, const int32_t *__restrict__ pt_row,
                Slice sl, uint32_t pt_base, uint32_t page_bits) {
  const uint32_t want =
      sl.len ? (((sl.start + sl.len - 1u) >> page_bits) - pt_base + 1u) : 0u;
  const uint32_t npt = want < PT_WIN ? want : PT_WIN;
  for (uint32_t i = threadIdx.x; i < npt; i += BS) {
    s_pt[i] = pt_row[pt_base + i];
  }
}

template <int PTMODE, bool PRANK = false, int PBLK = 0>
__global__ __launch_bounds__(BS) void k_scatter(
    const float *__restrict__ logits, const int32_t *__restrict__ row_ends,
    const int32_t *__restrict__ page_table, int32_t *__restrict__ out,
    uint32_t *__restrict__ ghist, int32_t *__restrict__ cursor,
    int32_t *__restrict__ cand_idx, float *__restrict__ cand_val,
    int64_t lg_stride, int64_t pt_stride, uint32_t page_bits,
    uint32_t page_mask, uint32_t cap, uint32_t G) {
  // Three working sets with disjoint lifetimes share one 16 KB block: the flat
  // histogram, then the staging buffers, then the refinement's keys and slots.
  // s_radix stays separate -- it is live at the same time as the keys.
  __shared__ union {
    uint32_t hist[HIST_BINS];
    struct {
      int32_t wbuf[STAGE];
      int32_t cidx[STAGE];
      float cval[STAGE];
    } stage;
    struct {
      uint32_t key[REF_CAP];
      int32_t slot[REF_CAP];
    } ref;
  } s_pool;
  static_assert(sizeof(s_pool) == HIST_BINS * 4u,
                "the flat histogram is the widest phase");
  __shared__ uint32_t s_radix[RADIX];
  __shared__ uint32_t s_grp[NWAVE]; // cross-wave scan fixup (see find_thr)
  __shared__ uint32_t s_thr, s_above;
  // Per-block staging.  Without it every emitted element would need its own
  // returning atomic on the row's single cursor.
  __shared__ uint32_t s_wcnt, s_ccnt;
  __shared__ int32_t s_wbase, s_cbase;
  // PTMODE is a template parameter and not a runtime flag for the same reason
  // IN_LDS is in refine_core: a runtime select would emit both forms.
  __shared__ int32_t s_pt[PTMODE == 2 ? PT_WIN : 1];
  __shared__ uint32_t s_emit;
  __shared__ uint32_t s_last;
  uint32_t *const s_hist = s_pool.hist;
  int32_t *const s_wbuf = s_pool.stage.wbuf;
  int32_t *const s_cidx = s_pool.stage.cidx;
  float *const s_cval = s_pool.stage.cval;
  uint32_t *const s_key = s_pool.ref.key;
  int32_t *const s_slot = s_pool.ref.slot;

  const uint32_t row = blockIdx.y;
  const uint32_t g = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  const int32_t rl_s = row_ends[row];
  // Issued before row_ends comes back: the address depends only on blockIdx.
  // Later would serialise row_ends -> coarse bins -> fine group.
  uint32_t pre_crs = 0u;
  pre_crs = ((const uint32_t *)ghist)[(size_t)row * GH_STRIDE + GH_CRS_OFF +
                                      (threadIdx.x & (WAVE - 1u))];
  // row_ends is device data, so no host check can bound it.  Unclamped, an
  // oversized row reads past its page table, and under PTMODE 2 past s_pt --
  // which stays inside LDS and so returns garbage rather than faulting.
  const uint32_t pt_cap_pt = (uint32_t)((uint64_t)pt_stride << page_bits);
  // ...and past the logits row, which the scan below walks over [0, row_len).
  // The paired logits kernel bounds its write by the same two quantities, and
  // the two must agree: a row binned there but skipped here is never zeroed.
  const uint32_t pt_cap_lg = (uint32_t)lg_stride;
  const uint32_t pt_cap = pt_cap_pt < pt_cap_lg ? pt_cap_pt : pt_cap_lg;
  const uint32_t rl_u = rl_s > 0 ? (uint32_t)rl_s : 0u;
  const uint32_t row_len = rl_u < pt_cap ? rl_u : pt_cap;
  if (row_len <= TOPK) {
    // Rows short enough to need no selection: this kernel is their only
    // writer, so it emits the whole row here.
    const int32_t *__restrict__ pt0 = page_table + (int64_t)row * pt_stride;
    int32_t *__restrict__ o0 = out + (size_t)row * TOPK;
    for (uint32_t i = g * BS + tx; i < TOPK; i += G * BS) {
      o0[i] = i < row_len ? slot_of(pt0, i, page_bits, page_mask) : -1;
    }
    return;
  }

  int32_t *__restrict__ rc = cursor + (size_t)row * RC_STRIDE;

  // Issued first: independent of everything else, so the page-table staging
  // rides along under the barrier below.
  const Slice sl = slice_of(row_len, g, G);
  // First page-table entry this block can possibly touch.  Every position it
  // emits is inside its own slice, so the window is [pt_base, pt_base+npt).
  const uint32_t pt_base = (PTMODE == 2) ? (sl.start >> page_bits) : 0u;
  if constexpr (PTMODE == 2) {
    stage_pt_window(s_pt, page_table + (int64_t)row * pt_stride, sl, pt_base,
                    page_bits);
  }

  const uint32_t *__restrict__ gh =
      (const uint32_t *)ghist + (size_t)row * GH_STRIDE;
  uint32_t thr_h = 0u;
  bool need_flat = false; // HIER: the flat histogram is the fallback only
  // Issued before the barrier so its two dependent loads overlap the
  // page-table window staging above.  Needs neither LDS nor a barrier.
  thr_h = find_thr_hier_impl(gh, TOPK, pre_crs, row_len);
  need_flat = (thr_h == HIER_BAD); // block-uniform

  if (need_flat) {
    for (uint32_t i = tx; i < HIST_BINS; i += BS) {
      s_hist[i] = gh[i];
    }
  }
  if (tx == 0) {
    s_thr = 0;
    s_above = 0;
    s_wcnt = 0;
    s_ccnt = 0;
  }
  __syncthreads();

  if (need_flat) {
    find_thr<HIST_BINS>(s_hist, s_grp, TOPK, &s_thr, &s_above);
    __syncthreads();
  }

  const uint32_t thr = need_flat ? s_thr : thr_h;

  const float *__restrict__ in = logits + (int64_t)row * lg_stride;
  const int32_t *__restrict__ pt = page_table + (int64_t)row * pt_stride;
  int32_t *__restrict__ o = out + (size_t)row * TOPK;
  int32_t *__restrict__ ci = cand_idx + (size_t)row * cap;
  float *__restrict__ cv = cand_val + (size_t)row * cap;

  // The page-table lookup.  `if constexpr` so exactly one form is
  // emitted (see the s_pt declaration).
  auto SLOT = [&](uint32_t p) -> int32_t {
    if constexpr (PTMODE == 2) {
      return (s_pt[(p >> page_bits) - pt_base] << page_bits) |
             (int32_t)(p & page_mask);
    } else {
      return slot_of(pt, p, page_bits, page_mask);
    }
  };

  scan_slice(in, sl, [&](float v, uint32_t gi) {
    const uint32_t kb = order_key16(v) >> LOW_BITS;
    if (kb > thr) {
      // Winner outright: monotonicity of order_key16 guarantees it beats
      // every element of the threshold bin.
      const uint32_t p = atomicAdd(&s_wcnt, 1u);
      if (p < STAGE) {
        s_wbuf[p] = (int32_t)gi;
      } else {
        const unsigned long long old =
            atomicAdd((unsigned long long *)rc, 1ull);
        const uint32_t q = (uint32_t)old;
        // The threshold guarantees q < TopK; the bound is belt-and-braces
        // so that no reachable state can scribble past the output row.
        if (q < TOPK) {
          o[q] = SLOT(gi);
        }
      }
    } else if (kb == thr) {
      const uint32_t p = atomicAdd(&s_ccnt, 1u);
      if (p < STAGE) {
        s_cidx[p] = (int32_t)gi;
        s_cval[p] = v;
      } else {
        const unsigned long long old =
            atomicAdd((unsigned long long *)rc, 1ull << 32);
        const uint32_t q = (uint32_t)(old >> 32);
        if (q < cap) {
          __hip_atomic_store(&ci[q], SLOT(gi), __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
          __hip_atomic_store(&cv[q], v, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  });

  __syncthreads();

  const uint32_t wn = s_wcnt < STAGE ? s_wcnt : STAGE;
  const uint32_t cn = s_ccnt < STAGE ? s_ccnt : STAGE;

  // Issue the page-table gathers first: they do not depend on the base this
  // block is about to reserve, so they overlap the atomic's round trip.
  constexpr uint32_t PERT = (STAGE + BS - 1u) / BS;
  int32_t wslot[PERT], cslot[PERT];
  float cvalr[PERT];
#pragma unroll
  for (uint32_t u = 0; u < PERT; ++u) {
    const uint32_t i = tx + u * BS;
    if (i < wn) {
      wslot[u] = SLOT((uint32_t)s_wbuf[i]);
    }
    if (i < cn) {
      cslot[u] = SLOT((uint32_t)s_cidx[i]);
      cvalr[u] = s_cval[i];
    }
  }

  // One thread reserves this block's span in both output streams with a single
  // packed atomic, so the two counters cannot be observed out of step.
  if (tx == 0) {
    const unsigned long long pack =
        ((unsigned long long)cn << 32) | (unsigned long long)wn;
    const unsigned long long old = atomicAdd((unsigned long long *)rc, pack);
    s_wbase = (int32_t)(uint32_t)old;
    s_cbase = (int32_t)(uint32_t)(old >> 32);
  }
  __syncthreads();

#pragma unroll
  for (uint32_t u = 0; u < PERT; ++u) {
    const uint32_t i = tx + u * BS;
    if (i < wn) {
      const uint32_t p = (uint32_t)s_wbase + i;
      if (p < TOPK) {
        o[p] = wslot[u];
      }
    }
    if (i < cn) {
      const uint32_t q = (uint32_t)s_cbase + i;
      if (q < cap) {
        // The only bytes another block reads inside this kernel, which is why
        // just these stores are agent-scope.
        __hip_atomic_store(&ci[q], cslot[u], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&cv[q], cvalr[u], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }

  // Release, the cheap way: the candidate stores above are already
  // write-through to a coherent point, so waiting for them to retire is
  // enough -- no __threadfence().
  __builtin_amdgcn_s_waitcnt(/*vmcnt(0)*/ 0x0f70);
  __syncthreads();
  if (tx == 0) {
    const uint32_t old =
        __hip_atomic_fetch_add((uint32_t *)&rc[RC_ARR], 1u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
    // PRANK needs this block's arrival rank, the non-PRANK form only the
    // is-last flag, and no form needs both, so they share s_last.
    s_last = PRANK ? old : ((old + 1u == G) ? 1u : 0u);
  }
  __syncthreads();
  const uint32_t PB_N = (PBLK == 0 || (uint32_t)PBLK > G) ? G : (uint32_t)PBLK;
  const uint32_t pfirst = G - PB_N;
  const uint32_t pidx = s_last - pfirst; // valid only if s_last>=pfirst
  // Under PRANK the arrival counter stops being an election (one block
  // continues, G-1 return) and becomes the arrival half of a row barrier.
  if constexpr (PRANK) {
    if (s_last < pfirst) {
      return;
    }
    if (tx == 0) {
      uint32_t v = __hip_atomic_load((uint32_t *)&rc[RC_ARR], __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT);
      while (v < G) {
        __builtin_amdgcn_s_sleep(2);
        v = __hip_atomic_load((uint32_t *)&rc[RC_ARR], __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    __syncthreads();
  } else if (!s_last) {
    return;
  }

  // ACQUIRE side of the handshake: the cursor reads below are agent-scope
  // atomics, so no fence of our own.

  static_assert(GH_STRIDE % 4u == 0u, "vectorised reset needs 4 | stride");
  // Under PRANK the reset is spread over the row's blocks instead; the
  // barrier above guarantees every block has finished reading it.
  uint4 *__restrict__ ghw4 = (uint4 *)(ghist + (size_t)row * GH_STRIDE);
  const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
  for (uint32_t i = PRANK ? pidx * BS + tx : tx; i < GH_STRIDE / 4u;
       i += PRANK ? PB_N * BS : BS) {
    ghw4[i] = z4;
  }

  const uint32_t above = (uint32_t)__hip_atomic_load(
      &rc[RC_WIN], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  const uint32_t nraw = (uint32_t)__hip_atomic_load(
      &rc[RC_CAND], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // The candidate prefetch is issued alongside the two cursor reads rather
  // than behind them.  Unconditionally in bounds (cap >= TOPK >= BS);
  // entries at or past n are discarded.
  const int32_t pre_slot = ld_idx(&cand_idx[(size_t)row * cap + tx]);
  const float pre_val = ld_val(&cand_val[(size_t)row * cap + tx]);
  __syncthreads(); // every thread has read them before they are cleared
  if (!PRANK && tx == 0) {
    rc[RC_WIN] = 0;
    rc[RC_CAND] = 0;
    rc[RC_ARR] = 0;
  }
  refine_row<PRANK>(row, above, nraw, cap, cand_idx, cand_val, o, s_radix,
                    s_grp, s_key, s_slot, &s_thr, &s_above, &s_emit, true,
                    pre_slot, pre_val, pidx, PB_N);
  // Departure half of the row barrier: the block that finishes first cannot
  // reset the counters while the other G-1 are still reading them, so the
  // last one out does it.
  if constexpr (PRANK) {
    __syncthreads();
    if (tx == 0) {
      const uint32_t d =
          __hip_atomic_fetch_add((uint32_t *)&rc[RC_DEP], 1u, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
      if (d + 1u == PB_N) {
        rc[RC_WIN] = 0;
        rc[RC_CAND] = 0;
        rc[RC_ARR] = 0;
        rc[RC_DEP] = 0;
      }
    }
  }
}

} // namespace dsa_topk

// PRANK liveness guard.
static bool prank_resident_ok(const void *fn, int blocks_per_row, int rows) {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) {
    return false;
  }
  // Both queries are per-device constants but hipGetDeviceProperties is a
  // heavyweight host call, and this runs once per launch -- once per DSA layer
  // on any path that is not inside a captured graph.  Cache per device.
  static constexpr int kMaxDev = 16;
  static int s_cu[kMaxDev] = {0};
  if (dev < 0 || dev >= kMaxDev) {
    return false;
  }
  if (s_cu[dev] == 0) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
      return false;
    }
    s_cu[dev] = prop.multiProcessorCount;
  }
  int per_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &per_cu, fn, (int)dsa_topk::BS, 0) != hipSuccess) {
    return false;
  }
  if (per_cu <= 0) {
    return false;
  }
  // Conservative on purpose. Only the last PBLK blocks of a row spin (the
  // earlier G - PBLK increment the arrival counter and return, freeing their
  // slots), so the barrier needs PBLK * rows co-resident, not G * rows. The
  // 4x margin at PBLK=16, G=64 is what absorbs the occupancy figure being a
  // whole-device theoretical maximum that ignores concurrently resident work.
  const long long slots = (long long)per_cu * (long long)s_cu[dev];
  return (long long)blocks_per_row * (long long)rows <= slots;
}

// host entry

// PTMODE 2 stages a per-block page-table window in LDS.  Page-table strides
// too wide for that fixed window use the numerically identical PTMODE 0
// global-gather specialization instead.
void topk_transform(torch::Tensor logits, torch::Tensor row_ends,
                    torch::Tensor page_table, torch::Tensor out,
                    torch::Tensor ghist, torch::Tensor cursor,
                    torch::Tensor cand_idx, torch::Tensor cand_val,
                    int64_t g_per_row, int64_t page_size) {
  const int64_t R = logits.size(0);
  const int64_t L = logits.stride(0);
  const int64_t PTS = page_table.stride(0);
  const int64_t cap = cand_idx.size(1);
  const uint32_t G = (uint32_t)g_per_row;
  // page_size must be 1 or a power of two; both give the identical mapping.
  uint32_t PB = 0, PM = 0;
  for (int64_t ps = page_size; ps > 1; ps >>= 1) {
    ++PB;
  }
  PM = (page_size > 1) ? (uint32_t)(page_size - 1) : 0u;
  TORCH_CHECK((page_size & (page_size - 1)) == 0 && page_size >= 1,
              "page_size must be a power of two, got ", page_size);

  // The histogram width is a compile-time property of the kernel.  Getting it
  // wrong on the host silently walks off the end of ghist and corrupts the
  // buffers next to it, which is exactly what happened once already.
  TORCH_CHECK(ghist.numel() == R * (int64_t)dsa_topk::GH_STRIDE,
              "ghist must be [rows, ", dsa_topk::GH_STRIDE, "], got ",
              ghist.numel());
  // TOPK is compiled in and every row write is `out + row * TOPK`, so a
  // narrower or shorter `out` overruns each row in turn.  The rest of these
  // bound the tensors the kernel indexes by blockIdx.y or by `cap`; each one
  // holds today only because fused_decode.py happens to allocate it that way.
  TORCH_CHECK(out.dim() == 2 && out.size(0) >= R &&
                  out.size(1) == (int64_t)dsa_topk::TOPK,
              "out must be [>=rows, ", dsa_topk::TOPK, "], got ", out.sizes());
  TORCH_CHECK(
      row_ends.scalar_type() == at::kInt && row_ends.is_contiguous() &&
          row_ends.numel() >= R,
      "row_ends must be a contiguous int32 tensor with one entry per row");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) >= R,
              "page_table must be [>=rows, width], got ", page_table.sizes());
  TORCH_CHECK(cand_val.dim() == 2 && cand_val.size(0) >= R &&
                  cand_val.size(1) >= cap,
              "cand_val must be at least as wide as cand_idx (", cap, ")");
  TORCH_CHECK(cand_idx.size(0) >= R, "cand_idx must have one row per row");
  // scan_slice reads the row base as float4, so the row stride has to keep
  // that base 16B-aligned; the slice starts are multiples of 4 by construction.
  TORCH_CHECK(
      logits.stride(1) == 1 && L % 4 == 0,
      "logits rows must be unit-stride with a stride(0) divisible by 4, got ",
      L);
  // The cursor carries the same zero-in/zero-out invariant as ghist, and the
  // consequence of getting it wrong is worse: under PRANK a stale arrival
  // counter hangs the row barrier rather than returning a wrong answer.
  TORCH_CHECK(cursor.numel() == R * (int64_t)dsa_topk::RC_STRIDE,
              "cursor must be [rows, ", dsa_topk::RC_STRIDE, "], got ",
              cursor.numel(), " elements for ", R, " rows");
  TORCH_CHECK(cap >= (int64_t)dsa_topk::TOPK,
              "candidate capacity must be >= TopK");
  TORCH_CHECK(logits.scalar_type() == at::kFloat &&
                  out.scalar_type() == at::kInt,
              "dtype");
  // PTMODE 2 is faster when its per-block page-table slice fits in LDS.  PTS is
  // a static graph property while row_ends is replay-time data, so use PTS as
  // the conservative upper bound.  PTMODE 0 has no page-table-width limit.
  const bool use_pt_window =
      (PTS + (int64_t)G - 1) / (int64_t)G + 2 <= (int64_t)dsa_topk::PT_WIN;

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)G, (unsigned)R, 1);
  dim3 blk(dsa_topk::BS, 1, 1);
#define SCATTER_ARGS                                                           \
  grid, blk, 0, stream, logits.data_ptr<float>(),                              \
      row_ends.data_ptr<int32_t>(), page_table.data_ptr<int32_t>(),            \
      out.data_ptr<int32_t>(), (uint32_t *)ghist.data_ptr<int32_t>(),          \
      cursor.data_ptr<int32_t>(), cand_idx.data_ptr<int32_t>(),                \
      cand_val.data_ptr<float>(), L, PTS, PB, PM, (uint32_t)cap, G

  // The persistent-rank tail finishes the exact rank behind a row barrier, so
  // its blocks must be co-resident.  When they would not be, the ordinary
  // kernel-boundary form is launched instead and the selector cannot hang.
  if (use_pt_window &&
      prank_resident_ok((const void *)dsa_topk::k_scatter<2, true, 16>, (int)G,
                        (int)R)) {
    hipLaunchKernelGGL((dsa_topk::k_scatter<2, true, 16>), SCATTER_ARGS);
  } else if (use_pt_window) {
    hipLaunchKernelGGL((dsa_topk::k_scatter<2>), SCATTER_ARGS);
  } else if (prank_resident_ok((const void *)dsa_topk::k_scatter<0, true, 16>,
                               (int)G, (int)R)) {
    hipLaunchKernelGGL((dsa_topk::k_scatter<0, true, 16>), SCATTER_ARGS);
  } else {
    hipLaunchKernelGGL((dsa_topk::k_scatter<0>), SCATTER_ARGS);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef SCATTER_ARGS
}

// Row stride of the ghist workspace = fine bins + the coarse summary.
// The workspace MUST be sized on this, not on hist_bins(): getting it wrong
// walks off the end of ghist into whatever is allocated next.
int64_t hist_stride() { return (int64_t)dsa_topk::GH_STRIDE; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_transform", &topk_transform, "phase D top-k + page transform",
        py::arg("logits"), py::arg("row_ends"), py::arg("page_table"),
        py::arg("out"), py::arg("ghist"), py::arg("cursor"),
        py::arg("cand_idx"), py::arg("cand_val"), py::arg("g_per_row"),
        py::arg("page_size"));
  m.def("hist_stride", &hist_stride,
        "ghist row stride (fine bins + coarse summary)");
}
