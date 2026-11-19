// bf16 specialization of the candidate top-k: one CTA per row, at most 12288 bf16
// candidates (all >= the row's threshold T) -> k key indices, exact over the bf16 values;
// which copies of the k-th value are taken is unspecified.
//
// The scheme of the fp32 kernel in the bf16 domain: y = score - T computed as bf16x2 (two elements
// per instruction, rounded to nearest, so monotone in the score); the high bits of y (exponent and
// 5 or 6 mantissa bits) over the window [2^-16, 2^16), clamped, index a 1024- or 2048-bin
// histogram; a block prefix from the top finds the bin holding rank k; a second pass compares y
// against the bin's two bf16 edges, emits the higher bins and collects the rank-k bin. The counters
// are 32-bit and count by one: bf16 rows put many lanes of a warp into one bin, and only
// `red.add 1` merges same-address lanes.
//
// The rank-k bin is buffered as (order key of the score) << 16 | (0xFFFF - position) and ranked by
// one warp ballot per tie up to 64 ties; up to 256 ties, a bin of one value takes any `quota` of
// them and otherwise a CTA radix select over the 16-bit order keys (two rounds) finds the k-th
// value. A wider bin (an exact tie of > 256 copies at large scores) ranks its first 256 elements; the
// part of the quota beyond them is emitted as -1.
//
// The whole row (24 KB) stays resident in three shared-memory stages, filled by bulk TMA. T = -inf
// rows (short rows that keep every key) use the row minimum as T.
//
// NaN scores are excluded by construction (the indexer emits finite logits >= T).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "topk_candidates_common.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {

struct TopKCandidatesBF16Params {
  const uint32_t* __restrict__ scores;  // [rows, cap] bf16, viewed as packed pairs
  const int32_t* __restrict__ indices;  // [rows, cap]
  const int32_t* __restrict__ count;
  const float* __restrict__ thr;
  const int32_t* __restrict__ out_offsets;  // nullable: added to every emitted index
  int32_t* __restrict__ out;
  int64_t scores_stride;  // in packed pairs
  int64_t indices_stride;
  int64_t out_stride;
  uint32_t cap;  // slab width: a row's count past it is clipped (the indexer keeps counting on overflow)
};

// The bf16 candidate top-k of one row: one CTA per row, `run` is the kernel body.
// kTopK_ Selected elements per row (512 / 1024 / 2048).
template <uint32_t kTopK_>
struct TopKCandidatesBF16Impl {
  static constexpr uint32_t kTopK = kTopK_;
  static constexpr uint32_t kBlockSize = 256;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kMaxCands = 12288;
  static constexpr uint32_t kStageElems = 4096;
  static constexpr uint32_t kNumStages = kMaxCands / kStageElems;  // the whole row stays resident
  static constexpr uint32_t kStageBytes = kStageElems * 2;
  static constexpr uint32_t kVecElems = 8;                                   // bf16 per 16 B vector
  static constexpr uint32_t kGroups = kStageElems / kVecElems / kBlockSize;  // vectors per thread per stage
  static constexpr uint32_t kElemsPerThread = kGroups * kVecElems;           // per stage
  // bins: (bf16 bits of y) >> kBinShift, i.e. exponent and 7 - kBinShift mantissa bits, minus the
  // window base 2^-16; 32 octaves. 1024 bins for k <= 1024 (the smaller histogram is what fits 7
  // CTAs per SM there; measured 7-11% faster than 2048 bins at 6); 2048 for k = 2048, whose
  // 8 KB of selected ids hold it at 6 CTAs anyway and whose bf16 rows have wide ties
  static constexpr uint32_t kBinShift = kTopK == 2048 ? 1 : 2;
  static constexpr uint32_t kNumBins = 32u << (7 - kBinShift);
  static constexpr uint32_t kBinBase = (127 - 16) << (7 - kBinShift);
  static constexpr uint32_t kTieCap = 256;  // one tie key per thread in radix_ties
  static_assert(kTopK == 512 || kTopK == 1024 || kTopK == 2048);
  static_assert(kElemsPerThread <= 16);  // one 16-bit mask per class per stage
  static_assert(kMaxCands < 0x10000);    // every counter packs in 16 bits, positions in 16 bits

  struct Smem {
    union {
      alignas(16) uint32_t hist[kNumBins + 4];  // + a junk bin (kNumBins) for the invalid tail lanes
      struct {
        int32_t ids[kTopK];         // selected positions in the row
        uint32_t tie_key[kTieCap];  // composite keys of the rank-k bin: higher = taken first
      } sel;
    };
    union {
      uint32_t warp_sums[kNumWarps];
      float warp_min[kNumWarps];  // T = -inf rows: the warps' minima
    };
    float warp_max[kNumWarps];  // T = -inf rows: the warps' maxima
    uint32_t pivot_bin;
    uint32_t count_above;
    uint32_t emit_counter;  // packed (ties << 16 | above-pivot)
    float row_min, row_max;
    alignas(8) uint64_t full_bar[kNumStages];
    union alignas(128) {
      uint32_t data[kNumStages][kStageElems / 2];  // packed bf16 pairs
      struct {                                     // radix_ties scratch: the stage buffers are dead after pass 2
        uint32_t hist[2][256];
        uint32_t match[3];  // bin, count above it, count in it
        uint32_t counter, eq_counter;
        uint32_t value_min[kNumWarps], value_max[kNumWarps];
      } radix;
    };
  };
  static constexpr uint32_t kSmemBytes = sizeof(Smem);
  // the launch-bounds occupancy target: whatever the shared memory allows on a 228 KB SM
  static constexpr uint32_t kOccupancy = std::min(228u * 1024u / (kSmemBytes + 1024u), 2048u / kBlockSize);

  Smem& smem;
  const uint32_t tx, warp_id, lane_id;
  const uint32_t* row;        // packed scores of this row
  uint32_t n;                 // candidates, at most kMaxCands
  float thr;                  // T; the row minimum on T = -inf rows
  uint32_t thr_x2;            // T rounded toward -inf to bf16, twice
  uint32_t num_stages;        // 1 .. 3
  uint32_t pivot;             // the rank-k bin
  uint32_t hi_x2, lo_x2;      // pass 2: bf16 values of the pivot bin's upper and lower edges, twice
  float lin_scale, lin_bias;  // T = -inf rows: bin = (score - min) * kNumBins / (max - min)

  // ---- per-element helpers -----------------------------------------------------------------

  // y = score - T for the two bf16 of `w`, as bf16x2 bits (>= 0 by the slab contract; -0 when
  // score = -0 and T = +0).
  SGL_DEVICE uint32_t y_of(uint32_t w) const {
    uint32_t y;
    asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(y) : "r"(w), "r"(thr_x2));
    return y;
  }

  // The two bins of a bf16x2 y, packed u16x2: (bits >> kBinShift) clamped to the window (masking
  // the sign of a -0 y). The clamp keeps the order: y below 2^-16 (incl. 0) in bin 0, above 2^16 in
  // the top bin.
  SGL_DEVICE static uint32_t bins_of(uint32_t y) {
    constexpr uint32_t kLo = kBinBase << 16 | kBinBase, kHi = (kBinBase + kNumBins - 1) * 0x10001u;
    uint32_t b = (y >> kBinShift) & ((0x7FFFu >> kBinShift) * 0x10001u);
    asm("max.u16x2 %0, %0, %1;\n\tmin.u16x2 %0, %0, %2;" : "+r"(b) : "r"(kLo), "r"(kHi));
    return b - kLo;  // no borrow between the halves: both are >= the base
  }

  // T = -inf rows: the two linear bins of a bf16x2 score over the row's [min, max], packed u16x2.
  SGL_DEVICE uint32_t lin_bins(uint32_t w) const {
    uint32_t lo, hi;
    asm("cvt.rzi.sat.u32.f32 %0, %1;" : "=r"(lo) : "f"(fmaf(__uint_as_float(w << 16), lin_scale, lin_bias)));
    asm("cvt.rzi.sat.u32.f32 %0, %1;" : "=r"(hi) : "f"(fmaf(__uint_as_float(w & 0xFFFF0000u), lin_scale, lin_bias)));
    return min(lo, kNumBins - 1) | min(hi, kNumBins - 1) << 16;
  }

  // Per-half 0xFFFF / 0x0000 mask of `a >= b` on packed bf16x2.
  SGL_DEVICE static uint32_t set_ge_bf16x2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("set.ge.u32.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
  }

  // One hit bit per element of an 8-wide vector from its four per-half masks.
  SGL_DEVICE static uint32_t pack_hits8(uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
    // one flag byte per element (0xFF / 0x00), then signed dot products turn them into bits
    const auto lo = __byte_perm(m0, m1, 0x7531);
    const auto hi = __byte_perm(m2, m3, 0x7531);
    const auto nib = __dp4a(static_cast<int>(lo), static_cast<int>(0xF8FCFEFFu), 0);                 // -1,-2,-4,-8
    return static_cast<uint32_t>(__dp4a(static_cast<int>(hi), static_cast<int>(0x80C0E0F0u), nib));  // -16..-128
  }

  // Composite rank key of the bf16 `raw` at row position `pos`: the order-preserving key of the
  // value (negatives below positives, -0 keyed as +0) above the inverted position.
  SGL_DEVICE static uint32_t composite_key(uint32_t raw, uint32_t pos) {
    raw = raw == 0x8000u ? 0u : raw;
    const uint32_t key = raw & 0x8000u ? ~raw & 0xFFFFu : raw | 0x8000u;
    return key << 16 | (0xFFFFu - pos);
  }

  // ---- stages ------------------------------------------------------------------------------
  // Stage s holds row positions s * kStageElems + [0, stage_len(s)). In a buffer, thread `tx`
  // owns `kGroups` vectors of 8: vector g is the 16 B vector g * kBlockSize + tx.

  SGL_DEVICE uint32_t stage_len(uint32_t s) const {
    return min(n - s * kStageElems, kStageElems);
  }

  // Bulk-copy stage `s` into its buffer. Issuer lane only.
  SGL_DEVICE void load_stage(uint32_t s) {
    const uint32_t bytes = (stage_len(s) * 2 + 15) / 16 * 16;  // the row has room past `count`
    device::ptx::mbar_arrive_expect_tx(&smem.full_bar[s], bytes);
    device::ptx::cp_async_bulk_g2s(smem.data[s], row + s * kStageElems / 2, bytes, &smem.full_bar[s]);
  }

  // Pass 1 over stage `s`: count it into the histogram (kLinear: the T = -inf rows' linear bins).
  template <bool kLinear>
  SGL_DEVICE void hist_stage(uint32_t s) {
    const uint32_t n_s = stage_len(s);
#pragma unroll
    for (uint32_t g = 0; g < kGroups; ++g) {
      const uint32_t pos = kVecElems * (g * kBlockSize + tx);
      if (pos >= n_s) break;
      device::AlignedVector<uint32_t, 4> v;
      v.load(smem.data[s], g * kBlockSize + tx);
      const uint32_t nv = min(n_s - pos, kVecElems);
#pragma unroll
      for (uint32_t w = 0; w < 4; ++w) {
        // elements past the row count into the junk bin: an unconditional +1 stays a merging
        // atomic, a guarded one becomes a branch per element
        const uint32_t b = kLinear ? lin_bins(v[w]) : bins_of(y_of(v[w]));
        atomicAdd(&smem.hist[2 * w < nv ? b & 0xFFFFu : kNumBins], 1u);
        atomicAdd(&smem.hist[2 * w + 1 < nv ? b >> 16 : kNumBins], 1u);
      }
    }
  }

  // Pass 2 over stage `s`: one 16-bit hit mask per thread per class (bit 8g + j: element j of
  // vector g) -> packed warp scan -> one shared atomic per warp (above-pivot | ties << 16); slots
  // land in arrival order. kLinear: the classes come from the linear bins against the pivot.
  template <bool kLinear>
  SGL_DEVICE void emit_stage(uint32_t s) {
    const uint32_t n_s = stage_len(s);
    uint32_t gt_mask = 0, ge_mask = 0;
#pragma unroll
    for (uint32_t g = 0; g < kGroups; ++g) {
      const uint32_t pos = kVecElems * (g * kBlockSize + tx);
      if (pos >= n_s) break;
      device::AlignedVector<uint32_t, 4> v;
      v.load(smem.data[s], g * kBlockSize + tx);
      uint32_t gt, ge;
      if constexpr (kLinear) {
        const uint32_t pivot_x2 = pivot * 0x10001u;
        uint32_t b[4];
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) {
          b[w] = lin_bins(v[w]);
        }
        gt = pack_hits8(
            __vcmpgtu2(b[0], pivot_x2),
            __vcmpgtu2(b[1], pivot_x2),
            __vcmpgtu2(b[2], pivot_x2),
            __vcmpgtu2(b[3], pivot_x2));
        ge = pack_hits8(
            __vcmpgeu2(b[0], pivot_x2),
            __vcmpgeu2(b[1], pivot_x2),
            __vcmpgeu2(b[2], pivot_x2),
            __vcmpgeu2(b[3], pivot_x2));
      } else {
        uint32_t y[4];
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) {
          y[w] = y_of(v[w]);
        }
        gt = pack_hits8(
            set_ge_bf16x2(y[0], hi_x2),
            set_ge_bf16x2(y[1], hi_x2),
            set_ge_bf16x2(y[2], hi_x2),
            set_ge_bf16x2(y[3], hi_x2));
        ge = pack_hits8(
            set_ge_bf16x2(y[0], lo_x2),
            set_ge_bf16x2(y[1], lo_x2),
            set_ge_bf16x2(y[2], lo_x2),
            set_ge_bf16x2(y[3], lo_x2));
      }
      const uint32_t valid = (1u << min(n_s - pos, kVecElems)) - 1;
      gt_mask |= (gt & valid) << (kVecElems * g);
      ge_mask |= (ge & valid) << (kVecElems * g);
    }
    uint32_t eq_mask = ge_mask & ~gt_mask;
    const uint32_t before =
        topk_cands::warp_reserve(__popc(gt_mask) | __popc(eq_mask) << 16, &smem.emit_counter, lane_id);
    uint32_t gt_slot = before & 0xFFFFu;
    uint32_t eq_slot = before >> 16;
    const uint32_t base = s * kStageElems;
    // above the pivot (dense per vector): predicated stores at compile-time positions, a one-add
    // running slot; a vector without hits (often a whole warp's, past the row) is skipped
#pragma unroll
    for (uint32_t g = 0; g < kGroups; ++g) {
      if (((gt_mask >> (kVecElems * g)) & 0xFFu) == 0) continue;
#pragma unroll
      for (uint32_t j = 0; j < kVecElems; ++j) {
        const bool hit = (gt_mask >> (kVecElems * g + j)) & 1u;
        if (hit) smem.sel.ids[gt_slot] = base + kVecElems * (g * kBlockSize + tx) + j;
        gt_slot += hit ? 1u : 0u;
      }
    }
    // the pivot bin (sparse)
    for (; eq_mask != 0; eq_mask &= eq_mask - 1, ++eq_slot) {
      const uint32_t e = __ffs(eq_mask) - 1;
      const uint32_t i = kVecElems * ((e / kVecElems) * kBlockSize + tx) + e % kVecElems;
      const uint32_t raw = (smem.data[s][i / 2] >> (16 * (i % 2))) & 0xFFFFu;
      const uint32_t key = composite_key(raw, base + i);
      if (eq_slot < kTieCap) smem.sel.tie_key[eq_slot] = key;
    }
  }

  // ---- selection ---------------------------------------------------------------------------

  // Prefix from the top over the histogram: lane L of warp w sums chunks j * 32 + L of
  // the warp's range (4 bins each, every LDS.128 of a warp contiguous), one packed shuffle scan
  // gives the lane suffixes, one barrier exchanges the warp totals, and the lane whose chunk holds
  // rank k re-reads its 4 bins. Publishes `pivot_bin` and `count_above`; whole CTA.
  SGL_DEVICE void find_pivot() {
    using vec_t = device::AlignedVector<uint32_t, 4>;
    constexpr uint32_t kVecsPerWarp = kNumBins / 4 / kNumWarps;
    constexpr uint32_t kChunks = kVecsPerWarp / 32;  // per lane: 2 or 1
    static_assert(kChunks == 1 || kChunks == 2);
    uint32_t local[kChunks], excl[kChunks], total[kChunks];
#pragma unroll
    for (uint32_t j = 0; j < kChunks; ++j) {
      vec_t h;
      h.load(smem.hist, warp_id * kVecsPerWarp + j * 32 + lane_id);
      local[j] = h[0] + h[1] + h[2] + h[3];
    }
    // all partial sums are < 2^16 (one row), so two chunks scan in one word
    const uint32_t packed = local[0] | (kChunks > 1 ? local[kChunks - 1] << 16 : 0u);
    const uint32_t inc = topk_cands::suffix_sum(packed);  // lanes >= L
    const uint32_t tot = __shfl_sync(device::warp::kFullMask, inc, 0);
    excl[0] = (inc - packed) & 0xFFFFu, total[0] = tot & 0xFFFFu;
    if constexpr (kChunks > 1) excl[1] = (inc - packed) >> 16, total[1] = tot >> 16;
    uint32_t warp_total = 0;
#pragma unroll
    for (uint32_t j = 0; j < kChunks; ++j) {
      warp_total += total[j];
    }
    if (lane_id == 0) smem.warp_sums[warp_id] = warp_total;
    __syncthreads();
    uint32_t above = 0;  // everything in the warps above this one
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      if (w > warp_id) above += smem.warp_sums[w];
    }
    // chunk order inside the warp: j descending, then lanes descending; bins inside a chunk
    // descend from word 3 to word 0
#pragma unroll
    for (int32_t j = kChunks - 1; j >= 0; --j) {
      const uint32_t a = above + excl[j];
      if (a < kTopK && kTopK <= a + local[j]) {
        vec_t h;
        const uint32_t vec = warp_id * kVecsPerWarp + j * 32 + lane_id;
        h.load(smem.hist, vec);
        uint32_t suffix = a;
#pragma unroll
        for (int32_t b = 3; b >= 0; --b) {
          if (suffix < kTopK && kTopK <= suffix + h[b]) {
            smem.pivot_bin = 4 * vec + static_cast<uint32_t>(b);
            smem.count_above = suffix;
          }
          suffix += h[b];
        }
      }
      above += total[j];
    }
    __syncthreads();
  }

  // Place the `quota` highest of the first `num` tie keys (num <= 32 * kSlots) at `ids[rank]`,
  // rank = number of higher keys. Lane L holds keys L and L + 32; warp w ranks targets w,
  // w + kNumWarps, ... with one ballot per held key. Padding keys are 0, below every real key.
  template <uint32_t kSlots>
  SGL_DEVICE void rank_ties(int32_t* ids, uint32_t num, uint32_t quota) {
    uint32_t held[kSlots];
#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s) {
      const uint32_t j = s * device::kWarpThreads + lane_id;
      held[s] = j < num ? smem.sel.tie_key[j] : 0;
    }
    constexpr uint32_t kTargets = host::div_ceil(kSlots * device::kWarpThreads, kNumWarps);
#pragma unroll
    for (uint32_t i = 0; i < kTargets; ++i) {
      const uint32_t t = i * kNumWarps + warp_id;
      if (t < num) {  // warp-uniform
        const uint32_t key = smem.sel.tie_key[t];
        uint32_t rank = 0;
#pragma unroll
        for (uint32_t s = 0; s < kSlots; ++s) {
          rank += __popc(__ballot_sync(device::warp::kFullMask, held[s] > key));
        }
        if (lane_id == 0 && rank < quota) ids[rank] = static_cast<int32_t>(0xFFFFu - (key & 0xFFFFu));
      }
    }
  }

  // Place the `quota` highest of `num` tie keys (64 < num <= kTieCap: one per thread) at `ids`,
  // any order and any copies of the k-th value. A bin of one value takes keys [0, quota); otherwise
  // a CTA radix select over the 16-bit order keys (2 rounds of 8 bits) finds the k-th value, the
  // higher keys are all taken and the copies of the k-th value claim the remaining slots. Whole CTA.
  SGL_DEVICE void radix_ties(int32_t* ids, uint32_t num, uint32_t quota) {
    static_assert(kTieCap <= kBlockSize && kBlockSize == 256);  // thread tx owns key tx and bin tx
    auto& r = smem.radix;
    const bool mine = tx < num;
    const uint32_t key = mine ? smem.sel.tie_key[tx] >> 16 : 0;  // the order key of the value
    const int32_t pos = mine ? static_cast<int32_t>(0xFFFFu - (smem.sel.tie_key[tx] & 0xFFFFu)) : 0;
    const uint32_t warp_min = __reduce_min_sync(device::warp::kFullMask, mine ? key : 0xFFFFu);
    const uint32_t warp_max = __reduce_max_sync(device::warp::kFullMask, key);
    if (lane_id == 0) r.value_min[warp_id] = warp_min, r.value_max[warp_id] = warp_max;
    r.hist[0][tx] = 0;
    if (tx == 0) r.counter = 0, r.eq_counter = 0;
    __syncthreads();
    uint32_t lo = 0xFFFFu, hi = 0;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      lo = min(lo, r.value_min[w]), hi = max(hi, r.value_max[w]);
    }
    if (lo == hi) {  // every tie holds the k-th value
      if (tx < quota) ids[tx] = pos;
      return;
    }
    bool active = mine;
    uint32_t remain = quota, total = num;
#pragma unroll
    for (uint32_t round = 0; round < 2; ++round) {
      const uint32_t shift = 8 - 8 * round;
      uint32_t* hist = r.hist[round];
      if (active) atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
      if (round == 0) r.hist[1][tx] = 0;
      __syncthreads();
      // bin tx: inclusive prefix from bin 0, so `total - prefix` keys sit in higher bins
      const uint32_t cnt = hist[tx];
      const uint32_t inc = topk_cands::prefix_sum(cnt);
      if (lane_id == device::kWarpThreads - 1) smem.warp_sums[warp_id] = inc;
      __syncthreads();
      uint32_t prefix = inc;
      for (uint32_t w = 0; w < warp_id; ++w) {
        prefix += smem.warp_sums[w];
      }
      const uint32_t above = total - prefix;
      if (above < remain && above + cnt >= remain) r.match[0] = tx, r.match[1] = above, r.match[2] = cnt;
      __syncthreads();
      const uint32_t bin = r.match[0];
      remain -= r.match[1];
      total = r.match[2];
      if (active) {
        const uint32_t b = (key >> shift) & 0xFFu;
        if (b > bin) ids[atomicAdd(&r.counter, 1u)] = pos;
        active = b == bin;
      }
    }
    // the survivors hold the k-th value: the first `remain` to claim a slot are taken
    if (active) {
      const uint32_t slot = atomicAdd(&r.eq_counter, 1u);
      if (slot < remain) ids[quota - remain + slot] = pos;
    }
  }

  // Rank-k bin of a row with a finite T (every candidate >= T): log bins of y = score - T, which are finest
  // right above T, where the k-th value of a sampled threshold lies. Publishes the pivot and count_above and
  // collects the pivot bin into the tie buffer.
  SGL_DEVICE void select_log() {
    using namespace device;
    {  // T toward -inf to bf16 (the slab's T normally is a bf16 value already): y stays >= 0
      const uint32_t bits = __float_as_uint(thr);
      const uint32_t t16 = (bits >> 16) + ((bits >> 31) & ((bits & 0xFFFFu) != 0));
      thr_x2 = t16 * 0x10001u;
    }
    // ---- pass 1: histogram of y's high bits ----
#pragma unroll
    for (uint32_t s = 0; s < kNumStages; ++s) {
      if (s >= num_stages) break;
      ptx::mbar_wait_parity(&smem.full_bar[s], 0);
      hist_stage<false>(s);
    }
    __syncthreads();
    find_pivot();
    pivot = smem.pivot_bin;
    // ---- pass 2: emit the bins above the pivot, collect the pivot bin into the tie buffer ----
    // bin >= b <=> y >= the bf16 value with bits (b + kBinBase) << 1, except at the clamped ends:
    // every y is in a bin >= 0, none in a bin > the top
    const uint32_t edge_hi = pivot + 1 < kNumBins ? (pivot + 1 + kBinBase) << kBinShift : 0x7F80u;  // +inf
    const uint32_t edge_lo = pivot > 0 ? (pivot + kBinBase) << kBinShift : 0u;
    hi_x2 = edge_hi * 0x10001u, lo_x2 = edge_lo * 0x10001u;
#pragma unroll
    for (uint32_t s = 0; s < kNumStages; ++s) {
      if (s >= num_stages) break;
      emit_stage<false>(s);
    }
  }

  // Rank-k bin of a T = -inf row (every key, a wide range of values): linear bins over the row's [min, max].
  // A row of one value takes its first kTopK keys and returns false (done); else as select_log.
  SGL_DEVICE void select_linear() {
    using namespace device;
    float lo = INFINITY, hi = -INFINITY;
    for (uint32_t s = 0; s < num_stages; ++s) {
      ptx::mbar_wait_parity(&smem.full_bar[s], 0);
      const uint32_t n_s = stage_len(s);
      for (uint32_t i = tx; i < n_s; i += kBlockSize) {
        const float v = __uint_as_float(((smem.data[s][i / 2] >> (16 * (i % 2))) & 0xFFFFu) << 16);
        lo = fminf(lo, v), hi = fmaxf(hi, v);
      }
    }
    lo = warp::reduce_min(lo), hi = warp::reduce_max(hi);
    if (lane_id == 0) smem.warp_min[warp_id] = lo, smem.warp_max[warp_id] = hi;
    __syncthreads();
    if (warp_id == 0) {
      lo = warp::reduce_min(lane_id < kNumWarps ? smem.warp_min[lane_id] : INFINITY);
      hi = warp::reduce_max(lane_id < kNumWarps ? smem.warp_max[lane_id] : -INFINITY);
      if (lane_id == 0) smem.row_min = lo, smem.row_max = hi;
    }
    __syncthreads();
    lo = smem.row_min, hi = smem.row_max;
    lin_scale = static_cast<float>(kNumBins) / (hi - lo);
    lin_bias = -lo * lin_scale;
#pragma unroll
    for (uint32_t s = 0; s < kNumStages; ++s) {
      if (s >= num_stages) break;
      hist_stage<true>(s);
    }
    __syncthreads();
    find_pivot();
    pivot = smem.pivot_bin;
#pragma unroll
    for (uint32_t s = 0; s < kNumStages; ++s) {
      if (s >= num_stages) break;
      emit_stage<true>(s);
    }
  }

  // ---- the kernel body ---------------------------------------------------------------------

  SGL_DEVICE void run(const TopKCandidatesBF16Params& params) {
    using namespace device;
    const auto bx = blockIdx.x;
    row = params.scores + bx * params.scores_stride;

    // ---- prologue: warp 0 owns the barriers and the bulk copies, the second warpgroup zeroes
    // the histogram ----
    if (warp_id == 0) {
      if (lane_id == 0) {  // one lane keeps init, fence and every bulk copy in program order
#pragma unroll
        for (uint32_t s = 0; s < kNumStages; ++s) {
          ptx::mbar_init(&smem.full_bar[s], 1);
        }
        ptx::fence_proxy_async_shared();
        ptx::mbar_arrive_expect_tx(&smem.full_bar[0], kStageBytes);
        PDLWaitPrimary<true>();
        // stage 0 needs no count: the slab row always holds a full stage
        ptx::cp_async_bulk_g2s(smem.data[0], row, kStageBytes, &smem.full_bar[0]);
      }
    } else if (warp_id == 1) {
      if (lane_id == 0) smem.emit_counter = 0;
    } else if (warp_id >= 4) {
      // 8-byte stores: STS.128 does not reach the full shared-memory store bandwidth
      AlignedVector<uint32_t, 2> zeros;
      zeros.fill(0);
      static_loop<sizeof(smem.hist) / 8, kBlockSize - 128>(tx - 128, [&](uint32_t i) { zeros.store(smem.hist, i); });
    }
    // every thread loads the per-row scalars itself; the loads overlap the barrier below
    PDLWaitPrimary<true>();
    n = min(static_cast<uint32_t>(params.count[bx]), params.cap);
    thr = params.thr[bx];
    const int32_t off = params.out_offsets != nullptr ? params.out_offsets[bx] : 0;
    __syncthreads();

    const auto indices_row = params.indices + bx * params.indices_stride;
    const auto out = params.out + bx * params.out_stride;
    // a T = -inf row holds every key of the row in key order: position t is key t, the index slab is not read
    const bool identity = thr == -INFINITY;
    if (n <= kTopK) {  // every candidate is selected, -1 past the row
      static_loop<kTopK, kBlockSize>(tx, [&](uint32_t t) {
        out[t] = t < n ? (identity ? static_cast<int32_t>(t) : indices_row[t]) + off : -1;  //
      });
      if (tx == 0) {
        ptx::mbar_wait_parity(&smem.full_bar[0], 0);  // no CTA exits with a copy in flight
      }
      return PDLTriggerSecondary<true>();
    }
    num_stages = div_ceil(n, kStageElems);
    __builtin_assume(num_stages == 1 || num_stages == 2 || num_stages == 3);
    const bool is_issuer = warp_id == 0 && lane_id == 0;
    if (is_issuer) {
      for (uint32_t s = 1; s < num_stages; ++s) {
        load_stage(s);
      }
      // the end gathers from the index slab: warm it in L2 meanwhile
      if (!identity) ptx::cp_async_bulk_prefetch_l2(indices_row, (n * 4 + 15) / 16 * 16);
    }

    if (thr == -INFINITY) {
      select_linear();
    } else {
      select_log();
    }

    PDLTriggerSecondary<true>();
    __syncthreads();
    const uint32_t count_above = smem.count_above;
    const uint32_t quota = kTopK - count_above;  // taken from the pivot bin, >= 1

    // ---- rank the rank-k bin by value ----
    // the pivot bin's population, clamped to the tie buffer (a wider bin keeps its first kTieCap)
    const uint32_t num_ties = min(smem.emit_counter >> 16, kTieCap);
    const uint32_t taken = min(quota, num_ties);
    // == kTopK unless the pivot bin overflowed: the slots past it gather a placeholder and store -1
    const uint32_t first_hole = count_above + taken;
    int32_t* const tie_ids = smem.sel.ids + count_above;
    for (uint32_t i = taken + tx; i < quota; i += kBlockSize) {
      tie_ids[i] = 0;
    }
    if (num_ties == taken) {  // every tie is taken
      if (tx < num_ties) tie_ids[tx] = static_cast<int32_t>(0xFFFFu - (smem.sel.tie_key[tx] & 0xFFFFu));
    } else if (num_ties <= kWarpThreads) {
      rank_ties<1>(tie_ids, num_ties, taken);
    } else if (num_ties <= 2 * kWarpThreads) {
      rank_ties<2>(tie_ids, num_ties, taken);
    } else {
      radix_ties(tie_ids, num_ties, taken);
    }
    __syncthreads();

    // ---- gather: quads of selected positions -> key indices (+ offset) -> out ----
    constexpr uint32_t kQuads = kTopK / 4;
    constexpr uint32_t kPerThread = host::div_ceil(kQuads, kBlockSize);
    AlignedVector<int32_t, 4> res[kPerThread];
    // one copy per row kind, so the index loads of a sampled row stay unconditional (all in flight before a store)
    const auto gather = [&](auto identity_t) {
      constexpr bool kIdentity = decltype(identity_t)::value;
#pragma unroll
      for (uint32_t r = 0; r < kPerThread; ++r) {  // every load in flight before the first store
        const uint32_t q4 = r * kBlockSize + tx;
        if (q4 < kQuads) {
          AlignedVector<int32_t, 4> ids;
          ids.load(smem.sel.ids, q4);
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) {
            // unsigned compare: an id outside the row (negative included) never reaches the gather
            const auto id = static_cast<uint32_t>(ids[i]);
            res[r][i] = id < n ? (kIdentity ? static_cast<int32_t>(id) : indices_row[id]) + off : -1;
          }
        }
      }
    };
    if (identity) {
      gather(std::true_type{});
    } else {
      gather(std::false_type{});
    }
#pragma unroll
    for (uint32_t r = 0; r < kPerThread; ++r) {
      const uint32_t q4 = r * kBlockSize + tx;
      if (first_hole < kTopK) {
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          if (4 * q4 + i >= first_hole) res[r][i] = -1;
        }
      }
      if (q4 < kQuads) res[r].store(out, q4);
    }
  }
};

template <typename Impl>
__global__ __launch_bounds__(Impl::kBlockSize, Impl::kOccupancy)  //
    void topk_candidates_bf16_kernel(const __grid_constant__ TopKCandidatesBF16Params params) {
  alignas(128) extern __shared__ uint8_t smem_raw[];
  const uint32_t tx = threadIdx.x;
  Impl impl{
      .smem = *reinterpret_cast<typename Impl::Smem*>(smem_raw),
      .tx = tx,
      .warp_id = device::warp::reduce_max(tx / device::kWarpThreads),
      .lane_id = device::warp::get_lane_id(),
      .row = nullptr,
      .n = 0,
      .thr = 0.0f,
      .thr_x2 = 0,
      .num_stages = 0,
      .pivot = 0,
      .hi_x2 = 0,
      .lo_x2 = 0,
  };
  impl.run(params);
}

template <uint32_t kTopK>
struct TopKCandidatesBF16Kernel {
  using C = TopKCandidatesBF16Impl<kTopK>;
  static constexpr auto kernel = topk_candidates_bf16_kernel<C>;

  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView count,
      const tvm::ffi::TensorView thr,
      const tvm::ffi::Optional<tvm::ffi::TensorView> out_offsets,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto L = SymbolicSize{"cap"};
    auto S = SymbolicSize{"scores_stride"};
    auto SI = SymbolicSize{"indices_stride"};
    auto O = SymbolicSize{"out_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, L})  // scores
        .with_strides({S, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(scores);
    TensorMatcher({R, L})  // indices
        .with_strides({SI, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(indices);
    TensorMatcher({R})  // count
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(count);
    TensorMatcher({R})  // thr
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(thr);
    TensorMatcher({R, kTopK})  // out
        .with_strides({O, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(out);
    // stage 0 is copied whole whatever the count, and every copy is 16 B granular
    CHECK_HOST(L.unwrap() >= C::kStageElems && L.unwrap() <= C::kMaxCands && L.unwrap() % 8 == 0 && S.unwrap() % 8 == 0)
        << "cap must be in [" << C::kStageElems << ", " << C::kMaxCands << "], rows 16-byte aligned";
    auto params = TopKCandidatesBF16Params{
        .scores = static_cast<const uint32_t*>(scores.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .thr = static_cast<const float*>(thr.data_ptr()),
        .out_offsets = nullptr,
        .out = static_cast<int32_t*>(out.data_ptr()),
        .scores_stride = S.unwrap() / 2,
        .indices_stride = SI.unwrap(),
        .out_stride = O.unwrap(),
        .cap = static_cast<uint32_t>(L.unwrap()),
    };
    if (out_offsets.has_value()) {
      TensorMatcher({R})  // out_offsets
          .with_dtype<int32_t>()
          .with_device(device_)
          .verify(out_offsets.value());
      params.out_offsets = static_cast<const int32_t*>(out_offsets.value().data_ptr());
    }
    if (R.unwrap() == 0) return;
    const auto device = device_.unwrap();
    runtime::set_max_dynamic_smem_per_device<kernel>(device, C::kSmemBytes);
    LaunchKernel(static_cast<uint32_t>(R.unwrap()), C::kBlockSize, device, C::kSmemBytes)
        .config({.use_pdl = true})(kernel, params);
  }
};

}  // namespace sglang
