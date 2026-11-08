/// \file topk_impl.cuh
/// \brief DeepSeek-V4 (DSA indexer) top-k implementation classes.
///
/// This header holds ONLY the device-side implementation classes + helpers; the
/// `__global__` kernels and the host dispatcher live in csrc/deepseek_v4/topk_v2.cuh.
///
/// Design notes:
///  - top-k (`topk`) is a *runtime* value (<= kMaxTopK = 2048), never a
///    compile-time constant.
///  - the dispatcher optionally transforms selected raw indices through a page
///    table after the device implementation writes them.
///  - each block reads its own `seq_len` (per-batch ragged lengths) -- the host
///    launches one universal kernel and dispatches per block.
///  - the dispatcher selects cluster size 8 or 16 from the probed occupancy.
///
/// Algorithm: fp16 coarse histogram -> threshold bin -> fp32-boundary collect ->
/// exact radix tie-break, plus (exact modes) an exact-key refinement when the
/// threshold bin overflows the staging buffer (refine_ties).

#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>

#if !defined(USE_ROCM)
// currently only apply cluster for SM90 & SM100, SM120 has poor cluster performance
#define SUPPORT_CLUSTER (SGL_CUDA_ARCH >= 900 && SGL_CUDA_ARCH < 1100)
#else
// AMD doesn't support cluster
#define SUPPORT_CLUSTER false
#endif

#if SUPPORT_CLUSTER
#include <sgl_kernel/math.cuh>

#include <sgl_kernel/ptx/barrier.cuh>

#include <cooperative_groups.h>
#endif

// The exactness contract, chosen once per process (SGLANG_TOPK_V2_MODE -> -DSGL_TOPK_MODE).
#ifdef SGL_TYPE_CHECKING
#define SGL_TOPK_MODE 0
#endif

#ifndef SGL_TOPK_MODE
#error "SGL_TOPK_MODE must be defined: 0 fast, 1 exact, 2 deterministic"
#endif
static_assert(SGL_TOPK_MODE >= 0 && SGL_TOPK_MODE <= 2, "SGL_TOPK_MODE must be 0, 1 or 2");
// Each mode's symbols live in their own inline namespace, so modules built with
// different modes can be loaded into one process without colliding.
#define SGL_TOPK_CONCAT_IMPL(a, b) a##b
#define SGL_TOPK_CONCAT(a, b) SGL_TOPK_CONCAT_IMPL(a, b)
#ifdef SGL_TOPK_SCORE_BF16
#define SGL_TOPK_MODE_NAMESPACE SGL_TOPK_CONCAT(SGL_TOPK_CONCAT(topk_mode_, SGL_TOPK_MODE), _bf16)
#else
#define SGL_TOPK_MODE_NAMESPACE SGL_TOPK_CONCAT(topk_mode_, SGL_TOPK_MODE)
#endif

namespace sglang {

namespace device::topk {

inline namespace SGL_TOPK_MODE_NAMESPACE {

/// What the selection guarantees beyond the above-threshold set.
/// - FAST: a threshold bin with more than kMaxNumTie members keeps an arbitrary subset of them.
/// - EXACT: the selected values equal torch.topk's multiset; ties at the k-th value are arbitrary.
/// - DETERMINISTIC: EXACT, and the selected set is the same on every run.
enum class Exactness : uint32_t { FAST = 0, EXACT = 1, DETERMINISTIC = 2 };
inline constexpr Exactness kExactness = static_cast<Exactness>(SGL_TOPK_MODE);
inline constexpr bool kExact = kExactness != Exactness::FAST;
inline constexpr bool kDeterministic = kExactness == Exactness::DETERMINISTIC;
/// Mode 2's cluster tie keys order equal values by a 20-bit row index; longer rows are rejected.
inline constexpr uint32_t kDeterministicIdxBits = 20;

#ifdef SGL_TOPK_SCORE_BF16
using score_t = bf16_t;
#else
using score_t = float;
#endif

SGL_DEVICE float score_to_float(score_t x) {
  if constexpr (std::is_same_v<score_t, float>) {
    return x;
  } else {
    return __bfloat162float(x);
  }
}

/// Hints that `value` is warp-uniform so it can live in a uniform register. The
/// caller must already guarantee that: on ROCm this is the identity, since the
/// 32-bit mask below covers only half of a 64-lane wavefront and there is no
/// uniform register file to hint at.
template <typename T>
SGL_DEVICE T broadcast(T value, uint32_t src = 0) {
#if defined(USE_ROCM)
  static_cast<void>(src);
  return value;
#else
  return __shfl_sync(0xFFFFFFFF, value, src);
#endif
}

template <typename... Smems>
struct MaxSmem {
  static constexpr size_t kSize = std::max({sizeof(Smems)...});
  static constexpr size_t kAlign = std::max({alignof(Smems)...});
  alignas(kAlign) uint8_t storage[kSize];
};

// ---------------------------------------------------------------------------
// Order-preserving float -> integer key extraction
// ---------------------------------------------------------------------------

SGL_DEVICE uint32_t extract_exact_bin(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/// extract_exact_bin for tie resolution. The deterministic mode orders -0 and +0 as
/// the equal floats they are, so its index tie-break also spans the two zeros.
SGL_DEVICE uint32_t extract_tie_key(float x) {
  if constexpr (kDeterministic) x = x == 0.0f ? 0.0f : x;
  return extract_exact_bin(x);
}

constexpr float padding_value() {
  return std::numeric_limits<float>::quiet_NaN();
}

constexpr float infinity_value() {
  return std::numeric_limits<float>::infinity();
}

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  const auto bits = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(x))) << 16;
  const auto sign = static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31);
  return (bits ^ (sign | 0x80000000u)) >> (32 - kBits);
}

/// fp32 bounds {lo, hi} of coarse bin `bin`: "above" iff v >= hi, "tie" iff v >= lo and
/// not above, agreeing with extract_coarse_bin<kBits> on every non-NaN score. A zero hi
/// moves to the smallest subnormal, since fp32 compares cannot put -0.0 below +0.0. The
/// inf bin holds +inf, so its hi is a NaN (nothing is above it; write the tie test as
/// !(v >= hi)); bins above it hold only NaN and get NaN edges (nothing collected).
template <uint32_t kBits>
SGL_DEVICE float2 coarse_bin_bounds(uint32_t bin) {
  constexpr uint32_t kShift = 16 - kBits;
  static_assert(kShift >= 1, "edge keys must be even for the rounding rule below");
  constexpr uint32_t kNegInfBin = 0x03FFu >> kShift;  // bin holding the -inf key
  constexpr uint32_t kInfBin = 0xFC00u >> kShift;     // bin holding the +inf key
  // ordered16 edge keys, {lo, hi} packed in the {low, high} halves.
  const uint32_t keys = (bin << kShift) | ((bin + 1) << (kShift + 16));
  // ordered16 -> fp16 bits per half: ~key below 0x8000 (negative), key ^ 0x8000 above.
  const auto to_fp32 = [](uint32_t k) {
    const uint32_t bits = k ^ 0x80008000u ^ ((~k >> 15) & 0x00010001u) * 0x7FFFu;
    return cast<fp32x2_t>(*reinterpret_cast<const fp16x2_t*>(&bits));
  };
  const auto [v_lo, v_hi] = to_fp32(keys);
  const auto [u_lo, u_hi] = to_fp32(keys - 0x00010001u);
  // Exact midpoints of adjacent fp16 values; the clamp puts one against inf on the
  // +/-65520 overflow point.
  const auto midpoint = [](float a, float b) { return fminf(fmaxf(0.5f * (a + b), -65520.0f), 65520.0f); };
  const int32_t lo = __float_as_int(midpoint(v_lo, u_lo));
  const int32_t hi = __float_as_int(midpoint(v_hi, u_hi));
  // Ties round to even and edge keys are even, so negative edges sit one ulp above
  // the midpoint. A zero hi steps past +0.
  const float lo_edge = __int_as_float(lo + (lo >> 31));
  const float hi_edge = __int_as_float(hi + (hi >> 31) + (hi == 0));
  const float nan = __int_as_float(0x7F800001);
  return {bin <= kNegInfBin ? -infinity_value() : bin > kInfBin ? nan : lo_edge, bin >= kInfBin ? nan : hi_edge};
}

SGL_DEVICE uint32_t warp_sum_bool(bool pred, uint32_t mask = 0xFFFFFFFF) {
#ifdef USE_ROCM
  // The ballot covers the whole hardware wave, which on wave64 holds two of
  // these 32-lane logical warps, so a plain __popc would report the wave's
  // lower half to both of them. Shift the caller's mask onto this warp's half
  // and count all 64 bits. __lane_id() / kWarpSize is 0 on wave32.
  const uint32_t half = __lane_id() / kWarpSize;
  return __popcll(__ballot(pred) & (static_cast<uint64_t>(mask) << (kWarpSize * half)));
#else
  return __popc(__ballot_sync(mask, pred));
#endif
}

struct alignas(8) TieValue {
  float value;
  uint32_t idx;
  inline static constexpr TieValue invalid() {
    return TieValue{padding_value(), 0xFFFFFFFFu};
  }
};

// ---------------------------------------------------------------------------
// Per-batch problem description + page-table transform sink
// ---------------------------------------------------------------------------

struct TopKProblem {
  const score_t* __restrict__ in;
  int32_t* __restrict__ out;  // page_indices [topk]
  uint32_t topk;
  uint32_t seq_len;
  int32_t bias = 0;
  uint32_t input_start = 0;  // needed by ragged mode

  SGL_DEVICE void emit(uint32_t pos, uint32_t raw_idx) const {
    out[pos] = static_cast<int32_t>(raw_idx) + bias;
  }
};

// ---------------------------------------------------------------------------
// Shared configuration + tie handling (exact radix select on the threshold bin)
// ---------------------------------------------------------------------------

struct TopKConfig {
  static constexpr uint32_t kMaxTopK = 2048;
  static constexpr uint32_t kBlockSize = 1024;
  static constexpr uint32_t kOccupancy = 2;
  static constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;
  // kMaxNumTie must be >= kMaxTopK: the collect pass keeps at most kMaxNumTie
  // threshold-bin candidates, and up to `topk` output slots may have to be
  // filled from them (above_count can be 0, e.g. heavily tied or all-inf
  // scores). A smaller cap leaves slots that handle_tie can only pad, and
  // padded slots inside the first min(seq_len, topk) entries are dereferenced
  // by downstream sparse attention.
  static constexpr uint32_t kMaxNumTie = 2048;
  static constexpr uint32_t kRadixSize = 1 << 8;
  // Radix width of the threshold-bin refinement (refine_ties). Wider
  // than kRadixSize so that ceil(32 / kRefineBits) == 3 rounds cover the whole
  // key: the histogram overlays the tie staging buffer, which that path
  // re-derives anyway, so the extra bins are free. 12 is the widest that fits.
  static constexpr uint32_t kRefineBits = 12;
  static constexpr uint32_t kRefineSize = 1 << kRefineBits;
  static constexpr uint32_t kRefineItems = kRefineSize / kBlockSize;
  static_assert(
      kRefineSize * sizeof(uint32_t) <= kMaxNumTie * sizeof(TieValue),
      "the refinement histogram must fit the tie staging buffer it overlays");
  static_assert(kRefineSize % kBlockSize == 0);
  static constexpr uint32_t kTopKItems = (kMaxTopK + kBlockSize - 1) / kBlockSize;
  // tie candidates owned per thread in the strided handle_tie loops
  static constexpr uint32_t kTieItems = kMaxNumTie / kBlockSize;
  static_assert(kMaxNumTie >= kMaxTopK && kMaxNumTie % kBlockSize == 0 && kBlockSize % kNumWarps == 0);

  struct TieHandleSmem {
    struct MatchBin {
      uint32_t bin;
      uint32_t above_count;
      uint32_t equal_count;
    };
    uint32_t counter;
    uint32_t counter_final;
    MatchBin match;
    uint32_t warp_sum[kNumWarps];
    uint32_t histogram[2][kRadixSize];
  };

  /// Resolve the threshold bin's ties exactly. `base` is the number of strictly
  /// "above" elements already emitted (final output starts at slot `base`);
  /// `topk` here is the number of remaining slots to fill (== global_topk - base).
  SGL_DEVICE static void handle_tie(  //
      const TieValue* tie_buffer,
      const TopKProblem& problem,
      const uint32_t base,
      const uint32_t num_ties,
      const uint32_t topk,
      TieHandleSmem* smem) {
    constexpr auto is_greater = [](const TieValue& a, const TieValue& b) {
      return (a.value > b.value) || (a.value == b.value && a.idx < b.idx);
    };
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);
    static_assert(kNumWarps == kWarpSize);

    if (num_ties <= topk) {
      for (uint32_t t = tx; t < num_ties; t += kBlockSize) {
        problem.emit(base + t, tie_buffer[t].idx);
      }
      for (uint32_t t = num_ties + tx; t < topk; t += kBlockSize) {
        problem.emit(base + t, base + t);
      }
    } else if (num_ties <= kWarpSize) {
      if (lane_id >= num_ties || warp_id >= num_ties) return;  // some threads are idle
      /// NOTE: use long long to avoid mask overflow when num_tie == 32
      const uint32_t mask = (1ull << num_ties) - 1u;
      const auto tie = tie_buffer[lane_id];
      const auto target = tie_buffer[warp_id];
      const auto rank = warp_sum_bool(is_greater(tie, target), mask);
      if (lane_id == 0 && rank < topk) problem.emit(base + rank, target.idx);
    } else if (num_ties <= kWarpSize * 2) {
      // 64 x 64 topk implementation: each thread takes 2 elements
      const auto warp_id_1 = warp_id + kWarpSize;
      const auto lane_id_1 = lane_id + kWarpSize;
      const auto invalid = TieValue::invalid();
      const auto tie_0 = tie_buffer[lane_id];
      const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
      const auto target_0 = tie_buffer[warp_id];
      const auto target_1 = tie_buffer[warp_id_1];
      {  // warp_id < kNumWarps < num_ties: always a valid target
        const auto rank_0 = warp_sum_bool(is_greater(tie_0, target_0));
        const auto rank_1 = warp_sum_bool(is_greater(tie_1, target_0));
        const auto rank = rank_0 + rank_1;
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target_0.idx);
      }
      if (warp_id_1 < num_ties) {
        const auto rank_0 = warp_sum_bool(is_greater(tie_0, target_1));
        const auto rank_1 = warp_sum_bool(is_greater(tie_1, target_1));
        const auto rank = rank_0 + rank_1;
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target_1.idx);
      }
    } else if (num_ties <= kWarpSize * 4) {
      // 128 x 128 topk implementation: each thread takes 4 elements and does local sort + merge
      const auto invalid = TieValue::invalid();
      const TieValue tie[] = {
          tie_buffer[lane_id + 0 * kWarpSize],
          tie_buffer[lane_id + 1 * kWarpSize],
          lane_id + 2 * kWarpSize < num_ties ? tie_buffer[lane_id + 2 * kWarpSize] : invalid,
          lane_id + 3 * kWarpSize < num_ties ? tie_buffer[lane_id + 3 * kWarpSize] : invalid,
      };
      const TieValue target[] = {
          tie_buffer[warp_id + 0 * kWarpSize],
          tie_buffer[warp_id + 1 * kWarpSize],
          tie_buffer[warp_id + 2 * kWarpSize],
          tie_buffer[warp_id + 3 * kWarpSize],
      };
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= 2 && warp_id + i * kWarpSize >= num_ties) break;
        uint32_t rank = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          rank += warp_sum_bool(is_greater(tie[j], target[i]));
        }
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target[i].idx);
      }
    } else if (num_ties <= kBlockSize) {
      // Common case: one candidate per thread.
      return radix_tie_select<1>(tie_buffer, problem, base, num_ties, topk, smem);
    } else {
      // Rare overflow case.
      static_assert(kTieItems == 2);
      return radix_tie_select<2>(tie_buffer, problem, base, num_ties, topk, smem);
    }
  }

  /// Threshold-digit search over the kRefineSize-bin refinement histogram in
  /// `hist`, by the same rule radix_tie_select uses. One thread owns kRefineItems
  /// contiguous bins. Publishes the result in `smem->match`. Block-wide; ends
  /// with a barrier.
  SGL_DEVICE static void refine_find_threshold(  //
      const uint32_t* hist,
      const uint32_t total_active,
      const uint32_t topk_remain,
      TieHandleSmem* smem) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);

    uint32_t counts[kRefineItems];
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      counts[i] = hist[tx * kRefineItems + i];
      local_sum += counts[i];
    }
    const auto warp_inc = warp::inclusive_sum(local_sum, lane_id);
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    __syncthreads();

    // Inclusive prefix over all bins below this thread's first one.
    uint32_t prefix = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
    prefix += warp_inc - local_sum;
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      prefix += counts[i];
      const auto above = total_active - prefix;  // active elements in bins ABOVE this one
      if (above < topk_remain && above + counts[i] >= topk_remain) {
        smem->match = {tx * kRefineItems + i, above, counts[i]};
      }
    }
    __syncthreads();
  }

  /// Emits the `need` smallest of the `count` distinct indices in `list` at slots
  /// [pos, pos + need), in no particular order. Deterministic-mode tie break among
  /// equal values; rare, so out of line. Block-wide.
  __device__ __noinline__ static void select_smallest_indices(
      const uint32_t* list,
      const uint32_t count,
      uint32_t need,
      const TopKProblem problem,
      const uint32_t pos,
      TieHandleSmem* smem) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);
    // Radix select of the largest `top - idx`, 8 bits a round.
    const auto top = problem.seq_len - 1;
    uint32_t ikey[kTieItems];
    bool active[kTieItems];
#pragma unroll
    for (uint32_t i = 0; i < kTieItems; ++i) {
      const auto t = tx + i * kBlockSize;
      active[i] = t < count;
      ikey[i] = active[i] ? top - list[t] : 0u;
    }
    if (tx < kRadixSize) smem->histogram[0][tx] = 0;
    if (tx == kRadixSize) smem->counter = 0;
    __syncthreads();
    uint32_t total = count;
    uint32_t parity = 0;
    const uint32_t bits = top == 0 ? 1u : 32u - __clz(top);
#pragma unroll 1
    for (int shift = (bits - 1) / 8 * 8; shift >= 0; shift -= 8) {
      const auto histogram = smem->histogram[parity];
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        if (active[i]) atomicAdd(&histogram[(ikey[i] >> shift) & 0xFFu], 1);
      }
      if (tx < kRadixSize) smem->histogram[parity ^ 1][tx] = 0;
      parity ^= 1;
      __syncthreads();
      uint32_t hist_val = 0;
      uint32_t warp_inc = 0;
      if (tx < kRadixSize) {
        hist_val = histogram[tx];
        warp_inc = warp::inclusive_sum(hist_val, lane_id);
        if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
      }
      __syncthreads();
      if (tx < kRadixSize) {
        const auto inter = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
        const auto above = total - (inter + warp_inc);
        if (above < need && above + hist_val >= need) smem->match = {tx, above, hist_val};
      }
      __syncthreads();
      const auto [threshold_bin, above_count, equal_count] = smem->match;
      total = equal_count;
      need -= above_count;
      // Indices are distinct, so the last digit leaves a group of exactly `need`.
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        if (!active[i]) continue;
        const uint32_t bin = (ikey[i] >> shift) & 0xFFu;
        if (bin > threshold_bin || (bin == threshold_bin && shift == 0)) {
          problem.emit(pos + atomicAdd(&smem->counter, 1), top - ikey[i]);
          active[i] = false;
        } else if (bin < threshold_bin) {
          active[i] = false;
        }
      }
      if (need == 0 || (equal_count == need)) {
        // the whole remaining group fits
#pragma unroll
        for (uint32_t i = 0; i < kTieItems; ++i) {
          if (active[i]) problem.emit(pos + atomicAdd(&smem->counter, 1), top - ikey[i]);
        }
        return;
      }
    }
  }

  /// Exact radix select over the tie candidates: each thread owns kItems
  /// strided elements (inactive beyond num_ties). Requires
  /// num_ties <= kItems * kBlockSize. In the deterministic mode, an exact-equal
  /// group straddling the last slot is cut by smallest index (extra radix rounds
  /// over the index, only on such rows).
  template <uint32_t kItems>
  SGL_DEVICE static void radix_tie_select(  //
      const TieValue* tie_buffer,
      const TopKProblem& problem,
      const uint32_t base,
      const uint32_t num_ties,
      const uint32_t topk,
      TieHandleSmem* smem) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);

    bool active[kItems];
    uint32_t key[kItems];
    uint32_t idx[kItems];
    uint32_t write_pos[kItems];
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      const auto t = tx + i * kBlockSize;
      active[i] = t < num_ties;
      const auto tie = active[i] ? tie_buffer[t] : TieValue::invalid();
      key[i] = extract_tie_key(tie.value);
      idx[i] = tie.idx;
      write_pos[i] = topk;
    }
    uint32_t topk_remain = topk;
    if (tx < kRadixSize) smem->histogram[0][tx] = 0;
    if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
    __syncthreads();
    uint32_t total_active = num_ties;
    // Deterministic mode: an exact-equal group straddling the last slot is listed here
    // (every thread has read its candidates, so tie_buffer is free) and cut by index.
    const auto list = reinterpret_cast<uint32_t*>(const_cast<TieValue*>(tie_buffer));
    bool straddle = false;

#pragma unroll
    for (int round = 0; round < 4; round++) {
      const uint32_t shift = 24 - round * 8;
      const auto hist_idx = round % 2;
      const auto histogram = smem->histogram[hist_idx];

#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (active[i]) atomicAdd(&histogram[(key[i] >> shift) & 0xFFu], 1);
      }
      if (round < 3 && tx < kRadixSize) {
        smem->histogram[hist_idx ^ 1][tx] = 0;
      }
      __syncthreads();

      uint32_t hist_val = 0;
      uint32_t warp_inc = 0;
      if (tx < kRadixSize) {
        hist_val = histogram[tx];
        warp_inc = warp::inclusive_sum(hist_val, lane_id);
        if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
      }
      __syncthreads();
      if (tx < kRadixSize) {
        const auto inter = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
        const auto prefix = inter + warp_inc;      // inclusive prefix through this bin
        const auto above = total_active - prefix;  // elements in bins ABOVE this one
        if (above < topk_remain && above + hist_val >= topk_remain) {
          smem->match = {tx, above, hist_val};
        }
      }
      __syncthreads();

      const auto [threshold_bin, above_count, equal_count] = smem->match;
      total_active = equal_count;
      topk_remain -= above_count;
      if (round == 3 && kDeterministic) straddle = equal_count > topk_remain;

#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (!active[i]) continue;
        const uint32_t bin = (key[i] >> shift) & 0xFFu;
        if (bin > threshold_bin) {
          write_pos[i] = atomicAdd(&smem->counter, 1);
          active[i] = false;
        } else if (bin < threshold_bin) {
          active[i] = false;
        } else if (round == 3) {
          const auto slot = atomicAdd(&smem->counter_final, 1);
          if (straddle) {
            list[slot] = idx[i];
          } else {
            write_pos[i] = topk - topk_remain + slot;
          }
        }
        // bin == threshold_bin before the last round: stays active
      }

      if (round == 3 || topk_remain == 0) break;
    }

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      if (write_pos[i] < topk) problem.emit(base + write_pos[i], idx[i]);
    }
    if constexpr (kDeterministic) {
      if (straddle) [[unlikely]] {
        __syncthreads();
        const auto pos = base + topk - topk_remain;
        if (total_active <= kWarpSize) {
          // Real rows tie a handful of elements: rank them all-to-all in one warp.
          if (warp_id == 0 && lane_id < total_active) {
            const auto mine = list[lane_id];
            uint32_t rank = 0;
            for (uint32_t j = 0; j < total_active; ++j) {
              rank += list[j] < mine ? 1u : 0u;
            }
            if (rank < topk_remain) problem.emit(pos + rank, mine);
          }
        } else {
          select_smallest_indices(list, total_active, topk_remain, problem, pos, smem);
        }
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Radix base: histogram storage + input iteration + threshold-bin search
// ---------------------------------------------------------------------------

template <uint32_t kHistBits_>
struct TopKRadixBase : TopKConfig {
  static constexpr uint32_t kVecSize = 4;
  static constexpr uint32_t kHistBits = kHistBits_;
  static constexpr uint32_t kHistSize = 1 << kHistBits;
  using vec_t = AlignedVector<float, kVecSize>;

  SGL_DEVICE static void load_scores(vec_t& v, const score_t* in, int64_t vi) {
    if constexpr (std::is_same_v<score_t, float>) {
      v.load(in, vi);
    } else {
      AlignedVector<score_t, kVecSize> raw;
      raw.load(in, vi);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        v[j] = score_to_float(raw[j]);
      }
    }
  }

  struct Smem {
    uint32_t count_eq;
    uint32_t count_gt;
    float v_hi;
    float v_lo;
    uint32_t thr_count;  // coarse-histogram population of the threshold bin
    union {
      uint32_t warp_sum[kNumWarps];
      uint32_t tie_nonzero;
    };
    // The coarse histogram is dead once find_threshold() has published its
    // boundaries, and the tie machinery only comes alive after that, so the two
    // phases overlay. tie_handle and tie_values are live TOGETHER, so they sit
    // side by side inside the overlay, not in a union with each other.
    union {
      alignas(16) uint32_t histogram[kHistSize];
      struct {
        // refine_ties re-derives the candidates instead of trusting
        // what the collect pass staged, so the buffer is free to carry that
        // path's radix histogram until its staging pass writes values back.
        union {
          TieValue tie_values[kMaxNumTie];
          uint32_t refine_hist[kRefineSize];
        };
        TieHandleSmem tie_handle;
      };
    };
  };

 protected:
  /// True when a threshold bin starting at `lower` may skip refine_ties once its
  /// members are proven all +0, since any subset of them is then exact (ROCm only).
  /// The deterministic mode still orders those ties by index, so it never may.
  SGL_DEVICE static bool zero_bin_shortcut([[maybe_unused]] float lower) {
#ifdef USE_ROCM
    return !kDeterministic && __float_as_uint(lower) == 0u;
#else
    return false;
#endif
  }

  template <uint32_t N = 1, typename F>
  SGL_DEVICE static void for_each_input(const score_t* __restrict__ in_, uint32_t seq_len, F&& fn) {
    // Only the one-block paths: the cluster scan measured slower with LDG than LD.E.
    const auto in = N == 1 ? assume_global(in_) : in_;
    constexpr auto kStride = N * kBlockSize;
    const auto tx = threadIdx.x;
    const auto num_full = seq_len / kVecSize;  // fully-in-bounds vectors
    const auto kChunk = 128u;
    // lane | rank | warp
    auto vi = N == 1 ? tx : (tx % kChunk) + blockIdx.y * kChunk + (tx / kChunk) * (N * kChunk);
    if (vi < num_full) {
      vec_t next_vec;
      load_scores(next_vec, in, vi);
#pragma unroll 1
      do {
        const auto cur = next_vec;
        vi += kStride;
        if (vi < num_full) load_scores(next_vec, in, vi);
        const auto base = (vi - kStride) * kVecSize;
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          fn(cur[j], base + j);
        }
      } while (vi < num_full);
    }

    if (vi == num_full) {
      const auto base = vi * kVecSize;
      if (base == seq_len) return;
      vec_t cur;
      load_scores(cur, in, vi);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        if (base + j < seq_len) fn(cur[j], base + j);
      }
    }
  }

  template <uint32_t kSize>
  SGL_DEVICE static void init_histogram(uint32_t (&histogram)[kSize], uint32_t tx) {
    static_assert(kSize % kBlockSize == 0);
    constexpr uint32_t kItems = kSize / kBlockSize;
    AlignedVector<uint32_t, kItems> vec;
    vec.fill(0);
    vec.store(histogram, tx);
  }

  /// Fills the slots after the count_gt "above" ones from the staged ties.
  template <typename Smem>
  SGL_DEVICE static void handle_staged_ties(const TopKProblem& problem, Smem* smem) {
    const auto topk = problem.topk;
    const auto count_gt = smem->count_gt;
    const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
    const auto tie_count = min(smem->count_eq, kMaxNumTie);
    handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
  }

  /// Finds the coarse bin holding the topk-th largest of `seq_len` scores in
  /// smem->histogram and calls fn(bin) on the one thread that owns it. Block-wide.
  template <typename Smem, typename Fn>
  SGL_DEVICE static void find_threshold(const uint32_t topk, const uint32_t seq_len, Smem* smem, Fn fn) {
    const auto tx = threadIdx.x;
    // the histogram's own size: the cluster path may bin finer than this base
    constexpr uint32_t kItems = sizeof(Smem::histogram) / sizeof(uint32_t) / kBlockSize;
    uint32_t local_exc_sum[kItems + 1];
    AlignedVector<uint32_t, kItems> hist_vec;
    hist_vec.load(smem->histogram, tx);

    local_exc_sum[0] = 0;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      local_exc_sum[i + 1] = hist_vec[i] + local_exc_sum[i];
    }

    const auto local_sum = local_exc_sum[kItems];
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);
    const auto warp_inc_sum = warp::inclusive_sum(local_sum, lane_id);
    const auto warp_exc_sum = warp_inc_sum - local_sum;
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc_sum;

    __syncthreads();

    const auto lane_warp_sum = smem->warp_sum[lane_id];
    const auto warp_prefix_sum = warp::reduce_sum(lane_id < warp_id ? lane_warp_sum : 0);
    const auto exc_sum = static_cast<int32_t>(warp_prefix_sum + warp_exc_sum);
    const auto remained = static_cast<int32_t>(seq_len - topk - exc_sum);
    // only 1 lane will execute this
    if (remained >= 0 && remained < static_cast<int32_t>(local_sum)) [[unlikely]] {
      uint32_t target = 0;
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        const auto prev = static_cast<int32_t>(local_exc_sum[i + 0]);
        const auto next = static_cast<int32_t>(local_exc_sum[i + 1]);
        if (remained >= prev && remained < next) target = tx * kItems + i;
      }
      fn(target);
    }

    __syncthreads();
  }

  /// Digits of refine_ties are taken from the key relative to the bin's lower edge:
  /// members of one coarse bin share their top key bits, so absolute digits would
  /// waste a round on them. A +0 edge also admits -0, whose key sits just below
  /// key(+0). Returns {relative key origin, bits of the largest relative key}.
  SGL_DEVICE static uint2 refine_key_range(const float v_lo, const float v_hi) {
    const auto key_lo = extract_exact_bin(v_lo == 0.0f ? -0.0f : v_lo);
    const auto span = extract_exact_bin(v_hi) - key_lo;  // relative keys are < span
    return {key_lo, span <= 1u ? 1u : 32u - __clz(span - 1u)};
  }

  /// Shift of refine_ties' first digit.
  SGL_DEVICE static uint32_t refine_first_shift(const uint32_t key_bits) {
    return key_bits > kRefineBits ? key_bits - kRefineBits : 0u;
  }

  /// Re-stages the tie buffer exactly when the threshold bin overflows it: radix
  /// rounds narrow the bin to one bucket, then a final scan emits what lies above
  /// it and stages the bucket for the caller's handle_tie. In the deterministic mode,
  /// an exact-equal group that still overflows is narrowed by index, smallest first.
  /// `first_round_done`: the caller has already built the first round's histogram
  /// (digit refine_first_shift of the relative tie key) in refine_hist.
  SGL_DEVICE static void refine_ties(  //
      const TopKProblem problem,
      Smem* smem,
      const float v_lo,
      const float v_hi,
      const uint32_t count_eq,
      const bool first_round_done = false) {
    constexpr uint32_t kRefineRounds = (32 + kRefineBits - 1) / kRefineBits;
    const auto topk = problem.topk;
    if (smem->count_gt >= topk) [[unlikely]]
      return;
    const auto tx = threadIdx.x;
    const auto handle = &smem->tie_handle;
    const auto hist = smem->refine_hist;
    // for_each_input's elements with one copy of `fn` and reversed lanes, so this
    // cold scan shares no load address (hence no registers) with the fast path.
    const auto in = assume_global(problem.in);
    const auto scan = [&](auto&& fn) {
      const auto seq_len = problem.seq_len;
      const auto num_vec = (seq_len + kVecSize - 1) / kVecSize;
      uint32_t vi = kBlockSize - 1 - tx;
      vec_t cur;
      if (vi < num_vec) load_scores(cur, in, vi);
#pragma unroll 1
      while (vi < num_vec) {
        const auto next_vi = vi + kBlockSize;
        vec_t next;
        if (next_vi < num_vec) load_scores(next, in, next_vi);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          if (vi * kVecSize + j < seq_len) fn(cur[j], vi * kVecSize + j);
        }
        cur = next;
        vi = next_vi;
      }
    };

    // refine_find_threshold needs 0 < remain <= active; its match rule keeps it.
    uint32_t remain = topk - smem->count_gt;
    uint32_t active = count_eq;
    const auto [key_lo, key_bits] = refine_key_range(v_lo, v_hi);
    // Candidates are [bucket_lo, bucket_lo + bucket_mask] (relative); round 0 takes the whole bin.
    uint32_t bucket_lo = 0;
    uint32_t bucket_mask = ~0u;

#pragma unroll
    for (uint32_t round = 0; round < kRefineRounds; ++round) {
      const uint32_t used = (round + 1) * kRefineBits;
      const uint32_t lo = key_bits > used ? key_bits - used : 0u;
      if (round > 0 || !first_round_done) {
#pragma unroll
        for (uint32_t i = 0; i < kRefineItems; ++i) {
          hist[tx * kRefineItems + i] = 0;
        }
        __syncthreads();
        // bucket_lo is aligned to the previous digit, so (key - bucket_lo) >> lo is this digit.
        scan([&](float val, uint32_t) {
          if (!(val >= v_lo && !(val >= v_hi))) return;
          const auto d = extract_tie_key(val) - key_lo - bucket_lo;
          if (d <= bucket_mask) atomicAdd(&hist[d >> lo], 1);
        });
        __syncthreads();
      }
      refine_find_threshold(hist, active, remain, handle);
      const auto match = handle->match;
      bucket_lo |= match.bin << lo;
      bucket_mask = (1u << lo) - 1u;
      remain -= match.above_count;
      active = match.equal_count;
      if (lo == 0 || active <= kMaxNumTie) break;
    }

    // Deterministic mode, more than kMaxNumTie members of one exact key: narrow them
    // by index too, and stage there.
    if constexpr (kDeterministic) {
      if (active > kMaxNumTie) [[unlikely]] {
        return refine_by_index(problem, smem, v_lo, v_hi, key_lo, bucket_lo, active, remain);
      }
    }

    // Relative keys are < span, so bucket_hi cannot wrap.
    // Staging overwrites the histogram, which refine_find_threshold has consumed.
    const auto bucket_hi = bucket_lo + bucket_mask + 1u;
    if (tx == 0) smem->count_eq = 0;
    __syncthreads();
    scan([&](float val, uint32_t idx) {
      if (!(val >= v_lo && !(val >= v_hi))) return;
      const auto d = extract_tie_key(val) - key_lo;
      if (d >= bucket_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]]
          problem.emit(pos, idx);
      } else if (d >= bucket_lo) {
        const auto slot = atomicAdd(&smem->count_eq, 1);
        if (slot < kMaxNumTie) smem->tie_values[slot] = {val, idx};
      }
    });
    __syncthreads();
  }

  /// Deterministic-mode tail of refine_ties, once its value rounds leave more than
  /// kMaxNumTie scores of one tie key (key_lo + bucket): radix rounds over top - idx
  /// (larger = smaller index) among them, then refine_ties' staging scan with the
  /// equal-key members ordered by index. Rare, so out of line.
  __device__ __noinline__ static void refine_by_index(
      const TopKProblem problem,
      Smem* smem,
      const float v_lo,
      const float v_hi,
      const uint32_t key_lo,
      const uint32_t bucket,
      uint32_t active,
      uint32_t remain) {
    const auto key = key_lo + bucket;
    const auto tx = threadIdx.x;
    const auto hist = smem->refine_hist;
    const auto in = assume_global(problem.in);
    const auto top = problem.seq_len - 1;
    const uint32_t idx_bits = top == 0 ? 1u : 32u - __clz(top);
    const auto num_vec = (problem.seq_len + kVecSize - 1) / kVecSize;
    uint32_t ibucket_lo = 0;
    uint32_t imask = ~0u;
#pragma unroll 1
    for (uint32_t used = kRefineBits;; used += kRefineBits) {
      const uint32_t lo = idx_bits > used ? idx_bits - used : 0u;
#pragma unroll
      for (uint32_t i = 0; i < kRefineItems; ++i) {
        hist[tx * kRefineItems + i] = 0;
      }
      __syncthreads();
#pragma unroll 1
      for (uint32_t vi = kBlockSize - 1 - tx; vi < num_vec; vi += kBlockSize) {
        vec_t cur;
        load_scores(cur, in, vi);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          const auto idx = vi * kVecSize + j;
          const auto val = cur[j];
          if (idx >= problem.seq_len || !(val >= v_lo && !(val >= v_hi)) || extract_tie_key(val) != key) continue;
          const auto e = top - idx - ibucket_lo;
          if (e <= imask) atomicAdd(&hist[e >> lo], 1);
        }
      }
      __syncthreads();
      refine_find_threshold(hist, active, remain, &smem->tie_handle);
      const auto match = smem->tie_handle.match;
      ibucket_lo |= match.bin << lo;
      imask = (1u << lo) - 1u;
      remain -= match.above_count;
      active = match.equal_count;
      if (lo == 0 || active <= kMaxNumTie) break;
    }
    const auto ibucket_hi = ibucket_lo + imask + 1u;
    if (tx == 0) smem->count_eq = 0;
    __syncthreads();
#pragma unroll 1
    for (uint32_t vi = kBlockSize - 1 - tx; vi < num_vec; vi += kBlockSize) {
      vec_t cur;
      load_scores(cur, in, vi);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const auto idx = vi * kVecSize + j;
        const auto val = cur[j];
        if (idx >= problem.seq_len || !(val >= v_lo && !(val >= v_hi))) continue;
        const auto d = extract_tie_key(val) - key_lo;
        const auto ikey = top - idx;
        // Equal-key members with a smaller index than the index bucket count as above it.
        if (d > bucket || (d == bucket && ikey >= ibucket_hi)) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < problem.topk) [[likely]] {
            problem.emit(pos, idx);
          }
        } else if (d == bucket && ikey >= ibucket_lo) {
          const auto slot = atomicAdd(&smem->count_eq, 1);
          if (slot < kMaxNumTie) smem->tie_values[slot] = {val, idx};
        }
      }
    }
    __syncthreads();
  }

  /// refine_ties out of line, for callers whose fast path it would otherwise crowd.
  __device__ __noinline__ static void
  refine_ties_cold(const TopKProblem problem, Smem* smem, const float v_lo, const float v_hi, const uint32_t count_eq) {
    refine_ties(problem, smem, v_lo, v_hi, count_eq);
  }
};

// ---------------------------------------------------------------------------
// Register path: scores stay resident in registers across both passes (read
// once). Templated on kLocalVecs so the caller picks the smallest covering
// kernel -- a larger kLocalVecs raises kMaxSeqLen but its fixed-unrolled loop
// wastes work on shorter sequences.
// ---------------------------------------------------------------------------

template <uint32_t kLocalVecs_>
struct TopKRegister : TopKRadixBase<12> {
  static constexpr uint32_t kLocalVecs = kLocalVecs_;
  static constexpr uint32_t kMaxSeqLen = kBlockSize * kVecSize * kLocalVecs;

  template <bool kUsePDL>
  SGL_DEVICE static void forward(const TopKProblem& problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }

    __syncthreads();
    PDLWaitPrimary<kUsePDL>();
    const auto in = assume_global(problem.in);
    const uint32_t num_full = div_ceil(problem.seq_len, kVecSize);

    // Phase 1: load full vectors + build histogram
    vec_t local_vecs[kLocalVecs];
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi < num_full) load_scores(local_vecs[i], in, vi);
    }

    const auto tail_start = (problem.seq_len - 1) % kVecSize + 1;
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
      if (vi == num_full - 1) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          if (j >= tail_start) local_vecs[i][j] = padding_value();
        }
      }
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(local_vecs[i][j])], 1);
      }
    }
    const auto num_padding = kVecSize - tail_start + problem.input_start;
    if (tx == 0 && num_padding > 0) {
      // Ask the histogram's own binning where the platform's NaN landed.
      atomicSub(&smem->histogram[extract_coarse_bin<kHistBits>(padding_value())], num_padding);
      atomicAdd(&smem->histogram[0], num_padding);
    }
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, num_full * kVecSize, smem, [&](uint32_t threshold_bin) {
      const auto [v_lo, v_hi] = coarse_bin_bounds<kHistBits>(threshold_bin);
      smem->v_hi = v_hi;
      smem->v_lo = v_lo;
    });

    // Phase 3: collect by two fp32 boundaries
    const auto topk = problem.topk;
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    const bool zero_bin = kExact && zero_bin_shortcut(v_lo);
    if (zero_bin) {
      // find_threshold has finished using the overlaid warp sums.
      if (tx == 0) smem->tie_nonzero = 0;
      __syncthreads();
    }

#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      const auto base = vi * kVecSize;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const auto idx = base + j;
        const auto val = local_vecs[i][j];
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            problem.emit(pos, idx);
          }
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (zero_bin && __float_as_uint(val) != 0u) atomicOr(&smem->tie_nonzero, 1u);
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      }
    }

    // Phase 4: Handle ties.
    __syncthreads();
    if constexpr (kExact) {
      if (smem->count_eq > kMaxNumTie && (!zero_bin || smem->tie_nonzero != 0)) [[unlikely]] {
        // Buffer holds an arrival-order subset; re-stage it on the exact key.
        refine_ties_cold(problem, smem, v_lo, v_hi, smem->count_eq);
      }
    }
    handle_staged_ties(problem, smem);
  }
};

// ---------------------------------------------------------------------------
// Streaming path: rows too long for the register path -- two vectorized passes
// over global memory
// ---------------------------------------------------------------------------

struct TopKStreaming : TopKRadixBase<12> {
  /// Collect pass of a row whose threshold bin overflows: emits what lies above the
  /// bin and histograms the bin's members by refine_ties' first digit, then lets
  /// refine_ties finish from its second round. Out of line, so the normal row's
  /// collect pass keeps its register allocation.
  __device__ __noinline__ static void
  collect_overflow(const TopKProblem problem, Smem* smem, const float v_lo, const float v_hi) {
    const auto tx = threadIdx.x;
    const auto topk = problem.topk;
    const auto hist = smem->refine_hist;
    const auto [key_lo, key_bits] = refine_key_range(v_lo, v_hi);
    const auto shift = refine_first_shift(key_bits);
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      hist[tx * kRefineItems + i] = 0;
    }
    __syncthreads();
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]] {
          problem.emit(pos, idx);
        }
      } else if (val >= v_lo) {
        atomicAdd(&smem->count_eq, 1);
        atomicAdd(&hist[(extract_tie_key(val) - key_lo) >> shift], 1);
      }
    });
    __syncthreads();
    refine_ties(problem, smem, v_lo, v_hi, smem->count_eq, /*first_round_done=*/true);
    handle_staged_ties(problem, smem);
  }

  template <bool kUsePDL>
  SGL_DEVICE static void forward(TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }
    __syncthreads();
    PDLWaitPrimary<kUsePDL>();

    // Phase 1: Load and build histogram
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t) {
      const auto bin = extract_coarse_bin<kHistBits>(val);
      atomicAdd(&smem->histogram[bin], 1);
    });
    const auto num_padding = problem.input_start;
    if (tx == 0 && num_padding != 0) {
      atomicSub(&smem->histogram[extract_coarse_bin<kHistBits>(padding_value())], num_padding);
      atomicAdd(&smem->histogram[0], num_padding);
    }
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, problem.seq_len, smem, [&](uint32_t threshold_bin) {
      const auto [v_lo, v_hi] = coarse_bin_bounds<kHistBits>(threshold_bin);
      smem->v_hi = v_hi;
      smem->v_lo = v_lo;
      if constexpr (kExact) smem->thr_count = smem->histogram[threshold_bin];
    });

    // Phase 3: collect by the bin's two fp32 edges (coarse_bin_bounds), which avoids
    // recomputing the fp16 bin of every score.
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    const auto topk = problem.topk;
    const bool zero_bin = kExact && zero_bin_shortcut(v_lo);
    if constexpr (kExact) {
      if (!zero_bin && smem->thr_count > kMaxNumTie) [[unlikely]] {
        // The threshold bin overflows the staging buffer (known from the histogram):
        // the collect pass builds refine_ties' first histogram instead of staging.
        return collect_overflow(problem, smem, v_lo, v_hi);
      }
    }
    if (zero_bin) {
      if (tx == 0) smem->tie_nonzero = 0;
      __syncthreads();
    }
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]] {
          problem.emit(pos, idx);
        }
      } else if (val >= v_lo) {
        const auto pos = atomicAdd(&smem->count_eq, 1);
        if (zero_bin && __float_as_uint(val) != 0u) atomicOr(&smem->tie_nonzero, 1u);
        if (pos < kMaxNumTie) [[likely]] {
          smem->tie_values[pos] = {val, idx};
        }
      }
    });

    // Phase 4: handle ties. The output layout follows the collect counts, not the
    // histogram's, so it stays consistent with the fp32 classification above.
    __syncthreads();
    if constexpr (kExact) {
      if (smem->count_eq > kMaxNumTie && (!zero_bin || smem->tie_nonzero != 0)) [[unlikely]] {
        // See the register path.
        refine_ties(problem, smem, v_lo, v_hi, smem->count_eq);
      }
    }
    handle_staged_ties(problem, smem);
  }
};

// ---------------------------------------------------------------------------
// Cluster path: very long seq_len, small batch. `kClusterSize` blocks cooperate
// on one batch element via distributed shared memory (one cluster per element).
//
// CUDA only: thread-block clusters and distributed shared memory have no CDNA
// equivalent.
//
// An overflowing threshold bin is refined cluster-wide in the exact modes (the
// cold path of TopKCluster), not with the block-local refine_ties: the
// candidates are split across kClusterSize ranks.
// ---------------------------------------------------------------------------

#if SUPPORT_CLUSTER

/// Single-instruction PTX wrappers of the cluster path, on top of sgl_kernel/ptx/barrier.cuh.
namespace ptx {

using device::ptx::mbar_init;
using device::ptx::to_shared;

/// Address of `addr` (a .shared offset) in cluster rank `rank`'s shared memory.
SGL_DEVICE uint32_t map_rank(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
/// Generic pointer to `ptr`'s counterpart in cluster rank `rank`'s shared memory.
template <typename T>
SGL_DEVICE T* map_rank(T* ptr, uint32_t rank) {
  T* r;
  asm volatile("mapa.u64 %0, %1, %2;" : "=l"(r) : "l"(ptr), "r"(rank));
  return r;
}
SGL_DEVICE void st_async_v4(uint32_t dst, uint4 v, uint32_t mbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
               "r"(v.x),
               "r"(v.y),
               "r"(v.z),
               "r"(v.w),
               "r"(mbar)
               : "memory");
}
SGL_DEVICE void st_async_v2(uint32_t dst, uint32_t a, uint32_t b, uint32_t mbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(dst),
               "r"(a),
               "r"(b),
               "r"(mbar)
               : "memory");
}
SGL_DEVICE void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
// The waits below differ from ptx/barrier.cuh: a "memory" clobber, and a cluster-scope acquire
// that pairs with the remote st.async complete_tx. They wait on phase 0 only: after every
// used phase the CTA completes the next one empty (mbar_skip_phase), so each problem
// starts every mbarrier on an even phase.
SGL_DEVICE void mbar_arrive_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(addr), "r"(bytes) : "memory");
}
SGL_DEVICE void mbar_wait(uint32_t addr) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "LAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra LAB_WAIT_%=;\n}\n" ::"r"(addr)
      : "memory");
}
/// Waits for phase 0 of a transaction mbarrier that peers complete with st.async, ordering
/// only the shared::cluster writes it covers: a relaxed wait plus an acquire fence restricted
/// to shared::cluster, so no L1 invalidation (the cluster-scope acquire wait above emits
/// CCTL.IVALL, which drops the cached score lines the collect pass would re-read).
SGL_DEVICE void mbar_wait_shared(uint32_t addr) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "LAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra LAB_WAIT_%=;\n}\n" ::"r"(addr)
      : "memory");
  asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
}
/// Completes the mbarrier's current phase at once (count 1, nothing pending) and observes it,
/// as PTX requires of every phase before the next arrive. Call only after every waiter of
/// the previous phase has passed it.
SGL_DEVICE void mbar_skip_phase(uint32_t addr) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%0], 1;\n}\n" ::"r"(addr)
      : "memory");
}
/// Release of this thread's prior shared::cta accesses at cluster scope (MEMBAR.ALL.CTA, not the
/// GPU-scope membar of a release arrive); with a relaxed barrier arrive it forms a release arrive.
SGL_DEVICE void fence_release_shared_cta() {
  asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
}
SGL_DEVICE void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
SGL_DEVICE void barrier_cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
SGL_DEVICE void barrier_cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
/// Named barrier `id` over `num_threads` threads (a multiple of the warp size).
SGL_DEVICE void bar_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}
SGL_DEVICE void cluster_sync() {
  barrier_cluster_arrive_release();
  barrier_cluster_wait();
}

}  // namespace ptx

/// Cluster path. Every rank builds a coarse histogram of its interleaved chunks,
/// then the threshold bin is found by every rank itself from two all-gathers over
/// DSMEM (st.async + transaction mbarrier, no cluster barrier): first the per-rank
/// sums of 32-bin groups, then the per-rank bins of the selected group. The
/// all-gathered per-rank counts also give every rank its own output and tie
/// bases, so the collect pass emits directly (above -> output, ties -> rank 0's
/// tie buffer through st.async); rank 0 waits for exactly the tie bytes and
/// resolves them. An overflowing threshold bin (more than kMaxNumTie members)
/// is known cluster-wide before the collect pass and takes a cold refine path.
template <uint32_t kClusterSize>
struct TopKCluster : TopKRadixBase<11> {
  static constexpr uint32_t kHistItems = kHistSize / kBlockSize;  // bins per thread
  static constexpr uint32_t kGroupBins = 32;
  static constexpr uint32_t kGroups = kHistSize / kGroupBins;
  static constexpr uint32_t kGroupChunks = kGroups / 4;  // 16-byte chunks of one rank's group sums
  static constexpr uint32_t kBinRow = kGroupBins + 4;    // + this rank's count above the group, padded
  static constexpr uint32_t kBinChunks = kBinRow / 4;
  static constexpr uint32_t kNegZeroBin = 0x7FFFu >> (16 - kHistBits);  // negatives that round to fp16 -0
  using MatchBin = TieHandleSmem::MatchBin;
  static constexpr uint32_t kNumMbars = 5;
  struct Smem {
    // Transaction mbarriers: the group and bin all-gathers, rank 0's tie arrivals,
    // and the fine path's group and bin all-gathers. `mbars` views them for init.
    union {
      uint64_t mbars[kNumMbars];
      struct {
        uint64_t mbar_group;
        uint64_t mbar_bin;
        uint64_t mbar_ties;
        uint64_t mbar_fine_group;
        uint64_t mbar_fine_bin;
      };
    };
    uint32_t count_eq;
    uint32_t count_gt;
    uint32_t local_start_eq;  // refine_cluster: this rank's base in rank 0's tie buffer
    union {
      uint32_t local_start_gt;  // refine_cluster: this rank's base in the output
      uint32_t fine_count_gt;   // fine_path: above-digit members emitted by pass 2
    };
    uint32_t refine_gt;
    uint32_t refine_remain;
    MatchBin refine_match;
    uint32_t warp_sum[kNumWarps];
    alignas(16) uint32_t group_local[kGroups];
    alignas(16) uint32_t recv_group[kClusterSize][kGroups];
    alignas(16) uint32_t recv_bin[kClusterSize][kBinRow];  // group's bins + count above the group
    uint32_t rank_suffix[kClusterSize][2];                 // fine path, per rank: members above / at the digit
    uint32_t sel_group, above_group, own_above_group, sel_bin, above_total, eq_total;
    uint32_t tie_ctr_gt, tie_ctr_eq;  // rank 0's tie select
    MatchBin tie_digit;
    TieValue tie_stage[kWarpSize];
    union {
      alignas(16) uint32_t histogram[kHistSize];
      TieHandleSmem tie_handle;
      int32_t stage_out_idxs[kMaxTopK];  // refine_cluster, ranks > 0: above indices until the base is known
    };
    union {
      TieValue tie_values[kMaxNumTie];
      alignas(16) uint32_t refine_hist[kRefineSize];  // cold paths: digits of the exact key - key_low(v_lo)
    };
    // Phase 2d's results for this rank: bases of its above and tie slots, bin edges.
    uint32_t base_gt, base_eq, my_gt, my_eq;
    float v_lo, v_hi;
    // Fine path: this rank's threshold-bin members, staged by pass 1 for pass 2.
    uint32_t member_count;
    TieValue members[kMaxNumTie];
  };

  /// fp32 edges {lo, hi} of the threshold bin under this path's binning, which keys
  /// -0.0 as +0.0: "above" iff val >= hi, "tie" iff val >= lo and not above, so the
  /// collected counts equal the histogram's on every non-NaN score.
  SGL_DEVICE static float2 bin_bounds(uint32_t bin) {
    auto [v_lo, v_hi] = coarse_bin_bounds<kHistBits>(bin);
    if (bin == kNegZeroBin) v_hi = 0.0f;  // -0.0 is binned as +0.0, i.e. above
    return {v_lo, v_hi};
  }
  /// Origin of the in-bin exact keys. -0.0 and +0.0 compare equal: the range starts
  /// at -0.0 so both sit at or above it.
  SGL_DEVICE static uint32_t key_low(float v_lo) {
    return extract_exact_bin(v_lo == 0.0f ? -0.0f : v_lo);
  }
  /// Bits of the largest in-bin key relative to key_low(v_lo). Kept apart from
  /// refine_key_range: the two differ for spans <= 1 (0 vs 1 bit).
  SGL_DEVICE static uint32_t key_bits(float v_lo, float v_hi) {
    return 32 - __clz(extract_exact_bin(v_hi) - key_low(v_lo) - 1);
  }

  /// st.async of `v` to `ptr`'s counterpart in rank `dst_rank`, completing on that rank's `bar`.
  SGL_DEVICE static void push_v4(const uint32_t* ptr, uint4 v, uint32_t dst_rank, uint64_t* bar) {
    ptx::st_async_v4(ptx::map_rank(ptx::to_shared(ptr), dst_rank), v, ptx::map_rank(ptx::to_shared(bar), dst_rank));
  }

  /// extract_coarse_bin<kHistBits>(x + 0) of two scores, x in the low half of the result:
  /// one packed add and one packed fp16 conversion for both, and a byte permute that
  /// replicates each half's sign over it (the ordered fp16 key is ~h for negatives, h | 0x8000 otherwise).
  SGL_DEVICE static uint2 coarse_bins2(float x, float y) {
#if SGL_CUDA_ARCH >= 1000
    const auto v = math::add_f32x2(fp32x2_t{x, y}, fp32x2_t{0.0f, 0.0f});
#else
    const auto v = fp32x2_t{__fadd_rn(x, 0.0f), __fadd_rn(y, 0.0f)};
#endif
    const auto h = cast<fp16x2_t>(v);
    const uint32_t u = *reinterpret_cast<const uint32_t*>(&h);
    uint32_t neg;
    asm("prmt.b32 %0, %1, 0, 0xBB99;" : "=r"(neg) : "r"(u));
    const uint32_t key = u ^ (neg | 0x80008000u);
    return {(key >> (16 - kHistBits)) & (kHistSize - 1), key >> (32 - kHistBits)};
  }

  static constexpr uint32_t kScanStride = kClusterSize * kBlockSize;  // vectors per cluster round
  /// This thread's first vector of the cluster scan: 128-vector chunks, lane | rank | warp.
  SGL_DEVICE static uint32_t scan_start() {
    constexpr uint32_t kChunk = 128;
    const auto tx = threadIdx.x;
    return (tx % kChunk) + blockIdx.y * kChunk + (tx / kChunk) * (kClusterSize * kChunk);
  }

  /// for_each_input<kClusterSize> by whole vectors: fn_vec(vec, idx of vec[0]) for each of the
  /// first `num_vecs` vectors, a partial last one included (pad_tail).
  template <typename FV>
  SGL_DEVICE static void for_each_vec(const score_t* __restrict__ in, uint32_t num_vecs, FV&& fn_vec) {
    auto vi = scan_start();
    if (vi < num_vecs) {
      vec_t next_vec;
      load_scores(next_vec, in, vi);
#pragma unroll 1
      do {
        const auto cur = next_vec;
        vi += kScanStride;
        if (vi < num_vecs) load_scores(next_vec, in, vi);
        fn_vec(cur, (vi - kScanStride) * kVecSize);
      } while (vi < num_vecs);
    }
  }

  /// Fills the partial last vector of the row with NaN in place, so both scans read whole
  /// vectors: NaN is never collected, and its histogram count is taken back here. Only the
  /// thread that scans that vector writes it, so its own program order covers its later loads.
  /// Needs the row's storage to reach the next multiple of 4 scores (score_stride % 4 == 0).
  SGL_DEVICE static void pad_tail(const TopKProblem& problem, Smem* smem) {
    const auto last = problem.seq_len / kVecSize;
    const auto num_valid = problem.seq_len % kVecSize;
    const auto start = scan_start();
    if (num_valid == 0 || last < start || (last - start) % kScanStride != 0) return;
    const auto tail = const_cast<score_t*>(problem.in) + last * kVecSize;
#pragma unroll
    for (uint32_t j = 1; j < kVecSize; ++j) {  // a runtime trip count here spills on C8
      if (j >= num_valid) tail[j] = static_cast<score_t>(padding_value());
    }
    atomicSub(&smem->histogram[extract_coarse_bin<kHistBits>(padding_value())], kVecSize - num_valid);
  }

  /// The collect pass: fn(val, idx) on every score that may be at or above v_lo. A whole
  /// vector is tested first (its max below v_lo: none is), which most vectors are.
  template <typename F>
  SGL_DEVICE static void collect_scan(const TopKProblem& problem, float v_lo, F&& fn) {
    for_each_vec(problem.in, div_ceil(problem.seq_len, kVecSize), [&](const vec_t& v, uint32_t base) {
      if (fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3])) < v_lo) return;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        fn(v[j], base + j);
      }
    });
  }

  /// Once per CTA, before the first `forward`.
  SGL_DEVICE static void init(void* _smem) {
    const auto smem = static_cast<Smem*>(_smem);
    if (threadIdx.x >= kNumMbars) return;
    ptx::mbar_init(&smem->mbars[threadIdx.x], 1);
    ptx::fence_mbar_init();
  }

  /// Phase 2: the threshold bin across the cluster, from every rank's coarse histogram. Two
  /// st.async all-gathers: the 64 group sums (warp 0 picks the group), then the group's 32 bins
  /// plus each rank's count above it (warp kClusterSize picks the bin). Publishes sel_bin,
  /// above_total, eq_total, this rank's base_gt / base_eq / my_gt / my_eq and v_lo / v_hi in smem.
  SGL_DEVICE static void find_cluster_threshold(Smem* smem, const uint32_t topk) {
    static_assert(kGroups == 64 && kGroups == 2 * kNumWarps && kGroupBins == kWarpSize);
    const auto tx = threadIdx.x;
    const auto lane_id = get_lane_id();
    const auto warp_id = broadcast(tx / kWarpSize);
    const uint32_t this_rank = blockIdx.y;
    // The last thread neither pushes nor searches here: it arms the two mbarriers.
    constexpr uint32_t kIdleThread = kBlockSize - 1;
    static_assert(kClusterSize * std::max(kGroupChunks, kBinChunks) < kIdleThread);
    // Phase 2a: group sums, all-gathered. Warp w sums groups 2w and 2w + 1, one bin of each per
    // lane, with one full-warp redux each; every lane stores the same pair.
    {
      const auto group = 2 * warp_id;
      const auto lo = smem->histogram[(group + 0) * kGroupBins + lane_id];
      const auto hi = smem->histogram[(group + 1) * kGroupBins + lane_id];
      AlignedVector<uint32_t, 2> vec;
      vec[0] = warp::reduce_sum(lo);
      vec[1] = warp::reduce_sum(hi);
      vec.store(&smem->group_local[group]);
    }
    __syncthreads();
    static_assert(kClusterSize * kGroupChunks <= kBlockSize);
    // Whole warps (warp_id is uniform), so the collectives of 2b below need no divergence path.
    static_assert(kClusterSize * kGroupChunks % kWarpSize == 0);
    if (warp_id < kClusterSize * kGroupChunks / kWarpSize) {
      const uint32_t chunk = tx % kGroupChunks, dst_rank = tx / kGroupChunks;
      // Raw uint4 / uint2 views of the DSMEM rows: AlignedVector adds instructions here.
      const auto v = reinterpret_cast<const uint4*>(smem->group_local)[chunk];
      push_v4(&smem->recv_group[this_rank][chunk * 4], v, dst_rank, &smem->mbar_group);

      // Phase 2b: warp 0 finds the group holding the topk-th score; lane l holds groups 2l and 2l + 1.
      if (warp_id == 0) {
        // This rank's own sums of groups 2l and 2l + 1, ready since 2a: loaded ahead of the wait.
        const auto local = reinterpret_cast<const uint2*>(smem->group_local)[lane_id];
        ptx::mbar_wait_shared(ptx::to_shared(&smem->mbar_group));
        uint32_t sum_lo = 0, sum_hi = 0;  // groups 2l and 2l + 1 over the cluster
#pragma unroll
        for (uint32_t r = 0; r < kClusterSize; ++r) {
          const auto v = reinterpret_cast<const uint2*>(smem->recv_group[r])[lane_id];
          sum_lo += v.x;
          sum_hi += v.y;
        }
        const auto incl = warp::inclusive_sum(sum_lo + sum_hi, lane_id);
        const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
        const auto above_hi = total - incl;       // above group 2l + 1: the groups of higher lanes
        const auto above_lo = above_hi + sum_hi;  // above group 2l
        // One lane holds the topk-th score: in group 2l + 1 if above_lo already reaches topk.
        const bool hit = above_hi < topk && above_lo + sum_lo >= topk;
        const bool in_hi = above_lo >= topk;
        // Exactly one lane hits (the counts run from 0 to seq_len > topk), so max-reduxes (CREDUX,
        // uniform results) read its values.
        const auto group = warp::reduce_max(hit ? 2 * lane_id + (in_hi ? 1u : 0u) : 0u);
        const auto group_above = warp::reduce_max(hit ? (in_hi ? above_hi : above_lo) : 0u);
        // This rank's elements in groups above group (2l + 1 > group iff 2l >= group).
        const auto own = warp::reduce_sum((2 * lane_id > group ? local.x : 0u) + (2 * lane_id >= group ? local.y : 0u));
        smem->sel_group = group;  // every lane stores the same values
        smem->above_group = group_above;
        smem->own_above_group = own;
      }
    } else if (tx == kIdleThread) {
      ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar_group), kClusterSize * kGroups * sizeof(uint32_t));
      ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar_bin), kClusterSize * kBinRow * sizeof(uint32_t));
    }

    __syncthreads();

    // Phase 2c: all-gather this rank's bins of the selected group, plus its count above the group.
    static_assert(kClusterSize * kBinChunks <= kBlockSize);
    if (tx < kClusterSize * kBinChunks) {
      const uint32_t chunk = tx % kBinChunks, dst_rank = tx / kBinChunks;
      const auto sel_group = smem->sel_group;
      const auto v = chunk < kGroupBins / 4
                         ? reinterpret_cast<const uint4*>(smem->histogram + sel_group * kGroupBins)[chunk]
                         : uint4{smem->own_above_group, 0, 0, 0};
      push_v4(&smem->recv_bin[this_rank][chunk * 4], v, dst_rank, &smem->mbar_bin);
    }

    // Phase 2d: warp kClusterSize alone finds the threshold bin, this rank's bases and the bin
    // edges; the other warps read them back after the barrier.
    static_assert(kClusterSize < kNumWarps);
    if (warp_id == kClusterSize) {
      ptx::mbar_wait_shared(ptx::to_shared(&smem->mbar_bin));
      uint32_t bin_count = 0, before = 0, mine = 0;  // bin lane_id: all ranks, ranks before this one, this rank
#pragma unroll
      for (uint32_t r = 0; r < kClusterSize; ++r) {
        const auto v = smem->recv_bin[r][lane_id];
        bin_count += v;
        before += r < this_rank ? v : 0u;
        mine = r == this_rank ? v : mine;
      }
      // rank lane_id's count above the group
      const auto row_above = lane_id < kClusterSize ? smem->recv_bin[lane_id][kGroupBins] : 0u;
      const auto incl = warp::inclusive_sum(bin_count, lane_id);
      const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
      const auto above = smem->above_group + total - incl;
      // Exactly one lane hits: the group's bins sum to its count, which straddles topk (2b). So
      // max-reduxes (CREDUX, uniform results) read the hit lane's values.
      const bool hit = above < topk && above + bin_count >= topk;
      const auto bin = warp::reduce_max(hit ? lane_id : 0u);
      const auto bin_above = warp::reduce_max(hit ? above : 0u);
      const auto bin_eq = warp::reduce_max(hit ? bin_count : 0u);
      const auto base_eq = warp::reduce_max(hit ? before : 0u);
      const auto my_eq = warp::reduce_max(hit ? mine : 0u);
      const auto base_gt = warp::reduce_sum((lane_id < this_rank ? row_above : 0u) + (lane_id > bin ? before : 0u));
      const auto my_gt = warp::reduce_sum((lane_id == this_rank ? row_above : 0u) + (lane_id > bin ? mine : 0u));
      const auto [v_lo, v_hi] = bin_bounds(smem->sel_group * kGroupBins + bin);
      smem->sel_bin = bin;  // every lane stores the same values
      smem->above_total = bin_above;
      smem->eq_total = bin_eq;
      smem->base_gt = base_gt;
      smem->base_eq = base_eq;
      smem->my_gt = my_gt;
      smem->my_eq = my_eq;
      smem->v_lo = v_lo;
      smem->v_hi = v_hi;
    }
    __syncthreads();
    // Every waiter has passed both phases: complete the next ones empty, for the next problem.
    if (tx == 0) {
      ptx::mbar_skip_phase(ptx::to_shared(&smem->mbar_group));
      ptx::mbar_skip_phase(ptx::to_shared(&smem->mbar_bin));
    }
  }

  /// Selects the top-k of `problem` across the cluster, cold rows included; `init` runs first.
  /// kInlineCold: inline the cold path instead of calling it. A call makes ptxas keep per-item
  /// state on the stack in a 32-register kernel (the persistent pool); inlining it slows the
  /// small-batch kernel, which also holds the one-block paths.
  template <bool kUsePDL, bool kInlineCold = false>
  SGL_DEVICE static void forward(TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);
    const uint32_t this_rank = blockIdx.y;
    const auto topk = problem.topk;

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
      smem->tie_ctr_gt = 0;
      smem->tie_ctr_eq = 0;
    }
    // bar-0: no rank pushes into another, and no warp of this CTA counts, before every rank has
    // reset the state above and read the previous problem's receive buffers. A shared::cta release
    // fence + relaxed arrive publishes exactly that (the release arrive is a MEMBAR.ALL.GPU that
    // also waits for the previous output stores); its latency hides behind the PDL wait.
    // The wait comes before the scan: its acquire invalidates L1, which the collect pass re-reads.
    ptx::fence_release_shared_cta();
    ptx::barrier_cluster_arrive_relaxed();
    PDLWaitPrimary<kUsePDL>();
    ptx::barrier_cluster_wait();

    pad_tail(problem, smem);
    // Phase 1: histogram over this rank's interleaved chunks.
    // Bin -0.0 as +0.0 (x + 0 under round-to-nearest), as fp32 compares see it.
    for_each_vec(problem.in, div_ceil(problem.seq_len, kVecSize), [&](const vec_t& v, uint32_t) {
      const auto [b0, b1] = coarse_bins2(v[0], v[1]);
      const auto [b2, b3] = coarse_bins2(v[2], v[3]);
      atomicAdd(&smem->histogram[b0], 1);
      atomicAdd(&smem->histogram[b1], 1);
      atomicAdd(&smem->histogram[b2], 1);
      atomicAdd(&smem->histogram[b3], 1);
    });
    __syncthreads();

    find_cluster_threshold(smem, topk);
    const uint32_t above_total = smem->above_total;
    const uint32_t eq_total = smem->eq_total;
    // Rank 0's coarse histogram is dead (pushed, consumed): it becomes the tie select's digit histogram.
    if (this_rank == 0) init_histogram(smem->histogram, tx);
    // CTA-uniform: let them live in uniform registers through the collect loop.
    const uint32_t base_gt = broadcast(smem->base_gt);
    const uint32_t base_eq = broadcast(smem->base_eq);
    const uint32_t my_gt = broadcast(smem->my_gt);
    const uint32_t my_eq = broadcast(smem->my_eq);
    const float v_lo = broadcast(smem->v_lo), v_hi = broadcast(smem->v_hi);

    // Cold path (exact modes): the bin overflows the tie buffer. if constexpr: mode 0 never
    // instantiates the fine path.
    if constexpr (kExact) {
      if (eq_total > kMaxNumTie) [[unlikely]] {
        __syncthreads();  // everyone has read the 2d results before the fine path reuses them
        if constexpr (kInlineCold) {
          return cold_path(problem, smem);
        } else {
          return cold_path_noinline(problem, smem);
        }
      }
    }

    // Phase 3: collect. Above -> output at base_gt; ties -> rank 0's tie buffer at base_eq.
    // The collected counts equal the histogram's (bin_bounds), except that a NaN
    // score is counted but never collected, so no store needs a bound; mode 0 only
    // truncates the tie buffer at kMaxNumTie (in rank order). NaN holes are filled below.
    const uint32_t tie_cap = broadcast(min(my_eq, base_eq < kMaxNumTie ? kMaxNumTie - base_eq : 0u));
    const uint32_t tie_total = min(eq_total, kMaxNumTie);
    if (this_rank == 0) {
      if (tx == 0) {
        ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar_ties), (tie_total - tie_cap) * sizeof(TieValue));
      }
      collect_scan(problem, v_lo, [&](float val, uint32_t idx) {
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          problem.emit(pos, idx);
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (kExact || pos < tie_cap) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      });
    } else {
      const uint32_t tie_dst = broadcast(ptx::map_rank(ptx::to_shared(&smem->tie_values[base_eq]), 0));
      const uint32_t tie_bar = broadcast(ptx::map_rank(ptx::to_shared(&smem->mbar_ties), 0));
      auto out = problem;
      out.out += base_gt;
      collect_scan(problem, v_lo, [&](float val, uint32_t idx) {
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          out.emit(pos, idx);
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (kExact || pos < tie_cap) [[likely]] {
            ptx::st_async_v2(tie_dst + pos * sizeof(TieValue), __float_as_uint(val), idx, tie_bar);
          }
        }
      });
    }
    __syncthreads();
    {
      // The counts promise holes only for NaN scores; fill them with legal entries.
      const auto got_gt = smem->count_gt, got_eq = smem->count_eq;
      if (got_gt < my_gt) [[unlikely]] {
        for (uint32_t t = got_gt + tx; t < my_gt; t += kBlockSize) {
          problem.emit(base_gt + t, base_gt + t);  // a slot index is < topk < seq_len
        }
      }
      if (got_eq < tie_cap) [[unlikely]] {
        for (uint32_t t = got_eq + tx; t < tie_cap; t += kBlockSize) {
          const auto v = TieValue{-infinity_value(), base_eq + t};
          if (this_rank == 0) {
            smem->tie_values[t] = v;
          } else {
            ptx::st_async_v2(
                ptx::map_rank(ptx::to_shared(&smem->tie_values[base_eq + t]), 0),
                __float_as_uint(v.value),
                v.idx,
                ptx::map_rank(ptx::to_shared(&smem->mbar_ties), 0));
          }
        }
        __syncthreads();
      }
    }
    {
      if (this_rank != 0) return;
      // Rank 0's tie stage runs on the warps it uses (one per tie, or one per target for at
      // most a warp of ties); the rest leave now.
      const auto remain_topk = above_total < topk ? topk - above_total : 0;
      // Slots to fill: every tie, or every remaining slot when the ties fit (holes only with NaN).
      const auto tie_slots = tie_total <= remain_topk ? remain_topk : tie_total;
      const bool by_target = tie_total <= kWarpSize && tie_total > remain_topk;
      const uint32_t tie_warps = max(by_target ? tie_total : (tie_slots + kWarpSize - 1) / kWarpSize, 1u);
      if (tx >= min(tie_warps, kNumWarps) * kWarpSize) return;
      if (tx < kWarpSize) ptx::mbar_wait_shared(ptx::to_shared(&smem->mbar_ties));
      ptx::bar_sync(1, min(tie_warps, kNumWarps) * kWarpSize);
      if (tx == 0) ptx::mbar_skip_phase(ptx::to_shared(&smem->mbar_ties));
      if (tie_total <= kWarpSize || tie_total <= remain_topk) {
        handle_tie(smem->tie_values, problem, above_total, tie_total, remain_topk, &smem->tie_handle);
      } else {
        tie_select_digits(problem, smem, above_total, tie_total, remain_topk, key_low(v_lo), key_bits(v_lo, v_hi));
      }
    }
  }

  static constexpr uint32_t kTieDigitBits = 6;
  /// Rank 0: exact select of `remain` of the `n` staged ties (remain < n <= kMaxNumTie),
  /// on the warps holding a tie. Rounds of kTieDigitBits-bit digits of the key relative to key(v_lo) (mode 2:
  /// followed by the inverted index, so equal values are cut at the smallest indices) narrow
  /// the bucket holding the remain-th largest tie: every thread histograms its bucket members
  /// over the zeroed histogram, warp 0 finds the digit, and the members above it are emitted.
  /// The final digit's members are emitted whole, cut (an all-equal digit, mode 1), or ranked
  /// all-pairs by warp 0 when at most a warp of them is left.
  SGL_DEVICE static void tie_select_digits(
      const TopKProblem& problem,
      Smem* smem,
      const uint32_t base,
      const uint32_t n,
      const uint32_t remain,
      const uint32_t key_lo,
      const uint32_t num_bits) {
    if (remain == 0) return;  // only with NaN-polluted counts
    using Key = std::conditional_t<kDeterministic, uint64_t, uint32_t>;
    constexpr uint32_t kIdxBits = kDeterministic ? kDeterministicIdxBits : 0;
    const auto tx = threadIdx.x;
    const auto lane_id = get_lane_id();
    const auto warp_id = broadcast(tx / kWarpSize);
    // Only the warps holding a tie take part (named barrier 1); the rest leave.
    const uint32_t num_threads = min((n + kWarpSize - 1) / kWarpSize * kWarpSize, kBlockSize);
    if (tx >= num_threads) return;
    const auto key_of = [=](const TieValue& tie) -> Key {
      const uint32_t key = extract_tie_key(tie.value);
      const Key rel = key >= key_lo ? key - key_lo : 0u;  // filler ties (NaN rows) sit below key_lo
      if constexpr (kDeterministic) {
        return (rel << kIdxBits) | ((1u << kIdxBits) - 1 - tie.idx);
      } else {
        return rel;
      }
    };
    TieValue tie[kTieItems];
    Key key[kTieItems];
    bool active[kTieItems];
#pragma unroll
    for (uint32_t i = 0; i < kTieItems; ++i) {
      const auto t = tx + i * kBlockSize;
      active[i] = t < n;
      tie[i] = active[i] ? smem->tie_values[t] : TieValue::invalid();
      key[i] = key_of(tie[i]);
    }
    const uint32_t total_bits = num_bits + kIdxBits;
    uint32_t shift = total_bits > kTieDigitBits ? total_bits - kTieDigitBits : 0;
    Key bucket_lo = 0;
    uint32_t count = n, need = remain, done = 0;  // done: slots emitted above the bucket
#pragma unroll 1
    while (true) {
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        if (active[i]) atomicAdd(&smem->histogram[static_cast<uint32_t>((key[i] - bucket_lo) >> shift)], 1);
      }
      ptx::bar_sync(1, num_threads);
      if (warp_id == 0) {
        const uint32_t c0 = smem->histogram[2 * lane_id], c1 = smem->histogram[2 * lane_id + 1];
        smem->histogram[2 * lane_id] = 0;
        smem->histogram[2 * lane_id + 1] = 0;
        const auto incl = warp::inclusive_sum(c0 + c1, lane_id);
        const uint32_t above1 = count - incl, above0 = above1 + c1;
        const bool hit1 = above1 < need && above1 + c1 >= need;
        const bool hit0 = above0 < need && above0 + c0 >= need;
        if (hit1) smem->tie_digit = {2 * lane_id + 1, above1, c1};
        if (hit0) smem->tie_digit = {2 * lane_id, above0, c0};
      }
      ptx::bar_sync(1, num_threads);
      const auto [sel, above, sel_count] = smem->tie_digit;
      const bool last = sel_count == need - above || sel_count <= kWarpSize || shift == 0;
      // The final digit's members: all of them, or the first ones of an all-equal digit
      // (mode 1 only: the index makes mode-2 keys distinct), or staged for the ranking.
      const bool rank_digit = sel_count != need - above && shift != 0;
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        if (!active[i]) continue;
        const auto digit = static_cast<uint32_t>((key[i] - bucket_lo) >> shift);
        if (digit > sel) {
          problem.emit(base + atomicAdd(&smem->tie_ctr_gt, 1), tie[i].idx);
          active[i] = false;
        } else if (digit < sel) {
          active[i] = false;
        } else if (last) {
          const auto pos = atomicAdd(&smem->tie_ctr_eq, 1);
          if (rank_digit) {
            smem->tie_stage[pos] = tie[i];
          } else if (pos < need - above) {
            problem.emit(base + done + above + pos, tie[i].idx);
          }
        }
      }
      need -= above;
      done += above;
      count = sel_count;
      if (last) {
        if (!rank_digit) return;
        break;
      }
      // tie_ctr_gt keeps counting: the next round's members above go after these.
      bucket_lo += static_cast<Key>(sel) << shift;
      shift = shift > kTieDigitBits ? shift - kTieDigitBits : 0;
    }
    ptx::bar_sync(1, num_threads);
    if (warp_id != 0) return;
    const auto mine = lane_id < count ? smem->tie_stage[lane_id] : TieValue::invalid();
    const Key my_key = key_of(mine);
    uint32_t rank = 0;
#pragma unroll
    for (uint32_t j = 0; j < kWarpSize; ++j) {
      const Key other = __shfl_sync(0xFFFFFFFF, my_key, j);
      rank += j < count && (other > my_key || (other == my_key && j < lane_id)) ? 1 : 0;
    }
    if (lane_id < count && rank < need) problem.emit(base + done + rank, mine.idx);
  }

  // Cold paths (exact modes): the threshold bin holds more than kMaxNumTie members.

  /// Every cold case. Takes only the problem: the state is re-read from shared memory, so a
  /// call to it (cold_path_noinline) keeps few registers live across the hot path.
  SGL_DEVICE static void cold_path(const TopKProblem problem, Smem* smem) {
    const float v_lo = smem->v_lo, v_hi = smem->v_hi;
    const uint32_t base_gt = smem->base_gt, my_gt = smem->my_gt;
    fine_path(problem, smem, blockIdx.y, v_lo, v_hi, smem->above_total, base_gt, my_gt);
  }
  __device__ __noinline__ static void cold_path_noinline(const TopKProblem problem, Smem* smem) {
    cold_path(problem, smem);
  }

  /// Cold path for a threshold bin with more than kMaxNumTie members (not an edge
  /// bin). The above elements are exact and emitted as on the normal path;
  /// the same pass builds a 4096-bin histogram of the bin's members over 12-bit
  /// digits of key - key(v_lo), which is all-gathered like the coarse one (64 group
  /// sums, then the selected group's 64 bins), so every rank finds the k-th digit
  /// and its bases itself. A second pass emits the members above that digit and
  /// st.asyncs the digit's members to rank 0, which resolves them. If the digit
  /// still holds more than kMaxNumTie members (runs of near-equal keys), the
  /// barrier-based refine (refine_cluster) takes over.
  SGL_DEVICE static void fine_path(
      const TopKProblem& problem,
      Smem* smem,
      const uint32_t this_rank,
      const float v_lo,
      const float v_hi,
      const uint32_t above_total,
      const uint32_t base_gt,
      const uint32_t my_gt) {
    constexpr uint32_t kFineGroups = 64;
    constexpr uint32_t kFineGroupBins = kRefineSize / kFineGroups;  // 64
    constexpr uint32_t kFineRow = kFineGroupBins + 4;
    static_assert(kFineGroups <= kGroups && sizeof(uint32_t) * kClusterSize * kFineRow <= sizeof(uint32_t) * kHistSize);
    const auto tx = threadIdx.x;
    const auto lane_id = get_lane_id();
    const auto warp_id = broadcast(tx / kWarpSize);
    const auto topk = problem.topk;
    const uint32_t key_lo = key_low(v_lo);
    const uint32_t num_bits = key_bits(v_lo, v_hi);
    const uint32_t shift = num_bits > kRefineBits ? num_bits - kRefineBits : 0;
    const uint32_t remain = topk - above_total;  // 0 < remain < members of the bin
    const auto recv_fine_bin = reinterpret_cast<uint32_t (*)[kFineRow]>(smem->histogram);

    zero_refine_hist(smem);
    if (tx == 0) {
      smem->fine_count_gt = 0;
      smem->member_count = 0;
    }
    __syncthreads();
    // Pass 1: above as on the normal path; members into the fine histogram.
    for_each_input<kClusterSize>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < my_gt) [[likely]] {
          problem.emit(base_gt + pos, idx);
        }
      } else if (val >= v_lo) {
        atomicAdd(&smem->refine_hist[(extract_tie_key(val) - key_lo) >> shift], 1);
        if (const auto pos = atomicAdd(&smem->member_count, 1); pos < kMaxNumTie) smem->members[pos] = {val, idx};
      }
    });
    __syncthreads();
    if (const auto got_gt = smem->count_gt; got_gt < my_gt) {
      for (uint32_t t = got_gt + tx; t < my_gt; t += kBlockSize) {
        problem.emit(base_gt + t, base_gt + t);
      }
    }
    // F1: group sums, all-gathered. As in phase 2a: warp w sums groups 2w and 2w + 1 (two
    // bins of each per lane), one full-warp redux each.
    static_assert(kFineGroups == 2 * kNumWarps && kFineGroupBins == 2 * kWarpSize);
    {
      const auto bins = smem->refine_hist + 2 * warp_id * kFineGroupBins + lane_id;
      smem->group_local[2 * warp_id] = warp::reduce_sum(bins[0] + bins[kWarpSize]);
      smem->group_local[2 * warp_id + 1] = warp::reduce_sum(bins[2 * kWarpSize] + bins[3 * kWarpSize]);
    }
    if (tx == 0)
      ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar_fine_group), kClusterSize * kFineGroups * sizeof(uint32_t));
    __syncthreads();
    static_assert(kClusterSize * (kFineGroups / 4) <= kBlockSize);
    if (tx < kClusterSize * (kFineGroups / 4)) {
      const uint32_t chunk = tx % (kFineGroups / 4), dst_rank = tx / (kFineGroups / 4);
      // Raw uint4 / uint2 views of the DSMEM rows: AlignedVector adds instructions here.
      const auto v = reinterpret_cast<const uint4*>(smem->group_local)[chunk];
      push_v4(&smem->recv_group[this_rank][chunk * 4], v, dst_rank, &smem->mbar_fine_group);
    }
    if (warp_id == 0) {
      ptx::mbar_wait(ptx::to_shared(&smem->mbar_fine_group));
      uint32_t sum_even = 0, sum_odd = 0;
#pragma unroll
      for (uint32_t r = 0; r < kClusterSize; ++r) {
        const auto v = reinterpret_cast<const uint2*>(smem->recv_group[r])[lane_id];
        sum_even += v.x;
        sum_odd += v.y;
      }
      const auto incl = warp::inclusive_sum(sum_even + sum_odd, lane_id);
      const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
      const uint32_t above_odd = total - incl, above_even = above_odd + sum_odd;
      const bool hit_odd = above_odd < remain && above_odd + sum_odd >= remain;
      const bool hit_even = above_even < remain && above_even + sum_even >= remain;
      const auto mask = __ballot_sync(0xFFFFFFFF, hit_even || hit_odd);
      const auto src = mask ? __ffs(mask) - 1 : 0;
      const auto group = __shfl_sync(0xFFFFFFFF, 2 * lane_id + (hit_odd ? 1 : 0), src);
      const auto group_above = __shfl_sync(0xFFFFFFFF, hit_odd ? above_odd : above_even, src);
      uint32_t own = (2 * lane_id > group ? smem->group_local[2 * lane_id] : 0) +
                     (2 * lane_id + 1 > group ? smem->group_local[2 * lane_id + 1] : 0);
      own = warp::reduce_sum(own);
      if (lane_id == 0) {
        smem->sel_group = group;
        smem->above_group = group_above;
        smem->own_above_group = own;
      }
    }
    if (tx == 0)
      ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar_fine_bin), kClusterSize * kFineRow * sizeof(uint32_t));
    __syncthreads();
    // F2: the selected group's 64 bins plus the count above the group.
    static_assert(kClusterSize * (kFineRow / 4) <= kBlockSize);
    if (tx < kClusterSize * (kFineRow / 4)) {
      const uint32_t chunk = tx % (kFineRow / 4), dst_rank = tx / (kFineRow / 4);
      const auto group = smem->sel_group;
      const auto v = chunk < kFineGroupBins / 4
                         ? reinterpret_cast<const uint4*>(smem->refine_hist + group * kFineGroupBins)[chunk]
                         : uint4{smem->own_above_group, 0, 0, 0};
      push_v4(&recv_fine_bin[this_rank][chunk * 4], v, dst_rank, &smem->mbar_fine_bin);
    }
    if (warp_id == 0) {
      ptx::mbar_wait(ptx::to_shared(&smem->mbar_fine_bin));
      uint32_t sum_even = 0, sum_odd = 0;
#pragma unroll
      for (uint32_t r = 0; r < kClusterSize; ++r) {
        const auto v = reinterpret_cast<const uint2*>(recv_fine_bin[r])[lane_id];
        sum_even += v.x;
        sum_odd += v.y;
      }
      const auto incl = warp::inclusive_sum(sum_even + sum_odd, lane_id);
      const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
      const uint32_t above_odd = smem->above_group + total - incl, above_even = above_odd + sum_odd;
      const bool hit_odd = above_odd < remain && above_odd + sum_odd >= remain;
      const bool hit_even = above_even < remain && above_even + sum_even >= remain;
      const auto mask = __ballot_sync(0xFFFFFFFF, hit_even || hit_odd);
      const auto src = mask ? __ffs(mask) - 1 : 0;
      const auto bin = __shfl_sync(0xFFFFFFFF, 2 * lane_id + (hit_odd ? 1 : 0), src);
      const auto bin_above = __shfl_sync(0xFFFFFFFF, hit_odd ? above_odd : above_even, src);
      const auto bin_count = __shfl_sync(0xFFFFFFFF, hit_odd ? sum_odd : sum_even, src);
      if (lane_id == 0) {
        smem->sel_bin = bin;
        smem->above_total = bin_above;
        smem->eq_total = bin_count;
      }
    } else if (warp_id <= kClusterSize) {
      ptx::mbar_wait(ptx::to_shared(&smem->mbar_fine_bin));
    }
    __syncthreads();
    const uint32_t bin = smem->sel_bin;
    const uint32_t digit = smem->sel_group * kFineGroupBins + bin;
    const uint32_t above_digit = smem->above_total;  // bin members above the digit, cluster-wide
    const uint32_t eq_digit = smem->eq_total;
    if (warp_id < kClusterSize) {
      const auto row = recv_fine_bin[warp_id];
      const auto v = reinterpret_cast<const uint2*>(row)[lane_id];
      uint32_t gt = (2 * lane_id > bin ? v.x : 0) + (2 * lane_id + 1 > bin ? v.y : 0);
      gt = warp::reduce_sum(gt) + row[kFineGroupBins];
      if (lane_id == 0) {
        smem->rank_suffix[warp_id][0] = gt;
        smem->rank_suffix[warp_id][1] = row[bin];
      }
    }
    __syncthreads();
    uint32_t digit_base_gt, digit_base_eq, digit_my_gt, digit_my_eq;
    {
      const bool mine = lane_id < kClusterSize;
      const auto gt = mine ? smem->rank_suffix[lane_id][0] : 0u;
      const auto eq = mine ? smem->rank_suffix[lane_id][1] : 0u;
      const bool before = lane_id < this_rank;
      digit_base_gt = warp::reduce_sum(before ? gt : 0u);
      digit_base_eq = warp::reduce_sum(before ? eq : 0u);
      digit_my_gt = __shfl_sync(0xFFFFFFFF, gt, this_rank);
      digit_my_eq = __shfl_sync(0xFFFFFFFF, eq, this_rank);
    }
    if (tx == 0) {
      ptx::mbar_skip_phase(ptx::to_shared(&smem->mbar_fine_group));
      ptx::mbar_skip_phase(ptx::to_shared(&smem->mbar_fine_bin));
    }

    if (eq_digit > kMaxNumTie) [[unlikely]] {
      // refine_cluster re-emits every selected slot at new positions: drop the slots pass 1 wrote,
      // or a rank's own staging (PAGE_TABLE / DUAL_OUTPUT) keeps stale entries that it transforms.
      for (uint32_t t = tx; t < topk; t += kBlockSize) {
        problem.out[t] = -1;
      }
      if (this_rank == 0 && tx == 0) ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar_ties), 0);
      const auto smem_0 = ptx::map_rank(smem, 0);
      if (this_rank == 0) {
        refine_cluster<true>(problem, smem, smem, v_lo, v_hi);
        ptx::mbar_wait(ptx::to_shared(&smem->mbar_ties));
        __syncthreads();
        if (tx == 0) ptx::mbar_skip_phase(ptx::to_shared(&smem->mbar_ties));
      } else {
        refine_cluster<false>(problem, smem, smem_0, v_lo, v_hi);
      }
      return;
    }

    // Pass 2: members above the digit -> output; the digit's members -> rank 0.
    if (this_rank == 0) {
      init_histogram(smem->histogram, tx);  // recv_fine_bin consumed; tie_select_single_digit's digit histogram
      if (tx == 0)
        ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar_ties), (eq_digit - digit_my_eq) * sizeof(TieValue));
    }
    const uint32_t out_base = above_total + digit_base_gt;
    const uint32_t tie_dst = ptx::map_rank(ptx::to_shared(&smem->tie_values[digit_base_eq]), 0);
    const uint32_t tie_bar = ptx::map_rank(ptx::to_shared(&smem->mbar_ties), 0);
    const auto pass2 = [&](float val, uint32_t idx) {
      if (!(val >= v_lo && !(val >= v_hi))) return;
      const auto member_digit = (extract_tie_key(val) - key_lo) >> shift;
      if (member_digit > digit) {
        const auto pos = atomicAdd(&smem->fine_count_gt, 1);
        if (pos < digit_my_gt) problem.emit(out_base + pos, idx);
      } else if (member_digit == digit) {
        const auto pos = atomicAdd(&smem->count_eq, 1);
        if (pos < digit_my_eq) {
          if (this_rank == 0) {
            smem->tie_values[pos] = {val, idx};
          } else {
            ptx::st_async_v2(tie_dst + pos * sizeof(TieValue), __float_as_uint(val), idx, tie_bar);
          }
        }
      }
    };
    // Rescan only the members pass 1 staged, unless they overflowed the stage.
    if (const auto members = smem->member_count; members <= kMaxNumTie) {
      for (uint32_t t = tx; t < members; t += kBlockSize) {
        const auto m = smem->members[t];
        pass2(m.value, m.idx);
      }
    } else {
      for_each_input<kClusterSize>(problem.in, problem.seq_len, pass2);
    }
    __syncthreads();
    if (this_rank == 0) {
      if (tx < kWarpSize) ptx::mbar_wait(ptx::to_shared(&smem->mbar_ties));
      __syncthreads();
      if (tx == 0) ptx::mbar_skip_phase(ptx::to_shared(&smem->mbar_ties));
      const auto base = above_total + above_digit;
      const auto need = remain - above_digit;
      if (eq_digit <= kWarpSize * 4 || eq_digit <= need) {
        handle_tie(smem->tie_values, problem, base, eq_digit, need, &smem->tie_handle);
      } else {
        tie_select_single_digit(problem, smem, base, eq_digit, need, key_lo + (digit << shift), shift);
      }
    }
  }

  /// Rank 0: exact select of `remain` of the `n` staged ties (n > remain, n <= kMaxNumTie).
  /// One 11-bit digit of the key relative to key(v_lo) over the zeroed histogram;
  /// elements above the k-th digit are emitted, the k-th digit's members are ranked
  /// by (value desc, idx asc) in one warp when they are at most 32, else
  /// compacted and handed to the generic handle_tie.
  SGL_DEVICE static void tie_select_single_digit(
      const TopKProblem& problem,
      Smem* smem,
      uint32_t base,
      uint32_t n,
      uint32_t remain,
      uint32_t key_lo,
      uint32_t num_bits) {
    const auto tx = threadIdx.x;
    const auto lane_id = get_lane_id();
    const auto warp_id = broadcast(tx / kWarpSize);
    const uint32_t shift = num_bits > kHistBits ? num_bits - kHistBits : 0;
    TieValue tie[kTieItems];
    uint32_t digit[kTieItems];
#pragma unroll
    for (uint32_t i = 0; i < kTieItems; ++i) {
      const auto t = tx + i * kBlockSize;
      tie[i] = t < n ? smem->tie_values[t] : TieValue::invalid();
      const auto key = extract_tie_key(tie[i].value);
      // Filler ties (NaN rows) sit below key_lo: put them in digit 0.
      digit[i] = min((key >= key_lo ? key - key_lo : 0u) >> shift, kHistSize - 1);
      if (t < n) atomicAdd(&smem->histogram[digit[i]], 1);
    }
    __syncthreads();
    // The digit holding the remain-th largest tie (same scan as find_threshold, from the top).
    {
      AlignedVector<uint32_t, kHistItems> h;
      h.load(smem->histogram, tx);
      uint32_t local = 0;
#pragma unroll
      for (uint32_t i = 0; i < kHistItems; ++i) {
        local += h[i];
      }
      const auto warp_inc = warp::inclusive_sum(local, lane_id);
      if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
      __syncthreads();
      const auto lane_warp_sum = smem->warp_sum[lane_id];
      uint32_t prefix =
          warp::reduce_sum(lane_id < warp_id ? lane_warp_sum : 0) + warp_inc - local;  // ties in lower digits
#pragma unroll
      for (uint32_t i = 0; i < kHistItems; ++i) {
        prefix += h[i];
        const auto above = n - prefix;
        if (above < remain && above + h[i] >= remain) smem->tie_digit = {tx * kHistItems + i, above, h[i]};
      }
      __syncthreads();
    }
    const auto [sel, above, count] = smem->tie_digit;
    const auto need = remain - above;
#pragma unroll
    for (uint32_t i = 0; i < kTieItems; ++i) {
      if (tx + i * kBlockSize >= n) continue;
      if (digit[i] > sel) {
        problem.emit(base + atomicAdd(&smem->tie_ctr_gt, 1), tie[i].idx);
      } else if (digit[i] == sel) {
        const auto pos = atomicAdd(&smem->tie_ctr_eq, 1);
        if (need == count) {
          problem.emit(base + above + pos, tie[i].idx);
        } else if (count <= kWarpSize) {
          smem->tie_stage[pos] = tie[i];
        } else {
          smem->tie_values[pos] = tie[i];  // compacted in place: every tie was read before the scan
        }
      }
    }
    if (need == count) return;
    __syncthreads();
    if (count > kWarpSize) [[unlikely]] {
      return handle_tie(smem->tie_values, problem, base + above, count, need, &smem->tie_handle);
    }
    if (warp_id == 0) {
      const auto mine = lane_id < count ? smem->tie_stage[lane_id] : TieValue::invalid();
      uint32_t rank = 0;
      for (uint32_t j = 0; j < count; ++j) {
        const auto other = smem->tie_stage[j];
        rank += (other.value > mine.value) || (other.value == mine.value && other.idx < mine.idx);
      }
      if (lane_id < count && rank < need) problem.emit(base + above + rank, mine.idx);
    }
  }

  /// Refine from scratch, for an edge bin or a fine-path digit that still holds more
  /// than kMaxNumTie members: cluster-wide radix rounds over the exact key (12-bit
  /// digits relative to key(v_lo), round 0 fused with the collect pass, merged into
  /// rank 0 with remote atomics), then the final bucket is staged and resolved by
  /// rank 0's handle_tie. Self-contained: counts from the scans, rank bases from
  /// atomics on rank 0, its own cluster barriers. kIsRank0 stays a template
  /// parameter: as a runtime flag it changes the generated code.
  template <bool kIsRank0>
  SGL_DEVICE static void refine_cluster(const TopKProblem& problem, Smem* smem, Smem* smem_0, float v_lo, float v_hi) {
    const auto tx = threadIdx.x;
    const auto topk = problem.topk;
    zero_refine_hist(smem);
    if (tx == 0) {
      smem->count_gt = 0;
      smem->count_eq = 0;
      smem->refine_gt = 0;
    }
    __syncthreads();
    ptx::cluster_sync();  // every rank's counters and histogram are zero before any remote update

    const uint32_t key_lo = key_low(v_lo);
    const uint32_t span = extract_exact_bin(v_hi) - key_lo;  // in-bin keys are < key_lo + span
    const uint32_t num_bits = 32 - __clz(span - 1);
    // Mode 2 radixes (key, smallest index first) so the final bucket is one element
    // and the selected set is deterministic (idx < 2^20: scores are at most 1M long).
    using Key = std::conditional_t<kDeterministic, uint64_t, uint32_t>;
    constexpr uint32_t kIdxBits = kDeterministic ? kDeterministicIdxBits : 0;
    const auto comp = [=](float val, uint32_t idx) -> Key {
      const Key rel = extract_tie_key(val) - key_lo;
      if constexpr (kDeterministic) {
        return (rel << kIdxBits) | ((1u << kIdxBits) - 1 - idx);
      } else {
        return rel;
      }
    };
    const uint32_t total_bits = num_bits + kIdxBits;
    uint32_t shift = total_bits > kRefineBits ? total_bits - kRefineBits : 0;
    const auto emit_above = [&](uint32_t idx) {
      const auto pos = atomicAdd(&smem->count_gt, 1);
      if constexpr (kIsRank0) {
        if (pos < topk) [[likely]] {
          problem.emit(pos, idx);
        }
      } else {
        if (pos < topk) [[likely]] {
          smem->stage_out_idxs[pos] = idx;
        }
      }
    };

    for_each_input<kClusterSize>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      // NaN-safe: a NaN upper edge (inf bin) puts nothing above and +inf in the bin.
      if (val >= v_hi) {
        emit_above(idx);
      } else if (val >= v_lo) {
        atomicAdd(&smem->refine_hist[static_cast<uint32_t>(comp(val, idx) >> shift)], 1);
      }
    });
    Key bucket_lo = 0;
    Key bucket_mask = (static_cast<Key>(span - 1) << kIdxBits) | ((Key(1) << kIdxBits) - 1);
    uint32_t remain = 0;
#pragma unroll 1
    for (uint32_t round = 0;; ++round) {
      __syncthreads();
      if constexpr (!kIsRank0) {
        AlignedVector<uint32_t, kRefineItems> counts;
        counts.load(smem->refine_hist, tx);
        zero_refine_hist(smem);
#pragma unroll
        for (uint32_t i = 0; i < kRefineItems; ++i) {
          if (counts[i] != 0) atomicAdd(&smem_0->refine_hist[tx * kRefineItems + i], counts[i]);
        }
        if (round == 0 && tx == 0) atomicAdd(&smem_0->refine_gt, smem->count_gt);
        ptx::barrier_cluster_arrive_release();  // A: histograms merged
        ptx::barrier_cluster_wait();
        ptx::barrier_cluster_arrive_relaxed();  // B: digit published
        ptx::barrier_cluster_wait();
      } else {
        if (round == 0 && tx == 0) atomicAdd(&smem->refine_gt, smem->count_gt);
        ptx::barrier_cluster_arrive_relaxed();  // A
        ptx::barrier_cluster_wait();
        __syncthreads();
        if (round == 0) {
          const auto gt = smem->refine_gt;
          remain = gt < topk ? topk - gt : 0;
          if (tx == 0) smem->refine_remain = remain;
        }
        if (remain != 0) refine_find_digit(smem, remain);
        __syncthreads();
        ptx::barrier_cluster_arrive_release();  // B: publishes refine_match / refine_remain
        ptx::barrier_cluster_wait();
      }
      if (round == 0) {
        remain = smem_0->refine_remain;
        if (remain == 0) break;  // above elements fill every slot
      }
      const MatchBin match = smem_0->refine_match;
      bucket_lo += static_cast<Key>(match.bin) << shift;
      bucket_mask = (Key(1) << shift) - 1;
      remain -= match.above_count;
      if (shift == 0 || match.equal_count <= kMaxNumTie) break;
      if constexpr (kIsRank0) __syncthreads();  // all have read refine_match before the next find rewrites it
      shift = shift > kRefineBits ? shift - kRefineBits : 0;
      for_each_input<kClusterSize>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (!(val >= v_lo && !(val >= v_hi))) return;
        const Key d = comp(val, idx) - bucket_lo;
        if (d <= bucket_mask) atomicAdd(&smem->refine_hist[static_cast<uint32_t>(d >> shift)], 1);
      });
    }

    __syncthreads();  // everyone has read the histogram before tie_values overwrite it
    if (remain != 0) {
      for_each_input<kClusterSize>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (!(val >= v_lo && !(val >= v_hi))) return;
        const Key rel = comp(val, idx);
        if (rel < bucket_lo) return;
        if (rel - bucket_lo > bucket_mask) {
          emit_above(idx);
        } else {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      });
    }
    __syncthreads();
    if constexpr (!kIsRank0) {
      const auto local_count_gt = smem->count_gt;
      const auto local_count_eq = min(smem->count_eq, kMaxNumTie);
      if (tx == 0) {
        smem->local_start_gt = atomicAdd(&smem_0->count_gt, local_count_gt);
        smem->local_start_eq = atomicAdd(&smem_0->count_eq, local_count_eq);
      }
      __syncthreads();
      const auto local_start_gt = smem->local_start_gt;
      const auto local_start_eq = smem->local_start_eq;
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        const auto t = tx + i * kBlockSize;
        if (t < local_count_eq && local_start_eq + t < kMaxNumTie) {
          smem_0->tie_values[local_start_eq + t] = smem->tie_values[t];
        }
      }
      ptx::cluster_sync();
#pragma unroll
      for (uint32_t i = 0; i < kTopKItems; ++i) {
        if (const auto t = tx + i * kBlockSize; t < local_count_gt && local_start_gt + t < topk) {
          problem.emit(local_start_gt + t, smem->stage_out_idxs[t]);
        }
      }
    } else {
      ptx::cluster_sync();
      handle_staged_ties(problem, smem);
    }
  }

  SGL_DEVICE static void zero_refine_hist(Smem* smem) {
    AlignedVector<uint32_t, kRefineItems> zero;
    zero.fill(0);
    zero.store(smem->refine_hist, threadIdx.x);
  }

  /// Rank 0: digit search over the merged kRefineSize-bin histogram (zeroed after reading).
  SGL_DEVICE static void refine_find_digit(Smem* smem, const uint32_t remain) {
    const auto tx = threadIdx.x;
    const auto lane_id = get_lane_id();
    const auto warp_id = broadcast(tx / kWarpSize);
    AlignedVector<uint32_t, kRefineItems> counts;
    counts.load(smem->refine_hist, tx);
    zero_refine_hist(smem);
    if (tx == 0) smem->refine_match = {0, 0, 0};  // stays if nothing matches (NaN-polluted counts)
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      local_sum += counts[i];
    }
    const auto warp_inc = warp::inclusive_sum(local_sum, lane_id);
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    __syncthreads();
    const auto lane_warp_sum = smem->warp_sum[lane_id];
    const auto total = warp::reduce_sum(lane_warp_sum);
    uint32_t prefix = warp::reduce_sum(lane_id < warp_id ? lane_warp_sum : 0) + warp_inc - local_sum;
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      prefix += counts[i];
      const auto above = total - prefix;
      if (above < remain && above + counts[i] >= remain) {
        smem->refine_match = {tx * kRefineItems + i, above, counts[i]};
      }
    }
  }
};

/// The coarse-bin cluster path of the DCP top-k candidates kernel: rank 0 folds every
/// rank's histogram and resolves ties. kBits: the coarse bin width (fp16 key bits); finer
/// bins overflow the 2048-entry tie buffer less often but cost a larger DSMEM fold.
template <uint32_t N, uint32_t kBits = 10>
struct TopKClusterCoarse : TopKRadixBase<10> {
 public:
  static constexpr uint32_t kClusterSize = N;
  static constexpr uint32_t kCoarseBits = kBits;
  static constexpr uint32_t kCoarseSize = 1u << kBits;
  static_assert(kBits >= 10 && kBits <= 12 && kCoarseSize % kBlockSize == 0);
  static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();
  struct Smem {
    uint32_t count_eq;
    uint32_t count_gt;
    uint32_t local_start_eq;
    uint32_t local_start_gt;
    float v_lo;
    float v_hi;
    uint32_t warp_sum[kNumWarps];
    union {
      alignas(16) uint32_t histogram[kCoarseSize];
      TieHandleSmem tie_handle;
      int32_t stage_out_idxs[kMaxTopK];
    };
    TieValue tie_values[kMaxNumTie];
  };

  SGL_DEVICE static void barrier_cluster_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  }

  SGL_DEVICE static void barrier_cluster_arrive_release() {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  }

  SGL_DEVICE static void barrier_cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  }

  template <bool kUsePDL>
  SGL_DEVICE static void forward(TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);
    const auto cluster = cooperative_groups::this_cluster();
    const auto this_rank = blockIdx.y;

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }
    __syncthreads();
    // Rank 0's shared memory is read by its peers: the zeroed histogram they fold
    // into after bar-0, v_hi / v_lo after bar-2. Those arrives release so the
    // peers' acquire wait orders the reads after the writes at cluster scope;
    // __syncthreads() alone is CTA-scoped. The peers publish nothing at these two
    // barriers and keep the cheaper relaxed arrive.
    if (this_rank == 0) {
      barrier_cluster_arrive_release();  // bar-0 arrive
    } else {
      barrier_cluster_arrive_relaxed();  // bar-0 arrive
    }
    PDLWaitPrimary<kUsePDL>();

    // Phase 1: Load and build histogram over this rank's contiguous chunk.
    for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t) {
      const auto bin = extract_coarse_bin<kCoarseBits>(val);
      atomicAdd(&smem->histogram[bin], 1);
    });

    barrier_cluster_wait();  // bar-0 wait
    __syncthreads();
    if (this_rank != 0) {
      const auto smem_0 = cluster.map_shared_rank(smem, 0);
      // Phase 2. atomic flush all histogram into rank 0
      constexpr uint32_t kFoldItems = kCoarseSize / kBlockSize;
#pragma unroll
      for (uint32_t k = 0; k < kFoldItems; ++k) {
        if (const auto count = smem->histogram[tx + k * kBlockSize]; count != 0) {
          atomicAdd(&smem_0->histogram[tx + k * kBlockSize], count);
        }
      }

      barrier_cluster_arrive_release();  // bar-1 arrive
      barrier_cluster_wait();            // bar-1 wait

      barrier_cluster_arrive_relaxed();  // bar-2 arrive
      barrier_cluster_wait();            // bar-2 wait

      // Phase 4. non-0 rank stage to local smem, then write to rank-0 via DSMEM
      const auto topk = problem.topk;
      const auto v_hi = smem_0->v_hi;
      const auto v_lo = smem_0->v_lo;
      for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            smem->stage_out_idxs[pos] = idx;
          }
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      });
      __syncthreads();
      const auto local_count_gt = smem->count_gt;
      const auto local_count_eq = min(smem->count_eq, kMaxNumTie);
      if (tx == 0) {
        const auto gt = atomicAdd(&smem_0->count_gt, local_count_gt);
        const auto eq = atomicAdd(&smem_0->count_eq, local_count_eq);
        smem->local_start_gt = gt;
        smem->local_start_eq = eq;
      }
      __syncthreads();
      const auto local_start_gt = smem->local_start_gt;
      const auto local_start_eq = smem->local_start_eq;
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        const auto t = tx + i * kBlockSize;
        if (t < local_count_eq && local_start_eq + t < kMaxNumTie) {
          smem_0->tie_values[local_start_eq + t] = smem->tie_values[t];
        }
      }

      cluster.sync();  // bar 3

      const auto start_write = local_start_gt;
      const auto num_write = local_count_gt;
#pragma unroll
      for (uint32_t i = 0; i < kTopKItems; ++i) {
        if (const auto t = tx + i * kBlockSize; t < num_write && start_write + t < topk) {
          problem.emit(start_write + t, smem->stage_out_idxs[t]);
        }
      }
    } else {
      barrier_cluster_arrive_relaxed();  // bar-1 arrive
      barrier_cluster_wait();            // bar-1 wait

      // Phase 3. rank-0 find threshold and write to local smem for other ranks to read
      find_threshold(problem.topk, problem.seq_len, smem, [&](uint32_t threshold_bin) {
        const auto [v_lo, v_hi] = coarse_bin_bounds<kCoarseBits>(threshold_bin);
        smem->v_hi = v_hi;
        smem->v_lo = v_lo;
      });

      barrier_cluster_arrive_release();  // bar-2 arrive: publishes v_hi / v_lo
      barrier_cluster_wait();            // bar-2 wait

      // Phase 4. rank-0 directly write to output
      const auto topk = problem.topk;
      const auto v_hi = smem->v_hi;
      const auto v_lo = smem->v_lo;
      for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            problem.emit(pos, idx);
          }
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      });

      cluster.sync();  // bar-3

      // Phase 4: Handle ties.
      const auto count_gt = smem->count_gt;
      const auto count_eq = smem->count_eq;
      const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
      const auto tie_count = min(count_eq, kMaxNumTie);
      handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
    }
  }
};

#endif  // SUPPORT_CLUSTER

}  // namespace SGL_TOPK_MODE_NAMESPACE

}  // namespace device::topk

}  // namespace sglang
