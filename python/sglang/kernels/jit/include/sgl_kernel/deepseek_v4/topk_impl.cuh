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

#if !defined(USE_ROCM)
// currently only apply cluster for SM90 & SM100, SM120 has poor cluster performance
#define SUPPORT_CLUSTER (SGL_CUDA_ARCH >= 900 && SGL_CUDA_ARCH < 1100)
#else
// AMD doesn't support cluster
#define SUPPORT_CLUSTER false
#endif

#if SUPPORT_CLUSTER
#include <sgl_kernel/mbarrier.cuh>
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
#define SGL_TOPK_MODE_NAMESPACE SGL_TOPK_CONCAT(topk_mode_, SGL_TOPK_MODE)

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
  uint32_t b = (uint32_t)__half_as_ushort(__float2half_rn(x)) << 16;
  uint32_t s = (uint32_t)((int32_t)b >> 31);
  return (b ^ (s | 0x80000000u)) >> (32 - kBits);
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

/// The score pointer, asserted global. Loads through a pointer that crossed an asm
/// memory clobber (the PDL wait) or a call otherwise lose the global address space
/// and become generic LD.E; call this on a local copy after the clobber.
SGL_DEVICE const float* assume_global(const float* ptr) {
#if !defined(USE_ROCM)
  __builtin_assume(__isGlobal(ptr));
#endif
  return ptr;
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
  const float* __restrict__ in;
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
      const auto warp_id_0 = warp_id;
      const auto warp_id_1 = warp_id + kWarpSize;
      const auto lane_id_1 = lane_id + kWarpSize;
      const auto invalid = TieValue::invalid();
      const auto tie_0 = tie_buffer[lane_id];
      const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
      const auto target_0 = tie_buffer[warp_id_0];
      const auto target_1 = tie_buffer[warp_id_1];
      if (true) {  // NOTE: warp_id_0 <= kNumWarps < num_ties
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
        // 3. Find threshold bin
        if (above < topk_remain && above + hist_val >= topk_remain) {
          smem->match = {tx, above, hist_val};
        }
      }
      __syncthreads();

      const auto [threshold_bin, above_count, equal_count] = smem->match;
      total_active = equal_count;
      topk_remain -= above_count;
      if (round == 3 && kDeterministic) straddle = equal_count > topk_remain;

      // 4. Scatter
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
        // my_bin == thr && round < 3: stay active for next round
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
 public:
  static constexpr uint32_t kVecSize = 4;
  static constexpr uint32_t kHistBits = kHistBits_;
  static constexpr uint32_t kHistSize = 1 << kHistBits;
  using vec_t = AlignedVector<float, kVecSize>;

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
  SGL_DEVICE static void for_each_input(const float* __restrict__ in_, uint32_t seq_len, F&& fn) {
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
      next_vec.load(in, vi);
#pragma unroll 1
      do {
        const auto cur = next_vec;
        vi += kStride;
        if (vi < num_full) next_vec.load(in, vi);
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
      cur.load(in, vi);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        if (base + j < seq_len) fn(cur[j], base + j);
      }
    }
  }

  SGL_DEVICE static void init_histogram(uint32_t (&histogram)[kHistSize], uint32_t tx) {
    constexpr uint32_t kItems = kHistSize / kBlockSize;
    AlignedVector<uint32_t, kItems> vec;
    vec.fill(0);
    vec.store(histogram, tx);
  }

  /// Same, but scanning a histogram that need not be `smem`'s own -- the cluster
  /// path merges into one rank's copy and scans it there.
  template <typename Smem, typename Fn>
  SGL_DEVICE static void find_threshold(const uint32_t topk, const uint32_t seq_len, Smem* smem, Fn fn) {
    const auto tx = threadIdx.x;
    constexpr uint32_t kItems = kHistSize / kBlockSize;
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

    const auto tmp = smem->warp_sum[lane_id];
    const auto warp_prefix_sum = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
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

  /// Re-stages the tie buffer exactly when the threshold bin overflows it: radix
  /// rounds narrow the bin to one bucket, then a final scan emits what lies above
  /// it and stages the bucket for the caller's handle_tie. In the deterministic mode,
  /// an exact-equal group that still overflows is narrowed by index, smallest first.
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
      if (vi < num_vec) cur.load(in, vi);
#pragma unroll 1
      while (vi < num_vec) {
        const auto next_vi = vi + kBlockSize;
        vec_t next;
        if (next_vi < num_vec) next.load(in, next_vi);
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
        cur.load(in, vi);
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
      cur.load(in, vi);
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
  using Smem = typename TopKRadixBase<12>::Smem;

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
      if (vi < num_full) local_vecs[i].load(in, vi);
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
    const auto count_gt = smem->count_gt;
    const auto count_eq = smem->count_eq;
    const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
    const auto tie_count = min(count_eq, kMaxNumTie);
    handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
  }
};

// ---------------------------------------------------------------------------
// Streaming path: seq_len > 8192 -- two vectorized passes over global memory
// ---------------------------------------------------------------------------

struct TopKStreaming : TopKRadixBase<12> {
 public:
  static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();

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
    const auto count_gt = smem->count_gt;
    const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
    const auto tie_count = min(smem->count_eq, kMaxNumTie);
    handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
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

    // Phase 3: Collect candidates and sort. Classify by two fp32 boundaries derived
    // from the threshold bin instead of recomputing the fp16 bin per element: an
    // element is "above" iff val >= v_hi (bin > threshold) and a "tie" iff
    // v_lo <= val < v_hi (bin == threshold). This drops the F2F + bit-twiddle from
    // the second full pass over the input.
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    const auto topk = problem.topk;
    const bool zero_bin = kExact && zero_bin_shortcut(v_lo);
    if (kExact && !zero_bin && smem->thr_count > kMaxNumTie) [[unlikely]] {
      // The threshold bin overflows the staging buffer (known from the histogram):
      // the collect pass builds refine_ties' first histogram instead of staging.
      return collect_overflow(problem, smem, v_lo, v_hi);
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

    // Phase 4: Handle ties. Drive the output layout from the *collect* counts so it
    // is self-consistent with the fp32 classification above (rather than the fp16
    // histogram counts), even if rounding moves a boundary element between the
    // "above" and "tie" sets. above_count is < topk by the threshold-bin invariant,
    // so the count_gt guard above effectively never triggers.
    __syncthreads();
    if constexpr (kExact) {
      if (smem->count_eq > kMaxNumTie && (!zero_bin || smem->tie_nonzero != 0)) [[unlikely]] {
        // See the register path.
        refine_ties(problem, smem, v_lo, v_hi, smem->count_eq);
      }
    }
    const auto count_gt = smem->count_gt;
    const auto count_eq = smem->count_eq;
    const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
    const auto tie_count = min(count_eq, kMaxNumTie);
    handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
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

/// Single-instruction PTX wrappers of the cluster path, on top of sgl_kernel/mbarrier.cuh.
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
// The two below differ from mbarrier.cuh: a "memory" clobber, and a cluster-scope acquire
// that pairs with the remote st.async complete_tx.
SGL_DEVICE void mbar_arrive_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(addr), "r"(bytes) : "memory");
}
SGL_DEVICE void mbar_wait(uint32_t addr, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "LAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT_%=;\n}\n" ::"r"(addr),
      "r"(parity)
      : "memory");
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
template <uint32_t N>
struct TopKCluster : TopKRadixBase<11> {
 public:
  static constexpr uint32_t kClusterSize = N;
  static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kHistItems = kHistSize / kBlockSize;  // bins per thread
  static constexpr uint32_t kGroupBins = 32;
  static constexpr uint32_t kGroups = kHistSize / kGroupBins;
  static constexpr uint32_t kGroupChunks = kGroups / 4;  // 16-byte chunks of one rank's group sums
  static constexpr uint32_t kBinRow = kGroupBins + 4;    // + this rank's count above the group, padded
  static constexpr uint32_t kBinChunks = kBinRow / 4;
  static_assert(N < kNumWarps, "phase 2d uses warps 0..N");
  static_assert(kGroups == 32 || kGroups == 64, "the group search maps 1 or 2 groups to a lane");
  static constexpr uint32_t kNegZeroBin = 0x7FFFu >> (16 - kHistBits);  // negatives that round to fp16 -0
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
  // Cold paths: 12-bit digits of the exact key relative to key(v_lo).
  static constexpr uint32_t kOvfBits = 12;
  static constexpr uint32_t kOvfSize = 1u << kOvfBits;
  static constexpr uint32_t kOvfItems = kOvfSize / kBlockSize;
  using MatchBin = TieHandleSmem::MatchBin;
  struct Smem {
    uint64_t mbar[5];    // group all-gather, bin all-gather, rank 0 tie arrivals, fine-path group / bin all-gathers
    uint32_t fine_uses;  // phases completed by mbar[3] / mbar[4] (fine path only)
    uint32_t cold_parity;
    uint32_t count_eq;
    uint32_t count_gt;
    uint32_t local_start_eq;
    uint32_t local_start_gt;
    uint32_t ovf_gt;
    uint32_t ovf_remain;
    MatchBin ovf_match;
    uint32_t warp_sum[kNumWarps];
    alignas(16) uint32_t group_local[kGroups];
    alignas(16) uint32_t recv_group[N][kGroups];
    alignas(16) uint32_t recv_bin[N][kBinRow];  // group's bins + count above the group
    uint32_t rank_suffix[N][kGroupBins + 1];    // per rank: elements >= bin b of the group
    uint32_t sel_group, above_group, own_above_group, sel_bin, above_total, eq_total;
    uint32_t tie_ctr_gt, tie_ctr_eq;  // rank 0's tie select
    MatchBin tie_digit;
    TieValue tie_stage[kWarpSize];
    union {
      alignas(16) uint32_t histogram[kHistSize];
      TieHandleSmem tie_handle;
      int32_t stage_out_idxs[kMaxTopK];
    };
    union {
      TieValue tie_values[kMaxNumTie];
      alignas(16) uint32_t ovf_hist[kOvfSize];  // overflow path only
    };
  };
  static_assert(sizeof(TieValue) * kMaxNumTie >= sizeof(uint32_t) * kOvfSize);

  SGL_DEVICE static void zero_ovf_hist(Smem* smem) {
    AlignedVector<uint32_t, kOvfItems> zero;
    zero.fill(0);
    zero.store(smem->ovf_hist, threadIdx.x);
  }

  /// Rank 0: digit search over the merged kOvfSize-bin histogram (zeroed after reading).
  SGL_DEVICE static void ovf_find_digit(Smem* smem, const uint32_t remain) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);
    AlignedVector<uint32_t, kOvfItems> counts;
    counts.load(smem->ovf_hist, tx);
    zero_ovf_hist(smem);
    if (tx == 0) smem->ovf_match = {0, 0, 0};  // stays if nothing matches (NaN-polluted counts)
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kOvfItems; ++i) {
      local_sum += counts[i];
    }
    const auto warp_inc = warp::inclusive_sum(local_sum, lane_id);
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    __syncthreads();
    const auto ws = smem->warp_sum[lane_id];
    const auto total = warp::reduce_sum(ws);
    uint32_t prefix = warp::reduce_sum(lane_id < warp_id ? ws : 0) + warp_inc - local_sum;
#pragma unroll
    for (uint32_t i = 0; i < kOvfItems; ++i) {
      prefix += counts[i];
      const auto above = total - prefix;
      if (above < remain && above + counts[i] >= remain) {
        smem->ovf_match = {tx * kOvfItems + i, above, counts[i]};
      }
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
  SGL_DEVICE static void overflow_path(const TopKProblem& problem, Smem* smem, Smem* smem_0, float v_lo, float v_hi) {
    const auto tx = threadIdx.x;
    const auto topk = problem.topk;
    zero_ovf_hist(smem);
    if (tx == 0) {
      smem->count_gt = 0;
      smem->count_eq = 0;
      smem->ovf_gt = 0;
    }
    __syncthreads();
    ptx::cluster_sync();  // every rank's counters and histogram are zero before any remote update

    // NaN-safe: a NaN upper edge (inf bin) puts nothing above and +inf in the bin.
    const auto is_above = [=](float val) { return val >= v_hi; };
    const auto in_bin = [=](float val) { return val >= v_lo && !(val >= v_hi); };
    const uint32_t key_lo = key_low(v_lo);
    const uint32_t span = extract_exact_bin(v_hi) - key_lo;  // in-bin keys are < key_lo + span
    const uint32_t num_bits = 32 - __clz(span - 1);
    // Mode 2 radixes (key, smallest index first) so the final bucket is one element
    // and the selected set is deterministic (idx < 2^20: scores are at most 1M long).
    using Key = std::conditional_t<kDeterministic, uint64_t, uint32_t>;
    constexpr uint32_t kIdxBits = kDeterministic ? 20 : 0;
    const auto comp = [=](float val, uint32_t idx) -> Key {
      const Key rel = extract_tie_key(val) - key_lo;
      if constexpr (kDeterministic) {
        return (rel << kIdxBits) | ((1u << kIdxBits) - 1 - idx);
      } else {
        return rel;
      }
    };
    const uint32_t total_bits = num_bits + kIdxBits;
    uint32_t shift = total_bits > kOvfBits ? total_bits - kOvfBits : 0;
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

    for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (is_above(val)) {
        emit_above(idx);
      } else if (in_bin(val)) {
        atomicAdd(&smem->ovf_hist[static_cast<uint32_t>(comp(val, idx) >> shift)], 1);
      }
    });
    Key bucket_lo = 0;
    Key bucket_mask = (static_cast<Key>(span - 1) << kIdxBits) | ((Key(1) << kIdxBits) - 1);
    uint32_t remain = 0;
#pragma unroll 1
    for (uint32_t round = 0;; ++round) {
      __syncthreads();
      if constexpr (!kIsRank0) {
        AlignedVector<uint32_t, kOvfItems> counts;
        counts.load(smem->ovf_hist, tx);
        zero_ovf_hist(smem);
#pragma unroll
        for (uint32_t i = 0; i < kOvfItems; ++i) {
          if (counts[i] != 0) atomicAdd(&smem_0->ovf_hist[tx * kOvfItems + i], counts[i]);
        }
        if (round == 0 && tx == 0) atomicAdd(&smem_0->ovf_gt, smem->count_gt);
        ptx::barrier_cluster_arrive_release();  // A: histograms merged
        ptx::barrier_cluster_wait();
        ptx::barrier_cluster_arrive_relaxed();  // B: digit published
        ptx::barrier_cluster_wait();
      } else {
        if (round == 0 && tx == 0) atomicAdd(&smem->ovf_gt, smem->count_gt);
        ptx::barrier_cluster_arrive_relaxed();  // A
        ptx::barrier_cluster_wait();
        __syncthreads();
        if (round == 0) {
          const auto gt = smem->ovf_gt;
          remain = gt < topk ? topk - gt : 0;
          if (tx == 0) smem->ovf_remain = remain;
        }
        if (remain != 0) ovf_find_digit(smem, remain);
        __syncthreads();
        ptx::barrier_cluster_arrive_release();  // B: publishes ovf_match / ovf_remain
        ptx::barrier_cluster_wait();
      }
      if (round == 0) {
        remain = smem_0->ovf_remain;
        if (remain == 0) break;  // above elements fill every slot
      }
      const MatchBin match = smem_0->ovf_match;
      bucket_lo += static_cast<Key>(match.bin) << shift;
      bucket_mask = (Key(1) << shift) - 1;
      remain -= match.above_count;
      if (shift == 0 || match.equal_count <= kMaxNumTie) break;
      if constexpr (kIsRank0) __syncthreads();  // all have read ovf_match before the next find rewrites it
      shift = shift > kOvfBits ? shift - kOvfBits : 0;
      for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (!in_bin(val)) return;
        const Key d = comp(val, idx) - bucket_lo;
        if (d <= bucket_mask) atomicAdd(&smem->ovf_hist[static_cast<uint32_t>(d >> shift)], 1);
      });
    }

    __syncthreads();  // everyone has read the histogram before tie_values overwrite it
    if (remain != 0) {
      for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (!in_bin(val)) return;
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
      const auto count_gt = smem->count_gt;
      const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
      const auto tie_count = min(smem->count_eq, kMaxNumTie);
      handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
    }
  }

  /// Cold path for a threshold bin with more than kMaxNumTie members (not an edge
  /// bin). The above elements are exact and emitted as on the normal path;
  /// the same pass builds a 4096-bin histogram of the bin's members over 12-bit
  /// digits of key - key(v_lo), which is all-gathered like the coarse one (64 group
  /// sums, then the selected group's 64 bins), so every rank finds the k-th digit
  /// and its bases itself. A second pass emits the members above that digit and
  /// st.asyncs the digit's members to rank 0, which resolves them. If the digit
  /// still holds more than kMaxNumTie members (runs of near-equal keys), the
  /// barrier-based refine (overflow_path) takes over.
  SGL_DEVICE static void fine_path(
      const TopKProblem& problem,
      Smem* smem,
      const uint32_t this_rank,
      const float v_lo,
      const float v_hi,
      const uint32_t above_total,
      const uint32_t base_gt,
      const uint32_t my_gt,
      const uint32_t item_parity) {
    constexpr uint32_t kFineGroups = 64;
    constexpr uint32_t kFineGroupBins = kOvfSize / kFineGroups;  // 64
    constexpr uint32_t kFineRow = kFineGroupBins + 4;
    constexpr uint32_t kLanesPerFineGroup = kFineGroupBins / kOvfItems;
    static_assert(kFineGroups <= kGroups && sizeof(uint32_t) * N * kFineRow <= sizeof(uint32_t) * kHistSize);
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);
    const auto topk = problem.topk;
    const uint32_t key_lo = key_low(v_lo);
    const uint32_t num_bits = key_bits(v_lo, v_hi);
    const uint32_t shift = num_bits > kOvfBits ? num_bits - kOvfBits : 0;
    const uint32_t remain = topk - above_total;  // 0 < remain < members of the bin
    const uint32_t parity = smem->fine_uses & 1;
    const auto recv2 = reinterpret_cast<uint32_t (*)[kFineRow]>(smem->histogram);

    zero_ovf_hist(smem);
    if (tx == 0) smem->local_start_gt = 0;  // counter of the second pass
    __syncthreads();
    // Pass 1: above as on the normal path; members into the fine histogram.
    for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < my_gt) [[likely]] {
          problem.emit(base_gt + pos, idx);
        }
      } else if (val >= v_lo) {
        atomicAdd(&smem->ovf_hist[(extract_tie_key(val) - key_lo) >> shift], 1);
      }
    });
    __syncthreads();
    if (const auto got_gt = smem->count_gt; got_gt < my_gt) {
      for (uint32_t t = got_gt + tx; t < my_gt; t += kBlockSize) {
        problem.emit(base_gt + t, base_gt + t);
      }
    }
    // F1: group sums, all-gathered.
    {
      AlignedVector<uint32_t, kOvfItems> v;
      v.load(smem->ovf_hist, tx);
      uint32_t sum = 0;
#pragma unroll
      for (uint32_t i = 0; i < kOvfItems; ++i) {
        sum += v[i];
      }
      sum = warp::reduce_sum<kLanesPerFineGroup>(sum);
      if (tx % kLanesPerFineGroup == 0) smem->group_local[tx / kLanesPerFineGroup] = sum;
    }
    if (tx == 0) ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar[3]), N * kFineGroups * sizeof(uint32_t));
    __syncthreads();
    if (tx < N * (kFineGroups / 4)) {
      const uint32_t c = tx % (kFineGroups / 4), dst = tx / (kFineGroups / 4);
      // Raw uint4 / uint2 views of the DSMEM rows: AlignedVector adds instructions here.
      const auto v = reinterpret_cast<const uint4*>(smem->group_local)[c];
      ptx::st_async_v4(
          ptx::map_rank(ptx::to_shared(&smem->recv_group[this_rank][c * 4]), dst),
          v,
          ptx::map_rank(ptx::to_shared(&smem->mbar[3]), dst));
    }
    if (warp_id == 0) {
      ptx::mbar_wait(ptx::to_shared(&smem->mbar[3]), parity);
      uint32_t t0 = 0, t1 = 0;
#pragma unroll
      for (uint32_t r = 0; r < N; ++r) {
        const auto v = reinterpret_cast<const uint2*>(smem->recv_group[r])[lane_id];
        t0 += v.x;
        t1 += v.y;
      }
      const auto incl = warp::inclusive_sum(t0 + t1, lane_id);
      const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
      const uint32_t above1 = total - incl, above0 = above1 + t1;
      const bool hit1 = above1 < remain && above1 + t1 >= remain;
      const bool hit0 = above0 < remain && above0 + t0 >= remain;
      const auto mask = __ballot_sync(0xFFFFFFFF, hit0 || hit1);
      const auto src = mask ? __ffs(mask) - 1 : 0;
      const auto g = __shfl_sync(0xFFFFFFFF, 2 * lane_id + (hit1 ? 1 : 0), src);
      const auto g_above = __shfl_sync(0xFFFFFFFF, hit1 ? above1 : above0, src);
      uint32_t own = (2 * lane_id > g ? smem->group_local[2 * lane_id] : 0) +
                     (2 * lane_id + 1 > g ? smem->group_local[2 * lane_id + 1] : 0);
      own = warp::reduce_sum(own);
      if (lane_id == 0) {
        smem->sel_group = g;
        smem->above_group = g_above;
        smem->own_above_group = own;
      }
    }
    if (tx == 0) ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar[4]), N * kFineRow * sizeof(uint32_t));
    __syncthreads();
    // F2: the selected group's 64 bins plus the count above the group.
    if (tx < N * (kFineRow / 4)) {
      const uint32_t c = tx % (kFineRow / 4), dst = tx / (kFineRow / 4);
      const auto g = smem->sel_group;
      const auto v = c < kFineGroupBins / 4 ? reinterpret_cast<const uint4*>(smem->ovf_hist + g * kFineGroupBins)[c]
                                            : uint4{smem->own_above_group, 0, 0, 0};
      ptx::st_async_v4(
          ptx::map_rank(ptx::to_shared(&recv2[this_rank][c * 4]), dst),
          v,
          ptx::map_rank(ptx::to_shared(&smem->mbar[4]), dst));
    }
    if (warp_id == 0) {
      ptx::mbar_wait(ptx::to_shared(&smem->mbar[4]), parity);
      uint32_t t0 = 0, t1 = 0;
#pragma unroll
      for (uint32_t r = 0; r < N; ++r) {
        const auto v = reinterpret_cast<const uint2*>(recv2[r])[lane_id];
        t0 += v.x;
        t1 += v.y;
      }
      const auto incl = warp::inclusive_sum(t0 + t1, lane_id);
      const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
      const uint32_t above1 = smem->above_group + total - incl, above0 = above1 + t1;
      const bool hit1 = above1 < remain && above1 + t1 >= remain;
      const bool hit0 = above0 < remain && above0 + t0 >= remain;
      const auto mask = __ballot_sync(0xFFFFFFFF, hit0 || hit1);
      const auto src = mask ? __ffs(mask) - 1 : 0;
      const auto b = __shfl_sync(0xFFFFFFFF, 2 * lane_id + (hit1 ? 1 : 0), src);
      const auto b_above = __shfl_sync(0xFFFFFFFF, hit1 ? above1 : above0, src);
      const auto b_count = __shfl_sync(0xFFFFFFFF, hit1 ? t1 : t0, src);
      if (lane_id == 0) {
        smem->sel_bin = b;
        smem->above_total = b_above;
        smem->eq_total = b_count;
      }
    } else if (warp_id <= N) {
      ptx::mbar_wait(ptx::to_shared(&smem->mbar[4]), parity);
    }
    __syncthreads();
    const uint32_t b = smem->sel_bin;
    const uint32_t digit = smem->sel_group * kFineGroupBins + b;
    const uint32_t above_digit = smem->above_total;  // bin members above the digit, cluster-wide
    const uint32_t eq_digit = smem->eq_total;
    if (warp_id < N) {
      const auto row = recv2[warp_id];
      const auto v = reinterpret_cast<const uint2*>(row)[lane_id];
      uint32_t gt = (2 * lane_id > b ? v.x : 0) + (2 * lane_id + 1 > b ? v.y : 0);
      gt = warp::reduce_sum(gt) + row[kFineGroupBins];
      if (lane_id == 0) {
        smem->rank_suffix[warp_id][0] = gt;
        smem->rank_suffix[warp_id][1] = row[b];
      }
    }
    __syncthreads();
    uint32_t base_gt2, base_eq2, my_gt2, my_eq2;
    {
      const bool mine = lane_id < N;
      const auto gt = mine ? smem->rank_suffix[lane_id][0] : 0u;
      const auto eq = mine ? smem->rank_suffix[lane_id][1] : 0u;
      const bool before = lane_id < this_rank;
      base_gt2 = warp::reduce_sum(before ? gt : 0u);
      base_eq2 = warp::reduce_sum(before ? eq : 0u);
      my_gt2 = __shfl_sync(0xFFFFFFFF, gt, this_rank);
      my_eq2 = __shfl_sync(0xFFFFFFFF, eq, this_rank);
    }
    if (tx == 0) smem->fine_uses += 1;  // read again only by the next item, after its first barrier

    if (eq_digit > kMaxNumTie) [[unlikely]] {
      if (this_rank == 0 && tx == 0) ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar[2]), 0);
      const auto smem_0 = ptx::map_rank(smem, 0);
      if (this_rank == 0) {
        overflow_path<true>(problem, smem, smem, v_lo, v_hi);
        ptx::mbar_wait(ptx::to_shared(&smem->mbar[2]), item_parity);
      } else {
        overflow_path<false>(problem, smem, smem_0, v_lo, v_hi);
      }
      return;
    }

    // Pass 2: members above the digit -> output; the digit's members -> rank 0.
    if (this_rank == 0) {
      init_histogram(smem->histogram, tx);  // recv2 consumed; tie_select's digit histogram
      if (tx == 0) ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar[2]), (eq_digit - my_eq2) * sizeof(TieValue));
    }
    const uint32_t out_base = above_total + base_gt2;
    const uint32_t tie_dst = ptx::map_rank(ptx::to_shared(&smem->tie_values[base_eq2]), 0);
    const uint32_t tie_bar = ptx::map_rank(ptx::to_shared(&smem->mbar[2]), 0);
    for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (!(val >= v_lo && !(val >= v_hi))) return;
      const auto d = (extract_tie_key(val) - key_lo) >> shift;
      if (d > digit) {
        const auto pos = atomicAdd(&smem->local_start_gt, 1);
        if (pos < my_gt2) problem.emit(out_base + pos, idx);
      } else if (d == digit) {
        const auto pos = atomicAdd(&smem->count_eq, 1);
        if (pos < my_eq2) {
          if (this_rank == 0) {
            smem->tie_values[pos] = {val, idx};
          } else {
            ptx::st_async_v2(tie_dst + pos * sizeof(TieValue), __float_as_uint(val), idx, tie_bar);
          }
        }
      }
    });
    __syncthreads();
    if (this_rank == 0) {
      if (tx < kWarpSize) ptx::mbar_wait(ptx::to_shared(&smem->mbar[2]), item_parity);
      __syncthreads();
      const auto base = above_total + above_digit;
      const auto need = remain - above_digit;
      if (eq_digit <= kWarpSize * 4 || eq_digit <= need) {
        handle_tie(smem->tie_values, problem, base, eq_digit, need, &smem->tie_handle);
      } else {
        tie_select(problem, smem, base, eq_digit, need, key_lo + (digit << shift), shift);
      }
    }
  }

  /// Every cold case (the threshold bin holds more than kMaxNumTie members). Takes
  /// only the problem: the state is re-read from shared memory, so a call to it
  /// (cold_path_noinline) keeps few registers live across the hot path.
  SGL_DEVICE static void cold_path(const TopKProblem problem, Smem* smem) {
    const uint32_t parity = smem->cold_parity;
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const uint32_t this_rank = blockIdx.y;
    const uint32_t b = smem->sel_bin;
    const uint32_t sel_bin = smem->sel_group * kGroupBins + b;
    const auto [v_lo, v_hi] = bin_bounds(sel_bin);
    const bool mine = lane_id < N;
    const auto gt = mine ? smem->rank_suffix[lane_id][b + 1] : 0u;
    const auto base_gt = warp::reduce_sum(lane_id < this_rank ? gt : 0u);
    const auto my_gt = __shfl_sync(0xFFFFFFFF, gt, this_rank);
    const auto above_total = smem->above_total;
    __syncthreads();  // everyone has read the 2d results before the fine path reuses them
    fine_path(problem, smem, this_rank, v_lo, v_hi, above_total, base_gt, my_gt, parity);
  }
  __device__ __noinline__ static void cold_path_noinline(const TopKProblem problem, Smem* smem) {
    cold_path(problem, smem);
  }

  /// Rank 0: exact select of `remain` of the `n` staged ties (n > remain, n <= kMaxNumTie).
  /// One 11-bit digit of the key relative to key(v_lo) over the zeroed histogram;
  /// elements above the k-th digit are emitted, the k-th digit's members are ranked
  /// by (value desc, idx asc) in one warp when they are at most 32, else
  /// compacted and handed to the generic handle_tie.
  SGL_DEVICE static void tie_select(
      const TopKProblem& problem,
      Smem* smem,
      uint32_t base,
      uint32_t n,
      uint32_t remain,
      uint32_t key_lo,
      uint32_t num_bits) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
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
      const auto ws = smem->warp_sum[lane_id];
      uint32_t prefix = warp::reduce_sum(lane_id < warp_id ? ws : 0) + warp_inc - local;  // ties in lower digits
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

  /// Selects the top-k of `problem` across the cluster, cold rows included.
  /// `item`: 0 for the first problem this CTA runs (initializes the mbarriers), +1 per later one.
  /// kInlineCold: inline the cold path instead of calling it. A call makes ptxas keep per-item
  /// state on the stack in a 32-register kernel (the persistent pool); inlining it makes the
  /// small-batch kernel, which also holds the one-block paths, spill to shared memory.
  template <bool kUsePDL, bool kInlineCold = false>
  SGL_DEVICE static void forward(TopKProblem problem, void* _smem, const uint32_t item = 0) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto smem = static_cast<Smem*>(_smem);
    const uint32_t this_rank = blockIdx.y;
    const uint32_t parity = item & 1;
    const auto topk = problem.topk;

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
      smem->tie_ctr_gt = 0;
      smem->tie_ctr_eq = 0;
      if (item == 0) {
#pragma unroll
        for (uint32_t i = 0; i < 5; ++i) {
          ptx::mbar_init(&smem->mbar[i], 1);
        }
        ptx::fence_mbar_init();
        smem->fine_uses = 0;
      }
    }
    __syncthreads();
    // bar-0: every rank's mbarriers are initialized (and the previous item's
    // receive buffers consumed) before any rank pushes into them.
    ptx::barrier_cluster_arrive_release();
    PDLWaitPrimary<kUsePDL>();

    // Phase 1: histogram over this rank's interleaved chunks.
    for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t) {
      // Bin -0.0 as +0.0 (x + 0 under round-to-nearest), as fp32 compares see it.
      atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(__fadd_rn(val, 0.0f))], 1);
    });
    __syncthreads();

    // Phase 2a: group sums (32 bins per group), all-gathered.
    AlignedVector<uint32_t, kHistItems> my_bins;
    my_bins.load(smem->histogram, tx);
    {
      constexpr uint32_t kLanesPerGroup = kGroupBins / kHistItems;
      uint32_t sum = 0;
#pragma unroll
      for (uint32_t i = 0; i < kHistItems; ++i) {
        sum += my_bins[i];
      }
      sum = warp::reduce_sum<kLanesPerGroup>(sum);
      if (tx % kLanesPerGroup == 0) smem->group_local[tx / kLanesPerGroup] = sum;
    }
    if (tx == 0) ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar[0]), N * kGroups * sizeof(uint32_t));
    __syncthreads();
    ptx::barrier_cluster_wait();  // bar-0 wait
    if (tx < N * kGroupChunks) {
      const uint32_t c = tx % kGroupChunks, dst = tx / kGroupChunks;
      // Raw uint4 / uint2 views of the DSMEM rows: AlignedVector adds instructions here.
      const auto v = reinterpret_cast<const uint4*>(smem->group_local)[c];
      ptx::st_async_v4(
          ptx::map_rank(ptx::to_shared(&smem->recv_group[this_rank][c * 4]), dst),
          v,
          ptx::map_rank(ptx::to_shared(&smem->mbar[0]), dst));
    }
    // Only the warps that read a received buffer poll its mbarrier; the rest go
    // on to the next CTA barrier (no spinning next to a co-resident CTA).

    // Phase 2b: warp 0 finds the group holding the k-th element and this rank's
    // count above that group.
    const auto warp_id = broadcast(tx / kWarpSize);
    if (warp_id == 0) {
      ptx::mbar_wait(ptx::to_shared(&smem->mbar[0]), parity);
      uint32_t t[kGroups / 32];
#pragma unroll
      for (uint32_t i = 0; i < kGroups / 32; ++i) {
        t[i] = 0;
      }
#pragma unroll
      for (uint32_t r = 0; r < N; ++r) {
        if constexpr (kGroups == 32) {
          t[0] += smem->recv_group[r][lane_id];
        } else {
          const auto v = reinterpret_cast<const uint2*>(smem->recv_group[r])[lane_id];
          t[0] += v.x;
          t[1] += v.y;
        }
      }
      uint32_t lane_sum = 0;
#pragma unroll
      for (uint32_t i = 0; i < kGroups / 32; ++i) {
        lane_sum += t[i];
      }
      const auto incl = warp::inclusive_sum(lane_sum, lane_id);
      const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
      uint32_t above = total - incl;  // groups of higher lanes
      uint32_t hit = 0, hit_above = 0;
#pragma unroll
      for (int i = kGroups / 32 - 1; i >= 0; --i) {
        if (above < topk && above + t[i] >= topk) {
          hit = i + 1;
          hit_above = above;
        }
        above += t[i];
      }
      const auto mask = __ballot_sync(0xFFFFFFFF, hit != 0);
      const auto src = mask ? __ffs(mask) - 1 : 0;
      auto g = __shfl_sync(0xFFFFFFFF, lane_id * (kGroups / 32) + hit - 1, src);
      auto g_above = __shfl_sync(0xFFFFFFFF, hit_above, src);
      if (mask == 0) {  // only with NaN-polluted counts
        g = 0;
        g_above = 0;
      }
      // This rank's elements in groups above g.
      uint32_t own = 0;
#pragma unroll
      for (uint32_t i = 0; i < kGroups / 32; ++i) {
        const auto grp = lane_id * (kGroups / 32) + i;
        own += grp > g ? smem->group_local[grp] : 0;
      }
      own = warp::reduce_sum(own);
      if (lane_id == 0) {
        smem->sel_group = g;
        smem->above_group = g_above;
        smem->own_above_group = own;
      }
    }
    if (tx == 0) ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar[1]), N * kBinRow * sizeof(uint32_t));
    __syncthreads();

    // Phase 2c: all-gather this rank's bins of the selected group, plus its count above the group.
    if (tx < N * kBinChunks) {
      const uint32_t c = tx % kBinChunks, dst = tx / kBinChunks;
      const auto sel_group = smem->sel_group;
      const auto v = c < kGroupBins / 4 ? reinterpret_cast<const uint4*>(smem->histogram + sel_group * kGroupBins)[c]
                                        : uint4{smem->own_above_group, 0, 0, 0};
      ptx::st_async_v4(
          ptx::map_rank(ptx::to_shared(&smem->recv_bin[this_rank][c * 4]), dst),
          v,
          ptx::map_rank(ptx::to_shared(&smem->mbar[1]), dst));
    }
    if (tx < (N + 1) * kWarpSize) ptx::mbar_wait(ptx::to_shared(&smem->mbar[1]), parity);

    // Phase 2d: warps 0..N-1 turn rank w's row into "elements >= bin b" suffix
    // counts; warp N finds the threshold bin. Every thread then derives its bases.
    if (warp_id < N) {
      const auto row = smem->recv_bin[warp_id];
      const auto incl = warp::inclusive_sum(row[lane_id], lane_id);
      const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
      const auto prev = __shfl_up_sync(0xFFFFFFFF, incl, 1);
      smem->rank_suffix[warp_id][lane_id] = row[kGroupBins] + total - (lane_id == 0 ? 0 : prev);
      if (lane_id == 0) smem->rank_suffix[warp_id][kGroupBins] = row[kGroupBins];
    } else if (warp_id == N) {
      uint32_t tb = 0;
#pragma unroll
      for (uint32_t r = 0; r < N; ++r) {
        tb += smem->recv_bin[r][lane_id];
      }
      const auto incl = warp::inclusive_sum(tb, lane_id);
      const auto total = __shfl_sync(0xFFFFFFFF, incl, kWarpSize - 1);
      const auto above = smem->above_group + total - incl;
      const bool hit = above < topk && above + tb >= topk;
      const auto mask = __ballot_sync(0xFFFFFFFF, hit);
      const auto b = mask ? __ffs(mask) - 1 : 0;
      const auto above_b = __shfl_sync(0xFFFFFFFF, above, b);
      const auto eq_b = __shfl_sync(0xFFFFFFFF, tb, b);
      if (lane_id == 0) {
        smem->sel_bin = b;
        smem->above_total = above_b;
        smem->eq_total = eq_b;
      }
    }
    __syncthreads();
    const uint32_t b = smem->sel_bin;
    const uint32_t sel_bin = smem->sel_group * kGroupBins + b;
    const uint32_t above_total = smem->above_total;
    const uint32_t eq_total = smem->eq_total;
    // Rank 0's coarse histogram is dead (pushed, consumed): it becomes the tie select's digit histogram.
    if (this_rank == 0) init_histogram(smem->histogram, tx);
    uint32_t base_gt, base_eq, my_gt, my_eq;
    {
      const bool mine = lane_id < N;
      const auto ge = mine ? smem->rank_suffix[lane_id][b] : 0u;      // elements >= bin b
      const auto gt = mine ? smem->rank_suffix[lane_id][b + 1] : 0u;  // elements > bin b
      const bool before = lane_id < this_rank;
      base_gt = warp::reduce_sum(before ? gt : 0u);
      base_eq = warp::reduce_sum(before ? ge - gt : 0u);
      my_gt = __shfl_sync(0xFFFFFFFF, gt, this_rank);
      my_eq = __shfl_sync(0xFFFFFFFF, ge, this_rank) - my_gt;
      // CTA-uniform: let them live in uniform registers through the collect loop.
      base_gt = broadcast(base_gt);
      base_eq = broadcast(base_eq);
      my_gt = broadcast(my_gt);
      my_eq = broadcast(my_eq);
    }
    const auto [v_lo, v_hi] = bin_bounds(sel_bin);

    // Cold path (exact modes): the bin overflows the tie buffer.
    if (kExact && eq_total > kMaxNumTie) [[unlikely]] {
      // The cold path re-derives everything else from shared memory.
      if (tx == 0) smem->cold_parity = parity;
      __syncthreads();
      if constexpr (kInlineCold) {
        cold_path(problem, smem);
      } else {
        cold_path_noinline(problem, smem);
      }
      return;
    }

    // Phase 3: collect. Above -> output at base_gt; ties -> rank 0's tie buffer at base_eq.
    // The collected counts equal the histogram's (bin_bounds), except that a NaN
    // score is counted but never collected, so no store needs a bound; mode 0 only
    // truncates the tie buffer at kMaxNumTie (in rank order). NaN holes are filled below.
    const auto v_hi_u = broadcast(v_hi), v_lo_u = broadcast(v_lo);
    const uint32_t tie_cap = broadcast(min(my_eq, base_eq < kMaxNumTie ? kMaxNumTie - base_eq : 0u));
    const uint32_t tie_total = min(eq_total, kMaxNumTie);
    if (this_rank == 0) {
      if (tx == 0) ptx::mbar_arrive_expect_tx(ptx::to_shared(&smem->mbar[2]), (tie_total - tie_cap) * sizeof(TieValue));
      for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (val >= v_hi_u) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          problem.emit(pos, idx);
        } else if (val >= v_lo_u) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (kExact || pos < tie_cap) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      });
    } else {
      const uint32_t tie_dst = broadcast(ptx::map_rank(ptx::to_shared(&smem->tie_values[base_eq]), 0));
      const uint32_t tie_bar = broadcast(ptx::map_rank(ptx::to_shared(&smem->mbar[2]), 0));
      auto out = problem;
      out.out += base_gt;
      for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (val >= v_hi_u) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          out.emit(pos, idx);
        } else if (val >= v_lo_u) {
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
                ptx::map_rank(ptx::to_shared(&smem->mbar[2]), 0));
          }
        }
        __syncthreads();
      }
    }
    if (this_rank == 0) {
      if (tx < kWarpSize) ptx::mbar_wait(ptx::to_shared(&smem->mbar[2]), parity);
      __syncthreads();
      const auto remain_topk = above_total < topk ? topk - above_total : 0;
      if (tie_total <= kWarpSize * 4 || tie_total <= remain_topk) {
        handle_tie(smem->tie_values, problem, above_total, tie_total, remain_topk, &smem->tie_handle);
      } else {
        tie_select(problem, smem, above_total, tie_total, remain_topk, key_low(v_lo), key_bits(v_lo, v_hi));
      }
    }
  }
};

#endif  // !USE_ROCM

}  // namespace SGL_TOPK_MODE_NAMESPACE

}  // namespace device::topk

}  // namespace sglang
