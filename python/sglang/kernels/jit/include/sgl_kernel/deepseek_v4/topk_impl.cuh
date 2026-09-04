/// \file topk_impl.cuh
/// \brief DeepSeek-V4 (DSA indexer) top-k implementation classes.
///
/// This header holds ONLY the device-side implementation classes + helpers; the
/// `__global__` kernels and the host dispatcher live in csrc/deepseek_v4/topk_v2.cuh.
///
/// Design notes:
///  - top-k (`topk`) is a *runtime* value (<= kMaxTopK = 2048), never a
///    compile-time constant.
///  - the output is the page-table transform of the selected raw indices
///    (`TopKProblem::emit` then `transform_output`).
///  - each block reads its own `seq_len` (per-batch ragged lengths) -- the host
///    launches one universal kernel and dispatches per block.
///  - the cluster size is fixed at 8 (dynamic persistent clusters are hard).
///
/// Algorithm: fp16 coarse histogram -> threshold bin -> fp32-boundary collect ->
/// exact radix tie-break. A threshold coarse bin that holds more candidates
/// than the staging buffer takes an extra exact-key radix refinement first (see
/// TopKRadixBase::refine_and_handle_tie).

#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cfloat>
#include <cstdint>
#include <limits>

#ifndef USE_ROCM
#include <cooperative_groups.h>
#endif

namespace sglang {

namespace device::topk {

#ifndef USE_ROCM
namespace cg = cooperative_groups;
#endif

/// sgl_kernel names the warp size `kWarpThreads`; alias it locally as `kWarpSize`.
inline constexpr uint32_t kWarpSize = kWarpThreads;

// ---------------------------------------------------------------------------
// Shared-memory storage sized/aligned for several impl `Smem` types
// ---------------------------------------------------------------------------

/// Compile-time max over a non-empty pack (avoids an <algorithm> dependency).
template <typename T>
constexpr T ct_max(T a) {
  return a;
}
template <typename T, typename... Ts>
constexpr T ct_max(T a, Ts... rest) {
  const T m = ct_max(rest...);
  return a > m ? a : m;
}

/// Static shared-memory buffer sized + aligned to hold any one of the given
/// impl `Smem` types. A kernel that dispatches across several paths (e.g. the
/// fused small-batch kernel runs either Streaming or Cluster; the main kernel
/// runs any of Register2/Register4/Streaming) declares one
/// `__shared__ MaxSmem<...> smem` and hands `&smem` to whichever forward() it
/// calls -- instead of hand-picking "the largest" type and relying on it
/// staying the largest. `&smem` converts to the `void*` the forwards expect;
/// the buffer is aligned to the strictest member, so the cast is well-aligned.
template <typename... Smems>
struct MaxSmem {
  static constexpr size_t kSize = ct_max(sizeof(Smems)...);
  static constexpr size_t kAlign = ct_max(alignof(Smems)...);
  alignas(kAlign) uint8_t storage[kSize];
};

// ---------------------------------------------------------------------------
// Order-preserving float -> integer key extraction
// ---------------------------------------------------------------------------

SGL_DEVICE uint32_t extract_exact_bin(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  const auto hx = cast<fp16_t>(x);
  const uint16_t bits = *reinterpret_cast<const uint16_t*>(&hx);
  const uint16_t key = (bits & 0x8000) ? ~bits : bits | 0x8000;
  return key >> (16 - kBits);
}

// Smallest fp32 value `v` for which `extract_coarse_bin<kBits>(v) >= bin`, i.e. the
// lower fp32 boundary of coarse bin `bin`. Because `extract_coarse_bin` is monotonic
// non-decreasing in its argument, the collect pass can classify an element with two
// fp32 comparisons against these boundaries instead of recomputing the fp16 bin --
// removing the F2F conversion and bit-twiddle from the (compute-bound) second pass.
// Returns -inf for bin 0 (everything qualifies) and +inf for bins past the top.
template <uint32_t kBits>
SGL_DEVICE float coarse_bin_lower_bound(uint32_t bin) {
  constexpr uint32_t kShift = 16 - kBits;
  const uint32_t key = bin << kShift;  // ordered16 key at the low edge of `bin`
  // ordered16 -> fp16 value (inverse of the transform in extract_coarse_bin);
  // finite keys only.
  const auto to_finite_val = [](uint32_t okey) -> float {
    const uint16_t ob = static_cast<uint16_t>(okey);
    const uint16_t hb = (ob & 0x8000) ? static_cast<uint16_t>(ob ^ 0x8000) : static_cast<uint16_t>(~ob);
    return cast<float>(*reinterpret_cast<const fp16_t*>(&hb));
  };
  // Fast path, hoisted above the per-key special cases so both keys are
  // range-checked at once: `key` and `key - 1` both land in the finite band
  // [0x0401, 0xFBFF] -- every boundary a finite-score threshold produces.
  // fp16 rounds to nearest, so the fp32 boundary is the midpoint between the
  // fp16 values at `key` and `key - 1`. (Verified bit-exact against the slow
  // path for every bin of kBits 10 and 12, and measured faster than either
  // per-key dispatch or an ordered-bit decrement trick -- the two conversions
  // are independent and issue in parallel.)
  if (key - 0x0401u <= 0xFBFFu - 0x0401u && bin < (1u << kBits)) {
    return 0.5f * (to_finite_val(key) + to_finite_val(key - 1));
  }
  // Slow path: an edge of `bin` touches the +/-inf keys or NaN key space.
  // The ordered-key line is: [0, 0x03FF) negative-NaN space, 0x03FF = -inf,
  // [0x0400, 0xFC00) finite, 0xFC00 = +inf, (0xFC00, 0xFFFF] positive-NaN
  // space. Treat the +/-inf keys as +/-65536 (one ideal step past fp16 max,
  // so the midpoint lands exactly on +/-65520 -- the fp32->fp16
  // round-to-nearest overflow threshold) and saturate NaN-space keys, keeping
  // the returned boundaries finite-or-inf and monotone. Otherwise a threshold
  // bin at/next to the inf bin gets NaN boundaries, the collect pass matches
  // nothing, and rows whose scores contain >= topk (+/-)inf or >65504 values
  // come back short -- the padded slots then illegal-address downstream.
  if (bin == 0) return -FLT_MAX;
  if (bin >= (1u << kBits)) return FLT_MAX;
  const auto to_val = [&](uint32_t okey) -> float {
    constexpr float k_Inf = std::numeric_limits<float>::infinity();
    if (okey < 0x03FFu) return -k_Inf;
    if (okey == 0x03FFu) return -65536.0f;
    if (okey == 0xFC00u) return 65536.0f;
    if (okey > 0xFC00u) return FLT_MAX;
    return to_finite_val(okey);
  };
  return 0.5f * (to_val(key) + to_val(key - 1));
}

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
#ifndef USE_ROCM
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
#else
    uint32_t n = __shfl_up_sync(kFullMask, val, offset, kWarpThreads);
#endif
    if (lane_id >= offset) val += n;
  }
  return val;
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
    return TieValue{-FLT_MAX, 0xFFFFFFFFu};
  }
};

// ---------------------------------------------------------------------------
// Per-batch problem description + page-table transform sink
// ---------------------------------------------------------------------------

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

/// One batch element's worth of work. `emit(pos, raw_idx)` writes the selected raw
/// index to output slot `pos`; `transform_output` then applies the page-table
/// transform in a separate pass.
struct TopKProblem {
  const float* __restrict__ in;
  int32_t* __restrict__ out;  // page_indices [topk]
  const int32_t* __restrict__ page_table;
  uint32_t topk;
  uint32_t seq_len;
  uint32_t page_bits;
  int32_t bias = 0;  // needed by ragged mode

  SGL_DEVICE void emit(uint32_t pos, uint32_t raw_idx) const {
    out[pos] = static_cast<int32_t>(raw_idx) + bias;
  }
  SGL_DEVICE void transform_output(uint32_t t, int32_t raw) const {
    out[t] = raw < 0 ? -1 : page_to_indices(page_table, raw, page_bits);
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
  static constexpr uint32_t kTopKItems = (kMaxTopK + kBlockSize - 1) / kBlockSize;
  // tie candidates owned per thread in the strided handle_tie loops
  static constexpr uint32_t kTieItems = kMaxNumTie / kBlockSize;
  static_assert(kMaxNumTie >= kMaxTopK && kMaxNumTie % kBlockSize == 0 && kBlockSize % kNumWarps == 0);

  struct TieHandleSmem {
    struct alignas(16) MatchBin {
      uint32_t bin;
      uint32_t above_count;
      uint32_t equal_count;
      uint32_t _pad = 0;
    };
    alignas(128) uint32_t counter;
    alignas(128) uint32_t counter_final;
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
    const auto warp_id = tx / kWarpSize;
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
      radix_tie_select<1>(tie_buffer, problem, base, num_ties, topk, smem);
    } else {
      // Rare overflow case (kBlockSize < num_ties <= kMaxNumTie), kept out of
      // the common path so it alone pays the multi-item register cost.
      radix_tie_select<kTieItems>(tie_buffer, problem, base, num_ties, topk, smem);
    }
  }

  /// Threshold-byte search over the 256-bin refinement histogram in
  /// `smem->histogram[0]`. Mirrors the search inside radix_tie_select: exactly
  /// one bin `b` satisfies `above(b) < topk_remain <= above(b) + count(b)`,
  /// where `above(b)` counts the still-active elements in bins strictly above
  /// `b`. Publishes it in `smem->match`. Block-wide; ends with a barrier.
  SGL_DEVICE static void refine_find_threshold(  //
      const uint32_t total_active,
      const uint32_t topk_remain,
      TieHandleSmem* smem) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    uint32_t hist_val = 0;
    uint32_t warp_inc = 0;
    if (tx < kRadixSize) {
      hist_val = smem->histogram[0][tx];
      warp_inc = warp_inclusive_sum(lane_id, hist_val);
      if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    }
    __syncthreads();
    if (tx < kRadixSize) {
      const auto inter = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
      const auto prefix = inter + warp_inc;      // inclusive prefix through this bin
      const auto above = total_active - prefix;  // active elements in bins ABOVE this one
      if (above < topk_remain && above + hist_val >= topk_remain) {
        smem->match = {tx, above, hist_val};
      }
    }
    __syncthreads();
  }

  /// Exact radix select over the tie candidates: each thread owns kItems
  /// strided elements (inactive beyond num_ties). Requires
  /// num_ties <= kItems * kBlockSize.
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
    const auto warp_id = tx / kWarpSize;

    bool active[kItems];
    uint32_t key[kItems];
    uint32_t idx[kItems];
    uint32_t write_pos[kItems];
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      const auto t = tx + i * kBlockSize;
      active[i] = t < num_ties;
      const auto tie = active[i] ? tie_buffer[t] : TieValue::invalid();
      key[i] = extract_exact_bin(tie.value);
      idx[i] = tie.idx;
      write_pos[i] = topk;
    }
    uint32_t topk_remain = topk;
    if (tx < kRadixSize) smem->histogram[0][tx] = 0;
    if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
    __syncthreads();
    uint32_t total_active = num_ties;

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
        warp_inc = warp_inclusive_sum(lane_id, hist_val);
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

      const auto [threshold_bin, above_count, equal_count, __] = smem->match;
      if (round < 3) total_active = equal_count;
      topk_remain -= above_count;

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
          write_pos[i] = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
        }
        // my_bin == thr && round < 3: stay active for next round
      }

      if (round == 3 || topk_remain == 0) break;
    }

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      if (write_pos[i] < topk) problem.emit(base + write_pos[i], idx[i]);
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

  struct Smem {
    using kHistVec = AlignedVector<uint32_t, kHistSize / kBlockSize>;
    alignas(128) uint32_t count_eq;
    alignas(128) uint32_t count_gt;
    uint32_t threshold_bin;
    uint32_t warp_sum[kNumWarps];
    // The coarse histogram is dead once find_threshold() has published
    // threshold_bin, and the tie machinery only comes alive after that: the
    // collect pass fills tie.values, then handle_tie works over them with
    // tie.handle as scratch. Overlaying the two phases keeps the
    // kMaxNumTie-candidate buffer from growing the block's shared-memory
    // footprint. tie.handle and tie.values are live TOGETHER, so they sit
    // side by side inside the overlay, not in a union with each other.
    union {
      uint32_t histogram[kHistSize];
      kHistVec hist_vecs[kBlockSize];
      struct {
        TieHandleSmem handle;
        TieValue values[kMaxNumTie];
      } tie;
    };
  };

 protected:
  template <typename F>
  SGL_DEVICE static void for_each_input(const float* __restrict__ in, uint32_t seq_len, F&& fn) {
    const auto tx = threadIdx.x;
    const uint32_t num_full = seq_len / kVecSize;  // fully-in-bounds vectors

    vec_t next_vec;
    uint32_t vi = tx;
    if (vi < num_full) next_vec.load(in, vi);
    while (vi < num_full) {
      const auto cur = next_vec;
      const auto base = vi * kVecSize;
      vi += kBlockSize;
      if (vi < num_full) next_vec.load(in, vi);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        fn(cur[j], base + j);
      }
    }

    // Tail: at most one partial vector, `rem` in [0, kVecSize).
    static_assert(kVecSize <= kBlockSize);  // ensure tail correctness
    const uint32_t tail_start = num_full * kVecSize;
    if (tx < seq_len - tail_start) {
      const auto idx = tail_start + tx;
      fn(in[idx], idx);
    }
  }

  SGL_DEVICE static void find_threshold(const uint32_t topk, const uint32_t seq_len, Smem* smem) {
    const auto tx = threadIdx.x;
    constexpr uint32_t kItems = kHistSize / kBlockSize;
    uint32_t orig[kItems];
    const auto hist_vec = smem->hist_vecs[tx];
    uint32_t tmp_local_sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      orig[i] = hist_vec[i];
      tmp_local_sum += orig[i];
    }

    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    const auto warp_inc = warp_inclusive_sum(lane_id, tmp_local_sum);
    const auto warp_exc = warp_inc - tmp_local_sum;
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;

    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    // Exactly one bin satisfies: above < K && above + count >= K
    uint32_t prefix_sum = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
    prefix_sum += warp_exc;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      prefix_sum += orig[i];
      const auto above = seq_len - prefix_sum;
      if (above < topk && above + orig[i] >= topk) {
        smem->threshold_bin = tx * kItems + i;
      }
    }
    __syncthreads();
  }

  /// Exact-key refinement of an OVERFLOWING threshold coarse bin.
  ///
  /// `extract_coarse_bin` keeps only the top kHistBits of the fp16 form of a
  /// score, so a single coarse bin spans a whole quarter-binade and can hold
  /// far more than kMaxNumTie elements with distinct fp32 values. The phase-3
  /// collect stages threshold-bin candidates into a fixed kMaxNumTie buffer and
  /// DROPS every arrival past the cap, so in that case handle_tie would rank an
  /// arrival-order subset of the candidates and emit a wrong selected-value
  /// multiset -- silently. This routine replaces phase 4 in exactly that case
  /// (`equal_count > kMaxNumTie`); the fast path is untouched and pays one
  /// comparison. Raising kMaxNumTie instead would only move the trigger point,
  /// and would cost shared memory on the fast path.
  ///
  /// Method: narrow the candidate set with radix passes of 8 bits at a time
  /// over the order-preserving 32-bit key `extract_exact_bin`, restricted to
  /// the elements the coarse pass already classified into the threshold bin
  /// (`v_lo <= val < v_hi`). Each pass histograms the current key byte, finds
  /// the refined threshold byte with the same `above < remain <= above + count`
  /// rule the coarse pass uses, emits the candidates that are now provably
  /// above it (accumulating into `count_gt`, so the output slots stay
  /// contiguous with the ones phase 3 already wrote), and keeps only the
  /// candidates whose key matches the refined prefix.
  ///
  /// TERMINATION / CORRECTNESS INVARIANT -- the loop exits when either
  ///   (a) the refined equal-set is <= kMaxNumTie: it then fits the staging
  ///       buffer whole and the existing handle_tie ranks it exactly; or
  ///   (b) all 32 key bits have been consumed (4 passes of 8 bits).
  ///       `extract_exact_bin` is injective on fp32 bit patterns, so an equal
  ///       key means a bit-identical value: the survivors are interchangeable
  ///       and truncating to an arbitrary kMaxNumTie-subset still yields the
  ///       correct selected-value multiset.
  /// Both exits are exact, and the loop is bounded by 4 passes.
  ///
  /// Exit (b) is also detected UP FRONT by a min/max reduction over the exact
  /// key: a candidate set with a single distinct key is bit-identical, so the
  /// loop can be skipped entirely and phase 3's staged subset ranked directly.
  /// See the early-out block for why that is exact.
  ///
  /// The `above < remain <= above + count` rule needs `0 < remain <= active` on
  /// entry to every pass. That holds: phase 3 classifies by the same two
  /// boundaries used here, so `count_gt + count_eq` is the number of elements
  /// in bins >= threshold, which the coarse threshold-bin invariant puts at
  /// >= topk; and each pass restores it by construction.
  ///
  /// Every pass iterates through `for_each_input(problem.in, problem.seq_len,
  /// ...)`, the same live-length bound phases 1 and 3 use, so a position past
  /// the live length can never become a candidate. Re-reading the scores from
  /// global memory instead of reusing the register-resident copy the register
  /// path holds is deliberate: it keeps this cold path from extending the live
  /// range of `local_vecs` into phase 4 and perturbing the fast path's register
  /// allocation. Shared memory is likewise borrowed from the tie machinery
  /// (`TieHandleSmem::histogram[0]`), so the block's footprint is unchanged.
  SGL_DEVICE static void refine_and_handle_tie(  //
      const TopKProblem& problem,
      Smem* smem,
      const float v_lo,
      const float v_hi,
      const uint32_t equal_count) {
    const auto tx = threadIdx.x;
    const auto topk = problem.topk;
    const auto handle = &smem->tie.handle;

    // DEGENERATE-ROW EARLY-OUT. If every threshold-bin candidate carries the
    // same exact key, the radix passes below cannot separate them and would
    // burn ~4 histogram scans plus ~4 emit scans to arrive at the answer phase
    // 3 already staged. This is termination exit (b) recognised up front:
    // `extract_exact_bin` is injective on fp32 bit patterns, so one distinct
    // key means the candidates are bit-identical, and under the
    // selected-value-multiset contract any kMaxNumTie-subset of them is a
    // correct answer -- including the arrival-order subset already sitting in
    // `smem->tie.values`. So hand that buffer straight to `handle_tie`.
    //
    // The test is exact and general: it compares min and max of the 32-bit
    // exact key, so it fires on an all-equal row of ANY value (an all-zero row
    // is just the common instance) and never on a row with two distinct
    // values, however close. Cost is one scan, against the ~9 the loop below
    // would pay; non-degenerate overflow rows pay that one extra scan and are
    // otherwise untouched.
    {
      uint32_t key_min = 0xFFFFFFFFu;
      uint32_t key_max = 0u;
      for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t) {
        if (val >= v_lo && val < v_hi) {
          const auto key = extract_exact_bin(val);
          key_min = min(key_min, key);
          key_max = max(key_max, key);
        }
      });
      // Reduce per-thread extrema through the borrowed tie histogram; two
      // atomics per thread rather than two per candidate.
      if (tx == 0) {
        handle->histogram[0][0] = 0xFFFFFFFFu;
        handle->histogram[0][1] = 0u;
      }
      __syncthreads();
      atomicMin(&handle->histogram[0][0], key_min);
      atomicMax(&handle->histogram[0][1], key_max);
      __syncthreads();
      const bool bit_identical = handle->histogram[0][0] == handle->histogram[0][1];
      __syncthreads();  // all threads read the scratch before the loop clears it
      if (bit_identical) {
        // equal_count > kMaxNumTie on entry, so phase 3 filled the whole buffer.
        const auto above_count = smem->count_gt;
        const auto remain_topk = above_count < topk ? topk - above_count : 0;
        handle_tie(smem->tie.values, problem, above_count, kMaxNumTie, remain_topk, handle);
        return;
      }
    }

    uint32_t cand_count = equal_count;
    uint32_t remain = smem->count_gt < topk ? topk - smem->count_gt : 0;
    uint32_t prefix = 0;  // refined key bits agreed on so far
    uint32_t mask = 0;    // which key bits `prefix` pins down

    for (uint32_t round = 0; round < 4 && cand_count > kMaxNumTie && remain > 0; ++round) {
      const uint32_t shift = 24 - round * 8;

      if (tx < kRadixSize) handle->histogram[0][tx] = 0;
      __syncthreads();
      for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t) {
        if (val >= v_lo && val < v_hi) {
          const auto key = extract_exact_bin(val);
          if ((key & mask) == prefix) atomicAdd(&handle->histogram[0][(key >> shift) & 0xFFu], 1);
        }
      });
      __syncthreads();

      refine_find_threshold(cand_count, remain, handle);
      const auto match = handle->match;

      for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (val >= v_lo && val < v_hi) {
          const auto key = extract_exact_bin(val);
          if ((key & mask) == prefix && ((key >> shift) & 0xFFu) > match.bin) {
            const auto pos = atomicAdd(&smem->count_gt, 1);
            if (pos < topk) [[likely]]
              problem.emit(pos, idx);
          }
        }
      });

      prefix |= match.bin << shift;
      mask |= 0xFFu << shift;
      remain -= match.above_count;
      cand_count = match.equal_count;
      __syncthreads();  // `match` is read above and overwritten by the next pass
    }

    // Stage the survivors for the existing exact ranker. Exit (b) can still
    // leave more than kMaxNumTie of them, but they are bit-identical by then,
    // so the cap picks an equivalent subset rather than a wrong one.
    if (tx == 0) smem->count_eq = 0;
    __syncthreads();
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (val >= v_lo && val < v_hi) {
        const auto key = extract_exact_bin(val);
        if ((key & mask) == prefix) {
          const auto slot = atomicAdd(&smem->count_eq, 1);
          if (slot < kMaxNumTie) smem->tie.values[slot] = {val, idx};
        }
      }
    });
    __syncthreads();

    const auto above_count = smem->count_gt;
    const auto tie_count = min(smem->count_eq, kMaxNumTie);
    const auto remain_topk = above_count < topk ? topk - above_count : 0;
    handle_tie(smem->tie.values, problem, above_count, tie_count, remain_topk, handle);
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
  SGL_DEVICE static void forward(const TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    {
      Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;
    }
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }

    __syncthreads();
    PDLWaitPrimary<kUsePDL>();

    // A vector `vi` is fully in bounds iff vi < num_full; only full vectors are
    // vector-loaded (16B aligned, never straddling seq_len). The <kVecSize tail is
    // a scalar remainder on the LAST lanes (which own the fewest full vectors, so
    // it overlaps the busy lanes' extra vector). The full path has no per-element
    // bounds check, keeping register pressure low enough to hold all vectors.
    const uint32_t num_full = problem.seq_len / kVecSize;
    const uint32_t tail_start = num_full * kVecSize;
    const uint32_t tail = problem.seq_len - tail_start;

    // Phase 1: load full vectors + build histogram
    vec_t local_vecs[kLocalVecs];
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
      local_vecs[i].load(problem.in, vi);
    }
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j)
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(local_vecs[i][j])], 1);
    }
    if (tx >= kBlockSize - tail) {
      const uint32_t idx = tail_start + tx - (kBlockSize - tail);
      atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(problem.in[idx])], 1);
    }
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, problem.seq_len, smem);

    // Phase 3: collect by two fp32 boundaries (raw indices; transform applied later)
    const auto topk = problem.topk;
    const auto threshold_bin = smem->threshold_bin;
    const auto v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
    const auto v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);
    const auto collect = [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]]
          problem.emit(pos, idx);
      } else if (val >= v_lo) {
        const auto count_eq = atomicAdd(&smem->count_eq, 1);
        if (count_eq < kMaxNumTie) [[likely]]
          smem->tie.values[count_eq] = {val, idx};
      }
    };
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      const auto base = vi * kVecSize;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j)
        collect(local_vecs[i][j], base + j);
    }
    if (tx >= kBlockSize - tail) {
      const uint32_t idx = tail_start + tx - (kBlockSize - tail);
      collect(problem.in[idx], idx);
    }

    // Phase 4: Handle ties.
    __syncthreads();
    const auto above_count = smem->count_gt;
    const auto equal_count = smem->count_eq;
    const auto remain_topk = above_count < topk ? topk - above_count : 0;
    if (equal_count > kMaxNumTie) [[unlikely]] {
      // The threshold coarse bin overflowed the staging buffer, so the buffer
      // holds an arrival-order subset of the candidates. Refine on the exact
      // key before ranking -- see refine_and_handle_tie.
      refine_and_handle_tie(problem, smem, v_lo, v_hi, equal_count);
      return;
    }
    const auto tie_count = min(equal_count, kMaxNumTie);
    handle_tie(smem->tie.values, problem, above_count, tie_count, remain_topk, &smem->tie.handle);
  }
};

// ---------------------------------------------------------------------------
// Streaming path: seq_len > 8192 -- two vectorized passes over global memory
// ---------------------------------------------------------------------------

struct TopKStreaming : TopKRegister<2> {
 public:
  static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();

  template <bool kUsePDL>
  SGL_DEVICE static void forward(const TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    {
      Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;
    }
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
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, problem.seq_len, smem);

    // Phase 3: Collect candidates and sort. Classify by two fp32 boundaries derived
    // from the threshold bin instead of recomputing the fp16 bin per element: an
    // element is "above" iff val >= v_hi (bin > threshold) and a "tie" iff
    // v_lo <= val < v_hi (bin == threshold). This drops the F2F + bit-twiddle from
    // the second full pass over the input.
    const auto threshold_bin = smem->threshold_bin;
    const float v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
    const float v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);
    const auto topk = problem.topk;
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]] {
          problem.emit(pos, idx);
        }
      } else if (val >= v_lo) {
        const auto count_eq = atomicAdd(&smem->count_eq, 1);
        if (count_eq < kMaxNumTie) [[likely]] {
          smem->tie.values[count_eq] = {val, idx};
        }
      }
    });

    // Phase 4: Handle ties. Drive the output layout from the *collect* counts so it
    // is self-consistent with the fp32 classification above (rather than the fp16
    // histogram counts), even if rounding moves a boundary element between the
    // "above" and "tie" sets. above_count is < topk by the threshold-bin invariant,
    // so the count_gt guard above effectively never triggers.
    __syncthreads();
    const auto above_count = smem->count_gt;
    const auto equal_count = smem->count_eq;
    const auto remain_topk = above_count < topk ? topk - above_count : 0;
    if (equal_count > kMaxNumTie) [[unlikely]] {
      // See the register path: overflowing coarse bins get an exact-key radix
      // refinement instead of an arrival-order truncation.
      refine_and_handle_tie(problem, smem, v_lo, v_hi, equal_count);
      return;
    }
    const auto tie_count = min(equal_count, kMaxNumTie);
    handle_tie(smem->tie.values, problem, above_count, tie_count, remain_topk, &smem->tie.handle);
  }
};

// ---------------------------------------------------------------------------
// Cluster path: very long seq_len, small batch. `kClusterSize` blocks cooperate
// on one batch element via distributed shared memory (one cluster per element).
//
// CUDA only: thread-block clusters and distributed shared memory have no CDNA
// equivalent.
//
// KNOWN GAP: this path still truncates an overflowing threshold coarse bin to
// kMaxNumTie the way the register and streaming paths did before
// refine_and_handle_tie. Refining here is not the same routine: the candidate
// set is split across kClusterSize ranks, so every radix pass needs a
// cluster-wide histogram all-reduce (phase 1.5) and a cluster-wide emit
// counter, not the block-local ones refine_and_handle_tie uses. Left
// unaddressed deliberately -- the path does not exist on ROCm, which is where
// the defect was measured.
// ---------------------------------------------------------------------------

#ifndef USE_ROCM

template <uint32_t kClusterSize_>
struct TopKCluster : TopKRadixBase<10> {
 public:
  static constexpr uint32_t kClusterSize = kClusterSize_;
  static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();
  using Base = TopKRadixBase<10>;
  struct Smem : Base::Smem {
    using kHistVec = Base::Smem::kHistVec;
    uint32_t start_eq_local, start_gt_local;
    int32_t tmp_out[kMaxTopK];
  };

  // Process ONE batch element (one cluster). NO PDL and NO trailing barrier --
  // the persistent kernel does PDLWaitPrimary once before its item loop and a
  // cluster.sync() after each forward(). Writes raw indices to out; the kernel's
  // transform pass applies the page-table transform.
  template <bool kUsePDL>
  SGL_DEVICE static void forward(TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);
    const auto cluster = cg::this_cluster();
    const auto this_rank = blockIdx.y;
    const bool is_primary = (this_rank == 0);

    constexpr uint32_t kAlignElems = kWarpSize * kVecSize;
    const uint32_t chunk_size = div_ceil(problem.seq_len, kClusterSize * kAlignElems) * kAlignElems;
    const uint32_t chunk_start = min(this_rank * chunk_size, problem.seq_len);
    const uint32_t chunk_finish = min(chunk_start + chunk_size, problem.seq_len);
    const uint32_t local_seq_len = chunk_finish - chunk_start;
    problem.in += chunk_start;

    {
      typename Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;
    }
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }
    __syncthreads();
    PDLWaitPrimary<kUsePDL>();

    // Phase 1: Load and build histogram over this rank's contiguous chunk.
    for_each_input(problem.in, local_seq_len, [&](float val, uint32_t) {
      const auto bin = extract_coarse_bin<kHistBits>(val);
      atomicAdd(&smem->histogram[bin], 1);
    });
    __syncthreads();

    // Phase 1.5: reduce the histogram across the cluster
    {
      // 1-shot all-reduce: each rank owns kPartition consecutive bins;
      // for each owned bin, gather the kClusterSize peer values (one per
      // consecutive lane) via DSMEM, sum across the lanes, then scatter back.
      cluster.sync();
      static_assert(kHistSize == kBlockSize);  // we optimize on top of this
      constexpr uint32_t kPartition = kHistSize / kClusterSize;
      const auto start = this_rank * kPartition;
      const auto which = start + tx / kClusterSize;
      const auto peer_rank = tx % kClusterSize;
      const auto addr = cluster.map_shared_rank(&smem->histogram[which], peer_rank);
      const auto value = *addr;
      *addr = warp::reduce_sum<kClusterSize>(value);
      cluster.sync();
    }

    // Phase 2: Find the threshold bin (uses global seq_len)
    find_threshold(problem.topk, problem.seq_len, smem);

    // Phase 3: Collect candidates over this rank's chunk; convert local indices
    // back to global by adding chunk_start. Classify by two fp32 boundaries derived
    // from the (global) threshold bin instead of recomputing the fp16 bin per
    // element -- see TopKStreaming for the rationale. threshold_bin is identical
    // across ranks, so v_hi/v_lo are too.
    const auto topk = problem.topk;
    const auto threshold_bin = smem->threshold_bin;
    const float v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
    const float v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);

    // Phase 3: collect candidates. The primary scatters straight into
    // `problem.out`, the others stage into block-local `smem->tmp_out`.
    //
    // DO NOT merge these two loops back into one by selecting the destination
    // first (`cur_out = is_primary ? problem.out : smem->tmp_out`). `problem.out`
    // can be a shared::cluster (DSMEM) alias of the elected rank's buffer while
    // `tmp_out` is shared::cta; merging them into a single pointer variable makes
    // cicc 13.1+ mis-lower the block-local arm on sm_90a and *silently drop every
    // non-primary rank's staged output* -- `tmp_out` stays zero, and phase 3.5
    // then faithfully copies zeros to correct DSMEM addresses. The result is a
    // top-k output where only the primary's slots and the handle_tie tail are
    // valid, which downstream sparse attention dereferences as garbage KV indices.
    if (!is_primary) {
      // stage to tmp_out first before writing to global/DSMEM
      for_each_input(problem.in, local_seq_len, [&](float val, uint32_t local_idx) {
        const auto idx = chunk_start + local_idx;
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            smem->tmp_out[pos] = idx;
          }
        } else if (val >= v_lo) {
          const auto count_eq = atomicAdd(&smem->count_eq, 1);
          if (count_eq < kMaxNumTie) [[likely]] {
            smem->tie.values[count_eq] = {val, idx};
          }
        }
      });
      __syncthreads();
      const auto local_above_count = smem->count_gt;
      const auto local_equal_count = min(smem->count_eq, kMaxNumTie);
      const auto smem_0 = cluster.map_shared_rank(smem, 0);
      if (tx == 0) {
        const auto gt = atomicAdd(&smem_0->count_gt, local_above_count);
        const auto eq = atomicAdd(&smem_0->count_eq, local_equal_count);
        smem->start_gt_local = gt;
        smem->start_eq_local = eq;
      }
      __syncthreads();
      const auto start_gt_local = smem->start_gt_local;
      const auto start_eq_local = smem->start_eq_local;
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        const auto t = tx + i * kBlockSize;
        if (t < local_equal_count && start_eq_local + t < kMaxNumTie) {
          smem_0->tie.values[start_eq_local + t] = smem->tie.values[t];
        }
      }

      cluster.sync();

      const auto start_write = start_gt_local;
      const auto num_write = local_above_count;
#pragma unroll
      for (uint32_t i = 0; i < kTopKItems; ++i) {
        if (const auto t = tx + i * kBlockSize; t < num_write && start_write + t < topk) {
          problem.emit(start_write + t, smem->tmp_out[t]);
        }
      }
    } else {
      for_each_input(problem.in, local_seq_len, [&](float val, uint32_t local_idx) {
        const auto idx = chunk_start + local_idx;
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            problem.emit(pos, idx);
          }
        } else if (val >= v_lo) {
          const auto count_eq = atomicAdd(&smem->count_eq, 1);
          if (count_eq < kMaxNumTie) [[likely]] {
            smem->tie.values[count_eq] = {val, idx};
          }
        }
      });

      cluster.sync();

      // Phase 4: Handle ties.
      const auto above_count = smem->count_gt;
      const auto equal_count = smem->count_eq;
      const auto remain_topk = above_count < topk ? topk - above_count : 0;
      const auto tie_count = min(equal_count, kMaxNumTie);
      handle_tie(smem->tie.values, problem, above_count, tie_count, remain_topk, &smem->tie.handle);
    }
  }
};

#endif  // !USE_ROCM

}  // namespace device::topk

}  // namespace sglang
