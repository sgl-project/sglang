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
/// exact radix tie-break.

#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

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
#include <cooperative_groups.h>
#endif

namespace sglang {

namespace device::topk {

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

/// sgl_kernel names the warp size `kWarpThreads`; alias it locally as `kWarpSize`.
inline constexpr uint32_t kWarpSize = kWarpThreads;

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

// template <uint32_t kBits>
// SGL_DEVICE uint32_t extract_coarse_bin(float x) {
//   static_assert(0 < kBits && kBits < 15);
//   const auto hx = cast<fp16_t>(x);
//   const uint16_t bits = *reinterpret_cast<const uint16_t*>(&hx);
//   const uint16_t key = (bits & 0x8000) ? ~bits : bits | 0x8000;
//   return key >> (16 - kBits);
// }

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  uint32_t b = (uint32_t)__half_as_ushort(__float2half_rn(x)) << 16;
  uint32_t s = (uint32_t)((int32_t)b >> 31);
  return (b ^ (s | 0x80000000u)) >> (32 - kBits);
}

// Smallest fp32 `v` for which `extract_coarse_bin<kBits>(v) >= bin`, i.e. the
// lower fp32 boundary of coarse bin `bin`. The collect pass classifies with two
// comparisons against these instead of recomputing the fp16 bin per element, so
// this must agree with `extract_coarse_bin` on every value -- a score sitting
// exactly on a boundary included. Two pairs no fp32 threshold can separate are
// left: -0.0 at the zero bin, and +inf at a NaN-key bin.
template <uint32_t kBits>
SGL_DEVICE float coarse_bin_lower_bound(uint32_t bin) {
  constexpr uint32_t kShift = 16 - kBits;
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr uint32_t kInfBin = 0xFC00u >> kShift;  // bin holding the +inf key
  const uint32_t key = bin << kShift;              // ordered16 key at the low edge
  // ordered16 -> fp16 value (inverse of the transform in extract_coarse_bin);
  // finite keys only.
  constexpr auto to_finite_bits = [](uint32_t okey) -> uint16_t {
    const uint16_t ob = static_cast<uint16_t>(okey);
    return (ob & 0x8000) ? static_cast<uint16_t>(ob ^ 0x8000) : static_cast<uint16_t>(~ob);
  };
  constexpr auto to_finite_val = [](uint32_t okey) -> float {
    const uint16_t hb = to_finite_bits(okey);
    return cast<float>(*reinterpret_cast<const fp16_t*>(&hb));
  };
  constexpr auto step_up = [](float v) -> float {
    const int32_t b = __float_as_int(v);
    return __int_as_float(b >= 0 ? b + 1 : b - 1);
  };
  // Fast path, hoisted above the per-key special cases so both keys are
  // range-checked at once: `key` and `key - 1` both land in the finite band
  // [0x0401, 0xFBFF] -- every boundary a finite-score threshold produces. fp16
  // rounds to nearest, so the boundary is the midpoint between the fp16 values
  // at `key` and `key - 1`. (Measured faster than a per-key dispatch: the two
  // conversions are independent and issue in parallel.)
  if (key - 0x0401u <= 0xFBFFu - 0x0401u) {
    const float mid = 0.5f * (to_finite_val(key) + to_finite_val(key - 1));
    // fp32 -> fp16 rounds to nearest EVEN, so on the ~half of bins whose fp16
    // value has an odd significand the midpoint still bins as `bin - 1`.
    return (to_finite_bits(key) & 1u) ? step_up(mid) : mid;
  }
  // Slow path: an edge of `bin` touches the +/-inf keys or NaN key space. The
  // ordered-key line is: [0, 0x03FF) negative-NaN space, 0x03FF = -inf,
  // [0x0400, 0xFC00) finite, 0xFC00 = +inf, (0xFC00, 0xFFFF] positive-NaN
  // space. The +/-inf keys stand in as +/-65536, one ideal step past fp16 max,
  // so the midpoint lands on the +/-65520 fp32 -> fp16 overflow threshold.
  if (bin == 0) return -kInf;      // every value bins at >= 0
  if (bin > kInfBin) return kInf;  // NaN key space: nothing bins that high
  const auto to_val = [&](uint32_t okey) -> float {
    if (okey < 0x03FFu) return -kInf;
    if (okey == 0x03FFu) return -65536.0f;
    if (okey == 0xFC00u) return 65536.0f;
    return to_finite_val(okey);
  };
  // The +/-65536 stand-ins are not real fp16 neighbours, so the parity rule
  // does not apply here; test the property directly instead.
  const float mid = 0.5f * (to_val(key) + to_val(key - 1));
  return extract_coarse_bin<kBits>(mid) < bin ? step_up(mid) : mid;
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

struct TopKProblem {
  const float* __restrict__ in;
  int32_t* __restrict__ out;  // page_indices [topk]
  uint32_t topk;
  uint32_t seq_len;
  int32_t bias = 0;  // needed by ragged mode

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

      const auto [threshold_bin, above_count, equal_count] = smem->match;
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
    uint32_t warp_sum[kNumWarps];
    // The coarse histogram is dead once find_threshold() has published
    // threshold_bin, and the tie machinery only comes alive after that: the
    // collect pass fills tie_values, then handle_tie works over them with
    // tie_handle as scratch. Overlaying the two phases keeps the
    // kMaxNumTie-candidate buffer from growing the block's shared-memory
    // footprint. tie_handle and tie_values are live TOGETHER, so they sit
    // side by side inside the overlay, not in a union with each other.
    union {
      alignas(16) uint32_t histogram[kHistSize];
      struct {
        TieValue tie_values[kMaxNumTie];
        TieHandleSmem tie_handle;
      };
    };
  };

 protected:
  template <uint32_t N = 1, typename F>
  SGL_DEVICE static void for_each_input(const float* __restrict__ in, uint32_t seq_len, F&& fn) {
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
    const auto warp_inc_sum = warp_inclusive_sum(lane_id, local_sum);
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
    const uint32_t num_full = div_ceil(problem.seq_len, kVecSize);

    // Phase 1: load full vectors + build histogram
    vec_t local_vecs[kLocalVecs];
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi < num_full) local_vecs[i].load(problem.in, vi);
    }

    const auto tail_start = (problem.seq_len - 1) % kVecSize + 1;
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
      if (vi == num_full - 1) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          if (j >= tail_start) local_vecs[i][j] = -FLT_MAX;
        }
      }
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(local_vecs[i][j])], 1);
      }
    }
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, num_full * kVecSize, smem, [&](uint32_t threshold_bin) {
      const auto v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
      const auto v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin + 0);
      smem->v_hi = v_hi;
      smem->v_lo = v_lo;
    });

    // Phase 3: collect by two fp32 boundaries
    const auto topk = problem.topk;
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;

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
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      }
    }

    // Phase 4: Handle ties.
    __syncthreads();
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
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, problem.seq_len, smem, [&](uint32_t threshold_bin) {
      const auto v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
      const auto v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin + 0);
      smem->v_hi = v_hi;
      smem->v_lo = v_lo;
    });

    // Phase 3: Collect candidates and sort. Classify by two fp32 boundaries derived
    // from the threshold bin instead of recomputing the fp16 bin per element: an
    // element is "above" iff val >= v_hi (bin > threshold) and a "tie" iff
    // v_lo <= val < v_hi (bin == threshold). This drops the F2F + bit-twiddle from
    // the second full pass over the input.
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    const auto topk = problem.topk;
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
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

    // Phase 4: Handle ties. Drive the output layout from the *collect* counts so it
    // is self-consistent with the fp32 classification above (rather than the fp16
    // histogram counts), even if rounding moves a boundary element between the
    // "above" and "tie" sets. above_count is < topk by the threshold-bin invariant,
    // so the count_gt guard above effectively never triggers.
    __syncthreads();
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
// ---------------------------------------------------------------------------

#if SUPPORT_CLUSTER

template <uint32_t N>
struct TopKCluster : TopKRadixBase<10> {
 public:
  static constexpr uint32_t kClusterSize = N;
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
      alignas(16) uint32_t histogram[kHistSize];
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
    barrier_cluster_arrive_relaxed();  // bar-0 arrive
    PDLWaitPrimary<kUsePDL>();

    // Phase 1: Load and build histogram over this rank's contiguous chunk.
    for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t) {
      const auto bin = extract_coarse_bin<kHistBits>(val);
      atomicAdd(&smem->histogram[bin], 1);
    });

    barrier_cluster_wait();  // bar-0 wait
    __syncthreads();
    if (this_rank != 0) {
      const auto smem_0 = cluster.map_shared_rank(smem, 0);
      // Phase 2. atomic flush all histogram into rank 0
      static_assert(kHistSize == kBlockSize);  // one bin per thread

      if (const auto count = smem->histogram[tx]; count != 0) {
        atomicAdd(&smem_0->histogram[tx], count);
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
        smem->v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
        smem->v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin + 0);
      });

      // NOTE: zero write to DSMEM, so relaxed is strong enough here
      barrier_cluster_arrive_relaxed();  // bar-2 arrive
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

#endif  // !USE_ROCM

}  // namespace device::topk

}  // namespace sglang
