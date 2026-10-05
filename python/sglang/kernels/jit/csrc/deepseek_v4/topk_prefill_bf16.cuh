#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cfloat>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#ifndef SGL_TOPK
#define SGL_TOPK 512
#endif

namespace sglang {
namespace {

// ============================================================
// SM80 BF16 prefill streaming top-k kernel.
//
// Pipeline: `cp.async` (via __pipeline_memcpy_async) doubly-buffered
// staging of bf16 scores into shared memory, then a 12-bit coarse
// histogram select followed by a tie refinement pass.
// ============================================================

namespace prefill_bf16 {

constexpr uint32_t kTopK = SGL_TOPK;
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kNumWarps = kBlockSize / 32;

constexpr uint32_t kHistBits = 12;
constexpr uint32_t kHistBins = 1u << kHistBits;          // 4096
constexpr uint32_t kHistItems = kHistBins / kBlockSize;  // 4

constexpr uint32_t kNumStages = 2;                                         // double buffer (cp.async)
constexpr uint32_t kElemPerStage = 8;                                      // bf16 elems per thread per stage
constexpr uint32_t kSizePerStage = kElemPerStage * kBlockSize;             // 8192 bf16 / stage
constexpr uint32_t kCpAsyncBytes = kElemPerStage * sizeof(__nv_bfloat16);  // 16B

constexpr uint32_t kMaxTies = kTopK;         // Must equal kTopK to cover all tie slots when scores are NaN
constexpr uint32_t kMax1PassLength = 16384;  // single-pass register cap
constexpr uint32_t kMax2PassLength = 2 * kMax1PassLength;  // 32768
constexpr uint32_t kVecsPerThread = 4;

}  // namespace prefill_bf16

// ============================================================
// Public params (mirrors TopKPrefillParams but with bf16 scores).
// ============================================================
struct TopKPrefillBF16Params {
  const __nv_bfloat16* __restrict__ scores;
  const int32_t* __restrict__ row_starts;
  const int32_t* __restrict__ row_ends;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;  // optional, may be nullptr
  const int64_t score_stride;
  const int64_t page_table_stride;
  uint32_t page_bits;
};

// ============================================================
// Lightweight POD helpers (kept local to this TU to avoid
// colliding with topk_bf16.cuh when both share the same JIT cache).
// ============================================================
struct alignas(16) MatchBinV3 {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

struct alignas(8) TieV3 {
  uint32_t idx;
  float score;
};

// ============================================================
// Shared memory layout (~50.5 KB; SM80 max dynamic smem 164KB).
//   counter_gt  : 128B (alignas)
//   counter_eq  : 128B
//   match       : 128B
//   warp_sum    : 128B (32 warps * 4B = 128B)
//   union { histogram[4096]  -> 16KB
//           tie_buffer[kTopK] -> kTopK*8B }
//   bf16_buffer[2][8192]     -> 32KB
// s_topk_indices (kTopK * int32) = 2KB lives at the tail of dynamic smem
// (allocated separately by the caller).
// ============================================================
struct alignas(128) PrefillBF16Smem {
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  alignas(128) MatchBinV3 match;
  alignas(128) uint32_t warp_sum[prefill_bf16::kNumWarps];
  union {
    uint32_t histogram[prefill_bf16::kHistBins];
    TieV3 tie_buffer[prefill_bf16::kMaxTies];
  };
  alignas(128) __nv_bfloat16 bf16_buffer[prefill_bf16::kNumStages][prefill_bf16::kSizePerStage];
};

// ============================================================
// Register-fast-path shared memory layout (~80 KB).
//
// Mirrors `RegisterTopKSmem` from topk_bf16.cuh but parameterized
// against the `prefill_bf16` namespace so we can share dynamic
// smem with the streaming layout via overlay.
//   counter_gt / counter_eq : 256B
//   match + warp_sum        : 32B + 128B
//   union { histogram[4096] -> 16KB
//           histogram_vec   -> same storage as histogram
//           tie_buffer[kTopK]-> kTopK*8B }
//   bf16_stage[16384]       -> 32KB
// ============================================================
using Vec4PF = device::AlignedVector<float, 4>;

struct RegisterTopKSmemPF {
  using HistVec = device::AlignedVector<uint32_t, prefill_bf16::kHistBins / prefill_bf16::kBlockSize>;
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBinV3 match;
  uint32_t warp_sum[prefill_bf16::kNumWarps];
  union {
    uint32_t histogram[prefill_bf16::kHistBins];
    HistVec histogram_vec[prefill_bf16::kBlockSize];
    TieV3 tie_buffer[prefill_bf16::kMaxTies];
  };
  alignas(128) __nv_bfloat16 bf16_stage[prefill_bf16::kMax1PassLength];
};

// ============================================================
// Helpers
// ============================================================

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin_pf(float x) {
  // Order-preserving fp32->fp16->uint16 mapping, then top-kBits as bin idx.
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

SGL_DEVICE int32_t page_to_indices_pf(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

SGL_DEVICE uint32_t cdiv_pf(uint32_t a, uint32_t b) {
  return (a + b - 1u) / b;
}

// Element-granularity misalignment of a bf16 pointer w.r.t. 16B vector access.
// `scores + row_id * score_stride + row_start` carries arbitrary parity because
// both score_stride and row_start (== ks) are plain token counts, so the row
// base is only 2B aligned in general. cp.async / uint4 loads require 16B, so
// callers align the base down by this amount and shift the scan window instead.
SGL_DEVICE uint32_t bf16_misalign_elems_pf(const __nv_bfloat16* p) {
  return static_cast<uint32_t>((reinterpret_cast<uintptr_t>(p) >> 1) & (prefill_bf16::kElemPerStage - 1u));
}

SGL_DEVICE uint32_t warp_inclusive_sum_pf(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

// ============================================================
// stream_pass_bf16
//
// Doubly-buffered cp.async pipeline. When kIsScatter == false this
// only accumulates the 12-bit histogram. When kIsScatter == true,
// elements with bin > thr_bin are emitted into s_topk_indices and
// equal-bin elements are deferred to the tie buffer.
//
// `scores_bf16` must be 16B aligned; the logical row lives in the
// window [base_offset, base_offset + length) so that an arbitrarily
// aligned row start can still be served by 16B cp.async.
// ============================================================
template <bool kIsScatter>
SGL_DEVICE static void stream_pass_bf16(
    const __nv_bfloat16* __restrict__ scores_bf16,
    const uint32_t base_offset,
    const uint32_t length,
    const uint32_t thr_bin,
    int32_t* __restrict__ s_topk_indices,
    PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;
  const auto total = base_offset + length;
  const auto num_iters = cdiv_pf(total, kSizePerStage);

  auto issue_stage = [&](uint32_t s) {
    // Always issues a commit even when out of range, so callers can
    // pair every issue with `__pipeline_commit` without branching.
    if (s >= num_iters) return;
    const uint32_t buf = s % kNumStages;
    const uint32_t base = s * kSizePerStage;
    const uint32_t local = tx * kElemPerStage;
    const uint32_t global_base = base + local;

    if (global_base + kElemPerStage <= total) {
      // Fast path: full 16-byte cp.async copy of 8 bf16s.
      __pipeline_memcpy_async(&smem->bf16_buffer[buf][local], scores_bf16 + global_base, kCpAsyncBytes);
    } else if (global_base < total) {
      // Tail path: do an aligned full-size copy if the remaining tile
      // is still entirely valid; otherwise fall back to per-element
      // synchronous loads with sentinel padding.
#pragma unroll
      for (uint32_t e = 0; e < kElemPerStage; ++e) {
        const uint32_t g = global_base + e;
        smem->bf16_buffer[buf][local + e] = (g < total) ? scores_bf16[g] : __float2bfloat16(-FLT_MAX);
      }
    } else {
      // Entire chunk is OOB: prefill sentinel so processing never
      // observes garbage values for masked positions.
#pragma unroll
      for (uint32_t e = 0; e < kElemPerStage; ++e) {
        smem->bf16_buffer[buf][local + e] = __float2bfloat16(-FLT_MAX);
      }
    }
  };

  // ------------------------------------------------------------
  // Prologue: kick off kNumStages stages, each in its own group.
  // ------------------------------------------------------------
#pragma unroll
  for (uint32_t s = 0; s < kNumStages; ++s) {
    issue_stage(s);
    __pipeline_commit();
  }

  // ------------------------------------------------------------
  // Main loop: wait for the head group, consume it, prefetch.
  // ------------------------------------------------------------
  for (uint32_t iter = 0; iter < num_iters; ++iter) {
    __pipeline_wait_prior(kNumStages - 1);
    __syncthreads();

    const uint32_t buf = iter % kNumStages;
    const uint32_t base = iter * kSizePerStage;

#pragma unroll
    for (uint32_t e = 0; e < kElemPerStage; ++e) {
      const uint32_t local = tx * kElemPerStage + e;
      const uint32_t w = base + local;
      if (w >= total) break;
      // Unsigned wrap makes the `base_offset` head elements compare as huge,
      // so this single check drops both the head padding and the tail.
      const uint32_t global_idx = w - base_offset;
      if (global_idx >= length) continue;

      const float val = __bfloat162float(smem->bf16_buffer[buf][local]);
      const uint32_t bin = extract_coarse_bin_pf<kHistBits>(val);

      if constexpr (kIsScatter) {
        if (bin > thr_bin) {
          const auto pos = atomicAdd(&smem->counter_gt, 1u);
          if (pos < kTopK) {
            s_topk_indices[pos] = static_cast<int32_t>(global_idx);
          }
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1u);
          if (pos < kMaxTies) {
            smem->tie_buffer[pos] = TieV3{.idx = global_idx, .score = val};
          }
        }
      } else {
        atomicAdd(&smem->histogram[bin], 1u);
      }
    }
    __syncthreads();

    // Prefetch the (iter + kNumStages)-th stage into the buffer we
    // just freed and start a new pipeline group.
    issue_stage(iter + kNumStages);
    __pipeline_commit();
  }

  // Drain any extra in-flight prefetches issued past num_iters.
  __pipeline_wait_prior(0);
  __syncthreads();
}

// ============================================================
// scatter_pass_bf16_warp_steal
//
// Work-stealing scatter pass for ultra-long sequences. Each warp
// independently claims tiles from a shared atomic counter and
// loads data directly from HBM via vectorized loads (uint4).
// No block-level __syncthreads() in the main loop -- warps that
// miss topk hits immediately steal the next unscanned tile.
//
// Same aligned-base + [base_offset, base_offset + length) window
// contract as stream_pass_bf16, required by the uint4 loads.
// ============================================================
SGL_DEVICE static void scatter_pass_bf16_warp_steal(
    const __nv_bfloat16* __restrict__ scores_bf16,
    const uint32_t base_offset,
    const uint32_t length,
    const uint32_t thr_bin,
    int32_t* __restrict__ s_topk_indices,
    PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;
  const auto lane_id = tx & 31u;
  const auto warp_id = tx >> 5u;

  // Warp-tile granularity: 32 lanes * 8 elements = 256 BF16 values.
  constexpr uint32_t kWarpTileElems = 32u * kElemPerStage;  // 256
  const uint32_t total = base_offset + length;
  const uint32_t num_warp_tiles = cdiv_pf(total, kWarpTileElems);

  // Reuse warp_sum[0] as shared work-steal counter.
  // Pre-assign first kNumWarps tiles statically to avoid initial stampede.
  if (tx == 0) {
    smem->warp_sum[0] = kNumWarps;
  }
  __syncthreads();  // single init barrier

  // Each warp starts with its statically-assigned tile.
  uint32_t my_tile = warp_id;

  while (my_tile < num_warp_tiles) {
    const uint32_t base = my_tile * kWarpTileElems;
    const uint32_t global_base = base + lane_id * kElemPerStage;

    if (global_base + kElemPerStage <= total) {
      // Fast path: vectorized 16-byte load (8 bf16 = uint4).
      const uint4 data = *reinterpret_cast<const uint4*>(scores_bf16 + global_base);
      const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&data);

#pragma unroll
      for (uint32_t e = 0; e < kElemPerStage; ++e) {
        // Unsigned wrap drops the `base_offset` head elements.
        const uint32_t global_idx = global_base + e - base_offset;
        if (global_idx >= length) continue;

        const float fval = __bfloat162float(vals[e]);
        const uint32_t bin = extract_coarse_bin_pf<kHistBits>(fval);

        if (bin > thr_bin) {
          const auto pos = atomicAdd(&smem->counter_gt, 1u);
          if (pos < kTopK) {
            s_topk_indices[pos] = static_cast<int32_t>(global_idx);
          }
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1u);
          if (pos < kMaxTies) {
            smem->tie_buffer[pos] = TieV3{.idx = global_idx, .score = fval};
          }
        }
      }
    } else if (global_base < total) {
      // Tail path: element-by-element for partial tile.
      for (uint32_t e = 0; e < kElemPerStage; ++e) {
        const uint32_t w = global_base + e;
        if (w >= total) break;
        const uint32_t global_idx = w - base_offset;
        if (global_idx >= length) continue;

        const float fval = __bfloat162float(scores_bf16[w]);
        const uint32_t bin = extract_coarse_bin_pf<kHistBits>(fval);

        if (bin > thr_bin) {
          const auto pos = atomicAdd(&smem->counter_gt, 1u);
          if (pos < kTopK) {
            s_topk_indices[pos] = static_cast<int32_t>(global_idx);
          }
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1u);
          if (pos < kMaxTies) {
            smem->tie_buffer[pos] = TieV3{.idx = global_idx, .score = fval};
          }
        }
      }
    }
    // else: entire lane range OOB, skip.

    // Warp leader atomically claims the next tile; broadcast to all lanes.
    if (lane_id == 0) {
      my_tile = atomicAdd(&smem->warp_sum[0], 1u);
    }
    my_tile = __shfl_sync(0xFFFFFFFF, my_tile, 0);
  }

  // Block-level barrier: all warps must complete the scatter phase before
  // the caller reads counter_eq / tie_buffer in tie_handle_and_transform.
  // Without this, fast-finishing warps race ahead into tie-handling while
  // slow warps are still writing atomics → stale reads → garbage indices → crash.
  __syncthreads();
}

// ============================================================
// find_threshold_bf16
//
// 4096 bins / 1024 threads = 4 bins/thread. Builds a warp-then-block
// inclusive prefix scan of the histogram and locates the bin whose
// inclusive prefix straddles the K-th element.
// ============================================================
SGL_DEVICE static void find_threshold_bf16(uint32_t length, PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  uint32_t orig[kHistItems];
  uint32_t local_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < kHistItems; ++i) {
    orig[i] = smem->histogram[tx * kHistItems + i];
    local_sum += orig[i];
  }

  const auto warp_inc = warp_inclusive_sum_pf(lane_id, local_sum);
  const auto warp_exc = warp_inc - local_sum;
  if (lane_id == 31) smem->warp_sum[warp_id] = warp_inc;
  __syncthreads();

  // Inter-warp inclusive prefix; kNumWarps == 32 fits in one lane vector.
  const uint32_t tmp = smem->warp_sum[lane_id];
  uint32_t prefix_sum = device::warp::reduce_sum(lane_id < warp_id ? tmp : 0u);
  prefix_sum += warp_exc;

#pragma unroll
  for (uint32_t i = 0; i < kHistItems; ++i) {
    prefix_sum += orig[i];
    const auto above = length - prefix_sum;  // strict-above count
    if (above < kTopK && above + orig[i] >= kTopK) {
      smem->match = MatchBinV3{
          .bin = tx * kHistItems + i,
          .above_count = above,
          .equal_count = orig[i],
      };
    }
  }
  __syncthreads();
}

// ============================================================
// tie_handle_and_transform_bf16
//
// Resolves the equality bin and translates the final raw indices
// through the page table. When num_above + num_equal == kTopK we
// just append all ties; otherwise rank ties by (score desc, idx asc).
// ============================================================
SGL_DEVICE static void tie_handle_and_transform_bf16(
    const TopKPrefillBF16Params& params,
    int32_t* __restrict__ s_topk_indices,
    int32_t* __restrict__ page_indices_ptr,
    int32_t* __restrict__ raw_indices_ptr,
    const int32_t* __restrict__ page_ptr,
    PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  const auto num_above = smem->match.above_count;
  const auto num_equal = smem->counter_eq;
  const uint32_t num_ties = (num_equal < kMaxTies) ? num_equal : kMaxTies;
  const uint32_t topk_remain = (kTopK > num_above) ? (kTopK - num_above) : 0u;
  const bool need_tiebreak = (num_above + num_equal > kTopK);

  if (topk_remain == 0u) {
    // Already filled by the strictly-greater bucket; nothing to do.
  } else if (!need_tiebreak) {
    // Fast path: every tie element fits.
    for (uint32_t i = tx; i < num_ties; i += kBlockSize) {
      const auto pos = num_above + i;
      if (pos < kTopK) {
        s_topk_indices[pos] = static_cast<int32_t>(smem->tie_buffer[i].idx);
      }
    }
  } else {
    // Tie-break: rank each tie within the bin and keep top `topk_remain`.
    const auto is_greater = [](const TieV3& a, const TieV3& b) {
      if (a.score > b.score) return true;
      if (a.score < b.score) return false;
      // Neither > nor < means equal OR NaN; use idx as deterministic tiebreaker
      return a.idx < b.idx;
    };

    if (num_ties <= 32u) {
      // Single-warp ballot fast path.
      if (lane_id < num_ties && warp_id < num_ties) {
        const uint32_t mask = (num_ties == 32u) ? 0xFFFFFFFFu : ((1u << num_ties) - 1u);
        const auto tie = smem->tie_buffer[lane_id];
        const auto target = smem->tie_buffer[warp_id];
        const bool pred = is_greater(tie, target);
        const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
        if (lane_id == 0 && rank < topk_remain) {
          s_topk_indices[num_above + rank] = static_cast<int32_t>(target.idx);
        }
      }
    } else if (num_ties <= 64u) {
      // Two-batch ballot: each warp ranks one target via two halves.
      const auto invalid = TieV3{.idx = 0xFFFFFFFFu, .score = -FLT_MAX};
      const auto tie_0 = smem->tie_buffer[lane_id];
      const auto tie_1 = (lane_id + 32u) < num_ties ? smem->tie_buffer[lane_id + 32u] : invalid;

      auto rank_target = [&](const TieV3& target) {
        const bool p0 = is_greater(tie_0, target);
        const bool p1 = is_greater(tie_1, target);
        const auto r0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFFu, p0)));
        const auto r1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFFu, p1)));
        return r0 + r1;
      };

      if (warp_id < num_ties) {
        const auto target = smem->tie_buffer[warp_id];
        const auto rank = rank_target(target);
        if (lane_id == 0 && rank < topk_remain) {
          s_topk_indices[num_above + rank] = static_cast<int32_t>(target.idx);
        }
      }
      if (warp_id + 32u < num_ties) {
        const auto target = smem->tie_buffer[warp_id + 32u];
        const auto rank = rank_target(target);
        if (lane_id == 0 && rank < topk_remain) {
          s_topk_indices[num_above + rank] = static_cast<int32_t>(target.idx);
        }
      }
    } else {
      // Generic path: each warp independently ranks one target at a time.
      for (uint32_t i = warp_id; i < num_ties; i += kNumWarps) {
        const auto target = smem->tie_buffer[i];
        uint32_t local_rank = 0;
        for (uint32_t j = lane_id; j < num_ties; j += 32) {
          const auto t = smem->tie_buffer[j];
          if (is_greater(t, target)) ++local_rank;
        }
        const auto rank = device::warp::reduce_sum(local_rank);
        if (lane_id == 0 && rank < topk_remain) {
          s_topk_indices[num_above + rank] = static_cast<int32_t>(target.idx);
        }
      }
    }
  }
  __syncthreads();

  // Page-translate the final raw indices and emit. Sentinels from
  // initialization stay as -1.
  for (uint32_t i = tx; i < kTopK; i += kBlockSize) {
    const auto raw = s_topk_indices[i];
    if (raw >= 0) {
      page_indices_ptr[i] = page_to_indices_pf(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = raw;
    } else {
      page_indices_ptr[i] = -1;
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = -1;
    }
  }
}

// ============================================================
// naive_transform_bf16
//
// Length <= kTopK: every position is selected; just translate.
// ============================================================
SGL_DEVICE static void naive_transform_bf16(
    const int32_t* __restrict__ page_ptr,
    int32_t* __restrict__ page_indices_ptr,
    int32_t* __restrict__ raw_indices_ptr,
    const uint32_t length,
    const uint32_t page_bits) {
  using namespace prefill_bf16;
  for (uint32_t i = threadIdx.x; i < kTopK; i += kBlockSize) {
    if (i < length) {
      page_indices_ptr[i] = page_to_indices_pf(page_ptr, i, page_bits);
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = static_cast<int32_t>(i);
    } else {
      page_indices_ptr[i] = -1;
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = -1;
    }
  }
}

// ============================================================
// register_topk_bf16_pf
//
// Adapted from topk_bf16.cuh's `register_topk_bf16<kIs2Pass>`.
// Loads the entire row (<= 32K bf16 scores) into per-thread
// registers (and a 16K-float smem staging buffer for the second
// pass), then runs the same coarse 12-bit histogram + tie-breaking
// pipeline used by the streaming variant. Avoids the cp.async
// double buffer entirely for short rows.
//
// `_smem` is reinterpret_cast<RegisterTopKSmemPF*>; the caller is
// responsible for sizing the dynamic smem accordingly.
// `indices` (== s_topk_indices) MUST be initialized to -1 by the
// caller before this is invoked, since num_above + num_equal == kTopK
// is only guaranteed when length > kTopK.
// ============================================================
template <bool kIs2Pass>
[[maybe_unused]]
SGL_DEVICE void register_topk_bf16_pf(
    const __nv_bfloat16* __restrict__ scores, int32_t* __restrict__ indices, const uint32_t length, void* _smem) {
  auto* smem = static_cast<RegisterTopKSmemPF*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  // Initialize shared memory histogram
  {
    typename RegisterTopKSmemPF::HistVec hist_vec;
    hist_vec.fill(0);
    smem->histogram_vec[tx] = hist_vec;
    if (tx == 0) {
      smem->counter_gt = smem->counter_eq = 0;
    }
    __syncthreads();
  }

  // Load scores into registers (bf16 -> float cast per element)
  Vec4PF local[prefill_bf16::kVecsPerThread];
#pragma unroll
  for (uint32_t v = 0; v < prefill_bf16::kVecsPerThread; ++v) {
    const uint32_t base = (tx + v * prefill_bf16::kBlockSize) * 4;
    if (base >= length) break;
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t idx = base + e;
      local[v][e] = (idx < length) ? static_cast<float>(scores[idx]) : 0.0f;
    }
  }

  // Async copy bf16 data from global to shared memory staging buffer, then cast to float
  if constexpr (kIs2Pass) {
    const uint32_t extra_length = length - prefill_bf16::kMax1PassLength;
    const __nv_bfloat16* src = scores + prefill_bf16::kMax1PassLength;

    // `src` inherits the row base alignment, which is only 2B in general because
    // score_stride / row_start are plain token counts. A 16B cp.async on a
    // misaligned source faults, so use a plain strided copy in that case.
    if ((reinterpret_cast<uintptr_t>(src) & 15u) != 0u) {
      for (uint32_t i = tx; i < extra_length; i += prefill_bf16::kBlockSize) {
        smem->bf16_stage[i] = src[i];
      }
      __pipeline_commit();  // keep the commit/wait pairing below intact
    } else {
      // Issue async copies: each thread copies kElemPerStage (8) bf16 elements per round.
      // extra_length is at most kMax1PassLength (16384) and kSizePerStage = 8192, so at most 2 rounds.
#pragma unroll 2
      for (uint32_t r = 0; r < 2; ++r) {
        const uint32_t global_base = r * prefill_bf16::kSizePerStage + tx * prefill_bf16::kElemPerStage;
        if (global_base + prefill_bf16::kElemPerStage <= extra_length) {
          __pipeline_memcpy_async(&smem->bf16_stage[global_base], &src[global_base], prefill_bf16::kCpAsyncBytes);
        } else if (global_base < extra_length) {
#pragma unroll
          for (uint32_t e = 0; e < prefill_bf16::kElemPerStage; ++e) {
            if (global_base + e < extra_length) {
              smem->bf16_stage[global_base + e] = src[global_base + e];
            }
          }
        }
      }
      __pipeline_commit();
    }
  }

  // Accumulate histogram via shared-memory atomics
#pragma unroll
  for (uint32_t v = 0; v < prefill_bf16::kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      if constexpr (!kIs2Pass) {
        const uint32_t idx = (tx + v * prefill_bf16::kBlockSize) * 4 + e;
        if (idx >= length) goto LABEL_ACC_FINISH;
      }
      atomicAdd(&smem->histogram[extract_coarse_bin_pf<prefill_bf16::kHistBits>(local[v][e])], 1);
    }
  }
  if constexpr (kIs2Pass) {
    const uint32_t extra_length = length - prefill_bf16::kMax1PassLength;
    __pipeline_wait_prior(0);
    __syncthreads();

    // Read 32-bit (2 bf16) from staging buffer and accumulate histogram directly
    const __nv_bfloat162* bf16_pairs = reinterpret_cast<const __nv_bfloat162*>(smem->bf16_stage);
    const uint32_t pair_len = extra_length / 2;
    for (uint32_t i = tx; i < pair_len; i += prefill_bf16::kBlockSize) {
      const __nv_bfloat162 pair = bf16_pairs[i];
      const float val0 = __bfloat162float(__low2bfloat16(pair));
      const float val1 = __bfloat162float(__high2bfloat16(pair));
      atomicAdd(&smem->histogram[extract_coarse_bin_pf<prefill_bf16::kHistBits>(val0)], 1);
      atomicAdd(&smem->histogram[extract_coarse_bin_pf<prefill_bf16::kHistBits>(val1)], 1);
    }
    // Handle trailing odd element
    if ((extra_length & 1) && tx == 0) {
      const float val = __bfloat162float(smem->bf16_stage[extra_length - 1]);
      atomicAdd(&smem->histogram[extract_coarse_bin_pf<prefill_bf16::kHistBits>(val)], 1);
    }
  }
[[maybe_unused]] LABEL_ACC_FINISH:
  __syncthreads();

  // Phase 2: Exclusive prefix scan -> find threshold bin
  {
    constexpr uint32_t kItems = prefill_bf16::kHistBins / prefill_bf16::kBlockSize;
    uint32_t orig[kItems];
    const auto hist_vec = smem->histogram_vec[tx];
    uint32_t tmp_local_sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      orig[i] = hist_vec[i];
      tmp_local_sum += orig[i];
    }

    const auto warp_inc = warp_inclusive_sum_pf(lane_id, tmp_local_sum);
    const auto warp_exc = warp_inc - tmp_local_sum;
    if (lane_id == 31) {
      smem->warp_sum[warp_id] = warp_inc;
    }

    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    uint32_t prefix_sum = device::warp::reduce_sum(lane_id < warp_id ? tmp : 0u);
    prefix_sum += warp_exc;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      prefix_sum += orig[i];
      const auto above = length - prefix_sum;
      if (above < prefill_bf16::kTopK && above + orig[i] >= prefill_bf16::kTopK) {
        smem->match = {
            .bin = tx * kItems + i,
            .above_count = above,
            .equal_count = orig[i],
        };
      }
    }
    __syncthreads();
  }

  const auto [thr_bin, num_above, num_equal] = smem->match;

  // Phase 3: Scatter
  constexpr uint32_t kMaxTolerance = 0;
  const bool need_tiebreak = (num_equal + num_above > prefill_bf16::kTopK + kMaxTolerance);
  const auto topk_indices = indices;
  const auto tie_buffer = smem->tie_buffer;

#pragma unroll
  for (uint32_t v = 0; v < prefill_bf16::kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t idx = (tx + v * prefill_bf16::kBlockSize) * 4 + e;
      if constexpr (!kIs2Pass) {
        if (idx >= length) goto LABEL_SCATTER_DONE;
      }
      const uint32_t bin = extract_coarse_bin_pf<prefill_bf16::kHistBits>(local[v][e]);
      if (bin > thr_bin) {
        topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
      } else if (bin == thr_bin) {
        const auto pos = atomicAdd(&smem->counter_eq, 1);
        if (need_tiebreak) {
          if (pos < prefill_bf16::kMaxTies) {
            tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
          }
        } else {
          if (const auto which = pos + num_above; which < prefill_bf16::kTopK) {
            topk_indices[which] = idx;
          }
        }
      }
    }
    // prefetch the next scores from bf16_stage into registers (for 2-pass)
    if constexpr (kIs2Pass) {
      const uint32_t pf_base = (tx + v * prefill_bf16::kBlockSize) * 4;
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        local[v][e] = __bfloat162float(smem->bf16_stage[pf_base + e]);
      }
    }
  }

  // 16K ~ 32K: process second chunk from registers (now loaded from bf16_stage)
  if constexpr (kIs2Pass) {
#pragma unroll
    for (uint32_t v = 0; v < prefill_bf16::kVecsPerThread; ++v) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t idx = (tx + v * prefill_bf16::kBlockSize) * 4 + e + prefill_bf16::kMax1PassLength;
        if (idx >= length) goto LABEL_SCATTER_DONE;
        const uint32_t bin = extract_coarse_bin_pf<prefill_bf16::kHistBits>(local[v][e]);
        if (bin > thr_bin) {
          topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (need_tiebreak) {
            if (pos < prefill_bf16::kMaxTies) {
              tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
            }
          } else {
            if (const auto which = pos + num_above; which < prefill_bf16::kTopK) {
              topk_indices[which] = idx;
            }
          }
        }
      }
    }
  }

[[maybe_unused]] LABEL_SCATTER_DONE:
  if (!need_tiebreak) return;

  // Phase 4: Tie-breaking within the threshold bin
  __syncthreads();

  const uint32_t num_ties = min(num_equal, prefill_bf16::kMaxTies);
  const uint32_t topk_remain = prefill_bf16::kTopK - num_above;

  const auto is_greater = [](const TieV3& a, const TieV3& b) {
    if (a.score > b.score) return true;
    if (a.score < b.score) return false;
    // Neither > nor < means equal OR NaN; use idx as deterministic tiebreaker
    return a.idx < b.idx;
  };

  if (num_ties <= 32u) {
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = tie_buffer[lane_id];
    const auto target_tie = tie_buffer[warp_id];
    const bool pred = is_greater(tie, target_tie);
    const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
    if (lane_id == 0 && rank < topk_remain) {
      topk_indices[num_above + rank] = target_tie.idx;
    }
  } else if (num_ties <= 64u) {
    const auto lane_id_1 = lane_id + 32;
    const auto warp_id_1 = warp_id + 32;
    const auto invalid = TieV3{.idx = 0xFFFFFFFF, .score = -FLT_MAX};
    const auto tie_0 = tie_buffer[lane_id];
    const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
    if (true) {
      const auto target = tie_buffer[warp_id];
      const bool pred_0 = is_greater(tie_0, target);
      const bool pred_1 = is_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    }
    if (warp_id_1 < num_ties) {
      const auto target = tie_buffer[warp_id_1];
      const bool pred_0 = is_greater(tie_0, target);
      const bool pred_1 = is_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    }
  } else {
    [[unlikely]];
    for (auto i = warp_id; i < num_ties; i += prefill_bf16::kNumWarps) {
      const auto target_tie = tie_buffer[i];
      uint32_t local_rank = 0;
      for (auto j = lane_id; j < num_ties; j += 32) {
        const auto tie = tie_buffer[j];
        if (is_greater(tie, target_tie)) local_rank++;
      }
      const auto rank = device::warp::reduce_sum(local_rank);
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target_tie.idx;
      }
    }
  }
}

// ============================================================
// register_transform_bf16
//
// Page-translates the indices written by `register_topk_bf16_pf`
// (raw absolute offsets within the row) and emits to the global
// outputs. Mirrors the tail of `tie_handle_and_transform_bf16`.
// ============================================================
SGL_DEVICE static void register_transform_bf16(
    const TopKPrefillBF16Params& params,
    const int32_t* __restrict__ s_topk_indices,
    int32_t* __restrict__ page_indices_ptr,
    int32_t* __restrict__ raw_indices_ptr,
    const int32_t* __restrict__ page_ptr) {
  using namespace prefill_bf16;
  for (uint32_t i = threadIdx.x; i < kTopK; i += kBlockSize) {
    const auto raw = s_topk_indices[i];
    if (raw >= 0) {
      page_indices_ptr[i] = page_to_indices_pf(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = raw;
    } else {
      page_indices_ptr[i] = -1;
      if (raw_indices_ptr != nullptr) raw_indices_ptr[i] = -1;
    }
  }
}

// ============================================================
// streaming_topk_bf16
//
// Drives the full streaming flow over an arbitrary length:
//   A. histogram pass  (cp.async double buffered)
//   B. find threshold
//   C. scatter pass    (cp.async double buffered)
//
// `scores_bf16` is the 16B-aligned-down row base and `base_offset`
// (< kElemPerStage) is how many elements the real row start sits
// past it; both passes mask the head elements out.
//
// The caller is responsible for calling tie_handle_and_transform_bf16
// afterwards.
// ============================================================
SGL_DEVICE static void streaming_topk_bf16(
    const __nv_bfloat16* __restrict__ scores_bf16,
    const uint32_t base_offset,
    const uint32_t length,
    int32_t* __restrict__ s_topk_indices,
    PrefillBF16Smem* smem) {
  using namespace prefill_bf16;
  const auto tx = threadIdx.x;

  // Init shared state: histogram, counters, sentinel topk slots.
  {
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      smem->histogram[tx * kHistItems + i] = 0u;
    }
    if (tx == 0) {
      smem->counter_gt = 0u;
      smem->counter_eq = 0u;
    }
    for (uint32_t i = tx; i < kTopK; i += kBlockSize) {
      s_topk_indices[i] = -1;
    }
    __syncthreads();
  }

  // Phase A: build histogram via streaming cp.async pipeline.
  stream_pass_bf16<false>(scores_bf16, base_offset, length, /*thr_bin=*/0u, /*topk=*/nullptr, smem);

  // Phase B: locate threshold bin.
  find_threshold_bf16(length, smem);

  // Reset counters before scatter; tie_buffer aliases histogram so
  // we must not touch histogram between this point and the scatter.
  if (tx == 0) {
    smem->counter_gt = 0u;
    smem->counter_eq = 0u;
  }
  __syncthreads();

  // Phase C: scatter pass.
  const auto thr_bin = smem->match.bin;
  scatter_pass_bf16_warp_steal(scores_bf16, base_offset, length, thr_bin, s_topk_indices, smem);
}

// ============================================================
// Register kernel (short/medium sequences <= 32K)
//
// Handles naive passthrough (<=kTopK), 1-pass register (<=16K),
// and 2-pass register (<=32K) paths. Only allocates
// RegisterTopKSmemPF + s_topk_indices smem.
// ============================================================
__global__ __launch_bounds__(prefill_bf16::kBlockSize, 1) void topk_prefill_bf16_register_kernel(
    const TopKPrefillBF16Params params) {
  using namespace prefill_bf16;

  extern __shared__ char smem_raw[];
  auto* s_topk_indices = reinterpret_cast<int32_t*>(smem_raw + sizeof(RegisterTopKSmemPF));

  const auto row_id = blockIdx.x;
  const auto row_start = params.row_starts[row_id];
  const auto row_end = params.row_ends[row_id];
  const auto length = row_end > row_start ? static_cast<uint32_t>(row_end - row_start) : 0u;

  const auto scores_ptr = params.scores + static_cast<int64_t>(row_id) * params.score_stride + row_start;
  const auto page_ptr = params.page_table + static_cast<int64_t>(row_id) * params.page_table_stride;
  auto page_indices_ptr = params.page_indices + static_cast<int64_t>(row_id) * kTopK;
  auto raw_indices_ptr =
      params.raw_indices == nullptr ? nullptr : params.raw_indices + static_cast<int64_t>(row_id) * kTopK;

  if (length <= kTopK) {
    // Use stride loop to handle kTopK > kBlockSize correctly.
    const auto tx = threadIdx.x;
    for (uint32_t i = tx; i < kTopK; i += kBlockSize) {
      if (i < length) {
        page_indices_ptr[i] = page_to_indices_pf(page_ptr, i, params.page_bits);
        if (raw_indices_ptr != nullptr) {
          raw_indices_ptr[i] = static_cast<int32_t>(i);
        }
      } else {
        page_indices_ptr[i] = -1;
        if (raw_indices_ptr != nullptr) {
          raw_indices_ptr[i] = -1;
        }
      }
    }
  } else if (length <= kMax1PassLength) {
    register_topk_bf16_pf<false>(scores_ptr, s_topk_indices, length, smem_raw);
    __syncthreads();
    // Use stride loop to handle kTopK > kBlockSize correctly.
    const auto tx = threadIdx.x;
    for (uint32_t i = tx; i < kTopK; i += kBlockSize) {
      const auto raw = s_topk_indices[i];
      if (raw >= 0) {
        page_indices_ptr[i] = page_to_indices_pf(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
        if (raw_indices_ptr != nullptr) {
          raw_indices_ptr[i] = raw;
        }
      } else {
        page_indices_ptr[i] = -1;
        if (raw_indices_ptr != nullptr) {
          raw_indices_ptr[i] = -1;
        }
      }
    }
  } else if (length <= kMax2PassLength) {
    [[likely]];
    register_topk_bf16_pf<true>(scores_ptr, s_topk_indices, length, smem_raw);
    __syncthreads();
    // Use stride loop to handle kTopK > kBlockSize correctly.
    const auto tx = threadIdx.x;
    for (uint32_t i = tx; i < kTopK; i += kBlockSize) {
      const auto raw = s_topk_indices[i];
      if (raw >= 0) {
        page_indices_ptr[i] = page_to_indices_pf(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
        if (raw_indices_ptr != nullptr) {
          raw_indices_ptr[i] = raw;
        }
      } else {
        page_indices_ptr[i] = -1;
        if (raw_indices_ptr != nullptr) {
          raw_indices_ptr[i] = -1;
        }
      }
    }
  }
}

// ============================================================
// Streaming kernel (long sequences > 32K)
//
// Handles the cp.async double-buffered streaming top-k path.
// Allocates PrefillBF16Smem + s_topk_indices smem.
// For batch rows whose actual length happens to be short,
// falls back to naive passthrough or streaming (since smem is
// already sized for streaming).
// ============================================================
__global__ __launch_bounds__(prefill_bf16::kBlockSize, 1) void topk_prefill_bf16_stream_kernel(
    const TopKPrefillBF16Params params) {
  using namespace prefill_bf16;

  extern __shared__ char smem_raw[];
  auto* smem = reinterpret_cast<PrefillBF16Smem*>(smem_raw);
  auto* s_topk_indices = reinterpret_cast<int32_t*>(smem_raw + sizeof(PrefillBF16Smem));

  const auto row_id = blockIdx.x;
  const auto row_start = params.row_starts[row_id];
  const auto row_end = params.row_ends[row_id];
  const auto length = row_end > row_start ? static_cast<uint32_t>(row_end - row_start) : 0u;

  const auto scores_raw = params.scores + static_cast<int64_t>(row_id) * params.score_stride + row_start;
  const auto page_ptr = params.page_table + static_cast<int64_t>(row_id) * params.page_table_stride;
  auto page_indices_ptr = params.page_indices + static_cast<int64_t>(row_id) * kTopK;
  auto raw_indices_ptr =
      params.raw_indices == nullptr ? nullptr : params.raw_indices + static_cast<int64_t>(row_id) * kTopK;

  if (length <= kTopK) {
    naive_transform_bf16(page_ptr, page_indices_ptr, raw_indices_ptr, length, params.page_bits);
    return;
  }

  // `score_stride` and `row_start` are plain token counts, so the row base can
  // land on any 2B boundary while cp.async / uint4 need 16B. Align the base down
  // and let the passes skip the resulting head elements. Aligning down never
  // leaves the tensor: the allocator guarantees a 16B-aligned `params.scores`,
  // so the misalignment is always <= the row's element offset.
  const uint32_t base_offset = bf16_misalign_elems_pf(scores_raw);
  const auto scores_ptr = scores_raw - base_offset;

  // All rows with length > kTopK use the streaming path in this kernel,
  // since smem is already sized for PrefillBF16Smem.
  streaming_topk_bf16(scores_ptr, base_offset, length, s_topk_indices, smem);
  tie_handle_and_transform_bf16(params, s_topk_indices, page_indices_ptr, raw_indices_ptr, page_ptr, smem);
}

// ============================================================
// Host-side launcher
// ============================================================
template <auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once_pf(host::DebugInfo where = {}) {
  [[maybe_unused]]
  static const auto result = [] {
    const auto fptr = std::bit_cast<const void*>(f);
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
  host::RuntimeDeviceCheck(result, where);
}

struct TopKPrefillBF16Kernel {
  static void transform(
      const tvm::ffi::TensorView scores,        // [batch, max_len] bf16
      const tvm::ffi::TensorView row_starts,    // [batch] int32
      const tvm::ffi::TensorView row_ends,      // [batch] int32
      const tvm::ffi::TensorView page_table,    // [batch, num_pages] int32
      const tvm::ffi::TensorView page_indices,  // [batch, kTopK] int32
      const uint32_t page_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<bf16_t>().with_device(device_).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_starts);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_ends);
    TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({B, prefill_bf16::kTopK}).with_dtype<int32_t>().with_device(device_).verify(page_indices);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, prefill_bf16::kTopK}).with_dtype<int32_t>().with_device(device_).verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto params = TopKPrefillBF16Params{
        .scores = static_cast<const __nv_bfloat16*>(scores.data_ptr()),
        .row_starts = static_cast<int32_t*>(row_starts.data_ptr()),
        .row_ends = static_cast<int32_t*>(row_ends.data_ptr()),
        .page_table = static_cast<int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = page_bits,
    };

    // Host-side smem routing based on max possible per-row length.
    // The score tensor's second dim (score_stride S) is an upper bound on
    // (row_ends[i] - row_starts[i]), so we can safely pick the kernel with
    // the minimal smem footprint.
    const auto max_seq_len = static_cast<uint32_t>(S.unwrap());

    if (max_seq_len <= prefill_bf16::kMax2PassLength) {
      // Short/medium sequences: register kernel only needs RegisterTopKSmemPF.
      constexpr auto kernel = topk_prefill_bf16_register_kernel;
      constexpr auto kSMEM_ = sizeof(RegisterTopKSmemPF) + prefill_bf16::kTopK * sizeof(int32_t);
      setup_kernel_smem_once_pf<kernel, kSMEM_>();
      LaunchKernel(batch_size, prefill_bf16::kBlockSize, device_.unwrap(), kSMEM_)(kernel, params);
    } else {
      // Long sequences: streaming kernel needs PrefillBF16Smem.
      constexpr auto kernel = topk_prefill_bf16_stream_kernel;
      constexpr auto kSMEM_ = sizeof(PrefillBF16Smem) + prefill_bf16::kTopK * sizeof(int32_t);
      setup_kernel_smem_once_pf<kernel, kSMEM_>();
      LaunchKernel(batch_size, prefill_bf16::kBlockSize, device_.unwrap(), kSMEM_)(kernel, params);
    }
  }
};

}  // namespace
}  // namespace sglang
