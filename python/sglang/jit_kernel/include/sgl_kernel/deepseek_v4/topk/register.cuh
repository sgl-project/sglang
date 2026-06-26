#pragma once

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include "common.cuh"
#include "ptx.cuh"
#include <cfloat>
#include <cstdint>

namespace device::top512 {

template <uint32_t K>
struct RegisterTopK {
  static constexpr uint32_t kHistBits = 12;
  static constexpr uint32_t kHistBins = 1 << kHistBits;
  static constexpr uint32_t kVecsPerThread = 4;
  static constexpr uint32_t kMaxTolerance = 0;
  static constexpr uint32_t kMax1PassLength = kVecsPerThread * 4 * kBlockSize;
  static constexpr uint32_t kMaxExtraLength = kMax1PassLength;
  static constexpr uint32_t kMax2PassLength = kMax1PassLength + kMaxExtraLength;

  struct Smem {
    using HistVec = AlignedVector<uint32_t, kHistBins / kBlockSize>;
    alignas(128) uint32_t counter_gt;
    alignas(128) uint32_t counter_eq;
    uint64_t mbarrier;  // for cp.async
    MatchBin match;
    uint32_t warp_sum[kNumWarps];
    union {
      uint32_t histogram[kHistBins];
      HistVec histogram_vec[kBlockSize];
      Tie tie_buffer[kMaxTies];
    };
    alignas(16) float score_buffer[kMaxExtraLength];
  };

  template <bool kIs2Pass = false>
  SGL_DEVICE static void
  run(const float* scores,  //
      int32_t* indices,
      const uint32_t length,
      void* _smem,
      const bool use_pdl = false) {
    const auto smem = static_cast<Smem*>(_smem);
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpThreads;
    const auto warp_id = tx / kWarpThreads;

    // Initialize shared memory histogram
    {
      typename Smem::HistVec hist_vec;
      hist_vec.fill(0);
      smem->histogram_vec[tx] = hist_vec;
      if (tx == 0) {
        smem->counter_gt = smem->counter_eq = 0;
        if constexpr (kIs2Pass) {
          ptx::mbarrier_init(&smem->mbarrier, 1);
        }
      }
      __syncthreads();
    }

    if (use_pdl) device::PDLWaitPrimary<true>();

    // Load scores into registers
    Vec4 local[kVecsPerThread];
#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v) {
      const uint32_t base = (tx + v * kBlockSize) * 4;
      if (base >= length) break;
      local[v].load(scores, tx + v * kBlockSize);
    }

    // Fetch the next chunk of scores
    if constexpr (kIs2Pass) {
      if (ptx::elect_sync_cta(tx)) {
        const auto length_aligned = (length + 3u - kMax1PassLength) & ~3u;
        const auto size_bytes = length_aligned * sizeof(float);
        ptx::tma_load(smem->score_buffer, scores + kMax1PassLength, size_bytes, &smem->mbarrier);
        ptx::mbarrier_arrive_expect_tx(&smem->mbarrier, size_bytes);
      }
      __syncwarp();  // avoid warp divergence on
    }

    // Accumulate histogram via shared-memory atomics
#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        if constexpr (!kIs2Pass) {
          const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
          if (idx >= length) goto LABEL_ACC_FINISH;
        }
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(local[v][e])], 1);
      }
    }
    if constexpr (kIs2Pass) {
      // 16K ~ 32K. `i` is a float4 index
      if (lane_id == 0) ptx::mbarrier_wait(&smem->mbarrier, 0);
      __syncwarp();
      for (uint32_t i = tx; i + kMax1PassLength < length; i += kBlockSize) {
        const auto val = smem->score_buffer[i];
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(val)], 1);
      }
    }
  [[maybe_unused]] LABEL_ACC_FINISH:
    __syncthreads();

    // Phase 2: Exclusive prefix scan -> find threshold bin
    {
      constexpr uint32_t kItems = kHistBins / kBlockSize;
      uint32_t orig[kItems];
      const auto hist_vec = smem->histogram_vec[tx];
      uint32_t tmp_local_sum = 0;

#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        orig[i] = hist_vec[i];
        tmp_local_sum += orig[i];
      }

      const auto warp_inc = warp_inclusive_sum(lane_id, tmp_local_sum);
      const auto warp_exc = warp_inc - tmp_local_sum;
      if (lane_id == kWarpThreads - 1) {
        smem->warp_sum[warp_id] = warp_inc;
      }

      __syncthreads();

      const auto tmp = smem->warp_sum[lane_id];
      // Exactly one bin satisfies: above < K && above + count >= K
      uint32_t prefix_sum = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
      prefix_sum += warp_exc;
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        prefix_sum += orig[i];
        const auto above = length - prefix_sum;
        if (above < K && above + orig[i] >= K) {
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
    // Elements strictly above threshold go directly to output.
    // Tied elements: simple path admits first-come; tiebreak path collects into tie_buffer.
    const bool need_tiebreak = (num_equal + num_above > K + kMaxTolerance);
    const auto topk_indices = indices;
    const auto tie_buffer = smem->tie_buffer;

#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
        if constexpr (!kIs2Pass) {
          if (idx >= length) goto LABEL_SCATTER_DONE;
        }
        const uint32_t bin = extract_coarse_bin<kHistBits>(local[v][e]);
        if (bin > thr_bin) {
          topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (need_tiebreak) {
            if (pos < kMaxTies) {
              tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
            }
          } else {
            if (const auto which = pos + num_above; which < K) {
              topk_indices[which] = idx;
            }
          }
        }
      }
      // prefetch the next scores
      if constexpr (kIs2Pass) {
        local[v].load(smem->score_buffer, tx + v * kBlockSize);
      }
    }

    // 16K ~ 32K, already in registers: similar loop as above but read from smem->score_buffer
    if constexpr (kIs2Pass) {
#pragma unroll
      for (uint32_t v = 0; v < kVecsPerThread; ++v) {
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
          const uint32_t idx = (tx + v * kBlockSize) * 4 + e + kMax1PassLength;
          if (idx >= length) goto LABEL_SCATTER_DONE;
          const uint32_t bin = extract_coarse_bin<kHistBits>(local[v][e]);
          if (bin > thr_bin) {
            topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
          } else if (bin == thr_bin) {
            const auto pos = atomicAdd(&smem->counter_eq, 1);
            if (need_tiebreak) {
              if (pos < kMaxTies) {
                tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
              }
            } else {
              if (const auto which = pos + num_above; which < K) {
                topk_indices[which] = idx;
              }
            }
          }
        }
      }
    }

  [[maybe_unused]] LABEL_SCATTER_DONE:
    if (!need_tiebreak) return;

    // Phase 4: Tie-breaking within the threshold bin.
    // Assume num_ties <= kBlockSize (at most 1 block of ties).
    // Each thread takes one tied element, computes its rank (number of
    // elements with strictly higher score, breaking exact float ties by
    // original index), and writes to output if rank < topk_remain.
    __syncthreads();
    static_assert(kMaxTies <= kBlockSize);

    const uint32_t num_ties = min(num_equal, kMaxTies);
    const uint32_t topk_remain = K - num_above;

    const auto is_greater = [](const Tie& a, const Tie& b) {
      return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
    };

    if (num_ties <= kWarpThreads) {
      static_assert(kWarpThreads <= kNumWarps);
      if (lane_id >= num_ties || warp_id >= num_ties) return;  // some threads are idle
      /// NOTE: use long long to avoid mask overflow when num_ties == 32
      const uint32_t mask = (1ull << num_ties) - 1u;
      const auto tie = tie_buffer[lane_id];
      const auto target_tie = tie_buffer[warp_id];
      const bool pred = is_greater(tie, target_tie);
      const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target_tie.idx;
      }
    } else if (num_ties <= kWarpThreads * 2) {
      // 64 x 64 topk implementation: each thread takes 2 elements
      const auto lane_id_1 = lane_id + kWarpThreads;
      const auto warp_id_1 = warp_id + kWarpThreads;
      const auto invalid = Tie{.idx = 0xFFFFFFFF, .score = -FLT_MAX};
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
      /// NOTE: Based on my observation, this path is very rarely reached
      [[unlikely]];
      // Block-level: each thread reads from tie_buffer in shared memory
      for (auto i = warp_id; i < num_ties; i += kNumWarps) {
        const auto target_tie = tie_buffer[i];
        uint32_t local_rank = 0;
        for (auto j = lane_id; j < num_ties; j += kWarpThreads) {
          const auto tie = tie_buffer[j];
          if (is_greater(tie, target_tie)) local_rank++;
        }
        // sum the rank across the warp
        const auto rank = warp::reduce_sum(local_rank);
        if (lane_id == 0 && rank < topk_remain) {
          topk_indices[num_above + rank] = target_tie.idx;
        }
      }
    }
  }

  SGL_DEVICE static void transform(const TransformParams params) {
    __syncthreads();
    if (const auto tx = threadIdx.x; tx < K) params.transform(tx);
  }
};

}  // namespace device::top512
