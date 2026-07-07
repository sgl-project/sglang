#pragma once
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include "common.cuh"
#include "ptx.cuh"
#include <cooperative_groups.h>
#include <cstdint>

namespace device::top512 {

template <uint32_t K>
struct ClusterTopK {
  static constexpr uint32_t kClusterSize = 8;
  static constexpr uint32_t kHistBits = 10;
  static constexpr uint32_t kHistBins = 1 << kHistBits;
  static constexpr uint32_t kRadixBins = 256;
  static constexpr uint32_t kElemPerStage = 8;
  static constexpr uint32_t kSizePerStage = kElemPerStage * kBlockSize;
  static constexpr uint32_t kNumStages = 4;
  static constexpr uint32_t kMaxLength = kClusterSize * kNumStages * kSizePerStage;
  static constexpr uint32_t kStoreLane = kBlockSize - 1;
  static constexpr uint32_t kAboveBits = 11;

  // ---------------------------------------------------------------------------
  // Shared memory layouts
  // ---------------------------------------------------------------------------

  struct Smem {
    uint64_t barrier[kNumStages];
    uint32_t local_above_equal[kClusterSize];
    uint32_t prefix_above_equal;
    alignas(128) uint32_t counter_gt;
    alignas(128) uint32_t counter_eq;
    alignas(128) MatchBin match;
    alignas(128) uint32_t warp_sum[kNumWarps];
    uint32_t histogram[kHistBins];
    alignas(128) float score_buffer[kNumStages][kSizePerStage];
    Tie tie_buffer[kMaxTies];
  };

  struct alignas(16) Metadata {
    uint32_t batch_id;
    uint32_t seq_len;
    bool has_next;
  };

  struct WorkSpace {
    uint2 metadata;  // {num_above, num_ties}
    Tie ties[kMaxTies];
  };

  static constexpr uint32_t kWorkspaceInts = sizeof(WorkSpace) / sizeof(uint32_t);

  // ---------------------------------------------------------------------------
  // Stage 1: histogram + cluster reduce + find threshold + scatter
  // ---------------------------------------------------------------------------

  SGL_DEVICE static void stage1_init(void* _smem) {
    const auto tx = threadIdx.x;
    __builtin_assume(tx < kBlockSize);
    const auto smem = static_cast<Smem*>(_smem);
    if (tx < kHistBins) smem->histogram[tx] = 0;
    if (tx < kNumStages) ptx::mbarrier_init(&smem->barrier[tx], 1);
    __syncthreads();
  }

  SGL_DEVICE static void stage1_prologue(const float* scores, uint32_t length, void* _smem) {
    if (threadIdx.x == 0) {
      const auto smem = static_cast<Smem*>(_smem);
      const auto num_stages = (length + kSizePerStage - 1) / kSizePerStage;
      const auto length_aligned = (length + 3u) & ~3u;  // align to 4 for TMA
#pragma unroll
      for (uint32_t stage = 0; stage < kNumStages; stage++) {
        if (stage >= num_stages) break;
        const auto offset = stage * kSizePerStage;
        const auto size = min(kSizePerStage, length_aligned - offset);
        const auto size_bytes = size * sizeof(float);
        const auto bar = &smem->barrier[stage];
        ptx::tma_load(smem->score_buffer[stage], scores + offset, size_bytes, bar);
        ptx::mbarrier_arrive_expect_tx(bar, size_bytes);
      }
    }
  }

  SGL_DEVICE static void stage1(int32_t* indices, uint32_t length, void* _smem, bool reuse = false) {
    const auto smem = static_cast<Smem*>(_smem);
    const auto tx = threadIdx.x;
    __builtin_assume(tx < kBlockSize);
    const auto lane_id = tx % kWarpThreads;
    const auto warp_id = tx / kWarpThreads;

    // Initialize shared memory histogram, counters, and barriers
#pragma unroll
    for (uint32_t stage = 0; stage < kNumStages; stage++) {
      const auto offset = stage * kSizePerStage;
      if (offset >= length) break;
      const auto size = min(kSizePerStage, length - offset);
      if (lane_id == 0) ptx::mbarrier_wait(&smem->barrier[stage], 0);
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; ++i) {
        const auto idx = tx + i * kBlockSize;
        if (idx >= size) break;
        const auto score = smem->score_buffer[stage][idx];
        const auto bin = extract_coarse_bin<kHistBits>(score);
        atomicAdd(&smem->histogram[bin], 1);
      }
    }

    static_assert(kHistBins <= kBlockSize);

    // 2-shot all-reduce
    {
      auto cluster = cooperative_groups::this_cluster();
      cluster.sync();
      const auto cluster_rank = blockIdx.y;
      const auto kLocalSize = kHistBins / kClusterSize;
      const auto offset = kLocalSize * cluster_rank;

      const auto src_tx = tx / kClusterSize;
      const auto src_rank = tx % kClusterSize;

      if (tx < kHistBins) {
        const auto addr = &smem->histogram[offset + src_tx];
        const auto src_addr = cluster.map_shared_rank(addr, src_rank);
        *src_addr = warp::reduce_sum<kClusterSize>(*src_addr);
      }
      cluster.sync();
    }

    // now each block holds the whole histogram, find the threshold bin
    {
      const auto value = tx < kHistBins ? smem->histogram[tx] : 0;
      const auto warp_inc = warp_inclusive_sum(lane_id, value);
      if (lane_id == kWarpThreads - 1) {
        smem->warp_sum[warp_id] = warp_inc;
      }

      __syncthreads();
      const auto tmp = smem->warp_sum[lane_id];
      // total_length = sum of all bins in the globally-reduced histogram
      // (problem.length is block-local; after cluster reduction we need the global total)
      const auto total_length = warp::reduce_sum(tmp);
      uint32_t prefix_sum = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
      prefix_sum += warp_inc;
      const auto above = total_length - prefix_sum;
      if (tx < kHistBins && above < K && above + value >= K) {
        smem->counter_gt = smem->counter_eq = 0;
        smem->match = {
            .bin = tx,
            .above_count = above,
            .equal_count = value,
        };
      }
      __syncthreads();
    }

    const auto [thr_bin, num_above, num_equal] = smem->match;

    // write above and equal results to global memory
#pragma unroll
    for (uint32_t stage = 0; stage < kNumStages; stage++) {
      const auto offset = stage * kSizePerStage;
      if (offset >= length) break;
#pragma unroll
      for (uint32_t i = 0; i < kElemPerStage; ++i) {
        const auto buf_idx = tx + i * kBlockSize;
        const auto global_idx = offset + buf_idx;
        if (global_idx >= length) break;
        const auto score = smem->score_buffer[stage][buf_idx];
        const auto bin = extract_coarse_bin<kHistBits>(score);
        if (bin > thr_bin) {
          indices[atomicAdd(&smem->counter_gt, 1)] = global_idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (pos < kMaxTies) smem->tie_buffer[pos] = {global_idx, score};
        }
      }
    }
    if (reuse) {
      const auto num_stages = (length + kSizePerStage - 1) / kSizePerStage;
      if (tx < kHistBins) smem->histogram[tx] = 0;
      if (tx < num_stages) ptx::mbarrier_arrive(&smem->barrier[tx]);
    }
    __syncthreads();
  }

  // ---------------------------------------------------------------------------
  // Stage 1 epilogue: cross-block prefix sum + page translate + tie store
  // ---------------------------------------------------------------------------

  SGL_DEVICE static void stage1_epilogue(const TransformParams params, const uint32_t offset, void* _ws, void* _smem) {
    auto cluster = cooperative_groups::this_cluster();
    const auto smem = static_cast<Smem*>(_smem);
    const auto tx = threadIdx.x;
    const auto local_above = smem->counter_gt;
    const auto local_equal = smem->counter_eq;
    const auto cluster_rank = blockIdx.y;

    constexpr uint32_t kAboveMask = (1 << kAboveBits) - 1;
    static_assert(kAboveMask >= K);

    // Pack local counts -- NO alignment rounding (contiguous layout)
    static_assert(kMaxTies <= kBlockSize);
    const auto idx_above = tx < local_above ? params.indices_in[tx] : 0;
    const auto tie_value = tx < local_equal ? smem->tie_buffer[tx] : Tie{0, 0.0f};

    // push to remote shared memory, can reduce latency of reading remote
    if (tx < kClusterSize) {
      const auto value = (local_equal << kAboveBits) | local_above;
      const auto dst_addr = cluster.map_shared_rank(smem->local_above_equal, tx);
      dst_addr[cluster_rank] = value;
    }
    // after this last sync, only read local shared memory
    // so that it is safe when peer rank has already exited the kernel
    cluster.sync();
    if (tx < kClusterSize) {
      const auto value = tx < cluster_rank ? smem->local_above_equal[tx] : 0;
      const auto kActiveMask = (1u << kClusterSize) - 1;
      smem->prefix_above_equal = warp::reduce_sum<kClusterSize>(value, kActiveMask);
    }
    __syncthreads();

    const auto prefix_packed = smem->prefix_above_equal;
    const auto prefix_above = prefix_packed & kAboveMask;
    const auto prefix_equal = prefix_packed >> kAboveBits;

    // Page-translate above elements
    if (tx < local_above) {
      params.write(tx + prefix_above, idx_above + offset);
    }
    // Contiguous tie store via regular global writes (no TMA, no gaps)
    const auto ws = static_cast<WorkSpace*>(_ws);
    if (tx < local_equal && tx + prefix_equal < kMaxTies) {
      ws->ties[tx + prefix_equal] = {tie_value.idx + offset, tie_value.score};
    }
    // Block 0 writes global metadata {num_above, num_ties}
    if (cluster_rank == kClusterSize - 1 && tx == 0) {
      const auto sum_above = prefix_above + local_above;
      const auto sum_equal = prefix_equal + local_equal;
      ws->metadata = make_uint2(sum_above, sum_equal);
    }
  }

  SGL_DEVICE static void transform(const TransformParams params, const void* _ws, void* _smem) {
    const auto ws = static_cast<const WorkSpace*>(_ws);
    const auto meta = &ws->metadata;
    const auto [num_above, num_equal] = *meta;
    if (num_above >= K || num_equal == 0) return;
    const auto clamped_ties = min(num_equal, kMaxTies);
    tie_handle_transform(ws->ties, clamped_ties, num_above, K, params, _smem);
  }
};

}  // namespace device::top512
