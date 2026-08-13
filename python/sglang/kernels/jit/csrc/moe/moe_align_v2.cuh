#include <sgl_kernel/allocator.h>
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>

namespace sglang {

constexpr uint32_t kCTASize = 1024;

#define ALIGN_KERNEL __global__ __launch_bounds__(kCTASize, 1)

/// \brief Kernel arguments shared by both paths.
struct MoEAlignParams {
  const int32_t* topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ expert_ids;
  int32_t* __restrict__ total_tokens_post_pad;
  /// Caller-owned scratch, the general path's per-bucket row cursors.
  uint32_t* __restrict__ cumsum_buffer;
  /// Kernel-owned single word, the small path's `atomic::Event`.
  void* __restrict__ event;
  uint32_t numel;  // batch_size * topk
  uint32_t block_size;
  uint32_t buffer_vecs;
  uint32_t num_buckets;  // E + 1 under the "+1 offset" convention
  bool pad_sorted_token_ids;
};

/**
 * \brief Scatter each (token, slot) pair into its slot. One thread per pair.
 *
 * `cumsum_buffer` already holds each bucket's row offset; the atomicAdd both
 * reserves a slot and advances the cursor, so intra-bucket order is atomicAdd
 * scheduling order.
 */
template <bool kIgnoreInvalid, bool kUsePDL>
ALIGN_KERNEL void count_and_sort_expert_tokens_kernel(const __grid_constant__ MoEAlignParams params) {
  const auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx < params.numel) {
    device::PDLWaitPrimary<kUsePDL>();
    const auto expert_id = params.topk_ids[global_idx];
    if constexpr (kIgnoreInvalid) {
      if (expert_id < 0) return;
    }
    const auto rank_post_pad = atomicAdd(&params.cumsum_buffer[expert_id + 1], 1);
    params.sorted_token_ids[rank_post_pad] = global_idx;
  }
}

/**
 * \brief Histogram, padded scan, expert_ids, and the pad prefill.
 *
 * Block 0 does all the compute; every other block only prefills
 * `sorted_token_ids`, so the bandwidth-heavy fill spreads over as many SMs as the
 * launcher asks for instead of sitting on one.
 */
template <bool kIgnoreInvalid, bool kUsePDL>
ALIGN_KERNEL void moe_align_block_size_kernel(const __grid_constant__ MoEAlignParams params) {
  using namespace device;
  using vec_t = AlignedVector<int32_t, 4>;
  constexpr uint32_t kVecSize = 4;

  if (blockIdx.x > 0) {
    if (params.pad_sorted_token_ids) {
      PDLWaitPrimary<kUsePDL>();
      vec_t fill_vec;
      fill_vec.fill(static_cast<int32_t>(params.numel));
      // Stride over the FILL blocks only. Counting block 0 in would leave the
      // last blockDim.x-sized window of the buffer unwritten.
      const auto global_idx = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
      const auto stride = (gridDim.x - 1) * blockDim.x;
      for (auto idx = global_idx; idx < params.buffer_vecs; idx += stride) {
        fill_vec.store(params.sorted_token_ids, idx);
      }
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  constexpr uint32_t kNumWarps = kCTASize / kWarpThreads;
  static_assert(kNumWarps <= kWarpThreads, "per-warp sums must fit one warp of lanes");

  // Sized for the widest bucket count this kernel accepts; the launcher checks it.
  __shared__ uint32_t s_counts[kCTASize];
  __shared__ uint32_t s_warp_sums[kNumWarps];
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto num_buckets = params.num_buckets;
  s_counts[tx] = 0;
  if (tx < kNumWarps) s_warp_sums[tx] = 0;
  __syncthreads();

  PDLWaitPrimary<kUsePDL>();

  const auto for_each_expert = [&](auto&& fn) {
    const auto complete_vecs = params.numel / kVecSize;
    for (auto idx = tx; idx < complete_vecs; idx += kCTASize) {
      vec_t topk_id;
      topk_id.load(params.topk_ids, idx);
#pragma unroll
      for (uint32_t v = 0; v < kVecSize; ++v) {
        const auto expert_id = topk_id[v];
        if constexpr (kIgnoreInvalid) {
          if (expert_id < 0) continue;
        }
        fn(expert_id, idx * kVecSize + v);
      }
    }
    if (tx < params.numel % kVecSize) {
      const auto idx = complete_vecs * kVecSize + tx;
      const auto expert_id = params.topk_ids[idx];
      if constexpr (kIgnoreInvalid) {
        if (expert_id < 0) return;
      }
      fn(expert_id, idx);
    }
  };

  for_each_expert([&](int32_t expert_id, uint32_t) {
    // +1 offset: bucket = expert_id + 1, so an EP-filtered -1 lands in bucket 0
    atomicAdd(&s_counts[expert_id + 1], 1);
  });
  __syncthreads();

  // Exclusive scan of per-bucket BLOCK counts. Scanning blocks rather than padded
  // rows keeps the expert_ids fill a plain loop over block indices and avoids
  // dividing by the runtime block_size.
  const auto block_size = params.block_size;
  const auto count = tx < num_buckets ? s_counts[tx] : 0u;
  const auto num_blocks = div_ceil(count, block_size);
  const auto warp_inc = warp::inclusive_sum(lane_id, num_blocks);
  const auto warp_sum = __shfl_sync(warp::kFullMask, warp_inc, kWarpThreads - 1);
  // Warp w adds its total into the base of every LATER warp.
  if (lane_id > warp_id) atomicAdd(&s_warp_sums[lane_id], warp_sum);
  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

  if (tx < num_buckets) {
    const auto block_base = s_warp_sums[warp_id] + warp_inc - num_blocks;  // exclusive
    // count_and_sort uses this as a row cursor, so publish rows, not blocks.
    params.cumsum_buffer[tx] = block_base * block_size;
    // One entry per output block, value = bucket - 1 (bucket 0 -> -1).
    for (uint32_t j = 0; j < num_blocks; ++j) {
      params.expert_ids[block_base + j] = static_cast<int32_t>(tx) - 1;
    }
    if (tx == num_buckets - 1) {
      *params.total_tokens_post_pad = static_cast<int32_t>((block_base + num_blocks) * block_size);
    }
  }
}

/**
 * \brief Tiny-batch variant: one warp per pair, single CTA, no expert axis.
 *
 * Capacity is `kWarpThreads` pairs -- warp `w` owns pair `w` and its lanes hold
 * every pair, so all the bucket bookkeeping is three warp reductions over a
 * 32x32 relation. Nothing here is sized by the bucket count, which is what the
 * other paths pay: they zero and scan `kCTASize` shared counters even when a
 * handful of buckets are non-empty.
 *
 * The launcher sizes the block at `numel * kWarpThreads`, so every warp owns a
 * live pair. The fill does not suffer for it: `buffer_vecs` is proportional to
 * `numel` too, leaving `div_ceil(block_size, 4 * kWarpThreads)` iterations no
 * matter how small the batch is.
 *
 * The offset trick: `rank` counts pairs in strictly smaller buckets, so it is
 * the *unpadded* prefix. It is strictly increasing across distinct non-empty
 * buckets, hence usable as a dense key -- the representative of a bucket adds
 * its block count into every `s_cumsum[k]` with `k >= rank`. That inclusive
 * form gives a pair its own bucket's start as `s_cumsum[rank] - num_blocks`,
 * and leaves the grand total sitting in the last lane.
 *
 * Dead lanes (past `numel`, or filtered under kIgnoreInvalid) take the sentinel
 * bucket INT_MAX: they never compare smaller, so they cannot inflate any live
 * pair's `rank`, and they are excluded from every store.
 */
template <bool kIgnoreInvalid, bool kUsePDL>
ALIGN_KERNEL void moe_align_small_kernel(const __grid_constant__ MoEAlignParams params) {
  using namespace device;
  constexpr int32_t kDead = INT_MAX;
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;

  // Block counts, prefix-summed and indexed by `rank`. `rank` is bounded by the
  // pair count, itself capped at one warp, so this is sized in lanes -- not in
  // warps and not in buckets.
  __shared__ uint32_t s_cumsum[kWarpThreads];

  if (warp_id == 0) s_cumsum[tx] = 0;
  __syncthreads();

  // Ahead of the topk_ids loads: this is what makes the producer's writes visible.
  PDLWaitPrimary<kUsePDL>();

  // NOTE: 1 warp for 1 item
  const auto load_one = [&](uint32_t i) -> int32_t {
    if (i >= params.numel) return kDead;
    const auto v = params.topk_ids[i];
    if constexpr (kIgnoreInvalid) {
      return v < 0 ? kDead : v;
    }
    return v;
  };
  const auto base = load_one(lane_id);
  const auto self = load_one(warp_id);

  // Placed after those loads on purpose: nothing here consumes them, so the fill
  // covers their latency.
  if (params.pad_sorted_token_ids) {
    AlignedVector<int32_t, 4> fill_vec;
    fill_vec.fill(static_cast<int32_t>(params.numel));
    for (auto idx = tx; idx < params.buffer_vecs; idx += blockDim.x) {
      fill_vec.store(params.sorted_token_ids, idx);
    }
  }

  const auto is_live = self != kDead;
  const auto is_equal = (self == base);
  const auto is_greater = (self > base);
  const auto is_greater_in_equal = is_equal && (lane_id < warp_id);
  // Every lane joins each reduction -- dead lanes contribute 0 rather than exit.
  const auto rank = warp::reduce_sum(static_cast<uint32_t>(is_greater));
  const auto rank_in_equal = warp::reduce_sum(static_cast<uint32_t>(is_greater_in_equal));
  const auto num_equal = warp::reduce_sum(static_cast<uint32_t>(is_equal));
  const auto num_blocks = div_ceil(num_equal, params.block_size);
  const auto is_leader = is_live && rank_in_equal == 0;

  // NOTE: inclusive sum here
  if (is_leader && lane_id >= rank) {
    atomicAdd(&s_cumsum[lane_id], num_blocks);
  }

  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();
  const auto block_base = s_cumsum[rank] - num_blocks;
  if (warp_id == params.numel - 1) {
    *params.total_tokens_post_pad = s_cumsum[warp_id] * params.block_size;
  }
  if (!is_live) return;
  const auto offset = block_base * params.block_size;
  params.sorted_token_ids[offset + rank_in_equal] = warp_id;
  // One entry per block this bucket spans, not just the block it starts in.
  if (is_leader && lane_id < num_blocks) {
    params.expert_ids[block_base + lane_id] = self;
  }
}

/**
 * \brief The same scheme with two pairs per warp, reaching 2 * kWarpThreads.
 *
 * Written out rather than folded into the kernel above as a `kUnroll` template:
 * one pair per warp is the hot shape and it does not deserve the indexing that
 * a general form drags in. Three pairs would need a third slot everywhere and
 * buys nothing, so this stops at two.
 *
 * Lane `j` holds the pairs `j` and `j + kWarpThreads`; warp `w` owns the pairs
 * `w` and `w + num_warps`. The lane mapping has to span the whole capacity,
 * while the warp mapping is packed so no slot lands past `numel` unless the
 * count is odd. Each lane folds its two held pairs into one contribution first,
 * so every quantity is still one warp reduction; `s_cumsum` doubles and each
 * lane drives two of its entries.
 */
template <bool kIgnoreInvalid, bool kUsePDL>
ALIGN_KERNEL void moe_align_small_x2_kernel(const __grid_constant__ MoEAlignParams params) {
  using namespace device;
  constexpr int32_t kDead = INT_MAX;
  constexpr uint32_t kCapacity = 2 * kWarpThreads;
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto num_warps = blockDim.x / kWarpThreads;

  __shared__ uint32_t s_cumsum[kCapacity];

  if (warp_id == 0) {
    s_cumsum[lane_id] = 0;
    s_cumsum[lane_id + kWarpThreads] = 0;
  }
  __syncthreads();

  // Ahead of the topk_ids loads: this is what makes the producer's writes visible.
  PDLWaitPrimary<kUsePDL>();

  const auto load_one = [&](uint32_t i) -> int32_t {
    if (i >= params.numel) return kDead;
    const auto v = params.topk_ids[i];
    if constexpr (kIgnoreInvalid) {
      return v < 0 ? kDead : v;
    }
    return v;
  };
  const auto base_lo = load_one(lane_id);
  const auto base_hi = load_one(lane_id + kWarpThreads);
  const auto idx_lo = warp_id;
  const auto idx_hi = warp_id + num_warps;
  const auto self_lo = load_one(idx_lo);
  const auto self_hi = load_one(idx_hi);

  // Placed after those loads on purpose: nothing here consumes them, so the fill
  // covers their latency.
  if (params.pad_sorted_token_ids) {
    AlignedVector<int32_t, 4> fill_vec;
    fill_vec.fill(static_cast<int32_t>(params.numel));
    for (auto idx = tx; idx < params.buffer_vecs; idx += blockDim.x) {
      fill_vec.store(params.sorted_token_ids, idx);
    }
  }

  const auto eq_lo_lo = (self_lo == base_lo);
  const auto eq_lo_hi = (self_lo == base_hi);
  const auto eq_hi_lo = (self_hi == base_lo);
  const auto eq_hi_hi = (self_hi == base_hi);
  // `idx_lo` is a warp id, so it is below kWarpThreads and can never follow a
  // pair a lane holds in its high slot -- that term is dropped.
  const auto before_lo = static_cast<uint32_t>(eq_lo_lo && lane_id < idx_lo);
  const auto before_hi = static_cast<uint32_t>(eq_hi_lo && lane_id < idx_hi) +
                         static_cast<uint32_t>(eq_hi_hi && lane_id + kWarpThreads < idx_hi);

  // Every lane joins each reduction -- dead lanes contribute 0 rather than exit.
  const auto rank_lo =
      warp::reduce_sum(static_cast<uint32_t>(self_lo > base_lo) + static_cast<uint32_t>(self_lo > base_hi));
  const auto rank_hi =
      warp::reduce_sum(static_cast<uint32_t>(self_hi > base_lo) + static_cast<uint32_t>(self_hi > base_hi));
  const auto rank_in_equal_lo = warp::reduce_sum(before_lo);
  const auto rank_in_equal_hi = warp::reduce_sum(before_hi);
  const auto num_blocks_lo =
      div_ceil(warp::reduce_sum(static_cast<uint32_t>(eq_lo_lo) + static_cast<uint32_t>(eq_lo_hi)), params.block_size);
  const auto num_blocks_hi =
      div_ceil(warp::reduce_sum(static_cast<uint32_t>(eq_hi_lo) + static_cast<uint32_t>(eq_hi_hi)), params.block_size);

  const auto is_live_lo = self_lo != kDead;
  const auto is_live_hi = self_hi != kDead;
  const auto is_leader_lo = is_live_lo && rank_in_equal_lo == 0;
  const auto is_leader_hi = is_live_hi && rank_in_equal_hi == 0;

  // NOTE: inclusive sum here, over both of this lane's entries.
  if (is_leader_lo) {
    if (lane_id >= rank_lo) atomicAdd(&s_cumsum[lane_id], num_blocks_lo);
    if (lane_id + kWarpThreads >= rank_lo) atomicAdd(&s_cumsum[lane_id + kWarpThreads], num_blocks_lo);
  }
  if (is_leader_hi) {
    if (lane_id >= rank_hi) atomicAdd(&s_cumsum[lane_id], num_blocks_hi);
    if (lane_id + kWarpThreads >= rank_hi) atomicAdd(&s_cumsum[lane_id + kWarpThreads], num_blocks_hi);
  }

  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();

  // No leader ranks above numel - 1, so that entry is the grand total. Warps only
  // reach num_warps, which is half of it, so a single thread publishes instead.
  if (tx == 0) {
    *params.total_tokens_post_pad = s_cumsum[params.numel - 1] * params.block_size;
  }

  if (is_live_lo) {
    const auto block_base = s_cumsum[rank_lo] - num_blocks_lo;
    params.sorted_token_ids[block_base * params.block_size + rank_in_equal_lo] = idx_lo;
    // One entry per block this bucket spans, not just the block it starts in.
    if (is_leader_lo && lane_id < num_blocks_lo) {
      params.expert_ids[block_base + lane_id] = self_lo;
    }
  }
  if (is_live_hi) {
    const auto block_base = s_cumsum[rank_hi] - num_blocks_hi;
    params.sorted_token_ids[block_base * params.block_size + rank_in_equal_hi] = idx_hi;
    if (is_leader_hi && lane_id < num_blocks_hi) {
      params.expert_ids[block_base + lane_id] = self_hi;
    }
  }
}

/**
 * \brief Low-latency variant: one pair per thread, no grid-stride loop.
 *
 * Split across the grid: blocks `[0, gridDim.x - 1)` only prefill
 * `sorted_token_ids`, and the LAST block does all the alignment compute. The
 * bandwidth-heavy fill therefore overlaps the latency-critical histogram+scan,
 * and only the final scatter waits on it, through `params.event`.
 *
 * Capacity is `kCTASize * kUnroll` pairs -- there is no fallback loop, so the
 * launcher must not call this above that.
 *
 * The trick that makes the scatter free: the counting `atomicAdd` already returns
 * the pair's rank within its bucket, so keeping that in a register removes the
 * second atomic pass entirely.
 *
 * The compute block is the LAST one on purpose: it spins on the fill blocks, and
 * blocks are dispatched in increasing blockIdx order, so the spinner is the one
 * scheduled last. The launcher additionally bounds the grid to what is resident,
 * since that dispatch order is a convention rather than a guarantee.
 *
 * \tparam kUnroll Pairs per thread. Power of two; the vectorized load needs
 *                 `topk_ids` aligned to `4 * kUnroll` bytes, which the launcher
 *                 checks.
 */
template <uint32_t kUnroll, bool kIgnoreInvalid, bool kUsePDL>
ALIGN_KERNEL void moe_align_fused_kernel(const __grid_constant__ MoEAlignParams params) {
  using namespace device;
  constexpr uint32_t kNumWarps = kCTASize / kWarpThreads;
  static_assert(kNumWarps <= kWarpThreads, "per-warp sums must fit one warp of lanes");

  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto event_ptr = static_cast<atomic::Event*>(params.event);
  if (bx != gridDim.x - 1) {
    const auto global_idx = bx * kCTASize + tx;
    if (global_idx < params.buffer_vecs) {
      AlignedVector<int32_t, 4> fill_vec;
      fill_vec.fill(static_cast<int32_t>(params.numel));
      fill_vec.store(params.sorted_token_ids, global_idx);
    }
    __syncthreads();
    if (tx == 0) event_ptr->arrive();
    return PDLTriggerSecondary<kUsePDL>();
  }

  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto num_buckets = params.num_buckets;
  const auto owns_bucket = tx < num_buckets;

  // Staging costs an extra barrier and a shared round-trip, which only pays off
  // once there are enough pairs for the scatter to dominate. Measured crossover
  // is around numel = 1024 -- exactly where the launcher leaves kUnroll = 1.
  constexpr bool kDoStaging = kUnroll > 1;

  __shared__ uint32_t s_warp_sums[kNumWarps];
  __shared__ uint32_t s_counts[kCTASize];
  // Compact (unpadded) prefix when staging, padded block prefix otherwise.
  __shared__ uint32_t s_base[kCTASize];
  __shared__ int32_t s_stage[kDoStaging ? kCTASize * kUnroll : 1];

  s_counts[tx] = 0;
  if (tx < kNumWarps) s_warp_sums[tx] = 0;
  __syncthreads();

  PDLWaitPrimary<kUsePDL>();

  const auto offset = tx * kUnroll;
  AlignedVector<int32_t, kUnroll> topk_id;
  uint32_t slot_id[kUnroll];
  if (offset < params.numel) {
    topk_id.load(params.topk_ids + offset);
  }
  // One predicate for all three passes below: in range, and routed to this rank
  // when the caller asked for -1 to be dropped rather than bucketed into 0.
  const auto is_live = [&](uint32_t v) {
    if (offset + v >= params.numel) return false;
    if constexpr (kIgnoreInvalid) return topk_id[v] >= 0;
    return true;
  };
#pragma unroll
  for (uint32_t v = 0; v < kUnroll; ++v) {
    if (is_live(v)) slot_id[v] = atomicAdd(&s_counts[topk_id[v] + 1], 1);
  }
  __syncthreads();

  // One scan yields both prefixes: the padded BLOCK prefix that places a bucket
  // in the output, and the compact COUNT prefix that lays out the staging area.
  // They ride in one word -- neither field can carry into the other, since both
  // totals are bounded by numel.
  static_assert(kCTASize * kUnroll <= 0xffffu, "packed scan fields would overflow");
  const auto block_size = params.block_size;
  const auto count = owns_bucket ? s_counts[tx] : 0u;
  const auto num_blocks = div_ceil(count, block_size);
  const auto packed = (count << 16) | num_blocks;
  const auto warp_inc = warp::inclusive_sum(lane_id, packed);
  const auto warp_sum = __shfl_sync(warp::kFullMask, warp_inc, kWarpThreads - 1);
  if (lane_id > warp_id) atomicAdd(&s_warp_sums[lane_id], warp_sum);
  __syncthreads();

  const auto excl = s_warp_sums[warp_id] + warp_inc - packed;  // exclusive, packed
  const auto compact_base = excl >> 16;
  const auto block_base = excl & 0xffffu;
  if (owns_bucket) {
    s_base[tx] = kDoStaging ? compact_base : block_base;
    if (tx == num_buckets - 1) {
      *params.total_tokens_post_pad = (block_base + num_blocks) * block_size;
    }
  }

  if constexpr (kDoStaging) {
    __syncthreads();
    // Stage the pair indices in shared, grouped by bucket. A shared scatter has
    // no coalescing requirement, and it turns the global write-out below into one
    // dense run per bucket instead of one 32-byte sector per pair.
#pragma unroll
    for (uint32_t v = 0; v < kUnroll; ++v) {
      if (is_live(v)) {
        s_stage[s_base[topk_id[v] + 1] + slot_id[v]] = static_cast<int32_t>(offset + v);
      }
    }
  }

  // One thread waits on the fill blocks.
  if (tx == kCTASize - 1) {
    event_ptr->wait(gridDim.x - 1);
    PDLTriggerSecondary<kUsePDL>();  // only safe to trigger after CAS
  }
  __syncthreads();  // also publishes s_stage to the bucket owners

  if constexpr (kDoStaging) {
    // aligned write, better performance
    if (owns_bucket) {
      constexpr uint32_t kVec = 4;
      const auto row_base = block_base * block_size;
      for (uint32_t i = 0; i < count; i += kVec) {
        AlignedVector<int32_t, kVec> out;
#pragma unroll
        for (uint32_t k = 0; k < kVec; ++k) {
          out[k] = i + k < count ? s_stage[compact_base + i + k] : static_cast<int32_t>(params.numel);
        }
        out.store(params.sorted_token_ids + row_base + i);
      }
      for (uint32_t j = 0; j < num_blocks; ++j) {
        params.expert_ids[block_base + j] = static_cast<int32_t>(tx) - 1;
      }
    }
  } else {
    // normal scatter write
#pragma unroll
    for (uint32_t v = 0; v < kUnroll; ++v) {
      if (is_live(v)) {
        const auto pos = s_base[topk_id[v] + 1] * block_size + slot_id[v];
        params.sorted_token_ids[pos] = static_cast<int32_t>(offset + v);
        if (pos % block_size == 0) params.expert_ids[pos / block_size] = topk_id[v];
      }
    }
  }
}

/**
 * \brief Host launcher. Picks the small path when the pairs fit one launch.
 *
 * Argument list and semantics match `moe_align_block_size` so this is a drop-in
 * replacement for the AOT/JIT kernel, `ignore_invalid_expert` aside.
 *
 * \param topk_ids            [num_tokens, topk] int32 expert ids; -1 = EP-filtered
 * \param num_experts         BUCKET count E + 1, i.e. what the moe_runner call site
 *                            passes as `num_experts + 1`. Capped at kCTASize,
 *                            except on the tiny-batch path, which is unbounded
 *                            in buckets because it never scans that axis.
 * \param block_size          GEMM tile height every bucket is padded up to
 * \param sorted_token_ids    [max_num_tokens_padded] out
 * \param expert_ids          [max_num_m_blocks] out
 * \param num_tokens_post_pad [1] out
 * \param cumsum_buffer       [>= num_experts] scratch for the general path
 * \param pad_sorted_token_ids Whether to prefill the pad slots with `numel`
 *
 * \tparam kIgnoreInvalid Drop -1 pairs instead of bucketing them into 0. A
 *         template parameter, not an argument: as a runtime bool ptxas keeps the
 *         test inside the histogram loop instead of unswitching it (~2x on the
 *         loop body), and only the variant the caller asks for gets compiled.
 */
template <bool kIgnoreInvalid, bool kUsePDL>
void moe_align_v2(
    tvm::ffi::TensorView topk_ids,
    int64_t num_experts,
    int64_t block_size,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_pad,
    tvm::ffi::TensorView cumsum_buffer,
    bool pad_sorted_token_ids) {
  using namespace host;

  auto device_sym = SymbolicDevice{};
  device_sym.set_options<kDLCUDA>();

  auto N = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"topk"};
  TensorMatcher({N, K})  //
      .with_dtype<int32_t>()
      .with_device(device_sym)
      .verify(topk_ids);

  auto P = SymbolicSize{"max_num_tokens_padded"};
  TensorMatcher({P})  //
      .with_dtype<int32_t>()
      .with_device(device_sym)
      .verify(sorted_token_ids);

  auto B = SymbolicSize{"max_num_m_blocks"};
  TensorMatcher({B})  //
      .with_dtype<int32_t>()
      .with_device(device_sym)
      .verify(expert_ids);

  TensorMatcher({1})  //
      .with_dtype<int32_t>()
      .with_device(device_sym)
      .verify(num_tokens_post_pad);

  auto C = SymbolicSize{"cumsum_size"};
  TensorMatcher({C})  //
      .with_dtype<int32_t>()
      .with_device(device_sym)
      .verify(cumsum_buffer);

  const int64_t numel = N.unwrap() * K.unwrap();
  const int64_t num_buckets = num_experts;
  CHECK_HOST(block_size > 0) << "block_size must be positive, got " << block_size;
  // Both paths prefill with 4-wide vector stores, and the real total is a multiple of block_size.
  CHECK_HOST(block_size % 4 == 0) << "block_size must be a multiple of 4, got " << block_size;
  // Pair indices and the pad sentinel both land in an int32 buffer.
  CHECK_HOST(numel <= std::numeric_limits<int32_t>::max())
      << "topk_ids has " << numel << " elements, which does not fit int32";
  // The small path launches numel * kWarpThreads threads, and no path writes
  // total_tokens_post_pad without a live pair to write it.
  CHECK_HOST(numel > 0) << "topk_ids must not be empty";
  // Worst case: every non-empty bucket pads out to a full extra block. This is
  // exactly the bound the moe_runner call site sizes its buffers to.
  const int64_t worst_total = numel + std::min<int64_t>(numel, num_buckets) * (block_size - 1);
  CHECK_HOST(P.unwrap() >= worst_total) << "sorted_token_ids holds " << P.unwrap() << " rows, needs " << worst_total;
  CHECK_HOST(B.unwrap() >= worst_total / block_size)
      << "expert_ids holds " << B.unwrap() << " entries, needs " << worst_total / block_size;

  const auto device = device_sym.unwrap();
  const auto stream = LaunchKernel::resolve_device(device);

  // The general path uses the caller's cumsum_buffer; only the small path's Event
  // is kernel-owned, and it self-resets so it is zeroed exactly once.
  const auto& event = allocate_once(stream, [&] {
    const auto dl_int32 = DLDataType{.code = kDLInt, .bits = 32, .lanes = 1};
    auto tensor = ffi::empty({1}, dl_int32, device);
    CHECK_CUDA(cudaMemsetAsync(tensor.data_ptr(), 0, sizeof(int32_t), stream));
    return tensor;
  });

  // Floor, not ceil: `worst_total` is not necessarily a multiple of 4 (e.g.
  // numel=1024, E+1=897, block_size=128 gives 114943), and rounding up would put
  // the last vector store one element past a buffer sized to exactly that bound.
  // Flooring still covers everything, because the real total is a multiple of
  // block_size and so of 4.
  const auto buffer_vecs = static_cast<uint32_t>(worst_total / 4);
  const auto params = MoEAlignParams{
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(sorted_token_ids.data_ptr()),
      static_cast<int32_t*>(expert_ids.data_ptr()),
      static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
      static_cast<uint32_t*>(cumsum_buffer.data_ptr()),
      event.data_ptr(),
      static_cast<uint32_t>(numel),
      static_cast<uint32_t>(block_size),
      buffer_vecs,
      static_cast<uint32_t>(num_buckets),
      pad_sorted_token_ids,
  };

  // Tiny batches first: these kernels never touch the bucket axis and load
  // topk_ids one element at a time, so none of the constraints below apply to
  // them -- any bucket count, any alignment, no cumsum_buffer. One pair per warp
  // reaches a warp of pairs, two pairs per warp reach twice that.
  if (numel <= device::kWarpThreads) {
    return LaunchKernel(1, numel * device::kWarpThreads, stream)  //
        .enable_pdl(kUsePDL)(moe_align_small_kernel<kIgnoreInvalid, kUsePDL>, params);
  }
  if (numel <= 2 * device::kWarpThreads) {
    return LaunchKernel(1, div_ceil(numel, 2u) * device::kWarpThreads, stream)  //
        .enable_pdl(kUsePDL)(moe_align_small_x2_kernel<kIgnoreInvalid, kUsePDL>, params);
  }

  // Everything past here scans the bucket axis and loads topk_ids as int4.
  // One thread per bucket in the scan, and s_counts is kCTASize wide.
  CHECK_HOST(num_buckets <= kCTASize) << "num_experts (bucket count) must be <= " << kCTASize << ", got "
                                      << num_buckets;
  CHECK_HOST(C.unwrap() >= num_buckets) << "cumsum_buffer holds " << C.unwrap() << " entries, needs " << num_buckets;
  const bool aligned = reinterpret_cast<uintptr_t>(topk_ids.data_ptr()) % 16 == 0;
  CHECK_HOST(aligned) << "topk_ids must be aligned to 16 bytes for the vectorized load, but got: "
                      << topk_ids.data_ptr();

  static const uint32_t sm_count = runtime::get_sm_count(device.device_id);
  const uint32_t small_grid = div_ceil(buffer_vecs, kCTASize) + 1;
  const uint32_t pairs_per_thread = div_ceil(numel, kCTASize);

  constexpr uint32_t kMaxUnroll = 4;
  if (pairs_per_thread <= kMaxUnroll && small_grid <= sm_count) {
    const auto kernel = (pairs_per_thread <= 1) ? moe_align_fused_kernel<1, kIgnoreInvalid, kUsePDL>
                        : pairs_per_thread <= 2 ? moe_align_fused_kernel<2, kIgnoreInvalid, kUsePDL>
                                                : moe_align_fused_kernel<4, kIgnoreInvalid, kUsePDL>;
    return LaunchKernel(small_grid, kCTASize, stream)  //
        .enable_pdl(kUsePDL)(kernel, params);
  }

  // General path. Block 0 computes, the rest prefill; the fill is a grid-stride
  // loop so any grid works, capped at what the machine can run at once.
  const uint32_t fill_blocks = std::clamp<uint32_t>(div_ceil(buffer_vecs, 2 * kCTASize), 1, sm_count - 1);
  LaunchKernel(1 + fill_blocks, kCTASize, stream)  //
      .enable_pdl(kUsePDL)(moe_align_block_size_kernel<kIgnoreInvalid, kUsePDL>, params);
  const uint32_t sort_blocks = div_ceil(numel, kCTASize);
  LaunchKernel(sort_blocks, kCTASize, stream)  //
      .enable_pdl(kUsePDL)(count_and_sort_expert_tokens_kernel<kIgnoreInvalid, kUsePDL>, params);
}

}  // namespace sglang
