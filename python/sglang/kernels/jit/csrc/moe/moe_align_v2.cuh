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
 *                            passes as `num_experts + 1`. Capped at kCTASize.
 * \param block_size          GEMM tile height every bucket is padded up to
 * \param sorted_token_ids    [max_num_tokens_padded] out
 * \param expert_ids          [max_num_m_blocks] out
 * \param num_tokens_post_pad [1] out
 * \param cumsum_buffer       [>= num_experts] scratch for the general path
 * \param pad_sorted_token_ids Whether to prefill the pad slots with `numel`
 * \param ignore_invalid_expert Drop -1 pairs instead of bucketing them into 0
 */
template <bool kUsePDL>
void moe_align_v2(
    tvm::ffi::TensorView topk_ids,
    int64_t num_experts,
    int64_t block_size,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_pad,
    tvm::ffi::TensorView cumsum_buffer,
    bool pad_sorted_token_ids,
    bool ignore_invalid_expert) {
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
  // One thread per bucket in the scan, and s_counts is kCTASize wide.
  CHECK_HOST(num_buckets <= kCTASize) << "num_experts (bucket count) must be <= " << kCTASize << ", got "
                                      << num_buckets;
  CHECK_HOST(C.unwrap() >= num_buckets) << "cumsum_buffer holds " << C.unwrap() << " entries, needs " << num_buckets;
  // Pair indices and the pad sentinel both land in an int32 buffer.
  CHECK_HOST(numel <= std::numeric_limits<int32_t>::max())
      << "topk_ids has " << numel << " elements, which does not fit int32";
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

  static const uint32_t sm_count = runtime::get_sm_count(device.device_id);
  const uint32_t small_grid = div_ceil(buffer_vecs, kCTASize) + 1;
  const uint32_t pairs_per_thread = div_ceil(numel, kCTASize);
  const bool aligned = reinterpret_cast<uintptr_t>(topk_ids.data_ptr()) % 16 == 0;
  CHECK_HOST(aligned) << "topk_ids must be aligned to 16 bytes for the vectorized load, but got: "
                      << topk_ids.data_ptr();

  // The small path needs three things, and any miss falls back to the general
  // two-launch path:
  //  1. `kCTASize * kUnroll` capacity, since it has no loop over the pairs;
  //  2. its fill blocks resident alongside the compute block, which spins on them.
  //    `__launch_bounds__(kCTASize, 1)` guarantees one block per SM, so the SM
  //    count is a safe (conservative) bound;
  //  3. `topk_ids` aligned for the vectorized load, which a sliced tensor is not.
  constexpr uint32_t kMaxUnroll = 4;
  // `ignore_invalid_expert` is a compile-time constant in the kernels: as a
  // runtime bool ptxas keeps the test inside the histogram loop instead of
  // unswitching it, which measured ~2x on the loop body.
  const auto launch = [&](auto invalid_tag) {
    constexpr bool kIgnoreInvalid = decltype(invalid_tag)::value;
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
  };

  if (ignore_invalid_expert) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

}  // namespace sglang
