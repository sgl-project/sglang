#pragma once

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/push/epilogue/common.cuh>
#include <sgl_kernel/distributed/push/port.cuh>
#include <sgl_kernel/distributed/push/prologue/common.cuh>

#include <cstdint>

/// The fused all-reduce kernels, one row per cluster: 1-shot push and 2-shot push. A caller instantiates them with
/// its prologue and epilogue (see `prologue/common.cuh` and `epilogue/common.cuh`).

namespace sglang::device::distributed::push {

// One row of kHiddenSize bf16 over a cluster of kClusterSize CTAs, one 16-byte vector per thread.
template <uint32_t kHiddenSize, uint32_t kClusterSize>
struct RowTile {
  static constexpr uint32_t kRowVecs = kHiddenSize / 8;
  static constexpr uint32_t kNumThreads = kRowVecs / kClusterSize;
  static constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;
  static_assert(kHiddenSize % 8 == 0 && kRowVecs % kClusterSize == 0, "the cluster must divide the row's vectors");
  static_assert(kNumThreads % kWarpThreads == 0 && kNumThreads <= 1024, "whole warps, at most 1024 threads");
  static_assert(1 <= kClusterSize && kClusterSize <= 8, "portable cluster sizes only");
};

template <uint32_t kWorldSize, Prologue PrologueT, Epilogue EpilogueT>
struct PushParams {
  typename PrologueT::Params prologue;
  typename EpilogueT::Params epilogue;
  PushWorkSpace<kWorldSize> ws;
  uint32_t rank;
};

#define AR_PUSH                                   \
  __global__ __cluster_dims__(1, kClusterSize, 1) \
      __launch_bounds__(RowTile<kHiddenSize, kClusterSize>::kNumThreads, kOccupancy)

// Grid `(rows, kClusterSize)`, cluster `(1, kClusterSize)`: `blockIdx.x` is the row, `blockIdx.y` the CTA's rank in
// the row's cluster.
template <
    uint32_t kWorldSize,
    uint32_t kHiddenSize,
    uint32_t kClusterSize,
    bool kUsePDL,
    bool kTriggerAfterPush,
    bool kMulticast,
    uint32_t kOccupancy,
    Prologue PrologueT,
    Epilogue EpilogueT>
AR_PUSH void fused_all_reduce_1shot(const __grid_constant__ PushParams<kWorldSize, PrologueT, EpilogueT> p) {
  using Tile = RowTile<kHiddenSize, kClusterSize>;
  __shared__ typename EpilogueT::template Smem<kClusterSize, Tile::kNumWarps> smem;
  smem.init();

  const auto row = blockIdx.x;
  const auto col = blockIdx.y * Tile::kNumThreads + threadIdx.x;
  const auto vid = row * Tile::kRowVecs + col;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = PushEpoch<kWorldSize>{p.ws};

  const auto prologue = PrologueT{p.prologue, row, col, vid};
  if constexpr (kMulticast) {
    const auto port = MulticastPushPort<bf16_t>{p.ws.mc_workspace, epoch.slot_offset(p.rank)};
    port.store(vid, prologue.load());
  } else {
    const auto port = PushPort<bf16_t, kWorldSize>{p.ws.workspaces.data(), epoch.slot_offset(p.rank)};
    port.store(vid, prologue.load());
  }

  if constexpr (kTriggerAfterPush) PDLTriggerSecondary<kUsePDL>();
  const auto epilogue = EpilogueT{p.epilogue, row, col, vid};

  // poll: every rank's partial, summed in rank order and rounded to bf16
  const auto poll = PollPort<bf16_t, kWorldSize>{p.ws.workspaces[p.rank] + epoch.slot_offset(), p.ws.slot_bytes};
  const auto x = poll.wait_reduce(vid);

  epilogue.store(x, smem);

  if constexpr (!kTriggerAfterPush) PDLTriggerSecondary<kUsePDL>();

  // restore the empty markers; every thread has read the epoch before the CTA counts itself in
  poll.reset(vid);
  __syncthreads();
  epoch.commit(blockIdx.x * kClusterSize + blockIdx.y, gridDim.x * kClusterSize);
}

// The 2-shot kernel: grid and parameters as the 1-shot kernel's, the dependent grid released after the epilogue.
// Its rows exceed one wave, so it keeps the default launch bounds.
template <
    uint32_t kWorldSize,
    uint32_t kHiddenSize,
    uint32_t kClusterSize,
    bool kUsePDL,
    bool kTriggerAfterPush,
    bool kMulticast,
    uint32_t kOccupancy,
    Prologue PrologueT,
    Epilogue EpilogueT>
AR_PUSH void fused_all_reduce_2shot(const __grid_constant__ PushParams<kWorldSize, PrologueT, EpilogueT> p) {
  using Tile = RowTile<kHiddenSize, kClusterSize>;
  __shared__ typename EpilogueT::template Smem<kClusterSize, Tile::kNumWarps> smem;
  smem.init();

  const auto row = blockIdx.x;
  const auto col = blockIdx.y * Tile::kNumThreads + threadIdx.x;
  const auto vid = row * Tile::kRowVecs + col;
  const auto owner = row % kWorldSize;
  const bool is_owner = owner == p.rank;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = PushEpoch<kWorldSize>{p.ws};

  const auto port = PushPort<bf16_t, kWorldSize>{p.ws.workspaces.data(), epoch.slot_offset(p.rank)};
  const auto partial = PrologueT{p.prologue, row, col, vid}.load();
  port.store(owner, vid, partial);
  const auto epilogue = EpilogueT{p.epilogue, row, col, vid};

  // owner: every rank's partial, summed in rank order and rounded to bf16, returned to the other ranks with the -0
  // remap they receive, so every rank runs the epilogue on the same bits; others: the owner's sum
  const auto poll = PollPort<bf16_t, kWorldSize>{p.ws.workspaces[p.rank] + epoch.slot_offset(), p.ws.slot_bytes};
  VecStorage<bf16_t> x;
  if (is_owner) {
    x = poll.wait_reduce(vid);
    VecLamport<bf16_t>::clear_pos_zero(x.data());
#pragma unroll
    for (uint32_t i = 1; i < kWorldSize; ++i) {  // rank-rotated order: measured faster than 0..W-1 on GB300 at 4 ranks
      port.store((p.rank + i) % kWorldSize, vid, x);
    }
  } else {
    x = poll.wait(owner, vid);
  }

  epilogue.store(x, smem);
  PDLTriggerSecondary<kUsePDL>();

  // restore the empty markers; every thread has read the epoch before the CTA counts itself in
  if (is_owner) {
    poll.reset(vid);
  } else {
    poll.reset(owner, vid);
  }
  __syncthreads();
  epoch.commit(blockIdx.x * kClusterSize + blockIdx.y, gridDim.x * kClusterSize);
}

#undef AR_PUSH

}  // namespace sglang::device::distributed::push
