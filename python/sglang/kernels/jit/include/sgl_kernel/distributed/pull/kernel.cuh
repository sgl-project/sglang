#pragma once

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/fusion.cuh>
#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/distributed/pull/epilogue/common.cuh>
#include <sgl_kernel/distributed/pull/row_sum.cuh>

#include <cstdint>

/// The fused 2-shot NVLS pull all-reduce: every rank `multimem.ld_reduce`s the rows of its shard from the partials'
/// multicast VA (the reduce-scatter happens in the switch), runs the epilogue on them, and multicasts what every rank
/// needs (the all-gather). A caller instantiates it with its epilogue (see `epilogue/common.cuh`).

namespace sglang::device::distributed::pull {

template <Epilogue Epi>
struct PullParams {
  typename Epi::Params epilogue;
  const uint8_t* partial_mc;  // [num_tokens, hidden] bf16, the multicast VA of every rank's partial
  Semaphore* sem_local;       // this rank's pull-plane semaphores
  Semaphore* sem_mc;          // their multicast VA
  uint32_t world_size;
  uint32_t row0;  // first row of this rank's shard
  uint32_t rows;  // rows of this rank's shard (may be 0: the CTAs still meet the barriers)
};

/// The ld_reduce of a group of `kNumRows` rows, issued before any of them is used, then the epilogue.
template <uint32_t kNumRows, Epilogue Epi, typename Smem>
SGL_DEVICE void pull_rows(const Epi& epi, const PullParams<Epi>& p, Smem& smem, RowGroup group, uint32_t phase) {
  using Tile = typename Epi::Tile;
  VecStorage<bf16_t> x[kNumRows];
#pragma unroll
  for (uint32_t u = 0; u < kNumRows; ++u) {
    ptx::ld_multimem_16B(x[u], p.partial_mc, group.template vec<Tile::kRowVecs>(u));
  }
  const auto inputs = epi.template prefetch<kNumRows>(group);
  epi.template store<kNumRows>(group, x, inputs, smem, phase);
}

// Grid: any number of CTAs, the same on every rank (CTA `b` meets CTA `b` of every peer at the barriers). CTA `b`
// takes groups of `kRowsPerGroup` rows of the shard, round-robin; the full groups run without a predicate, the last
// partial group row by row.
template <uint32_t kRowsPerGroup, bool kUsePDL, Epilogue Epi>
__global__ __launch_bounds__(Epi::Tile::kNumThreads, 1)  //
    void fused_all_reduce_pull(const __grid_constant__ PullParams<Epi> p) {
  __shared__ GroupRowSum<kRowsPerGroup, Epi::Tile::kNumWarps, Epi::kNumSums> smem;
  const auto barrier = McBarrier{p.sem_local, p.sem_mc, p.world_size};

  // the arrival follows the PDL wait: it asserts that this rank's partial is written
  PDLWaitPrimary<kUsePDL>();
  barrier.arrive_relaxed();
  __syncthreads();
  const auto epi = Epi{p.epilogue};

  uint32_t phase = 0;
  uint32_t local = blockIdx.x * kRowsPerGroup;
  for (; local + kRowsPerGroup <= p.rows; local += gridDim.x * kRowsPerGroup, ++phase) {
    pull_rows<kRowsPerGroup>(epi, p, smem, RowGroup{p.row0 + local, local}, phase);
  }
  if constexpr (kRowsPerGroup > 1) {  // the partial group (at most one CTA has one)
    const uint32_t end = local + kRowsPerGroup < p.rows ? local + kRowsPerGroup : p.rows;
    for (; local < end; ++local, ++phase) {
      pull_rows<1>(epi, p, smem, RowGroup{p.row0 + local, local}, phase);
    }
  }

  // every peer has read this rank's partial, and every multicast store has landed, before the next kernel
  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq();
}

}  // namespace sglang::device::distributed::pull
