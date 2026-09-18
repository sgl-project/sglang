/**
 * \file topk_v2.cuh
 * \brief TopK kernel for DeepSeek v4.
 * Adapted from
 * 1:
 *   https://github.com/vllm-project/vllm/blob/a8c6ee9b787d273916206a29b77feebadb80c368/csrc/persistent_topk.cuh
 * 2:
 *   https://github.com/flashinfer-ai/flashinfer/blob/c2b4db2b1a84448d802f0e6ac445243312bd6a4c/include/flashinfer/topk.cuh
 * DarkSharpness never took a detailed look at these 2 implementation, but his claude code did.
 * So we add credit to the reference implementations.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <utility>

namespace sglang {

namespace impl = device::topk;
using impl::TopKProblem;

enum class TopKMode {
  INDICES,      ///< raw selected indices into `out`; `page_table` unused
  PAGE_TABLE,   ///< page-table-transformed indices into `out`
  DUAL_OUTPUT,  ///< page-table-transformed indices into `out` and raw indices into `raw_indices`
};

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)

/// Metadata tensor rows (each 8 B / 2 int32). Row 0 is the global plan result;
/// rows 1..N are the (batch_id, seq_len) of items routed to the cluster pool.
struct alignas(8) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;  // N = number of items routed to the cluster pool
};
struct alignas(8) PlanItem {
  uint32_t batch_id;
  uint32_t seq_len;
};
static_assert(sizeof(GlobalMetadata) == 2 * sizeof(int32_t) && sizeof(PlanItem) == sizeof(GlobalMetadata));

struct PageTransform {
  const int32_t* __restrict__ page_table;
  uint32_t page_bits;
  int32_t* __restrict__ raw_out;  // the row's raw output, written in DUAL_OUTPUT only

  SGL_DEVICE int32_t page_to_indices(uint32_t i) const {
    const uint32_t mask = (1u << page_bits) - 1u;
    return (page_table[i >> page_bits] << page_bits) | (i & mask);
  }
};

struct TopKPagedParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;      // DUAL_OUTPUT only, nullptr otherwise
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;
  uint32_t static_cluster_floor;  // only used in small batch variant
  uint32_t batch_size;

  SGL_DEVICE const GlobalMetadata& global() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE uint32_t cluster_threshold() const {
    return global().cluster_threshold;
  }
  SGL_DEVICE const PlanItem& item(uint32_t i) const {
    return metadata[1 + i];
  }
  SGL_DEVICE int32_t* get_output_ptr(uint32_t batch_id) const {
    return page_indices + batch_id * static_cast<int64_t>(topk);
  }
  SGL_DEVICE PageTransform get_transform(uint32_t batch_id) const {
    return {
        page_table == nullptr ? nullptr : page_table + batch_id * page_table_stride,
        page_bits,
        raw_indices == nullptr ? nullptr : raw_indices + batch_id * static_cast<int64_t>(topk)};
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        .in = scores + batch_id * score_stride,
        .out = page_indices + batch_id * k,
        .topk = topk,
        .seq_len = seq_len,
    };
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    return this->problem(batch_id, static_cast<uint32_t>(seq_lens[batch_id]));
  }
};

struct TopKRaggedParams {
  float* __restrict__ scores;  // NOTE: may write
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ row_starts;
  const int32_t* __restrict__ out_offsets;
  int32_t* __restrict__ topk_indices;
  int64_t score_stride;
  uint32_t topk;
};

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  static_assert(kMaxTopK % kBlockSize == 0);
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      __builtin_assume(tx < kMaxTopK);
      f(tx, i);
    }
  }
}

template <bool kPDL, TopKMode kMode>
SGL_DEVICE void trivial_transform(const TopKProblem& problem, const PageTransform& transform) {
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t) {
    if constexpr (kMode == TopKMode::INDICES) {
      problem.out[tx] = tx < problem.seq_len ? static_cast<int32_t>(tx) : -1;
    } else {
      problem.out[tx] = tx < problem.seq_len ? transform.page_to_indices(tx) : -1;
      if constexpr (kMode == TopKMode::DUAL_OUTPUT) {
        transform.raw_out[tx] = tx < problem.seq_len ? static_cast<int32_t>(tx) : -1;
      }
    }
  });
}

template <TopKMode kMode>
SGL_DEVICE void paged_transform(const TopKProblem& problem, int32_t* out, const PageTransform& transform) {
  static_assert(kMode != TopKMode::INDICES, "paged_transform requires page-table output");
  static_assert(kMaxTopK % kBlockSize == 0);
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
  int32_t indices[kNumElems];
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) {
    // load into register at once
    indices[i] = problem.out[tx];
  });
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) {
    // safe write to output
    out[tx] = indices[i] >= 0 ? transform.page_to_indices(indices[i]) : -1;
    if constexpr (kMode == TopKMode::DUAL_OUTPUT) transform.raw_out[tx] = indices[i];
  });
}

/**
 * \brief Ragged (prefill) top-k: select inside a per-row window, emit indices
 * rebased onto the flattened KV.
 *
 * Row `b` selects the top-k of `scores[b][ks : ks + seq_lens[b]]` (`ks =
 * row_starts[b]`) and writes `selected_position + out_offsets[b]`, `-1` padded.
 * No page table and no plan: the DeepGEMM contiguous-KV indexer emits columns
 * that are already absolute positions in the batch's flattened KV, so an add is
 * the whole transform. One block per row -- prefill has thousands of rows, so
 * the cluster path (which exists to split ONE row across blocks) is never worth
 * it here.
 *
 * The window start is an arbitrary token offset, so the 16-byte vectorized load
 * needs the row pointer rounded down to a 4-float boundary. The <= 3 elements
 * that pulls in are columns of a preceding request -- real finite scores that
 * would otherwise win the selection -- so they are masked in place first. That
 * write races with nothing and needs no barrier of its own:
 *   - one block owns the row, and a column of row `b` is read by no other row;
 *   - the score buffer is dead once the top-k has run;
 *   - every forward() below opens with its smem init and a `__syncthreads()`
 *     before it reads any score. That barrier both publishes the mask to
 *     whichever thread loads the head vector and keeps the compiler from
 *     hoisting those loads above the store -- store and loads reach the same row
 *     through two `__restrict__` pointers, which otherwise licenses exactly that
 *     reordering.
 * It must however land after the PDL wait, or the indexer overwrites it.
 */
template <bool kPDL>
TOPK_KERNEL void topk_ragged_kernel(const __grid_constant__ TopKRaggedParams params) {
  device::enable_smem_spilling();
  constexpr uint32_t kVecSize = impl::TopKStreaming::kVecSize;
  const auto bx = blockIdx.x;
  // issue all metadata prefetch ahead of time
  const auto seq_len = static_cast<uint32_t>(params.seq_lens[bx]);
  const auto offset = params.out_offsets[bx];
  const auto row_start = params.row_starts == nullptr ? 0u : params.row_starts[bx];
  const auto topk = params.topk;
  const auto out = params.topk_indices + bx * static_cast<int64_t>(topk);

  if (seq_len <= topk) {
    device::PDLWaitPrimary<kPDL>();
    for_each_item(topk, [&](uint32_t tx, uint32_t) {
      out[tx] = tx < seq_len ? static_cast<int32_t>(tx) + offset : -1;  // note: need offset
    });
    return;
  }

  const auto rem = row_start % kVecSize;
  const auto score = params.scores + bx * params.score_stride;
  if (rem != 0) {
    // The mask has to land after the indexer has retired
    // Otherwise it may be accidentally overwritten by DG upstream
    device::PDLWaitPrimary<kPDL>();
    static_assert(kVecSize <= kBlockSize, "not enough threads ");
    if (const auto tx = threadIdx.x; tx < rem) {
      score[row_start - rem + tx] = impl::padding_value();
    }
  }
  using device::topk::broadcast;
  const auto problem = TopKProblem{
      .in = score + (row_start - rem),
      .out = out,
      .topk = topk,
      .seq_len = seq_len + rem,
      .bias = broadcast(offset - static_cast<int32_t>(rem)),
      .input_start = broadcast(rem),
  };
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= Register2::kMaxSeqLen) {
    Register2::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= Register4::kMaxSeqLen) {
    Register4::forward<kPDL>(problem, &smem);
  } else {
    Streaming::forward<kPDL>(problem, &smem);
  }
  // PDL trigger secondary at the end the block typically has no use, so ignore it
}

/**
 * \brief Main kernel for the short items and epilogue of long items.
 * \tparam kPDL whether to use PDL to synchronize with the cluster kernel (if any)
 * \tparam kLevel:
 * - Level 0: max_seq_len <= 8192           -> trivial + register<2>
 * - Level 1: max_seq_len <= 16384          -> trivial + register<4>
 * - Level 2: max_seq_len <= cluster_floor  -> trivial + register<4> + streaming
 * - Level 3: max_seq_len > cluster_floor   -> + epilogue process of cluster path
 */
template <bool kPDL, int kLevel, TopKMode kMode>
TOPK_KERNEL void topk_main_kernel(const __grid_constant__ TopKPagedParams params) {
  device::enable_smem_spilling();
  constexpr bool kNeedStaging = kMode != TopKMode::INDICES;
  constexpr bool kHandleCluster = (kLevel == 3);
  // Only the cluster path consumes the cluster kernel's output, so only it waits
  // on that kernel (kPDLFinal). Every other path waits at most on the indexer
  // (kPDLEarly) and must not be held on an SM slot until the long-running
  // persistent pool retires -- that would serialize the short items behind it.
  constexpr bool kPDLEarly = kPDL && !kHandleCluster;
  constexpr bool kPDLFinal = kPDL && kHandleCluster;
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  __shared__ int32_t s_topk_indices[kMaxTopK];

  const auto bx = blockIdx.x;
  auto problem = params.problem(bx);
  if (problem.seq_len <= problem.topk) {
    return trivial_transform<kPDLEarly, kMode>(problem, params.get_transform(bx));
  }
  if constexpr (kNeedStaging) {
    problem.out = s_topk_indices;  // write into stage buffer in smem first
  }

  // non-trivial path: dispatch based on level and seq_len
  if constexpr (kLevel == 0) {
    __builtin_assume(problem.seq_len <= kReg2MaxSeqLen);
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    __builtin_assume(problem.seq_len <= kReg4MaxSeqLen);
    Register4::forward<kPDL>(problem, &smem);  // max_seq_len <= 16384 guarantees seq <= 16384
  } else {
    const auto cluster_threshold = kHandleCluster ? params.cluster_threshold() : UINT_MAX;
    static_assert(kLevel == 2 || kLevel == 3, "we only support level = 0,1,2,3 now");
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= cluster_threshold) {
      Streaming::forward<kPDLEarly>(problem, &smem);
    } else [[unlikely]] {
      // Cluster path: the pool already selected into our output row; the only
      // work left is the epilogue, so this is the one path that waits for it.
      if constexpr (kNeedStaging) {
        device::PDLWaitPrimary<kPDLFinal>();
        problem.out = params.get_output_ptr(bx);  // in-place transform
        device::PDLTriggerSecondary<kPDL>();
        return paged_transform<kMode>(problem, problem.out, params.get_transform(bx));
      } else {
        return device::PDLTriggerSecondary<kPDL>();
      }
    }
  }

  device::PDLTriggerSecondary<kPDL>();
  if constexpr (kNeedStaging) {
    __syncthreads();
    paged_transform<kMode>(problem, params.get_output_ptr(bx), params.get_transform(bx));
  }
}

#if SUPPORT_CLUSTER

#ifndef SGL_TOPK_V2_MAX_C8_OCC2
#if SGL_ARCH_BLACKWELL_OR_GREATER
#define SGL_TOPK_V2_MAX_C8_OCC2 33  // NOTE: B200
#else
#define SGL_TOPK_V2_MAX_C8_OCC2 30  // NOTE: H200
#endif
#endif

#ifndef SGL_TOPK_V2_MAX_C16_OCC1
#define SGL_TOPK_V2_MAX_C16_OCC1 7
#endif

constexpr uint32_t kNumPersistentClusters = SGL_TOPK_V2_MAX_C8_OCC2;
constexpr uint32_t kMaxCluster16BatchSize = SGL_TOPK_V2_MAX_C16_OCC1;
constexpr uint32_t kClusterMaxBatch = 512;
#define CLUSTER_TOPK_KERNEL TOPK_KERNEL __cluster_dims__(1, kClusterSize, 1)

/// Persistent cluster kernel for the items the plan routed to the pool; topk_main_kernel handles the rest.
template <bool kPDL, uint32_t kClusterSize>
CLUSTER_TOPK_KERNEL void topk_persistent_cluster_kernel(const __grid_constant__ TopKPagedParams params) {
  device::enable_smem_spilling();
  using ClusterN = impl::TopKCluster<kClusterSize>;
  __shared__ impl::MaxSmem<typename ClusterN::Smem> smem;
  const auto bx = blockIdx.x;
  const auto num_cluster_items = params.global().num_cluster_items;
  device::PDLWaitPrimary<kPDL>();
  if (bx >= params.batch_size) return;
  device::PDLTriggerSecondary<kPDL>();
  auto idx = static_cast<int32_t>(num_cluster_items - 1 - bx);
#pragma unroll 1
  while (idx >= 0) {
    const auto it = params.item(idx);
    const auto problem = params.problem(it.batch_id, it.seq_len);
    ClusterN::template forward<false>(problem, &smem);
    idx -= kNumPersistentClusters;
    if (idx >= 0) __syncthreads();
  }
}

template <bool kPDL, TopKMode kMode, uint32_t kClusterSize, uint32_t kOccupancy>
CLUSTER_TOPK_KERNEL void topk_small_batch_cluster_kernel(const __grid_constant__ TopKPagedParams params) {
  device::enable_smem_spilling();
  constexpr bool kNeedStaging = kMode != TopKMode::INDICES;
  const auto bx = blockIdx.x;
  const auto by = blockIdx.y;
  auto problem = params.problem(bx);
  __shared__ int32_t s_topk_indices[kMaxTopK];
  using ClusterN = impl::TopKCluster<kClusterSize>;
  __shared__ impl::MaxSmem<Register4::Smem, Streaming::Smem, typename ClusterN::Smem> smem;

  // randomly elect one worker rank to avoid workload imbalance
  const auto worker_rank = bx % kClusterSize;
  if (problem.seq_len <= problem.topk) {
    if (by != worker_rank) return;
    return trivial_transform<kPDL, kMode>(problem, params.get_transform(bx));
  }

  if constexpr (kNeedStaging) {
    problem.out = s_topk_indices;  // write into stage buffer in smem first
  }
  // for small batch, we will fuse in the cluster case
  if (problem.seq_len <= kReg4MaxSeqLen) {
    if (by != worker_rank) return;
    Register4::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= params.static_cluster_floor) {
    if (by != worker_rank) return;
    Streaming::forward<kPDL>(problem, &smem);
  } else {
    auto cluster = cooperative_groups::this_cluster();
    if constexpr (kNeedStaging) {
      problem.out = cluster.map_shared_rank(s_topk_indices, 0);
    }
    ClusterN::forward<kPDL>(problem, &smem);
    if constexpr (kNeedStaging) {
      device::PDLTriggerSecondary<kPDL>();
      cluster.sync();
      if (by != 0) return;
      problem.out = s_topk_indices;
      return paged_transform<kMode>(problem, params.get_output_ptr(bx), params.get_transform(bx));
    } else {
      return device::PDLTriggerSecondary<kPDL>();
    }
  }

  device::PDLTriggerSecondary<kPDL>();
  if constexpr (kNeedStaging) {
    __syncthreads();
    paged_transform<kMode>(problem, params.get_output_ptr(bx), params.get_transform(bx));
  }
}

// --- Plan: choose cluster_threshold from the seq_len distribution -----------
__global__ __launch_bounds__(kBlockSize, 1) void topk_plan_cluster(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,  // [0]=GlobalMetadata, [1+i]=PlanItem
    const uint32_t batch_size,
    const int32_t static_cluster_threshold) {
  // Candidate (threshold T_j, cap_j) pairs, T strictly increasing. The plan lowers
  // cluster_threshold to T_j while #(items with seq_len > T_j) <= cap_j, so cap_j
  // bounds how many long items go to the persistent pool. The pool runs N items in
  // ceil(N / kNumPersistentClusters) waves; the longer the seq the more waves pay
  // off (streaming a single block over a long item is very slow), so cap_j is the
  // measured cluster-vs-streaming crossover (B200, occ2) and GROWS with T -- a flat
  // cap = pool size only fits the shortest (~98K, one-wave) bucket. (Plan is tunable.)
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
#if SGL_ARCH_BLACKWELL_OR_GREATER  // tuned on B200
      {32768, 48},
      {131072, 66},
      {163840, 99},
      {196608, 132},
      {262144, 198},
      {393216, 231},
      {524288, 264},
#else  // tuned on H200
      {65536, 30},
      {98304, 45},
      {131072, 60},
      {196608, 80},
      {262144, 112},
      {393216, 128},
#endif
  };
  constexpr uint32_t kNumCandidates = std::size(kCandidates);

  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;
  __shared__ uint32_t s_count;

  const auto tx = threadIdx.x;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  if (tx == 0) s_count = 0;
  __syncthreads();

  if (static_cluster_threshold >= 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t seq_len = seq_lens[i];
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (seq_len > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) atomicAdd(&s_counts[count - 1], 1);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kCandidates[kNumCandidates - 1].threshold;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];  // # items with seq_len > kCandidates[j].threshold
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();

  constexpr uint32_t kClusterFloor = 32768;  // a very loose lower bound on threshold
  const auto cluster_threshold = max(s_threshold, kClusterFloor);

  // Compact items with seq_len > threshold into metadata[1..N]: their batch ids
  // are the work list the persistent cluster pool fetches.
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t seq_len = seq_lens[i];
    assert(static_cast<int32_t>(seq_len) >= 0 && "negative seq_len detected");
    if (seq_len > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, seq_len};
    }
  }
  __syncthreads();
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {.cluster_threshold = cluster_threshold, .num_cluster_items = s_count};
  }
}

#endif  // SUPPORT_CLUSTER

#ifdef USE_ROCM
// ---------------------------------------------------------------------------
// Split path (ROCm): one row across several blocks, cooperating through global
// memory instead of a cluster.
//
// This is the CDNA answer to TopKCluster, not a port of it. The cluster path
// needs thread-block clusters and distributed shared memory -- one cluster owns
// a row, the ranks all-reduce their histograms over DSMEM and synchronise with
// cluster.sync() -- and CDNA has neither primitive. What it does have is the
// pattern v1's topk.hip already uses on this hardware: put the shared state in
// global memory and let a kernel boundary be the barrier.
//
// Two launches, which is the fewest this can be done in without assuming the
// blocks of a row are co-resident:
//
//   1. topk_split_hist    each rank histograms its own chunk and folds it into
//                         the row's shared histogram with global atomics.
//   2. topk_split_select  each rank reads that histogram, finds the threshold,
//                         scans its chunk again and appends its candidates.
//                         The last rank to arrive resolves the tie tail and
//                         applies the page-table transform.
//
// Nothing spins: the epilogue runs in whichever block arrives last, so the path
// makes no forward-progress assumption that a plain launch does not already
// guarantee.
//
// The shared histogram is accumulated rather than given one plane per rank and
// summed by the reader. Per-rank planes need no zeroing, which is tidier, but
// the reader then walks `split` planes in a loop the compiler cannot software
// pipeline, so it pays the memory latency `split` times over -- 23 us of a
// 26 us kernel, measured. Accumulating costs a zeroed buffer instead, and the
// epilogue block clears the row on its way out so the next launch finds it
// clean (the allocation is zeroed once, for the first call).
//
// That reset makes the workspace single-stream state: two streams running this
// path on one device would interleave their histograms. Every consumer runs the
// model on one stream, and the counters below have the same shape of problem,
// but it is the reason this scratch cannot simply be shared more widely.
// ---------------------------------------------------------------------------

constexpr uint32_t kSplitMax = 32;  ///< most blocks one row may take
constexpr uint32_t kSplitMin = 4;   ///< fewest that pays for the second launch

/// Blocks a launch may spread its rows over: the cross-block cost grows with
/// rows * split, while the scan it buys back only shrinks as L / split.
constexpr uint32_t kSplitBlocks = 64;

/// Shortest row worth splitting: the second launch and the once-per-row
/// epilogue have to be covered, and their cost tracks the batch, not the spread.
inline constexpr uint32_t split_floor(uint32_t batch_size) {
  return batch_size <= 8 ? 40960 : batch_size <= 16 ? 49152 : batch_size <= 32 ? 65536 : 114688;
}

/// A cache line each: rows reserve their output slots with atomics on these,
/// and packing four rows into one line makes those atomics serialize across
/// rows that have nothing to do with each other.
struct alignas(128) SplitCounters {
  uint32_t count_gt;
  uint32_t count_eq;
  uint32_t arrive;
  uint32_t _pad;
};

struct SplitWorkspace {
  uint32_t* __restrict__ hist;        ///< [rows][kHistSize], accumulated, left zeroed
  SplitCounters* __restrict__ ctr;    ///< [rows]
  impl::TieValue* __restrict__ ties;  ///< [rows][kMaxNumTie]
  uint32_t split;
  uint32_t floor;  ///< same value the host dispatched on
};

/// 12 histogram bits, the width TopKStreaming uses: a 10-bit threshold bin holds
/// more unequal scores than kMaxNumTie can stage, and drops the rest in silence.
struct TopKSplit : impl::TopKRadixBase<12> {
  using Base = impl::TopKRadixBase<12>;
  static_assert(kHistSize % kBlockSize == 0, "the histogram is transferred kHistItems bins per thread");
  /// Bins per thread, in the contiguous tx * kHistItems layout the base uses.
  static constexpr uint32_t kHistItems = kHistSize / kBlockSize;
  static constexpr uint32_t kWarp = kBlockSize / impl::TopKConfig::kNumWarps;

  struct Smem : Base::Smem {
    uint32_t base_gt, base_eq, total_gt, total_eq, is_last;
    int32_t staged[kMaxTopK];
  };

  struct Chunk {
    uint32_t start, len;
  };

  /// This rank's slice. Chunk starts are rounded up to a whole wavefront of
  /// vector loads so that for_each_input's 16-byte path stays aligned on every
  /// rank, not just the first.
  SGL_DEVICE static Chunk chunk_of(uint32_t seq_len, uint32_t rank, uint32_t split) {
    constexpr uint32_t kAlign = kWarp * kVecSize;
    const uint32_t per = (seq_len + split - 1) / split;
    const uint32_t size = ((per + kAlign - 1) / kAlign) * kAlign;
    const uint32_t start = min(rank * size, seq_len);
    return {start, min(start + size, seq_len) - start};
  }

  SGL_DEVICE static void
  histogram_chunk(const TopKProblem& problem, Chunk chunk, uint32_t* __restrict__ row_hist, Smem* smem) {
    const auto tx = threadIdx.x;
    init_histogram(smem->histogram, tx);
    __syncthreads();
    for_each_input(problem.in + chunk.start, chunk.len, [&](float val, uint32_t) {
      atomicAdd(&smem->histogram[impl::extract_coarse_bin<kHistBits>(val)], 1);
    });
    __syncthreads();
    // One atomic per bin per rank, so `split` of them per address at worst.
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      const auto bin = tx * kHistItems + i;
      if (const auto n = smem->histogram[bin]; n != 0) atomicAdd(&row_hist[bin], n);
    }
  }

  /// Hand a row's histogram back zeroed, so the next launch needs no reset.
  SGL_DEVICE static void clear_row(uint32_t* __restrict__ row_hist) {
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i)
      row_hist[threadIdx.x * kHistItems + i] = 0;
  }

  /// Scan this rank's chunk and append what clears the threshold.
  ///
  /// Counted in LDS first and committed with one global atomic per block per
  /// class, the way the cluster path stages through `tmp_out`. Taking a slot
  /// per candidate straight from the global counter looks tempting because the
  /// above-threshold candidates are bounded by topk -- but the threshold bin
  /// itself is not, and a bin holding a few thousand elements turns into a few
  /// thousand serialized atomics on one address, which measured five times
  /// slower than not splitting at all.
  SGL_DEVICE static void select_chunk(
      const TopKProblem& problem,
      Chunk chunk,
      const uint32_t* __restrict__ row_hist,
      SplitCounters* __restrict__ ctr,
      impl::TieValue* __restrict__ ties,
      Smem* smem) {
    const auto tx = threadIdx.x;
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      const auto bin = tx * kHistItems + i;
      smem->histogram[bin] = row_hist[bin];
    }
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
      smem->v_hi = impl::padding_value();
      smem->v_lo = impl::padding_value();
    }
    __syncthreads();
    // The full row's histogram and the full row's seq_len, so every rank picks
    // the same bin and the appends below agree on what "above" means.
    find_threshold(problem.topk, problem.seq_len, smem, [&](uint32_t threshold_bin) {
      smem->v_hi = impl::coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
      smem->v_lo = impl::coarse_bin_lower_bound<kHistBits>(threshold_bin + 0);
    });

    const auto topk = problem.topk;
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    __syncthreads();

    for_each_input(problem.in + chunk.start, chunk.len, [&](float val, uint32_t local) {
      const auto idx = chunk.start + local;
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1u);
        // The whole row has fewer than topk of these, so this rank has too.
        if (pos < topk) [[likely]]
          smem->staged[pos] = static_cast<int32_t>(idx);
      } else if (val >= v_lo) {
        const auto slot = atomicAdd(&smem->count_eq, 1u);
        if (slot < kMaxNumTie) [[likely]]
          smem->tie_values[slot] = {val, idx};
      }
    });
    __syncthreads();

    const auto n_gt = min(smem->count_gt, topk);
    const auto n_eq = min(smem->count_eq, kMaxNumTie);
    if (tx == 0) {
      smem->base_gt = atomicAdd(&ctr->count_gt, n_gt);
      smem->base_eq = atomicAdd(&ctr->count_eq, n_eq);
    }
    __syncthreads();
    const auto base_gt = smem->base_gt;
    const auto base_eq = smem->base_eq;

    for (uint32_t t = tx; t < n_gt; t += kBlockSize) {
      if (base_gt + t < topk) problem.emit(base_gt + t, static_cast<uint32_t>(smem->staged[t]));
    }
    for (uint32_t t = tx; t < n_eq; t += kBlockSize) {
      if (base_eq + t < kMaxNumTie) ties[base_eq + t] = smem->tie_values[t];
    }
  }

  /// True in exactly one block per row, once every rank's appends are visible.
  /// Nobody waits: the epilogue simply runs wherever the last arrival lands.
  SGL_DEVICE static bool arrive_last(SplitCounters* __restrict__ ctr, uint32_t split, Smem* smem) {
    __syncthreads();  // this block's appends are done and visible to thread 0
    if (threadIdx.x == 0) {
      // Both fences belong to this thread alone. A device-scope fence on a
      // multi-die part is a cross-L2 operation, and letting all 1024 threads
      // issue one costs 8 us of a 26 us kernel; the __syncthreads above
      // already gave thread 0 the rest of the block's writes to push out.
      __threadfence();  // release: the appends land before the arrival does
      const bool last = atomicAdd(&ctr->arrive, 1u) == split - 1;
      smem->is_last = last ? 1u : 0u;
      if (last) {
        __threadfence();  // acquire the other ranks' appends
        // Through the atomic path that wrote them: a plain load could be
        // served out of this CU's own stale cache.
        smem->total_gt = atomicAdd(&ctr->count_gt, 0u);
        smem->total_eq = atomicAdd(&ctr->count_eq, 0u);
      }
    }
    __syncthreads();
    return smem->is_last != 0;
  }

  /// Fill the slots the threshold bin has to break ties for. handle_tie takes a
  /// plain pointer, so it could read the workspace directly, but its ranking
  /// pass is all-to-all over the candidates; staging them into LDS first keeps
  /// that out of global memory.
  SGL_DEVICE static void finish_ties(const TopKProblem& problem, const impl::TieValue* ties, Smem* smem) {
    const auto tx = threadIdx.x;
    const auto above_count = smem->total_gt;
    const auto tie_count = min(smem->total_eq, kMaxNumTie);
    const auto remain_topk = above_count < problem.topk ? problem.topk - above_count : 0;
    for (uint32_t t = tx; t < tie_count; t += kBlockSize)
      smem->tie_values[t] = ties[t];
    __syncthreads();
    handle_tie(smem->tie_values, problem, above_count, tie_count, remain_topk, &smem->tie_handle);
  }
};

template <bool kPDL>
TOPK_KERNEL void topk_split_hist(const __grid_constant__ TopKPagedParams params, const SplitWorkspace ws) {
  device::enable_smem_spilling();
  const auto row = blockIdx.x;
  const auto rank = blockIdx.y;
  const auto tx = threadIdx.x;
  // One launch boundary ahead of the only reader, so no fence is needed.
  if (rank == 0 && tx < sizeof(SplitCounters) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t*>(&ws.ctr[row])[tx] = 0;
  }

  const auto problem = params.problem(row);
  if (problem.seq_len <= ws.floor) return;  // the select pass takes it whole

  __shared__ impl::MaxSmem<TopKSplit::Smem> smem;
  const auto chunk = TopKSplit::chunk_of(problem.seq_len, rank, ws.split);
  auto* row_hist = ws.hist + static_cast<size_t>(row) * TopKSplit::kHistSize;
  device::PDLWaitPrimary<kPDL>();
  TopKSplit::histogram_chunk(problem, chunk, row_hist, reinterpret_cast<TopKSplit::Smem*>(&smem));
}

template <bool kPDL, TopKMode kMode>
TOPK_KERNEL void topk_split_select(const __grid_constant__ TopKPagedParams params, const SplitWorkspace ws) {
  device::enable_smem_spilling();
  const auto row = blockIdx.x;
  const auto rank = blockIdx.y;
  auto problem = params.problem(row);
  constexpr bool kNeedStaging = kMode != TopKMode::INDICES;
  __shared__ impl::MaxSmem<Register4::Smem, Streaming::Smem, TopKSplit::Smem> smem;

  // Rows too short to be worth splitting were skipped by the histogram pass;
  // rank 0 runs them on the ordinary one-block paths and the rest retire, the
  // same election the cluster kernel makes for its short items.
  if (problem.seq_len <= ws.floor) {
    if (rank != 0) return;
    __shared__ int32_t s_topk_indices[kNeedStaging ? kMaxTopK : 1];
    if (problem.seq_len <= problem.topk) {
      return trivial_transform<kPDL, kMode>(problem, params.get_transform(row));
    }
    if constexpr (kNeedStaging) problem.out = s_topk_indices;
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDL>(problem, &smem);
    } else {
      Streaming::forward<kPDL>(problem, &smem);
    }
    device::PDLTriggerSecondary<kPDL>();
    if constexpr (kNeedStaging) {
      __syncthreads();
      paged_transform<kMode>(problem, params.get_output_ptr(row), params.get_transform(row));
    }
    return;
  }

  auto* const split_smem = reinterpret_cast<TopKSplit::Smem*>(&smem);
  auto* const ctr = &ws.ctr[row];
  auto* const ties = ws.ties + static_cast<size_t>(row) * TopKSplit::kMaxNumTie;
  const auto chunk = TopKSplit::chunk_of(problem.seq_len, rank, ws.split);
  auto* const row_hist = ws.hist + static_cast<size_t>(row) * TopKSplit::kHistSize;

  TopKSplit::select_chunk(problem, chunk, row_hist, ctr, ties, split_smem);
  if (!TopKSplit::arrive_last(ctr, ws.split, split_smem)) return;

  TopKSplit::clear_row(row_hist);
  TopKSplit::finish_ties(problem, ties, split_smem);
  device::PDLTriggerSecondary<kPDL>();
  if constexpr (kNeedStaging) {
    // problem.out is already the destination, and paged_transform reads every
    // slot into registers before writing any, so transforming in place is safe.
    __syncthreads();
    paged_transform<kMode>(problem, problem.out, params.get_transform(row));
  }
}

/// Per-device scratch for the split path, allocated once and never freed.
///
/// Never freed on purpose. Growing the buffer would be worse than wasteful: a
/// HIP graph captured while an earlier allocation was current bakes that
/// address into its kernel arguments, so releasing it leaves those graphs
/// writing into memory the allocator has since handed to someone else -- the
/// same hazard v1's topk.hip avoids by taking its scratch from the caching
/// allocator per call. Allocating the worst case up front sidesteps both. The
/// worst case is small because the path is only taken when rows * split fits
/// the machine, so the histograms are bounded by the CU count and not by the
/// batch: about 2 MB on a 256-CU part.
struct SplitResources {
  int cu = 0;
  uint32_t max_rows = 0;
  SplitWorkspace ws{};

  SplitResources() = default;  // the "no split path here" state

  explicit SplitResources(int device_id) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_id) != hipSuccess) {
      (void)hipGetLastError();  // do not leave it for the next launch to trip on
      return;
    }
    cu = prop.multiProcessorCount;
    max_rows = std::max<uint32_t>(cu / kSplitMin, 1);

    const size_t hist_bytes = static_cast<size_t>(max_rows) * TopKSplit::kHistSize * sizeof(uint32_t);
    const size_t ctr_bytes = static_cast<size_t>(max_rows) * sizeof(SplitCounters);
    const size_t tie_bytes = static_cast<size_t>(max_rows) * TopKSplit::kMaxNumTie * sizeof(impl::TieValue);

    int prev = 0;
    (void)hipGetDevice(&prev);
    (void)hipSetDevice(device_id);
    void* base = nullptr;
    const auto total = hist_bytes + ctr_bytes + tie_bytes;
    // Zeroed once here; from then on each launch leaves the histogram clean.
    const bool ok = hipMalloc(&base, total) == hipSuccess && hipMemset(base, 0, total) == hipSuccess;
    (void)hipSetDevice(prev);
    if (!ok) {
      (void)hipGetLastError();
      cu = 0;  // the caller falls back to one block per row
      return;
    }
    auto* p = static_cast<char*>(base);
    ws.hist = reinterpret_cast<uint32_t*>(p);
    p += hist_bytes;
    ws.ctr = reinterpret_cast<SplitCounters*>(p);
    p += ctr_bytes;
    ws.ties = reinterpret_cast<impl::TieValue*>(p);
  }
};

inline const SplitResources& split_resources(int device_id) {
  // One slot per device: the buffer is a raw device pointer, so a single shared
  // one would be valid on exactly one of them.
  constexpr int kMaxDevices = 16;
  static std::once_flag once[kMaxDevices];
  static const SplitResources* slots[kMaxDevices] = {};
  static const SplitResources kNone{};
  if (device_id < 0 || device_id >= kMaxDevices) return kNone;
  std::call_once(once[device_id], [device_id] { slots[device_id] = new SplitResources(device_id); });
  return *slots[device_id];
}

/// How many blocks to give each row, and the scratch they share. Zero means the
/// ordinary one-block-per-row dispatch.
///
/// Both bounds are measured: split_floor for how long the row has to be,
/// kSplitBlocks for how far it is worth spreading.
inline auto split_plan(uint32_t batch_size, uint32_t max_seq_len, DLDevice device)
    -> std::pair<uint32_t, SplitWorkspace> {
  const auto& res = split_resources(device.device_id);
  if (res.cu <= 0 || batch_size == 0 || batch_size > res.max_rows) return {0, {}};
  const auto split = std::clamp<uint32_t>(kSplitBlocks / batch_size, kSplitMin, kSplitMax);
  const auto floor = split_floor(batch_size);
  if (split < kSplitMin || max_seq_len <= floor) return {0, {}};
  auto ws = res.ws;
  ws.split = split;
  ws.floor = floor;
  return {split, ws};
}
#endif  // USE_ROCM

template <bool kUsePDL>
struct TopKKernel {
  static void plan(  //
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView metadata,
      const int32_t static_cluster_threshold) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({-1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    RuntimeCheck(metadata.size(0) == B.unwrap() + 1, "invalid metadata shape");
#if SUPPORT_CLUSTER
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    // persistent cluster not supported
    if (kNumPersistentClusters == 0) return;
    // will not route to persistent cluster
    if (batch_size <= kNumPersistentClusters || batch_size > kClusterMaxBatch) return;
    const auto device = device_.unwrap();
    LaunchKernel(1, kBlockSize, device)(  //
        topk_plan_cluster,
        static_cast<const uint32_t*>(seq_lens.data_ptr()),
        static_cast<PlanItem*>(metadata.data_ptr()),
        batch_size,
        static_cluster_threshold);
#else
    static_cast<void>(static_cluster_threshold);
#endif
  }

  static void transform_paged(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::TensorView metadata,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    // Absent means "no page transform": `page_indices` then receives the raw
    // selected indices and nothing dereferences a page table.
    const int32_t* page_table_ptr = nullptr;
    int64_t page_table_stride = 0;
    if (page_table.has_value()) {
      TensorMatcher({B, -1})  // page_table
          .with_strides({-1, 1})
          .with_dtype<int32_t>()
          .with_device(device_)
          .verify(page_table.value());
      page_table_ptr = static_cast<const int32_t*>(page_table.value().data_ptr());
      page_table_stride = (page_table.value()).stride(0);
    }
    TensorMatcher({B, K})  // page_indices
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);
    TensorMatcher({-1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);
    // Present means "both outputs": `page_indices` receives the page-table
    // transform and `raw_indices` the selected raw indices, same -1 padding.
    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      RuntimeCheck(page_table.has_value(), "raw_indices requires a page table");
      TensorMatcher({B, K})  // raw_indices
          .with_dtype<int32_t>()
          .with_device(device_)
          .verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
    RuntimeCheck(metadata.size(0) == B.unwrap() + 1, "invalid metadata shape");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto max_seq_len = static_cast<uint32_t>(L.unwrap());
    const auto device = device_.unwrap();

    constexpr auto get_static_cluster_floor = [](uint32_t batch_size) -> uint32_t {
      // NOTE: 15 is exactly 0.5 wave which saturate all cluster-8 SMs on Hopper/Blackwell
      if constexpr (SGL_ARCH_BLACKWELL_OR_GREATER) {
        return batch_size <= 15 ? 24576 : 30720;
      } else if constexpr (SGL_ARCH_HOPPER_OR_GREATER) {
        return batch_size <= 15 ? 32768 : 65536;
      } else {
        return UINT_MAX;
      }
    };

    const auto params = TopKPagedParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = page_table_ptr,
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .metadata = static_cast<const PlanItem*>(metadata.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = page_table_stride,
        .topk = topk,
        .page_bits = page_bits,
        // only used in small batch variant
        .static_cluster_floor = get_static_cluster_floor(batch_size),
        // used for persistent cluster kernel and main kernel
        .batch_size = batch_size,
    };

    const auto dispatch = [&]<typename F>(F&& f) {
      const auto mode = raw_indices.has_value()  ? TopKMode::DUAL_OUTPUT
                        : page_table.has_value() ? TopKMode::PAGE_TABLE
                                                 : TopKMode::INDICES;
      switch (mode) {
        case TopKMode::INDICES:
          return f.template operator()<TopKMode::INDICES>();
        case TopKMode::PAGE_TABLE:
          return f.template operator()<TopKMode::PAGE_TABLE>();
        case TopKMode::DUAL_OUTPUT:
          return f.template operator()<TopKMode::DUAL_OUTPUT>();
        default:
          Panic("Invalid mode, this path should be unreachable");
      }
    };
    dispatch([&]<TopKMode kMode>() {
#if SUPPORT_CLUSTER
      const bool use_cluster = (max_seq_len > params.static_cluster_floor) && (batch_size <= kClusterMaxBatch);
      if (use_cluster) {
        if constexpr (kMaxCluster16BatchSize > 0) {
          if (batch_size <= kMaxCluster16BatchSize) {
            constexpr uint32_t kClusterSize = 16;
            // Widths above 8 are non-portable; the launch is rejected without this.
            const auto kernel = topk_small_batch_cluster_kernel<kUsePDL, kMode, kClusterSize, 1>;
            [[maybe_unused]]
            static const bool _ = [&kernel] {
              const auto kernel_ptr = reinterpret_cast<const void*>(kernel);
              CHECK_CUDA(::cudaFuncSetAttribute(kernel_ptr, ::cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
              return true;
            }();
            return LaunchKernel({batch_size, kClusterSize}, kBlockSize, device)
                .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
                .launch(kernel, params);
          }
        }

        if constexpr (kNumPersistentClusters > 0) {
          if (batch_size <= kNumPersistentClusters) {
            constexpr uint32_t kClusterSize = 8;
            return LaunchKernel({batch_size, kClusterSize}, kBlockSize, device)
                .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
                .launch(topk_small_batch_cluster_kernel<kUsePDL, kMode, kClusterSize, 2>, params);
          } else {
            constexpr uint32_t kClusterSize = 8;
            const uint32_t num_clusters = std::min(batch_size, kNumPersistentClusters);
            LaunchKernel({num_clusters, kClusterSize}, kBlockSize, device)
                .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
                .launch(topk_persistent_cluster_kernel<kUsePDL, kClusterSize>, params);
            LaunchKernel(batch_size, kBlockSize, device)
                .config({.use_pdl = kUsePDL})
                .launch(topk_main_kernel<kUsePDL, /*kLevel=*/3, kMode>, params);
            return void();
          }
        }
      }
#elif defined(USE_ROCM)
      // Split dispatch. One block per row leaves a long row latency bound on one
      // CU however idle the rest is; split_plan decides where a second launch pays.
      if (const auto [split, split_ws] = split_plan(batch_size, max_seq_len, device); split >= kSplitMin) {
        LaunchKernel({batch_size, split}, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_split_hist<kUsePDL>, params, split_ws);
        LaunchKernel({batch_size, split}, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_split_select<kUsePDL, kMode>, params, split_ws);
        return;
      }
#endif
      if (max_seq_len <= kReg2MaxSeqLen) {
        LaunchKernel(batch_size, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_main_kernel<kUsePDL, /*kLevel=*/0, kMode>, params);
      } else if (max_seq_len <= kReg4MaxSeqLen) {
        LaunchKernel(batch_size, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_main_kernel<kUsePDL, /*kLevel=*/1, kMode>, params);
      } else {
        LaunchKernel(batch_size, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_main_kernel<kUsePDL, /*kLevel=*/2, kMode>, params);
      }
    });
  }

  /**
   * \brief Ragged (prefill) variant of `transform`: per-row window, additive
   * output transform, no page table and no plan.
   *
   * `scores` is written in place: the <= 3 columns the 16-byte-aligned read base
   * pulls in ahead of each row's window are masked out (see
   * `topk_ragged_kernel`). They are invalid for that row, and the buffer has no
   * consumer after this call.
   *
   * `row_starts` absent means every window starts at column 0, which is the
   * single-request case; `out_offsets` is added to every selected position and
   * is what rebases them onto the flattened KV.
   */
  static void transform_ragged(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts,
      const tvm::ffi::TensorView out_offsets,
      const tvm::ffi::TensorView topk_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B})  // out_offsets
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(out_offsets);
    TensorMatcher({B, K})  // topk_indices
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(topk_indices);
    const int32_t* row_starts_ptr = nullptr;
    if (row_starts.has_value()) {
      TensorMatcher({B})  // row_starts
          .with_dtype<int32_t>()
          .with_device(device_)
          .verify(row_starts.value());
      row_starts_ptr = static_cast<const int32_t*>(row_starts.value().data_ptr());
    }

    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

    const auto params = TopKRaggedParams{
        .scores = static_cast<float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .row_starts = row_starts_ptr,
        .out_offsets = static_cast<const int32_t*>(out_offsets.data_ptr()),
        .topk_indices = static_cast<int32_t*>(topk_indices.data_ptr()),
        .score_stride = S.unwrap(),
        .topk = topk,
    };
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), kBlockSize, device_.unwrap())
        .config({.use_pdl = kUsePDL})
        .launch(topk_ragged_kernel<kUsePDL>, params);
  }
};

}  // namespace sglang
