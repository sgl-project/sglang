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

#include <bit>
#include <climits>
#include <cstdint>
#include <iterator>

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
    return {page_table + batch_id * page_table_stride, page_bits, raw_indices + batch_id * static_cast<int64_t>(topk)};
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

/**
 * \brief Persistent cluster kernel for the long items. It will handle long inputs.
 * The short items are handled by the separate topk_kernel.
 */
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
