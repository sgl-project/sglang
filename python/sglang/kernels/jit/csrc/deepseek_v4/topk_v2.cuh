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
#include <cstdint>
#include <iterator>
#include <limits>

namespace sglang {

namespace impl = device::topk;
using impl::TopKProblem;

enum class TopKMode {
  INDICES,     ///< raw selected indices into `out`; `page_table` unused
  PAGE_TABLE,  ///< page-table-transformed indices into `out`
};

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;
using Cluster = impl::TopKCluster<8>;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)
#define CLUSTER_TOPK_KERNEL TOPK_KERNEL __cluster_dims__(1, kClusterSize, 1)

constexpr uint32_t kClusterFloor = 65536;
constexpr uint32_t kClusterMaxBatch = 512;
constexpr uint32_t kNumPersistentClusters = 15 * kOccupancy;

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

struct TopKPagedParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;
  uint32_t cluster_floor;  // seq_len > this routes to the cluster path (batch-aware, host-set)

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
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        .in = scores + batch_id * score_stride,
        .out = page_indices + batch_id * k,
        .page_table = page_table + batch_id * page_table_stride,
        .topk = topk,
        .seq_len = seq_len,
        .page_bits = page_bits,
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

/**
 * \brief Persistent cluster kernel for the long items. It will handle long inputs.
 * The short items are handled by the separate topk_kernel.
 */
template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_persistent_cluster_kernel(const __grid_constant__ TopKPagedParams params) {
  device::enable_smem_spilling();
  __shared__ impl::MaxSmem<Cluster::Smem> smem;
  const uint32_t num_cluster_items = params.global().num_cluster_items;
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
#pragma unroll 1
  for (uint32_t w = blockIdx.x; w < num_cluster_items; w += kNumPersistentClusters) {
    const auto it = params.item(w);
    const auto problem = params.problem(it.batch_id, it.seq_len);
    Cluster::forward<false>(problem, &smem);
    __syncthreads();
  }
}

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
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
SGL_DEVICE void trivial_transform(const TopKProblem& problem) {
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t) {
    const auto idx = tx < problem.seq_len ? static_cast<int32_t>(tx) : -1;
    if constexpr (kMode == TopKMode::INDICES) {
      problem.emit(tx, idx);
    } else {
      problem.transform_output(tx, idx);
    }
  });
}

SGL_DEVICE void problem_transform(TopKProblem& problem, int32_t* output_ptr) {
  static_assert(kMaxTopK % kBlockSize == 0);
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
  int32_t source_index[kNumElems];
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { source_index[i] = problem.out[tx]; });
  problem.out = output_ptr;
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { problem.transform_output(tx, source_index[i]); });
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
      score[row_start - rem + tx] = -std::numeric_limits<float>::max();
    }
  }

  const auto problem = TopKProblem{
      .in = score + (row_start - rem),
      .out = out,
      .page_table = nullptr,  // unused
      .topk = topk,
      .seq_len = seq_len + rem,
      .page_bits = 1,  // unused
      .bias = offset - static_cast<int32_t>(rem),
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
  auto problem = params.problem(blockIdx.x);
  constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();
  constexpr bool kHandleCluster = (kLevel == 3);
  // Only the cluster path consumes the cluster kernel's output, so only it waits
  // on that kernel (kPDLFinal). Every other path waits at most on the indexer
  // (kPDLEarly) and must not be held on an SM slot until the long-running
  // persistent pool retires -- that would serialize the short items behind it.
  constexpr bool kPDLEarly = kPDL && !kHandleCluster;
  constexpr bool kPDLFinal = kPDL && kHandleCluster;
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDLEarly, kMode>(problem);

  constexpr bool kNeedStaging = kMode != TopKMode::INDICES;
  __shared__ int32_t s_topk_indices[kNeedStaging ? kMaxTopK : 1];
  if constexpr (kNeedStaging) problem.out = s_topk_indices;

  // non-trivial path: dispatch based on level and seq_len
  const auto cluster_threshold = kHandleCluster ? params.cluster_threshold() : kU32Max;
  if constexpr (kLevel == 0) {
    __builtin_assume(problem.seq_len <= kReg2MaxSeqLen);
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    __builtin_assume(problem.seq_len <= kReg4MaxSeqLen);
    Register4::forward<kPDL>(problem, &smem);  // max_seq_len <= 16384 guarantees seq <= 16384
  } else {
    static_assert(kLevel == 2 || kLevel == 3, "we only support level = 0,1,2,3 now");
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= cluster_threshold) {
      Streaming::forward<kPDLEarly>(problem, &smem);
    } else {
      // Cluster path: the pool already selected into our output row; the only
      // work left is the epilogue, so this is the one path that waits for it.
      problem.out = params.get_output_ptr(blockIdx.x);
      device::PDLWaitPrimary<kPDLFinal>();
    }
  }

  device::PDLTriggerSecondary<kPDL>();
  if constexpr (kNeedStaging) {
    __syncthreads();
    problem_transform(problem, params.get_output_ptr(blockIdx.x));
  }
}

template <bool kPDL, TopKMode kMode>
CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __grid_constant__ TopKPagedParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  __shared__ impl::MaxSmem<Streaming::Smem, Cluster::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDL, kMode>(problem);

  constexpr bool kNeedStaging = kMode != TopKMode::INDICES;
  __shared__ int32_t s_topk_indices[kNeedStaging ? kMaxTopK : 1];
  if constexpr (kNeedStaging) problem.out = s_topk_indices;

  // randomly elect one worker rank to avoid workload imbalance
  const auto worker_rank = blockIdx.x % kClusterSize;

  // for small batch, we will fuse in the cluster case
  if (problem.seq_len <= kReg4MaxSeqLen) {
    if (blockIdx.y != worker_rank) return;
    Register4::forward<kPDL>(problem, &smem);
    __syncthreads();
  } else if (problem.seq_len <= params.cluster_floor) {
    if (blockIdx.y != worker_rank) return;
    Streaming::forward<kPDL>(problem, &smem);
    __syncthreads();
  } else {
    auto cluster = cooperative_groups::this_cluster();
    if constexpr (kNeedStaging) {
      problem.out = cluster.map_shared_rank(s_topk_indices, worker_rank);
    }
    Cluster::forward<kPDL>(problem, &smem);
    if constexpr (kNeedStaging) {
      cluster.sync();
      if (blockIdx.y != worker_rank) return;
    }
  }

  device::PDLTriggerSecondary<kPDL>();
  if constexpr (kNeedStaging) {
    // Only the elected worker reaches here, and it mapped `topk_indices` to
    // itself, so `problem.out` is this block's own buffer. Stating that keeps the
    // shared::cluster address out of the load problem_transform issues -- which is
    // load-bearing, not an optimization: without it cicc segfaults on CUDA 13.1+
    // for sm_90a (issue #32830, previously worked around by copying `problem` in
    // #32910). Verified: dropping this line reproduces the crash on 13.1/13.2/13.3.
    __builtin_assume(problem.out == s_topk_indices);
    problem_transform(problem, params.get_output_ptr(blockIdx.x));
  }
}

// --- Plan: choose cluster_threshold from the seq_len distribution -----------
__global__ __launch_bounds__(kBlockSize, 1) void topk_plan(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,  // [0]=GlobalMetadata, [1+i]=PlanItem
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
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
      {65536, 30},    // (65536,98304]:    ~1 pool wave, streams beyond 30
      {98304, 48},    // (98304,131072]
      {131072, 60},   // (131072,196608]
      {196608, 80},   // (196608,262144]
      {262144, 112},  // (262144,393216]
      {393216, 128},  // (393216,inf):     longest -- worth many pool waves; a top
                      // threshold here lets overloaded ~280-393K batches still stream
  };
  constexpr uint32_t kNumCandidates = std::size(kCandidates);
  static_assert(kCandidates[0].threshold == kClusterFloor);

  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;
  __shared__ uint32_t s_count;

  const auto tx = threadIdx.x;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  if (tx == 0) s_count = 0;
  __syncthreads();

  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
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
  const auto cluster_threshold = max(s_threshold, kClusterFloor);

  // Compact items with seq_len > threshold into metadata[1..N]: their batch ids
  // are the work list the persistent cluster pool fetches.
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl};
    }
  }
  __syncthreads();
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {.cluster_threshold = cluster_threshold, .num_cluster_items = s_count};
  }
}

struct TopKKernel {
  static void plan(  //
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView metadata,
      const uint32_t static_cluster_threshold) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({Bp1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto device = device_.unwrap();
    LaunchKernel(1, kBlockSize, device)(  //
        topk_plan,
        static_cast<const uint32_t*>(seq_lens.data_ptr()),
        static_cast<PlanItem*>(metadata.data_ptr()),
        batch_size,
        static_cluster_threshold);
  }

  static void transform_paged(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::TensorView metadata) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

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
          .with_strides({P, 1})
          .with_dtype<int32_t>()
          .with_device(device_)
          .verify(page_table.value());
      page_table_ptr = static_cast<const int32_t*>(page_table.value().data_ptr());
      page_table_stride = P.unwrap();
    }
    TensorMatcher({B, K})  // page_indices
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);
    TensorMatcher({Bp1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto max_seq_len = static_cast<uint32_t>(L.unwrap());
    const auto device = device_.unwrap();

    // The fused kernel runs one 8-block cluster per batch element, and B200 fits one
    // wave of exactly 15 such clusters (occ2). For batch <= 15 it stays latency-bound,
    // so the 8-way split beats streaming from a much lower seq (measured crossover
    // ~36-40K); batch 16 spills into a 2nd wave (+25%) and keeps the 64K floor.
    // The floor is chosen on the host per launch.
    constexpr uint32_t kClusterFloorSmall = 32768;
    constexpr uint32_t kSmallBatchLowFloor = 15;
    const auto params = TopKPagedParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = page_table_ptr,
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .metadata = static_cast<const PlanItem*>(metadata.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = page_table_stride,
        .topk = topk,
        .page_bits = page_bits,
        .cluster_floor = (batch_size <= kSmallBatchLowFloor) ? kClusterFloorSmall : kClusterFloor,
    };

    const bool use_cluster = (max_seq_len > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
    constexpr bool kUsePDL = true;
    const auto mode = page_table.has_value() ? TopKMode::PAGE_TABLE : TopKMode::INDICES;
    const auto dispatch = [&]<typename F>(F&& f) {
      switch (mode) {
        case TopKMode::INDICES:
          return f.template operator()<TopKMode::INDICES>();
        default:
          return f.template operator()<TopKMode::PAGE_TABLE>();
      }
    };
    dispatch([&]<TopKMode kMode>() {
      if (use_cluster) {
        if (batch_size <= kNumPersistentClusters) {
          LaunchKernel({batch_size, kClusterSize}, kBlockSize, device)
              .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
              .launch(topk_small_batch_kernel<kUsePDL, kMode>, params);
        } else {
          const uint32_t num_clusters = std::min(batch_size, kNumPersistentClusters);
          LaunchKernel({num_clusters, kClusterSize}, kBlockSize, device)
              .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
              .launch(topk_persistent_cluster_kernel<kUsePDL>, params);
          LaunchKernel(batch_size, kBlockSize, device)
              .config({.use_pdl = kUsePDL})
              .launch(topk_main_kernel<kUsePDL, /*kLevel=*/3, kMode>, params);
        }
      } else if (max_seq_len <= kReg2MaxSeqLen) {
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
    device_.set_options<kDLCUDA>();

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

    constexpr bool kUsePDL = true;
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
