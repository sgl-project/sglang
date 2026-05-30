/// \file topk_v2.cuh
/// \brief DeepSeek-V4 (DSA indexer) top-k transform: __global__ kernels + host
///        dispatcher. The implementation classes live in the single header
///        <sgl_kernel/deepseek_v4/topk.cuh>.
///
/// Universal, runtime-`topk` (<= 2048) top-k + page-table transform with ragged
/// per-batch seq_lens. Dynamic (plan-routed) dispatch:
///   - `plan` (single block) decides a `cluster_threshold` from the seq_len
///     distribution + batch size and writes it to a relocatable metadata tensor.
///   - `topk_cluster_kernel` (one cluster of 8 blocks per element, occupancy 2 =>
///     ~30 concurrent on B200) processes the "long" items (seq > threshold) and
///     skips the rest.
///   - `topk_kernel` (one block per element) processes the "short" items
///     (trivial/register/streaming) and skips the long ones.
///
/// Why non-persistent clusters: a persistent pool (fixed 30 clusters looping over
/// items) was measured ~20% slower for long contexts -- the serial per-cluster
/// item loop with cluster.sync barriers loses the inter-wave pipelining that a
/// plain non-persistent launch gets for free at occupancy 2. The plan's
/// `cluster_threshold` is what keeps medium-context items on the cheaper
/// streaming path (fixing the mid-batch regressions).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/topk.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <iterator>

namespace {

namespace impl = device::topk;
using impl::TopKProblem;

using Trivial = impl::TopKTrivial;
using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;
using Cluster = impl::TopKCluster<8>;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384 (max_register_vecs=4)

// Below this context length the 8-way cluster split is too fine to beat plain
// streaming, so never cluster (also the smallest plan candidate threshold).
constexpr uint32_t kClusterFloor = 65536;
// Above this batch, one-block-per-element streaming already fills the GPU at
// occupancy 2, so the cluster path cannot help -- don't launch it.
constexpr uint32_t kClusterMaxBatch = 128;
// Persistent cluster pool size (clusters of 8 blocks). ~15 = one co-scheduled
// wave on B200's 8 GPCs; 30 = two waves (occupancy 2). The pool loops to fetch
// work from the plan list, so the all-short case wastes only this many clusters
// (not batch_size). Tunable.
constexpr uint32_t kNumPersistentClusters = 30;

constexpr bool kUsePDL = true;

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

struct TopKLaunchParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;

  SGL_DEVICE const GlobalMetadata& global() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE uint32_t cluster_threshold() const { return global().cluster_threshold; }
  SGL_DEVICE const PlanItem& item(uint32_t i) const { return metadata[1 + i]; }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        .in = scores + batch_id * score_stride,
        .out = page_indices + batch_id * k,
        .raw_out = nullptr,
        .page_table = page_table + batch_id * page_table_stride,
        .topk = topk,
        .seq_len = seq_len,
        .page_bits = page_bits,
    };
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    return problem(batch_id, static_cast<uint32_t>(seq_lens[batch_id]));
  }
};

// One block per batch element (no cluster). Specialized on the host-known
// max_seq_len bucket `kLevel` so a small-max_seq_len launch only compiles the
// paths it needs (leaner kernel, less register/smem-spill reservation):
//   kLevel 0: max_seq_len <= 8192   -> trivial + register<2>
//   kLevel 1: max_seq_len <= 16384  -> + register<4>
//   kLevel 2: max_seq_len  > 16384  -> + streaming  (the "main" kernel)
// When kSkipLong, items routed to the cluster path are skipped (full
// one-block-per-element parallelism for the short items, no cluster cap).
template <bool kPDL, bool kSkipLong, int kLevel>
__global__ __launch_bounds__(kBlockSize, kOccupancy) void topk_kernel(
    const __grid_constant__ TopKLaunchParams params) {
  impl::enable_smem_spilling();
  const auto problem = params.problem(blockIdx.x);
  if constexpr (kSkipLong) {
    if (problem.seq_len > params.cluster_threshold()) return;  // cluster path handles it
  }
  alignas(alignof(Streaming::Smem)) __shared__ uint8_t smem[sizeof(Streaming::Smem)];
  if (problem.seq_len <= problem.topk) {
    Trivial::forward<kPDL>(problem);
  } else if (problem.seq_len <= kReg2MaxSeqLen) {
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    Register4::forward<kPDL>(problem, &smem);  // max_seq_len <= 16384 guarantees seq <= 16384
  } else if constexpr (kLevel >= 2) {
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDL>(problem, &smem);
    } else {
      Streaming::forward<kPDL>(problem, &smem);
    }
  }
  // Pipelined page-table transform pass (gathers kept out of the hot scatter loop),
  // then trigger the dependent kernel only after the full output is written.
  __syncthreads();
  for (uint32_t t = threadIdx.x; t < problem.topk; t += kBlockSize) impl::transform_output(problem, t);
  device::PDLTriggerSecondary<kPDL>();
}

// Persistent cluster pool: a FIXED gridDim.x (<= kNumPersistentClusters) clusters
// of 8 blocks loop to fetch the compacted long items from the plan list (stride
// gridDim.x). Bounded launch -- the all-short case wastes only the pool's clusters
// in the first wave (not batch_size*8), PDL overlaps it, and all-long small-batch
// finishes in the first wave. Long items always have seq_len > cluster_threshold
// >= kClusterFloor > kMaxTopK, so they take the radix (never trivial) path.
template <bool kPDL>
__global__ __launch_bounds__(kBlockSize, kOccupancy) __cluster_dims__(1, kClusterSize, 1) void topk_persistent_cluster_kernel(
    const __grid_constant__ TopKLaunchParams params) {
  impl::enable_smem_spilling();
  alignas(alignof(Cluster::Smem)) __shared__ uint8_t smem[sizeof(Cluster::Smem)];
  const uint32_t n = params.global().num_cluster_items;
  device::PDLWaitPrimary<kPDL>();
  for (uint32_t w = blockIdx.x; w < n; w += gridDim.x) {
    const auto it = params.item(w);
    const auto problem = params.problem(it.batch_id, it.seq_len);
    Cluster::forward(problem, &smem);  // writes raw indices to out
    // Barrier: all ranks' raw writes are visible before the transform pass; the
    // 8 ranks then split the topk slots for the page-table transform.
    cooperative_groups::this_cluster().sync();
    for (uint32_t t = blockIdx.y * kBlockSize + threadIdx.x; t < problem.topk; t += kClusterSize * kBlockSize)
      impl::transform_output(problem, t);
  }
  device::PDLTriggerSecondary<kPDL>();
}

// --- Plan: choose cluster_threshold from the seq_len distribution -----------
__global__ __launch_bounds__(kBlockSize, 1) void topk_plan(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,  // [0]=GlobalMetadata, [1+i]=PlanItem
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
  // Candidate thresholds (strictly increasing) with the max number of "long"
  // items at which clustering still beats streaming for that context length.
  // Cluster efficiency rises with context length (the 8-way split amortizes), so
  // longer thresholds tolerate more long items. Tuned from the cluster/stream
  // crossover on B200.
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
      {65536, 16},
      {98304, 48},
      {131072, 96},
      {196608, 128},
      {262144, 128},
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

struct CombinedTopKKernel {
  static void plan(  //
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView metadata,
      const uint32_t static_cluster_threshold) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({Bp1, 2}).with_dtype<int32_t>().with_device(device_).verify(metadata);

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

  /// `workspace` is accepted to preserve the FFI interface (the new kernels use
  /// distributed shared memory, no global workspace); it is ignored.
  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      [[maybe_unused]] const tvm::ffi::TensorView workspace,
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

    TensorMatcher({B, L}).with_strides({S, 1}).with_dtype<float>().with_device(device_).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device_).verify(page_indices);
    TensorMatcher({Bp1, 2}).with_dtype<int32_t>().with_device(device_).verify(metadata);

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto max_seq_len = static_cast<uint32_t>(L.unwrap());
    const auto device = device_.unwrap();

    const auto params = TopKLaunchParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .metadata = static_cast<const PlanItem*>(metadata.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .topk = topk,
        .page_bits = page_bits,
    };

    const bool use_cluster = (max_seq_len > kClusterFloor) && (batch_size <= kClusterMaxBatch);
    if (use_cluster) {
      // Persistent cluster pool (bounded grid) handles the long items via the plan
      // work list; the per-element kernel handles the short ones (skips long).
      // Short items can be up to kClusterFloor, so the per-element kernel is level 2.
      const uint32_t pool = batch_size < kNumPersistentClusters ? batch_size : kNumPersistentClusters;
      LaunchKernel(dim3{pool, kClusterSize}, kBlockSize, device)
          .enable_cluster(dim3{1, kClusterSize})
          .enable_pdl(kUsePDL)(topk_persistent_cluster_kernel<kUsePDL>, params);
      LaunchKernel(batch_size, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(topk_kernel<kUsePDL, /*kSkipLong=*/true, /*kLevel=*/2>, params);
    } else if (max_seq_len <= kReg2MaxSeqLen) {
      LaunchKernel(batch_size, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(topk_kernel<kUsePDL, /*kSkipLong=*/false, /*kLevel=*/0>, params);
    } else if (max_seq_len <= kReg4MaxSeqLen) {
      LaunchKernel(batch_size, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(topk_kernel<kUsePDL, /*kSkipLong=*/false, /*kLevel=*/1>, params);
    } else {
      LaunchKernel(batch_size, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(topk_kernel<kUsePDL, /*kSkipLong=*/false, /*kLevel=*/2>, params);
    }
  }
};

}  // namespace
