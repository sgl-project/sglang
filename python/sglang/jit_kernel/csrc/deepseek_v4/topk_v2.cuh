/// \file topk_v2.cuh
/// \brief DeepSeek-V4 (DSA indexer) top-k transform: __global__ kernels + host
///        dispatcher. The implementation classes live in the single header
///        <sgl_kernel/deepseek_v4/topk.cuh>.
///
/// Universal, runtime-`topk` (<= 2048) replacement for the old compile-time-K
/// v2. Computes the per-row top-k of `scores` and writes the page-table
/// transform of the selected indices into `page_indices`. Per-batch ragged
/// `seq_lens` are supported: the host launches one universal kernel and each
/// block dispatches on its own length.
///
/// The public FFI interface (`transform` taking a `workspace` + `metadata`, and
/// `plan` producing that `metadata`) is preserved for the future persistent /
/// dynamic dispatcher. The current dispatch is deliberately naive and ignores
/// `workspace` and `metadata` (we may tune / consume them later).

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
using Register = impl::TopKRegister;
using Streaming = impl::TopKStreaming;
using Cluster = impl::TopKCluster<8>;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kRegisterMaxSeqLen = Register::kMaxSeqLen;  // 8192

// Naive dispatch thresholds. NOTE: deliberately untuned -- the cluster path is
// used only when one-block-per-element under-fills the GPU (small batch) AND
// contexts are long enough to amortize the cross-block histogram reduction.
constexpr uint32_t kClusterMinSeqLen = 65536;  // >= 64K
constexpr uint32_t kClusterMaxBatch = 128;

constexpr bool kUsePDL = true;

/// Launch parameters shared by both transform kernels. `problem(batch_id)`
/// materializes one batch element's work, reading its ragged `seq_len`.
struct TopKLaunchParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;

  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        .in = scores + batch_id * score_stride,
        .out = page_indices + batch_id * k,
        .raw_out = nullptr,
        .page_table = page_table + batch_id * page_table_stride,
        .topk = topk,
        .seq_len = static_cast<uint32_t>(seq_lens[batch_id]),
        .page_bits = page_bits,
    };
  }
};

// One block per batch element: trivial / register / streaming by length.
template <bool kPDL>
__global__ __launch_bounds__(kBlockSize, kOccupancy) void topk_kernel(
    const __grid_constant__ TopKLaunchParams params) {
  impl::enable_smem_spilling();
  const auto problem = params.problem(blockIdx.x);
  alignas(alignof(Streaming::Smem)) __shared__ uint8_t smem[sizeof(Streaming::Smem)];
  if (problem.seq_len <= problem.topk) {
    Trivial::forward<kPDL>(problem);
  } else if (problem.seq_len <= kRegisterMaxSeqLen) {
    Register::forward<kPDL>(problem, &smem);
  } else {
    Streaming::forward<kPDL>(problem, &smem);
  }
}

// kClusterSize blocks per batch element (one cluster per element). Short rows
// fall back to the trivial path on rank 0; everything else uses the cluster.
template <bool kPDL>
__global__ __launch_bounds__(kBlockSize, kOccupancy) __cluster_dims__(1, kClusterSize, 1) void topk_cluster_kernel(
    const __grid_constant__ TopKLaunchParams params) {
  impl::enable_smem_spilling();
  const auto problem = params.problem(blockIdx.x);
  alignas(alignof(Cluster::Smem)) __shared__ uint8_t smem[sizeof(Cluster::Smem)];
  if (problem.seq_len <= problem.topk) {
    if (blockIdx.y == 0) Trivial::forward<kPDL>(problem);
  } else {
    Cluster::forward<kPDL>(problem, &smem);
  }
}

// ---------------------------------------------------------------------------
// Plan kernel (persistent / dynamic dispatcher metadata). Preserved verbatim
// from the previous v2 for the future scheduler; its output is NOT consumed by
// the naive transform above. Operates purely on seq_lens -> metadata.
// ---------------------------------------------------------------------------

/// Per-item routing entry; one 16-byte row of the metadata tensor.
struct alignas(16) PlanItem {
  uint32_t batch_id;
  uint32_t seq_len;
  bool has_next;
};
/// metadata row 0: global routing decision (one PlanItem-sized slot).
struct alignas(16) GlobalMetadata {
  uint32_t cluster_threshold;  // decided per-batch in plan kernel
  uint32_t num_cluster_items;  // N = number of items routed to the cluster path
  uint32_t reserved[2];
};
static_assert(sizeof(GlobalMetadata) == sizeof(PlanItem), "metadata row 0 must occupy one PlanItem slot");

constexpr uint32_t kNumClusters = 15;            // hardware-limited persistent clusters
constexpr uint32_t kPlanMax2PassLength = 32768;  // below this, never cluster
constexpr uint32_t kPlanMaxSupportedLength = 262144;

__global__ __launch_bounds__(kBlockSize, 1) void topk_plan(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
  // Candidate thresholds, strictly increasing; chosen to give the auto-heuristic
  // reasonable granularity without a full sort. All >= kPlanMax2PassLength.
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
      {32768, 30},
      {40960, 45},
      {49152, 45},
      {65536, 60},
      {98304, 60},
      {131072, 75},
      {196608, 90},
      {262144, 105},
  };
  constexpr uint32_t kNumCandidates = std::size(kCandidates);
  constexpr uint32_t kMinBatchSize = kCandidates[0].max_batch_size;
  static_assert(kCandidates[0].threshold == kPlanMax2PassLength);
  static_assert(kCandidates[kNumCandidates - 1].threshold == kPlanMaxSupportedLength);

  __shared__ uint32_t s_count;  // final N after compaction
  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;

  const auto tx = threadIdx.x;
  if (tx == 0) s_count = 0;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  __syncthreads();

  // --- Phase 1: decide threshold ------------------------------------------
  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else if (batch_size <= kMinBatchSize) {
    if (tx == 0) s_threshold = kPlanMax2PassLength;  // always prefer cluster
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) {
        atomicAdd(&s_counts[count - 1], 1);
      }
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kPlanMaxSupportedLength;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  const auto cluster_threshold = max(s_threshold, kPlanMax2PassLength);

  // --- Phase 2: compact items with seq_len > threshold into metadata[1..] --
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl, false};
    }
  }
  __syncthreads();
  const auto N = s_count;

  // --- Phase 3: has_next + sentinels + GlobalMetadata ---------------------
  for (uint32_t i = tx; i < N; i += kBlockSize) {
    if (i + kNumClusters < N) metadata[1 + i].has_next = true;
  }
  if (tx < kNumClusters && tx >= N) metadata[1 + tx] = {0, 0, false};
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {
        .cluster_threshold = cluster_threshold,
        .num_cluster_items = N,
        .reserved = {0, 0},
    };
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

    TensorMatcher({B})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({Bp1, 4})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    if (batch_size <= kNumClusters) return;  // metadata unused in fused path

    const auto device = device_.unwrap();
    LaunchKernel(1, kBlockSize, device)(  //
        topk_plan,
        static_cast<const uint32_t*>(seq_lens.data_ptr()),
        static_cast<PlanItem*>(metadata.data_ptr()),
        batch_size,
        static_cluster_threshold);
  }

  /// `workspace` and `metadata` are accepted to preserve the FFI interface for
  /// the future persistent dispatcher; the current naive dispatch ignores them.
  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      [[maybe_unused]] const tvm::ffi::TensorView workspace,
      [[maybe_unused]] const tvm::ffi::TensorView metadata) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // strided scores
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens, contiguous
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, -1})  // strided page table
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_table);
    TensorMatcher({B, K})  // page_indices output, contiguous
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
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
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .topk = topk,
        .page_bits = page_bits,
    };

    const bool use_cluster = (max_seq_len >= kClusterMinSeqLen) && (batch_size <= kClusterMaxBatch);
    if (use_cluster) {
      LaunchKernel(dim3{batch_size, kClusterSize}, kBlockSize, device)
          .enable_cluster(dim3{1, kClusterSize})
          .enable_pdl(kUsePDL)(topk_cluster_kernel<kUsePDL>, params);
    } else {
      LaunchKernel(batch_size, kBlockSize, device).enable_pdl(kUsePDL)(topk_kernel<kUsePDL>, params);
    }
  }
};

}  // namespace
