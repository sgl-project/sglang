#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/topk/cluster.cuh>
#include <sgl_kernel/deepseek_v4/topk/register.cuh>
#include <sgl_kernel/deepseek_v4/topk/streaming.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>

#include <cfloat>
#include <cstdint>
#include <iterator>

namespace {

#ifndef SGL_TOPK
#define SGL_TOPK 512
#endif

inline constexpr uint32_t K = SGL_TOPK;

template <auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once(host::DebugInfo where = {}) {
  [[maybe_unused]]
  static const auto result = [] {
    const auto fptr = std::bit_cast<const void*>(f);
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
  host::RuntimeDeviceCheck(result, where);
}

namespace impl = device::top512;
using Large = impl::ClusterTopK<K>;
using Medium = impl::StreamingTopK<K>;
using Small = impl::RegisterTopK<K>;

using Metadata = Large::Metadata;
constexpr uint32_t kBlockSize = impl::kBlockSize;
constexpr uint32_t kNumClusters = 15;  // based on hardware limits
constexpr uint32_t kClusterSize = Large::kClusterSize;
constexpr uint32_t kMax2PassLength = Small::kMax2PassLength;
constexpr uint32_t kMaxSupportedLength = Large::kMaxLength;

/// Common metadata lives at metadata[0] (first row of the [batch_size+1, 4] tensor).
/// Per-item metadata starts at metadata[1..batch_size]. The plan kernel writes both.
struct alignas(16) GlobalMetadata {
  uint32_t cluster_threshold;  // decided per-batch in plan kernel
  uint32_t num_cluster_items;  // N = number of items routed to the cluster path
  uint32_t reserved[2];
};
static_assert(sizeof(GlobalMetadata) == sizeof(Metadata), "layout: row 0 must occupy one Metadata-sized slot");

// optimize occupancy for prefill
#define SMALL_TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, 2)
// cluster at y dim
#define LARGE_CLUSTER __cluster_dims__(1, kClusterSize, 1)
// stage-1 is persistent cluster, and shared memory usage is huge (can not 2)
#define LARGE_TOPK_STAGE_1 __global__ __launch_bounds__(kBlockSize, 1) LARGE_CLUSTER
// stage-2 is non-persistent non-cluster, with less shared memory and higher occupancy
#define LARGE_TOPK_STAGE_2 __global__ __launch_bounds__(kBlockSize, 2)
// fused into 1 stage when batch-size <= kNumPersistentClusters
#define FUSED_COMBINE_KERNEL __global__ __launch_bounds__(kBlockSize, 1) LARGE_CLUSTER
// plan runs once as a single block before the combine kernels
#define PLAN_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

struct TopKParams {
  const uint32_t* __restrict__ seq_lens;
  const float* __restrict__ scores;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint8_t* __restrict__ workspace;  // [batch, kWorkspaceBytes] -- internally allocated
  /// Pointer to the full metadata tensor: metadata[0] is GlobalMetadata, metadata[1..]
  /// are per-item entries (at most kNumClusters * rounds of them).
  const Metadata* __restrict__ metadata = nullptr;
  int64_t workspace_stride;  // bytes per batch
  uint32_t batch_size;
  uint32_t page_bits;

  SGL_DEVICE const float* get_scores(const uint32_t batch_id) const {
    return scores + batch_id * score_stride;
  }
  SGL_DEVICE impl::TransformParams get_transform(const uint32_t batch_id, int32_t* indices) const {
    return {
        .page_table = page_table + batch_id * page_table_stride,
        .indices_in = indices,
        .indices_out = page_indices + batch_id * K,
        .page_bits = page_bits,
    };
  }
  SGL_DEVICE const GlobalMetadata& get_global_metadata() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE const Metadata& get_item_metadata(uint32_t work_id) const {
    return metadata[1 + work_id];  // +1 to skip the GlobalMetadata row
  }
};

SGL_DEVICE uint2 partition_work(uint32_t length, uint32_t rank) {
  constexpr uint32_t kTMAAlign = 4;
  const auto total_units = (length + kTMAAlign - 1) / kTMAAlign;
  const auto base = total_units / kClusterSize;
  const auto extra = total_units % kClusterSize;
  const auto local_units = base + (rank < extra ? 1u : 0u);
  const auto offset_units = rank * base + min(rank, extra);
  const auto offset = offset_units * kTMAAlign;
  const auto finish = min(offset + local_units * kTMAAlign, length);
  return {offset, finish - offset};
}

/// Persistent scheduler. A single block:
///  1. Decides a cluster_threshold from the real seq_lens distribution (or
///     uses the caller-supplied `static_cluster_threshold` when non-zero).
///  2. Writes that threshold + N into metadata[0] (the GlobalMetadata row).
///  3. Compacts items with seq_len > threshold into metadata[1..N+1), laid out
///     to match the persistent consumer's round-robin stride (kNumClusters).
///     Entries for clusters that get no work are zero-filled.
PLAN_KERNEL void topk_plan(
    const uint32_t* __restrict__ seq_lens,
    Metadata* __restrict__ metadata,
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
  // Candidate thresholds, strictly increasing. Picked to give the auto-heuristic
  // reasonable granularity without needing a full sort. Must all be >= kMax2PassLength.

  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  /// NOTE: only tuned on B200
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
  static_assert(kCandidates[0].threshold == kMax2PassLength);
  static_assert(kCandidates[kNumCandidates - 1].threshold == kMaxSupportedLength);

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
    if (tx == 0) s_threshold = kMax2PassLength;  // always prefer cluster
  } else {
    // Count items above each candidate threshold. Monotonically non-increasing in T.
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      assert(sl <= kMaxSupportedLength);
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
      uint32_t chosen = kMaxSupportedLength;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];
        /// NOTE: `accum` increasing, while `max_batch_size` decreasing
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  // sanity check: below 2 pass threshold, must fits in small path
  const auto cluster_threshold = max(s_threshold, kMax2PassLength);

  // --- Phase 2: compact items with seq_len > threshold into metadata[1..] -
  // Per-item rows live at metadata[1 + pos]; metadata[0] is the GlobalMetadata row.
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
  // Zero-fill the first kNumClusters sentinel slots that got no valid entry.
  if (tx < kNumClusters && tx >= N) metadata[1 + tx] = {0, 0, false};
  // Write global metadata (row 0).
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {
        .cluster_threshold = cluster_threshold,
        .num_cluster_items = N,
        .reserved = {0, 0},
    };
  }
}

SMALL_TOPK_KERNEL void  // short context
topk_short_transform(const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[K];
  const auto batch_id = blockIdx.x;
  const auto seq_len = params.seq_lens[batch_id];
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  // trivial case
  if (seq_len <= K) {
    impl::trivial_transform(transform, seq_len, K);
  } else {
    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem, /*use_pdl=*/true);
    device::PDLTriggerSecondary<true>();
    Small::transform(transform);
  }
}

LARGE_TOPK_STAGE_1 void  // long context, middle to large batch size
topk_combine_preprocess(const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[K];
  uint32_t work_id = blockIdx.x;
  uint32_t batch_id;
  uint32_t seq_len;
  bool has_next;
  uint32_t length;
  uint32_t offset;
  const auto cluster_rank = blockIdx.y;

  const auto prefetch_metadata = [&] {
    const auto metadata = params.get_item_metadata(work_id);
    batch_id = metadata.batch_id;
    seq_len = metadata.seq_len;
    has_next = metadata.has_next;
    work_id += kNumClusters;  // advance to the next item for this cluster
  };
  const auto launch_prologue = [&] {
    const auto partition = partition_work(seq_len, cluster_rank);
    offset = partition.x;
    length = partition.y;
    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
  };

  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();

  prefetch_metadata();
  if (seq_len == 0) return;
  Large::stage1_init(smem);
  launch_prologue();
  while (true) {
    const auto this_length = length;
    const auto this_offset = offset;
    const auto need_prefetch = has_next;
    const auto transform = params.get_transform(batch_id, s_topk_indices);
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    if (need_prefetch) prefetch_metadata();
    Large::stage1(s_topk_indices, this_length, smem, /*reuse=*/true);
    if (need_prefetch) launch_prologue();
    Large::stage1_epilogue(transform, this_offset, ws, smem);
    if (!need_prefetch) break;
  }
}

LARGE_TOPK_STAGE_2 void  // long context, middle to large batch size
topk_combine_transform(const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[K];
  const auto batch_id = blockIdx.x;
  const auto seq_len = params.seq_lens[batch_id];
  const auto cluster_threshold = params.get_global_metadata().cluster_threshold;
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  if (seq_len <= K) {
    impl::trivial_transform(transform, seq_len, K);
  } else if (seq_len <= kMax2PassLength) {
    if (seq_len <= Small::kMax1PassLength) {
      Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
    } else {
      __syncwarp();
      Small::run<true>(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
    }
    Small::transform(transform);
  } else if (seq_len <= cluster_threshold) {
    Medium::run(params.get_scores(batch_id), seq_len, s_topk_indices, smem);
    Medium::transform(transform, smem);
  } else {
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    device::PDLWaitPrimary<true>();
    Large::transform(transform, ws, smem);
  }
}

FUSED_COMBINE_KERNEL void  // long context, small batch size
topk_fused_transform(const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  __shared__ int32_t s_topk_indices[K];
  const auto batch_id = blockIdx.x;
  const auto cluster_rank = blockIdx.y;
  const auto seq_len = params.seq_lens[batch_id];
  const auto transform = params.get_transform(batch_id, s_topk_indices);
  if (seq_len <= K) {
    if (cluster_rank != 0) return;  // only first rank work
    impl::trivial_transform(transform, seq_len, K);
  } else if (seq_len <= Small::kMax1PassLength) {
    if (cluster_rank != 0) return;  // only first rank work
    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem, /*use_pdl=*/true);
    Small::transform(transform);
  } else {
    const auto [offset, length] = partition_work(seq_len, cluster_rank);
    const auto ws = params.workspace + batch_id * params.workspace_stride;
    Large::stage1_init(smem);
    device::PDLWaitPrimary<true>();
    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
    Large::stage1(s_topk_indices, length, smem);
    Large::stage1_epilogue(transform, offset, ws, smem);
    cooperative_groups::this_cluster().sync();
    if (cluster_rank != 0) return;  // only first rank do the stage-2
    Large::transform(transform, ws, smem);
  }
}

struct CombinedTopKKernel {
  static constexpr auto kStage1SMEM = sizeof(Large::Smem) + 128;
  static constexpr auto kStage2SMEM = std::max(sizeof(Small::Smem), sizeof(Medium::Smem)) + 128;

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
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1);
    if (batch_size <= kNumClusters) return;  // metadata unused in fused path

    const auto device = device_.unwrap();
    constexpr auto kernel = topk_plan;
    LaunchKernel(1, kBlockSize, device)(  //
        kernel,
        static_cast<uint32_t*>(seq_lens.data_ptr()),
        static_cast<Metadata*>(metadata.data_ptr()),
        batch_size,
        static_cluster_threshold);
  }

  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::TensorView workspace,
      const tvm::ffi::TensorView metadata) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto W = SymbolicSize{"workspace_stride"};
    constexpr auto D = Large::kWorkspaceInts;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, L})  //
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, -1})  //
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_table);
    TensorMatcher({B, K})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);
    TensorMatcher({B, D})  //
        .with_strides({W, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(workspace);
    TensorMatcher({Bp1, 4})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto max_seq_len = static_cast<uint32_t>(L.unwrap());
    const auto device = device_.unwrap();
    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (TMA 16-byte alignment)");
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");

    // NOTE: this should be fixed later
    // RuntimeCheck(max_seq_len <= kMaxSupportedLength, max_seq_len, " exceeds the maximum supported length");

    const auto params = TopKParams{
        .seq_lens = static_cast<uint32_t*>(seq_lens.data_ptr()),
        .scores = static_cast<float*>(scores.data_ptr()),
        .page_table = static_cast<int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .workspace = static_cast<uint8_t*>(workspace.data_ptr()),
        .metadata = static_cast<const Metadata*>(metadata.data_ptr()),
        .workspace_stride = W.unwrap() * static_cast<int64_t>(sizeof(int32_t)),
        .batch_size = batch_size,
        .page_bits = page_bits,
    };

    if (max_seq_len <= Small::kMax1PassLength) {
      // All items fit in the short path -- no stage-1 needed
      constexpr auto kernel = topk_short_transform;
      setup_kernel_smem_once<kernel, kStage2SMEM>();
      LaunchKernel(batch_size, kBlockSize, device, kStage2SMEM)  //
          .enable_pdl(true)(kernel, params);
    } else {
      // Some items may be large -- launch stage-1 + main
      if (batch_size <= kNumClusters) {
        // can fuse into 1 stage
        constexpr auto kernel = topk_fused_transform;
        constexpr auto kSMEM = std::max(kStage1SMEM, kStage2SMEM);
        setup_kernel_smem_once<kernel, kSMEM>();
        LaunchKernel({batch_size, kClusterSize}, kBlockSize, device, kSMEM)
            .enable_cluster({1, kClusterSize})
            .enable_pdl(true)(kernel, params);
      } else {
        // stage 1 + stage 2
        constexpr auto kernel_stage_1 = topk_combine_preprocess;
        setup_kernel_smem_once<kernel_stage_1, kStage1SMEM>();
        const auto num_clusters = std::min(batch_size, kNumClusters);
        LaunchKernel({num_clusters, kClusterSize}, kBlockSize, device, kStage1SMEM)
            .enable_cluster({1, kClusterSize})
            .enable_pdl(true)(kernel_stage_1, params);
        constexpr auto kernel_stage_2 = topk_combine_transform;
        setup_kernel_smem_once<kernel_stage_2, kStage2SMEM>();
        LaunchKernel(batch_size, kBlockSize, device, kStage2SMEM)  //
            .enable_pdl(true)(kernel_stage_2, params);
      }
    }
  }
};

}  // namespace
