// paged_mqa_metadata: batch-size-adaptive dispatch.
//
// Replaces upstream's single-block kernel (grid=1, Phase-3 lane-serial
// advance, O(bs) dependent loads on the critical path) with three internal
// kernels dispatched by batch_size, all sharing the same Phase-1/2 prefix
// sum and a `num_sm + 1`-thread parallel upper_bound for Phase 3.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cub/block/block_scan.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr uint32_t kSplitKV = 256;  // const for both SM90 and SM100

constexpr uint32_t kTinyBlock = 256;
constexpr uint32_t kTinyMax = 64;

constexpr uint32_t kSmallBlock = 256;
constexpr uint32_t kSmallMax = 2048;
constexpr uint32_t kSmallItemsPerThread = 8;
static_assert(kSmallBlock * kSmallItemsPerThread == kSmallMax);

constexpr uint32_t kMBTileSize = 4096;
constexpr uint32_t kMBBlockSize = 1024;
constexpr uint32_t kMBItemsPerThread = 4;
constexpr uint32_t kKernelBThreads = 256;
static_assert(kMBBlockSize * kMBItemsPerThread == kMBTileSize);

struct MetadataParams {
  uint32_t batch_size;
  uint32_t num_sm;
  const uint32_t* __restrict__ context_lens;
  uint32_t* __restrict__ schedule_metadata;
};

// bs <= 64. Warp-0 inclusive scan, 256 B static smem.
__global__ __launch_bounds__(kTinyBlock, 1)  //
    void paged_mqa_metadata_tiny_kernel(const MetadataParams params) {
  __shared__ uint32_t s_prefix[kTinyMax];
  __shared__ uint32_t s_global_sum;

  const uint32_t tx = threadIdx.x;
  const uint32_t bs = params.batch_size;
  const uint32_t num_sm = params.num_sm;

  if (tx < 32) {
    uint32_t running = 0;
#pragma unroll
    for (uint32_t base = 0; base < kTinyMax; base += 32) {
      const uint32_t idx = base + tx;
      uint32_t v = 0;
      if (idx < bs) {
        const uint32_t length = params.context_lens[idx];
        v = (length + kSplitKV - 1) >> 8;
      }
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        uint32_t y = __shfl_up_sync(0xffffffff, v, o);
        if (tx >= static_cast<uint32_t>(o)) v += y;
      }
      v += running;
      if (idx < bs) s_prefix[idx] = v;
      running = __shfl_sync(0xffffffff, v, 31);
    }
    if (tx == 0) s_global_sum = running;
  }
  __syncthreads();

  const uint32_t global_sum = s_global_sum;
  const uint32_t avg = global_sum / num_sm;
  const uint32_t ret = global_sum % num_sm;

  // Stride loop so num_sm > blockDim.x - 1 is fully written.
  for (uint32_t i = tx; i <= num_sm; i += blockDim.x) {
    const uint32_t target = i * avg + ::min(i, ret);

    uint32_t lo = 0;
    uint32_t hi = bs;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) >> 1;
      if (s_prefix[mid] <= target)
        lo = mid + 1;
      else
        hi = mid;
    }
    const uint32_t q = lo;

    if (q >= bs) {
      params.schedule_metadata[2 * i + 0] = bs;
      params.schedule_metadata[2 * i + 1] = 0;
    } else {
      const uint32_t prefix_prev = (q == 0) ? 0u : s_prefix[q - 1];
      params.schedule_metadata[2 * i + 0] = q;
      params.schedule_metadata[2 * i + 1] = target - prefix_prev;
    }
  }
}

// 64 < bs <= 2048. CUB BlockScan, 8 KB static smem.
__global__ __launch_bounds__(kSmallBlock, 1)  //
    void paged_mqa_metadata_small_kernel(const MetadataParams params) {
  using BlockScan = cub::BlockScan<uint32_t, kSmallBlock, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ uint32_t s_prefix[kSmallMax];
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ uint32_t s_global_sum;

  const uint32_t tx = threadIdx.x;
  const uint32_t bs = params.batch_size;
  const uint32_t num_sm = params.num_sm;

  uint32_t thread_items[kSmallItemsPerThread];
#pragma unroll
  for (uint32_t k = 0; k < kSmallItemsPerThread; ++k) {
    const uint32_t i = tx * kSmallItemsPerThread + k;
    if (i < bs) {
      const uint32_t length = params.context_lens[i];
      thread_items[k] = (length + kSplitKV - 1) >> 8;
    } else {
      thread_items[k] = 0;
    }
  }

  uint32_t block_aggregate;
  BlockScan(temp_storage).InclusiveSum(thread_items, thread_items, block_aggregate);

  if (tx == 0) s_global_sum = block_aggregate;

#pragma unroll
  for (uint32_t k = 0; k < kSmallItemsPerThread; ++k) {
    const uint32_t i = tx * kSmallItemsPerThread + k;
    if (i < bs) s_prefix[i] = thread_items[k];
  }
  __syncthreads();

  const uint32_t global_sum = s_global_sum;
  const uint32_t avg = global_sum / num_sm;
  const uint32_t ret = global_sum % num_sm;

  // Stride loop so num_sm > blockDim.x - 1 is fully written.
  for (uint32_t i = tx; i <= num_sm; i += blockDim.x) {
    const uint32_t target = i * avg + ::min(i, ret);

    uint32_t lo = 0;
    uint32_t hi = bs;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) >> 1;
      if (s_prefix[mid] <= target)
        lo = mid + 1;
      else
        hi = mid;
    }
    const uint32_t q = lo;

    if (q >= bs) {
      params.schedule_metadata[2 * i + 0] = bs;
      params.schedule_metadata[2 * i + 1] = 0;
    } else {
      const uint32_t prefix_prev = (q == 0) ? 0u : s_prefix[q - 1];
      params.schedule_metadata[2 * i + 0] = q;
      params.schedule_metadata[2 * i + 1] = target - prefix_prev;
    }
  }
}

// bs > 2048, Phase 1: ceil(bs / kMBTileSize) blocks each emit an in-tile
// inclusive prefix into scratch_prefix and a per-tile sum into tile_sums.
__global__ __launch_bounds__(kMBBlockSize, 1)  //
    void phase1_tile_scan_kernel(
        const MetadataParams params, uint32_t* __restrict__ scratch_prefix, uint32_t* __restrict__ tile_sums) {
  using TileBlockScan = cub::BlockScan<uint32_t, kMBBlockSize, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename TileBlockScan::TempStorage temp_storage;

  const uint32_t bs = params.batch_size;
  const uint32_t tile_idx = blockIdx.x;
  const uint32_t tile_base = tile_idx * kMBTileSize;
  const uint32_t tx = threadIdx.x;

  uint32_t thread_items[kMBItemsPerThread];
#pragma unroll
  for (uint32_t k = 0; k < kMBItemsPerThread; ++k) {
    const uint32_t i = tile_base + tx * kMBItemsPerThread + k;
    if (i < bs) {
      const uint32_t length = params.context_lens[i];
      thread_items[k] = (length + kSplitKV - 1) >> 8;
    } else {
      thread_items[k] = 0;
    }
  }

  uint32_t block_aggregate;
  TileBlockScan(temp_storage).InclusiveSum(thread_items, thread_items, block_aggregate);

#pragma unroll
  for (uint32_t k = 0; k < kMBItemsPerThread; ++k) {
    const uint32_t i = tile_base + tx * kMBItemsPerThread + k;
    if (i < bs) scratch_prefix[i] = thread_items[k];
  }

  if (tx == 0) tile_sums[tile_idx] = block_aggregate;
}

// bs > 2048, Phase 2/3: one block, kKernelBThreads threads. Warp-0 scans
// tile_sums into s_tile_prefix; then num_sm+1 threads do tile-level
// upper_bound + within-tile upper_bound to recover (batch_idx, offset).
__global__ __launch_bounds__(kKernelBThreads, 1)  //
    void schedule_from_tiles_kernel(
        const MetadataParams params,
        const uint32_t* __restrict__ scratch_prefix,
        const uint32_t* __restrict__ tile_sums,
        uint32_t num_tiles) {
  extern __shared__ uint32_t s_tile_prefix[];
  __shared__ uint32_t s_global_sum;

  const uint32_t tx = threadIdx.x;
  const uint32_t bs = params.batch_size;
  const uint32_t num_sm = params.num_sm;

  if (tx < 32) {
    uint32_t running = 0;
    for (uint32_t base = 0; base < num_tiles; base += 32) {
      const uint32_t idx = base + tx;
      uint32_t v = (idx < num_tiles) ? tile_sums[idx] : 0;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        uint32_t y = __shfl_up_sync(0xffffffff, v, o);
        if (tx >= static_cast<uint32_t>(o)) v += y;
      }
      v += running;
      if (idx < num_tiles) s_tile_prefix[idx] = v;
      running = __shfl_sync(0xffffffff, v, 31);
    }
    if (tx == 0) s_global_sum = running;
  }
  __syncthreads();

  const uint32_t global_sum = s_global_sum;
  const uint32_t avg = global_sum / num_sm;
  const uint32_t ret = global_sum % num_sm;

  // Stride loop so num_sm > blockDim.x - 1 is fully written. `continue`
  // replaces the original early-out `return` so later strided targets
  // still get processed.
  for (uint32_t i = tx; i <= num_sm; i += blockDim.x) {
    const uint32_t target = i * avg + ::min(i, ret);

    uint32_t t_lo = 0;
    uint32_t t_hi = num_tiles;
    while (t_lo < t_hi) {
      const uint32_t mid = (t_lo + t_hi) >> 1;
      if (s_tile_prefix[mid] <= target)
        t_lo = mid + 1;
      else
        t_hi = mid;
    }
    const uint32_t tile = t_lo;

    if (tile >= num_tiles) {
      params.schedule_metadata[2 * i + 0] = bs;
      params.schedule_metadata[2 * i + 1] = 0;
      continue;
    }

    const uint32_t tile_offset = (tile == 0) ? 0u : s_tile_prefix[tile - 1];
    const uint32_t tile_start = tile * kMBTileSize;
    uint32_t tile_end = tile_start + kMBTileSize;
    if (tile_end > bs) tile_end = bs;
    const uint32_t local_target = target - tile_offset;

    uint32_t lo = tile_start;
    uint32_t hi = tile_end;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) >> 1;
      if (scratch_prefix[mid] <= local_target)
        lo = mid + 1;
      else
        hi = mid;
    }
    const uint32_t q = lo;

    if (q >= bs) {
      params.schedule_metadata[2 * i + 0] = bs;
      params.schedule_metadata[2 * i + 1] = 0;
    } else {
      const uint32_t prefix_prev = (q == tile_start) ? tile_offset : (scratch_prefix[q - 1] + tile_offset);
      params.schedule_metadata[2 * i + 0] = q;
      params.schedule_metadata[2 * i + 1] = target - prefix_prev;
    }
  }
}

struct IndexerMetadataKernel {
  static void run(tvm::ffi::TensorView seq_lens, tvm::ffi::TensorView metadata, tvm::ffi::TensorView workspace) {
    using namespace host;
    auto N = SymbolicSize{"batch_size"};
    auto M = SymbolicSize{"num_sm"};
    auto W = SymbolicSize{"workspace"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens);
    TensorMatcher({M, 2})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(metadata);
    TensorMatcher({W})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(workspace);

    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    const auto num_sm = static_cast<uint32_t>(M.unwrap()) - 1;
    RuntimeCheck(num_sm >= 1 && num_sm <= 1024);

    const auto params = MetadataParams{
        .batch_size = batch_size,
        .num_sm = num_sm,
        .context_lens = static_cast<uint32_t*>(seq_lens.data_ptr()),
        .schedule_metadata = static_cast<uint32_t*>(metadata.data_ptr()),
    };

    const auto dl_device = device.unwrap();

    if (batch_size <= kTinyMax) {
      LaunchKernel(1, kTinyBlock, dl_device)(paged_mqa_metadata_tiny_kernel, params);
    } else if (batch_size <= kSmallMax) {
      LaunchKernel(1, kSmallBlock, dl_device)(paged_mqa_metadata_small_kernel, params);
    } else {
      const auto num_tiles = (batch_size + kMBTileSize - 1) / kMBTileSize;
      const auto required = static_cast<int64_t>(batch_size) + num_tiles;
      RuntimeCheck(static_cast<int64_t>(W.unwrap()) >= required, "workspace too small for multi-block path");
      auto* scratch_prefix = static_cast<uint32_t*>(workspace.data_ptr());
      auto* tile_sums = scratch_prefix + batch_size;

      LaunchKernel(num_tiles, kMBBlockSize, dl_device)(phase1_tile_scan_kernel, params, scratch_prefix, tile_sums);
      const auto kb_smem_bytes = static_cast<size_t>(num_tiles) * sizeof(uint32_t);
      LaunchKernel(1, kKernelBThreads, dl_device, kb_smem_bytes)(
          schedule_from_tiles_kernel, params, scratch_prefix, tile_sums, num_tiles);
    }
  }
};

}  // namespace
