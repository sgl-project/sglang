#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <limits>

namespace sglang {

/// Finalises the block table layer 20 publishes for DeepGEMM's sparse indexer:
/// the top-k block ids a row selected (any order, -1 padded) become, in place,
/// the same ids ascending with INT32_MAX past the row's count, plus each block
/// as a pool slot / 8 (`page_table[b, id / bpp] * bpp + id % bpp`, `bpp` blocks
/// per index page). A row with at most `topk` blocks keeps every block and gets
/// the identity table without reading its input.
///
/// Counting sort over a bitmap of the row's blocks (one bit per block, 16 KiB
/// for the 128K blocks of a 1M-token row): set the selected bits, exclusive-scan
/// the popcounts, emit every set bit at its rank. A word with a single bit is
/// emitted by its owner (one `ffs`, no loop); a word with more goes to a
/// block-wide queue that the warps drain one word per step, one lane per bit,
/// so a dense cluster of selected blocks is spread over all warps.
struct SortConfig {
  static constexpr uint32_t kBlockSize = 1024;
  static constexpr uint32_t kOccupancy = 2;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kBlockTokens = 8;
  static constexpr uint32_t kMaxSeqLen = 128 * 1024;  // blocks: 1M tokens / kBlockTokens
  static constexpr uint32_t kMaxTopK = 2048;
  static constexpr uint32_t kWordsPerThread = kMaxSeqLen / 32 / kBlockSize;
  static_assert(kWordsPerThread == 4 && kNumWarps == device::kWarpThreads);
  static constexpr int32_t kPad = std::numeric_limits<int32_t>::max();
  using word_vec_t = device::AlignedVector<uint32_t, kWordsPerThread>;
  struct WriteItem {
    uint32_t start;  // rank of the word's first bit | word index << 16
    uint32_t bits;
  };
  struct Smem {
    uint32_t queue_size;
    uint32_t warp_sum[kNumWarps];
    union {
      alignas(16) uint32_t bitmap[kMaxSeqLen / 32];
      WriteItem write_queue[kMaxTopK];  // a queued word holds >= 2 of the topk bits
    };
  };
};

struct SortParams {
  const uint32_t* __restrict__ seq_len;    // [rows] tokens
  const int32_t* __restrict__ page_table;  // [rows, pages] index-pool pages
  int32_t* __restrict__ indices;           // [rows, topk] blocks, -1 padded in, ascending + kPad out
  int32_t* __restrict__ out_pages;         // [rows, topk] the same blocks as pool slots / 8
  int64_t page_table_stride;
  int64_t indices_stride;
  int64_t out_pages_stride;
  uint32_t topk;
  uint32_t page_bits;  // log2(page_size / kBlockTokens)
};

/// One CTA per row.
template <bool kUsePDL>
__global__ __launch_bounds__(SortConfig::kBlockSize, SortConfig::kOccupancy)  //
    void sort_128k_transform(const __grid_constant__ SortParams params) {
  using namespace device;
  using C = SortConfig;
  __shared__ C::Smem smem;
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto lanemask_lt = (1u << lane_id) - 1u;

  PDLWaitPrimary<kUsePDL>();  // indices is the block top-k's output
  const auto seq_len = params.seq_len[bx];
  const auto nblocks = (seq_len + C::kBlockTokens - 1) / C::kBlockTokens;
  const auto* __restrict__ table = params.page_table + bx * params.page_table_stride;
  auto* __restrict__ indices = params.indices + bx * params.indices_stride;
  auto* __restrict__ pages = params.out_pages + bx * params.out_pages_stride;
  const auto bpp_mask = (1u << params.page_bits) - 1u;
  const auto emit = [&](uint32_t rank, uint32_t id) {
    indices[rank] = static_cast<int32_t>(id);
    pages[rank] = (table[id >> params.page_bits] << params.page_bits) | static_cast<int32_t>(id & bpp_mask);
  };
  const auto pad = [&](uint32_t rank) {
    indices[rank] = C::kPad;
    pages[rank] = C::kPad;
  };

  if (nblocks <= params.topk) {  // every block is selected: the identity table
    for (uint32_t t = tx; t < params.topk; t += C::kBlockSize) {
      if (t < nblocks) {
        emit(t, t);
      } else {
        pad(t);
      }
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  // 1. the selected blocks as a bitmap
  C::word_vec_t words;
  words.fill(0u);
  words.store(smem.bitmap, tx);
  if (tx == 0) smem.queue_size = 0;
  __syncthreads();
  for (uint32_t t = tx; t < params.topk; t += C::kBlockSize) {
    const auto id = indices[t];
    if (id >= 0) atomicOr(&smem.bitmap[id >> 5], 1u << (id & 31));
  }
  __syncthreads();

  // 2. rank of every word's first bit: block-wide exclusive scan of the popcounts
  words.load(smem.bitmap, tx);
  uint32_t count[C::kWordsPerThread];
  uint32_t local = 0;
#pragma unroll
  for (uint32_t j = 0; j < C::kWordsPerThread; ++j) {
    count[j] = __popc(words[j]);
    local += count[j];
  }
  const auto warp_inc = warp::inclusive_sum(lane_id, local);
  if (lane_id == kWarpThreads - 1) smem.warp_sum[warp_id] = warp_inc;
  __syncthreads();  // also: every thread holds its words, the bitmap may become the queue
  const auto peer_sum = smem.warp_sum[lane_id];
  const auto warp_prefix = warp::reduce_sum(lane_id < warp_id ? peer_sum : 0u);
  const auto total = warp::reduce_sum(peer_sum);
  uint32_t base = warp_prefix + warp_inc - local;
  PDLTriggerSecondary<kUsePDL>();

  // 3. single bits by their owner, denser words queued for the warps
#pragma unroll
  for (uint32_t j = 0; j < C::kWordsPerThread; ++j) {
    const auto word_idx = tx * C::kWordsPerThread + j;
    if (count[j] == 1) {
      emit(base, word_idx * 32 + __ffs(words[j]) - 1);
    } else if (count[j] >= 2) {
      const auto slot = atomicAdd(&smem.queue_size, 1u);
      smem.write_queue[slot] = {base | (word_idx << 16), words[j]};
    }
    base += count[j];
  }
  for (uint32_t t = total + tx; t < params.topk; t += C::kBlockSize) {
    pad(t);
  }
  __syncthreads();

  // 4. drain the queue: one word per warp step, one lane per bit
  const auto queue_size = smem.queue_size;
  for (uint32_t q = warp_id; q < queue_size; q += C::kNumWarps) {
    const auto item = smem.write_queue[q];
    if ((item.bits >> lane_id) & 1u) {
      emit((item.start & 0xFFFFu) + __popc(item.bits & lanemask_lt), (item.start >> 16) * 32 + lane_id);
    }
  }
}

/// The page transform alone, for a block top-k that already emits its ids
/// ascending (e.g. DeepSelect with `sorted_index`): `out_pages[t]` is the pool
/// slot / 8 of `indices[t]` for `t < min(topk, ceil(seq_len / 8))`, INT32_MAX
/// past that; `indices` is left as it is. Those first entries must be valid
/// block ids of the row.
template <bool kUsePDL>
__global__ __launch_bounds__(SortConfig::kBlockSize, SortConfig::kOccupancy)  //
    void page_transform_128k(const __grid_constant__ SortParams params) {
  using namespace device;
  using C = SortConfig;
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();
  const auto seq_len = params.seq_len[bx];
  const auto nblocks = (seq_len + C::kBlockTokens - 1) / C::kBlockTokens;
  const auto num_valid = min(nblocks, params.topk);
  const auto* __restrict__ table = params.page_table + bx * params.page_table_stride;
  const auto* __restrict__ indices = params.indices + bx * params.indices_stride;
  auto* __restrict__ pages = params.out_pages + bx * params.out_pages_stride;
  const auto bpp_mask = (1u << params.page_bits) - 1u;
  for (uint32_t t = tx; t < params.topk; t += C::kBlockSize) {
    if (t < num_valid) {
      const auto id = static_cast<uint32_t>(indices[t]);
      pages[t] = (table[id >> params.page_bits] << params.page_bits) | static_cast<int32_t>(id & bpp_mask);
    } else {
      pages[t] = C::kPad;
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

/// Host entry: `indices` is rewritten in place; `page_size` is the index pool's,
/// a power of two >= 8, and the row's page table must cover its length.
template <bool kPDL>
struct SortIdxKernel {
  /// Sort + page transform, in place on `indices`.
  static void transform(
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out_pages,
      const uint32_t page_size) {
    launch<sort_128k_transform<kPDL>>(indices, seq_lens, page_table, out_pages, page_size);
  }

  /// Page transform only, `indices` already ascending and left untouched.
  static void transform_pages(
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out_pages,
      const uint32_t page_size) {
    launch<page_transform_128k<kPDL>>(indices, seq_lens, page_table, out_pages, page_size);
  }

 private:
  template <auto kKernel>
  static void launch(
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out_pages,
      const uint32_t page_size) {
    using namespace host;
    using C = SortConfig;
    auto B = SymbolicSize{"batch_size"};
    auto K = SymbolicSize{"topk_blocks"};
    auto Si = SymbolicSize{"indices_stride"};
    auto Sp = SymbolicSize{"out_pages_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({B, K}).with_strides({Si, 1}).with_dtype<int32_t>().with_device(device_).verify(indices);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({B, K}).with_strides({Sp, 1}).with_dtype<int32_t>().with_device(device_).verify(out_pages);
    RuntimeCheck(
        std::has_single_bit(page_size) && page_size >= C::kBlockTokens,
        "page_size must be a power of two of at least 8");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= C::kMaxTopK, "topk_blocks must be in (0, kMaxTopK]");
    const auto params = SortParams{
        .seq_len = static_cast<const uint32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .out_pages = static_cast<int32_t*>(out_pages.data_ptr()),
        .page_table_stride = page_table.stride(0),
        .indices_stride = Si.unwrap(),
        .out_pages_stride = Sp.unwrap(),
        .topk = topk,
        .page_bits = static_cast<uint32_t>(std::countr_zero(page_size / C::kBlockTokens)),
    };
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), C::kBlockSize, device_.unwrap())
        .config({.use_pdl = kPDL})
        .launch(kKernel, params);
  }
};

}  // namespace sglang
