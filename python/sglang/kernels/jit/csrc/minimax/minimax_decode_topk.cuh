#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>
#if defined(__HIP_PLATFORM_AMD__)
static constexpr unsigned long long kWarpSyncMask = 0xFFFFFFFFFFFFFFFFull;
#else
#include <math_constants.h>
static constexpr unsigned int kWarpSyncMask = 0xFFFFFFFFu;
#endif

namespace {

// Block top-k selection over a per-(head, batch) row of block scores, run by one
// CTA of TopKTrait::kCTASize threads. Picks the `topk` highest-scoring block ids
// (k_eff = min(topk, num_blocks)). Three size regimes, chosen by num_blocks:
//   * <= kSmallThreshold : O(n^2) rank-by-compare (no radix).
//   * <= kCTASize        : 4-pass 8-bit radix, one element per thread in a reg.
//   * <= kMaxNumBlocks   : 4-pass 8-bit radix, kIters elements per thread cached
//                          in registers (row read from global exactly once);
//                          liveness is a uint32_t bitmask, selection is an
//                          in-loop scatter -- nothing is cached in shared memory.
// The trivial case num_blocks <= topk (every block selected) is handled by the
// kernels below, outside the Trait.
struct TopKTrait {
  static constexpr uint32_t kMaxTopK = 32;
  static constexpr uint32_t kCTASize = 512;
  static constexpr uint32_t kNumWarps = kCTASize / device::kWarpThreads;
  static constexpr uint32_t kMaxNumBlocks = 4096;  // block topk
  static constexpr uint32_t kSmallThreshold = 8 * kNumWarps;
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadixSize = 1 << kRadixBits;
  static constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  struct Smem {
    uint32_t warp_sum[kNumWarps];
    alignas(128) uint32_t counter;
    alignas(128) uint32_t counter_final;
    alignas(128) uint32_t threshold_bin;
    uint32_t equal_count;
    uint32_t above_count;
    uint32_t histogram[2][kRadixSize];    // 8 bit radix
    float small_scores[kSmallThreshold];  // small (O(n^2)) path only
  };

  SGL_DEVICE static void forward(
      const float* __restrict__ scores,
      const uint32_t num_blocks,
      int32_t* __restrict__ topk_out,
      const uint32_t topk,
      Smem* smem) {
    using namespace device;
    const auto tx = threadIdx.x;
    __builtin_assume(tx < kCTASize);
    const auto warp_id = tx / kWarpThreads;
    const auto lane_id = tx % kWarpThreads;

    constexpr auto is_greater = [](float x, float y, int32_t delta) {
      return (x > y) || ((x == y) && delta < 0);  // lower block id wins
    };
    constexpr auto warp_inclusive_sum = [](uint32_t lane_id, uint32_t val) {
#pragma unroll
      for (uint32_t offset = 1; offset < 32; offset *= 2) {
        uint32_t n = __shfl_up_sync(kWarpSyncMask, val, offset, 32);
        if (lane_id >= offset) val += n;
      }
      return val;
    };
    constexpr auto clip_nan = [](float x) { return x != x ? kNegInf : x; };
    constexpr auto score_to_key = [](float x) {
      uint32_t b = __float_as_uint(x);
      return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    };
    // Find the radix bin holding the topk_remain-th largest of `total_active`
    // elements currently counted in `histogram`. Writes threshold_bin (the bin),
    // above_count (elements strictly above it), equal_count (elements in it).
    const auto find_threshold = [&](uint32_t* histogram, uint32_t total_active, uint32_t topk_remain) {
      using namespace device;
      uint32_t hist_val = 0;
      uint32_t warp_inc = 0;
      if (tx < kRadixSize) {
        hist_val = histogram[tx];
        warp_inc = warp_inclusive_sum(lane_id, hist_val);
        if (lane_id == kWarpThreads - 1) smem->warp_sum[warp_id] = warp_inc;
      }
      __syncthreads();
      if (tx < kRadixSize) {
        const auto inter = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
        const auto prefix = inter + warp_inc;      // count in bins [0, tx]
        const auto above = total_active - prefix;  // count in bins ABOVE tx
        if (above < topk_remain && above + hist_val >= topk_remain) {
          smem->threshold_bin = tx;
          smem->above_count = above;
          smem->equal_count = hist_val;
        }
      }
      __syncthreads();
    };

    if (num_blocks <= kSmallThreshold) {
      // O(n^2) compare: each block's rank = #blocks that outrank it; the ones
      // with rank < topk are selected (rank is its position in topk_out).
      static_assert(kSmallThreshold <= kCTASize);
      if (tx < num_blocks) smem->small_scores[tx] = clip_nan(scores[tx]);
      __syncthreads();
      constexpr uint32_t kNumCandidates = kSmallThreshold / kNumWarps;
      constexpr uint32_t kNumTargets = kSmallThreshold / kWarpThreads;
      float candidates[kNumCandidates];
      float target[kNumTargets];
#pragma unroll
      for (uint32_t i = 0; i < kNumTargets; ++i) {
        const auto idx = lane_id + i * kWarpThreads;
        target[i] = (idx < num_blocks) ? smem->small_scores[idx] : kNegInf;
      }
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto idx = warp_id + i * kNumWarps;
        candidates[i] = (idx < num_blocks) ? smem->small_scores[idx] : kNegInf;
      }

#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const int32_t idx = warp_id + i * kNumWarps;
        if (idx >= static_cast<int32_t>(num_blocks)) break;
        uint32_t rank = 0;
#pragma unroll
        for (uint32_t j = 0; j < kNumTargets; ++j) {
          const int32_t delta = lane_id + j * kWarpThreads - idx;
          // partial rank = how many of this lane's targets outrank the candidate
          rank += is_greater(target[j], candidates[i], delta);
        }
        // full rank = sum of the per-lane partial ranks across the warp
        rank = warp::reduce_sum(rank);
        if (rank < topk) topk_out[rank] = idx;
      }
    } else if (num_blocks <= kCTASize) {
      // 4-pass 8-bit radix select, one element per thread held in a register.
      bool active = tx < num_blocks;
      const auto value = active ? clip_nan(scores[tx]) : kNegInf;
      const auto key = score_to_key(value);
      uint32_t topk_remain = topk;
      uint32_t write_pos = topk;  // sentinel: not selected
      if (tx < kRadixSize) smem->histogram[0][tx] = 0;
      if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
      __syncthreads();
      uint32_t total_active = num_blocks;

#pragma unroll
      for (int round = 0; round < 4; round++) {
        const uint32_t shift = 24 - round * 8;
        const uint32_t bin = (key >> shift) & 0xFFu;
        const auto hist_idx = round % 2;
        const auto histogram = smem->histogram[hist_idx];

        if (active) atomicAdd(&histogram[bin], 1);
        if (round < 3 && tx < kRadixSize) smem->histogram[hist_idx ^ 1][tx] = 0;
        __syncthreads();

        find_threshold(histogram, total_active, topk_remain);

        const auto threshold_bin = smem->threshold_bin;
        const auto above_count = smem->above_count;
        const auto equal_count = smem->equal_count;

        if (round < 3) total_active = equal_count;
        topk_remain -= above_count;

        // scatter: above -> selected now; equal at the last pass -> keep the rest
        if (active) {
          if (bin > threshold_bin) {
            write_pos = atomicAdd(&smem->counter, 1);
            active = false;
          } else if (bin < threshold_bin) {
            active = false;
          } else if (round == 3) {
            write_pos = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
          }
          // bin == threshold && round < 3: stay active for the next pass
        }

        if (round == 3 || topk_remain == 0) break;
      }

      if (write_pos < topk) topk_out[write_pos] = tx;
    } else {
      // num_blocks in (kCTASize, kMaxNumBlocks]: each thread caches its (up to
      // kIters) slice of the row in registers -- read from global exactly ONCE --
      // then runs the same 4-pass radix select as the single-element path looped
      // over those slots. Liveness is a uint32_t bitmask (bit i = slot i still in
      // the running set), so there is no per-element flag array; selection is an
      // in-loop scatter, so there is no per-element position array. Nothing is
      // cached in shared memory beyond the histogram.
      constexpr uint32_t kIters = kMaxNumBlocks / kCTASize;
      static_assert(kIters <= 32, "active liveness is packed into a uint32_t");
      uint32_t key[kIters];
      uint32_t active = 0;
#pragma unroll
      for (uint32_t i = 0; i < kIters; ++i) {
        const uint32_t idx = i * kCTASize + tx;
        if (idx < num_blocks) {
          key[i] = score_to_key(clip_nan(scores[idx]));
          active |= 1u << i;
        }
      }
      if (tx < kRadixSize) smem->histogram[0][tx] = 0;
      if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
      __syncthreads();

      uint32_t topk_remain = topk;
      uint32_t total_active = num_blocks;

#pragma unroll
      for (int round = 0; round < 4; ++round) {
        const uint32_t shift = 24 - round * 8;
        const auto hb = round & 1;

#pragma unroll
        for (uint32_t i = 0; i < kIters; ++i)
          if (active & (1u << i)) atomicAdd(&smem->histogram[hb][(key[i] >> shift) & 0xFFu], 1);
        if (round < 3 && tx < kRadixSize) smem->histogram[hb ^ 1][tx] = 0;
        __syncthreads();

        find_threshold(smem->histogram[hb], total_active, topk_remain);
        const auto threshold_bin = smem->threshold_bin;
        const auto above_count = smem->above_count;
        const auto equal_count = smem->equal_count;

        if (round < 3) total_active = equal_count;
        topk_remain -= above_count;

#pragma unroll
        for (uint32_t i = 0; i < kIters; ++i) {
          if (active & (1u << i)) {
            const uint32_t bin = (key[i] >> shift) & 0xFFu;
            if (bin > threshold_bin) {
              topk_out[atomicAdd(&smem->counter, 1)] = i * kCTASize + tx;
              active &= ~(1u << i);
            } else if (bin < threshold_bin) {
              active &= ~(1u << i);
            } else if (round == 3) {
              const auto pos = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
              if (pos < topk) topk_out[pos] = i * kCTASize + tx;
            }
            // bin == threshold && round < 3: slot stays live for the next pass
          }
        }

        if (round == 3 || topk_remain == 0) break;
      }
    }
  }
};

// -------------------------------------------------------------------------
// Kernels: one CTA (kCTASize threads) per (head, batch) row. The trivial case
// num_blocks <= topk (every block selected) is special-judged here, outside the
// Trait; otherwise the Trait selects the top-k block ids.
// -------------------------------------------------------------------------

// Block-id output: topk_idx[h, b, 0:k_eff) = selected block ids (front-packed,
// unordered), [k_eff:topk) = -1.
template <typename SeqLenT, bool kUsePDL>
__global__ void minimax_decode_topk_block_kernel(
    const float* __restrict__ score,
    const SeqLenT* __restrict__ seq_lens,
    int32_t* __restrict__ topk_idx,
    int batch,
    int num_heads,
    int max_seqblock,
    int block_size,
    int topk) {
  const int b = blockIdx.x;  // grid.x = batch
  const int h = blockIdx.y;  // grid.y = num_heads
  const int tx = threadIdx.x;

  // seq_lens is from an earlier kernel; prefetch it (and the cheap setup) before
  // waiting on the score producer so the prologue overlaps its tail (PDL).
  const int64_t seq_len = static_cast<int64_t>(seq_lens[b]);
  const int num_blocks_raw = static_cast<int>((seq_len + block_size - 1) / block_size);
  // Never scan past the materialized score columns.
  const int num_blocks = num_blocks_raw < max_seqblock ? num_blocks_raw : max_seqblock;
  int32_t* __restrict__ out = topk_idx + (static_cast<int64_t>(h) * batch + b) * topk;
  device::PDLWaitPrimary<kUsePDL>();

  if (num_blocks <= topk) {  // trivial: identity, -1 padded
    for (int i = tx; i < topk; i += TopKTrait::kCTASize)
      out[i] = (i < num_blocks) ? i : -1;
    return;
  }

  const float* __restrict__ row = score + (static_cast<int64_t>(h) * batch + b) * max_seqblock;
  __shared__ TopKTrait::Smem smem;
  TopKTrait::forward(row, static_cast<uint32_t>(num_blocks), out, static_cast<uint32_t>(topk), &smem);
}

// Page-table output: for each (batch b, kv-head h) pseudo-request emit the
// trtllm/fa3 page table -- selected blocks sorted ascending (so the final partial
// block's pages land last), each expanded to its ppb = block_size/page_size pages
// via req_to_token -- plus the effective KV length seq_lens_out.
//
// DP attention (num_kv_heads > 1): each kv head selects its OWN blocks, so the
// per-request page table can't be shared across heads. We flatten (b, h) into
// num_heads*batch pseudo-requests laid out batch-major (row = b*num_heads + h,
// matching q.view(bs, nkv, gqa, d).reshape(bs*nkv, gqa, d)). seq_lens / slot_ids /
// req_to_token are per-batch (head-independent: a token's cache slot is the same
// for every head). The page index is head-encoded (head-minor) as
// base_page*num_heads + h, which is exactly the page index into an HND cache
// [num_pages, nkv, page_size, D] reshaped to [num_pages*nkv, 1, page_size, D] (a
// free view when the cache is contiguous HND). num_heads == 1 (h == 0) reproduces
// the single-kv-head TP>=4 behavior (page index == base_page).
template <typename SeqLenT, bool kUsePDL>
__global__ void minimax_decode_topk_page_table_kernel(
    const float* __restrict__ score,
    const SeqLenT* __restrict__ seq_lens,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ slot_ids,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ seq_lens_out,
    int batch,
    int num_heads,
    int max_seqblock,
    int block_size,
    int topk,
    int page_size,
    int r2t_stride,
    int max_kv_len,
    int max_sparse_pages) {
  const int b = blockIdx.x;  // grid.x = batch
  const int h = blockIdx.y;  // grid.y = num_heads (kv head)
  const int tx = threadIdx.x;

  // Prefetch seq_lens / slot_ids (from earlier kernels) and the cheap setup
  // before waiting on the score producer, so the prologue overlaps its tail (PDL).
  const int64_t seq_len = static_cast<int64_t>(seq_lens[b]);
  const int num_blocks_raw = static_cast<int>((seq_len + block_size - 1) / block_size);
  const int num_blocks = num_blocks_raw < max_seqblock ? num_blocks_raw : max_seqblock;
  const int ppb = block_size / page_size;
  const int64_t out_row = static_cast<int64_t>(b) * num_heads + h;  // flattened pseudo-request
  int32_t* __restrict__ pt_row = page_table + out_row * max_sparse_pages;
  const int64_t r2t_base = static_cast<int64_t>(slot_ids[b]) * r2t_stride;
  device::PDLWaitPrimary<kUsePDL>();

  if (num_blocks <= topk) {  // trivial: every block selected, all tokens valid
    if (tx == 0) seq_lens_out[out_row] = static_cast<int>(seq_len);
    // block id == ascending slot, so the partial final block's pages land last
    const int total = num_blocks * ppb;
    for (int e = tx; e < total; e += TopKTrait::kCTASize) {
      const int slot = e / ppb;
      const int pp = e % ppb;
      int tok = slot * block_size + pp * page_size;
      if (tok >= max_kv_len) tok = max_kv_len - 1;
      pt_row[e] = req_to_token[r2t_base + tok] / page_size * num_heads + h;
    }
    return;
  }

  const int k_eff = topk;                                                                        // num_blocks > topk
  const float* __restrict__ row = score + (static_cast<int64_t>(h) * batch + b) * max_seqblock;  // head-major score
  __shared__ TopKTrait::Smem smem;
  __shared__ int32_t s_topk[TopKTrait::kMaxTopK];
  TopKTrait::forward(row, static_cast<uint32_t>(num_blocks), s_topk, static_cast<uint32_t>(topk), &smem);
  __syncthreads();  // s_topk fully written before the transform reads it

  // Sort the selected block ids ascending (k_eff <= kMaxTopK is tiny) so the
  // partial final block lands last, accumulating the effective KV length in the
  // same pass: each selected block contributes min(block_size, seq_len - c*block)
  // valid tokens (only the final block can be partial).
  __shared__ int32_t s_sorted[TopKTrait::kMaxTopK];
  __shared__ int s_eff_kv;
  if (tx == 0) s_eff_kv = 0;
  __syncthreads();
  for (int slot = tx; slot < k_eff; slot += TopKTrait::kCTASize) {
    const int32_t v = s_topk[slot];
    int rank = 0;
    for (int j = 0; j < k_eff; ++j)
      rank += (s_topk[j] < v);
    s_sorted[rank] = v;
    const int rem = static_cast<int>(seq_len - static_cast<int64_t>(v) * block_size);
    atomicAdd(&s_eff_kv, rem < block_size ? rem : block_size);
  }
  __syncthreads();
  if (tx == 0) seq_lens_out[out_row] = s_eff_kv;

  // Parallel page emit: one thread per output page.
  const int total = k_eff * ppb;
  for (int e = tx; e < total; e += TopKTrait::kCTASize) {
    const int slot = e / ppb;
    const int pp = e % ppb;
    int tok = s_sorted[slot] * block_size + pp * page_size;
    if (tok >= max_kv_len) tok = max_kv_len - 1;
    pt_row[e] = req_to_token[r2t_base + tok] / page_size * num_heads + h;
  }
}

// -------------------------------------------------------------------------
// Launchers
// -------------------------------------------------------------------------
template <typename SeqLenT, bool kUsePDL>
void minimax_decode_topk(
    tvm::ffi::TensorView score,     // [H, B, S] fp32
    tvm::ffi::TensorView seq_lens,  // [B] int32/int64
    tvm::ffi::TensorView topk_idx,  // [H, B, T] int32
    int64_t block_size,
    int64_t topk) {
  using namespace host;

  SymbolicSize H = {"num_heads"};
  SymbolicSize B = {"batch"};
  SymbolicSize S = {"max_seqblock"};
  SymbolicSize T = {"topk"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({H, B, S}).with_dtype<fp32_t>().with_device(device_).verify(score);
  TensorMatcher({B}).with_dtype<SeqLenT>().with_device(device_).verify(seq_lens);
  TensorMatcher({H, B, T}).with_dtype<int32_t>().with_device(device_).verify(topk_idx);

  const int num_heads = static_cast<int>(H.unwrap());
  const int batch = static_cast<int>(B.unwrap());
  const int max_seqblock = static_cast<int>(S.unwrap());
  const int topk_i = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(
      static_cast<int64_t>(topk_i) == topk,
      "minimax_decode_topk: topk arg (",
      topk,
      ") must match topk_idx last dim (",
      topk_i,
      ")");
  RuntimeCheck(block_size > 0, "block_size must be > 0, got ", block_size);
  if (batch == 0 || num_heads == 0) return;

  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(num_heads));
  LaunchKernel(grid, TopKTrait::kCTASize, device, 0)
      .enable_pdl(kUsePDL)(
          minimax_decode_topk_block_kernel<SeqLenT, kUsePDL>,
          static_cast<const float*>(score.data_ptr()),
          static_cast<const SeqLenT*>(seq_lens.data_ptr()),
          static_cast<int32_t*>(topk_idx.data_ptr()),
          batch,
          num_heads,
          max_seqblock,
          static_cast<int>(block_size),
          topk_i);
}

// Page-table variant: emit the per-(batch, kv-head) paged page table consumed by
// the dense backend (trtllm_mha / fa3) plus the effective KV length, instead of
// block ids. For DP attention (num_kv_heads > 1) each kv head selects its own
// blocks, so (b, h) pseudo-requests are flattened batch-major into the output
// (B*num_heads rows); num_heads == 1 is the TP>=4 single-kv-head case. The page
// index is head-encoded (head-minor) as base_page*num_heads + h -- the index into
// an HND cache [num_pages, nkv, ps, D] reshaped to [num_pages*nkv, 1, ps, D].
// page_table and seq_lens_out are allocated by the caller.
template <typename SeqLenT, bool kUsePDL>
void minimax_decode_topk_page_table(
    tvm::ffi::TensorView score,         // [H, B, S] fp32 (H = num_kv_heads)
    tvm::ffi::TensorView seq_lens,      // [B] int32/int64
    tvm::ffi::TensorView req_to_token,  // [max_reqs, max_kv_len] int32
    tvm::ffi::TensorView slot_ids,      // [B] int64 (req_pool_indices)
    tvm::ffi::TensorView page_table,    // [B*H, max_sparse_pages] int32 (out)
    tvm::ffi::TensorView seq_lens_out,  // [B*H] int32 (effective KV length, out)
    int64_t block_size,
    int64_t topk,
    int64_t page_size) {
  using namespace host;

  SymbolicSize H = {"num_heads"};
  SymbolicSize B = {"batch"};
  SymbolicSize S = {"max_seqblock"};
  SymbolicSize R = {"max_reqs"};
  SymbolicSize KV = {"max_kv_len"};
  SymbolicSize BH = {"batch_heads"};
  SymbolicSize P = {"max_sparse_pages"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({H, B, S}).with_dtype<fp32_t>().with_device(device_).verify(score);
  TensorMatcher({B}).with_dtype<SeqLenT>().with_device(device_).verify(seq_lens);
  TensorMatcher({R, KV}).with_dtype<int32_t>().with_device(device_).verify(req_to_token);
  TensorMatcher({B}).with_dtype<int64_t>().with_device(device_).verify(slot_ids);
  TensorMatcher({BH, P}).with_dtype<int32_t>().with_device(device_).verify(page_table);
  TensorMatcher({BH}).with_dtype<int32_t>().with_device(device_).verify(seq_lens_out);

  const int num_heads = static_cast<int>(H.unwrap());
  const int batch = static_cast<int>(B.unwrap());
  const int max_seqblock = static_cast<int>(S.unwrap());
  const int max_kv_len = static_cast<int>(KV.unwrap());
  const int max_sparse_pages = static_cast<int>(P.unwrap());
  const int r2t_stride = static_cast<int>(req_to_token.stride(0));
  const DLDevice device = device_.unwrap();

  RuntimeCheck(
      BH.unwrap() == static_cast<int64_t>(batch) * num_heads,
      "page_table rows (",
      BH.unwrap(),
      ") must equal batch*num_heads (",
      static_cast<int64_t>(batch) * num_heads,
      ")");
  RuntimeCheck(
      block_size > 0 && page_size > 0 && block_size % page_size == 0,
      "block_size must be a positive multiple of page_size");
  RuntimeCheck(topk <= static_cast<int64_t>(TopKTrait::kMaxTopK), "topk exceeds kMaxTopK for page-table mode");
  if (batch == 0 || num_heads == 0) return;

  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(num_heads));
  LaunchKernel(grid, TopKTrait::kCTASize, device, 0)
      .enable_pdl(kUsePDL)(
          minimax_decode_topk_page_table_kernel<SeqLenT, kUsePDL>,
          static_cast<const float*>(score.data_ptr()),
          static_cast<const SeqLenT*>(seq_lens.data_ptr()),
          static_cast<const int32_t*>(req_to_token.data_ptr()),
          static_cast<const int64_t*>(slot_ids.data_ptr()),
          static_cast<int32_t*>(page_table.data_ptr()),
          static_cast<int32_t*>(seq_lens_out.data_ptr()),
          batch,
          num_heads,
          max_seqblock,
          static_cast<int>(block_size),
          static_cast<int>(topk),
          static_cast<int>(page_size),
          r2t_stride,
          max_kv_len,
          max_sparse_pages);
}

}  // namespace
