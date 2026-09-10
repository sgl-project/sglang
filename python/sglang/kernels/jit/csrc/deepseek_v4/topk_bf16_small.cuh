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

namespace sglang {

struct TopKBF16Config {
  static constexpr uint32_t kBlockSize = 1024;
  static constexpr uint32_t kOccupancy = 2;
  static constexpr uint32_t kMaxSeqLen = 16384;
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  static constexpr uint32_t kNumItems = kMaxSeqLen / (kVecSize * kBlockSize);
  static constexpr uint32_t kMaxTopK = 4096;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static_assert(kMaxSeqLen == kNumItems * kVecSize * kBlockSize);
  using vec_t = device::AlignedVector<bf16x2_t, kVecSize / 2>;
  struct SmemApprox {
    static constexpr uint32_t kHistBits = 13;
    static constexpr uint32_t kHistSize = 1 << kHistBits;
    // The threshold search walks the histogram 16 B at a time: 4 bins per thread
    // per round, warp w owning the bin range [w * 256, (w + 1) * 256).
    static constexpr uint32_t kHistItems = 16 / sizeof(uint32_t);
    static constexpr uint32_t kHistRounds = kHistSize / (kHistItems * kBlockSize);
    using hist_vec_t = device::AlignedVector<uint32_t, kHistItems>;
    static_assert(kHistRounds == 2 && kNumWarps == device::kWarpThreads);
    uint32_t count_eq_gt;    // collect cursors, (gt << 16) | eq
    uint32_t threshold_bin;  // the bin holding the k-th largest
    uint32_t warp_sum[kNumWarps];
    union {
      alignas(16) uint32_t histogram[kHistSize];
      int32_t stage_out_idxs[kMaxTopK];
    };
  };
};

#define TOPK_BF16_KERNEL __global__ __launch_bounds__(TopKBF16Config::kBlockSize, TopKBF16Config::kOccupancy)

struct TopKBF16Params {
  const bf16_t* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  int64_t page_indices_stride;
  uint32_t topk;
  uint32_t page_bits;
};

SGL_DEVICE uint32_t extract_coarse_bin2(bf16x2_t pair) {
  using device::cast;
  const auto fp16_pair = cast<fp16x2_t>(cast<fp32x2_t>(pair));
  const auto bits = reinterpret_cast<const uint32_t&>(fp16_pair);
  const auto sign = ((bits >> 15) & 0x00010001u) * 0xFFFFu;
  const auto key = bits ^ (sign | 0x80008000u);
  return (key >> 3) & 0x1FFF1FFFu;  // 1 + 5 + 7 = 13 bit, drop 3 redundant bits
}

SGL_DEVICE uint32_t get_ptx_lane_id() {
  uint32_t lane_id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}

/// Locates the bin where `above < topk <= above + count`, `above` being the number
/// of elements in the bins beyond it, and publishes both. `total` is what the
/// histogram counted, so `topk <= total` or no bin qualifies (the caller passes
/// min(topk, seq_len)). One bin passes, so exactly one thread writes. The
/// counters are re-read for that bin rather than kept: the row is live in
/// registers across this and 8 more would spill.
SGL_DEVICE void topk_bf16_find_threshold(uint32_t topk, uint32_t total, TopKBF16Config::SmemApprox& smem) {
  using S = TopKBF16Config::SmemApprox;
  using hist_vec_t = S::hist_vec_t;
  constexpr auto kItems = S::kHistItems;
  constexpr auto kRounds = S::kHistRounds;
  constexpr auto kWarpSize = device::kWarpThreads;
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;
  const auto hist_vec_index = [&](uint32_t r) { return warp_id * kRounds * kWarpSize + lane_id + r * kWarpSize; };

  uint32_t local_sum[kRounds];
#pragma unroll
  for (uint32_t r = 0; r < kRounds; ++r) {
    hist_vec_t hist;
    hist.load(smem.histogram, hist_vec_index(r));
    local_sum[r] = 0;
#pragma unroll
    for (uint32_t j = 0; j < kItems; ++j) {
      local_sum[r] += hist[j];
    }
  }

  uint32_t warp_inc_sum[kRounds];
#pragma unroll
  for (uint32_t r = 0; r < kRounds; ++r) {
    warp_inc_sum[r] = device::warp::inclusive_sum(lane_id, local_sum[r]);
  }
  if (lane_id == kWarpSize - 1) smem.warp_sum[warp_id] = warp_inc_sum[0] + warp_inc_sum[1];
  // round 1 sits entirely above round 0 within the warp's range
  const auto warp_half_sum = __shfl_sync(0xFFFFFFFFu, warp_inc_sum[0], kWarpSize - 1);
  __syncthreads();

  const auto peer_sum = smem.warp_sum[lane_id];
  const auto warp_prefix_sum = device::warp::reduce_sum(lane_id < warp_id ? peer_sum : 0u);
  // elements in every bin below this thread's bins of round r
  const uint32_t exc_sum[kRounds] = {
      warp_prefix_sum + warp_inc_sum[0] - local_sum[0],
      warp_prefix_sum + warp_half_sum + warp_inc_sum[1] - local_sum[1],
  };

#pragma unroll
  for (uint32_t r = 0; r < kRounds; ++r) {
    // `above` only falls as the bin rises, so the threshold is inside this
    // thread's bins exactly when it straddles their two ends.
    const auto above_hi = total - exc_sum[r];
    const auto above_lo = above_hi - local_sum[r];
    if (above_lo >= topk || topk > above_hi) continue;
    hist_vec_t hist;
    hist.load(smem.histogram, hist_vec_index(r));
    auto prefix = exc_sum[r];
    const auto bin_base = hist_vec_index(r) * kItems;
#pragma unroll
    for (uint32_t j = 0; j < kItems; ++j) {
      prefix += hist[j];
      const auto above = total - prefix;
      if (above < topk && above + hist[j] >= topk) {
        smem.threshold_bin = bin_base + j;
        smem.count_eq_gt = above;
      }
    }
  }
}

/// TODO: this kernel is not optimized at all
template <bool kUsePDL>
TOPK_BF16_KERNEL void topk_bf16_approx_kernel(const __grid_constant__ TopKBF16Params params) {
  using namespace device;
  using C = TopKBF16Config;
  using vec_t = C::vec_t;
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto seq_len = static_cast<uint32_t>(params.seq_lens[bx]);
  const auto* __restrict__ scores_row = params.scores + bx * params.score_stride;

  using Smem = typename C::SmemApprox;
  __shared__ Smem smem;

  for (uint32_t i = 0; i < Smem::kHistSize / C::kBlockSize; ++i) {
    smem.histogram[tx + i * C::kBlockSize] = 0;
  }
  __syncthreads();
  PDLWaitPrimary<kUsePDL>();

  if (seq_len <= params.topk) {
    // every element is selected: the row's indices through the table, -1 past the row
    const auto* __restrict__ table = params.page_table + bx * params.page_table_stride;
    auto* __restrict__ out = params.page_indices + bx * params.page_indices_stride;
    const auto page_mask = (1u << params.page_bits) - 1;
    for (uint32_t t = tx; t < params.topk; t += C::kBlockSize) {
      out[t] =
          t < seq_len ? (table[t >> params.page_bits] << params.page_bits) | static_cast<int32_t>(t & page_mask) : -1;
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  vec_t scores[C::kNumItems];
  const auto num_full = div_ceil(seq_len, C::kVecSize);
#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
    const auto vid = i * C::kBlockSize + tx;
    if (vid < num_full) {
      scores[i].load(scores_row, vid);
    }
  }

  // 1 time 13-bit histogram, not 100% correct
  uint32_t bins[C::kNumItems][C::kVecSize / 2];
#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
      bins[i][j] = extract_coarse_bin2(scores[i][j]);
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
    const auto vid = i * C::kBlockSize + tx;
    if (vid + 1 == num_full) [[unlikely]] {
      // elements of the last vector that are inside the row
      const auto tail_length = (seq_len - 1) % C::kVecSize + 1;
#pragma unroll
      for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
        const auto lo = bins[i][j] & 0xFFFFu;
        const auto hi = bins[i][j] >> 16;
        const auto lo_valid = j * 2 + 0 < tail_length;
        const auto hi_valid = j * 2 + 1 < tail_length;
        // wish compiler can generate predicate instructions
        if (lo_valid) atomicAdd(&smem.histogram[lo], 1);
        if (hi_valid) atomicAdd(&smem.histogram[hi], 1);
        // mask invalid bins to lowest (0): no real value keys below the -inf bin
        // (0x7F), so they can never compare above or equal to the threshold
        if (!hi_valid) bins[i][j] = lo;
        if (!lo_valid) bins[i][j] = 0;
      }
    } else if (vid < num_full) [[likely]] {
#pragma unroll
      for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
        const auto lo = bins[i][j] & 0xFFFFu;
        const auto hi = bins[i][j] >> 16;
        atomicAdd(&smem.histogram[lo], 1);
        atomicAdd(&smem.histogram[hi], 1);
      }
    }
  }

  __syncthreads();
  topk_bf16_find_threshold(params.topk, seq_len, smem);
  __syncthreads();
  const auto threshold_bin = smem.threshold_bin;
  uint32_t local_eq = 0;
  uint32_t local_gt = 0;
#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
    const auto vid = i * C::kBlockSize + tx;
    if (vid < num_full) {
#pragma unroll
      for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
        const auto lo = bins[i][j] & 0xFFFFu;
        const auto hi = bins[i][j] >> 16;
        local_gt += lo > threshold_bin;
        local_gt += hi > threshold_bin;
        local_eq += lo == threshold_bin;
        local_eq += hi == threshold_bin;
      }
    }
  }
  const auto lane_id = get_ptx_lane_id();
  const auto local_payload = (local_gt << 16) | local_eq;
  const auto warp_inc_payload = warp::inclusive_sum(lane_id, local_payload);
  uint32_t warp_payload = 0;
  if (lane_id == kWarpThreads - 1) {
    warp_payload = atomicAdd(&smem.count_eq_gt, warp_inc_payload);
  }
  warp_payload = __shfl_sync(0xFFFFFFFFu, warp_payload, kWarpThreads - 1);
  const auto exc_payload = warp_payload + warp_inc_payload - local_payload;
  local_eq = exc_payload & 0xFFFFu;
  local_gt = exc_payload >> 16;
  const auto collect = [&](uint32_t bin, uint32_t idx) {
    if (bin == threshold_bin) {
      const auto pos = local_eq++;
      if (pos < params.topk) smem.stage_out_idxs[pos] = idx;
    } else if (bin > threshold_bin) {
      const auto pos = local_gt++;
      smem.stage_out_idxs[pos] = idx;
    }
  };

#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
    const auto vid = i * C::kBlockSize + tx;
    if (vid < num_full) {
#pragma unroll
      for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
        const auto lo = bins[i][j] & 0xFFFFu;
        const auto hi = bins[i][j] >> 16;
        collect(lo, vid * C::kVecSize + j * 2 + 0);
        collect(hi, vid * C::kVecSize + j * 2 + 1);
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

  // page-transform: a selected index i maps through this row's table to slot
  // table[i >> page_bits] << page_bits | (i & mask); -1 past what the row has
  const auto* __restrict__ table = params.page_table + bx * params.page_table_stride;
  auto* __restrict__ out = params.page_indices + bx * params.page_indices_stride;
  const auto page_mask = (1u << params.page_bits) - 1;
  constexpr uint32_t kRounds = C::kMaxTopK / C::kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kRounds; ++i) {
    const auto t = i * C::kBlockSize + tx;
    if (t < params.topk) {
      const auto idx = static_cast<uint32_t>(smem.stage_out_idxs[t]);
      out[t] = (table[idx >> params.page_bits] << params.page_bits) | static_cast<int32_t>(idx & page_mask);
    }
  }
}

/// Host entry: bf16 top-k over rows of at most kMaxSeqLen, selected indices
/// written through a per-row table as `table[i >> log2(page_size)] << log2(page_size)
/// | (i & mask)`, -1 past min(topk, seq_len).
template <bool kPDL>
struct TopKBF16Kernel {
  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size) {
    using namespace host;
    using C = TopKBF16Config;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto K = SymbolicSize{"topk"};
    auto O = SymbolicSize{"page_indices_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({B, L})  // scores
        .with_strides({S, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, -1})  // page_table
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_table);
    TensorMatcher({B, K})  // page_indices
        .with_strides({O, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);
    RuntimeCheck(std::has_single_bit(page_size), "page_size must be a power of 2");
    RuntimeCheck(L.unwrap() <= C::kMaxSeqLen, "rows longer than kMaxSeqLen take the streaming top-k");
    RuntimeCheck(S.unwrap() % C::kVecSize == 0, "score_stride must keep every row vector-aligned");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= C::kMaxTopK, "topk must be in (0, kMaxTopK]");
    const auto params = TopKBF16Params{
        .scores = static_cast<const bf16_t*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = page_table.stride(0),
        .page_indices_stride = O.unwrap(),
        .topk = topk,
        .page_bits = static_cast<uint32_t>(std::countr_zero(page_size)),
    };
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), C::kBlockSize, device_.unwrap())
        .config({.use_pdl = kPDL})
        .launch(topk_bf16_approx_kernel<kPDL>, params);
  }
};

}  // namespace sglang
