#pragma once
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>

namespace device::top512 {

inline constexpr uint32_t kMaxTopK = 1024;
inline constexpr uint32_t kBlockSize = 1024;
inline constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;
inline constexpr uint32_t kMaxTies = 1024;  // == kBlockSize: 1 element per thread in stage2
static constexpr uint32_t kRadixBins = 256;
static_assert(kMaxTopK <= kBlockSize && kMaxTies <= kBlockSize);

// always use float4 to load from global memory
using Vec4 = AlignedVector<float, 4>;

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

struct TransformParams {
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ indices_in;
  int32_t* __restrict__ indices_out;
  uint32_t page_bits;

  SGL_DEVICE void transform(const uint32_t idx) const {
    indices_out[idx] = page_to_indices(page_table, indices_in[idx], page_bits);
  }
  SGL_DEVICE void write(const uint32_t dst, const uint32_t src) const {
    indices_out[dst] = page_to_indices(page_table, src, page_bits);
  }
};

struct alignas(16) MatchBin {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

struct alignas(8) Tie {
  uint32_t idx;
  float score;
};

struct TieHandleSmem {
  alignas(128) uint32_t counter;  // output position counter
  alignas(128) MatchBin match;
  uint32_t histogram[kRadixBins];  // 256-bin radix histogram
  uint32_t warp_sum[kNumWarps];    // for 2-pass prefix sum
};

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  const auto hx = cast<fp16_t>(x);
  const uint16_t bits = *reinterpret_cast<const uint16_t*>(&hx);
  const uint16_t key = (bits & 0x8000) ? ~bits : bits | 0x8000;
  return key >> (16 - kBits);
}

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
  static_assert(kWarpThreads == 32);
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

/// Order-preserving float32 -> uint32 for radix select
SGL_DEVICE uint32_t extract_exact_bin(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE void trivial_transform(const TransformParams& params, uint32_t length, uint32_t K) {
  if (const auto tx = threadIdx.x; tx < length) {
    params.write(tx, tx);
  } else if (tx < K) {
    params.indices_out[tx] = -1;
  }
}

SGL_DEVICE void tie_handle_transform(
    const Tie* __restrict__ ties,  //
    const uint32_t num_ties,
    const uint32_t num_above,
    const uint32_t K,
    const TransformParams params,
    void* _smem) {
  auto* smem = static_cast<TieHandleSmem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpThreads;
  const auto warp_id = tx / kWarpThreads;

  // Each thread loads one element (or becomes inactive)
  const bool has_elem = tx < num_ties;
  const auto tie = has_elem ? ties[tx] : Tie{0, 0.0f};
  const uint32_t key = extract_exact_bin(tie.score);
  const uint32_t idx = tie.idx;
  bool active = has_elem;
  uint32_t topk_remain = K - num_above;
  uint32_t write_pos = K;

  smem->counter = 0;
  __syncthreads();

  // Number of warps covering the 256-bin histogram (256/32 = 8)
  constexpr uint32_t kRadixWarps = kRadixBins / kWarpThreads;

#pragma unroll
  for (int round = 0; round < 4; round++) {
    const uint32_t shift = 24 - round * 8;
    const uint32_t bin = (key >> shift) & 0xFFu;

    // 1. Build histogram
    if (tx < kRadixBins) smem->histogram[tx] = 0;
    __syncthreads();
    if (active) atomicAdd(&smem->histogram[bin], 1);
    __syncthreads();

    // 2. v2-style 2-pass prefix sum on 256 bins
    //    Only first 256 threads (8 warps) carry histogram bins.
    //    Other threads get hist_val=0 and harmless prefix results.
    uint32_t hist_val = 0;
    uint32_t warp_inc = 0;
    if (tx < kRadixBins) {
      hist_val = smem->histogram[tx];
      warp_inc = warp_inclusive_sum(lane_id, hist_val);
      if (lane_id == kWarpThreads - 1) smem->warp_sum[warp_id] = warp_inc;
    }
    __syncthreads();
    if (tx < kRadixBins) {
      // Inter-warp prefix (only first kHistWarps warp totals matter)
      const auto tmp = (lane_id < kRadixWarps) ? smem->warp_sum[lane_id] : 0;
      const auto total = warp::reduce_sum(tmp);
      const auto inter = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
      const auto prefix = inter + warp_inc;  // inclusive prefix through this bin
      const auto above = total - prefix;     // elements in bins ABOVE this one
      // 3. Find threshold bin
      if (above < topk_remain && above + hist_val >= topk_remain) {
        smem->match = {tx, above, topk_remain - above};
      }
    }
    __syncthreads();

    const auto [thr, n_above, _] = smem->match;

    // 4. Scatter
    if (active) {
      if (bin > thr) {
        write_pos = num_above + atomicAdd(&smem->counter, 1);
        active = false;
      } else if (bin < thr) {
        active = false;
      } else if (round == 3) {
        write_pos = K - atomicAdd(&smem->match.equal_count, -1u);
      }
      // my_bin == thr && round < 3: stay active for next round
    }

    topk_remain -= n_above;
    if (topk_remain == 0) break;
  }

  if (write_pos < K) params.write(write_pos, idx);
}

}  // namespace device::top512
