// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include "candidates.cuh"

// Helpers of the per-row top-K selections (row_select.cuh) over candidate fp24
// codes (higher code = higher score): the bin search of a histogram level and
// a warp prefix scan. K (the row's top-k, <= kMaxK) is a runtime value;
// shared-memory scratch is sized for kMaxK.
namespace fast_select {
constexpr unsigned kMaxK = 2048, Full = 0xffffffffu, kBins = 1024;

// Bin (descending) where the cumulative count first reaches `rank`;
// `above` = elements in higher bins. Lane l owns bins kBins-1-32l .. -31.
__device__ __forceinline__ unsigned find_bin(const unsigned* hist,
                                             unsigned rank, unsigned& above) {
  const unsigned lane = threadIdx.x & 31;
  constexpr unsigned kPerLane = kBins / 32;
  unsigned sum = 0;
#pragma unroll
  for (unsigned j = 0; j < kPerLane; ++j)
    sum += hist[kBins - 1 - lane * kPerLane - j];
  unsigned inclusive = sum;
#pragma unroll
  for (unsigned d = 1; d < 32; d *= 2) {
    const unsigned v = __shfl_up_sync(Full, inclusive, d);
    if (lane >= d) inclusive += v;
  }
  const unsigned before = inclusive - sum;
  const unsigned winner =
      __ffs(__ballot_sync(Full, before < rank && inclusive >= rank)) - 1;
  unsigned chosen = 0, acc = before;
  if (lane == winner) {
    for (unsigned j = 0; j < kPerLane; ++j) {
      const unsigned bin = kBins - 1 - lane * kPerLane - j, h = hist[bin];
      if (acc + h >= rank) {
        chosen = bin;
        break;
      }
      acc += h;
    }
  }
  above = __shfl_sync(Full, acc, winner);
  return __shfl_sync(Full, chosen, winner);
}

__device__ __forceinline__ unsigned warp_exclusive_scan(unsigned v,
                                                        unsigned& total) {
  const unsigned lane = threadIdx.x & 31;
  unsigned inc = v;
#pragma unroll
  for (unsigned d = 1; d < 32; d *= 2) {
    const unsigned t = __shfl_up_sync(Full, inc, d);
    if (lane >= d) inc += t;
  }
  total = __shfl_sync(Full, inc, 31);
  return inc - v;
}

// Bins of the histogram levels that resolve the exact threshold inside a
// bracket (row_select.cuh).
constexpr unsigned kMidBins = 512;

}  // namespace fast_select
