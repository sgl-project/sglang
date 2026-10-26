// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include "candidates.cuh"

// One warp selects one row's top-K from its candidate fp24 codes (higher code =
// higher score), staged once in shared memory. Candidate codes span a narrow
// range (gate .. row max), so each level histograms the range-relative code
// into at most kBins bins: collisions are rare, plain shared atomics suffice
// and each lane keeps U independent elements in flight. Levels repeat inside
// the boundary bin until it is one code wide; codes above it are emitted, then
// ties until K. K (the row's top-k, <= kMaxK) is a runtime value;
// shared-memory scratch is sized for kMaxK.
namespace fast_select {
constexpr unsigned kMaxK = 2048, Full = 0xffffffffu, kBins = 1024, U = 8;

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

// Requires count >= K (callers recover rows with fewer candidates).
__device__ void select_row(unsigned* codes, unsigned count,
                           const uint16_t* values, const int32_t* packed,
                           int32_t* out, unsigned K, unsigned* hist) {
  const unsigned lane = threadIdx.x & 31;
  unsigned lo = 0xffffffffu, hi = 0;
  unsigned j = lane;
  for (; j + 32 * (U - 1) < count; j += 32 * U) {
    uint16_t v[U];
    int32_t p[U];
#pragma unroll
    for (unsigned u = 0; u < U; ++u) {
      v[u] = values[j + 32 * u];
      p[u] = packed[j + 32 * u];
    }
#pragma unroll
    for (unsigned u = 0; u < U; ++u) {
      const unsigned c =
          fused_candidates::candidate_load_score_code(v[u], p[u]);
      codes[j + 32 * u] = c;
      lo = min(lo, c);
      hi = max(hi, c);
    }
  }
  for (; j < count; j += 32) {
    const unsigned c =
        fused_candidates::candidate_load_score_code(values[j], packed[j]);
    codes[j] = c;
    lo = min(lo, c);
    hi = max(hi, c);
  }
  lo = __reduce_min_sync(Full, lo);
  hi = __reduce_max_sync(Full, hi);
  __syncwarp();

  // Current bucket: codes in [lo, lo + width). rank = elements still needed
  // from it.
  unsigned width = hi - lo + 1, rank = K;
#pragma unroll 1
  while (width > 1) {
    const unsigned spread = (width - 1) >> 10;
    const unsigned shift = spread ? 32 - __clz(spread) : 0;
    for (unsigned b = lane; b < kBins; b += 32) hist[b] = 0;
    __syncwarp();
    const unsigned top = lo + width - 1;
    unsigned i = lane;
    for (; i + 32 * (U - 1) < count; i += 32 * U) {
      unsigned c[U];
#pragma unroll
      for (unsigned u = 0; u < U; ++u) c[u] = codes[i + 32 * u];
#pragma unroll
      for (unsigned u = 0; u < U; ++u)
        if (c[u] >= lo && c[u] <= top)
          atomicAdd(hist + ((c[u] - lo) >> shift), 1u);
    }
    for (; i < count; i += 32) {
      const unsigned c = codes[i];
      if (c >= lo && c <= top) atomicAdd(hist + ((c - lo) >> shift), 1u);
    }
    __syncwarp();
    unsigned above;
    const unsigned chosen = find_bin(hist, rank, above);
    rank -= above;
    lo += chosen << shift;
    width = min(1u << shift, top - lo + 1);
    __syncwarp();
  }

  // lo is now the K-th largest code: emit codes above it, then ties until K.
  const unsigned threshold = lo, lt = (1u << lane) - 1u;
  unsigned pos = 0;
#pragma unroll 1
  for (unsigned tie = 0; tie < 2; ++tie) {
    for (unsigned j0 = 0; j0 < count && pos < K; j0 += 32 * U) {
      bool pass[U];
      int32_t p[U];
#pragma unroll
      for (unsigned u = 0; u < U; ++u) {
        const unsigned e = j0 + 32 * u + lane;
        const unsigned c = e < count ? codes[e] : 0;
        pass[u] = e < count && (tie ? c == threshold : c > threshold);
        p[u] = pass[u] ? packed[e] : 0;
      }
#pragma unroll
      for (unsigned u = 0; u < U; ++u) {
        const unsigned mask = __ballot_sync(Full, pass[u]),
                       at = pos + __popc(mask & lt);
        if (pass[u] && at < K)
          out[at] = fused_candidates::candidate_decode_index(p[u]);
        pos += __popc(mask);
      }
    }
  }
}
// Register samples per lane for the sampled bracket selects (row_select.cuh).
constexpr unsigned kSamplesPerLane = 16;
// Most elements a bracket select resolves inside its bracket (row_select.cuh).
constexpr unsigned kMid3Cap = 1024;

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

template <unsigned Bins>
__device__ __forceinline__ unsigned find_bin_n(const unsigned* hist,
                                               unsigned rank, unsigned& above) {
  const unsigned lane = threadIdx.x & 31;
  constexpr unsigned kPerLane = Bins / 32;
  unsigned sum = 0;
#pragma unroll
  for (unsigned j = 0; j < kPerLane; ++j)
    sum += hist[Bins - 1 - lane * kPerLane - j];
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
      const unsigned bin = Bins - 1 - lane * kPerLane - j, h = hist[bin];
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

}  // namespace fast_select
