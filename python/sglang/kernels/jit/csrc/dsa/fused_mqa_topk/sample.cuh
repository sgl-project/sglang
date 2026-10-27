// Gate sample of the persistent kernel. Each math warp histograms its row's
// sampled scores relative to a per-row reference (the row's maximum over its
// first sample tile), in bins of fp16(x - ref): 1/64 octave of the distance to
// the reference, 4096 bins as 16-bit counters, two per word, 2048 words padded
// to 33 per 32. finish_sample_histogram picks the gate from them. Absolute fp16
// bins are ~2 wide at |x| ~ 100, wider than the whole top-k tail of layers
// whose scores sit at a large common offset (GLM-5.2 layer 1: ~+100, ranks
// 1024-4096 within 1.6-2.6), which puts the gate far too low.
#pragma once

#include "common.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang::fused_mqa_topk {

struct alignas(16) SampleShared {
  uint32_t hist[kBlockRows][padded_bins(2048)];
  // Reference of each row: ordered fp32 code of its maximum valid sample in
  // its first sample tile holding any (0: none yet / no sample, reference 0).
  uint32_t ref[kBlockRows];
  // The samples of a reference tile (NaN: not valid) until its maximum is known.
  float stage[kBlockRows][kTileKeys];
};

// Bins of a sample relative to its row's reference (ascending with the value).
inline constexpr uint32_t kSampleBinShift = 4;

// Counts one sample in bin b (16-bit half b & 1 of word b >> 1; a row holds at
// most 255 sample tiles = 65280 samples, see launch_config).
__device__ __forceinline__ void sample_add(uint32_t* hist, uint32_t b) {
  const uint32_t w = b >> 1;
  atomicAdd(hist + padded_bin(w), 1u << ((b & 1u) * 16));
}

// Reference value of an ordered code; code 0 (no sample) is reference 0.
__device__ __forceinline__ float sample_ref_value(uint32_t code) {
  return code == 0u ? 0.0f : ordered_float(code);
}

__device__ __forceinline__ uint32_t sample_pair_total(uint32_t v) {
  return (v & 0xffffu) + (v >> 16);
}

// Gate: the k-th largest sample (k = count * target / visible, count = the
// histogram total) of one row from its 4096-bin histogram; one warp per row.
// The gate is ref + the lower edge of the bin holding the k-th sample (a lower
// bound of it). -inf when the row sees at most `target` keys or has no
// samples; +inf when it sees at most top-k keys.
__device__ __noinline__ void finish_sample_histogram(
    const uint32_t* hist, float* gate, uint32_t visible, uint32_t target, float ref, uint32_t lane) {
  if (visible <= target) {
    // Rows that see at most top-k keys (padding rows past the end included)
    // are written directly (write_short) and need no candidates: a +inf gate
    // keeps them out of the any-pass vote.
    if (lane == 0) *gate = visible <= kTopK ? CUDART_INF_F : -CUDART_INF_F;
    __syncwarp();
    return;
  }
  // Lane l owns the 64 words (128 bins) of chunks 63-2l and 62-2l (bins
  // descending with the lane).
  const uint32_t* const h0 = hist + (63 - 2 * lane) * 33;
  const uint32_t* const h1 = hist + (62 - 2 * lane) * 33;
  uint32_t sum = 0;
#pragma unroll 8
  for (uint32_t j = 0; j < 32; ++j)
    sum += sample_pair_total(h0[j]) + sample_pair_total(h1[j]);
  uint32_t incl = sum;
  incl = warp_inclusive_scan(incl, lane);
  const uint32_t tot = __shfl_sync(kFullMask, incl, 31);
  if (tot == 0) {
    if (lane == 0) *gate = -CUDART_INF_F;
    __syncwarp();
    return;
  }
  const uint32_t rank = max(1u, min(tot, (tot * target + visible - 1) / visible));
  const uint32_t w = __ffs(__ballot_sync(kFullMask, incl >= rank)) - 1;
  uint32_t above = __shfl_sync(kFullMask, incl - sum, w), chosen = 0;
#pragma unroll 1
  for (uint32_t cc = 0; cc < 2; ++cc) {
    const uint32_t c = 63 - 2 * w - cc;
    // lane 0 = highest word of the chunk; its high half is the higher bin
    const uint32_t word = hist[c * 33 + 31 - lane];
    const uint32_t total = sample_pair_total(word);
    uint32_t v = total;
    v = warp_inclusive_scan(v, lane);
    const uint32_t mask = __ballot_sync(kFullMask, above + v >= rank);
    if (mask) {
      const uint32_t sel = __ffs(mask) - 1;
      const uint32_t before = above + __shfl_sync(kFullMask, v - total, sel),
                     high = __shfl_sync(kFullMask, word >> 16, sel);
      chosen = 2 * (c * 32 + 31 - sel) + (before + high >= rank ? 1u : 0u);
      break;
    }
    above += __shfl_sync(kFullMask, v, 31);
  }
  if (lane == 0) *gate = ref + half_bin_floor<kSampleBinShift>(chosen);
  __syncwarp();
}

}  // namespace sglang::fused_mqa_topk
