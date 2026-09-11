#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

namespace sglang::device::legacy_radix_topk {

inline constexpr int kRadix = 256;

// The legacy DSA selectors first partition finite FP32 scores by the high byte
// of their ordered FP16 representation. This keeps their common-case binning
// and performance unchanged. Exact refinement uses the monotone FP32 key.
__device__ __forceinline__ uint8_t coarse_key(float x) {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000u);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t exact_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Exact, allocation-free radix selection for the legacy DSA kernels.
//
// The normal path preserves the original two-stash algorithm. Its threshold
// bin is safe to materialize because it fits below the shared-memory capacity.
// At or above that capacity, we instead rescan the coarse bin while descending
// the four bytes of the ordered FP32 key. No candidate is emitted or discarded
// before the exact cutoff is known; the final scan then fills disjoint strict
// and exact-tie output ranges.
//
// Scores must be finite. For valid input (length > topk > 0), every output slot
// is filled with a distinct index in [0, length). Defensive initialization and
// bounds checks turn any violated invariant into -1 padding instead of an
// out-of-bounds store or a garbage page-table lookup.
template <int BlockSize, int OutputCapacity, int StashEntries>
__device__ __forceinline__ void
select(const float* __restrict__ input, int32_t* __restrict__ output, int row_start, int length, int topk) {
  static_assert(BlockSize >= kRadix + 1, "the block must initialize the complete radix histogram");
  static_assert(OutputCapacity <= StashEntries, "top-k output must fit in one shared-memory stash");

  alignas(128) __shared__ int histogram_buf[2][kRadix + 1];
  alignas(128) __shared__ int counter;
  alignas(128) __shared__ int threshold_bin_id;
  alignas(128) __shared__ int num_input[2];

  extern __shared__ int dynamic_smem[];
  auto* input_idx = reinterpret_cast<int (*)[StashEntries]>(dynamic_smem);
  auto& histogram = histogram_buf[0];
  const int tx = threadIdx.x;

  for (int i = tx; i < topk; i += BlockSize)
    output[i] = -1;
  if (tx < kRadix + 1) histogram[tx] = 0;
  if (tx == 0) {
    counter = 0;
    threshold_bin_id = -1;
    num_input[0] = 0;
    num_input[1] = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < length; idx += BlockSize) {
    atomicAdd(&histogram[coarse_key(input[row_start + idx])], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < kRadix) {
        const int offset = 1 << i;
        const int src = i & 1;
        int value = histogram_buf[src][tx];
        if (tx < kRadix - offset) value += histogram_buf[src][tx + offset];
        histogram_buf[src ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < kRadix && histogram[tx] > topk && histogram[tx + 1] <= topk) {
    threshold_bin_id = tx;
  }
  __syncthreads();

  const int coarse_threshold = threshold_bin_id;
  if (coarse_threshold < 0) return;
  const int coarse_above = histogram[coarse_threshold + 1];
  int remaining = topk - coarse_above;

  if (remaining == 0) {
    for (int idx = tx; idx < length; idx += BlockSize) {
      if (coarse_key(input[row_start + idx]) > coarse_threshold) {
        const int pos = atomicAdd(&counter, 1);
        if (pos < topk) output[pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

  const int coarse_population = histogram[coarse_threshold] - histogram[coarse_threshold + 1];
  __syncthreads();
  if (coarse_population >= StashEntries) {
    // Emit the values above the coarse threshold once. The oversized bin is
    // then refined without intermediate emissions: four histogram passes find
    // the exact cutoff, and one final pass writes the strictly-greater values
    // and the required number of exact ties into disjoint output ranges.
    for (int idx = tx; idx < length; idx += BlockSize) {
      if (coarse_key(input[row_start + idx]) > coarse_threshold) {
        const int pos = atomicAdd(&counter, 1);
        if (pos < topk) output[pos] = idx;
      }
    }
    __syncthreads();

    uint32_t prefix = 0;
    int strict_count = 0;
#pragma unroll 4
    for (int level = 0; level < 4; ++level) {
      const int shift = 24 - level * 8;
      const uint32_t prefix_mask = level == 0 ? 0u : (~0u << (32 - level * 8));

      if (tx < kRadix + 1) histogram[tx] = 0;
      if (tx == 0) threshold_bin_id = -1;
      __syncthreads();

      for (int idx = tx; idx < length; idx += BlockSize) {
        const float value = input[row_start + idx];
        if (coarse_key(value) != coarse_threshold) continue;
        const uint32_t key = exact_key(value);
        if ((key & prefix_mask) != prefix) continue;
        atomicAdd(&histogram[(key >> shift) & 0xFFu], 1);
      }
      __syncthreads();

      run_cumsum();
      if (tx < kRadix && histogram[tx] >= remaining && histogram[tx + 1] < remaining) {
        threshold_bin_id = tx;
      }
      __syncthreads();

      const int threshold = threshold_bin_id;
      if (threshold < 0) return;
      const int above = histogram[threshold + 1];
      strict_count += above;
      remaining -= above;
      prefix |= static_cast<uint32_t>(threshold) << shift;
      // Every thread consumes the shared threshold/cumulative histogram above;
      // do not let a faster warp clear them for the next refinement round.
      __syncthreads();
    }

    if (tx == 0) {
      num_input[0] = 0;
      num_input[1] = 0;
    }
    __syncthreads();
    for (int idx = tx; idx < length; idx += BlockSize) {
      const float value = input[row_start + idx];
      if (coarse_key(value) != coarse_threshold) continue;
      const uint32_t key = exact_key(value);
      if (key > prefix) {
        const int pos = atomicAdd(&num_input[0], 1);
        if (pos < strict_count) output[coarse_above + pos] = idx;
      } else if (key == prefix) {
        const int pos = atomicAdd(&num_input[1], 1);
        if (pos < remaining) output[coarse_above + strict_count + pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

  // The complete coarse threshold bin fits below capacity: keep the original
  // stash/refine path, now with unconditional state initialization and bounded
  // emissions.
  if (tx < kRadix + 1) histogram[tx] = 0;
  __syncthreads();
  for (int idx = tx; idx < length; idx += BlockSize) {
    const float value = input[row_start + idx];
    const int bin = coarse_key(value);
    if (bin > coarse_threshold) {
      const int pos = atomicAdd(&counter, 1);
      if (pos < topk) output[pos] = idx;
    } else if (bin == coarse_threshold) {
      const int pos = atomicAdd(&num_input[0], 1);
      if (pos < StashEntries) {
        input_idx[0][pos] = idx;
        atomicAdd(&histogram[(exact_key(value) >> 24) & 0xFFu], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const int current = round & 1;
    const int next = current ^ 1;
    const int raw_num_input = num_input[current];
    const int current_count = raw_num_input < StashEntries ? raw_num_input : StashEntries;

    if (tx == 0) {
      threshold_bin_id = -1;
      num_input[next] = 0;
    }
    run_cumsum();
    if (tx < kRadix && histogram[tx] > remaining && histogram[tx + 1] <= remaining) {
      threshold_bin_id = tx;
    }
    __syncthreads();

    const int threshold = threshold_bin_id;
    if (threshold < 0) return;
    const int above = histogram[threshold + 1];
    const int shift = 24 - round * 8;

    __syncthreads();
    if (round < 3 && tx < kRadix + 1) histogram[tx] = 0;
    __syncthreads();
    for (int i = tx; i < current_count; i += BlockSize) {
      const int idx = input_idx[current][i];
      const float value = input[row_start + idx];
      const int bin = static_cast<int>((exact_key(value) >> shift) & 0xFFu);
      if (bin > threshold) {
        const int pos = atomicAdd(&counter, 1);
        if (pos < topk) output[pos] = idx;
      } else if (bin == threshold && round < 3) {
        const int pos = atomicAdd(&num_input[next], 1);
        if (pos < StashEntries) {
          input_idx[next][pos] = idx;
          atomicAdd(&histogram[(exact_key(value) >> (shift - 8)) & 0xFFu], 1);
        }
      }
    }
    __syncthreads();

    remaining -= above;
    if (remaining == 0) return;
    if (round == 3) {
      for (int i = tx; i < current_count; i += BlockSize) {
        const int idx = input_idx[current][i];
        if (static_cast<int>(exact_key(input[row_start + idx]) & 0xFFu) == threshold) {
          const int pos = atomicAdd(&counter, 1);
          if (pos < topk) output[pos] = idx;
        }
      }
      __syncthreads();
      return;
    }
  }
}

}  // namespace sglang::device::legacy_radix_topk
