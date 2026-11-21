#pragma once

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>

namespace sglang::device::distributed::pull {

/**
 * \brief Row sums of one float per thread for a group of up to `kRowsPerGroup` rows held by one CTA of `kNumWarps`
 *        warps:
 *        `publish` runs the rows' warp butterflies interleaved, lane 0 of each warp stores its sums, one
 *        `__syncthreads` for the group; `total(u)` then adds row `u`'s warp sums in warp order (every thread of a row
 *        gets the same bits). Read the totals row by row where they are used: reading every row's first makes ptxas
 *        hoist all `kRowsPerGroup * kNumWarps` shared loads and spill.
 *
 * Double buffer: `phase` counts the kernel's groups and picks the buffer, `index < kNumSums` the sum within a row.
 * Buffer `(p, i)` written by group `g` is written again by group `g + 2` only; every thread reads group `g`'s sums
 * before it reaches group `g + 1`'s barrier, which every writer of group `g + 2` has passed.
 */
template <uint32_t kRowsPerGroup, uint32_t kNumWarps, uint32_t kNumSums>
struct GroupRowSum {
 public:
  template <uint32_t kNumRows>
  SGL_DEVICE void publish(float (&value)[kNumRows], uint32_t phase, uint32_t index = 0) {
    static_assert(kNumRows <= kRowsPerGroup);
    const uint32_t warp_id = threadIdx.x / kWarpThreads;
    auto& partial = m_partial[phase % 2][index];
#pragma unroll
    for (uint32_t u = 0; u < kNumRows; ++u) {
      value[u] = warp::reduce_sum(value[u]);
    }
    if (threadIdx.x % kWarpThreads == 0) {
#pragma unroll
      for (uint32_t u = 0; u < kNumRows; ++u) {
        partial[u][warp_id] = value[u];
      }
    }
    __syncthreads();
  }

  SGL_DEVICE float total(uint32_t u, uint32_t phase, uint32_t index = 0) const {
    const auto& partial = m_partial[phase % 2][index][u];
    float total = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      total += partial[w];
    }
    return total;
  }

 private:
  float m_partial[2][kNumSums][kRowsPerGroup][kNumWarps];
};

}  // namespace sglang::device::distributed::pull
