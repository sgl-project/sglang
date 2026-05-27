/// \file warp.cuh
/// \brief Warp-level reduction primitives.

#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>

namespace device::warp {

/// \brief Full warp active mask.
#ifndef USE_ROCM
static constexpr uint32_t kFullMask = 0xffffffffu;
using mask_t = uint32_t;
#else
static constexpr uint64_t kFullMask = 0xffffffffffffffffULL;
using mask_t = uint64_t;
#endif

/**
 * \brief Warp-level sum reduction.
 *
 * On CUDA: uses __shfl_xor_sync with width=32.
 * On HIP: uses __shfl_xor with explicit width parameter (supports wave64 sub-groups).
 */
template <uint32_t kNumThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_sum(T value, mask_t active_mask = kFullMask) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
  static_assert(std::has_single_bit(kNumThreads), "must be pow of 2");
#pragma unroll
  for (int mask = kNumThreads / 2; mask > 0; mask >>= 1)
#ifndef USE_ROCM
    value = value + __shfl_xor_sync(active_mask, value, mask, 32);
#else
    value = value + __shfl_xor(value, mask, kNumThreads);
#endif
  return value;
}

/**
 * \brief Warp-level max reduction.
 */
template <uint32_t kNumThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_max(T value, mask_t active_mask = kFullMask) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
  static_assert(std::has_single_bit(kNumThreads), "must be pow of 2");
#pragma unroll
  for (int mask = kNumThreads / 2; mask > 0; mask >>= 1)
#ifndef USE_ROCM
    value = math::max(value, __shfl_xor_sync(active_mask, value, mask, 32));
#else
    value = math::max(value, __shfl_xor(value, mask, kNumThreads));
#endif
  return value;
}

}  // namespace device::warp
