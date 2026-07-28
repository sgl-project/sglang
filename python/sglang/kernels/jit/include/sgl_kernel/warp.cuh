/// \file warp.cuh
/// \brief Warp-level reduction primitives.

#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <type_traits>

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
 * \brief Warp-level reduction.
 *
 * On CUDA: uses __shfl_xor_sync with width=32. Full-warp reductions
 * use a single `redux.sync` instruction when the target supports it.
 * On HIP: uses __shfl_xor with explicit width parameter (supports wave64 sub-groups).
 * \tparam OP Reduction operation to perform (SUM, MAX, MIN).
 * \tparam kNumThreads Number of threads as a group.
 * \tparam kInner Whether to perform within a group or not.
 * \tparam T Type of the value to reduce.
 *
 * \param value The value to reduce.
 * \param active_mask The active mask of threads participating in the reduction.
 *
 * \note We will divide into groups of `kNumThreads`.
 * e.g. kNumThreads = 8, we have 0..7, 8..15, 16..23, 24..31 as groups.
 * By reduction is performed within a group. Inter-group reduction will reduce
 * over the same offset in different groups. e.g. {0, 8, 16, 24} in the above example.
 */
template <ReductionOp OP, uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce(T value, mask_t active_mask = kFullMask) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
  static_assert(std::has_single_bit(kNumThreads), "must be pow of 2");
  using Trait = ReductionTrait<OP, T>;

#ifdef SGL_CUDA_ARCH
  // CUDA target only
  constexpr bool kFullReduction = (kNumThreads == kWarpThreads && kInner) || (kNumThreads == 1 && !kInner);
  if constexpr (kFullReduction) {
#if SGL_CUDA_ARCH >= 800
    // 32 bit integer reduction
    if constexpr (std::is_integral_v<T> && sizeof(T) <= 4) {
      if constexpr (OP == ReductionOp::SUM) {
        return __reduce_add_sync(active_mask, value);
      } else if constexpr (OP == ReductionOp::MAX) {
        return __reduce_max_sync(active_mask, value);
      } else if constexpr (OP == ReductionOp::MIN) {
        return __reduce_min_sync(active_mask, value);
      }
    }
#endif
#if SGL_CUDA_ARCH >= 1000 && SGL_CUDA_ARCH < 1100
    // 32-bit float reduction
    if constexpr (std::is_same_v<T, float>) {
      if constexpr (OP == ReductionOp::MAX) {
        float result;
        asm("redux.sync.max.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "r"(active_mask));
        return result;
      } else if constexpr (OP == ReductionOp::MIN) {
        float result;
        asm("redux.sync.min.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "r"(active_mask));
        return result;
      }
    }
#endif
  }
#endif  // redux.sync for CUDA only

  if constexpr (kInner) {
#pragma unroll
    for (uint32_t mask = kNumThreads / 2; mask >= 1; mask >>= 1) {
#ifndef USE_ROCM
      value = Trait::reduce(value, __shfl_xor_sync(active_mask, value, mask, 32));
#else
      value = Trait::reduce(value, __shfl_xor(value, mask, kNumThreads));
#endif
    }
  } else {
#pragma unroll
    for (uint32_t mask = kNumThreads; mask <= kWarpThreads / 2; mask <<= 1) {
#ifndef USE_ROCM
      value = Trait::reduce(value, __shfl_xor_sync(active_mask, value, mask, 32));
#else
      // Inter-group shuffle crosses kNumThreads-sized sub-groups, so the
      // shuffle width must span the whole warp.
      value = Trait::reduce(value, __shfl_xor(value, mask, kWarpThreads));
#endif
    }
  }
  return value;
}

/** \brief Warp-level sum reduction. */
template <uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce_sum(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::SUM, kNumThreads, kInner>(value, active_mask);
}

/** \brief Warp-level max reduction. */
template <uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce_max(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::MAX, kNumThreads, kInner>(value, active_mask);
}

/** \brief Warp-level min reduction. */
template <uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce_min(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::MIN, kNumThreads, kInner>(value, active_mask);
}

}  // namespace device::warp
