/// \file warp.cuh
/// \brief Warp-level reduction and cooperative-copy primitives.

#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

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

/// \brief Warp-cooperative gmem -> smem copy of a compile-time byte count.
///
/// Picks the widest vector width that divides both the per-thread share and
/// the byte total. The caller guarantees ``src`` is aligned to the picked
/// width (16B for kBytes % (16*32) == 0, else 8/4) and ``dst`` is the start
/// of a 16B-aligned per-warp smem slot.
// Warp-cooperative byte copy between any two address spaces, vectorised to the
// widest unit `kBytes` allows. Named for what it does rather than where it is
// used: the MLA call sites happen to target shared memory, but nothing here is
// global->shared specific -- no cp.async, no TMA, the payload moves through
// registers.
//
// The strategy was measured against the two async alternatives on B300 (sm_103,
// 148 SMs), copying one MLA row per warp out of a 512 MB pool so every row
// streams from HBM (grid 296, 64 rows/warp, 50 launches):
//
//   1152 B/warp (bf16, nope 1024 + rope 128)   576 B/warp (fp8, 512 + 64)
//     this (generic)     47.3 us  3.69 TB/s      40.8 us  2.14 TB/s
//     cp.async (ldgsts)  73.9 us  2.36 TB/s      56.4 us  1.55 TB/s
//     cp.async.bulk/TMA  50.0 us  3.50 TB/s      43.2 us  2.02 TB/s
//
// The generic path wins at both sizes: a ~1 KB row is too small to amortise
// cp.async's per-lane 16 B issues or TMA's fixed issue plus mbarrier round trip.
// Revisit if a call site ever copies substantially more than one row per warp.
template <int64_t kBytes>
SGL_DEVICE void copy_bytes(const void* __restrict__ src, void* __restrict__ dst) {
  constexpr int64_t kAlignment = (kBytes % (16 * kWarpThreads) == 0)  ? 16
                                 : (kBytes % (8 * kWarpThreads) == 0) ? 8
                                 : (kBytes % (4 * kWarpThreads) == 0) ? 4
                                 : (kBytes % 4 == 0)                  ? 4
                                                                      : 0;
  static_assert(kAlignment > 0, "kBytes must be a multiple of 4");

  using vec_t = AlignedStorage<uint32_t, kAlignment / 4>;
  constexpr auto kLoopBytes = sizeof(vec_t) * kWarpThreads;
  constexpr auto kLoopCount = kBytes / kLoopBytes;
  constexpr int64_t kTailVecs = (kBytes - kLoopCount * kLoopBytes) / sizeof(vec_t);

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto v = gmem.load(src, i);
    gmem.store(dst, v, i);
  }
  if constexpr (kTailVecs > 0) {
    if (gmem.in_bound(kLoopCount * kWarpThreads + kTailVecs, kLoopCount)) {
      const auto v = gmem.load(src, kLoopCount);
      gmem.store(dst, v, kLoopCount);
    }
  }
}

/// Inclusive prefix sum across one warp, thread-rank order. Distinct from
/// reduce_sum above: every lane keeps its own running total rather than the
/// whole-warp result.
SGL_DEVICE uint32_t inclusive_sum(uint32_t lane_id, uint32_t val) {
  static_assert(kWarpThreads == 32);
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
#ifndef USE_ROCM
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
#else
    uint32_t n = __shfl_up_sync(kFullMask, val, offset, kWarpThreads);
#endif
    if (lane_id >= offset) val += n;
  }
  return val;
}

// One elected lane, via elect.sync. Raw PTX rather than cute::elect_one_sync,
// which would drag the whole CuTe include path into elementwise JIT modules;
// cuda::ptx has no elect_sync in CUDA 13.0. Use this to gate a single-thread
// TMA issue instead of a lane-index predicate.
SGL_DEVICE bool elect_one_lane() {
  uint32_t pred;
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  .reg .b32  r;\n"
      "  elect.sync r|p, 0xFFFFFFFF;\n"
      "  selp.b32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(pred));
  return pred != 0;
}

}  // namespace device::warp
