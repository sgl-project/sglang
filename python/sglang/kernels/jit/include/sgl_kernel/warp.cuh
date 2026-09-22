/// \file warp.cuh
/// \brief Warp-level reduction and cooperative-copy primitives.

#pragma once
#include <sgl_kernel/bits.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <numeric>
#include <type_traits>

namespace sglang {

namespace device::warp {

/// \brief Full warp active mask and lane count: 32 on CUDA, wave64 on HIP.
#ifndef USE_ROCM
inline constexpr uint32_t kFullMask = 0xffffffffu;
inline constexpr uint32_t kFullWidth = 32u;
using mask_t = uint32_t;
#else
inline constexpr uint64_t kFullMask = 0xffffffffffffffffULL;
inline constexpr uint32_t kFullWidth = 64u;
using mask_t = uint64_t;
#endif

using ::sglang::device::get_lane_id;

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

/**
 * \brief Warp-level reduction over the lane range spanned by two widths.
 *
 * `kStart` and `kFinish` bound a range of lane-index bits: lane `i` reduces with
 * every lane differing from it only in bits `[log2(lo), log2(hi))`, where `lo`
 * and `hi` are the smaller and larger of the two. So `<N, 1>` reduces within
 * contiguous groups of `N` lanes (`0..7, 8..15, ...` for `N = 8`), while
 * `<kFullWidth, N>` reduces across groups at the same offset (`{0, 8, 16, 24}`).
 *
 * \tparam OP Reduction operation to perform (SUM, MAX, MIN).
 * \tparam kStart One end of the reduced lane range; power of two.
 * \tparam kFinish The other end of the reduced lane range; power of two.
 * \tparam T Type of the value to reduce.
 *
 * \param value The value to reduce.
 * \param active_mask The active mask of threads participating in the reduction.
 *
 * \note Symmetric: `<kStart, kFinish>` and `<kFinish, kStart>` reduce the same
 * lane set, only walking the range in the opposite order.
 * \note On CUDA a whole-warp reduction lowers to a single `redux.sync` where the
 * target supports it. On HIP the shuffles use `__shfl_xor` with max width 64.
 */
template <ReductionOp OP, uint32_t kStart = kWarpThreads, uint32_t kFinish = 1, typename T>
SGL_DEVICE T reduce(T value, mask_t active_mask = kFullMask) {
  static_assert(host::is_pow2(kStart) && host::is_pow2(kFinish));
  static_assert(kStart <= kFullWidth && kFinish <= kFullWidth);
  using Trait = ReductionTrait<OP, T>;

#ifdef SGL_CUDA_ARCH
  // CUDA target only
  constexpr bool kFullReduction = (kStart == 1 && kFinish == kFullWidth) || (kStart == kFullWidth && kFinish == 1);
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

  if constexpr (kStart > kFinish) {
#pragma unroll
    for (uint32_t mask = kStart / 2; mask >= kFinish; mask >>= 1) {
#ifndef USE_ROCM
      value = Trait::reduce(value, __shfl_xor_sync(active_mask, value, mask, kStart));
#else
      value = Trait::reduce(value, __shfl_xor(value, mask, kStart));
#endif
    }
  } else {
#pragma unroll
    for (uint32_t mask = kStart; mask <= kFinish / 2; mask <<= 1) {
#ifndef USE_ROCM
      value = Trait::reduce(value, __shfl_xor_sync(active_mask, value, mask, kFinish));
#else
      value = Trait::reduce(value, __shfl_xor(value, mask, kFinish));
#endif
    }
  }
  return value;
}

/** \brief Warp-level sum reduction. */
template <uint32_t kStart = kWarpThreads, uint32_t kFinish = 1, typename T>
SGL_DEVICE T reduce_sum(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::SUM, kStart, kFinish>(value, active_mask);
}

/** \brief Warp-level max reduction. */
template <uint32_t kStart = kWarpThreads, uint32_t kFinish = 1, typename T>
SGL_DEVICE T reduce_max(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::MAX, kStart, kFinish>(value, active_mask);
}

/** \brief Warp-level min reduction. */
template <uint32_t kStart = kWarpThreads, uint32_t kFinish = 1, typename T>
SGL_DEVICE T reduce_min(T value, mask_t active_mask = kFullMask) {
  return reduce<ReductionOp::MIN, kStart, kFinish>(value, active_mask);
}

/**
 * \brief Inclusive scan within each segment of `kWidth` lanes.
 *
 * Distinct from `reduce` above: every lane keeps its own running total rather
 * than the whole-group result. Scans forward when `kStart < kFinish` (lane `p`
 * accumulates lanes `<= p`) and backward when `kStart > kFinish` (lane `p`
 * accumulates lanes `>= p`). The default `<kWidth, 1, kWidth>` is a plain
 * forward scan over the whole segment.
 *
 * \tparam OP Reduction operation to combine with (SUM, MAX, MIN).
 * \tparam kWidth Segment size; a scan never crosses a segment boundary.
 * \tparam kStart First shuffle offset; power of two. `kStart > 1` assumes the
 * input is already scanned in blocks of `kStart`, and scans the strided
 * subsequences instead.
 * \tparam kFinish Exclusive bound on the shuffle offset; power of two.
 * \tparam T Type of the value to scan.
 *
 * \param val The value to scan.
 * \param lane_id This thread's index WITHIN its segment, i.e.
 * `threadIdx.x % kWidth` -- not `% kWarpThreads`. The shuffles are
 * segment-relative but this predicate is not, so a warp-relative id silently
 * corrupts every segment past the first whenever `kWidth < kWarpThreads`.
 * \param active_mask The active mask of threads participating in the scan.
 *
 * \note The backward direction accumulates over `[p, kStart)`, so it covers the
 * whole segment only when `kStart == kWidth`.
 */
template <ReductionOp OP, uint32_t kWidth = kWarpThreads, uint32_t kStart = 1, uint32_t kFinish = kWidth, typename T>
SGL_DEVICE T inclusive_reduce(T val, uint32_t lane_id = get_lane_id<kWidth>(), mask_t active_mask = kFullMask) {
  static_assert(host::is_pow2(kStart) && host::is_pow2(kFinish));
  static_assert(kStart <= kWidth && kFinish <= kWidth && kWidth <= kFullWidth);
  using Trait = ReductionTrait<OP, T>;
  if constexpr (kStart < kFinish) {
#pragma unroll
    for (uint32_t offset = kStart; offset < kFinish; offset *= 2) {
      const auto n = __shfl_up_sync(active_mask, val, offset, kWidth);
      if (lane_id >= offset) val = Trait::reduce(val, n);
    }
  } else {
#pragma unroll
    for (uint32_t offset = kFinish; offset < kStart; offset *= 2) {
      const auto n = __shfl_down_sync(active_mask, val, offset, kWidth);
      if (lane_id < kStart - offset) val = Trait::reduce(val, n);
    }
  }
  return val;
}

template <uint32_t kWidth = kWarpThreads, uint32_t kStart = 1, uint32_t kFinish = kWidth, typename T>
SGL_DEVICE T inclusive_sum(T val, uint32_t lane_id = get_lane_id<kWidth>(), mask_t active_mask = kFullMask) {
  return inclusive_reduce<ReductionOp::SUM, kWidth, kStart, kFinish>(val, lane_id, active_mask);
}

template <uint32_t kWidth = kWarpThreads, uint32_t kStart = 1, uint32_t kFinish = kWidth, typename T>
SGL_DEVICE T inclusive_max(T val, uint32_t lane_id = get_lane_id<kWidth>(), mask_t active_mask = kFullMask) {
  return inclusive_reduce<ReductionOp::MAX, kWidth, kStart, kFinish>(val, lane_id, active_mask);
}

template <uint32_t kWidth = kWarpThreads, uint32_t kStart = 1, uint32_t kFinish = kWidth, typename T>
SGL_DEVICE T inclusive_min(T val, uint32_t lane_id = get_lane_id<kWidth>(), mask_t active_mask = kFullMask) {
  return inclusive_reduce<ReductionOp::MIN, kWidth, kStart, kFinish>(val, lane_id, active_mask);
}

/**
 * \brief Broadcast one lane's value to every lane of its `kWidth` segment.
 * \param src_lane The source's index WITHIN the segment, i.e. in `[0, kWidth)`;
 * each segment reads its own lane `src_lane`, not a single warp-wide one.
 */
template <uint32_t kWidth = kWarpThreads, typename T>
SGL_DEVICE T broadcast(T value, uint32_t src_lane, mask_t active_mask = kFullMask) {
  static_assert(host::is_pow2(kWidth) && kWidth <= kFullWidth);
  return __shfl_sync(active_mask, value, src_lane, kWidth);
}

namespace details {

// array for load & store operations
template <typename T, std::size_t N, int64_t kBytes>
struct Array : public DeviceArray<T, N> {
  static_assert(alignof(T) == sizeof(T));
  static constexpr int64_t kVecBytes = static_cast<int64_t>(sizeof(T)) * N;
};

}  // namespace details

template <int64_t kBytes, int64_t kVecBytes>
struct CopyTrait {
  static_assert(kBytes % kVecBytes == 0 && kBytes > 0);
  using vec_t = AlignedStorage<uint8_t, kVecBytes>;
  static constexpr int64_t kLoopBytes = sizeof(vec_t) * kWarpThreads;
  static constexpr int64_t kLoopCount = kBytes / kLoopBytes;
  static constexpr int64_t kTailBytes = kBytes - kLoopCount * kLoopBytes;
  static constexpr int64_t kTailVecs = kTailBytes / sizeof(vec_t);
  using result_t = details::Array<vec_t, kLoopCount + (kTailVecs > 0 ? 1 : 0), kBytes>;

  template <typename F>
  SGL_DEVICE static void for_each(F&& f) {
    const auto mem = tile::Memory<vec_t>::warp();
#pragma unroll
    for (int64_t i = 0; i < kLoopCount; ++i) {
      f(mem, i);
    }
    if constexpr (kTailVecs > 0) {
      if (mem.in_bound(kBytes / sizeof(vec_t), kLoopCount)) {
        f(mem, kLoopCount);
      }
    }
  }

  SGL_DEVICE static result_t load(const void* src) {
    result_t result;
    for_each([&](const auto& mem, int64_t i) { result[i] = mem.load(src, i); });
    return result;
  }

  SGL_DEVICE static void store(void* dst, const result_t& result) {
    for_each([&](const auto& mem, int64_t i) { mem.store(dst, result[i], i); });
  }
};

struct LoadStorePattern {
  using enum LoadStoreBytes::type;
  enum type : int64_t {
    WARP_UNIFORM_GMEM = -MAX_GMEM,
    WARP_UNIFORM_SMEM = -MAX_SMEM,
    WARP_UNIFORM_4B = -4,
    WARP_UNIFORM_8B = -8,
    WARP_UNIFORM_16B = -16,
    WARP_UNIFORM_32B = -32,
  };

  template <int64_t kBytes, int64_t kMaxVecBytes>
  SGL_DEVICE_HOST static constexpr int64_t get_vec_bytes() {
    if constexpr (kMaxVecBytes < 0) {  // best-effort warp uniform load/store
      if constexpr (kBytes % (4 * device::kWarpThreads) != 0) {
        // at least guarantee 128B coalesced for better performance
        return std::gcd(kBytes, 4);
      } else {  // kBytes is at least 128B coalesced
        return std::gcd(kBytes / device::kWarpThreads, -kMaxVecBytes);
      }
    } else {
      return std::gcd(kBytes, kMaxVecBytes);
    }
  }
};

template <
    int64_t kBytes,
    int64_t kMaxVecBytes = LoadStorePattern::MAX_GMEM,
    int64_t kVecBytes = LoadStorePattern::get_vec_bytes<kBytes, kMaxVecBytes>()>
SGL_DEVICE auto load_bytes(const void* src) {
  return CopyTrait<kBytes, kVecBytes>::load(src);
}

template <
    int64_t kBytes,
    int64_t kMaxVecBytes = LoadStorePattern::MAX_GMEM,
    int64_t kVecBytes = LoadStorePattern::get_vec_bytes<kBytes, kMaxVecBytes>()>
SGL_DEVICE void store_bytes(void* dst, const auto& result) {
  return CopyTrait<kBytes, kVecBytes>::store(dst, result);
}

}  // namespace device::warp

}  // namespace sglang
