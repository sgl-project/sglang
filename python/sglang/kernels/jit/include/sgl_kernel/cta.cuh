/// \file cta.cuh
/// \brief CTA (Cooperative Thread Array / thread-block) level primitives.

#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

namespace sglang {

namespace device::cta {

/**
 * \brief Compute the sum of `value` across all threads in a fixed-size CTA.
 *
 * Uses a two-level reduction: first within each warp via `warp::reduce_sum`,
 * then one warp-wide reduction over the per-warp partials. The result is
 * returned by every thread without a trailing `__syncthreads()`.
 *
 * \tparam kNumThreads Threads in the CTA; must be a multiple of the warp size.
 * \tparam T Numeric type supported by `warp::reduce_sum`.
 * \param value Per-thread input value.
 * \param smem Shared-memory buffer with at least `kNumThreads / 32` elements.
 */
template <int kNumThreads, typename T>
SGL_DEVICE T reduce_sum(T value, T* smem) {
  static_assert(kNumThreads % kWarpThreads == 0);
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;
  value = warp::reduce_sum(value);
  if constexpr (kNumWarps == 1) {
    return value;
  } else {
    const auto lane = threadIdx.x % kWarpThreads;
    if (lane == 0) smem[threadIdx.x / kWarpThreads] = value;
    __syncthreads();
    return warp::reduce_sum(lane < kNumWarps ? smem[lane] : T{0});
  }
}

/**
 * \brief Compute the maximum of `value` across all threads in the CTA.
 *
 * Uses a two-level reduction: first within each warp via `warp::reduce_max`,
 * then across warps using shared memory. The final result is stored in
 * `smem[0]`.
 *
 * \tparam T Numeric type (must be supported by `warp::reduce_max`).
 * \param value Per-thread input value.
 * \param smem Shared memory buffer (must have at least `blockDim.x / 32`
 *             elements).
 * \param min_value Identity element for max (default 0.0f).
 * \note This function does NOT issue a trailing `__syncthreads()`.
 *       Callers must synchronize before reading `smem[0]`.
 */
template <typename T>
SGL_DEVICE void reduce_max(T value, float* smem, float min_value = 0.0f) {
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  smem[warp_id] = warp::reduce_max(value);
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    const auto local_value = tx * kWarpThreads < blockDim.x ? smem[tx] : min_value;
    const auto max_value = warp::reduce_max(local_value);
    smem[0] = max_value;
  }
  // no extra sync; it is caller's responsibility to sync if needed
}

}  // namespace device::cta

}  // namespace sglang
