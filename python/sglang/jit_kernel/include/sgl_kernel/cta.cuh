/// \file cta.cuh
/// \brief CTA (Cooperative Thread Array / thread-block) level primitives.

#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

namespace device::cta {

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
