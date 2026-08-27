#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

namespace device::cta {

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
