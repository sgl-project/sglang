#pragma once
#include <sgl_kernel/math.cuh>

// Some warp primitives
namespace device::warp {

static constexpr uint32_t kFullMask = 0xffffffffu;

template <typename T>
SGL_DEVICE T reduce_sum(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    value = value + __shfl_xor_sync(active_mask, value, mask, 32);
  return value;
}

template <typename T>
SGL_DEVICE T reduce_max(T value, uint32_t active_mask = kFullMask) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    value = math::max(value, __shfl_xor_sync(active_mask, value, mask, 32));
  return value;
}

}  // namespace device::warp
