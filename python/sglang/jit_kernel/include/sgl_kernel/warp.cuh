#pragma once

// Some warp primitives
namespace device::warp {

template <typename T>
__always_inline __device__ T reduce_sum(T val, uint32_t active_mask = 0xffffffff) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(active_mask, val, mask, 32);
  return val;
}

}  // namespace device::warp
