#pragma once
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstddef>

// Some warp primitives
namespace device::warp {

template <typename T>
__forceinline__ __device__ T reduce_sum(T val, uint32_t active_mask = 0xffffffff) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(active_mask, val, mask, 32);
  return val;
}

template <typename T, std::size_t kThreads = kWarpThreads>
__forceinline__ __device__ T load(const void* ptr) {
  return static_cast<const T*>(ptr)[threadIdx.x % kWarpThreads];
}

template <std::size_t kThreads = kWarpThreads, typename T>
__forceinline__ __device__ T load(const T* ptr) {
  return static_cast<const T*>(ptr)[threadIdx.x % kWarpThreads];
}

template <std::size_t kThreads = kWarpThreads, typename T>
__forceinline__ __device__ void store(void* ptr, T val) {
  static_cast<T*>(ptr)[threadIdx.x % kWarpThreads] = val;
}

}  // namespace device::warp
