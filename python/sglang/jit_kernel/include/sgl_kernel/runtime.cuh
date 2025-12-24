#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>

namespace host::runtime {

template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  RuntimeDeviceCheck(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

inline auto get_sm_count(int device_id) -> uint32_t {
  cudaDeviceProp device_prop;
  RuntimeDeviceCheck(cudaGetDeviceProperties(&device_prop, device_id));
  return static_cast<uint32_t>(device_prop.multiProcessorCount);
}

}  // namespace host::runtime
