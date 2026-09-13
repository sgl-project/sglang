/// \file runtime.cuh
/// \brief Host-side CUDA runtime query helpers.
///
/// Thin wrappers around CUDA occupancy and device-property APIs with
/// automatic error checking via `CHECK_CUDA`.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <utility>
#ifndef USE_ROCM
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#endif
#ifndef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#endif
#ifndef cudaOccupancyAvailableDynamicSMemPerBlock
inline hipError_t
cudaOccupancyAvailableDynamicSMemPerBlock(std::size_t* smem, const void* func, int num_blocks, int block_size) {
  // HIP does not expose this directly; return max shared mem as conservative estimate
  hipDeviceProp_t prop;
  int device;
  hipGetDevice(&device);
  hipGetDeviceProperties(&prop, device);
  *smem = prop.sharedMemPerBlock;
  return hipSuccess;
}
#endif
#endif

namespace sglang {

namespace host::runtime {

namespace details {

template <typename T, T kDefault>
struct DeviceCacheMap {
 public:
  // Generous bound on the device ordinals one process can see; a larger ordinal
  // is not an error, it just falls through to the driver query uncached.
  static constexpr uint32_t kNumStaticMaxDevice = 72;

  constexpr DeviceCacheMap() {
    for (uint32_t i = 0; i < kNumStaticMaxDevice; ++i) {
      m_data[i] = kDefault;
    }
  }

  template <typename Fn>
  T get_cached(int32_t device_, bool use_cache, Fn&& fn) {
    const auto device = static_cast<uint32_t>(device_);
    if (use_cache && device < kNumStaticMaxDevice && m_data[device] != kDefault) {
      return m_data[device];
    }
    const auto value = static_cast<T>(std::forward<Fn>(fn)(device_));
    if (device < kNumStaticMaxDevice) {
      m_data[device] = value;
    }
    return value;
  }

 private:
  T m_data[kNumStaticMaxDevice];
};

}  // namespace details

// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of SMs for the given device
inline auto get_sm_count(int device_id, bool use_cache = true) -> uint32_t {
  static details::DeviceCacheMap<uint32_t, 0> sm_count_cache;
  return sm_count_cache.get_cached(device_id, use_cache, [](int32_t device_id) {
    int sm_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
    return sm_count;
  });
}

// Return the Major compute capability for the given device
inline auto get_cc_major(int device_id, bool use_cache = true) -> int {
  static details::DeviceCacheMap<int, -1> cc_major_cache;
  return cc_major_cache.get_cached(device_id, use_cache, [](int32_t device_id) {
    int cc_major;
    CHECK_CUDA(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id));
    return cc_major;
  });
}

// Return the Minor compute capability for the given device
inline auto get_cc_minor(int device_id, bool use_cache = true) -> int {
  static details::DeviceCacheMap<int, -1> cc_minor_cache;
  return cc_minor_cache.get_cached(device_id, use_cache, [](int32_t device_id) {
    int cc_minor;
    CHECK_CUDA(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device_id));
    return cc_minor;
  });
}

// Return the SM version (major * 10 + minor) for the given device
inline auto get_sm_version(int device_id, bool use_cache = true) -> int {
  return get_cc_major(device_id, use_cache) * 10 + get_cc_minor(device_id, use_cache);
}

// Return the runtime version
inline auto get_runtime_version() -> int {
  int runtime_version;
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

// Return the maximum dynamic shared memory per block for the given kernel
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  CHECK_CUDA(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
  return smem_size;
}

}  // namespace host::runtime

}  // namespace sglang
