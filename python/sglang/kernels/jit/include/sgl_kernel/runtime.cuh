/// \file runtime.cuh
/// \brief Host-side CUDA runtime query helpers.
///
/// Thin wrappers around CUDA occupancy and device-property APIs with
/// automatic error checking via `CHECK_CUDA`.

#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <atomic>
#include <concepts>
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

template <auto V>
struct Tag {};

struct MaybeDevice {
  MaybeDevice(DLDevice device) : device_id(device.device_id) {}
  MaybeDevice(int32_t device_id_) : device_id(device_id_) {}
  uint32_t device_id;
};

}  // namespace details

inline constexpr uint32_t kNumStaticMaxDevice = 72;

template <typename T, T kDefault>
struct DeviceCacheMap {
 public:
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

  template <typename Fn>
  T get_cached(int32_t device_, Fn&& fn) {
    return this->get_cached(device_, true, std::forward<Fn>(fn));
  }

 private:
  T m_data[kNumStaticMaxDevice];
};

inline void* get_device_accessible_ptr(const tvm::ffi::TensorView& tensor) {
  void* ptr = tensor.data_ptr();
  const auto tensor_device_type = tensor.device().device_type;
  if (tensor_device_type != kDLCPU && tensor_device_type != kDLGPUHost) {
    return ptr;
  }

  void* device_ptr = nullptr;
#ifdef USE_ROCM
  CHECK_CUDA(::hipHostGetDevicePointer(&device_ptr, ptr, 0));
#else
  CHECK_CUDA(::cudaHostGetDevicePointer(&device_ptr, ptr, 0));
#endif
  return device_ptr;
}

// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of SMs for the given device
inline auto get_sm_count(int device_id, bool use_cache = true) -> uint32_t {
  static DeviceCacheMap<uint32_t, 0> sm_count_cache;
  return sm_count_cache.get_cached(device_id, use_cache, [](int32_t device_id) {
    int sm_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
    return sm_count;
  });
}

// Return the Major compute capability for the given device
inline auto get_cc_major(int device_id, bool use_cache = true) -> int {
  static DeviceCacheMap<int, -1> cc_major_cache;
  return cc_major_cache.get_cached(device_id, use_cache, [](int32_t device_id) {
    int cc_major;
    CHECK_CUDA(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id));
    return cc_major;
  });
}

// Return the Minor compute capability for the given device
inline auto get_cc_minor(int device_id, bool use_cache = true) -> int {
  static DeviceCacheMap<int, -1> cc_minor_cache;
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
inline auto get_available_dynamic_smem_per_block(T kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  CHECK_CUDA(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
  return smem_size;
}

struct L1Carveout {
  int carveout_percent;
  uint32_t blocks_per_sm;
};

template <typename T>
inline void set_smem_carveout(T kernel, int percent) {
#ifdef USE_ROCM
  (void)kernel, (void)percent;
#else
  CHECK_CUDA(::cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, percent));
#endif
}

template <typename T>
inline auto set_prefer_l1_carveout_with_occupancy(
    const int lower_bound_percent,
    const int upper_bound_percent,
    const uint32_t min_occupancy,
    T kernel,
    const uint32_t block_threads,
    const std::size_t dyn_smem_bytes = 0) -> L1Carveout {
#ifdef USE_ROCM
  (void)lower_bound_percent, (void)upper_bound_percent, (void)min_occupancy;
  return {-1, get_blocks_per_sm(kernel, block_threads, dyn_smem_bytes)};
#else
  const auto kernel_ptr = reinterpret_cast<const void*>(kernel);
  // The attribute and the occupancy query both act on the current device.
  static_assert(cudaSharedmemCarveoutMaxL1 <= cudaSharedmemCarveoutMaxShared);
  CHECK_HOST(
      min_occupancy > 0 && lower_bound_percent <= upper_bound_percent  //
      && lower_bound_percent >= cudaSharedmemCarveoutMaxL1             //
      && upper_bound_percent <= cudaSharedmemCarveoutMaxShared);
  for (int p = lower_bound_percent; p <= upper_bound_percent; ++p) {
    set_smem_carveout(kernel_ptr, p);
    const auto occupancy = get_blocks_per_sm(kernel_ptr, block_threads, dyn_smem_bytes);
    if (occupancy >= min_occupancy) return {p, occupancy};
  }
  host::Error() << "no carveout in [" << lower_bound_percent << ", " << upper_bound_percent
                << "] can satisfy target occupancy " << min_occupancy << " for kernel " << kernel_ptr;
  __builtin_unreachable();
#endif
}

/// \brief Prefer the largest L1 carveout that keeps `kernel`'s default occupancy on the current
/// device (PDL secondaries inherit the primary's carveout). The attribute is per device and sticky
/// for the process; panics when no carveout restores occupancy. ROCm has no such attribute and
/// reports percent -1. Wrap it in `init_per_device(device, ...)` to run it once per device.
template <typename T>
inline auto set_prefer_l1_carveout(T kernel, uint32_t block_threads, std::size_t dyn_smem_bytes = 0) -> L1Carveout {
#ifdef USE_ROCM
  return {-1, get_blocks_per_sm(kernel, block_threads, dyn_smem_bytes)};
#else
  const auto kernel_ptr = reinterpret_cast<const void*>(kernel);
  set_smem_carveout(kernel_ptr, cudaSharedmemCarveoutDefault);
  const auto default_occupancy = get_blocks_per_sm(kernel_ptr, block_threads, dyn_smem_bytes);
  return set_prefer_l1_carveout_with_occupancy(
      cudaSharedmemCarveoutMaxL1,
      cudaSharedmemCarveoutMaxShared,
      default_occupancy,
      kernel_ptr,
      block_threads,
      dyn_smem_bytes);
#endif
}

/// \brief Opt `kernel` into `dyn_smem_bytes` of dynamic shared memory on the current device.
template <typename T>
inline void set_max_dynamic_smem(T kernel, std::size_t dyn_smem_bytes) {
  const auto kernel_ptr = reinterpret_cast<const void*>(kernel);
  const auto num_bytes = static_cast<int>(dyn_smem_bytes);
  CHECK_CUDA(::cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, num_bytes));
}

/// \brief Run `fn` on the first call per device, keyed off the uniqueness of (`Tag`, `Fn`).
/// \note Relaxed on purpose, to keep the hot path a single atomic load: concurrent first callers
/// may all run `fn`, so it must be idempotent (e.g. setting a function attribute). A throwing `fn`
/// leaves the device uninitialized, and devices past `kNumStaticMaxDevice` rerun `fn` every call.
template <typename Tag = void, std::invocable Fn>
inline void init_per_device(int device_id, bool use_cache, Fn&& fn) {
  static_assert(kNumStaticMaxDevice % 8 == 0);
  constexpr uint32_t kNumBytes = kNumStaticMaxDevice / 8;
  static std::atomic_uint8_t s_initialized[kNumBytes]{};
  const auto device = static_cast<uint32_t>(device_id);
  const auto can_use_cache = use_cache && device < kNumStaticMaxDevice;
  const auto mask = static_cast<uint8_t>(1u << (device % 8));
  if (can_use_cache && s_initialized[device / 8].load() & mask) return;
  std::forward<Fn>(fn)();
  if (can_use_cache) s_initialized[device / 8].fetch_or(mask);
}

// Some wrappers of the above functions

template <typename Tag = void, std::invocable Fn>
inline void init_per_device(details::MaybeDevice device, Fn&& fn) {
  return init_per_device<Tag>(device.device_id, true, std::forward<Fn>(fn));
}

template <auto kernel>
inline void set_max_dynamic_smem_per_device(details::MaybeDevice device, std::size_t dyn_smem_bytes) {
  using Tag = details::Tag<kernel>;
  return init_per_device<Tag>(device.device_id, true, [=] { set_max_dynamic_smem(kernel, dyn_smem_bytes); });
}

}  // namespace host::runtime

}  // namespace sglang
