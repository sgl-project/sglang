#pragma once

#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace device {

inline constexpr auto kWarpThreads = 32u;
inline constexpr auto kFullMask = 0xffffffffu;

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
#ifndef USE_ROCM
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                     : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
#else
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
#endif
}

__device__ __forceinline__ float warpReduceMax(float value) {
  value = fmaxf(value, __shfl_xor_sync(kFullMask, value, 16));
  value = fmaxf(value, __shfl_xor_sync(kFullMask, value, 8));
  value = fmaxf(value, __shfl_xor_sync(kFullMask, value, 4));
  value = fmaxf(value, __shfl_xor_sync(kFullMask, value, 2));
  value = fmaxf(value, __shfl_xor_sync(kFullMask, value, 1));
  return value;
}

__device__ __forceinline__ float blockReduceMax(float value) {
  static __shared__ float warpLevelMaxs[kWarpThreads];
  const int laneId = threadIdx.x % kWarpThreads;
  const int warpId = threadIdx.x / kWarpThreads;

  value = warpReduceMax(value);

  if (laneId == 0) warpLevelMaxs[warpId] = value;
  __syncthreads();

  value = (threadIdx.x < blockDim.x / kWarpThreads) ? warpLevelMaxs[laneId] : 0;
  if (warpId == 0) value = warpReduceMax(value);

  return value;
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T, std::integral... U>
__always_inline __device__ auto offset(T* ptr, U... offset) -> void* {
  static_assert(std::is_same_v<T, void>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char*>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
__always_inline __device__ auto offset(const T* ptr, U... offset) -> const void* {
  static_assert(std::is_same_v<T, void>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char*>(ptr) + (... + offset);
}

}  // namespace pointer

template <typename T, std::size_t N>
struct device_vec {
  T data[N];
};

template <bool kUsePDL>
__forceinline__ __device__ void PDLWaitPrimary() {
#ifndef USE_ROCM
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;");
  }
#endif
}

template <bool kUsePDL>
__forceinline__ __device__ void PDLTriggerSecondary() {
#ifndef USE_ROCM
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
#endif
}

}  // namespace device

namespace host {

inline void RuntimeDeviceCheck(::cudaError_t error, DebugInfo location = {}) {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

inline void RuntimeDeviceCheck(DebugInfo location = {}) {
  return RuntimeDeviceCheck(::cudaGetLastError(), location);
}

struct LaunchKernel {
 public:
  explicit LaunchKernel(
      dim3 grid_dim,
      dim3 block_dim,
      DLDevice device,
      std::size_t dynamic_shared_mem_bytes = 0,
      DebugInfo location = {}) noexcept
      : m_config(s_make_config(grid_dim, block_dim, resolve_device(device), dynamic_shared_mem_bytes)),
        m_location(location) {}

  explicit LaunchKernel(
      dim3 grid_dim,
      dim3 block_dim,
      cudaStream_t stream,
      std::size_t dynamic_shared_mem_bytes = 0,
      DebugInfo location = {}) noexcept
      : m_config(s_make_config(grid_dim, block_dim, stream, dynamic_shared_mem_bytes)), m_location(location) {}

  LaunchKernel(const LaunchKernel&) = delete;
  LaunchKernel& operator=(const LaunchKernel&) = delete;

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  auto enable_pdl(bool enabled = true) -> LaunchKernel& {
    if (enabled) {
      m_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attrs[0].val.programmaticStreamSerializationAllowed = true;
      m_config.numAttrs = 1;
      m_config.attrs = m_attrs;
    } else {
      m_config.numAttrs = 0;
    }
    return *this;
  }

  template <typename T, typename... Args>
  auto operator()(T&& kernel, Args&&... args) const -> void {
    RuntimeDeviceCheck(::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...), m_location);
  }

 private:
  static auto s_make_config(  // Make a config for kernel launch
      dim3 grid_dim,
      dim3 block_dim,
      cudaStream_t stream,
      std::size_t smem) -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }

  cudaLaunchConfig_t m_config;
  const DebugInfo m_location;
  cudaLaunchAttribute m_attrs[1];
};

}  // namespace host
