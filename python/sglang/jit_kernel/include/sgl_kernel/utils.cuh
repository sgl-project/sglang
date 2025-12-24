#pragma once

#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace device {

inline constexpr auto kWarpThreads = 32u;

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
  /// TODO: We can add a queue to store the attributes (e.g. for PDL) if needed in the future.
};

}  // namespace host
