#pragma once

#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>
#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#ifndef __grid_constant__
#define __grid_constant__
#endif
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaLaunchConfig_t = hipLaunchConfig_t;
using cudaLaunchAttribute = hipLaunchAttribute;
inline constexpr auto cudaSuccess = hipSuccess;
#define cudaStreamPerThread hipStreamPerThread
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaLaunchKernel hipLaunchKernel
#endif

#ifndef USE_ROCM
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;

using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
using fp8x2_e5m2_t = __nv_fp8x2_e5m2;

using fp32x4_t = float4;
#else
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __hip_bfloat16;
using fp8_e4m3_t = uint8_t;
using fp8_e5m2_t = uint8_t;
using fp32x2_t = float2;
using fp16x2_t = half2;
using bf16x2_t = __hip_bfloat162;
using fp8x2_e4m3_t = uint16_t;
using fp8x2_e5m2_t = uint16_t;
using fp32x4_t = float4;
#endif

/*
 * LDG Support
 */
#ifndef USE_ROCM
#define SGLANG_LDG(arg) __ldg(arg)
#else
#define SGLANG_LDG(arg) *(arg)
#endif

namespace device {

#define SGL_DEVICE __forceinline__ __device__

inline constexpr auto kWarpThreads = 32u;
inline constexpr auto kFullMask = 0xffffffffu;

template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}

/**
 * \brief Load data with the specified type and offset from a void pointer.
 * \tparam T The type to load.
 * \param ptr The base pointer.
 * \param offset The offset in number of elements of type T.
 */
template <typename T>
SGL_DEVICE T load_as(const void* ptr, int64_t offset = 0) {
  return static_cast<const T*>(ptr)[offset];
}

/**
 * \brief Store data with the specified type and offset to a void pointer.
 * \tparam T The type to store.
 * \param ptr The base pointer.
 * \param val The value to store.
 * \param offset The offset in number of elements of type T.
 * \note we use type_identity_t to force the caller to explicitly specify
 * the template parameter `T`, which can avoid accidentally using the wrong type.
 */
template <typename T>
SGL_DEVICE void store_as(void* ptr, std::type_identity_t<T> val, int64_t offset = 0) {
  static_cast<T*>(ptr)[offset] = val;
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T = char, std::integral... U>
SGL_DEVICE auto offset(void* ptr, U... offset) -> void* {
  return static_cast<T*>(ptr) + (... + offset);
}

template <typename T = char, std::integral... U>
SGL_DEVICE auto offset(const void* ptr, U... offset) -> const void* {
  return static_cast<const T*>(ptr) + (... + offset);
}

}  // namespace pointer

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
#ifdef USE_ROCM
    (void)enabled;
    m_config.numAttrs = 0;
#else
    if (enabled) {
      m_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attrs[0].val.programmaticStreamSerializationAllowed = true;
      m_config.numAttrs = 1;
      m_config.attrs = m_attrs;
    } else {
      m_config.numAttrs = 0;
    }
#endif
    return *this;
  }

  template <typename T, typename... Args>
  auto operator()(T&& kernel, Args&&... args) const -> void {
#ifdef USE_ROCM
    hipLaunchKernelGGL(
        std::forward<T>(kernel),
        m_config.gridDim,
        m_config.blockDim,
        m_config.dynamicSmemBytes,
        m_config.stream,
        std::forward<Args>(args)...);
    RuntimeDeviceCheck(m_location);
#else
    RuntimeDeviceCheck(::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...), m_location);
#endif
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
