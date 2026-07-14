/// \file utils.cuh
/// \brief Core CUDA/device utilities: type aliases, PDL helpers,
///        typed pointer access, kernel launch wrapper, and error checking.
///
/// This header is included (directly or transitively) by nearly every
/// JIT kernel. It provides:
/// - Scalar/packed type aliases (`fp16_t`, `bf16_t`, `fp8_e4m3_t`, ...).
/// - `SGL_DEVICE` macro (forced-inline device function qualifier).
/// - `kWarpThreads` constant (32).
/// - PDL (Programmatic Dependent Launch) helpers for Hopper (sm_90+).
/// - Typed `load_as` / `store_as` for void-pointer access.
/// - `pointer::offset` for safe void-pointer arithmetic.
/// - `host::LaunchKernel` - kernel launcher with optional PDL.
/// - `host::RuntimeDeviceCheck` - CUDA error checking.

#pragma once

#include <sgl_kernel/ffi.h>
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
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
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

// DLPack device type for the current platform
#ifndef USE_ROCM
inline constexpr auto kDLGPU = kDLCUDA;
inline constexpr auto kDLGPUHost = kDLCUDAHost;
#else
inline constexpr auto kDLGPU = kDLROCM;
inline constexpr auto kDLGPUHost = kDLROCMHost;
#endif

namespace device {

/// \brief Macro: forced-inline device function qualifier.
#define SGL_DEVICE __forceinline__ __device__

// Architecture detection: SGL_CUDA_ARCH is injected by load_jit() and is
// available in both host and device compilation passes, whereas __CUDA_ARCH__
// is only defined by nvcc during the device pass.
#if !defined(USE_ROCM)
#if !defined(SGL_CUDA_ARCH)
#error "SGL_CUDA_ARCH is not defined. JIT compilation must inject -DSGL_CUDA_ARCH via load_jit()."
#endif
#if defined(__CUDA_ARCH__)
static_assert(
    __CUDA_ARCH__ == SGL_CUDA_ARCH, "SGL_CUDA_ARCH mismatch: injected arch flag does not match device target");
#endif
#define SGL_ARCH_HOPPER_OR_GREATER (SGL_CUDA_ARCH >= 900)
#define SGL_ARCH_BLACKWELL_OR_GREATER ((SGL_CUDA_ARCH >= 1000) && (CUDA_VERSION >= 12090))
#else  // USE_ROCM
#define SGL_ARCH_HOPPER_OR_GREATER 0
#define SGL_ARCH_BLACKWELL_OR_GREATER 0
#endif

// Maximum vector size in bytes supported by current architecture.
// Pre-Blackwell / AMD: 128-bit (16 bytes)
// Blackwell or greater: 256-bit (32 bytes)
inline constexpr std::size_t kMaxVecBytes = SGL_ARCH_BLACKWELL_OR_GREATER ? 32 : 16;

/// \brief Number of threads per warp (always 32 on NVIDIA/AMD GPUs).
inline constexpr auto kWarpThreads = 32u;
/// \brief Full warp active mask (all 32 lanes).
#ifndef USE_ROCM
inline constexpr auto kFullMask = 0xffffffffu;
#else
inline constexpr auto kFullMask = 0xffffffffffffffffULL;
#endif

/**
 * \brief PDL (Programmatic Dependent Launch): wait for the primary kernel.
 *
 * On Hopper (sm_90+), inserts a `griddepcontrol.wait` instruction to
 * synchronize with a preceding kernel in the same stream. On older
 * architectures or ROCm this is a no-op.
 */
template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
#if SGL_ARCH_HOPPER_OR_GREATER
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

/**
 * \brief PDL: trigger dependent (secondary) kernel launch.
 *
 * On Hopper (sm_90+), inserts a `griddepcontrol.launch_dependents`
 * instruction. On older architectures or ROCm this is a no-op.
 */
template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
#if SGL_ARCH_HOPPER_OR_GREATER
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}

template <std::integral T, std::integral U>
SGL_DEVICE constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
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

/// \brief Safe void-pointer arithmetic (byte-level by default).
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

/// PTX pragma that lets the compiler spill registers into otherwise-unused
/// shared memory instead of local memory. The radix kernels run at occupancy 2
/// (32 regs/thread) and rely on this to avoid local-memory traffic.
SGL_DEVICE void enable_smem_spilling() {
#if defined(__CUDA_ARCH__) && CUDART_VERSION >= 13000
  asm(".pragma \"enable_smem_spilling\";");
#endif
}

}  // namespace device

namespace host {

/**
 * \brief Check the CUDA error code and panic with location info on failure.
 */
inline void RuntimeDeviceCheck(::cudaError_t error, DebugInfo location = {}) {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

/// \brief Check the last CUDA error (calls `cudaGetLastError`).
inline void RuntimeDeviceCheck(DebugInfo location = {}) {
  return RuntimeDeviceCheck(::cudaGetLastError(), location);
}

inline int getSMVersion(int device_id) {
  int sm_major = 0;
  int sm_minor = 0;
#ifndef USE_ROCM
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device_id));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device_id));
#else
  // SM (compute-capability) version is CUDA-only; the cudaDevAttr* enums are not
  // declared under HIP, so this must be compiled out for hipcc (DeepSeek-V4 JIT
  // kernels include this header). ROCm dispatches on gfx arch, not SM version.
  (void)device_id;
#endif
  return sm_major * 10 + sm_minor;
}

inline auto alloc_workspace_tensor(size_t required_bytes, DLDevice device) -> tvm::ffi::Tensor {
  if (required_bytes == 0) return {};
  DLDataType u8 = {kDLUInt, 8, 1};
  int64_t shape[] = {static_cast<int64_t>(required_bytes)};
  return ffi::empty(tvm::ffi::ShapeView(shape, 1), u8, device);
}

/**
 * \brief Kernel launcher with automatic stream resolution and PDL support.
 *
 * Usage:
 * \code
 *   host::LaunchKernel(grid, block, device)
 *       .enable_pdl(true)(my_kernel, arg0, arg1);
 *   host::LaunchKernel(grid, block, stream)
 *       .config({.use_pdl = true, .cluster_dim = cluster_dim})(my_kernel, arg0);
 * \endcode
 *
 * The constructor resolves the CUDA stream from a `DLDevice` (via `TVMFFIEnvGetStream`)
 * or accepts a raw `cudaStream_t`. The call operator launches the kernel and checks for errors.
 */
struct LaunchKernel {
 private:
  struct KernelConfig {
    bool use_pdl = false;
    std::optional<dim3> cluster_dim = std::nullopt;
  };

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
      auto& attr = m_attrs[m_config.numAttrs++];
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = true;
      m_config.attrs = m_attrs;
    }
#endif
    return *this;
  }

  auto enable_cluster(dim3 cluster_dim) -> LaunchKernel& {
#ifdef USE_ROCM
    (void)cluster_dim;
#else
    auto& attr = m_attrs[m_config.numAttrs++];
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {cluster_dim.x, cluster_dim.y, cluster_dim.z};
    m_config.attrs = m_attrs;
#endif
    return *this;
  }

  /**
   * \brief Configure the kernel launch with the given options.
   * \param config The kernel configuration options.
   * \return A reference to this `LaunchKernel` for chaining.
   * \note This is a convenience method that applies multiple configurations at once.
   * We are in favor of this instead of `enable_pdl` and `enable_cluster`.
   * We enforce use of designated initializers for better readability.
   */
  auto config(const KernelConfig& config) -> LaunchKernel& {
    if (config.use_pdl) this->enable_pdl(true);
    if (config.cluster_dim) this->enable_cluster(*config.cluster_dim);
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

  template <typename T, typename... Args>
  auto launch(T&& kernel, Args&&... args) const -> void {
    return (*this)(std::forward<T>(kernel), std::forward<Args>(args)...);
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
  cudaLaunchAttribute m_attrs[2];
};

}  // namespace host
