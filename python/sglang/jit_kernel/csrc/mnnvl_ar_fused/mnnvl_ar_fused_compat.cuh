// Compatibility shim for the verbatim mnnvl fused-allreduce kernel port.
// Replaces the three flashinfer-internal includes (../exception.h,
// ../logging.h, ../utils.cuh) with behavior-equivalent definitions so the
// kernel body compiles unchanged outside the flashinfer tree. Definitions
// mirror flashinfer 0.6.12 (see baseline/upstream_ref and docs/results.md):
//   - ceil_div / round_up / round_down: same constexpr templates
//   - GetCudaComputeCapability / GetCudaMultiProcessorCount: same queries,
//     same cached-atomic behavior
//   - FLASHINFER_CHECK / FLASHINFER_ERROR: ostringstream message + throw
//   - FLASHINFER_CUDA_CALL: print to stderr and return the cudaError_t
//   - FLASHINFER_LOG_DEBUG: no-op (upstream routes to spdlog::debug, which is
//     compiled out at default log level; no kernel-side behavior depends on it)
#pragma once

#include <array>
#include <atomic>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#define MNNVL_AR_FUSED_STR_HELPER(x) #x
#define STR(x) MNNVL_AR_FUSED_STR_HELPER(x)

namespace flashinfer {

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 round_up(const T1 x, const T2 y) noexcept {
  return ceil_div(x, y) * y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 round_down(const T1 x, const T2 y) noexcept {
  return (x / y) * y;
}

inline std::pair<int, int> GetCudaComputeCapability() {
  int device_id = 0;
  cudaGetDevice(&device_id);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  return std::make_pair(major, minor);
}

// Cached like upstream; per-process single-GPU assumption noted upstream too.
inline int GetCudaMultiProcessorCount() {
  static std::atomic<int> sm_count{0};
  int cached = sm_count.load(std::memory_order_relaxed);
  if (cached == 0) {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    cached = device_prop.multiProcessorCount;
    sm_count.store(cached, std::memory_order_relaxed);
  }
  return cached;
}

inline void mnnvl_ar_fused_write_to_stream(std::ostringstream&) {}

template <typename T, typename... Args>
void mnnvl_ar_fused_write_to_stream(std::ostringstream& oss, T&& val, Args&&... args) {
  oss << std::forward<T>(val);
  mnnvl_ar_fused_write_to_stream(oss, std::forward<Args>(args)...);
}

}  // namespace flashinfer

#define FLASHINFER_ERROR(message) throw std::runtime_error(std::string(message))

#define FLASHINFER_CHECK(condition, ...)                              \
  do {                                                                \
    if (!(condition)) {                                               \
      std::ostringstream oss;                                         \
      flashinfer::mnnvl_ar_fused_write_to_stream(oss, ##__VA_ARGS__); \
      std::string msg = oss.str();                                    \
      if (msg.empty()) {                                              \
        msg = "Check failed: " #condition;                            \
      }                                                               \
      FLASHINFER_ERROR(msg);                                          \
    }                                                                 \
  } while (0)

#define FLASHINFER_LOG_DEBUG(...) \
  do {                            \
  } while (0)

#define FLASHINFER_CUDA_CALL(func, ...)                                                                              \
  {                                                                                                                  \
    cudaError_t e = (func);                                                                                          \
    if (e != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ << ": line " << __LINE__ \
                << " at function " << STR(func) << std::endl;                                                        \
      return e;                                                                                                      \
    }                                                                                                                \
  }
