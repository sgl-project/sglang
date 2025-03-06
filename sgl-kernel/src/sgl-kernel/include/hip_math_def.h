#pragma once

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_bf16.h>
#include <hip/hip_common.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

// Adapted from flashinfer-rocm [PR#491](https://github.com/flashinfer-ai/flashinfer/pull/491)

namespace amdgpu {

template <typename T>
__forceinline__ __device__ T shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);

template <typename srcDtype, typename destDtype>
__forceinline__ __device__ destDtype cast(srcDtype val);

// specialization
template <>
__forceinline__ __device__ float shfl_xor_sync(unsigned mask, float var, int laneMask, int width) {
  return __shfl_xor(var, laneMask, width);
}

template <>
__forceinline__ __device__ int shfl_xor_sync(unsigned mask, int var, int laneMask, int width) {
  return __shfl_xor(var, laneMask, width);
}

template <>
__forceinline__ __device__ float cast(float val) {
  return val;
}

template <>
__forceinline__ __device__ float cast(__half val) {
  return __half2float(val);
}

template <>
__forceinline__ __device__ float cast(__hip_bfloat16 val) {
  return __bfloat162float(val);
}

template<>
__forceinline__ __device__ uint8_t cast(float fval) {
#ifdef HIP_FP8_TYPE_FNUZ
  return __hip_cvt_float_to_fp8(fval, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
#else
# error "__hip_cvt_float_to_fp8 is not supported in this processor (arch < gfx942)."
#endif
}

}  // namespace amdgpu

template <typename T>
__forceinline__ __device__ T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize) {
  return amdgpu::shfl_xor_sync(mask, var, laneMask, width);
}

template <typename srcDtype>
__device__ __forceinline__ float castToFloat(srcDtype val) {
  return amdgpu::cast<srcDtype, float>(val);
}

template <typename srcDtype>
__device__ __forceinline__ uint8_t castToFP8Storage(srcDtype val) {
  return amdgpu::cast<srcDtype, uint8_t>(val);
}

#endif
