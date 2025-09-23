/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#ifdef USE_ROCM

#include <hip/hip_bf16.h>
#include <hip/hip_common.h>
#include <hip/hip_fp16.h>

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

template <>
__forceinline__ __device__ __half cast(float fval) {
  return __float2half(fval);
}

template <>
__forceinline__ __device__ __hip_bfloat16 cast(float fval) {
  return __float2bfloat16(fval);
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

template <typename dstDtype>
__device__ __forceinline__ dstDtype castFromFloat(float val) {
  return amdgpu::cast<float, dstDtype>(val);
}

// operator overload to support flashinfer
__host__ __device__ __forceinline__ __half operator*(const __half& x, const __half& y) {
  __half h_x = x;
  __half h_y = y;
  return __hmul(h_x, h_y);
}

#endif
