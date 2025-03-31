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

#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__)

// Add backward support of bf16, bf162 PTX intrinsics
#if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 900  // bf16x2 requires compute ability >= sm80

__inline__ __device__ half __bfloat162half(const __nv_bfloat16 bf16_val) {
  return half(__bfloat162float(bf16_val));
}

__inline__ __device__ __nv_bfloat16 __half2bfloat16(const half hval) {
  return __nv_bfloat16(__half2float(hval));
}

__inline__ __device__ half2 __bfloat1622half2(const __nv_bfloat162 bf162_val) {
  half2 ret;
  ret.x = __bfloat162half(bf162_val.x);
  ret.y = __bfloat162half(bf162_val.y);
  return ret;
}

__inline__ __device__ __nv_bfloat162 __half22bfloat162(const half2 h2val) {
  __nv_bfloat162 ret;
  ret.x = __half2bfloat16(h2val.x);
  ret.y = __half2bfloat16(h2val.y);
  return ret;
}

#endif  //  __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900

namespace nvgpu {  // Add new semantics

// By default we use rounding to nearest even mode, but other modes will be supported soon; see
// https://docs.nvidia.com/cuda/floating-point/index.html
template <typename T>
__inline__ __device__ void __hsub2_inplace(uint* a, const uint b);

template <typename T>
__inline__ __device__ void __hmul2_inplace(uint* a, const uint b);

// ===== speicalization =====

// fp16, fp162 PTX intrinsics
template <>
__inline__ __device__ void __hsub2_inplace<half2>(uint* a, const uint b) {
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "+r"(*a) : "r"(*a), "r"(b));
}

template <>
__inline__ __device__ void __hmul2_inplace<half2>(uint* a, const uint b) {
  asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "+r"(*a) : "r"(*a), "r"(b));
}

#if __CUDA_ARCH__ >= 900

template <>
__inline__ __device__ void __hsub2_inplace<nv_bfloat162>(uint* a, const uint b) {
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "+r"(*a) : "r"(*a), "r"(b));
}

template <>
__inline__ __device__ void __hmul2_inplace<nv_bfloat162>(uint* a, const uint b) {
  asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "+r"(*a) : "r"(*a), "r"(b));
}

#elif __CUDA_ARCH__ >= 800

template <>
__inline__ __device__ void __hsub2_inplace<nv_bfloat162>(uint* a, const uint b) {
  asm volatile("sub.bf16x2 %0, %1, %2;\n" : "+r"(*a) : "r"(*a), "r"(b));
}

template <>
__inline__ __device__ void __hmul2_inplace<nv_bfloat162>(uint* a, const uint b) {
#warning "mul.rn.bf16x2 not supported __CUDA_ARCH__(800) <= 900)"

  nv_bfloat162* bf16x2_view_lhs = reinterpret_cast<nv_bfloat162*>(a);
  const nv_bfloat162* bf16x2_view_rhs = reinterpret_cast<const nv_bfloat162*>(&b);

  *bf16x2_view_lhs = __hmul2(*bf16x2_view_lhs, *bf16x2_view_rhs);
}

#else

#endif  // __CUDA_ARCH__ >= 900

}  // namespace nvgpu

#endif  // CUDA
