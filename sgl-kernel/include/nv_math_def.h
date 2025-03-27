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

#if defined(CUDA_VERSION) && CUDA_VERSION <= 12020

__inline__ __device__ half __bfloat162half(const __nv_bfloat16 bf16_val) {
  return half(__bfloat162float(bf16_val));
}

__inline__ __device__ half2 __bfloat1622half2(const __nv_bfloat162 bf162_val) {
  half2 ret;
  ret.x = __bfloat162half(bf162_val.x);
  ret.y = __bfloat162half(bf162_val.y);
  return ret;
}

__inline__ __device__ __nv_bfloat16 __half2bfloat16(const half hval) {
  return __nv_bfloat16(__half2float(hval));
}

__inline__ __device__ __nv_bfloat162 __half22bfloat162(const half2 h2val) {
  __nv_bfloat162 ret;
  ret.x = __half2bfloat16(h2val.x);
  ret.y = __half2bfloat16(h2val.y);
  return ret;
}

#endif  // CUDA_VERSION <= 12020
