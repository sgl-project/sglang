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

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include "defines.h"
#include "params.h"
#include <cstdio>
#include <cstdlib>
#include <math_constants.h>

#define KU_ASSERT(cond)                                                             \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      fprintf(stderr, "KU_ASSERT failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

#define KU_CUDA_CHECK(call)                                                                     \
  do {                                                                                          \
    cudaError_t err = (call);                                                                   \
    if (err != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

namespace ku {

template <typename T>
__host__ __device__ __forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace ku
