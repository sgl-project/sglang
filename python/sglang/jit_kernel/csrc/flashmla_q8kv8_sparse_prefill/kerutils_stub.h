// kerutils_stub.h -- Minimal stub replacing <kerutils/kerutils.cuh> for JIT compilation.
// Provides: KU_ASSERT, KU_CUDA_CHECK, KU_CHECK_KERNEL_LAUNCH, ku::ceil_div
#pragma once

#include <cstdio>
#include <cstdlib>

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
