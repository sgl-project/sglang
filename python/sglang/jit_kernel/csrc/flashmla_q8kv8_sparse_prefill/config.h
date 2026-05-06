#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include "defines.h"
#include "params.h"
#include <cooperative_groups.h>
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

namespace sm90::fwd {

using namespace cute;

};  // namespace sm90::fwd
