/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#endif

#define CHECK_CUDA(call)                                                                            \
  do {                                                                                              \
    cudaError_t status_ = call;                                                                     \
    if (status_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)

inline int get_current_device() {
  int device;
  CHECK_CUDA(cudaGetDevice(&device));
  return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
  int capability_major, capability_minor;
  CHECK_CUDA(cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA(cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device));
  return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
  int multiprocessor_count;
  CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  return multiprocessor_count;
}
