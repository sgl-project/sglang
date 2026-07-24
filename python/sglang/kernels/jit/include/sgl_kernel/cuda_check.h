/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ================= common/cuda_check.h =================

#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   \
                     cudaGetErrorName(_e), __FILE__, __LINE__,                 \
                     cudaGetErrorString(_e));                                  \
        std::abort();                                                          \
    }                                                                          \
} while (0)

#define CU_CHECK(expr) do {                                                    \
    CUresult _e = (expr);                                                      \
    if (_e != CUDA_SUCCESS) {                                                  \
        const char* _s = nullptr;                                              \
        cuGetErrorString(_e, &_s);                                             \
        std::fprintf(stderr, "CU error at %s:%d: %s\n",                        \
                     __FILE__, __LINE__, _s ? _s : "?");                       \
        std::abort();                                                          \
    }                                                                          \
} while (0)
