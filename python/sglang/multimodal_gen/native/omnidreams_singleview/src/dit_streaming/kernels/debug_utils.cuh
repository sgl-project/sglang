// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Debug utilities for INT8 per-block quantization validation.
 *
 * These utilities help validate numerical correctness during development:
 * - Tensor validation (NaN/Inf detection)
 * - Tensor statistics (min, max, mean, std)
 * - Tensor comparison (MSE, max diff)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

namespace omnidreams_singleview {
namespace debug {

// Enable verbose debug output (0 = off, 1 = basic, 2 = detailed)
#ifndef DEBUG_BLOCK_QUANT
#define DEBUG_BLOCK_QUANT 0
#endif

// =============================================================================
// Tensor Validation Kernels
// =============================================================================

/**
 * Check if any element in tensor is NaN or Inf.
 * Sets error_flag to 1 if invalid value found.
 */
template<typename T>
__global__ void validate_tensor_kernel(
    const T* __restrict__ data,
    int64_t n,
    int* __restrict__ error_flag,
    int* __restrict__ nan_count,
    int* __restrict__ inf_count
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = static_cast<float>(data[idx]);
    if (isnan(v)) {
        atomicAdd(nan_count, 1);
        atomicExch(error_flag, 1);
    }
    if (isinf(v)) {
        atomicAdd(inf_count, 1);
        atomicExch(error_flag, 1);
    }
}

// Specialization for half
template<>
__global__ void validate_tensor_kernel<half>(
    const half* __restrict__ data,
    int64_t n,
    int* __restrict__ error_flag,
    int* __restrict__ nan_count,
    int* __restrict__ inf_count
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = __half2float(data[idx]);
    if (isnan(v)) {
        atomicAdd(nan_count, 1);
        atomicExch(error_flag, 1);
    }
    if (isinf(v)) {
        atomicAdd(inf_count, 1);
        atomicExch(error_flag, 1);
    }
}

// =============================================================================
// Tensor Statistics Kernels
// =============================================================================

/**
 * Compute min, max, sum, sum_sq for tensor statistics.
 * Uses atomic operations for simplicity (not optimal for performance).
 */
template<typename T>
__global__ void tensor_stats_kernel(
    const T* __restrict__ data,
    int64_t n,
    float* __restrict__ min_val,
    float* __restrict__ max_val,
    double* __restrict__ sum,
    double* __restrict__ sum_sq
) {
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ double s_sum;
    __shared__ double s_sum_sq;

    if (threadIdx.x == 0) {
        s_min = 1e30f;
        s_max = -1e30f;
        s_sum = 0.0;
        s_sum_sq = 0.0;
    }
    __syncthreads();

    float local_min = 1e30f;
    float local_max = -1e30f;
    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float v = static_cast<float>(data[i]);
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
        local_sum += v;
        local_sum_sq += v * v;
    }

    // Block reduction
    atomicMin(reinterpret_cast<int*>(&s_min), __float_as_int(local_min));
    atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(local_max));
    __syncthreads();

    // Global reduction (one thread per block)
    if (threadIdx.x == 0) {
        atomicMin(reinterpret_cast<int*>(min_val), __float_as_int(s_min));
        atomicMax(reinterpret_cast<int*>(max_val), __float_as_int(s_max));
        // Note: atomic double add not available on all architectures
        // Using atomicAdd for double requires sm_60+
        atomicAdd(sum, local_sum);
        atomicAdd(sum_sq, local_sum_sq);
    }
}

// Specialization for half
template<>
__global__ void tensor_stats_kernel<half>(
    const half* __restrict__ data,
    int64_t n,
    float* __restrict__ min_val,
    float* __restrict__ max_val,
    double* __restrict__ sum,
    double* __restrict__ sum_sq
) {
    float local_min = 1e30f;
    float local_max = -1e30f;
    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float v = __half2float(data[i]);
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
        local_sum += v;
        local_sum_sq += v * v;
    }

    // Global atomic reduction
    atomicMin(reinterpret_cast<int*>(min_val), __float_as_int(local_min));
    atomicMax(reinterpret_cast<int*>(max_val), __float_as_int(local_max));
    atomicAdd(sum, local_sum);
    atomicAdd(sum_sq, local_sum_sq);
}

// =============================================================================
// Tensor Comparison Kernels
// =============================================================================

/**
 * Compute MSE and max absolute difference between two tensors.
 */
template<typename T>
__global__ void compare_tensors_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int64_t n,
    double* __restrict__ sum_sq_diff,
    float* __restrict__ max_diff
) {
    double local_sum_sq_diff = 0.0;
    float local_max_diff = 0.0f;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float va = static_cast<float>(a[i]);
        float vb = static_cast<float>(b[i]);
        float diff = va - vb;
        local_sum_sq_diff += diff * diff;
        local_max_diff = fmaxf(local_max_diff, fabsf(diff));
    }

    atomicAdd(sum_sq_diff, local_sum_sq_diff);
    atomicMax(reinterpret_cast<int*>(max_diff), __float_as_int(local_max_diff));
}

// Specialization for half
template<>
__global__ void compare_tensors_kernel<half>(
    const half* __restrict__ a,
    const half* __restrict__ b,
    int64_t n,
    double* __restrict__ sum_sq_diff,
    float* __restrict__ max_diff
) {
    double local_sum_sq_diff = 0.0;
    float local_max_diff = 0.0f;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        float diff = va - vb;
        local_sum_sq_diff += diff * diff;
        local_max_diff = fmaxf(local_max_diff, fabsf(diff));
    }

    atomicAdd(sum_sq_diff, local_sum_sq_diff);
    atomicMax(reinterpret_cast<int*>(max_diff), __float_as_int(local_max_diff));
}

// =============================================================================
// Host-callable Debug Functions
// =============================================================================

/**
 * Validate tensor for NaN/Inf values.
 *
 * @param data Device pointer to tensor data
 * @param n Number of elements
 * @param name Tensor name for debug output
 * @param stream CUDA stream
 * @return true if tensor is valid (no NaN/Inf), false otherwise
 */
template<typename T>
bool validate_tensor(const T* data, int64_t n, const char* name, cudaStream_t stream) {
    int* d_error_flag;
    int* d_nan_count;
    int* d_inf_count;
    cudaMalloc(&d_error_flag, sizeof(int));
    cudaMalloc(&d_nan_count, sizeof(int));
    cudaMalloc(&d_inf_count, sizeof(int));
    cudaMemset(d_error_flag, 0, sizeof(int));
    cudaMemset(d_nan_count, 0, sizeof(int));
    cudaMemset(d_inf_count, 0, sizeof(int));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    validate_tensor_kernel<<<blocks, threads, 0, stream>>>(
        data, n, d_error_flag, d_nan_count, d_inf_count);

    int h_error_flag, h_nan_count, h_inf_count;
    cudaMemcpyAsync(&h_error_flag, d_error_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_nan_count, d_nan_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_inf_count, d_inf_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_error_flag);
    cudaFree(d_nan_count);
    cudaFree(d_inf_count);

    if (h_error_flag) {
        printf("[DEBUG] %s: INVALID - %d NaN, %d Inf (n=%lld)\n",
               name, h_nan_count, h_inf_count, (long long)n);
        return false;
    }

#if DEBUG_BLOCK_QUANT >= 2
    printf("[DEBUG] %s: VALID (n=%lld)\n", name, (long long)n);
#endif
    return true;
}

/**
 * Print tensor statistics (min, max, mean, std).
 *
 * @param data Device pointer to tensor data
 * @param n Number of elements
 * @param name Tensor name for output
 * @param stream CUDA stream
 */
template<typename T>
void print_tensor_stats(const T* data, int64_t n, const char* name, cudaStream_t stream) {
    float* d_min;
    float* d_max;
    double* d_sum;
    double* d_sum_sq;

    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(double));
    cudaMalloc(&d_sum_sq, sizeof(double));

    float init_min = 1e30f;
    float init_max = -1e30f;
    double init_zero = 0.0;
    cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &init_zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum_sq, &init_zero, sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 1024);

    tensor_stats_kernel<<<blocks, threads, 0, stream>>>(
        data, n, d_min, d_max, d_sum, d_sum_sq);

    float h_min, h_max;
    double h_sum, h_sum_sq;
    cudaMemcpyAsync(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sum_sq, d_sum_sq, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_sum_sq);

    double mean = h_sum / n;
    double var = h_sum_sq / n - mean * mean;
    double std = sqrt(max(var, 0.0));

    printf("[STATS] %s: n=%lld, min=%.6f, max=%.6f, mean=%.6f, std=%.6f\n",
           name, (long long)n, h_min, h_max, mean, std);
}

/**
 * Compare two tensors and report MSE and max difference.
 *
 * @param a First tensor
 * @param b Second tensor
 * @param n Number of elements
 * @param stream CUDA stream
 * @return MSE between tensors
 */
template<typename T>
float compare_tensors_mse(const T* a, const T* b, int64_t n, cudaStream_t stream) {
    double* d_sum_sq_diff;
    float* d_max_diff;

    cudaMalloc(&d_sum_sq_diff, sizeof(double));
    cudaMalloc(&d_max_diff, sizeof(float));

    double init_zero = 0.0;
    float init_float_zero = 0.0f;
    cudaMemcpy(d_sum_sq_diff, &init_zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_diff, &init_float_zero, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 1024);

    compare_tensors_kernel<<<blocks, threads, 0, stream>>>(
        a, b, n, d_sum_sq_diff, d_max_diff);

    double h_sum_sq_diff;
    float h_max_diff;
    cudaMemcpyAsync(&h_sum_sq_diff, d_sum_sq_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_sum_sq_diff);
    cudaFree(d_max_diff);

    float mse = static_cast<float>(h_sum_sq_diff / n);

#if DEBUG_BLOCK_QUANT >= 1
    printf("[COMPARE] n=%lld, MSE=%.6f, MaxDiff=%.6f\n",
           (long long)n, mse, h_max_diff);
#endif

    return mse;
}

// =============================================================================
// Debug Macros
// =============================================================================

#if DEBUG_BLOCK_QUANT >= 1
#define DEBUG_VALIDATE(ptr, n, name, stream) \
    omnidreams_singleview::debug::validate_tensor(ptr, n, name, stream)
#define DEBUG_PRINT_STATS(ptr, n, name, stream) \
    omnidreams_singleview::debug::print_tensor_stats(ptr, n, name, stream)
#define DEBUG_COMPARE(a, b, n, stream) \
    omnidreams_singleview::debug::compare_tensors_mse(a, b, n, stream)
#else
#define DEBUG_VALIDATE(ptr, n, name, stream) (true)
#define DEBUG_PRINT_STATS(ptr, n, name, stream) ((void)0)
#define DEBUG_COMPARE(a, b, n, stream) (0.0f)
#endif

} // namespace debug
} // namespace omnidreams_singleview
