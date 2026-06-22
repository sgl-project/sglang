// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Unified INT8 GEMM API for SM80+ Tensor Cores.
 *
 * Features:
 * - 64x64 output tiles, K=128 (matches quantization blocks)
 * - XOR swizzle for bank-conflict-free shared memory
 * - cp.async for asynchronous transfers
 * - Double-buffered for latency hiding
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "cute_int8_gemm.cuh"

namespace omnidreams_singleview {

/**
 * INT8 GEMM with pre-quantized activations.
 *
 * @param A           INT8 activations [M, K] row-major
 * @param A_scales    Per-block scales [M/128, K/128]
 * @param B_swizzled  INT8 weights pre-swizzled
 * @param B_scales    Per-block scales [K/128, N/128]
 * @param C           FP16 output [M, N]
 * @param bias        Optional FP16 bias [N]
 * @param M, N, K     Problem dimensions
 * @param stream      CUDA stream
 */
inline cudaError_t int8_gemm_unified(
    const int8_t* A,
    const float* A_scales,
    const int8_t* B_swizzled,
    const float* B_scales,
    half* C,
    const half* bias,
    int M, int N, int K,
    cudaStream_t stream
) {
    return int8_gemm(A, A_scales, B_swizzled, B_scales, C, bias, M, N, K, stream);
}

} // namespace omnidreams_singleview
