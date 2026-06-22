// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Per-block INT8 quantization kernels.
 *
 * This implements 2D per-block (128x128) INT8 quantization for activations
 * and weights. Per-block quantization adapts to local value distributions,
 * preserving precision across heterogeneous tensors.
 *
 * Quantization scheme:
 *   - Block size: 128x128
 *   - Activations [M, K]: scales [M/128, K/128]
 *   - Weights [K, N]: scales [K/128, N/128]
 *
 * For each block:
 *   amax = max(|block|)
 *   scale = amax / 127.0
 *   quantized = round(block / scale).clamp(-127, 127)
 *
 * Swizzled weight format (for tensor core optimization):
 *   - Input: [K, N] row-major (standard PyTorch layout)
 *   - Output: [N_padded, K_padded] with swizzle pattern applied
 *   - Stored as column-major for MMA B operand
 *   - Swizzle pattern: Swizzle<2, 4, 3> for bank-conflict-free ldmatrix
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <cstdint>

namespace omnidreams_singleview {

// Block size for per-block quantization
constexpr int QUANT_BLOCK_SIZE = 128;

// =============================================================================
// Kernel Declarations (implementations in wan_block_quant.cu)
// =============================================================================

/**
 * Per-block INT8 quantization kernel.
 *
 * Each CTA processes one 128x128 block:
 * 1. Load 128x128 tile to shared memory
 * 2. Compute block amax via parallel reduction
 * 3. Scale and convert to int8
 * 4. Store scale to scale tensor
 *
 * Grid: (num_k_blocks, num_m_blocks)
 * Block: (128) - one thread per column, processes 128 rows
 */
__global__ void quantize_per_block_128_kernel(
    const half* __restrict__ src,       // [M, K] input (row-major)
    int8_t* __restrict__ dst,            // [M, K] output (row-major)
    float* __restrict__ block_scales,    // [num_m_blocks, num_k_blocks] scales
    int M, int K,
    int num_m_blocks, int num_k_blocks);

/**
 * Optimized per-block quantization using shared memory tiling.
 *
 * This version loads the entire 128x128 block to shared memory for better
 * memory access patterns and reduced global memory traffic.
 *
 * Grid: (num_k_blocks, num_m_blocks)
 * Block: (32, 4) = 128 threads, each processes 32 elements
 */
__global__ void quantize_per_block_128_tiled_kernel(
    const half* __restrict__ src,       // [M, K] input (row-major)
    int8_t* __restrict__ dst,            // [M, K] output (row-major)
    float* __restrict__ block_scales,    // [num_m_blocks, num_k_blocks] scales
    int M, int K,
    int num_m_blocks, int num_k_blocks);

/**
 * Per-block dequantization kernel.
 *
 * Reconstructs fp16 tensor from int8 + per-block scales.
 *
 * Grid: (num_k_blocks, num_m_blocks)
 * Block: (128)
 */
__global__ void dequantize_per_block_128_kernel(
    const int8_t* __restrict__ src,      // [M, K] input (row-major)
    half* __restrict__ dst,               // [M, K] output (row-major)
    const float* __restrict__ block_scales, // [num_m_blocks, num_k_blocks] scales
    int M, int K,
    int num_m_blocks, int num_k_blocks);

// =============================================================================
// Host API Functions (implementations in wan_block_quant.cu)
// =============================================================================

/**
 * Quantize FP16 tensor to INT8 with per-128x128-block scales.
 *
 * @param src Input tensor [M, K] in fp16 (row-major)
 * @param dst Output tensor [M, K] in int8 (row-major)
 * @param block_scales Output scales [M/128, K/128] in float32 (row-major)
 * @param M Number of rows
 * @param K Number of columns
 * @param stream CUDA stream
 * @param debug Enable debug output
 * @return cudaSuccess on success
 */
cudaError_t quantize_per_block_128(
    const half* src,
    int8_t* dst,
    float* block_scales,
    int M, int K,
    cudaStream_t stream,
    bool debug = false);

/**
 * Dequantize INT8 tensor to FP16 with per-128x128-block scales.
 *
 * @param src Input tensor [M, K] in int8 (row-major)
 * @param dst Output tensor [M, K] in fp16 (row-major)
 * @param block_scales Input scales [M/128, K/128] in float32 (row-major)
 * @param M Number of rows
 * @param K Number of columns
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
cudaError_t dequantize_per_block_128(
    const int8_t* src,
    half* dst,
    const float* block_scales,
    int M, int K,
    cudaStream_t stream);

// =============================================================================
// Swizzled Weight Quantization (for Tensor Core optimization)
// =============================================================================

/**
 * Quantize weights and store with swizzle pattern for MMA B operand.
 *
 * This kernel performs three operations:
 * 1. Quantize FP16 weights to INT8 with per-block scales
 * 2. Transpose from [K, N] to [N, K] for column-major MMA access
 * 3. Apply swizzle pattern for bank-conflict-free ldmatrix loading
 *
 * The output format is optimized for the m16n8k32 INT8 tensor core MMA:
 * - B operand expects column-major data (N-major, K-minor)
 * - Swizzle ensures 32 threads can load without bank conflicts
 *
 * Grid: (num_n_tiles, num_k_tiles)
 * Block: (128) - processes one 128x128 output tile
 *
 * @param src Input weights [K, N] row-major (standard PyTorch layout)
 * @param dst_swizzled Output [N_padded, K_padded] swizzled for MMA
 * @param block_scales Output scales [K/128, N/128] row-major
 * @param K Input feature dimension (rows of weight matrix)
 * @param N Output feature dimension (cols of weight matrix)
 * @param K_padded Padded K dimension (multiple of 128)
 * @param N_padded Padded N dimension (multiple of 128)
 * @param num_k_blocks Number of K blocks
 * @param num_n_blocks Number of N blocks
 */
__global__ void quantize_weights_swizzled_128_kernel(
    const half* __restrict__ src,
    int8_t* __restrict__ dst_swizzled,
    float* __restrict__ block_scales,
    int K, int N,
    int K_padded, int N_padded,
    int num_k_blocks, int num_n_blocks);

/**
 * Host API for swizzled weight quantization.
 *
 * Allocates padded dimensions automatically and applies swizzle pattern.
 * The caller must provide output buffers of sufficient size:
 * - dst_swizzled: N_padded * K_padded bytes
 * - block_scales: (K/128) * (N/128) * sizeof(float) bytes
 *
 * Use get_swizzled_weight_dims() and get_swizzled_weight_size() from
 * wan_int8_swizzle.cuh to calculate required buffer sizes.
 *
 * @param src Input weights [K, N] row-major fp16
 * @param dst_swizzled Output [N_padded, K_padded] swizzled int8
 * @param block_scales Output scales [K/128, N/128] float32
 * @param K Input feature dimension
 * @param N Output feature dimension
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
cudaError_t quantize_weights_swizzled_128(
    const half* src,
    int8_t* dst_swizzled,
    float* block_scales,
    int K, int N,
    cudaStream_t stream);

/**
 * Dequantize swizzled weights back to FP16 for verification.
 *
 * This reverses the swizzle and transpose to produce [K, N] row-major output.
 * Primarily used for debugging and accuracy verification.
 *
 * @param src_swizzled Input [N_padded, K_padded] swizzled int8
 * @param dst Output [K, N] row-major fp16
 * @param block_scales Input scales [K/128, N/128] float32
 * @param K Output rows
 * @param N Output cols
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
cudaError_t dequantize_weights_swizzled_128(
    const int8_t* src_swizzled,
    half* dst,
    const float* block_scales,
    int K, int N,
    cudaStream_t stream);

/**
 * Swizzle pre-quantized INT8 weights (no re-quantization).
 *
 * This function takes already-quantized INT8 weights and applies only the
 * swizzle pattern needed for efficient tensor core access. Use this when
 * loading weights from a pre-quantized checkpoint.
 *
 * Input:  src [K, N] row-major INT8
 * Output: dst_swizzled [num_tiles, 128, 128] where num_tiles = (N/128) * (K/128)
 *
 * @param src Source INT8 weights [K, N] row-major
 * @param dst_swizzled Output swizzled weights [num_tiles, 128, 128]
 * @param K Input rows (in_features)
 * @param N Input cols (out_features)
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
cudaError_t swizzle_int8_weights(
    const int8_t* src,
    int8_t* dst_swizzled,
    int K, int N,
    cudaStream_t stream);

/**
 * Check if weights are already in swizzled format.
 *
 * Swizzled format: [num_tiles, 128, 128] where num_tiles = (K/128) * (N/128)
 * Non-swizzled:    [K, N]
 *
 * @param dim0, dim1, dim2 Tensor dimensions (dim2 = -1 for 2D tensors)
 * @param K Original K dimension
 * @param N Original N dimension
 * @return true if tensor is in swizzled format
 */
bool is_swizzled_format(int64_t dim0, int64_t dim1, int64_t dim2, int K, int N);

// =============================================================================
// Utility Functions (inline, safe to include in header)
// =============================================================================

/**
 * Calculate the number of bytes needed for per-block scales.
 *
 * @param M Number of rows in the tensor
 * @param K Number of columns in the tensor
 * @return Size in bytes for the scale tensor
 */
inline size_t calculate_block_scales_size(int M, int K) {
    int num_m_blocks = (M + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;
    int num_k_blocks = (K + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;
    return num_m_blocks * num_k_blocks * sizeof(float);
}

/**
 * Get the shape of the scale tensor.
 *
 * @param M Number of rows in the tensor
 * @param K Number of columns in the tensor
 * @param num_m_blocks Output: number of M blocks
 * @param num_k_blocks Output: number of K blocks
 */
inline void get_block_scales_shape(int M, int K, int* num_m_blocks, int* num_k_blocks) {
    *num_m_blocks = (M + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;
    *num_k_blocks = (K + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;
}

} // namespace omnidreams_singleview
