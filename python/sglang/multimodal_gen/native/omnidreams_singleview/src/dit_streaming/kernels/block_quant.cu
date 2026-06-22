// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-block INT8 quantization kernel implementations.
 *
 * This file contains the CUDA kernel implementations for per-block quantization.
 * The header file (wan_block_quant.cuh) contains only declarations.
 */

#include "block_quant.cuh"
#include "int8_swizzle.cuh"

namespace omnidreams_singleview {

// =============================================================================
// Per-Block Quantization Kernel
// =============================================================================

__global__ void quantize_per_block_128_kernel(
    const half* __restrict__ src,
    int8_t* __restrict__ dst,
    float* __restrict__ block_scales,
    int M, int K,
    int num_m_blocks, int num_k_blocks
) {
    // Block coordinates
    int k_blk = blockIdx.x;
    int m_blk = blockIdx.y;

    if (k_blk >= num_k_blocks || m_blk >= num_m_blocks) return;

    // Thread index
    int tid = threadIdx.x;

    // Block boundaries (handle edge cases)
    int m_start = m_blk * QUANT_BLOCK_SIZE;
    int m_end = min(m_start + QUANT_BLOCK_SIZE, M);
    int k_start = k_blk * QUANT_BLOCK_SIZE;
    int k_end = min(k_start + QUANT_BLOCK_SIZE, K);

    // Shared memory for parallel reduction
    __shared__ float s_amax[128];

    // Step 1: Each thread computes local amax for its column
    float local_amax = 0.0f;
    int k_col = k_start + tid;

    if (k_col < k_end) {
        for (int m_row = m_start; m_row < m_end; m_row++) {
            float v = __half2float(src[m_row * K + k_col]);
            local_amax = fmaxf(local_amax, fabsf(v));
        }
    }

    // Step 2: Block-level reduction for amax
    s_amax[tid] = local_amax;
    __syncthreads();

    // Parallel reduction
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + stride]);
        }
        __syncthreads();
    }

    // Block amax and scale
    float block_amax = s_amax[0];
    float scale = fmaxf(block_amax / 127.0f, 1e-8f);
    float inv_scale = 1.0f / scale;

    // Step 3: Store scale (one thread)
    if (tid == 0) {
        block_scales[m_blk * num_k_blocks + k_blk] = scale;
    }

    // Step 4: Quantize and store
    if (k_col < k_end) {
        for (int m_row = m_start; m_row < m_end; m_row++) {
            float v = __half2float(src[m_row * K + k_col]);
            float scaled = v * inv_scale;
            int8_t quantized = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(scaled))));
            dst[m_row * K + k_col] = quantized;
        }
    }
}

__global__ void quantize_per_block_128_tiled_kernel(
    const half* __restrict__ src,
    int8_t* __restrict__ dst,
    float* __restrict__ block_scales,
    int M, int K,
    int num_m_blocks, int num_k_blocks
) {
    // Block coordinates
    int k_blk = blockIdx.x;
    int m_blk = blockIdx.y;

    if (k_blk >= num_k_blocks || m_blk >= num_m_blocks) return;

    // Thread indices
    int tx = threadIdx.x;  // 0-31
    int ty = threadIdx.y;  // 0-3
    int tid = ty * 32 + tx; // 0-127

    // Block boundaries
    int m_start = m_blk * QUANT_BLOCK_SIZE;
    int m_end = min(m_start + QUANT_BLOCK_SIZE, M);
    int k_start = k_blk * QUANT_BLOCK_SIZE;
    int k_end = min(k_start + QUANT_BLOCK_SIZE, K);

    int block_m = m_end - m_start;
    int block_k = k_end - k_start;

    // Shared memory for reduction
    __shared__ float s_amax[128];

    // Step 1: Compute local amax
    // Each thread handles multiple elements: 128*128/128 = 128 elements per thread
    float local_amax = 0.0f;

    for (int elem = tid; elem < QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE; elem += 128) {
        int local_m = elem / QUANT_BLOCK_SIZE;
        int local_k = elem % QUANT_BLOCK_SIZE;

        if (local_m < block_m && local_k < block_k) {
            int global_m = m_start + local_m;
            int global_k = k_start + local_k;
            float v = __half2float(src[global_m * K + global_k]);
            local_amax = fmaxf(local_amax, fabsf(v));
        }
    }

    // Step 2: Block-level reduction
    s_amax[tid] = local_amax;
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + stride]);
        }
        __syncthreads();
    }

    float block_amax = s_amax[0];
    float scale = fmaxf(block_amax / 127.0f, 1e-8f);
    float inv_scale = 1.0f / scale;

    // Step 3: Store scale
    if (tid == 0) {
        block_scales[m_blk * num_k_blocks + k_blk] = scale;
    }

    // Step 4: Quantize and store
    for (int elem = tid; elem < QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE; elem += 128) {
        int local_m = elem / QUANT_BLOCK_SIZE;
        int local_k = elem % QUANT_BLOCK_SIZE;

        if (local_m < block_m && local_k < block_k) {
            int global_m = m_start + local_m;
            int global_k = k_start + local_k;
            float v = __half2float(src[global_m * K + global_k]);
            float scaled = v * inv_scale;
            int8_t quantized = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(scaled))));
            dst[global_m * K + global_k] = quantized;
        }
    }
}

__global__ void dequantize_per_block_128_kernel(
    const int8_t* __restrict__ src,
    half* __restrict__ dst,
    const float* __restrict__ block_scales,
    int M, int K,
    int num_m_blocks, int num_k_blocks
) {
    int k_blk = blockIdx.x;
    int m_blk = blockIdx.y;

    if (k_blk >= num_k_blocks || m_blk >= num_m_blocks) return;

    int tid = threadIdx.x;

    int m_start = m_blk * QUANT_BLOCK_SIZE;
    int m_end = min(m_start + QUANT_BLOCK_SIZE, M);
    int k_start = k_blk * QUANT_BLOCK_SIZE;
    int k_end = min(k_start + QUANT_BLOCK_SIZE, K);

    float scale = block_scales[m_blk * num_k_blocks + k_blk];

    int k_col = k_start + tid;
    if (k_col < k_end) {
        for (int m_row = m_start; m_row < m_end; m_row++) {
            float v = static_cast<float>(src[m_row * K + k_col]) * scale;
            dst[m_row * K + k_col] = __float2half(v);
        }
    }
}

// =============================================================================
// Fast Single-Pass Quantization Kernel
// =============================================================================
//
// Single-pass, register-resident, vectorized quantization.
// 256 threads per CTA, 2 threads per row, each handles 64 consecutive elements.
// Loads FP16 once into float32 registers, computes amax via warp shuffle + atomic,
// quantizes from registers, stores INT8 with vectorized int4 writes.
// One DRAM read instead of two. ~2x faster than the tiled kernel above.
//
// Requires K to be a multiple of 128 (always true in our pipeline: K=1536, 4608, 8960).

// Device function with compile-time IsEvenM specialization.
// Two non-template __global__ wrappers below avoid MSVC __cudaLaunch macro issues
// with template kernel launches.
template<bool IsEvenM>
__device__ __forceinline__ void quantize_block_fast_impl(
    const half* __restrict__ src,
    int8_t* __restrict__ dst,
    float* __restrict__ block_scales,
    int M, int K,
    int num_m_blocks, int num_k_blocks)
{
    constexpr int BLOCK = 128;
    constexpr int ELEMS = 64;

    const int k_blk = blockIdx.x;
    const int m_blk = blockIdx.y;
    const int tid = threadIdx.x;

    // 2 threads per row: thread handles cols [col_start, col_start+64)
    const int row = tid >> 1;            // 0..127
    const int col_half = tid & 1;        // 0 or 1
    const int col_start = col_half << 6; // 0 or 64

    const int global_m = m_blk * BLOCK + row;
    const int global_k = k_blk * BLOCK + col_start;
    const bool row_valid = IsEvenM || (global_m < M);

    // ---- Phase 1: Load FP16 -> float32 registers (8 x int4 = 64 halfs) ----
    float data[ELEMS];

    if (row_valid) {
        const int4* load_ptr = reinterpret_cast<const int4*>(
            src + static_cast<int64_t>(global_m) * K + global_k);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            union { int4 vec; half h[8]; } u;
            u.vec = load_ptr[i];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                data[i * 8 + j] = __half2float(u.h[j]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < ELEMS; i++) data[i] = 0.0f;
    }

    // ---- Phase 2: Block amax via warp shuffle + atomic ----
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS; i++)
        amax = fmaxf(amax, fabsf(data[i]));

    // Warp-level butterfly reduction (all lanes get warp max)
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));

    // CTA-level reduction: atomicMax on uint32 works for positive floats
    // because IEEE 754 bit representation is monotonically increasing.
    __shared__ unsigned int smem_amax;
    if (tid == 0) smem_amax = 0u;
    __syncthreads();

    atomicMax(&smem_amax, __float_as_uint(amax));
    __syncthreads();

    const float block_amax = __uint_as_float(smem_amax);
    const float scale = fmaxf(block_amax / 127.0f, 1e-8f);
    const float inv_scale = 1.0f / scale;

    if (tid == 0) {
        block_scales[m_blk * num_k_blocks + k_blk] = scale;
    }

    // ---- Phase 3: Quantize from registers + vectorized store (4 x int4) ----
    if (row_valid) {
        int4* store_ptr = reinterpret_cast<int4*>(
            dst + static_cast<int64_t>(global_m) * K + global_k);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Pack 16 int8 values into one int4 (16 bytes) and store
            union { int8_t b[16]; int4 vec; } pack;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int v = __float2int_rn(data[i * 16 + j] * inv_scale);
                v = max(-127, min(127, v));
                pack.b[j] = static_cast<int8_t>(v);
            }
            store_ptr[i] = pack.vec;
        }
    }
}

// M is a multiple of 128 -- no bounds checking on rows
__global__ __launch_bounds__(256)
void quantize_block_fast_even_kernel(
    const half* __restrict__ src,
    int8_t* __restrict__ dst,
    float* __restrict__ block_scales,
    int M, int K,
    int num_m_blocks, int num_k_blocks)
{
    quantize_block_fast_impl<true>(src, dst, block_scales, M, K, num_m_blocks, num_k_blocks);
}

// M is NOT a multiple of 128 -- last block checks row bounds
__global__ __launch_bounds__(256)
void quantize_block_fast_partial_kernel(
    const half* __restrict__ src,
    int8_t* __restrict__ dst,
    float* __restrict__ block_scales,
    int M, int K,
    int num_m_blocks, int num_k_blocks)
{
    quantize_block_fast_impl<false>(src, dst, block_scales, M, K, num_m_blocks, num_k_blocks);
}

// =============================================================================
// Host API Functions
// =============================================================================

cudaError_t quantize_per_block_128(
    const half* src,
    int8_t* dst,
    float* block_scales,
    int M, int K,
    cudaStream_t stream,
    bool debug
) {
    int num_m_blocks = (M + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;
    int num_k_blocks = (K + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;

    dim3 grid(num_k_blocks, num_m_blocks);

    if (K % QUANT_BLOCK_SIZE == 0) {
        // Fast path: single-pass vectorized kernel (requires K aligned to 128)
        dim3 block(256);
        if (M % QUANT_BLOCK_SIZE == 0) {
            quantize_block_fast_even_kernel<<<grid, block, 0, stream>>>(
                src, dst, block_scales, M, K, num_m_blocks, num_k_blocks);
        } else {
            quantize_block_fast_partial_kernel<<<grid, block, 0, stream>>>(
                src, dst, block_scales, M, K, num_m_blocks, num_k_blocks);
        }
    } else {
        // Fallback: old tiled kernel for non-128-aligned K
        dim3 block(32, 4);
        quantize_per_block_128_tiled_kernel<<<grid, block, 0, stream>>>(
            src, dst, block_scales, M, K, num_m_blocks, num_k_blocks);
    }

    if (debug) {
        cudaStreamSynchronize(stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] quantize_per_block_128: %s\n", cudaGetErrorString(err));
            return err;
        }
        printf("[DEBUG] quantize_per_block_128: M=%d, K=%d, blocks=(%d, %d)\n",
               M, K, num_m_blocks, num_k_blocks);
    }

    return cudaSuccess;
}

cudaError_t dequantize_per_block_128(
    const int8_t* src,
    half* dst,
    const float* block_scales,
    int M, int K,
    cudaStream_t stream
) {
    int num_m_blocks = (M + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;
    int num_k_blocks = (K + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;

    dim3 grid(num_k_blocks, num_m_blocks);
    dim3 block(128);

    dequantize_per_block_128_kernel<<<grid, block, 0, stream>>>(
        src, dst, block_scales, M, K, num_m_blocks, num_k_blocks);

    return cudaSuccess;
}

// =============================================================================
// Swizzled Weight Quantization Kernels
// =============================================================================

/**
 * Quantize weights with swizzle pattern for tensor core MMA.
 *
 * This kernel processes one 128x128 block of the output (N, K) tile.
 * Input is [K, N] row-major, output is [N, K] with swizzle applied.
 *
 * Each CTA:
 * 1. Loads a 128x128 tile from src[k_base:k_base+128, n_base:n_base+128]
 * 2. Computes per-block amax and scale
 * 3. Quantizes to INT8
 * 4. Stores to dst with swizzle pattern applied
 */
__global__ void quantize_weights_swizzled_128_kernel(
    const half* __restrict__ src,       // [K, N] row-major
    int8_t* __restrict__ dst_swizzled,  // [N_padded, K_padded] swizzled
    float* __restrict__ block_scales,   // [num_k_blocks, num_n_blocks]
    int K, int N,
    int K_padded, int N_padded,
    int num_k_blocks, int num_n_blocks
) {
    // Block coordinates: processing output tile [n_blk*128:(n_blk+1)*128, k_blk*128:(k_blk+1)*128]
    // which corresponds to input tile [k_blk*128:(k_blk+1)*128, n_blk*128:(n_blk+1)*128]
    int n_blk = blockIdx.x;
    int k_blk = blockIdx.y;

    if (n_blk >= num_n_blocks || k_blk >= num_k_blocks) return;

    int tid = threadIdx.x;

    // Input boundaries
    int k_start = k_blk * QUANT_BLOCK_SIZE;
    int k_end = min(k_start + QUANT_BLOCK_SIZE, K);
    int n_start = n_blk * QUANT_BLOCK_SIZE;
    int n_end = min(n_start + QUANT_BLOCK_SIZE, N);

    int block_k = k_end - k_start;
    int block_n = n_end - n_start;

    // Shared memory for reduction and temporary storage
    __shared__ float s_amax[128];
    __shared__ float s_scale;
    __shared__ float s_inv_scale;

    // Step 1: Compute local amax
    // Each thread processes multiple elements: 128*128/128 = 128 elements
    float local_amax = 0.0f;

    for (int elem = tid; elem < QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE; elem += 128) {
        int local_k = elem / QUANT_BLOCK_SIZE;
        int local_n = elem % QUANT_BLOCK_SIZE;

        if (local_k < block_k && local_n < block_n) {
            int global_k = k_start + local_k;
            int global_n = n_start + local_n;
            float v = __half2float(src[global_k * N + global_n]);
            local_amax = fmaxf(local_amax, fabsf(v));
        }
    }

    // Step 2: Block-level reduction for amax
    s_amax[tid] = local_amax;
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + stride]);
        }
        __syncthreads();
    }

    // Compute scale
    if (tid == 0) {
        float block_amax = s_amax[0];
        s_scale = fmaxf(block_amax / 127.0f, 1e-8f);
        s_inv_scale = 1.0f / s_scale;
        // Store scale: scales are [num_k_blocks, num_n_blocks] row-major
        block_scales[k_blk * num_n_blocks + n_blk] = s_scale;
    }
    __syncthreads();

    float inv_scale = s_inv_scale;

    // Step 3: Quantize and store with swizzle
    // Output is [N_padded, K_padded] - each row corresponds to one N index
    // Within each 128x128 output tile, we apply swizzle for bank-conflict-free access

    for (int elem = tid; elem < QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE; elem += 128) {
        int local_k = elem / QUANT_BLOCK_SIZE;
        int local_n = elem % QUANT_BLOCK_SIZE;

        int8_t quantized = 0;

        if (local_k < block_k && local_n < block_n) {
            int global_k = k_start + local_k;
            int global_n = n_start + local_n;
            float v = __half2float(src[global_k * N + global_n]);
            float scaled = v * inv_scale;
            quantized = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(scaled))));
        }

        // Output coordinates: [n_blk*128 + local_n, k_blk*128 + local_k]
        // Apply swizzle within the 128x128 tile
        int out_n = n_blk * QUANT_BLOCK_SIZE + local_n;
        int out_k = k_blk * QUANT_BLOCK_SIZE + local_k;

        if (out_n < N_padded && out_k < K_padded) {
            // Apply swizzle: local_n is the "row" within this tile (N dimension)
            // local_k is the "column" within this tile (K dimension)
            int swizzled_offset = swizzle_smem_offset(local_n, local_k, kSwizzledStrideK);

            // Base offset for this tile in global memory
            // Tiles are stored contiguously, each tile is 128 * kSwizzledStrideK bytes
            int tile_base = (n_blk * num_k_blocks + k_blk) * (QUANT_BLOCK_SIZE * kSwizzledStrideK);

            dst_swizzled[tile_base + swizzled_offset] = quantized;
        }
    }
}

/**
 * Alternative: Store swizzled weights in a more straightforward layout.
 *
 * This version stores the swizzled data as [N_padded][K_padded] with swizzle
 * applied per-row, which is more natural for loading into shared memory.
 */
__global__ void quantize_weights_swizzled_128_v2_kernel(
    const half* __restrict__ src,       // [K, N] row-major
    int8_t* __restrict__ dst_swizzled,  // [N_padded][kSwizzledStrideK * num_k_blocks] with swizzle
    float* __restrict__ block_scales,   // [num_k_blocks, num_n_blocks]
    int K, int N,
    int K_padded, int N_padded,
    int num_k_blocks, int num_n_blocks
) {
    // Block processes one 128x128 quantization block
    int n_blk = blockIdx.x;
    int k_blk = blockIdx.y;

    if (n_blk >= num_n_blocks || k_blk >= num_k_blocks) return;

    int tid = threadIdx.x;

    int k_start = k_blk * QUANT_BLOCK_SIZE;
    int k_end = min(k_start + QUANT_BLOCK_SIZE, K);
    int n_start = n_blk * QUANT_BLOCK_SIZE;
    int n_end = min(n_start + QUANT_BLOCK_SIZE, N);

    int block_k = k_end - k_start;
    int block_n = n_end - n_start;

    __shared__ float s_amax[128];

    // Compute amax
    float local_amax = 0.0f;
    for (int elem = tid; elem < QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE; elem += 128) {
        int local_k = elem / QUANT_BLOCK_SIZE;
        int local_n = elem % QUANT_BLOCK_SIZE;
        if (local_k < block_k && local_n < block_n) {
            float v = __half2float(src[(k_start + local_k) * N + (n_start + local_n)]);
            local_amax = fmaxf(local_amax, fabsf(v));
        }
    }

    s_amax[tid] = local_amax;
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_amax[tid] = fmaxf(s_amax[tid], s_amax[tid + stride]);
        }
        __syncthreads();
    }

    float scale = fmaxf(s_amax[0] / 127.0f, 1e-8f);
    float inv_scale = 1.0f / scale;

    if (tid == 0) {
        block_scales[k_blk * num_n_blocks + n_blk] = scale;
    }

    // Global stride for swizzled output: each row (N) has K_padded bytes
    // but stored with swizzle pattern applied per 128-column chunk
    int global_k_stride = kSwizzledStrideK * num_k_blocks;

    // Quantize and store with swizzle
    for (int elem = tid; elem < QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE; elem += 128) {
        int local_k = elem / QUANT_BLOCK_SIZE;
        int local_n = elem % QUANT_BLOCK_SIZE;

        int8_t quantized = 0;
        if (local_k < block_k && local_n < block_n) {
            float v = __half2float(src[(k_start + local_k) * N + (n_start + local_n)]);
            quantized = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(v * inv_scale))));
        }

        int out_n = n_start + local_n;
        if (out_n < N_padded && (k_start + local_k) < K_padded) {
            // Row in output = out_n
            // Column = k_blk * kSwizzledStrideK + swizzled local offset
            int swizzled_local_k = swizzle_smem_offset(local_n, local_k, kSwizzledStrideK) -
                                   local_n * kSwizzledStrideK;  // Just the column part

            int out_offset = out_n * global_k_stride +
                            k_blk * kSwizzledStrideK +
                            swizzled_local_k;

            dst_swizzled[out_offset] = quantized;
        }
    }
}

cudaError_t quantize_weights_swizzled_128(
    const half* src,
    int8_t* dst_swizzled,
    float* block_scales,
    int K, int N,
    cudaStream_t stream
) {
    int K_padded = pad_to_128(K);
    int N_padded = pad_to_128(N);
    int num_k_blocks = K_padded / QUANT_BLOCK_SIZE;
    int num_n_blocks = N_padded / QUANT_BLOCK_SIZE;

    dim3 grid(num_n_blocks, num_k_blocks);
    dim3 block(128);

    quantize_weights_swizzled_128_kernel<<<grid, block, 0, stream>>>(
        src, dst_swizzled, block_scales,
        K, N, K_padded, N_padded,
        num_k_blocks, num_n_blocks);

    return cudaGetLastError();
}

// =============================================================================
// Dequantize Swizzled Weights (for verification)
// =============================================================================

__global__ void dequantize_weights_swizzled_128_kernel(
    const int8_t* __restrict__ src_swizzled,  // [N_padded][K_padded] swizzled
    half* __restrict__ dst,                    // [K, N] row-major
    const float* __restrict__ block_scales,    // [num_k_blocks, num_n_blocks]
    int K, int N,
    int K_padded, int N_padded,
    int num_k_blocks, int num_n_blocks
) {
    int n_blk = blockIdx.x;
    int k_blk = blockIdx.y;

    if (n_blk >= num_n_blocks || k_blk >= num_k_blocks) return;

    int tid = threadIdx.x;

    int k_start = k_blk * QUANT_BLOCK_SIZE;
    int k_end = min(k_start + QUANT_BLOCK_SIZE, K);
    int n_start = n_blk * QUANT_BLOCK_SIZE;
    int n_end = min(n_start + QUANT_BLOCK_SIZE, N);

    int block_k = k_end - k_start;
    int block_n = n_end - n_start;

    float scale = block_scales[k_blk * num_n_blocks + n_blk];

    // Tile base in swizzled storage
    int tile_base = (n_blk * num_k_blocks + k_blk) * (QUANT_BLOCK_SIZE * kSwizzledStrideK);

    for (int elem = tid; elem < QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE; elem += 128) {
        int local_k = elem / QUANT_BLOCK_SIZE;
        int local_n = elem % QUANT_BLOCK_SIZE;

        if (local_k < block_k && local_n < block_n) {
            // Read from swizzled location
            int swizzled_offset = swizzle_smem_offset(local_n, local_k, kSwizzledStrideK);
            int8_t quantized = src_swizzled[tile_base + swizzled_offset];

            // Dequantize and store to original layout
            float v = static_cast<float>(quantized) * scale;
            int global_k = k_start + local_k;
            int global_n = n_start + local_n;
            dst[global_k * N + global_n] = __float2half(v);
        }
    }
}

cudaError_t dequantize_weights_swizzled_128(
    const int8_t* src_swizzled,
    half* dst,
    const float* block_scales,
    int K, int N,
    cudaStream_t stream
) {
    int K_padded = pad_to_128(K);
    int N_padded = pad_to_128(N);
    int num_k_blocks = K_padded / QUANT_BLOCK_SIZE;
    int num_n_blocks = N_padded / QUANT_BLOCK_SIZE;

    dim3 grid(num_n_blocks, num_k_blocks);
    dim3 block(128);

    dequantize_weights_swizzled_128_kernel<<<grid, block, 0, stream>>>(
        src_swizzled, dst, block_scales,
        K, N, K_padded, N_padded,
        num_k_blocks, num_n_blocks);

    return cudaGetLastError();
}

// =============================================================================
// Swizzle Pre-Quantized INT8 Weights (no re-quantization)
// =============================================================================

/**
 * Swizzle already-quantized INT8 weights into the format expected by the GEMM kernel.
 *
 * This is for weights that are already quantized (e.g., from a checkpoint).
 * It only applies the swizzle pattern without re-quantization.
 *
 * Input:  src [K, N] row-major INT8
 * Output: dst [num_n_blocks * num_k_blocks, 128, 128] swizzled INT8
 *
 * The swizzle pattern is: swizzled_k = local_k ^ ((local_n & 7) << 4)
 */
__global__ void swizzle_int8_weights_kernel(
    const int8_t* __restrict__ src,         // [K, N] row-major
    int8_t* __restrict__ dst_swizzled,      // [num_tiles, 128, 128] swizzled
    int K, int N,
    int K_padded, int N_padded,
    int num_k_blocks, int num_n_blocks
) {
    // Block coordinates: processing tile [n_blk, k_blk]
    int n_blk = blockIdx.x;
    int k_blk = blockIdx.y;

    int n_start = n_blk * QUANT_BLOCK_SIZE;
    int k_start = k_blk * QUANT_BLOCK_SIZE;

    // Output tile index (n-major ordering to match kernel expectations)
    int tile_idx = n_blk * num_k_blocks + k_blk;
    int8_t* tile_out = dst_swizzled + tile_idx * QUANT_BLOCK_SIZE * kSwizzledStrideK;

    // Each thread processes multiple elements
    int tid = threadIdx.x;
    int elements_per_thread = (QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE) / blockDim.x;

    for (int i = 0; i < elements_per_thread; ++i) {
        int linear_idx = tid + i * blockDim.x;
        int local_n = linear_idx / QUANT_BLOCK_SIZE;  // Row in output tile (N dimension)
        int local_k = linear_idx % QUANT_BLOCK_SIZE;  // Col in output tile (K dimension)

        int global_n = n_start + local_n;
        int global_k = k_start + local_k;

        // Read from source (handle out-of-bounds)
        int8_t val = 0;
        if (global_k < K && global_n < N) {
            val = src[global_k * N + global_n];
        }

        // Apply swizzle pattern: XOR row bits 0-2 into col bits 4-6
        int swizzled_k = local_k ^ ((local_n & 7) << 4);
        int out_offset = local_n * kSwizzledStrideK + swizzled_k;

        tile_out[out_offset] = val;
    }
}

cudaError_t swizzle_int8_weights(
    const int8_t* src,
    int8_t* dst_swizzled,
    int K, int N,
    cudaStream_t stream
) {
    int K_padded = pad_to_128(K);
    int N_padded = pad_to_128(N);
    int num_k_blocks = K_padded / QUANT_BLOCK_SIZE;
    int num_n_blocks = N_padded / QUANT_BLOCK_SIZE;

    dim3 grid(num_n_blocks, num_k_blocks);
    dim3 block(128);

    swizzle_int8_weights_kernel<<<grid, block, 0, stream>>>(
        src, dst_swizzled,
        K, N, K_padded, N_padded,
        num_k_blocks, num_n_blocks);

    return cudaGetLastError();
}

/**
 * Check if weights are already in swizzled format.
 *
 * Swizzled format: [num_tiles, 128, 128] where num_tiles = (K/128) * (N/128)
 * Non-swizzled:    [K, N]
 *
 * Returns true if the tensor shape matches swizzled format for given K, N.
 */
bool is_swizzled_format(int64_t dim0, int64_t dim1, int64_t dim2, int K, int N) {
    int K_padded = pad_to_128(K);
    int N_padded = pad_to_128(N);
    int num_k_blocks = K_padded / QUANT_BLOCK_SIZE;
    int num_n_blocks = N_padded / QUANT_BLOCK_SIZE;
    int expected_tiles = num_n_blocks * num_k_blocks;

    return (dim0 == expected_tiles && dim1 == 128 && dim2 == 128);
}

} // namespace omnidreams_singleview
