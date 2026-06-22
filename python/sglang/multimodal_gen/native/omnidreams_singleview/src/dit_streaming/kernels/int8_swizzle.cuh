// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Swizzle utilities for INT8 GEMM with SM80 Tensor Cores.
 *
 * This implements the memory swizzle pattern for bank-conflict-free shared memory
 * access when using ldmatrix instructions with INT8 data.
 *
 * Swizzle pattern: Swizzle<BBits=2, MBase=4, SShift=3>
 * - 2 swizzle bits for XOR pattern
 * - 16-byte alignment (MBase=4 means lower 4 bits constant)
 * - Shift distance of 3 for row contribution
 *
 * This creates a permutation that distributes accesses across all 32 banks,
 * eliminating bank conflicts when 32 threads in a warp access shared memory
 * in the pattern required by ldmatrix.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace omnidreams_singleview {

// =============================================================================
// Swizzle Constants
// =============================================================================

// Swizzle parameters for INT8 with 128-bit (16-byte) access
// Based on CUTLASS TensorOpMultiplicandCongruous pattern
constexpr int kSwizzleBBits = 2;   // 2 swizzle bits
constexpr int kSwizzleMBase = 4;   // Lower 4 bits constant (16-byte alignment)
constexpr int kSwizzleSShift = 3;  // Shift distance

// Shared memory tile dimensions
constexpr int kSwizzleTileM = 128;
constexpr int kSwizzleTileN = 128;
constexpr int kSwizzleTileK = 128;

// Stride for swizzled shared memory - using 128 (no padding) with XOR swizzle
// The XOR swizzle pattern ensures bank-conflict-free access without needing padding
constexpr int kSwizzledStrideK = 128;

// =============================================================================
// Swizzle Functions
// =============================================================================

/**
 * Apply swizzle XOR pattern to compute shared memory offset.
 *
 * For INT8 MMA m16n8k32 fragment loads, each warp has 32 threads loading:
 * - 8 different rows (groupID = lane_id >> 2, range 0-7)
 * - 4 different column offsets (tid_in_grp = lane_id & 3, accessing col = tid*4)
 *
 * To avoid bank conflicts, we XOR row bits [0:2] into column bits [4:6].
 * This is CUTLASS Swizzle<3,4,3>: XOR 3 bits from offset position 7
 * (=row bits) into offset position 4 (=col bits 4-6). This preserves
 * 16-byte alignment (bits 0-3 unchanged) for LDMATRIX compatibility.
 *
 * With stride 128:
 *   swizzled_col = col ^ ((row & 7) << 4)
 *   bank = (row * 128 / 4 + swizzled_col / 4) % 32
 *
 * This maps warp-wide access patterns to unique banks.
 *
 * @param row Row index within shared memory tile
 * @param col Column byte offset within row (0-127)
 * @param stride Row stride in bytes (should be kSwizzledStrideK = 128)
 * @return Swizzled byte offset into shared memory
 */
__device__ __forceinline__
int swizzle_smem_offset(int row, int col, int stride) {
    // Must match cute Swizzle<3, 4, 3>: XOR row bits 0-2 into column bits 4-6
    // Preserves 16-byte alignment (bits 0-3 unchanged) for LDMATRIX compatibility
    int swizzled_col = col ^ ((row & 7) << 4);
    return row * stride + swizzled_col;
}

/**
 * Compute swizzled shared memory address.
 *
 * Converts a logical (row, col) coordinate to a swizzled shared memory pointer.
 * The returned address can be used directly with cp.async or ldmatrix.
 *
 * @param smem_base Base pointer to shared memory tile
 * @param row Row index within tile
 * @param col Column byte offset within row
 * @param stride Row stride in bytes
 * @return Shared memory address (as uint32_t for PTX instructions)
 */
__device__ __forceinline__
uint32_t get_swizzled_smem_addr(const int8_t* smem_base, int row, int col, int stride) {
    int offset = swizzle_smem_offset(row, col, stride);
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_base + offset));
}

/**
 * Inverse swizzle: given swizzled offset, recover original (row, col).
 *
 * Useful for debugging and verification that swizzle is self-inverse
 * within the relevant bit positions.
 *
 * @param row Row index
 * @param swizzled_col Column after swizzle (0-127)
 * @param stride Row stride
 * @return Original column byte offset
 */
__device__ __forceinline__
int inverse_swizzle_col(int row, int swizzled_col, int stride) {
    // Swizzle<3,4,3>: XOR is self-inverse, so inverse = forward
    return swizzled_col ^ ((row & 7) << 4);
}

// =============================================================================
// Utility Functions for Padded Dimensions
// =============================================================================

/**
 * Round up dimension to multiple of 128 for swizzled storage.
 */
__host__ __device__ __forceinline__
int pad_to_128(int dim) {
    return ((dim + 127) / 128) * 128;
}

/**
 * Calculate padded dimensions for swizzled weight storage.
 *
 * Weights are stored as [N_padded, K_padded] for column-major MMA access.
 *
 * @param K Input feature dimension (original)
 * @param N Output feature dimension (original)
 * @param K_padded Output: padded K dimension
 * @param N_padded Output: padded N dimension
 */
__host__ __forceinline__
void get_swizzled_weight_dims(int K, int N, int* K_padded, int* N_padded) {
    *K_padded = pad_to_128(K);
    *N_padded = pad_to_128(N);
}

/**
 * Calculate storage size for swizzled weights in bytes.
 *
 * @param K Input feature dimension
 * @param N Output feature dimension
 * @return Size in bytes for swizzled weight storage
 */
__host__ __forceinline__
size_t get_swizzled_weight_size(int K, int N) {
    int K_padded = pad_to_128(K);
    int N_padded = pad_to_128(N);
    return static_cast<size_t>(N_padded) * K_padded;
}

// =============================================================================
// Bank Conflict Analysis (for debugging)
// =============================================================================

/**
 * Compute shared memory bank for a given address.
 *
 * Shared memory has 32 banks, each 4 bytes wide.
 * Bank = (address / 4) % 32
 *
 * @param addr Shared memory byte address
 * @return Bank index (0-31)
 */
__device__ __forceinline__
int get_smem_bank(int addr) {
    return (addr >> 2) & 0x1F;
}

/**
 * Check if swizzle pattern eliminates bank conflicts for ldmatrix access.
 *
 * For ldmatrix with INT8, 32 threads access 32 consecutive 4-byte chunks.
 * Without swizzle, this causes 4-way bank conflicts.
 * With proper swizzle, each thread accesses a different bank.
 *
 * This function is for debugging - call from a single thread to verify
 * the swizzle pattern.
 *
 * @param smem_base Base shared memory pointer
 * @param tile_row Starting row for the 16x32 tile
 * @param tile_col Starting column for the tile
 * @param stride Shared memory stride
 */
__device__ inline
void debug_check_bank_conflicts(const int8_t* smem_base, int tile_row, int tile_col, int stride) {
#ifdef DEBUG_SWIZZLE
    // Check banks for first row of ldmatrix pattern
    int banks[32];
    for (int lane = 0; lane < 32; ++lane) {
        int groupID = lane >> 2;
        int tid_in_grp = lane & 3;
        int row = tile_row + groupID;
        int col = tile_col + tid_in_grp * 4;
        int offset = swizzle_smem_offset(row, col, stride);
        banks[lane] = get_smem_bank(offset);
    }

    // Check for duplicates (bank conflicts)
    bool has_conflict = false;
    for (int i = 0; i < 32 && !has_conflict; ++i) {
        for (int j = i + 1; j < 32; ++j) {
            if (banks[i] == banks[j]) {
                printf("[SWIZZLE DEBUG] Bank conflict: lane %d and %d both use bank %d\n",
                       i, j, banks[i]);
                has_conflict = true;
                break;
            }
        }
    }
    if (!has_conflict) {
        printf("[SWIZZLE DEBUG] No bank conflicts detected for tile (%d, %d)\n",
               tile_row, tile_col);
    }
#endif
}

} // namespace omnidreams_singleview
