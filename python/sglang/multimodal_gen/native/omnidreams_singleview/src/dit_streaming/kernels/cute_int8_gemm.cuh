// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * INT8 GEMM for SM80+ Tensor Cores.
 *
 * High-performance implementation with:
 * - 128x128 output tiles with K=128 (matches quantization blocks)
 * - 256 threads (8 warps), 4x2 warp layout
 * - XOR swizzle for bank-conflict-free shared memory
 * - LDMATRIX (SM75+) for efficient shared-to-register loads
 * - cp.async for asynchronous global->shared transfers
 * - Triple buffering (3-stage pipeline) for latency hiding
 * - Per-block (128x128) quantization scale support
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "block_quant.cuh"

// CUTLASS cute for LDMATRIX S2R copy and MMA dispatch
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

namespace omnidreams_singleview {

using namespace cute;

// =============================================================================
// Configuration Constants
// =============================================================================

// Tile sizes
static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kTileK = 128;  // Match quantization block

// MMA shape (m16n8k32 for INT8)
static constexpr int kMmaM = 16;
static constexpr int kMmaN = 8;
static constexpr int kMmaK = 32;

// Thread configuration
static constexpr int kThreads = 256;  // 8 warps

// Warp layout: 4 warps along M, 2 along N
static constexpr int kWarpsM = 4;
static constexpr int kWarpsN = 2;

// Per-warp output tile
static constexpr int kWarpTileM = kTileM / kWarpsM;  // 32
static constexpr int kWarpTileN = kTileN / kWarpsN;   // 64

// MMA iterations per warp
static constexpr int kMmasM = kWarpTileM / kMmaM;  // 2
static constexpr int kMmasN = kWarpTileN / kMmaN;   // 8

// Shared memory layout
static constexpr int kSmemStrideK = 128;  // With XOR swizzle, no padding needed
static constexpr int kTileBytes = kTileM * kSmemStrideK;  // 16384 bytes per A or B tile
static constexpr int kStageBytes = 2 * kTileBytes;        // 32768 bytes per stage (A+B)
static constexpr int kSmemSize3 = 3 * kStageBytes;        // 98304 bytes (3-stage pipeline)
static constexpr int kSmemSize2 = 2 * kStageBytes;        // 65536 bytes (2-stage fallback)
// Legacy alias used by configure_smem -- always tries 3-stage first
static constexpr int kSmemSize = kSmemSize3;

// Pre-swizzled B storage: 128-row tiles (matches tile size exactly)
static constexpr int kBTile128Bytes = 128 * kSmemStrideK;  // 16384 bytes

// Quantization block size
static constexpr int kQuantBlock = 128;

// Loads per thread: 128*128 / (256*16) = 4
static constexpr int kLoadsPerThread = (kTileM * kTileK) / (kThreads * 16);

// Per-thread accumulator count: 2 MMA-M * 8 MMA-N * 4 outputs/MMA = 64
static constexpr int kAccSize = kMmasM * kMmasN * 4;

// Epilogue: coalesced store via shared memory
// Padded stride avoids bank conflicts on 16-byte reads (4 banks shift per row)
static constexpr int kEpilogueStride = kTileN + 8;            // 136 halves per row
static constexpr int kEpilogueColGroups = kTileN / 8;         // 16 groups of 8 halves
static constexpr int kEpilogueRowsPerIter = kThreads / kEpilogueColGroups;  // 16 rows
static constexpr int kEpilogueIters = kTileM / kEpilogueRowsPerIter;        // 8 iters

// INT8 epilogue: coalesced 16-byte (16 int8_t) stores
static constexpr int kInt8EpilogueStride = kTileN + 16;       // 144 bytes per row (pad to avoid bank conflicts)
static constexpr int kInt8ColGroups = kTileN / 16;            // 8 groups of 16 bytes
static constexpr int kInt8RowsPerIter = kThreads / kInt8ColGroups;  // 32 rows
static constexpr int kInt8Iters = kTileM / kInt8RowsPerIter;        // 4 iters

// =============================================================================
// Cute Type Definitions for LDMATRIX S2R and MMA
// =============================================================================

// Smem layout atom matching our XOR swizzle: col ^ ((row & 7) << 3)
// Swizzle<B=3, M=4, S=3>: XOR 3 bits from position 7 (row) into position 3 (col)
// This pattern is compatible with LDMATRIX 16-byte alignment
using SmemLayoutAtom_ = decltype(
    composition(
        Swizzle<3, 4, 3>{},
        make_layout(
            make_shape(Int<8>{}, Int<kTileK>{}),
            make_stride(Int<kTileK>{}, Int<1>{})
        )
    )
);

// Full tile layout (128x128) by tiling the 8-row atom
using SmemLayoutTile_ = decltype(
    tile_to_shape(SmemLayoutAtom_{}, make_shape(Int<kTileM>{}, Int<kTileK>{}))
);

// TiledMma: SM80 INT8 m16n8k32 with 4x2 warp layout
// Matches TurboDiffusion's proven configuration for 128x128 tiles
using MmaOp_ = SM80_16x8x32_S32S8S8S32_TN;
using CuteTiledMma_ = decltype(
    make_tiled_mma(
        MMA_Atom<MMA_Traits<MmaOp_>>{},
        make_layout(make_shape(_4{}, _2{}, _1{})),
        make_tile(Int<64>{}, Int<32>{}, Int<32>{})
    )
);

// S2R copy using LDMATRIX (warp-collective, 1 instruction per fragment)
using S2RCopyAtomA_ = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, int8_t>;
using S2RCopyAtomB_ = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, int8_t>;
using S2RTiledCopyA_ = decltype(make_tiled_copy_A(S2RCopyAtomA_{}, CuteTiledMma_{}));
using S2RTiledCopyB_ = decltype(make_tiled_copy_B(S2RCopyAtomB_{}, CuteTiledMma_{}));

// =============================================================================
// GELU Activation
// =============================================================================

__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float kBeta = 0.044715f;
    float y = kAlpha * (x + kBeta * x * x * x);
    return 0.5f * x * (1.0f + tanhf(y));
}

// =============================================================================
// Swizzle Functions (used by manual G2S copy)
// =============================================================================

__device__ __forceinline__
int swizzle_offset(int row, int col) {
    // Must match cute Swizzle<3, 4, 3>: XOR bits 7-9 into bits 4-6
    // = col ^ ((row & 7) << 4)
    int swizzled_col = col ^ ((row & 7) << 4);
    return row * kSmemStrideK + swizzled_col;
}

__device__ __forceinline__
uint32_t get_swizzled_smem_addr(const int8_t* smem_base, int row, int col) {
    int offset = swizzle_offset(row, col);
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_base + offset));
}

// =============================================================================
// cp.async Helpers (used by manual G2S copy)
// =============================================================================

__device__ __forceinline__
void cp_async_16B_swizzled(int8_t* smem_base, const int8_t* gmem_ptr,
                           int row, int col, bool valid) {
    uint32_t smem_addr = get_swizzled_smem_addr(smem_base, row, col);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.b32 [%0], {0, 0, 0, 0};\n"
        "}\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"((int)valid)
    );
#else
    if (valid) {
        *reinterpret_cast<int4*>(smem_base + swizzle_offset(row, col)) =
            *reinterpret_cast<const int4*>(gmem_ptr);
    }
#endif
}

__device__ __forceinline__
void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int N>
__device__ __forceinline__
void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// =============================================================================
// Main Kernel
// =============================================================================

template<bool ApplyGelu = false, bool EmitInt8 = false, bool FuseGatedResidual = false, int NumStages = 3>
static __global__ void __launch_bounds__(kThreads, 1)
int8_gemm_kernel(
    const int8_t* __restrict__ A_ptr,
    const float* __restrict__ A_scales,
    const int8_t* __restrict__ B_swizzled,
    const float* __restrict__ B_scales,
    half* __restrict__ C_ptr,              // FP16 output (unused when EmitInt8/FuseGatedResidual)
    const half* __restrict__ bias,
    int M, int N, int K,
    int num_k_blocks,                      // K/128
    int num_quant_n_blocks,                // N/128
    int8_t* __restrict__ C_int8 = nullptr, // INT8 output (EmitInt8 only)
    float* __restrict__ C_scales = nullptr, // Output scales [M/128, N/128] (EmitInt8 only)
    // Gated residual parameters (FuseGatedResidual only):
    half* __restrict__ residual = nullptr,          // In/Out hidden states [M, N]
    const half* __restrict__ gate_sst = nullptr,    // sst + gate_idx*N, broadcast [N]
    const half* __restrict__ gate_temb = nullptr,   // temb + gate_idx*N, per-row [M, stride]
    int temb_row_stride = 0                         // Row stride of temb
) {
    static_assert(!(EmitInt8 && FuseGatedResidual),
        "EmitInt8 and FuseGatedResidual are mutually exclusive");
    // Thread indexing
    const int tid = threadIdx.x;

    // Block coordinates (128x128 tiles = exactly one quantization block)
    const int m_tile = blockIdx.y;
    const int n_tile = blockIdx.x;
    const int m_base = m_tile * kTileM;
    const int n_base = n_tile * kTileN;

    // Quantization block indices (1:1 with tiles now)
    const int quant_m = m_tile;
    const int quant_n = n_tile;

    // Dynamic shared memory: [A0][B0][A1][B1][A2][B2]
    extern __shared__ int8_t int8_gemm_smem[];
    auto get_smem_A = [&](int stage) -> int8_t* { return int8_gemm_smem + stage * kStageBytes; };
    auto get_smem_B = [&](int stage) -> int8_t* { return int8_gemm_smem + stage * kStageBytes + kTileBytes; };

    // =========================================================================
    // Cute MMA and S2R setup
    // =========================================================================
    CuteTiledMma_ tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // Allocate register fragments using simple (non-swizzled) layouts
    // The fragment shape depends only on tile dimensions, not smem swizzle
    auto tCrA = thr_mma.partition_fragment_A(
        make_tensor(make_gmem_ptr(static_cast<int8_t const*>(nullptr)),
                    make_shape(Int<kTileM>{}, Int<kTileK>{})));
    auto tCrB = thr_mma.partition_fragment_B(
        make_tensor(make_gmem_ptr(static_cast<int8_t const*>(nullptr)),
                    make_shape(Int<kTileN>{}, Int<kTileK>{})));
    auto tDrC = thr_mma.partition_fragment_C(
        make_tensor(make_gmem_ptr(static_cast<int32_t const*>(nullptr)),
                    make_shape(Int<kTileM>{}, Int<kTileN>{})));

    // S2R copy with LDMATRIX
    S2RTiledCopyA_ s2r_copy_a;
    auto s2r_thr_a = s2r_copy_a.get_slice(tid);
    auto tCrA_view = s2r_thr_a.retile_D(tCrA);

    S2RTiledCopyB_ s2r_copy_b;
    auto s2r_thr_b = s2r_copy_b.get_slice(tid);
    auto tCrB_view = s2r_thr_b.retile_D(tCrB);

    // Coordinate mapping for epilogue (identity tensor partitioned like C)
    auto cC = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileN>{}));
    auto tDcC = thr_mma.partition_C(cC);

    // Float accumulators
    float acc[kAccSize];
    #pragma unroll
    for (int i = 0; i < kAccSize; ++i) acc[i] = 0.0f;

    // =========================================================================
    // G2S copy lambdas (manual, with our swizzle)
    // =========================================================================

    // Load A tile (128x128) from global to shared with swizzle
    // Use tid + i*kThreads so threads in a warp load from the same row (coalesced)
    auto load_A = [&](int8_t* buf, int k_base) {
        #pragma unroll
        for (int i = 0; i < kLoadsPerThread; ++i) {
            int linear_idx = tid + i * kThreads;
            int lm = linear_idx / (kTileK / 16);       // row within tile
            int lk = (linear_idx % (kTileK / 16)) * 16; // col within tile
            int gm = m_base + lm;
            int gk = k_base + lk;
            bool valid = (gm < M) && (gk + 15 < K);
            cp_async_16B_swizzled(buf, A_ptr + gm * K + gk, lm, lk, valid);
        }
    };

    // Load B tile (128x128, already pre-swizzled in 128-row tiles)
    auto load_B = [&](int8_t* buf, int k_blk) {
        int64_t b_tile_idx = (int64_t)n_tile * num_k_blocks + k_blk;
        const int8_t* b_tile_ptr = B_swizzled + b_tile_idx * kBTile128Bytes;
        #pragma unroll
        for (int i = 0; i < kLoadsPerThread; ++i) {
            int linear_idx = tid + i * kThreads;
            int byte_offset = linear_idx * 16;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(buf + byte_offset));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(smem_addr), "l"(b_tile_ptr + byte_offset));
#endif
        }
    };

    // =========================================================================
    // Prologue: fill pipeline stages (NumStages-1 stages prefilled)
    // =========================================================================
    if (num_k_blocks > 0) {
        load_A(get_smem_A(0), 0);
        load_B(get_smem_B(0), 0);
        cp_async_commit();
    }
    if constexpr (NumStages >= 3) {
        if (num_k_blocks > 1) {
            load_A(get_smem_A(1), kTileK);
            load_B(get_smem_B(1), 1);
            cp_async_commit();
        }
    }

    // =========================================================================
    // Main loop with NumStages-stage pipeline
    // =========================================================================
    for (int k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
        int curr = k_blk % NumStages;

        if constexpr (NumStages >= 3) {
            // 3-stage: prefetch k_blk+2
            if (k_blk + 2 < num_k_blocks) {
                int fill = (k_blk + 2) % NumStages;
                load_A(get_smem_A(fill), (k_blk + 2) * kTileK);
                load_B(get_smem_B(fill), k_blk + 2);
                cp_async_commit();
                cp_async_wait_group<2>();
            } else if (k_blk + 1 < num_k_blocks) {
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
        } else {
            // 2-stage: prefetch k_blk+1, wait for current
            if (k_blk + 1 < num_k_blocks) {
                int fill = (k_blk + 1) % NumStages;
                load_A(get_smem_A(fill), (k_blk + 1) * kTileK);
                load_B(get_smem_B(fill), k_blk + 1);
                cp_async_commit();
            }
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // Create cute smem tensors for current stage
        auto sA = make_tensor(make_smem_ptr<int8_t>(get_smem_A(curr)), SmemLayoutTile_{});
        auto sB = make_tensor(make_smem_ptr<int8_t>(get_smem_B(curr)), SmemLayoutTile_{});

        // Partition smem for S2R copy (compile-time layout, runtime pointer)
        auto tCsA = s2r_thr_a.partition_S(sA);
        auto tCsB = s2r_thr_b.partition_S(sB);

        // Get quantization scales for this K-block
        float scale_a = A_scales[quant_m * num_k_blocks + k_blk];
        float scale_b = B_scales[k_blk * num_quant_n_blocks + quant_n];
        float combined_scale = scale_a * scale_b;

        // Reset int32 accumulator for this K-block
        clear(tDrC);

        // Inner K loop: LDMATRIX S2R + tensor core MMA
        // nk = kTileK / kMmaK = 128 / 32 = 4 iterations
        #pragma unroll
        for (int ik = 0; ik < size<2>(tCrA); ++ik) {
            cute::copy(s2r_copy_a, tCsA(_, _, ik), tCrA_view(_, _, ik));
            cute::copy(s2r_copy_b, tCsB(_, _, ik), tCrB_view(_, _, ik));
            cute::gemm(tiled_mma, tDrC, tCrA(_, _, ik), tCrB(_, _, ik), tDrC);
        }

        // Dequantize int32 -> float and accumulate
        #pragma unroll
        for (int i = 0; i < kAccSize; ++i) {
            acc[i] += float(tDrC(i)) * combined_scale;
        }
    }

    // =========================================================================
    // Epilogue: Coalesced stores via shared memory
    // =========================================================================

    if constexpr (EmitInt8) {
        // =====================================================================
        // INT8 Epilogue: bias + GELU + absmax reduction + quantize + coalesced stores
        // Works entirely from acc[] registers — no FP16 smem intermediate.
        // Total syncs: 3 (reduction write, reduction read, INT8 scatter)
        // =====================================================================

        // Phase 1: Apply bias + GELU in-place to acc[], compute thread-local absmax
        float thread_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < kAccSize; ++i) {
            float val = acc[i];
            if (bias != nullptr) {
                val += __half2float(bias[n_base + get<1>(tDcC(i))]);
            }
            if constexpr (ApplyGelu) {
                val = gelu_tanh(val);
            }
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            acc[i] = val;
            thread_amax = fmaxf(thread_amax, fabsf(val));
        }

        // Phase 2: Tile-wide absmax reduction (warp shuffle + cross-warp smem)
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFF, thread_amax, offset));
        }

        float* warp_amax = reinterpret_cast<float*>(int8_gemm_smem);  // 8 floats = 32 bytes
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;

        if (lane_id == 0) warp_amax[warp_id] = thread_amax;
        __syncthreads();

        float tile_amax;
        if (tid < 8) {
            tile_amax = warp_amax[tid];
            #pragma unroll
            for (int offset = 4; offset >= 1; offset >>= 1) {
                tile_amax = fmaxf(tile_amax, __shfl_xor_sync(0xFF, tile_amax, offset));
            }
            if (tid == 0) warp_amax[0] = tile_amax;
        }
        __syncthreads();
        tile_amax = warp_amax[0];

        float inv_scale = (tile_amax > 0.0f) ? 127.0f / tile_amax : 0.0f;

        // Write per-tile scale (1:1 with quantization blocks)
        if (tid == 0) {
            C_scales[quant_m * num_quant_n_blocks + quant_n] = tile_amax / 127.0f;
        }

        // Phase 3: Quantize from acc[] registers and scatter INT8 to smem
        // smem layout: [128, kInt8EpilogueStride] int8_t starting at offset 32
        int8_t* smem_int8 = int8_gemm_smem + 32;  // After warp_amax (32 bytes)
        #pragma unroll
        for (int i = 0; i < kAccSize; ++i) {
            int lm = get<0>(tDcC(i));
            int ln = get<1>(tDcC(i));
            int q = __float2int_rn(acc[i] * inv_scale);
            q = max(-127, min(127, q));
            smem_int8[lm * kInt8EpilogueStride + ln] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // Phase 4: Coalesced 16-byte vectorized INT8 stores to global memory
        // 256 threads x 16 bytes = 4096 bytes/iter = 32 rows/iter, 4 iterations
        #pragma unroll
        for (int iter = 0; iter < kInt8Iters; ++iter) {
            int linear = iter * kThreads + tid;
            int row = linear / kInt8ColGroups;
            int col = (linear % kInt8ColGroups) * 16;
            int gm = m_base + row;
            int gn = n_base + col;

            if (gm < M && gn + 15 < N) {
                *reinterpret_cast<int4*>(&C_int8[gm * N + gn]) =
                    *reinterpret_cast<const int4*>(&smem_int8[row * kInt8EpilogueStride + col]);
            } else if (gm < M) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    if (gn + j < N) {
                        C_int8[gm * N + gn + j] = smem_int8[row * kInt8EpilogueStride + col + j];
                    }
                }
            }
        }
    } else {
        // =====================================================================
        // FP16 Epilogue: scatter to smem + coalesced 16-byte vectorized stores
        // =====================================================================
        half* smem_out = reinterpret_cast<half*>(int8_gemm_smem);

        // Phase 1: scatter-write accumulators to smem with bias + GELU
        #pragma unroll
        for (int i = 0; i < kAccSize; ++i) {
            int lm = get<0>(tDcC(i));
            int ln = get<1>(tDcC(i));
            float val = acc[i];
            if (bias != nullptr) {
                val += __half2float(bias[n_base + ln]);
            }
            if constexpr (ApplyGelu) {
                val = gelu_tanh(val);
            }
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            smem_out[lm * kEpilogueStride + ln] = __float2half(val);
        }
        __syncthreads();

        // Phase 2: coalesced 16-byte vectorized stores to global memory
        // 256 threads x 8 halves = 2048 elems/iter = 16 rows/iter, 8 iterations
        #pragma unroll
        for (int iter = 0; iter < kEpilogueIters; ++iter) {
            int linear = iter * kThreads + tid;
            int row = linear / kEpilogueColGroups;
            int col = (linear % kEpilogueColGroups) * 8;
            int gm = m_base + row;
            int gn = n_base + col;

            if constexpr (FuseGatedResidual) {
                // Fused gated residual: hidden = hidden + gemm_out * gate
                // gate = sst[gn] + temb[gm * stride + gn]
                // Eliminates separate dFF2_row write/read (~894 MB DRAM) and kernel #37
                if (gm < M && gn + 7 < N) {
                    int4 gemm_v = *reinterpret_cast<const int4*>(&smem_out[row * kEpilogueStride + col]);
                    int4 sst_v  = *reinterpret_cast<const int4*>(&gate_sst[gn]);
                    int4 temb_v = *reinterpret_cast<const int4*>(&gate_temb[(size_t)gm * temb_row_stride + gn]);
                    int4 res_v  = *reinterpret_cast<const int4*>(&residual[(size_t)gm * N + gn]);
                    half* gv = reinterpret_cast<half*>(&gemm_v);
                    half* sv = reinterpret_cast<half*>(&sst_v);
                    half* tv = reinterpret_cast<half*>(&temb_v);
                    half* rv = reinterpret_cast<half*>(&res_v);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float gate = __half2float(sv[j]) + __half2float(tv[j]);
                        float r = __half2float(rv[j]);
                        float s = __half2float(gv[j]);
                        rv[j] = __float2half(r + s * gate);
                    }
                    *reinterpret_cast<int4*>(&residual[(size_t)gm * N + gn]) = res_v;
                } else if (gm < M) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if (gn + j < N) {
                            float val = __half2float(smem_out[row * kEpilogueStride + col + j]);
                            float gate = __half2float(gate_sst[gn + j])
                                       + __half2float(gate_temb[(size_t)gm * temb_row_stride + gn + j]);
                            float r = __half2float(residual[(size_t)gm * N + gn + j]);
                            residual[(size_t)gm * N + gn + j] = __float2half(r + val * gate);
                        }
                    }
                }
            } else {
                if (gm < M && gn + 7 < N) {
                    *reinterpret_cast<int4*>(&C_ptr[gm * N + gn]) =
                        *reinterpret_cast<const int4*>(&smem_out[row * kEpilogueStride + col]);
                } else if (gm < M) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if (gn + j < N) {
                            C_ptr[gm * N + gn + j] = smem_out[row * kEpilogueStride + col + j];
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Host API
// =============================================================================

namespace detail {

// Determine pipeline stage count for INT8 GEMM.
// Ada/Ampere (SM < 10): 2-stage (64KB). All kernel variants fit.
// Blackwell+ (SM >= 10): 3-stage (96KB). More smem available per SM.
// We use compute capability as the discriminator because cudaFuncSetAttribute
// lies on Ada -- it reports success for 96KB but the GELU kernel variant
// (higher register pressure) actually can't launch with that much smem.
inline int get_num_stages() {
    static int cached = -1;
    if (cached > 0) return cached;

    int device;
    cudaGetDevice(&device);
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);

    if (major >= 10) {
        // Blackwell+: try 3-stage
        cached = 3;
    } else {
        // Ada/Ampere: 2-stage (safe for all kernel variants)
        cached = 2;
    }
    (void)cudaGetLastError();
    return cached;
}

// Configure smem for ONE specific kernel variant right before launch.
// Avoids error poisoning from configuring variants this GPU can't support.
template<bool AG, bool EI, bool FGR, int S>
inline void configure_one_kernel() {
    constexpr int smem = (S >= 3) ? kSmemSize3 : kSmemSize2;
    (void)cudaGetLastError();  // clear any prior error
    cudaFuncSetAttribute(int8_gemm_kernel<AG, EI, FGR, S>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}

}  // namespace detail

/**
 * Launch the INT8 GEMM kernel. Automatically selects 2 or 3 pipeline stages.
 */
inline cudaError_t int8_gemm(
    const int8_t* A, const float* A_scales,
    const int8_t* B, const float* B_scales,
    half* C, const half* bias,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    int num_m_tiles = (M + kTileM - 1) / kTileM;
    int num_n_tiles = (N + kTileN - 1) / kTileN;
    int num_k_blocks = (K + kQuantBlock - 1) / kQuantBlock;
    int num_quant_n_blocks = (N + kQuantBlock - 1) / kQuantBlock;
    dim3 grid(num_n_tiles, num_m_tiles);
    dim3 block(kThreads);

    int stages = detail::get_num_stages();
    if (stages >= 3) {
        detail::configure_one_kernel<false, false, false, 3>();
        int8_gemm_kernel<false, false, false, 3><<<grid, block, kSmemSize3, stream>>>(
            A, A_scales, B, B_scales, C, bias, M, N, K, num_k_blocks, num_quant_n_blocks);
    } else {
        detail::configure_one_kernel<false, false, false, 2>();
        int8_gemm_kernel<false, false, false, 2><<<grid, block, kSmemSize2, stream>>>(
            A, A_scales, B, B_scales, C, bias, M, N, K, num_k_blocks, num_quant_n_blocks);
    }
    return cudaGetLastError();
}

/**
 * INT8 GEMM with fused GELU. Automatically selects 2 or 3 pipeline stages.
 */
inline cudaError_t int8_gemm_gelu(
    const int8_t* A, const float* A_scales,
    const int8_t* B, const float* B_scales,
    half* C, const half* bias,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    int num_m_tiles = (M + kTileM - 1) / kTileM;
    int num_n_tiles = (N + kTileN - 1) / kTileN;
    int num_k_blocks = (K + kQuantBlock - 1) / kQuantBlock;
    int num_quant_n_blocks = (N + kQuantBlock - 1) / kQuantBlock;
    dim3 grid(num_n_tiles, num_m_tiles);
    dim3 block(kThreads);

    int stages = detail::get_num_stages();
    if (stages >= 3) {
        detail::configure_one_kernel<true, false, false, 3>();
        int8_gemm_kernel<true, false, false, 3><<<grid, block, kSmemSize3, stream>>>(
            A, A_scales, B, B_scales, C, bias, M, N, K, num_k_blocks, num_quant_n_blocks);
    } else {
        detail::configure_one_kernel<true, false, false, 2>();
        int8_gemm_kernel<true, false, false, 2><<<grid, block, kSmemSize2, stream>>>(
            A, A_scales, B, B_scales, C, bias, M, N, K, num_k_blocks, num_quant_n_blocks);
    }
    return cudaGetLastError();
}

/**
 * INT8 GEMM with fused GELU + requantize epilogue.
 * On Blackwell (3-stage supported): single fused kernel.
 * On Ada (2-stage fallback): decomposed into GEMM+GELU then separate quantize.
 */
inline cudaError_t int8_gemm_gelu_requant(
    const int8_t* A, const float* A_scales,
    const int8_t* B, const float* B_scales,
    int8_t* C_int8, float* C_scales,
    const half* bias,
    int M, int N, int K,
    cudaStream_t stream = nullptr,
    half* fp16_temp = nullptr  // Required for 2-stage (unfused) path; caller must provide M*N halves
) {
    int num_m_tiles = (M + kTileM - 1) / kTileM;
    int num_n_tiles = (N + kTileN - 1) / kTileN;
    int num_k_blocks = (K + kQuantBlock - 1) / kQuantBlock;
    int num_quant_n_blocks = (N + kQuantBlock - 1) / kQuantBlock;
    dim3 grid(num_n_tiles, num_m_tiles);
    dim3 block(kThreads);

    int stages = detail::get_num_stages();
    if (stages >= 3) {
        detail::configure_one_kernel<true, true, false, 3>();
        int8_gemm_kernel<true, true, false, 3><<<grid, block, kSmemSize3, stream>>>(
            A, A_scales, B, B_scales, nullptr, bias,
            M, N, K, num_k_blocks, num_quant_n_blocks, C_int8, C_scales);
        return cudaGetLastError();
    } else {
        if (fp16_temp == nullptr) return cudaErrorInvalidValue;
        detail::configure_one_kernel<true, false, false, 2>();
        int8_gemm_kernel<true, false, false, 2><<<grid, block, kSmemSize2, stream>>>(
            A, A_scales, B, B_scales, fp16_temp, bias,
            M, N, K, num_k_blocks, num_quant_n_blocks);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        return quantize_per_block_128(
            reinterpret_cast<const half*>(fp16_temp), C_int8, C_scales,
            M, N, stream, false);
    }
}

/**
 * INT8 GEMM with fused gated residual epilogue.
 * Computes: residual[i,j] = residual[i,j] + (A @ B + bias)[i,j] * gate[i,j]
 * where gate[i,j] = sst[gate_idx*N + j] + temb[i*temb_stride + gate_idx*N + j]
 *
 * Eliminates the separate residual_gate_add kernel and the dFF2_row intermediate
 * buffer write/read (~894 MB DRAM traffic saved).
 *
 * @param A               INT8 activations [M, K] row-major
 * @param A_scales        Per-block scales [M/128, K/128]
 * @param B               INT8 weights pre-swizzled (128x128 tiles)
 * @param B_scales        Per-block scales [K/128, N/128]
 * @param residual        In/Out FP16 hidden states [M, N]
 * @param bias            Optional FP16 bias [N]
 * @param sst             Scale-shift table [6, N] half
 * @param temb            Temporal embedding [M, temb_stride] half
 * @param temb_row_stride Row stride of temb (elements)
 * @param gate_idx        Gate index in sst/temb (e.g., 5 for FFN)
 * @param M, N, K         Problem dimensions
 * @param stream          CUDA stream
 */
/**
 * INT8 GEMM with fused gated residual epilogue.
 * Automatically selects 2 or 3 pipeline stages.
 */
inline cudaError_t int8_gemm_gated_residual(
    const int8_t* A, const float* A_scales,
    const int8_t* B, const float* B_scales,
    half* residual, const half* bias,
    const half* sst, const half* temb,
    int temb_row_stride, int gate_idx,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    int num_m_tiles = (M + kTileM - 1) / kTileM;
    int num_n_tiles = (N + kTileN - 1) / kTileN;
    int num_k_blocks = (K + kQuantBlock - 1) / kQuantBlock;
    int num_quant_n_blocks = (N + kQuantBlock - 1) / kQuantBlock;
    dim3 grid(num_n_tiles, num_m_tiles);
    dim3 block(kThreads);

    const half* gate_sst  = sst  + gate_idx * N;
    const half* gate_temb = temb + gate_idx * N;

    int stages = detail::get_num_stages();
    if (stages >= 3) {
        detail::configure_one_kernel<false, false, true, 3>();
        int8_gemm_kernel<false, false, true, 3><<<grid, block, kSmemSize3, stream>>>(
            A, A_scales, B, B_scales, nullptr, bias,
            M, N, K, num_k_blocks, num_quant_n_blocks,
            nullptr, nullptr, residual, gate_sst, gate_temb, temb_row_stride);
    } else {
        detail::configure_one_kernel<false, false, true, 2>();
        int8_gemm_kernel<false, false, true, 2><<<grid, block, kSmemSize2, stream>>>(
            A, A_scales, B, B_scales, nullptr, bias,
            M, N, K, num_k_blocks, num_quant_n_blocks,
            nullptr, nullptr, residual, gate_sst, gate_temb, temb_row_stride);
    }
    return cudaGetLastError();
}

} // namespace omnidreams_singleview
