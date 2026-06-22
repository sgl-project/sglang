// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cosmos_fp8_flash.cuh"

#include "cutlass/numeric_conversion.h"

#include <cmath>

namespace omnidreams_singleview {
namespace {

// V0 fused FlashAttention contract kernel. It is intentionally simple and
// exists to lock correctness/API behavior before replacing the inner loop with
// SM120 FP8 MMA tiling.
template <int HeadDim>
__global__ void cosmos_fp8_flash_online_warp_kernel(
    const cutlass::float_e4m3_t* __restrict__ q,
    const cutlass::float_e4m3_t* __restrict__ k,
    const cutlass::float_e4m3_t* __restrict__ v,
    cutlass::half_t* __restrict__ o,
    int B,
    int Mq,
    int Mk,
    int H,
    bool causal,
    float softmax_scale) {
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kElemsPerLane = (HeadDim + kWarpSize - 1) / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int row_linear = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int total_rows = B * H * Mq;
    if (row_linear >= total_rows) {
        return;
    }

    const int q_idx = row_linear % Mq;
    const int head_batch = row_linear / Mq;
    const int head = head_batch % H;
    const int batch = head_batch / H;

    cutlass::NumericConverter<float, cutlass::float_e4m3_t> to_float;
    cutlass::NumericConverter<cutlass::half_t, float> to_half;

    const size_t q_base =
        ((static_cast<size_t>(batch) * Mq + q_idx) * H + head) * HeadDim;
    const size_t o_base = q_base;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float q_frag[kElemsPerLane];
    float out_frag[kElemsPerLane];

#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
        const int d = lane + i * kWarpSize;
        q_frag[i] = d < HeadDim ? to_float(q[q_base + d]) : 0.0f;
        out_frag[i] = 0.0f;
    }

    for (int k_idx = 0; k_idx < Mk; ++k_idx) {
        const size_t kv_base =
            ((static_cast<size_t>(batch) * Mk + k_idx) * H + head) * HeadDim;

        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < kElemsPerLane; ++i) {
            const int d = lane + i * kWarpSize;
            if (d < HeadDim) {
                dot += q_frag[i] * to_float(k[kv_base + d]);
            }
        }

        unsigned mask = 0xffffffffu;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(mask, dot, offset);
        }

        float old_scale = 1.0f;
        float new_weight = 0.0f;
        if (lane == 0) {
            const bool valid = !causal || (k_idx <= q_idx);
            if (valid) {
                const float score = dot * softmax_scale;
                const float new_max = fmaxf(row_max, score);
                old_scale = row_sum == 0.0f ? 0.0f : __expf(row_max - new_max);
                new_weight = __expf(score - new_max);
                row_sum = row_sum * old_scale + new_weight;
                row_max = new_max;
            }
        }

        old_scale = __shfl_sync(mask, old_scale, 0);
        new_weight = __shfl_sync(mask, new_weight, 0);

#pragma unroll
        for (int i = 0; i < kElemsPerLane; ++i) {
            const int d = lane + i * kWarpSize;
            if (d < HeadDim) {
                const float v_value = to_float(v[kv_base + d]);
                out_frag[i] = out_frag[i] * old_scale + v_value * new_weight;
            }
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) {
        inv_sum = row_sum > 0.0f ? (1.0f / row_sum) : 0.0f;
    }
    inv_sum = __shfl_sync(0xffffffffu, inv_sum, 0);

#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
        const int d = lane + i * kWarpSize;
        if (d < HeadDim) {
            o[o_base + d] = to_half(out_frag[i] * inv_sum);
        }
    }
}

template <int HeadDim>
cudaError_t launch_cosmos_fp8_flash_online(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const cutlass::float_e4m3_t* v,
    cutlass::half_t* o,
    int B,
    int Mq,
    int Mk,
    int H,
    bool causal,
    cudaStream_t stream) {
    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreads = 32 * kWarpsPerBlock;
    const int total_rows = B * H * Mq;
    const int blocks = (total_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(HeadDim));
    cosmos_fp8_flash_online_warp_kernel<HeadDim>
        <<<blocks, kThreads, 0, stream>>>(q, k, v, o, B, Mq, Mk, H, causal, softmax_scale);
    return cudaGetLastError();
}

}  // namespace

cudaError_t run_cosmos_cute_flash_fp8(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const cutlass::float_e4m3_t* v,
    cutlass::half_t* o,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    cudaStream_t stream) {
    if (!q || !k || !v || !o || B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0) {
        return cudaErrorInvalidValue;
    }

    switch (D) {
    case 32:
        return launch_cosmos_fp8_flash_online<32>(q, k, v, o, B, Mq, Mk, H, causal, stream);
    case 64:
        return launch_cosmos_fp8_flash_online<64>(q, k, v, o, B, Mq, Mk, H, causal, stream);
    case 128:
        return launch_cosmos_fp8_flash_online<128>(q, k, v, o, B, Mq, Mk, H, causal, stream);
    default:
        return cudaErrorInvalidValue;
    }
}

}  // namespace omnidreams_singleview
