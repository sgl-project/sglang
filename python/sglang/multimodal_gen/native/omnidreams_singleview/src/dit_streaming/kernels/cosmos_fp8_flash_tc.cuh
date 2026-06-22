// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cutlass/numeric_types.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace omnidreams_singleview {

// Planned SM120 tensor-core FP8 FlashAttention path for Cosmos BMHK tensors.
// This is intentionally separate from `run_cosmos_cute_flash_fp8`, whose V0
// implementation is correctness-oriented and scalar/warp-online.
cudaError_t run_cosmos_fp8_flash_tc(
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
    cudaStream_t stream);

cudaError_t run_cosmos_fp8_flash_tc_workspace(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const cutlass::float_e4m3_t* v,
    cutlass::float_e4m3_t* q_bhmd,
    cutlass::float_e4m3_t* k_bhmd,
    cutlass::float_e4m3_t* v_bhdm,
    cutlass::bfloat16_t* scores,
    cutlass::bfloat16_t* score_c_scratch,
    cutlass::float_e4m3_t* probs,
    cutlass::bfloat16_t* out_bhmd,
    cutlass::bfloat16_t* out_c_scratch,
    cutlass::half_t* o,
    cutlass::float_e4m3_t* o_fp8,
    float* tc_scale,
    int64_t tc_scale_elems,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    bool tc_scale_is_ones,
    cudaStream_t stream);

cudaError_t run_cosmos_fp8_flash_tc_workspace_prepacked_kv(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k_bhmd,
    const cutlass::float_e4m3_t* v_bhdm,
    cutlass::float_e4m3_t* q_bhmd,
    cutlass::bfloat16_t* scores,
    cutlass::bfloat16_t* score_c_scratch,
    cutlass::float_e4m3_t* probs,
    cutlass::bfloat16_t* out_bhmd,
    cutlass::bfloat16_t* out_c_scratch,
    cutlass::half_t* o,
    cutlass::float_e4m3_t* o_fp8,
    float* tc_scale,
    int64_t tc_scale_elems,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    bool tc_scale_is_ones,
    cudaStream_t stream);

cudaError_t run_cosmos_fp8_flash_tc_workspace_prepacked_qkv(
    const cutlass::float_e4m3_t* q_bhmd,
    const cutlass::float_e4m3_t* k_bhmd,
    const cutlass::float_e4m3_t* v_bhdm,
    cutlass::bfloat16_t* scores,
    cutlass::bfloat16_t* score_c_scratch,
    cutlass::float_e4m3_t* probs,
    cutlass::bfloat16_t* out_bhmd,
    cutlass::bfloat16_t* out_c_scratch,
    cutlass::half_t* o,
    cutlass::float_e4m3_t* o_fp8,
    float* tc_scale,
    int64_t tc_scale_elems,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    bool tc_scale_is_ones,
    cudaStream_t stream);

}  // namespace omnidreams_singleview
