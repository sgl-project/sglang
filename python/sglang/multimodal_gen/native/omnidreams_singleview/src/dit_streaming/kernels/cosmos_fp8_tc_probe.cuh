// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_types.h>

namespace omnidreams_singleview {

cudaError_t run_cosmos_fp8_tc_probe_qk(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const float* q_scale,
    const float* k_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int Mq,
    int Mk,
    int D,
    cudaStream_t stream);

cudaError_t run_cosmos_fp8_tc_probe_qk_batched(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const float* q_scale,
    const float* k_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int Mq,
    int Mk,
    int D,
    int batch_count,
    cudaStream_t stream);

cudaError_t run_cosmos_fp8_tc_probe_pv(
    const cutlass::float_e4m3_t* probs,
    const cutlass::float_e4m3_t* v,
    const float* probs_scale,
    const float* v_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int Mq,
    int Mk,
    int D,
    cudaStream_t stream);

cudaError_t run_cosmos_fp8_tc_probe_pv_batched(
    const cutlass::float_e4m3_t* probs,
    const cutlass::float_e4m3_t* v,
    const float* probs_scale,
    const float* v_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int Mq,
    int Mk,
    int D,
    int batch_count,
    cudaStream_t stream);

}  // namespace omnidreams_singleview
