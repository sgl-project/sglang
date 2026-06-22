// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "attention.cuh"

#include <torch/extension.h>

namespace omnidreams_singleview {

bool sage3_is_built() {
  return false;
}

bool sage3_is_runtime_supported(int /*device*/) {
  return false;
}

std::vector<at::Tensor> sage3_quantize_cross_kv_bf16(
    at::Tensor /*k_bmhd*/,
    at::Tensor /*v_bmhd*/) {
  TORCH_CHECK(false, "SageAttention-3 was not built into OmniDreams single-view native extension");
  return {};
}

cudaError_t run_sage3_fmha_packed_qkv(
    const cutlass::bfloat16_t* /*Q*/,
    const cutlass::bfloat16_t* /*K*/,
    const cutlass::bfloat16_t* /*V*/,
    cutlass::bfloat16_t* /*O*/,
    int /*B*/, int /*Mq*/, int /*Mk*/,
    int /*H*/, int /*D*/,
    bool /*causal*/,
    float /*scale*/,
    cudaStream_t /*stream*/) {
  return cudaErrorNotSupported;
}

cudaError_t run_sage3_fmha_packed_qkv_fp8(
    const cutlass::float_e4m3_t* /*Q*/,
    const cutlass::float_e4m3_t* /*K*/,
    const cutlass::float_e4m3_t* /*V*/,
    cutlass::bfloat16_t* /*O*/,
    int /*B*/, int /*Mq*/, int /*Mk*/,
    int /*H*/, int /*D*/,
    bool /*causal*/,
    float /*scale*/,
    cudaStream_t /*stream*/) {
  return cudaErrorNotSupported;
}

cudaError_t run_sage3_fmha_packed_qfp4_kvfp8(
    const uint8_t* /*Q_fp4*/,
    const cutlass::float_e4m3_t* /*Q_sf*/,
    const cutlass::float_e4m3_t* /*K*/,
    const cutlass::float_e4m3_t* /*V*/,
    cutlass::bfloat16_t* /*O*/,
    int /*B*/, int /*Mq*/, int /*Mk*/,
    int /*H*/, int /*D*/,
    bool /*causal*/,
    float /*scale*/,
    int /*padded_mq*/,
    cudaStream_t /*stream*/) {
  return cudaErrorNotSupported;
}

cudaError_t run_sage3_fmha_packed_qkv_fp4(
    const uint8_t* /*Q_fp4*/,
    const uint8_t* /*K_fp4*/,
    const uint8_t* /*V_fp4*/,
    const cutlass::float_e4m3_t* /*Q_sf*/,
    const cutlass::float_e4m3_t* /*K_sf*/,
    const cutlass::float_e4m3_t* /*V_sf*/,
    cutlass::bfloat16_t* /*O*/,
    int /*B*/, int /*Mq*/, int /*Mk*/,
    int /*H*/, int /*D*/,
    bool /*causal*/,
    float /*scale*/,
    int /*padded_mq*/,
    int /*padded_mk*/,
    cudaStream_t /*stream*/) {
  return cudaErrorNotSupported;
}

cudaError_t sage3_quantize_q_bf16(
    const cutlass::bfloat16_t* /*Q*/,
    const cutlass::bfloat16_t* /*gamma*/,
    const cutlass::bfloat16_t* /*rope_cos*/,
    const cutlass::bfloat16_t* /*rope_sin*/,
    uint8_t* /*Q_fp4*/,
    cutlass::float_e4m3_t* /*Q_sf*/,
    int /*B*/, int /*Mq*/,
    int /*H*/, int /*D*/,
    int /*input_row_stride*/,
    int /*input_head_offset*/,
    bool /*apply_rope*/,
    int /*padded_mq*/,
    cudaStream_t /*stream*/) {
  return cudaErrorNotSupported;
}

}  // namespace omnidreams_singleview
