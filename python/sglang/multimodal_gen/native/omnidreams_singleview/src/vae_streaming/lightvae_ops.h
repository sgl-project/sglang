// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>
#include <tuple>

namespace omnidreams_singleview {

torch::Tensor lightvae_causal_conv3d_bcthw(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int pad_t_left,
    int pad_h,
    int pad_w,
    int stride_t,
    int stride_h,
    int stride_w);

torch::Tensor cutlass_conv3d_nthwc(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int pad_d,
    int pad_h,
    int pad_w,
    int stride_d,
    int stride_h,
    int stride_w);

torch::Tensor lightvae_fp8_quantize_bcthw_to_tin16(
    torch::Tensor input,
    torch::Tensor scale,
    int padded_height,
    int padded_width,
    int padded_channels);

torch::Tensor lightvae_fp8_dequantize_tin16_to_bcthw(
    torch::Tensor input,
    torch::Tensor scale,
    int real_channels,
    int real_height,
    int real_width);

torch::Tensor lightvae_fp8_rmsnorm_tin16(
    torch::Tensor input,
    torch::Tensor input_scale,
    torch::Tensor gamma,
    torch::Tensor output_scale,
    int real_channels,
    bool silu);

torch::Tensor lightvae_fp8_add_tin16(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor output_scale,
    int real_channels);

torch::Tensor lightvae_fp8_add_relu_tin16(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor output_scale,
    int real_channels);

torch::Tensor lightvae_fp8_add_relu_nhwc_tin16_to_tin16(
    torch::Tensor a_nhwc,
    torch::Tensor b_tin16,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor output_scale,
    int real_channels);

torch::Tensor lightvae_fp8_upsample2x_tin16(torch::Tensor input);

torch::Tensor lightvae_fp8_extract_mu_normalize_tin16(
    torch::Tensor input,
    torch::Tensor input_scale,
    torch::Tensor mean,
    torch::Tensor inv_std,
    int real_height,
    int real_width);

torch::Tensor cutlass_conv3x3_nhwc_fp8(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int pad,
    bool relu);

torch::Tensor lightvae_fp8_prepare_conv2d_weight_krsc(
    torch::Tensor weight_krsc,
    torch::Tensor input_scale,
    torch::Tensor output_scale);

torch::Tensor cutlass_conv2d_nhwc_fp8_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu);

torch::Tensor lightvae_fp8_tin16_to_nhwc(torch::Tensor input);

torch::Tensor lightvae_fp8_nhwc_to_tin16(torch::Tensor input);

torch::Tensor lightvae_fp8_nhwc_rescale_to_tin16(
    torch::Tensor input,
    torch::Tensor source_scale,
    torch::Tensor output_scale);

torch::Tensor lightvae_fp8_pack_causal3_tin16(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache);

torch::Tensor lightvae_fp8_pack_causal3_tin16_to_nhwc(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache);

torch::Tensor lightvae_fp8_update_tail_cache_tin16(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    int cache_frames);

torch::Tensor lightvae_fp8_spatial_pad_right_bottom_tin16(torch::Tensor input);

torch::Tensor lightvae_fp8_pack_temporal3_tin16(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache);

torch::Tensor lightvae_fp8_pack_temporal3_tin16_to_nhwc(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache);

torch::Tensor lightvae_fp8_causal_conv3_tin16_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu);

torch::Tensor lightvae_fp8_spatial_conv3_tin16_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom);

torch::Tensor lightvae_fp8_conv1_tin16_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu);

torch::Tensor lightvae_fp8_temporal_conv1_tin16_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lightvae_fp8_qkv_tin16_to_bmhd(
    torch::Tensor qkv);

torch::Tensor lightvae_fp8_bmhd_to_tin16(
    torch::Tensor input,
    int frames,
    int height,
    int width);

torch::Tensor lightvae_fp8_sdpa_bmhd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor qkv_scale,
    torch::Tensor sdpa_inverse_scale,
    torch::Tensor unit_scale,
    int batch,
    int seq,
    double attn_scale);

void lightvae_fp8_sdpa_cleanup();

torch::Tensor lightvae_fp8_causal_conv3_tin16_direct_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu);

torch::Tensor lightvae_fp8_spatial_conv3_tin16_direct_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom);

torch::Tensor lightvae_fp8_conv1_tin16_direct_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu);

torch::Tensor lightvae_fp8_temporal_conv1_tin16_direct_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu);

torch::Tensor lightvae_fp8_conv1_tin16_warp_mma_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu);

torch::Tensor lightvae_fp8_spatial_conv3_tin16_warp_mma_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom);

torch::Tensor lightvae_fp8_causal_conv3_tin16_warp_mma_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu);

torch::Tensor lightvae_fp8_spatial_conv3_tin16_warp_mma_scaled_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor epilogue_scale,
    c10::optional<torch::Tensor> bias_scaled,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom);

torch::Tensor lightvae_fp8_causal_conv3_tin16_warp_mma_scaled_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor epilogue_scale,
    c10::optional<torch::Tensor> bias_scaled,
    bool relu);

void bind_lightvae_ops(pybind11::module_& m);

}  // namespace omnidreams_singleview
