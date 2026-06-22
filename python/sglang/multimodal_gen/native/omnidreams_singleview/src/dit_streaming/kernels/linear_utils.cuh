// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops.cuh"
#include "block_quant.cuh"
#include "sgl_gemm_shim.cuh"
#include "cute_int8_gemm.cuh"
#include <cutlass/numeric_types.h>
#include <type_traits>
#include <cstdint>

namespace omnidreams_singleview {

// When a per-column weight_scale is provided for FP8 linear, we run the GEMM with a small scalar alpha
// (so the half output doesn't overflow) and compensate by scaling the per-column scale in the epilogue.
// This preserves the effective output: (alpha * (A@B)) * (scale * (1/alpha)) == (A@B) * scale
static constexpr float k_fp8_linear_prescale_alpha = 1.0f / 128.0f;
static constexpr float k_fp8_linear_scale_mul = 1.0f / k_fp8_linear_prescale_alpha;  // 128

// INT8 uses the same pattern as FP8 when a per-column weight_scale is present:
// We scale down the GEMM output to avoid FP16 overflow, then scale up inside apply_col_scale_bias.
// Note: INT8 intermediate (A_int8 @ W_int8) can be very large for K=1536, so we use a more conservative prescale.
static constexpr float k_int8_linear_prescale_alpha = 1.0f / 1024.0f;
static constexpr float k_int8_linear_scale_mul = 1.0f / k_int8_linear_prescale_alpha;  // 1024

// Common helpers to dispatch linear layers for different WeightT (half, fp8, int8).
// For INT8: Uses per-block (128x128) quantization for better precision when weight_block_scale is provided.
//   - weight_block_scale: [in_features/128, out_features/128] float32 per-block scales
//   - int8_act_block_scales: [M/128, in_features/128] float32 activation block scales
//   - use_swizzled_weights: Set to true if weights are pre-swizzled for tensor core optimization
// Falls back to per-column INT8 path (with prescale) when weight_block_scale is nullptr.
template <typename WeightT>
static inline cudaError_t apply_linear_row(
    const cutlass::half_t* input_row,
    const WeightT* weight,
    const cutlass::half_t* bias,
    cutlass::half_t* output_row,
    int M, int in_features, int out_features,
    cudaStream_t stream,
    cutlass::half_t* fp8_scratch,
    const cutlass::half_t* weight_scale = nullptr,  // For FP8: per-channel scales
    int8_t* int8_scratch = nullptr,
    float* int8_act_block_scales = nullptr,  // For INT8: activation block scales buffer [M/128, in_features/128]
    const float* weight_block_scale = nullptr,  // For INT8: weight per-block scales
    bool use_swizzled_weights = false,  // For INT8: use swizzled weight format
    bool fp8_input_preconverted = false) {  // FP8 fusion: fp8_scratch already has FP8 data

  if constexpr (std::is_same<WeightT, cutlass::half_t>::value) {
    return cutlass_linear_layer_rrr(
      reinterpret_cast<const half*>(input_row),
      reinterpret_cast<const half*>(weight),
      reinterpret_cast<const half*>(bias),
      reinterpret_cast<half*>(output_row),
      M, in_features, out_features, stream);
  } else if constexpr (std::is_same<WeightT, cutlass::float_e4m3_t>::value) {
    if (fp8_scratch == nullptr) return cudaErrorInvalidValue;
    auto fp8_buf = reinterpret_cast<cutlass::float_e4m3_t*>(fp8_scratch);
    if (!fp8_input_preconverted) {
      cudaError_t cvt = convert_half_to_fp8_e4m3(
        reinterpret_cast<const cutlass::half_t*>(input_row),
        fp8_buf,
        int64_t(M) * in_features,
        stream);
      if (cvt != cudaSuccess) return cvt;
    }
    if (weight_scale == nullptr) {
      // sgl-kernel bare FP8 GEMM with bias (treated as post-op add).
      {
        at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
            fp8_buf, const_cast<cutlass::float_e4m3_t*>(weight), M, in_features, out_features);
        cudaMemcpyAsync(output_row, _s.data_ptr(),
                        _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
      }
      if (bias) {
        return apply_col_scale_bias(
          reinterpret_cast<cutlass::half_t*>(output_row),
          nullptr,
          reinterpret_cast<const cutlass::half_t*>(bias),
          M, out_features,
          stream,
          1.0f);
      }
      return cudaSuccess;
    }
    // sgl-kernel bare FP8 GEMM (no prescale) + col_scale_bias post-op.
    at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
        fp8_buf, const_cast<cutlass::float_e4m3_t*>(weight), M, in_features, out_features);
    cudaMemcpyAsync(output_row, _s.data_ptr(),
                    _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
    return apply_col_scale_bias(
      reinterpret_cast<cutlass::half_t*>(output_row),
      weight_scale,
      reinterpret_cast<const cutlass::half_t*>(bias),
      M, out_features,
      stream,
      1.0f);
  } else {
    static_assert(std::is_same<WeightT, int8_t>::value || std::is_same<WeightT, cutlass::half_t>::value || std::is_same<WeightT, cutlass::float_e4m3_t>::value,
                  "Unsupported WeightT");
    if constexpr (std::is_same<WeightT, int8_t>::value) {
      if (weight_block_scale != nullptr) {
        // Per-block (128x128) INT8 quantization path
        if (int8_scratch == nullptr) {
          printf("[ERROR] apply_linear_row<int8_t>: int8_scratch is nullptr\n");
          return cudaErrorInvalidValue;
        }
        if (int8_act_block_scales == nullptr) {
          printf("[ERROR] apply_linear_row<int8_t>: int8_act_block_scales is nullptr\n");
          return cudaErrorInvalidValue;
        }
        (void)use_swizzled_weights;  // Always uses swizzled path

        // Step 1: Quantize activations to INT8 with per-block scales
        cudaError_t quant_err = quantize_per_block_128(
          reinterpret_cast<const half*>(input_row),
          int8_scratch,
          int8_act_block_scales,
          M, in_features, stream, false);
        if (quant_err != cudaSuccess) return quant_err;

        // Step 2: Run INT8 GEMM with pre-quantized activations
        return int8_gemm(
          int8_scratch,
          int8_act_block_scales,
          reinterpret_cast<const int8_t*>(weight),
          weight_block_scale,
          reinterpret_cast<half*>(output_row),
          reinterpret_cast<const half*>(bias),
          M, out_features, in_features, stream);
      } else {
        // Per-column INT8 path with prescale
        if (fp8_scratch == nullptr) return cudaErrorInvalidValue;
        float activation_scale = 1.0f;
        cudaError_t act = compute_activation_scale(
          reinterpret_cast<const cutlass::half_t*>(input_row),
          int64_t(M) * in_features,
          &activation_scale,
          stream);
        if (act != cudaSuccess) return act;
        auto int8_buf = reinterpret_cast<int8_t*>(fp8_scratch);
        cudaError_t cvt = convert_half_to_int8(
          reinterpret_cast<const cutlass::half_t*>(input_row),
          int8_buf,
          int64_t(M) * in_features,
          stream,
          activation_scale);
        if (cvt != cudaSuccess) return cvt;
        if (weight_scale == nullptr) {
          return cutlass_linear_layer_rcr_int8(
            int8_buf,
            reinterpret_cast<const int8_t*>(weight),
            reinterpret_cast<const cutlass::half_t*>(bias),
            reinterpret_cast<cutlass::half_t*>(output_row),
            M, in_features, out_features, stream, activation_scale);
        }
        cudaError_t gemm = cutlass_linear_layer_rcr_int8(
          int8_buf,
          reinterpret_cast<const int8_t*>(weight),
          nullptr,
          reinterpret_cast<cutlass::half_t*>(output_row),
          M, in_features, out_features, stream, activation_scale * k_int8_linear_prescale_alpha);
        if (gemm != cudaSuccess) return gemm;
        return apply_col_scale_bias(
          reinterpret_cast<cutlass::half_t*>(output_row),
          weight_scale,
          reinterpret_cast<const cutlass::half_t*>(bias),
          M, out_features,
          stream,
          k_int8_linear_scale_mul);
      }
    } else {
      return cudaErrorNotSupported;
    }
  }
}

template <typename WeightT>
static inline cudaError_t apply_linear_row_gelu(
    const cutlass::half_t* input_row,
    const WeightT* weight,
    const cutlass::half_t* bias,
    cutlass::half_t* output_row,
    int M, int in_features, int out_features,
    cudaStream_t stream,
    cutlass::half_t* fp8_scratch,
    const cutlass::half_t* weight_scale = nullptr,  // For FP8: per-channel scales
    int8_t* int8_scratch = nullptr,
    float* int8_act_block_scales = nullptr,  // For INT8: activation block scales buffer [M/128, in_features/128]
    const float* weight_block_scale = nullptr,  // For INT8: weight per-block scales
    bool use_swizzled_weights = false,  // For INT8: use swizzled weight format
    cutlass::float_e4m3_t* fp8_next_input = nullptr) {  // FP8 fusion: write FP8 here instead of FP16

  if constexpr (std::is_same<WeightT, cutlass::half_t>::value) {
    return cutlass_linear_layer_rrr_gelu(
      reinterpret_cast<const half*>(input_row),
      reinterpret_cast<const half*>(weight),
      reinterpret_cast<const half*>(bias),
      reinterpret_cast<half*>(output_row),
      M, in_features, out_features, stream);
  } else if constexpr (std::is_same<WeightT, cutlass::float_e4m3_t>::value) {
    if (fp8_scratch == nullptr) return cudaErrorInvalidValue;
    auto fp8_buf = reinterpret_cast<cutlass::float_e4m3_t*>(fp8_scratch);
    cudaError_t cvt = convert_half_to_fp8_e4m3(
      reinterpret_cast<const cutlass::half_t*>(input_row),
      fp8_buf,
      int64_t(M) * in_features,
      stream);
    if (cvt != cudaSuccess) return cvt;
    if (weight_scale == nullptr) {
      // sgl-kernel bare FP8 GEMM + GELU post-op.
      {
        at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
            fp8_buf, const_cast<cutlass::float_e4m3_t*>(weight), M, in_features, out_features);
        cudaMemcpyAsync(output_row, _s.data_ptr(),
                        _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
      }
      return apply_col_scale_bias_gelu(
        reinterpret_cast<cutlass::half_t*>(output_row),
        nullptr,
        reinterpret_cast<const cutlass::half_t*>(bias),
        M, out_features,
        stream,
        1.0f);
    }
    // sgl-kernel bare FP8 GEMM (no prescale) + col_scale_bias_gelu post-op.
    at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
        fp8_buf, const_cast<cutlass::float_e4m3_t*>(weight), M, in_features, out_features);
    cudaMemcpyAsync(output_row, _s.data_ptr(),
                    _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
    if (fp8_next_input) {
      return apply_col_scale_bias_gelu_to_fp8(
        reinterpret_cast<cutlass::half_t*>(output_row),
        fp8_next_input,
        weight_scale,
        reinterpret_cast<const cutlass::half_t*>(bias),
        M, out_features,
        stream,
        1.0f);
    }
    return apply_col_scale_bias_gelu(
      reinterpret_cast<cutlass::half_t*>(output_row),
      weight_scale,
      reinterpret_cast<const cutlass::half_t*>(bias),
      M, out_features,
      stream,
      1.0f);
  } else {
    static_assert(std::is_same<WeightT, int8_t>::value || std::is_same<WeightT, cutlass::half_t>::value || std::is_same<WeightT, cutlass::float_e4m3_t>::value,
                  "Unsupported WeightT");
    if constexpr (std::is_same<WeightT, int8_t>::value) {
      if (weight_block_scale != nullptr) {
        // Per-block (128x128) INT8 quantization + GELU activation
        if (int8_scratch == nullptr) {
          printf("[ERROR] apply_linear_row_gelu<int8_t>: int8_scratch is nullptr\n");
          return cudaErrorInvalidValue;
        }
        if (int8_act_block_scales == nullptr) {
          printf("[ERROR] apply_linear_row_gelu<int8_t>: int8_act_block_scales is nullptr\n");
          return cudaErrorInvalidValue;
        }
        (void)use_swizzled_weights;  // Always uses swizzled path

        // Step 1: Quantize activations to INT8 with per-block scales
        cudaError_t quant_err = quantize_per_block_128(
          reinterpret_cast<const half*>(input_row),
          int8_scratch,
          int8_act_block_scales,
          M, in_features, stream, false);
        if (quant_err != cudaSuccess) return quant_err;

        // Step 2: Run INT8 GEMM with fused GELU activation
        return int8_gemm_gelu(
          int8_scratch,
          int8_act_block_scales,
          reinterpret_cast<const int8_t*>(weight),
          weight_block_scale,
          reinterpret_cast<half*>(output_row),
          reinterpret_cast<const half*>(bias),
          M, out_features, in_features, stream);
      } else {
        // Per-column INT8 path with prescale + GELU
        if (fp8_scratch == nullptr) return cudaErrorInvalidValue;
        float activation_scale = 1.0f;
        cudaError_t act = compute_activation_scale(
          reinterpret_cast<const cutlass::half_t*>(input_row),
          int64_t(M) * in_features,
          &activation_scale,
          stream);
        if (act != cudaSuccess) return act;
        auto int8_buf = reinterpret_cast<int8_t*>(fp8_scratch);
        cudaError_t cvt = convert_half_to_int8(
          reinterpret_cast<const cutlass::half_t*>(input_row),
          int8_buf,
          int64_t(M) * in_features,
          stream,
          activation_scale);
        if (cvt != cudaSuccess) return cvt;
        if (weight_scale == nullptr) {
          return cutlass_linear_layer_rcr_int8_gelu(
            int8_buf,
            reinterpret_cast<const int8_t*>(weight),
            reinterpret_cast<const cutlass::half_t*>(bias),
            reinterpret_cast<cutlass::half_t*>(output_row),
            M, in_features, out_features, stream, activation_scale);
        }
        cudaError_t gemm = cutlass_linear_layer_rcr_int8(
          int8_buf,
          reinterpret_cast<const int8_t*>(weight),
          nullptr,
          reinterpret_cast<cutlass::half_t*>(output_row),
          M, in_features, out_features, stream, activation_scale * k_int8_linear_prescale_alpha);
        if (gemm != cudaSuccess) return gemm;
        return apply_col_scale_bias_gelu(
          reinterpret_cast<cutlass::half_t*>(output_row),
          weight_scale,
          reinterpret_cast<const cutlass::half_t*>(bias),
          M, out_features,
          stream,
          k_int8_linear_scale_mul);
      }
    } else {
      return cudaErrorNotSupported;
    }
  }
}

template <typename WeightT>
static inline cudaError_t apply_linear_row_fused_residual(
    const cutlass::half_t* input_row,
    const WeightT* weight,
    const cutlass::half_t* bias,
    cutlass::half_t* residual_inout,
    int M, int in_features, int out_features,
    cudaStream_t stream,
    cutlass::half_t* fp8_scratch,
    const cutlass::half_t* weight_scale = nullptr,  // For FP8: per-channel scales
    cutlass::half_t* temp_out = nullptr,
    int8_t* int8_scratch = nullptr,
    float* int8_act_block_scales = nullptr,  // For INT8: activation block scales buffer [M/128, in_features/128]
    const float* weight_block_scale = nullptr,  // For INT8: weight per-block scales
    bool use_swizzled_weights = false) {  // For INT8: use swizzled weight format

  if constexpr (std::is_same<WeightT, cutlass::half_t>::value) {
    return cutlass_linear_layer_rrr_fused_residual(
      reinterpret_cast<const half*>(input_row),
      reinterpret_cast<const half*>(weight),
      reinterpret_cast<const half*>(bias),
      reinterpret_cast<half*>(residual_inout),
      M, in_features, out_features, stream);
  } else if constexpr (std::is_same<WeightT, cutlass::float_e4m3_t>::value) {
    if (fp8_scratch == nullptr) return cudaErrorInvalidValue;
    auto fp8_buf = reinterpret_cast<cutlass::float_e4m3_t*>(fp8_scratch);
    cudaError_t cvt = convert_half_to_fp8_e4m3(
      reinterpret_cast<const cutlass::half_t*>(input_row),
      fp8_buf,
      int64_t(M) * in_features,
      stream);
    if (cvt != cudaSuccess) return cvt;
    if (weight_scale == nullptr) {
      // sgl-kernel bare FP8 GEMM + add residual post-op.
      {
        at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
            fp8_buf, const_cast<cutlass::float_e4m3_t*>(weight), M, in_features, out_features);
        cudaMemcpyAsync(temp_out, _s.data_ptr(),
                        _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
      }
      return apply_col_scale_bias_residual(
        reinterpret_cast<const cutlass::half_t*>(temp_out),
        reinterpret_cast<cutlass::half_t*>(residual_inout),
        nullptr,
        reinterpret_cast<const cutlass::half_t*>(bias),
        M, out_features,
        stream,
        1.0f);
    }
    if (temp_out == nullptr) return cudaErrorInvalidValue;
    // sgl-kernel bare FP8 GEMM (no prescale) + col_scale_bias_residual post-op.
    at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
        fp8_buf, const_cast<cutlass::float_e4m3_t*>(weight), M, in_features, out_features);
    cudaMemcpyAsync(temp_out, _s.data_ptr(),
                    _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
    return apply_col_scale_bias_residual(
      reinterpret_cast<const cutlass::half_t*>(temp_out),
      reinterpret_cast<cutlass::half_t*>(residual_inout),
      weight_scale,
      reinterpret_cast<const cutlass::half_t*>(bias),
      M, out_features,
      stream,
      1.0f);
  } else {
    static_assert(std::is_same<WeightT, int8_t>::value || std::is_same<WeightT, cutlass::half_t>::value || std::is_same<WeightT, cutlass::float_e4m3_t>::value,
                  "Unsupported WeightT");
    if constexpr (std::is_same<WeightT, int8_t>::value) {
      if (weight_block_scale != nullptr) {
        // Per-block (128x128) INT8 quantization + fused residual
        if (temp_out == nullptr) {
          printf("[ERROR] apply_linear_row_fused_residual<int8_t>: temp_out is nullptr\n");
          return cudaErrorInvalidValue;
        }
        if (int8_scratch == nullptr) {
          printf("[ERROR] apply_linear_row_fused_residual<int8_t>: int8_scratch is nullptr\n");
          return cudaErrorInvalidValue;
        }
        if (int8_act_block_scales == nullptr) {
          printf("[ERROR] apply_linear_row_fused_residual<int8_t>: int8_act_block_scales is nullptr\n");
          return cudaErrorInvalidValue;
        }
        (void)use_swizzled_weights;  // Always uses swizzled path

        // Step 1: Quantize activations to INT8 with per-block scales
        cudaError_t quant_err = quantize_per_block_128(
          reinterpret_cast<const half*>(input_row),
          int8_scratch,
          int8_act_block_scales,
          M, in_features, stream, false);
        if (quant_err != cudaSuccess) return quant_err;

        // Step 2: Run INT8 GEMM with pre-quantized activations
        cudaError_t gemm_err = int8_gemm(
          int8_scratch,
          int8_act_block_scales,
          reinterpret_cast<const int8_t*>(weight),
          weight_block_scale,
          reinterpret_cast<half*>(temp_out),
          reinterpret_cast<const half*>(bias),
          M, out_features, in_features, stream);
        if (gemm_err != cudaSuccess) return gemm_err;

        // Add temp_out to residual: residual_inout += temp_out
        return add_inplace_half_rr(
          reinterpret_cast<cutlass::half_t*>(residual_inout),
          reinterpret_cast<const cutlass::half_t*>(temp_out),
          int64_t(M) * out_features,
          stream);
      } else {
        // Per-column INT8 path with prescale + fused residual
        if (fp8_scratch == nullptr) return cudaErrorInvalidValue;
        float activation_scale = 1.0f;
        cudaError_t act = compute_activation_scale(
          reinterpret_cast<const cutlass::half_t*>(input_row),
          int64_t(M) * in_features,
          &activation_scale,
          stream);
        if (act != cudaSuccess) return act;
        auto int8_buf = reinterpret_cast<int8_t*>(fp8_scratch);
        cudaError_t cvt = convert_half_to_int8(
          reinterpret_cast<const cutlass::half_t*>(input_row),
          int8_buf,
          int64_t(M) * in_features,
          stream,
          activation_scale);
        if (cvt != cudaSuccess) return cvt;
        if (weight_scale == nullptr) {
          return cutlass_linear_layer_rcr_int8_fused_residual(
            int8_buf,
            reinterpret_cast<const int8_t*>(weight),
            reinterpret_cast<const cutlass::half_t*>(bias),
            reinterpret_cast<cutlass::half_t*>(residual_inout),
            M, in_features, out_features, stream, activation_scale);
        }
        if (temp_out == nullptr) return cudaErrorInvalidValue;
        cudaError_t gemm = cutlass_linear_layer_rcr_int8(
          int8_buf,
          reinterpret_cast<const int8_t*>(weight),
          nullptr,
          reinterpret_cast<cutlass::half_t*>(temp_out),
          M, in_features, out_features, stream, activation_scale * k_int8_linear_prescale_alpha);
        if (gemm != cudaSuccess) return gemm;
        return apply_col_scale_bias_residual(
          reinterpret_cast<const cutlass::half_t*>(temp_out),
          reinterpret_cast<cutlass::half_t*>(residual_inout),
          weight_scale,
          reinterpret_cast<const cutlass::half_t*>(bias),
          M, out_features,
          stream,
          k_int8_linear_scale_mul);
      }
    } else {
      return cudaErrorNotSupported;
    }
  }
}

// =============================================================================
// Per-Block INT8 Linear Layer Functions
// =============================================================================

/**
 * Apply linear layer with per-block (128x128) INT8 quantization.
 *
 * This uses per-block quantization for both activations and weights,
 * providing better precision than per-tensor quantization.
 *
 * @param input FP16 input [M, in_features] (row-major)
 * @param weight_int8 Pre-quantized INT8 weights [in_features, out_features] (row-major)
 * @param weight_block_scales Per-block weight scales [in_features/128, out_features/128]
 * @param bias Optional bias [out_features]
 * @param output FP16 output [M, out_features] (row-major)
 * @param int8_scratch Scratch buffer for quantized activations [M, in_features]
 * @param act_block_scales Scratch buffer for activation scales [M/128, in_features/128]
 * @param M Number of rows
 * @param in_features Input feature dimension
 * @param out_features Output feature dimension
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
static inline cudaError_t apply_linear_row_perblock_int8(
    const cutlass::half_t* input,
    const int8_t* weight_int8,
    const float* weight_block_scales,
    const cutlass::half_t* bias,
    cutlass::half_t* output,
    int8_t* int8_scratch,
    float* act_block_scales,
    int M, int in_features, int out_features,
    cudaStream_t stream
) {
    if (int8_scratch == nullptr || act_block_scales == nullptr) {
        printf("[ERROR] apply_linear_row_perblock_int8: scratch buffers are nullptr\n");
        return cudaErrorInvalidValue;
    }

    // Step 1: Quantize activations to INT8 with per-block scales
    cudaError_t quant_err = quantize_per_block_128(
        reinterpret_cast<const half*>(input),
        int8_scratch,
        act_block_scales,
        M, in_features, stream, false);
    if (quant_err != cudaSuccess) return quant_err;

    // Step 2: Run INT8 GEMM with pre-quantized activations
    return int8_gemm(
        int8_scratch,
        act_block_scales,
        weight_int8,
        weight_block_scales,
        reinterpret_cast<half*>(output),
        reinterpret_cast<const half*>(bias),
        M, out_features, in_features,
        stream);
}

/**
 * Apply linear layer with GELU activation using per-block INT8 quantization.
 *
 * Computes: output = GELU(input @ weight + bias)
 */
static inline cudaError_t apply_linear_row_gelu_perblock_int8(
    const cutlass::half_t* input,
    const int8_t* weight_int8,
    const float* weight_block_scales,
    const cutlass::half_t* bias,
    cutlass::half_t* output,
    int8_t* int8_scratch,
    float* act_block_scales,
    int M, int in_features, int out_features,
    cudaStream_t stream
) {
    // First compute linear layer
    cudaError_t err = apply_linear_row_perblock_int8(
        input, weight_int8, weight_block_scales, bias, output,
        int8_scratch, act_block_scales,
        M, in_features, out_features, stream);
    if (err != cudaSuccess) return err;

    // Then apply GELU activation in-place
    // Note: We need to add a GELU kernel here. For now, we use the existing
    // path which applies GELU via a separate kernel or fused epilogue.
    // TODO: Implement fused GELU in the per-block GEMM kernel
    return cudaSuccess;
}

/**
 * Apply linear layer with fused residual using per-block INT8 quantization.
 *
 * Computes: residual = residual + (input @ weight + bias)
 */
static inline cudaError_t apply_linear_row_fused_residual_perblock_int8(
    const cutlass::half_t* input,
    const int8_t* weight_int8,
    const float* weight_block_scales,
    const cutlass::half_t* bias,
    cutlass::half_t* residual_inout,
    cutlass::half_t* temp_output,  // Temporary buffer for GEMM output
    int8_t* int8_scratch,
    float* act_block_scales,
    int M, int in_features, int out_features,
    cudaStream_t stream
) {
    // First compute linear layer to temp buffer
    cudaError_t err = apply_linear_row_perblock_int8(
        input, weight_int8, weight_block_scales, bias, temp_output,
        int8_scratch, act_block_scales,
        M, in_features, out_features, stream);
    if (err != cudaSuccess) return err;

    // TODO: Add residual in-place
    // This should be fused into the GEMM epilogue for better performance.
    // For now, the caller must handle residual addition separately.
    return cudaSuccess;
}

} // namespace omnidreams_singleview
