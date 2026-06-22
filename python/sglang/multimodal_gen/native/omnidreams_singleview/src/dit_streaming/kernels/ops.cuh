// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "workspace.cuh"
#include "arch_traits.h"

namespace omnidreams_singleview {

// ============================================================================
// GEMM TILE CONFIGURATION
// Controls the M-dimension threshold for switching between tile sizes.
// For M < threshold: uses 64x64x64 tiles (more threadblocks, better for small M)
// For M >= threshold: uses 128x64x64 tiles (better arithmetic intensity)
// ============================================================================

// Set the GEMM tile threshold (default: 1024). Lower values use larger tiles
// for smaller M, which may improve performance for batched workloads.
void set_gemm_tile_threshold(int threshold);

// Get the current GEMM tile threshold
int get_gemm_tile_threshold();

struct CosmosFp8LinearTileSelection {
    std::string op_kind;
    std::string preset;
    std::string tile;
    std::string stage;
    std::string variant;
    std::string reason;
    bool tile_env_override = false;
    bool stage_env_override = false;
    bool variant_env_override = false;
};

// Select the SM120 FP8 RCR colscale kernel variant for a Cosmos linear shape.
// Supported op_kind values are "colscale_bf16", "residual_bf16", and
// "gelu_fp8". Environment overrides are applied at call time so autotune tools
// can sweep variants without rebuilding.
CosmosFp8LinearTileSelection select_cosmos_fp8_linear_tile(
    const std::string& op_kind,
    int rows,
    int in_features,
    int out_features);

// ============================================================================
// 3D ROTARY POSITION EMBEDDINGS (RoPE)
// REFERENCE: HuggingFace transformer_wan.py - class WanRotaryPosEmbed
// ============================================================================

/**
 * Generate 3D RoPE frequencies for temporal + spatial dimensions
 *
 * @param cos_out Output cosine table [seq_len, head_dim]
 * @param sin_out Output sine table [seq_len, head_dim]
 * @param post_T Number of patches in temporal dimension
 * @param post_H Number of patches in height dimension
 * @param post_W Number of patches in width dimension
 * @param head_dim Head dimension (must be divisible by 3 for T, H, W)
 * @param theta RoPE base frequency (typically 10000)
 * @param rope_max_seq_len Maximum sequence length for RoPE
 */
cudaError_t generate_rope_3d(
    float* cos_out,
    float* sin_out,
    int post_T, int post_H, int post_W,
    int head_dim,
    float theta,
    int rope_max_seq_len,
    float* cos_t,
    float* sin_t,
    float* cos_h,
    float* sin_h,
    float* cos_w,
    float* sin_w,
    cudaStream_t stream
);

// ============================================================================
// UNIFIED LINEAR LAYER
// PyTorch-compatible interface for all linear operations
// ============================================================================

// Row-major GEMM: A(row) x B(row) -> C(row), optional bias via stride trick
cudaError_t cutlass_linear_layer_rrr(
    const half* input_row,
    const half* weight_row,
    const half* bias,
    half* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream
);

// Row-major GEMM with explicit output leading dimension (row-major stride).
// Useful when writing into a submatrix of a larger row-major buffer.
cudaError_t cutlass_linear_layer_rrr_strided(
    const half* input_row,         // [N, in_features]
    const half* weight_row,        // [in_features, out_features]
    const half* bias,              // [out_features] or nullptr
    half* output_row,              // points to row-major buffer
    int N, int in_features, int out_features,
    int ld_out,                    // row stride (must be >= out_features)
    cudaStream_t stream
);

// Row-major GEMM with fused residual addition: residual = (A @ B + bias) + residual
// Computes output in-place into residual buffer, eliminating intermediate storage
cudaError_t cutlass_linear_layer_rrr_fused_residual(
    const half* input_row,
    const half* weight_row,
    const half* bias,
    half* residual_inout,  // Input: existing residual, Output: updated residual
    int N, int in_features, int out_features,
    cudaStream_t stream
);

// Row-major GEMM with fused SiLU activation: D = SiLU(A @ B + bias)
// Fuses linear layer and SiLU activation into GEMM epilogue, eliminating intermediate storage
cudaError_t cutlass_linear_layer_rrr_silu(
    const half* input_row,
    const half* weight_row,
    const half* bias,
    half* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream
);

// Row-major variant: A(row) x B(row) -> C(row), optional bias via stride trick
cudaError_t cutlass_linear_layer_rrr(
    const half* input_row,
    const half* weight_row,
    const half* bias,
    half* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream
);

// FP8 A/B (RRR) linear: A(row) x B(row) -> C(row), optional bias via stride trick
cudaError_t cutlass_linear_layer_rrr_fp8(
    const cutlass::float_e4m3_t* input_row,   // Row-major [N, in_features] FP8
    const cutlass::float_e4m3_t* weight_fp8,  // Row-major [in_features, out_features] FP8
    const cutlass::half_t* bias,              // [out_features] half or nullptr
    cutlass::half_t* output_row,              // Row-major [N, out_features] half
    int N, int in_features, int out_features,
    cudaStream_t stream
);

// Row-major GEMM (INT8 inputs/weights) with optional bias via stride trick
cudaError_t cutlass_linear_layer_rcr_int8(
    const int8_t* input_row,
    const int8_t* weight_int8,
    const cutlass::half_t* bias,
    cutlass::half_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream,
    float activation_scale = 1.0f
);

// Row-major GEMM with GELU epilogue: D = GELU(A @ B + bias)
cudaError_t cutlass_linear_layer_rrr_gelu(
    const half* input_row,         // Row-major [N, in_features]
    const half* weight_row,        // Row-major [in_features, out_features]
    const half* bias,              // [out_features] or nullptr
    half* output_row,              // Row-major [N, out_features]
    int N, int in_features, int out_features,
    cudaStream_t stream
);

// FP8 A/B (RRR) with GELU epilogue: D = GELU(A @ B + bias)
cudaError_t cutlass_linear_layer_rrr_fp8_gelu(
    const cutlass::float_e4m3_t* input_row,   // Row-major [N, in_features] FP8
    const cutlass::float_e4m3_t* weight_fp8,  // Row-major [in_features, out_features] FP8
    const cutlass::half_t* bias,              // [out_features] half or nullptr
    cutlass::half_t* output_row,              // Row-major [N, out_features] half
    int N, int in_features, int out_features,
    cudaStream_t stream
);

// INT8 A/B (RCR) with GELU epilogue: D = GELU(A @ B + bias)
cudaError_t cutlass_linear_layer_rcr_int8_gelu(
    const int8_t* input_row,            // Row-major [N, in_features] INT8
    const int8_t* weight_int8,          // Column-major [in_features, out_features] INT8
    const cutlass::half_t* bias,        // [out_features] or nullptr
    cutlass::half_t* output_row,        // Row-major [N, out_features] half
    int N, int in_features, int out_features,
    cudaStream_t stream,
    float activation_scale = 1.0f
);

// Utilities: type conversions
cudaError_t convert_half_to_fp8_e4m3(
    const cutlass::half_t* src,
    cutlass::float_e4m3_t* dst,
    int64_t num_elements,
    cudaStream_t stream
);
cudaError_t convert_half_to_int8(
    const cutlass::half_t* src,
    int8_t* dst,
    int64_t num_elements,
    cudaStream_t stream,
    float activation_scale = 1.0f
);
// Compute activation scale = max(|src|)/127 with floor clamp
cudaError_t compute_activation_scale(
    const cutlass::half_t* src,
    int64_t num_elements,
    float* activation_scale_out,
    cudaStream_t stream);
// Column-wise scale + optional bias on row-major [N, out_features]
cudaError_t apply_col_scale_bias(
    cutlass::half_t* data,
    const cutlass::half_t* scale,   // [out_features]
    const cutlass::half_t* bias,    // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul = 1.0f
);
// Column-wise scale + optional bias + GELU on row-major [N, out_features]
cudaError_t apply_col_scale_bias_gelu(
    cutlass::half_t* data,
    const cutlass::half_t* scale,   // [out_features]
    const cutlass::half_t* bias,    // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul = 1.0f
);
// Fused: col_scale + bias + FP8 conversion
cudaError_t apply_col_scale_bias_to_fp8(
    const cutlass::half_t* data,       // GEMM output [N, out_features]
    cutlass::float_e4m3_t* fp8_out,    // FP8 output [N, out_features]
    const cutlass::half_t* scale,      // [out_features]
    const cutlass::half_t* bias,       // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul = 1.0f,
    float output_scale = 1.0f
);
// Fused: col_scale + bias + GELU + FP8 conversion (eliminates intermediate FP16 write/read)
cudaError_t apply_col_scale_bias_gelu_to_fp8(
    const cutlass::half_t* data,       // GEMM output [N, out_features]
    cutlass::float_e4m3_t* fp8_out,    // FP8 output [N, out_features]
    const cutlass::half_t* scale,      // [out_features]
    const cutlass::half_t* bias,       // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul = 1.0f,
    float output_scale = 1.0f
);
// Column-wise scale + optional bias + residual add: out = residual + scale*data + bias
cudaError_t apply_col_scale_bias_residual(
    const cutlass::half_t* data,          // GEMM output [N, out_features]
    cutlass::half_t* residual_inout,      // in/out residual buffer [N, out_features]
    const cutlass::half_t* scale,         // [out_features]
    const cutlass::half_t* bias,          // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul = 1.0f
);
// INT8 A/B (RCR) with fused per-channel scale + optional bias, where activation scale is provided
// as a device scalar holding activation absmax (amax). The epilogue applies act = max(amax/127, 1e-6).
// Computes: D = (act * weight_scale[col]) * (A @ B) + bias[col]   (bias optional)
cudaError_t cutlass_linear_layer_rcr_int8_scale_bias_dev_amax(
    const int8_t* input_row,             // Row-major [N, in_features] INT8
    const int8_t* weight_int8,           // Column-major [in_features, out_features] INT8
    const cutlass::half_t* weight_scale, // [out_features] half (required)
    const cutlass::half_t* bias,         // [out_features] half or nullptr
    cutlass::half_t* output_row,         // Row-major [N, out_features] half
    int N, int in_features, int out_features,
    cudaStream_t stream,
    const float* activation_amax_dev     // device scalar: amax
);

// Row-major GEMM (INT8 inputs/weights) with fused residual addition:
// Computes: residual = (A @ B + bias) + residual
cudaError_t cutlass_linear_layer_rcr_int8_fused_residual(
    const int8_t* input_row,              // Row-major [N, in_features] INT8
    const int8_t* weight_int8,            // Column-major [in_features, out_features] INT8
    const cutlass::half_t* bias,          // [out_features] (half) or nullptr
    cutlass::half_t* residual_inout,      // Row-major [N, out_features] (half), in-place
    int N, int in_features, int out_features,
    cudaStream_t stream,
    float activation_scale = 1.0f
);

// ============================================================================
// INT8 QUANTIZATION UTILITIES
// ============================================================================

// Device-only variant: writes activation absmax to activation_scale_out_dev on device, no host sync.
cudaError_t compute_activation_scale_device(
    const cutlass::half_t* src,
    int64_t num_elements,
    float* activation_scale_out_dev,
    cudaStream_t stream);

cudaError_t convert_half_to_int8_device_scale(
    const cutlass::half_t* src,
    int8_t* dst,
    int64_t num_elements,
    cudaStream_t stream,
    const float* activation_scale_dev
);
cudaError_t convert_int8_to_half_device_scale(
    const int8_t* src,
    cutlass::half_t* dst,
    int64_t num_elements,
    cudaStream_t stream,
    const float* activation_scale_dev
);

// ============================================================================
// INT8 POST-OPS (scalar-aware versions for activation scale)
// ============================================================================

// Scalar-aware post-ops for INT8 projection path:
// - act_scale_dev is a device scalar holding activation absmax (amax). Kernels internally apply amax/127 with clamp.
// - weight_scale is optional per-output-channel scale (may be nullptr).
// - bias is optional (may be nullptr).
cudaError_t apply_col_scale_bias_scalar(
    cutlass::half_t* data,
    const cutlass::half_t* weight_scale,   // [out_features] or nullptr
    const cutlass::half_t* bias,           // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream,
    const float* act_scale_dev
);
cudaError_t apply_col_scale_bias_gelu_scalar(
    cutlass::half_t* data,
    const cutlass::half_t* weight_scale,   // [out_features] or nullptr
    const cutlass::half_t* bias,           // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream,
    const float* act_scale_dev
);
cudaError_t apply_col_scale_bias_residual_scalar(
    const cutlass::half_t* data,
    cutlass::half_t* residual_inout,
    const cutlass::half_t* weight_scale,   // [out_features] or nullptr
    const cutlass::half_t* bias,           // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream,
    const float* act_scale_dev
);

// Apply GELU activation and add bias (no scaling) on row-major [N, out_features]
// Used for per-block INT8 quantization where scaling is already handled
cudaError_t apply_gelu_bias(
    cutlass::half_t* data,          // [N, out_features] in/out
    const cutlass::half_t* bias,    // [out_features] or nullptr
    int N, int out_features,
    cudaStream_t stream
);

// Simple in-place add: dest += src (element-wise)
cudaError_t add_inplace_half_rr(
    cutlass::half_t* dest,          // [numel] destination (in-place)
    const cutlass::half_t* src,     // [numel] source to add
    long long numel,
    cudaStream_t stream
);

// ============================================================================
// TIMESTEP EMBEDDINGS
// REFERENCE: HuggingFace embeddings.py - class Timesteps, TimestepEmbedding
// ============================================================================

/**
 * Generate sinusoidal timestep embeddings
 *
 * @param timesteps Input timesteps [N] or [N*seq_len]
 * @param embeddings Output sinusoidal embeddings [N, freq_dim] or [N*seq_len, freq_dim]
 * @param num_timesteps Number of timesteps
 * @param freq_dim Frequency dimension (typically 256)
 * @param max_period Maximum period for sinusoidal encoding (typically 10000)
 */
cudaError_t timestep_sinusoidal_embedding(
    const int64_t* timesteps,
    half* embeddings,
    int num_timesteps,
    int freq_dim,
    int max_period,
    cudaStream_t stream
);

/**
 * Timestep MLP projection: Linear -> SiLU -> Linear
 * Projects sinusoidal embeddings to model dimension
 *
 * @param input Sinusoidal embeddings [N, freq_dim] row-major
 * @param output Projected embeddings [N, proj_dim] row-major
 * @param w1 First linear weight [freq_dim, hidden_dim] ROW-MAJOR
 * @param b1 First linear bias [hidden_dim]
 * @param w2 Second linear weight [hidden_dim, proj_dim] ROW-MAJOR
 * @param b2 Second linear bias [proj_dim]
 * @param hidden_buffer Workspace buffer [N, hidden_dim] for intermediate result
 *
 * NOTE: Weights must be pre-transposed to column-major by caller
 */
cudaError_t timestep_mlp_projection(
    const half* input,
    half* output,
    const half* w1, const half* b1,
    const half* w2, const half* b2,
    half* hidden_buffer,
    int N, int freq_dim, int hidden_dim, int proj_dim,
    cudaStream_t stream
);

// ============================================================================
// TEXT PROJECTION
// REFERENCE: HuggingFace embeddings.py - class PixArtAlphaTextProjection
// ============================================================================

/**
 * Project text embeddings to model dimension
 * Optional: Linear projection + LayerNorm
 *
 * @param input Text embeddings [N, seq_len, text_dim]
 * @param output Projected text [N, seq_len, model_dim]
 * @param weight Linear weight [model_dim, text_dim]
 * @param bias Linear bias [model_dim] (optional)
 * @param ln_weight LayerNorm weight [model_dim] (optional)
 * @param ln_bias LayerNorm bias [model_dim] (optional)
 */
cudaError_t text_projection(
    const half* input,
    half* output,
    const half* weight,
    const half* bias,
    const half* ln_weight,
    const half* ln_bias,
    int N, int seq_len, int text_dim, int model_dim,
    float eps,
    cudaStream_t stream
);

// ============================================================================
// I2V: Image embedder (WanImageEmbedding)
// norm1(img_dim) -> MLP(img_dim -> img_dim -> K) -> norm2(K) -> optional pos_embed
// ============================================================================
cudaError_t image_embedder_forward(
    const half* encoder_hidden_states_img,   // [N, img_seq, img_dim]
    half* out_img_k,                         // [N, img_seq, K]
    const half* norm1_gamma,                 // [img_dim]
    const half* norm1_beta,                  // [img_dim]
    const half* ff0_w,                       // [img_dim, img_dim] (row-major [in, out])
    const half* ff0_b,                       // [img_dim]
    const half* ff2_w,                       // [img_dim, K] (row-major [in, out])
    const half* ff2_b,                       // [K]
    const half* norm2_gamma,                 // [K]
    const half* norm2_beta,                  // [K]
    const half* pos_embed,                   // [1, img_seq, K] or nullptr
    int N, int img_seq, int img_dim, int K,
    half* scratch_ln1,                       // [N*img_seq, img_dim]
    half* scratch_ff0,                       // [N*img_seq, img_dim]
    cudaStream_t stream
);

// ============================================================================
// ACTIVATIONS
// REFERENCE: HuggingFace activations.py
// ============================================================================

/**
 * SiLU (Swish) activation: x * sigmoid(x)
 * Applied element-wise in-place
 */
cudaError_t apply_silu_inplace(
    half* data,
    int64_t total_elements,
    cudaStream_t stream
);

// ============================================================================
// PATCHIFY / UNPATCHIFY OPERATIONS
// ============================================================================

/**
 * Patchify: Extract patches from 3D video tensor
 * Converts spatial video to sequence of patches
 *
 * @param input Video tensor [N, C_in, T, H, W]
 * @param output Patches [N, post_T*post_H*post_W, C_in*pt*ph*pw]
 * @param N Batch size
 * @param C_in Input channels
 * @param T, H, W Input dimensions
 * @param pt, ph, pw Patch sizes
 * @param post_T, post_H, post_W Number of patches in each dimension
 *
 * REFERENCE: Inverse of unpatchify operation
 * Extracts non-overlapping patches with stride=patch_size
 */
cudaError_t patchify_3d(
    const half* input,
    half* output,
    int N, int C_in,
    int T, int H, int W,
    int pt, int ph, int pw,
    int post_T, int post_H, int post_W,
    cudaStream_t stream
);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Concatenate two tensors along dimension 1 (sequence dimension)
 * Used to combine image and text embeddings
 *
 * @param input1 First tensor [N, seq1, dim]
 * @param input2 Second tensor [N, seq2, dim]
 * @param output Concatenated tensor [N, seq1+seq2, dim]
 */
cudaError_t concatenate_seq_dim(
    const half* input1, int seq1,
    const half* input2, int seq2,
    half* output,
    int N, int dim,
    cudaStream_t stream
);

/**
 * Unflatten tensor dimension: [N, M*K] -> [N, M, K]
 * Used for reshaping timestep projections
 */
cudaError_t unflatten_dimension(
    const half* input,
    half* output,
    int N, int M, int K,
    cudaStream_t stream
);

// ============================================================================
// FORMAT CONVERSION UTILITIES
// Support conversion between PyTorch (NCTHW) and CUTLASS (NDHWC) formats
// ============================================================================

/**
 * Convert NCTHW -> NDHWC (PyTorch channels-first -> CUTLASS channels-last)
 * Input:  [N, C, T, H, W]
 * Output: [N, T, H, W, C]
 */
cudaError_t convert_ncthw_to_ndhwc(
    const half* input,
    half* output,
    int N, int C, int T, int H, int W,
    cudaStream_t stream
);

/**
 * Convert KCTRS -> KTRSC (PyTorch weight -> CUTLASS weight format)
 * Input:  [K, C, T, R, S]
 * Output: [K, T, R, S, C]
 */
cudaError_t convert_kctrs_to_ktrsc(
    const half* input,
    half* output,
    int K, int C, int T, int R, int S,
    cudaStream_t stream
);

/**
 * Flatten and transpose NDHWC to ColumnMajor
 * Input:  [N, T, H, W, C] NDHWC (channels last)
 * Output: [N*T*H*W, C] ColumnMajor
 *
 * Used after patch embedding to prepare data for transformer blocks.
 */
cudaError_t flatten_transpose_ndhwc_to_colmajor(
    const half* input,
    half* output,
    int N, int C, int T, int H, int W,
    cudaStream_t stream
);

// ============================================================================
// 3D CONVOLUTION FOR PATCH EMBEDDING
// REFERENCE: HuggingFace transformer_wan.py - patch_embedding layer
// ============================================================================

/**
 * 3D Convolution for patch embedding (channels-last native)
 * REFERENCE: WAN patch_embedding layer in transformer_wan.py
 *
 * This performs the initial spatial-to-sequence conversion in WAN transformer:
 *   input:  [N, T, H, W, C_in] -> patches -> output: [N, T', H', W', C_out]
 *
 * Typical WAN configuration: kernel=(1,2,2), stride=(1,2,2), no padding
 *   - Reduces spatial dimensions by 2x in H and W
 *   - Projects channels: 16 -> 1536 (inner_dim)
 *
 * @param input [N, T, H, W, C_in] NDHWC format (channels last, CUTLASS native)
 * @param weight [C_out, K_d, K_h, K_w, C_in] KTRSC format (channels last, CUTLASS native)
 * @param output [N, T_out, H_out, W_out, C_out] NDHWC format (channels last)
 * @param N Batch size
 * @param C_in Input channels
 * @param C_out Output channels
 * @param T, H, W Input spatial dimensions
 * @param K_d, K_h, K_w Kernel dimensions
 * @param stride_d, stride_h, stride_w Stride for each dimension
 * @param pad_d, pad_h, pad_w Padding for each dimension
 *
 * NOTE: This kernel works natively in channels-last (NDHWC) format.
 *       Call site is responsible for format conversions from/to PyTorch (NCTHW).
 */
cudaError_t patch_embedding_conv3d(
    const half* input,
    const half* weight,
    half* output,
    int N, int C_in, int C_out,
    int T, int H, int W,
    int K_d, int K_h, int K_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    cudaStream_t stream
);

// ============================================================================
// Pure C++ CUDA API for full transformer forward (no PyTorch types)
// ============================================================================

struct TransformerConfig {
    int num_attention_heads;
    int attention_head_dim;
    int num_layers;
    int ffn_dim;
    int patch_t;
    int patch_h;
    int patch_w;
    int freq_dim;           // e.g., 256
    int rope_max_seq_len;   // e.g., 1024
    int time_hidden_dim;    // hidden dim in timestep MLP
    int time_proj_dim;      // output dim of timestep MLP (often == K)
    int text_hidden_dim;    // hidden dim in 2-layer text embedder (if used)
    int out_channels;       // output latent channels (e.g., 16 for I2V)
    // Cross-attention KV sharing:
    // - 0 = default (project KV for each batch item)
    // - 1 = project KV once and broadcast to all batch items (requires identical encoder tokens across batch)
    int encoder_batch_size = 0;
};

template <typename WeightT>
struct BlockWeightsT {
    // Self-attention
    const WeightT* sa_w_qkv = nullptr;    // [3K,K]
    const half* sa_w_qkv_scale = nullptr; // [3K] per-channel scale (for FP8) or nullptr
    const float* sa_w_qkv_block_scale = nullptr; // [K/128, 3K/128] per-block scale (for INT8)
    const half* sa_b_qkv = nullptr;    // [3K]
    const half* sa_norm_q = nullptr;   // [K]
    const half* sa_norm_k = nullptr;   // [K]
    const WeightT* sa_w_out = nullptr;    // [K,K]
    const half* sa_w_out_scale = nullptr; // [K] per-channel scale or nullptr
    const float* sa_w_out_block_scale = nullptr; // [K/128, K/128] per-block scale (for INT8)
    const half* sa_b_out = nullptr;    // [K]

    // Cross-attention (separate Q and KV)
    const WeightT* ca_w_q = nullptr;      // [K,K]
    const half* ca_w_q_scale = nullptr;   // [K] per-channel scale or nullptr
    const float* ca_w_q_block_scale = nullptr; // [K/128, K/128] per-block scale (for INT8)
    const half* ca_b_q = nullptr;      // [K]
    const WeightT* ca_w_kv = nullptr;     // [2K,K]
    const half* ca_w_kv_scale = nullptr;  // [2K] per-channel scale or nullptr
    const float* ca_w_kv_block_scale = nullptr; // [K/128, 2K/128] per-block scale (for INT8)
    const half* ca_b_kv = nullptr;     // [2K]
    const half* ca_norm_q = nullptr;   // [K]
    const half* ca_norm_k = nullptr;   // [K]
    // I2V image cross-attention (add_k/add_v projections)
    const WeightT* ca_w_add_k = nullptr;   // [K, added_kv_proj_dim]
    const half* ca_b_add_k = nullptr;      // [K]
    const half* ca_w_add_k_scale = nullptr; // [K] or nullptr
    const WeightT* ca_w_add_v = nullptr;   // [K, added_kv_proj_dim]
    const half* ca_b_add_v = nullptr;      // [K]
    const half* ca_w_add_v_scale = nullptr; // [K] or nullptr
    const half* ca_norm_added_k = nullptr; // [K]
    const WeightT* ca_w_out = nullptr;    // [K,K]
    const half* ca_w_out_scale = nullptr; // [K] per-channel scale or nullptr
    const float* ca_w_out_block_scale = nullptr; // [K/128, K/128] per-block scale (for INT8)
    const half* ca_b_out = nullptr;    // [K]

    // Norms
    const half* ln1_gamma = nullptr;   // [K] optional
    const half* ln2_gamma = nullptr;   // [K] optional
    const half* ln2_bias = nullptr;    // [K] optional
    const half* ln3_gamma = nullptr;   // [K] optional

    // FFN
    const WeightT* ffn_w1 = nullptr;      // [FF,K]
    const half* ffn_w1_scale = nullptr;   // [FF] per-channel scale or nullptr
    const float* ffn_w1_block_scale = nullptr; // [K/128, FF/128] per-block scale (for INT8)
    const half* ffn_b1 = nullptr;      // [FF]
    const WeightT* ffn_w2 = nullptr;      // [K,FF]
    const half* ffn_w2_scale = nullptr;   // [K] per-channel scale or nullptr
    const float* ffn_w2_block_scale = nullptr; // [FF/128, K/128] per-block scale (for INT8)
    const half* ffn_b2 = nullptr;      // [K]

    // Per-block scale/shift table [6,K]
    const half* scale_shift_table = nullptr;
};

using BlockWeights = BlockWeightsT<cutlass::half_t>;

template <typename WeightT>
struct ModelWeightsT {
    // Patch embedding
    const half* patch_embedding_weight; // [K,C_in,pt,ph,pw] KCTRS
    const half* patch_embedding_bias;   // [K] optional

    // Timestep embedding MLP
    const half* time_proj_linear1_w; // [freq_dim, time_hidden_dim]
    const half* time_proj_linear1_b; // [time_hidden_dim]
    const half* time_proj_linear2_w; // [time_hidden_dim, time_proj_dim]
    const half* time_proj_linear2_b; // [time_proj_dim]

    // Final projection to 6*K
    const half* time_proj_final_w;   // [time_proj_dim, 6*K]
    const half* time_proj_final_b;   // [6*K]

    // Text projection (one of the two paths may be provided)
    const half* caption_proj_w;      // [text_dim, K]
    const half* caption_proj_b;      // [K]
    // Two-layer embedder
    const half* text_embedder_w1;    // [text_dim, text_hidden_dim]
    const half* text_embedder_b1;    // [text_hidden_dim]
    const half* text_embedder_w2;    // [text_hidden_dim, K]
    const half* text_embedder_b2;    // [K]

    // Image projection (optional)
    const half* image_proj_w;        // [img_dim, K] (legacy/simple projection; not used in HF-style image_embedder)
    const half* image_proj_b;        // [K]

    // HF-style image_embedder (WanImageEmbedding):
    // norm1(img_dim) -> MLP(img_dim -> img_dim -> K) -> norm2(K) -> optional pos_embed([1,img_seq,K])
    const half* image_norm1_gamma;   // [img_dim]
    const half* image_norm1_beta;    // [img_dim]
    const half* image_ff0_w;         // [img_dim, img_dim]  (MLP fc_in)
    const half* image_ff0_b;         // [img_dim]
    const half* image_ff2_w;         // [img_dim, K]        (MLP fc_out)
    const half* image_ff2_b;         // [K]
    const half* image_norm2_gamma;   // [K]
    const half* image_norm2_beta;    // [K]
    const half* image_pos_embed;     // [1, img_seq, K] optional

    // Transformer blocks
    const BlockWeightsT<WeightT>* blocks;   // array size num_layers

    // Final scale/shift table and projection
    const half* final_scale_shift_table; // [2,K]
    const half* proj_out_w;          // [K, patch_vol*C_in]
    const half* proj_out_b;          // [patch_vol*C_in] optional
};

using ModelWeights = ModelWeightsT<cutlass::half_t>;
struct ModelIO {
    // Shapes
    int N, C_in, T, H, W;        // input video
    int text_seq_len;            // encoder text seq length
    int img_seq_len;             // encoder image seq length (0 if none)
    int text_dim;                // encoder text embedding dim
    int img_dim;                 // encoder image embedding dim (0 if none)

    // Inputs
    const half* hidden_states_ncthw;       // [N,C,T,H,W]
    const int64_t* timesteps;              // [N] (long tensor from PyTorch)
    const half* encoder_hidden_states;     // [N,text_seq,text_dim]
    const half* encoder_hidden_states_img; // [N,img_seq,img_dim] or nullptr

    // Output
    half* out_ncthw;                       // [N,C_in,T,H,W]
};

// Full-model forward orchestrator (templated by block weight type and architecture).
// Arch defaults to WanArchTraits; callers that don't specify Arch get Wan behavior.
template <typename WeightT, typename Arch = WanArchTraits>
cudaError_t transformer_forward(
    const TransformerConfig& cfg,
    const ModelWeightsT<WeightT>& w,
    const ModelIO& io,
    ts::Workspace* workspace,
    cudaStream_t stream
);

} // namespace omnidreams_singleview
