// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ops.cuh"
#include "attention.cuh"
#include "cosmos_block.cuh"
#include "workspace.cuh"
#include "helper.h"
#include <unordered_map>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_bf16.h>
#include <cublasLt.h>
#ifndef OMNIDREAMS_SINGLEVIEW_NO_TORCH
#include <torch/extension.h>
#endif

#include "common/profile_config.h"
#include "common/profiler.h"
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass/functional.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/numeric_types.h"

// CUTLASS 3.x/4.x SM120 (Blackwell GeForce) includes
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/util/packed_stride.hpp"
#endif

#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv3d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/conv3d_problem_size.h>

#define PACK_INPUT_LAYOUT(problem) \
    cutlass::layout::TensorNDHWC::packed({ \
        (problem).N, (problem).D, (problem).H, (problem).W, (problem).C})

#define PACK_WEIGHT_LAYOUT(problem) \
    cutlass::layout::TensorNDHWC::packed({ \
        (problem).K, (problem).T, (problem).R, (problem).S, (problem).C})

#define PACK_OUTPUT_LAYOUT(problem) \
    cutlass::layout::TensorNDHWC::packed({ \
        (problem).N, (problem).Z, (problem).P, (problem).Q, (problem).K})

#define PACK_INPUT_LAYOUT_2D(problem) \
    cutlass::layout::TensorNHWC::packed({ \
        (problem).N, (problem).H, (problem).W, (problem).C})

#define PACK_WEIGHT_LAYOUT_2D(problem) \
    cutlass::layout::TensorNHWC::packed({ \
        (problem).K, (problem).R, (problem).S, (problem).C})

#define PACK_OUTPUT_LAYOUT_2D(problem) \
    cutlass::layout::TensorNHWC::packed({ \
        (problem).N, (problem).P, (problem).Q, (problem).K})

#include <cmath>

namespace omnidreams_singleview {

// Forward declarations for kernels used before definition
__global__ void gelu_erf_inplace_kernel(half* data, long long numel);
__global__ void add_pos_embed_img_kernel(half* __restrict__ data,
                                        const half* __restrict__ pos,
                                        int N, int img_seq, int K);
__global__ void add_bias_lastdim_kernel(half* __restrict__ data,
                                        const half* __restrict__ bias,
                                        long long total_elems,
                                        int K);
__global__ void col_scale_bias_to_bf16_kernel(
    const cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    cutlass::bfloat16_t* __restrict__ output,
    float scale_mul,
    int out_features,
    long long total_elems);
__global__ void col_scale_bias_to_bf16_vec2_kernel(
    const cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    cutlass::bfloat16_t* __restrict__ output,
    float scale_mul,
    int out_features,
    long long total_elems);
__global__ void col_scale_residual_bf16_kernel(
    const cutlass::half_t* __restrict__ data,
    cutlass::bfloat16_t* __restrict__ residual_inout,
    const cutlass::half_t* __restrict__ scale,
    float scale_mul,
    int out_features,
    long long total_elems);
__global__ void col_scale_residual_bf16_vec2_kernel(
    const cutlass::half_t* __restrict__ data,
    cutlass::bfloat16_t* __restrict__ residual_inout,
    const cutlass::half_t* __restrict__ scale,
    float scale_mul,
    int out_features,
    long long total_elems);
cudaError_t apply_col_scale_bias_gelu_to_fp8(
    const cutlass::half_t* data,
    cutlass::float_e4m3_t* fp8_out,
    const cutlass::half_t* scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul,
    float output_scale);
static bool ptr_aligned_4(const void* ptr);

// ============================================================================
// GEMM TILE CONFIGURATION
// ============================================================================

namespace {
  // Default threshold: 1024 (original behavior)
  // Lower values (e.g., 256, 512) use 128x64x64 tiles for smaller M,
  // which may improve performance for CFG batched workloads.
  int g_gemm_tile_threshold = 1024;
}

void set_gemm_tile_threshold(int threshold) {
  g_gemm_tile_threshold = threshold;
}

int get_gemm_tile_threshold() {
  return g_gemm_tile_threshold;
}

namespace {

bool env_value_is(const char* value, const char* expected) {
  return value && std::string(value) == expected;
}

bool byte_ranges_overlap(const void* lhs, size_t lhs_bytes, const void* rhs, size_t rhs_bytes) {
  if (!lhs || !rhs || lhs_bytes == 0 || rhs_bytes == 0) return false;
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  const auto lhs_end = lhs_begin + lhs_bytes;
  const auto rhs_end = rhs_begin + rhs_bytes;
  if (lhs_end < lhs_begin || rhs_end < rhs_begin) return true;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

bool valid_cosmos_fp8_linear_tile(const std::string& op_kind, const std::string& tile) {
  (void)op_kind;
  if (tile == "m128n128k128" || tile == "m128n64k128") {
    return true;
  }
  if (tile == "m128n32k128") {
    return true;
  }
  return false;
}

bool valid_cosmos_fp8_gelu_variant(const std::string& variant) {
  return variant == "default" ||
         variant == "nosrc_e128n16" ||
         variant == "nosrc_e128n32" ||
         variant == "nosrc_e128n64" ||
         variant == "nosrc_e128n128" ||
         variant == "nosrc_pingpong_e128n32";
}

std::string legacy_cosmos_fp8_linear_tile(const std::string& op_kind) {
  return op_kind == "gelu_fp8" ? "m128n128k128" : "m128n32k128";
}

std::string preset_540p_cosmos_fp8_linear_tile(
    const std::string& op_kind,
    int rows,
    int in_features,
    int out_features) {
  (void)op_kind;
  (void)rows;
  (void)in_features;
  (void)out_features;
  // 540p autotune winner, 2026-05-06:
  //   profiles/cosmos_fp8_attention/autotune_540p_20260506_180124_summary.json
  return "m128n128k128";
}

std::string preset_540p_cosmos_fp8_gelu_variant(
    const std::string& op_kind,
    int rows,
    int in_features,
    int out_features) {
  (void)rows;
  (void)in_features;
  (void)out_features;
  if (op_kind != "gelu_fp8") {
    return "default";
  }
  // 540p autotune winner, 2026-05-06:
  //   /tmp/native_cutlass_epilogue_exp/autotune_540p_20260506_202601_summary.json
  // Explicit OMNIDREAMS_DIT_FP8_GELU_VARIANT=default restores the stock CUTLASS epilogue.
  return "nosrc_pingpong_e128n32";
}

}  // namespace

CosmosFp8LinearTileSelection select_cosmos_fp8_linear_tile(
    const std::string& op_kind,
    int rows,
    int in_features,
    int out_features) {
  CosmosFp8LinearTileSelection selection;
  selection.op_kind = op_kind;

  const bool known_op =
      op_kind == "colscale_bf16" || op_kind == "residual_bf16" || op_kind == "gelu_fp8";
  const std::string normalized_op = known_op ? op_kind : "colscale_bf16";

  const char* preset_env = std::getenv("OMNIDREAMS_DIT_FP8_LINEAR_PRESET");
  selection.preset = (preset_env && preset_env[0] != '\0') ? std::string(preset_env) : "540p";

  const char* tile_env_name =
      normalized_op == "gelu_fp8" ? "OMNIDREAMS_DIT_FP8_GELU_TILE" : "OMNIDREAMS_DIT_FP8_LINEAR_TILE";
  const char* stage_env_name =
      normalized_op == "gelu_fp8" ? "OMNIDREAMS_DIT_FP8_GELU_STAGE" : "OMNIDREAMS_DIT_FP8_LINEAR_STAGE";
  const char* variant_env_name = "OMNIDREAMS_DIT_FP8_GELU_VARIANT";
  const char* tile_env = std::getenv(tile_env_name);
  const char* stage_env = std::getenv(stage_env_name);
  const char* variant_env = normalized_op == "gelu_fp8" ? std::getenv(variant_env_name) : nullptr;

  if (tile_env && tile_env[0] != '\0') {
    std::string requested(tile_env);
    if (valid_cosmos_fp8_linear_tile(normalized_op, requested)) {
      selection.tile = requested;
      selection.tile_env_override = true;
      selection.reason = std::string(tile_env_name) + "_override";
    } else {
      selection.reason = std::string("ignored_invalid_") + tile_env_name;
    }
  }

  if (selection.tile.empty()) {
    if (selection.preset == "legacy" || selection.preset == "env") {
      selection.tile = legacy_cosmos_fp8_linear_tile(normalized_op);
      if (selection.reason.empty()) {
        selection.reason = selection.preset == "env" ? "env_missing_tile_legacy_fallback" : "legacy";
      }
    } else {
      selection.tile = preset_540p_cosmos_fp8_linear_tile(
          normalized_op, rows, in_features, out_features);
      if (selection.reason.empty()) {
        selection.reason = selection.preset == "540p"
            ? "540p_preset"
            : "unknown_preset_540p_fallback";
      }
    }
  }

  if (stage_env && stage_env[0] != '\0') {
    if (env_value_is(stage_env, "2")) {
      selection.stage = "2";
      selection.stage_env_override = true;
    } else if (env_value_is(stage_env, "auto")) {
      selection.stage = "auto";
      selection.stage_env_override = true;
    } else {
      const bool preset_uses_540p_stage =
          selection.preset != "legacy" && selection.preset != "env";
      selection.stage = preset_uses_540p_stage ? "2" : "auto";
      if (selection.reason.empty() || selection.reason == "540p_preset") {
        selection.reason = std::string("ignored_invalid_") + stage_env_name;
      }
    }
  } else {
    const bool preset_uses_540p_stage =
        selection.preset != "legacy" && selection.preset != "env";
    selection.stage = preset_uses_540p_stage ? "2" : "auto";
  }

  if (!known_op) {
    selection.reason = "unknown_op_colscale_bf16_fallback";
  }
  const bool preset_uses_540p_variant =
      selection.preset != "legacy" && selection.preset != "env";
  selection.variant = preset_uses_540p_variant
      ? preset_540p_cosmos_fp8_gelu_variant(normalized_op, rows, in_features, out_features)
      : "default";
  if (normalized_op == "gelu_fp8" && variant_env && variant_env[0] != '\0') {
    std::string requested(variant_env);
    if (valid_cosmos_fp8_gelu_variant(requested)) {
      selection.variant = requested;
      selection.variant_env_override = true;
    } else if (selection.reason.empty() || selection.reason == "540p_preset") {
      selection.reason = std::string("ignored_invalid_") + variant_env_name;
    }
  }
  return selection;
}

// ============================================================================
// 3D ROTARY POSITION EMBEDDINGS (RoPE)
// REFERENCE: HuggingFace transformer_wan.py - WanRotaryPosEmbed (lines ~70-120)
// ============================================================================

/**
 * Helper: Generate 1D RoPE frequencies
 * REFERENCE: get_1d_rotary_pos_embed in embeddings.py
 *
 * Python equivalent:
 *   inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
 *   freqs = torch.outer(positions, inv_freq)
 *   cos = freqs.cos()
 *   sin = freqs.sin()
 *
 * NOTE: RoPE frequencies are duplicated for each even/odd pair.
 *       For dimension pairs (2i, 2i+1), both positions use the same frequency.
 */
__global__ void generate_rope_1d_freq_kernel(
    float* cos_out,  // [seq_len, dim] - full dimension with duplicates
    float* sin_out,  // [seq_len, dim] - full dimension with duplicates
    int seq_len,
    int dim,         // Must be even
    float theta,
    int rope_max_seq_len) {

    int seq_idx = blockIdx.x;
    int pair_idx = threadIdx.x;

    if (seq_idx >= seq_len || pair_idx >= dim / 2) return;

    // Compute inverse frequency for this dimension pair
    float freq_exp = float(pair_idx * 2) / float(dim);
    float inv_freq = 1.0f / powf(theta, freq_exp);

    // Scale position by frequency
    float angle = float(seq_idx) * inv_freq;

    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Store at both even and odd positions (for the pair)
    int even_idx = seq_idx * dim + pair_idx * 2;
    int odd_idx = even_idx + 1;

    cos_out[even_idx] = cos_val;
    sin_out[even_idx] = sin_val;

    if (pair_idx * 2 + 1 < dim) {
        cos_out[odd_idx] = cos_val;
        sin_out[odd_idx] = sin_val;
    }
}

/**
 * Combine 3 sets of 1D RoPE frequencies (T, H, W) into final 3D RoPE
 *
 * REFERENCE: WanRotaryPosEmbed.forward() - actual HF implementation uses unequal splits:
 * Python (actual):
 *   split_sizes = [D - 2*(D//3), D//3, D//3]  # e.g., [44, 42, 42] for D=128
 *   freqs_cos_t, freqs_cos_h, freqs_cos_w = freqs_cos.split(split_sizes, dim=1)
 *   freqs_cos_f = freqs_cos_t[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
 *   freqs_cos_h = freqs_cos_h[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
 *   freqs_cos_w = freqs_cos_w[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
 *   freqs = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1)
 *
 * NOTE: Input frequencies are already in full dimension (with even/odd duplicates)
 */
__global__ void combine_rope_3d_kernel(
    const float* cos_t, const float* sin_t,  // [T, D_t] - full dimension
    const float* cos_h, const float* sin_h,  // [H, D_h] - full dimension
    const float* cos_w, const float* sin_w,  // [W, D_w] - full dimension
    float* cos_out, float* sin_out,          // [T*H*W, D] where D = D_t + D_h + D_w
    int T, int H, int W, int D,
    int D_t, int D_h, int D_w) {              // Split sizes (full dimensions)

    int total_seq = T * H * W;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (seq_idx >= total_seq) return;

    // Decompose sequence index to (t, h, w)
    int w_idx = seq_idx % W;
    int h_idx = (seq_idx / W) % H;
    int t_idx = seq_idx / (H * W);

    // Copy temporal frequencies (first D_t elements)
    for (int d = 0; d < D_t; d++) {
        cos_out[seq_idx * D + d] = cos_t[t_idx * D_t + d];
        sin_out[seq_idx * D + d] = sin_t[t_idx * D_t + d];
    }

    // Copy height frequencies (next D_h elements)
    for (int d = 0; d < D_h; d++) {
        cos_out[seq_idx * D + D_t + d] = cos_h[h_idx * D_h + d];
        sin_out[seq_idx * D + D_t + d] = sin_h[h_idx * D_h + d];
    }

    // Copy width frequencies (last D_w elements)
    for (int d = 0; d < D_w; d++) {
        cos_out[seq_idx * D + D_t + D_h + d] = cos_w[w_idx * D_w + d];
        sin_out[seq_idx * D + D_t + D_h + d] = sin_w[w_idx * D_w + d];
    }
}

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
    cudaStream_t stream) {

    // HuggingFace uses unequal split matching this formula:
    //   h_dim = w_dim = 2 * (attention_head_dim // 6)
    //   t_dim = attention_head_dim - h_dim - w_dim
    // For D=128: t_dim=44, h_dim=w_dim=42
    // For D=64: t_dim=24, h_dim=w_dim=20
    int D_h_w = 2 * (head_dim / 6);        // Floor division
    int D_t = head_dim - 2 * D_h_w;        // Remainder goes to temporal

    if (D_t <= 0 || D_h_w <= 0) {
        return cudaErrorInvalidValue;  // head_dim too small
    }

    // Generate 1D frequencies for each dimension
    // Launch with (dim+1)/2 threads to handle odd dimensions properly
    int threads_t = (D_t + 1) / 2;   // Ceiling division
    int threads_h = (D_h_w + 1) / 2;
    int threads_w = (D_h_w + 1) / 2;

    generate_rope_1d_freq_kernel<<<post_T, threads_t, 0, stream>>>(
        cos_t, sin_t, post_T, D_t, theta, rope_max_seq_len);

    generate_rope_1d_freq_kernel<<<post_H, threads_h, 0, stream>>>(
        cos_h, sin_h, post_H, D_h_w, theta, rope_max_seq_len);

    generate_rope_1d_freq_kernel<<<post_W, threads_w, 0, stream>>>(
        cos_w, sin_w, post_W, D_h_w, theta, rope_max_seq_len);

    CUDA_CHECK(cudaGetLastError());

    // Combine into 3D RoPE with proper split sizes
    int total_seq = post_T * post_H * post_W;
    int threads = 256;
    int blocks = (total_seq + threads - 1) / threads;

    combine_rope_3d_kernel<<<blocks, threads, 0, stream>>>(
        cos_t, sin_t,
        cos_h, sin_h,
        cos_w, sin_w,
        cos_out, sin_out,
        post_T, post_H, post_W, head_dim,
        D_t, D_h_w, D_h_w);  // Pass split sizes

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

// ============================================================================
// TIMESTEP EMBEDDINGS
// REFERENCE: diffusers.models.embeddings.get_timestep_embedding
// ============================================================================

/**
 * Generate sinusoidal timestep embeddings
 * Python equivalent (with flip_sin_to_cos=True, downscale_freq_shift=0):
 *   half_dim = embedding_dim // 2
 *   freq = 10000^(-i/half_dim) for i in [0, half_dim)
 *   angle = timestep * freq
 *   output = [cos(angle), sin(angle)]  (concatenated)
 */
__global__ void timestep_sinusoidal_embedding_kernel(
    const int64_t* timesteps,
    half* embeddings,
    int num_timesteps,
    int freq_dim,
    int max_period) {

    int ts_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (ts_idx >= num_timesteps || dim_idx >= freq_dim) return;

    int half_dim = freq_dim / 2;

    // Compute frequency for this dimension
    float emb_scale = logf(float(max_period)) / float(half_dim);
    float freq;

    if (dim_idx < half_dim) {
        // Cos component (first half)
        freq = expf(float(dim_idx) * -emb_scale);
        float angle = float(timesteps[ts_idx]) * freq;
        embeddings[ts_idx * freq_dim + dim_idx] = __float2half(cosf(angle));
    } else {
        // Sin component (second half)
        int freq_idx = dim_idx - half_dim;
        freq = expf(float(freq_idx) * -emb_scale);
        float angle = float(timesteps[ts_idx]) * freq;
        embeddings[ts_idx * freq_dim + dim_idx] = __float2half(sinf(angle));
    }
}

cudaError_t timestep_sinusoidal_embedding(
    const int64_t* timesteps,
    half* embeddings,
    int num_timesteps,
    int freq_dim,
    int max_period,
    cudaStream_t stream) {

    timestep_sinusoidal_embedding_kernel<<<num_timesteps, freq_dim, 0, stream>>>(
        timesteps, embeddings, num_timesteps, freq_dim, max_period);

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

// ============================================================================
// TIMESTEP MLP PROJECTION
// REFERENCE: HuggingFace embeddings.py - TimestepEmbedding class
// Structure: Linear -> SiLU -> Linear
// ============================================================================

/**
 * SiLU (Swish) activation: x * sigmoid(x)
 * REFERENCE: activations.py - SiLU/Swish
 */
__global__ void silu_activation_kernel(
    half* data,
    int64_t total_elements) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = __half2float(data[idx]);
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    float result = x / (1.0f + expf(-x));
    data[idx] = __float2half(result);
}

cudaError_t apply_silu_inplace(
    half* data,
    int64_t total_elements,
    cudaStream_t stream) {

    int threads = 256;
    int64_t blocks = (total_elements + threads - 1) / threads;

    silu_activation_kernel<<<blocks, threads, 0, stream>>>(data, total_elements);

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

// ============================================================================
// SM120 (Blackwell GeForce) FP8 RCR GEMM kernel type definitions
// Uses CUTLASS 3.x/4.x CollectiveBuilder API with TMA + warp-specialized
// persistent kernels. SM120 native tensor core: F8F6F4 in TN layout only.
// ============================================================================
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

template <
    class CtaTileShapeMNK,
    class EpilogueTile,
    template <class> class ActivationFn,
    class ElementOutput,
    class ElementCompute,
    class ElementScalar,
    int AlignmentScalar,
    cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest>
struct CosmosSm120PerColScaleEltActNoSrcCallbacks
    : cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::Sm90Compute<
              ActivationFn,
              ElementOutput,
              ElementCompute,
              RoundStyle>,
          cutlass::epilogue::fusion::Sm90EVT<
              cutlass::epilogue::fusion::Sm90Compute<
                  cutlass::multiplies,
                  ElementCompute,
                  ElementCompute,
                  RoundStyle>,
              cutlass::epilogue::fusion::Sm90RowBroadcast<
                  0,
                  CtaTileShapeMNK,
                  ElementScalar,
                  ElementCompute,
                  cute::Stride<cute::_0, bool, int64_t>,
                  AlignmentScalar>,
              cutlass::epilogue::fusion::Sm90AccFetch>> {
    using Impl = cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            ActivationFn,
            ElementOutput,
            ElementCompute,
            RoundStyle>,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::multiplies,
                ElementCompute,
                ElementCompute,
                RoundStyle>,
            cutlass::epilogue::fusion::Sm90RowBroadcast<
                0,
                CtaTileShapeMNK,
                ElementScalar,
                ElementCompute,
                cute::Stride<cute::_0, bool, int64_t>,
                AlignmentScalar>,
            cutlass::epilogue::fusion::Sm90AccFetch>>;

    struct Arguments {
        ElementScalar alpha = ElementScalar(1);
        ElementScalar const* alpha_ptr = nullptr;

        using StrideAlpha = cute::Stride<cute::_0, bool, int64_t>;
        StrideAlpha dAlpha = {cute::_0{}, bool(1), 0};

        using ActivationArguments =
            typename cutlass::epilogue::fusion::Sm90Compute<
                ActivationFn,
                ElementOutput,
                ElementCompute,
                RoundStyle>::Arguments;
        ActivationArguments activation = ActivationArguments();

        operator typename Impl::Arguments() const {
            return {
                {
                    {alpha_ptr, alpha, dAlpha},
                    {},
                    {}
                },
                activation
            };
        }
    };

    using Impl::Impl;
};

namespace sm120_fp8_rcr_bias {
    // D = alpha * (A @ B) + bias
    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;     // TN: A is Row
    using LayoutB = cutlass::layout::ColumnMajor;   // TN: B is Col
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // No multicast on SM120

    using FusionOp = cutlass::epilogue::fusion::LinCombPerRowBias<
        ElementD, ElementCompute, ElementC, ElementC, ElementCompute>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace sm120_fp8_rcr_bias

namespace sm120_fp8_rcr_bias_gelu {
    // D = GELU(alpha * (A @ B) + bias)
    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using FusionOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
        cutlass::epilogue::thread::GELU,
        ElementD, ElementCompute, ElementC, ElementC, ElementCompute>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace sm120_fp8_rcr_bias_gelu

namespace sm120_fp8_rcr_colscale_bf16 {
    // D = (A @ B) * per_column_scale, with BF16 output.
    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::Identity,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace sm120_fp8_rcr_colscale_bf16

namespace sm120_fp8_rcr_colscale_bf16_m128n64k128 {
    // More N-side CTAs for the Cosmos target linears. Kept opt-in for autotune.
    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<_128, _64, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::Identity,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace sm120_fp8_rcr_colscale_bf16_m128n64k128

namespace sm120_fp8_rcr_colscale_bf16_m128n32k128 {
    // Higher N-side CTA count for Cosmos target linears. Opt-in until measured.
    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::Identity,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace sm120_fp8_rcr_colscale_bf16_m128n32k128

namespace sm120_fp8_rcr_colscale_gelu_fp8 {
    // D = fp8(GELU((A @ B) * per_column_scale)).
    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::float_e4m3_t;
    using ElementD = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::GELU,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace sm120_fp8_rcr_colscale_gelu_fp8

namespace sm120_fp8_rcr_colscale_gelu_fp8_m128n64k128 {
    // More N-side CTAs for target FFN1 GELU-to-FP8 GEMMs. Kept opt-in until measured.
    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::float_e4m3_t;
    using ElementD = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<_128, _64, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::GELU,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace sm120_fp8_rcr_colscale_gelu_fp8_m128n64k128

namespace sm120_fp8_rcr_colscale_gelu_fp8_m128n32k128 {
    // Highest N-side CTA count for FFN1 GELU-to-FP8 GEMMs. Kept selectable for
    // 540p autotune because GELU epilogue register pressure may prefer more
    // parallel CTAs on long token sequences.
    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::float_e4m3_t;
    using ElementD = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::GELU,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace sm120_fp8_rcr_colscale_gelu_fp8_m128n32k128

template <class TileShape_, int Stages>
struct Sm120Fp8RcrColscaleBf16StageConfig {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = TileShape_;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::Identity,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCount<Stages>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class TileShape_, int Stages>
struct Sm120Fp8RcrColscaleGeluFp8StageConfig {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::float_e4m3_t;
    using ElementD = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using MmaTileShape = TileShape_;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::GELU,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCount<Stages>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class TileShape_, class EpilogueTile_, int Stages, class KernelScheduleTag>
struct Sm120Fp8RcrColscaleGeluFp8NoSrcStageConfig {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = void;
    using ElementD = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentScale = 128 / cutlass::sizeof_bits<ElementScale>::value;

    using MmaTileShape = TileShape_;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using EpilogueTile = EpilogueTile_;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::GELU,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale,
        AlignmentScale,
        AlignmentScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        EpilogueTile,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCount<Stages>,
        KernelScheduleTag>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class TileShape_, class EpilogueTile_, class KernelScheduleTag>
struct Sm120Fp8RcrColscaleGeluFp8NoSrcAutoConfig {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = void;
    using ElementD = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScale = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentScale = 128 / cutlass::sizeof_bits<ElementScale>::value;

    using MmaTileShape = TileShape_;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using EpilogueTile = EpilogueTile_;

    using FusionOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::GELU,
        ElementD,
        ElementCompute,
        ElementScale,
        ElementC,
        ElementScale,
        AlignmentScale,
        AlignmentScale>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        EpilogueTile,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelScheduleTag>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

namespace {
    void* g_sm120_workspace = nullptr;
    size_t g_sm120_workspace_size = 0;
    cublasLtHandle_t g_cublaslt_handle = nullptr;
    void* g_cublaslt_workspace = nullptr;
    size_t g_cublaslt_workspace_size = 0;
    cutlass::half_t* g_cublaslt_half_scratch = nullptr;
    size_t g_cublaslt_half_scratch_elems = 0;

    void* ensure_sm120_workspace(size_t needed) {
        if (needed > g_sm120_workspace_size) {
            if (g_sm120_workspace) cudaFree(g_sm120_workspace);
            cudaMalloc(&g_sm120_workspace, needed);
            g_sm120_workspace_size = needed;
        }
        return g_sm120_workspace;
    }

    cudaError_t ensure_cublaslt_handle(cudaStream_t stream) {
        if (!g_cublaslt_handle) {
            cublasStatus_t st = cublasLtCreate(&g_cublaslt_handle);
            if (st != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
        }
        (void)stream;
        return cudaSuccess;
    }

    void destroy_cublaslt_descs(
        cublasLtMatmulPreference_t pref,
        cublasLtMatrixLayout_t d_desc,
        cublasLtMatrixLayout_t c_desc,
        cublasLtMatrixLayout_t b_desc,
        cublasLtMatrixLayout_t a_desc,
        cublasLtMatmulDesc_t op_desc) {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (d_desc) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    }

    cudaError_t ensure_cublaslt_workspace(size_t needed) {
        if (needed > g_cublaslt_workspace_size) {
            if (g_cublaslt_workspace) cudaFree(g_cublaslt_workspace);
            cudaError_t err = cudaMalloc(&g_cublaslt_workspace, needed);
            if (err != cudaSuccess) {
                g_cublaslt_workspace = nullptr;
                g_cublaslt_workspace_size = 0;
                return err;
            }
            g_cublaslt_workspace_size = needed;
        }
        return cudaSuccess;
    }

    cudaError_t ensure_cublaslt_half_scratch(size_t elems) {
        if (elems > g_cublaslt_half_scratch_elems) {
            if (g_cublaslt_half_scratch) cudaFree(g_cublaslt_half_scratch);
            cudaError_t err = cudaMalloc(&g_cublaslt_half_scratch, elems * sizeof(cutlass::half_t));
            if (err != cudaSuccess) {
                g_cublaslt_half_scratch = nullptr;
                g_cublaslt_half_scratch_elems = 0;
                return err;
            }
            g_cublaslt_half_scratch_elems = elems;
        }
        return cudaSuccess;
    }

    cudaError_t cublaslt_status_to_cuda(cublasStatus_t st) {
        if (st == CUBLAS_STATUS_SUCCESS) return cudaSuccess;
        if (st == CUBLAS_STATUS_NOT_SUPPORTED || st == CUBLAS_STATUS_ARCH_MISMATCH) {
            return cudaErrorNotSupported;
        }
        if (st == CUBLAS_STATUS_INVALID_VALUE) return cudaErrorInvalidValue;
        return cudaErrorUnknown;
    }

    // Default-on toggle for the SM120 fused FP8 GEMM + epilogue path. When
    // enabled, FP8 GEMMs route through the custom CUTLASS SM120 GMMA kernel
    // with fused col_scale / bias / GELU / residual / FP8-quantize epilogues
    // (eliminates the cuBLASLt FP8 GEMM + standalone post-op kernel pair).
    // The path depends on the native TMA-descriptor pool patch in
    // `cute/atom/copy_traits_sm90_tma.hpp` and `detail/tma_descriptor_pool.hpp`
    // to work around an nvcc 13.x SM120 __grid_constant__ spill bug.
    //
    // Set OMNIDREAMS_DIT_FP8_FUSED_EPILOGUE=0 (or false/no) to fall back to the
    // legacy cuBLASLt sm89_xmma_gemm_e4m3 + post-op kernel pair (useful for
    // A/B testing or if a future build environment regresses the SM120 path).
    bool prefer_sm120_fused_fp8_epilogue() {
        const char* value = std::getenv("OMNIDREAMS_DIT_FP8_FUSED_EPILOGUE");
        if (!value || !value[0]) return true;
        return value[0] != '0' &&
               value[0] != 'f' &&
               value[0] != 'F' &&
               value[0] != 'n' &&
               value[0] != 'N';
    }

    bool cosmos_fp8_fused_debug_enabled() {
        const char* value = std::getenv("OMNIDREAMS_DIT_FP8_FUSED_EPILOGUE_DEBUG");
        if (!value || !value[0]) return false;
        return value[0] != '0' &&
               value[0] != 'f' &&
               value[0] != 'F' &&
               value[0] != 'n' &&
               value[0] != 'N';
    }

    cudaError_t cublaslt_fp8_rcr_to_half(
        const cutlass::float_e4m3_t* input_row,
        const cutlass::float_e4m3_t* weight_fp8,
        cutlass::half_t* output_row,
        int M,
        int K,
        int N,
        float alpha,
        cudaStream_t stream) {
        if (!input_row || !weight_fp8 || !output_row || M <= 0 || K <= 0 || N <= 0) {
            return cudaErrorInvalidValue;
        }
        cudaError_t err = ensure_cublaslt_handle(stream);
        if (err != cudaSuccess) return err;

        cublasLtMatmulDesc_t op_desc = nullptr;
        cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr, d_desc = nullptr;
        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned_results = 0;
        constexpr size_t kWorkspaceBytes = 32ull * 1024ull * 1024ull;

        cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (st != CUBLAS_STATUS_SUCCESS) {
            destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
            return cublaslt_status_to_cuda(st);
        }
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_T;
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
        }

        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, M, K, K);
        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, N, K, K);
        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, M, N, N);
        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16F, M, N, N);
        cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
        }
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
        }
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
        }
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
        }
        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulPreferenceCreate(&pref);
        if (st == CUBLAS_STATUS_SUCCESS) {
            size_t workspace_bytes = kWorkspaceBytes;
            st = cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_bytes, sizeof(workspace_bytes));
        }
        if (st != CUBLAS_STATUS_SUCCESS) {
            destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
            return cublaslt_status_to_cuda(st);
        }

        err = ensure_cublaslt_workspace(kWorkspaceBytes);
        if (err != cudaSuccess) {
            destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
            return err;
        }

        st = cublasLtMatmulAlgoGetHeuristic(
            g_cublaslt_handle, op_desc, a_desc, b_desc, c_desc, d_desc,
            pref, 1, &heuristic, &returned_results);
        if (st != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
            destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
            return cudaErrorNotSupported;
        }

        float beta = 0.0f;
        st = cublasLtMatmul(
            g_cublaslt_handle,
            op_desc,
            &alpha,
            input_row,
            a_desc,
            weight_fp8,
            b_desc,
            &beta,
            output_row,
            c_desc,
            output_row,
            d_desc,
            &heuristic.algo,
            g_cublaslt_workspace,
            kWorkspaceBytes,
            stream);
        destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
        return cublaslt_status_to_cuda(st);
    }

    bool sm120_fp8_stage2(const CosmosFp8LinearTileSelection& selection) {
        return selection.stage == "2";
    }

    template <typename Gemm>
    cudaError_t run_sm120_fp8_colscale_bf16_gemm(
        const cutlass::float_e4m3_t* input_row,
        const cutlass::float_e4m3_t* weight_fp8,
        const cutlass::half_t* weight_scale,
        cutlass::bfloat16_t* output_row,
        int N, int in_features, int out_features,
        cutlass::half_t beta,
        cudaStream_t stream) {
        using StrideA = typename Gemm::GemmKernel::StrideA;
        using StrideB = typename Gemm::GemmKernel::StrideB;
        using StrideC = typename Gemm::GemmKernel::StrideC;
        using StrideD = typename Gemm::GemmKernel::StrideD;

        int M = N;
        int K = in_features;
        int N_gemm = out_features;

        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N_gemm, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N_gemm, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N_gemm, 1));

        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N_gemm, K, 1},
            {input_row, stride_A, weight_fp8, stride_B},
            {{},
             output_row, stride_C, output_row, stride_D}
        };
        arguments.epilogue.thread.alpha_ptr = weight_scale;
        arguments.epilogue.thread.beta = beta;
        arguments.epilogue.thread.bias_ptr = nullptr;

        Gemm gemm_op;
        auto status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            if (cosmos_fp8_fused_debug_enabled())
                std::fprintf(stderr, "[bf16gemm] can_implement FAIL status=%d (%s) M=%d N=%d K=%d\n",
                             int(status), cutlass::cutlassGetStatusString(status), M, N_gemm, K);
            return cudaErrorNotSupported;
        }

        size_t ws_size = Gemm::get_workspace_size(arguments);
        void* ws = ws_size > 0 ? ensure_sm120_workspace(ws_size) : nullptr;

        status = gemm_op.initialize(arguments, ws, stream);
        if (status != cutlass::Status::kSuccess) {
            if (cosmos_fp8_fused_debug_enabled())
                std::fprintf(stderr, "[bf16gemm] initialize FAIL status=%d (%s) M=%d N=%d K=%d ws=%zu\n",
                             int(status), cutlass::cutlassGetStatusString(status), M, N_gemm, K, ws_size);
            return cudaErrorUnknown;
        }
        status = gemm_op.run(stream);
        if (status != cutlass::Status::kSuccess) {
            if (cosmos_fp8_fused_debug_enabled())
                std::fprintf(stderr, "[bf16gemm] run FAIL status=%d (%s) M=%d N=%d K=%d\n",
                             int(status), cutlass::cutlassGetStatusString(status), M, N_gemm, K);
            return cudaErrorUnknown;
        }
        return cudaGetLastError();
    }

    template <typename Gemm>
    cudaError_t run_sm120_fp8_colscale_gelu_fp8_gemm(
        const cutlass::float_e4m3_t* input_row,
        const cutlass::float_e4m3_t* weight_fp8,
        const cutlass::half_t* weight_scale,
        cutlass::float_e4m3_t* output_fp8,
        int N, int in_features, int out_features,
        cudaStream_t stream) {
        using StrideA = typename Gemm::GemmKernel::StrideA;
        using StrideB = typename Gemm::GemmKernel::StrideB;
        using StrideC = typename Gemm::GemmKernel::StrideC;
        using StrideD = typename Gemm::GemmKernel::StrideD;

        int M = N;
        int K = in_features;
        int N_gemm = out_features;

        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N_gemm, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N_gemm, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N_gemm, 1));
        using ElementCArg = typename Gemm::ElementC;
        ElementCArg const* ptr_C = nullptr;
        if constexpr (!cute::is_void_v<ElementCArg>) {
            ptr_C = output_fp8;
        }

        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N_gemm, K, 1},
            {input_row, stride_A, weight_fp8, stride_B},
            {{},
             ptr_C, stride_C, output_fp8, stride_D}
        };
        arguments.epilogue.thread.alpha_ptr = weight_scale;
        using ThreadArgs = decltype(arguments.epilogue.thread);
        if constexpr (requires(ThreadArgs& t) { t.beta; }) {
            arguments.epilogue.thread.beta = cutlass::half_t(0.0f);
        }
        if constexpr (requires(ThreadArgs& t) { t.bias_ptr; }) {
            arguments.epilogue.thread.bias_ptr = nullptr;
        }

        Gemm gemm_op;
        auto status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            if (cosmos_fp8_fused_debug_enabled()) {
                std::fprintf(stderr,
                    "[cosmos_fp8_fused] can_implement failed status=%d M=%d N=%d K=%d "
                    "input=%p weight=%p scale=%p output=%p C=%p\n",
                    int(status), M, N_gemm, K,
                    static_cast<const void*>(input_row),
                    static_cast<const void*>(weight_fp8),
                    static_cast<const void*>(weight_scale),
                    static_cast<void*>(output_fp8),
                    static_cast<const void*>(ptr_C));
            }
            return cudaErrorNotSupported;
        }

        size_t ws_size = Gemm::get_workspace_size(arguments);
        void* ws = ws_size > 0 ? ensure_sm120_workspace(ws_size) : nullptr;
        if (cosmos_fp8_fused_debug_enabled()) {
            std::fprintf(stderr,
                "[cosmos_fp8_fused] launch M=%d N=%d K=%d workspace=%zu "
                "input=%p weight=%p scale=%p output=%p C=%p\n",
                M, N_gemm, K, ws_size,
                static_cast<const void*>(input_row),
                static_cast<const void*>(weight_fp8),
                static_cast<const void*>(weight_scale),
                static_cast<void*>(output_fp8),
                static_cast<const void*>(ptr_C));
        }

        status = gemm_op.initialize(arguments, ws, stream);
        if (status != cutlass::Status::kSuccess) {
            if (cosmos_fp8_fused_debug_enabled()) {
                std::fprintf(stderr, "[cosmos_fp8_fused] initialize failed status=%d\n", int(status));
            }
            return cudaErrorUnknown;
        }
        status = gemm_op.run(stream);
        if (status != cutlass::Status::kSuccess) {
            if (cosmos_fp8_fused_debug_enabled()) {
                std::fprintf(stderr, "[cosmos_fp8_fused] run failed status=%d\n", int(status));
            }
            return cudaErrorUnknown;
        }
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess && cosmos_fp8_fused_debug_enabled()) {
            std::fprintf(stderr,
                "[cosmos_fp8_fused] cudaGetLastError=%d (%s)\n",
                int(launch_err), cudaGetErrorString(launch_err));
        }
        return launch_err;
    }

    template <class EpilogueTile, class KernelScheduleTag>
    cudaError_t run_sm120_fp8_colscale_gelu_fp8_nosrc_variant(
        const cutlass::float_e4m3_t* input_row,
        const cutlass::float_e4m3_t* weight_fp8,
        const cutlass::half_t* weight_scale,
        cutlass::float_e4m3_t* output_fp8,
        int N, int in_features, int out_features,
        const CosmosFp8LinearTileSelection& selection,
        cudaStream_t stream) {
        using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
        if (cosmos_fp8_fused_debug_enabled()) {
            std::fprintf(stderr,
                "[cosmos_fp8_fused] variant=%s tile=%s stage=%s reason=%s rows=%d in=%d out=%d\n",
                selection.variant.c_str(),
                selection.tile.c_str(),
                selection.stage.c_str(),
                selection.reason.c_str(),
                N, in_features, out_features);
        }
        if (sm120_fp8_stage2(selection)) {
            return run_sm120_fp8_colscale_gelu_fp8_gemm<
                typename Sm120Fp8RcrColscaleGeluFp8NoSrcStageConfig<
                    TileShape, EpilogueTile, 2, KernelScheduleTag>::Gemm>(
                        input_row, weight_fp8, weight_scale, output_fp8,
                        N, in_features, out_features, stream);
        }
        return run_sm120_fp8_colscale_gelu_fp8_gemm<
            typename Sm120Fp8RcrColscaleGeluFp8NoSrcAutoConfig<
                TileShape, EpilogueTile, KernelScheduleTag>::Gemm>(
                    input_row, weight_fp8, weight_scale, output_fp8,
                    N, in_features, out_features, stream);
    }
}

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED

// Cached runtime SM major version (queried once, reused for all FP8 dispatch).
namespace {
    int g_runtime_sm_major = -1;
    int get_runtime_sm_major() {
        if (g_runtime_sm_major < 0) {
            int device = 0;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&g_runtime_sm_major, cudaDevAttrComputeCapabilityMajor, device);
        }
        return g_runtime_sm_major;
    }
}

// SM89 fallback for FP8 RCR GEMM (always compiled, used as fallback on non-Blackwell GPUs)
// Uses 128x64x128 tile (known-working from pre-anahmad-turbo codebase).

// SM89 fallback for FP8 RCR GEMM + GELU (always compiled)
// Uses 128x64x128 tile (known-working from pre-anahmad-turbo codebase).

// Strided-batched BF16 GEMM via cuBLASLt: D[b] = A[b] @ B[b]^T + (bias != null ? C[b] : 0)
//
// Used for the AdaLN-LoRA pre-stack path's "up" GEMM. With per-instance
// (M=B=1, N=3*model_channels=6144, K=lora_dim=256) and batchCount = num_blocks*3,
// the batch dim fills the GPU even though per-instance M is a vector.
//
// Layout matches PyTorch nn.Linear: A is row-major [M, K]; B is row-major
// [N, K] (PyTorch's [out, in] weight layout); D is row-major [M, N].
// bias (when non-null) is treated as the C matrix with the supplied stride;
// pass stride_bias=0 to broadcast a single bias matrix across all batches.
cudaError_t cublaslt_strided_batched_bf16_gemm(
    const cutlass::bfloat16_t* a_row,
    const cutlass::bfloat16_t* b_row,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* d_row,
    int M, int K, int N,
    int batchCount,
    int64_t stride_a, int64_t stride_b, int64_t stride_d, int64_t stride_bias,
    cudaStream_t stream) {
    if (!a_row || !b_row || !d_row || M <= 0 || K <= 0 || N <= 0 || batchCount <= 0) {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = ensure_cublaslt_handle(stream);
    if (err != cudaSuccess) return err;

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr, d_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    constexpr size_t kWorkspaceBytes = 32ull * 1024ull * 1024ull;

    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) {
        destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
        return cublaslt_status_to_cuda(st);
    }
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    }

    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, M, K, K);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, N, K, K);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, M, N, N);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, M, N, N);
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));

    int32_t batch_count = batchCount;
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));

    int64_t sa = stride_a, sb = stride_b, sd = stride_d;
    int64_t sc = (bias != nullptr) ? stride_bias : stride_d;
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sc, sizeof(sc));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sd, sizeof(sd));

    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulPreferenceCreate(&pref);
    if (st == CUBLAS_STATUS_SUCCESS) {
        size_t workspace_bytes = kWorkspaceBytes;
        st = cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes));
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
        destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
        return cublaslt_status_to_cuda(st);
    }

    err = ensure_cublaslt_workspace(kWorkspaceBytes);
    if (err != cudaSuccess) {
        destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
        return err;
    }

    st = cublasLtMatmulAlgoGetHeuristic(
        g_cublaslt_handle, op_desc, a_desc, b_desc, c_desc, d_desc,
        pref, 1, &heuristic, &returned_results);
    if (st != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
        destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
        return cudaErrorNotSupported;
    }

    float alpha_f = 1.0f;
    float beta_f = (bias != nullptr) ? 1.0f : 0.0f;
    const cutlass::bfloat16_t* c_ptr = (bias != nullptr) ? bias : d_row;
    st = cublasLtMatmul(
        g_cublaslt_handle, op_desc,
        &alpha_f, a_row, a_desc, b_row, b_desc,
        &beta_f, c_ptr, c_desc, d_row, d_desc,
        &heuristic.algo, g_cublaslt_workspace, kWorkspaceBytes,
        stream);
    destroy_cublaslt_descs(pref, d_desc, c_desc, b_desc, a_desc, op_desc);
    return cublaslt_status_to_cuda(st);
}




// FP8 GEMM (cuBLASLt) followed by a fused col_scale + residual + LayerNorm
// + modulate -> FP8 post-op kernel. Replaces the back-to-back pair
// (col_scale_residual_bf16_vec2 + cosmos_layernorm_modulate_to_fp8_only) that
// runs after SA-out / CA-out projections in the production FP8 path. The
// fused post-op keeps the BF16-truncated `x_new` row in shared memory between
// the residual write and the LN reduce, eliminating the BF16 re-read of x.
//
// Falls back to cudaErrorNotSupported when the cuBLASLt half-scratch path
// can't be set up; callers should run the unfused two-step pair in that case.
//
// Layout matches `cutlass_linear_layer_rcr_fp8_colscale_residual_bf16`: alpha
// is the precomputed per-column (weight_scale * gate * input_scale) FP16
// vector produced by `cosmos_scale_gate_half`.


// FP8 A/B (RRR) path with stride-trick bias fusion
// SM120 only supports TN layout for F8 types; RRR (TT) is not available.
// Guard out to avoid Sm89 2.x MMA template errors when SM120 headers are active.
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
cudaError_t cutlass_linear_layer_rrr_fp8(
    const cutlass::float_e4m3_t* input_row,   // Row-major [N, in_features] FP8
    const cutlass::float_e4m3_t* weight_fp8,  // Row-major [in_features, out_features] FP8
    const cutlass::half_t* bias,              // [out_features] half or nullptr
    cutlass::half_t* output_row,              // Row-major [N, out_features] half
    int N, int in_features, int out_features,
    cudaStream_t stream) {

    using ElementInputA = cutlass::float_e4m3_t;
    using ElementInputB = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;

    using LayoutC = cutlass::layout::RowMajor;

    const cutlass::half_t* c_ptr = bias ? bias : output_row;
    int ldc = bias ? 0 : out_features;
    float beta = bias ? 1.0f : 0.0f;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        8,                      // elements per access (reduce to avoid zero row iterations)
        ElementAccumulator,
        ElementAccumulator>;

    // 128x128x128, 8 warps, 16384 output elements/CTA (2x original)
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA, ElementInputB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 128, 128>, cutlass::gemm::GemmShape<64, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp>;
    Gemm gemm_op;
    Gemm::Arguments args({N, out_features, in_features},
                         {input_row, in_features},
                         {weight_fp8, out_features},
                         {c_ptr, ldc},   // bias broadcast when ldc==0
                         {output_row, out_features},
                         {1.0f, beta});

    cutlass::Status status;
    status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    return cudaSuccess;
}
#endif // !CUTLASS_ARCH_MMA_SM120_SUPPORTED

cudaError_t cutlass_linear_layer_rcr_int8(
    const int8_t* input_row,
    const int8_t* weight_int8,
    const cutlass::half_t* bias,
    cutlass::half_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream,
    float activation_scale) {

    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = int32_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    const cutlass::half_t* c_ptr = bias ? bias : output_row;
    int ldc = bias ? 0 : out_features;
    float beta = bias ? 1.0f : 0.0f;

    int const kElementsPerAccess = 8; // 8 half values per vectorized access

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPerAccess,
        ElementAccumulator,
        float>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp
    >;

    Gemm gemm_op;
    Gemm::Arguments args({N, out_features, in_features},
                         {input_row, in_features},
                         {weight_int8, in_features},
                         {c_ptr, ldc},
                         {output_row, out_features},
                         {activation_scale, beta});

    cutlass::Status status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    return cudaSuccess;
}

// Row-major GEMM with GELU epilogue: D = GELU(A @ B + bias)
cudaError_t cutlass_linear_layer_rrr_gelu(
    const half* input_row,
    const half* weight_row,
    const half* bias,
    half* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream) {

    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementOutput,
        kElementsPerAccess,
        ElementAccumulator,
        float
    >;

    // Use 256x128x32 tiles to match PyTorch/cuBLAS optimal configuration
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        EpilogueOp
    >;

    Gemm gemm_op;

    typename EpilogueOp::Params epilogue_params(1.0f, 1.0f);
    Gemm::Arguments args(
        {N, out_features, in_features},
        {reinterpret_cast<const cutlass::half_t*>(input_row), in_features},
        {reinterpret_cast<const cutlass::half_t*>(weight_row), out_features},
        {reinterpret_cast<const cutlass::half_t*>(bias), 0},  // Bias with stride 0
        {reinterpret_cast<cutlass::half_t*>(output_row), out_features},
        epilogue_params
    );

    cutlass::Status status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    return cudaSuccess;
}

// FP8 A/B (RCR) with GELU epilogue: D = GELU(A @ B + bias)

// FP8 A/B (RRR) with GELU epilogue: D = GELU(A @ B + bias)
// SM120 only supports TN layout for F8 types; RRR (TT) is not available.
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
cudaError_t cutlass_linear_layer_rrr_fp8_gelu(
    const cutlass::float_e4m3_t* input_row,   // Row-major [N, in_features] FP8
    const cutlass::float_e4m3_t* weight_fp8,  // Row-major [in_features, out_features] FP8
    const cutlass::half_t* bias,              // [out_features] half or nullptr
    cutlass::half_t* output_row,              // Row-major [N, out_features] half
    int N, int in_features, int out_features,
    cudaStream_t stream) {

    using ElementInputA = cutlass::float_e4m3_t;
    using ElementInputB = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementOutput,
        kElementsPerAccess,
        ElementAccumulator,
        float
    >;

    // 128x128x128, 8 warps, 16384 output elements/CTA (2x original)
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 128, 128>,
        cutlass::gemm::GemmShape<64, 32, 128>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp
    >;

    Gemm gemm_op;

    float beta = bias ? 1.0f : 0.0f;
    typename EpilogueOp::Params epilogue_params(1.0f, beta);
    const cutlass::half_t* c_ptr = bias ? bias : output_row;
    int ldc = bias ? 0 : out_features;
    Gemm::Arguments args(
        {N, out_features, in_features},
        {input_row, in_features},
        {weight_fp8, out_features},
        {c_ptr, ldc}, // bias broadcast when ldc==0
        {output_row, out_features},
        epilogue_params
    );

    cutlass::Status status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    return cudaSuccess;
}
#endif // !CUTLASS_ARCH_MMA_SM120_SUPPORTED

// INT8 A/B (RCR) with GELU epilogue: D = GELU(A @ B + bias)
cudaError_t cutlass_linear_layer_rcr_int8_gelu(
    const int8_t* input_row,
    const int8_t* weight_int8,
    const cutlass::half_t* bias,
    cutlass::half_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream,
    float activation_scale) {

    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = int32_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    int const kElementsPerAccess = 8;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementOutput,
        kElementsPerAccess,
        ElementAccumulator,
        float
    >;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp
    >;

    Gemm gemm_op;

    const cutlass::half_t* c_ptr = bias ? bias : output_row;
    int ldc = bias ? 0 : out_features;
    float beta = bias ? 1.0f : 0.0f;
    typename EpilogueOp::Params epilogue_params(activation_scale, beta);

    Gemm::Arguments args(
        {N, out_features, in_features},
        {input_row, in_features},
        {weight_int8, in_features},
        {c_ptr, ldc},
        {output_row, out_features},
        epilogue_params
    );

    cutlass::Status status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    return cudaSuccess;
}

// ============================================================================
// Utilities: conversions
// ============================================================================

__global__ void half_to_fp8_e4m3_kernel(
    const cutlass::half_t* __restrict__ src,
    cutlass::float_e4m3_t* __restrict__ dst,
    int64_t n) {
  // 4-wide vectorized: each thread processes 4 FP8 elements
  int64_t base = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (base >= n) return;
  const half2* src_h2 = reinterpret_cast<const half2*>(src);
  half2 v01 = src_h2[base / 2];
  half2 v23 = (base + 2 < n) ? src_h2[base / 2 + 1] : half2{__float2half(0.f), __float2half(0.f)};
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  dst[base]     = to_fp8(__half2float(v01.x));
  if (base + 1 < n) dst[base + 1] = to_fp8(__half2float(v01.y));
  if (base + 2 < n) dst[base + 2] = to_fp8(__half2float(v23.x));
  if (base + 3 < n) dst[base + 3] = to_fp8(__half2float(v23.y));
}

cudaError_t convert_half_to_fp8_e4m3(
    const cutlass::half_t* src,
    cutlass::float_e4m3_t* dst,
    int64_t num_elements,
    cudaStream_t stream) {
  int threads = 256;
  int64_t elems_per_block = threads * 4;
  int64_t blocks = (num_elements + elems_per_block - 1) / elems_per_block;
  half_to_fp8_e4m3_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(src, dst, num_elements);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

// ============================================================================
// INT8 QUANTIZATION UTILITIES
// ============================================================================

// Main's half_to_int8 kernel (host-side scale)
__global__ void half_to_int8_kernel(
    const cutlass::half_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t n,
    float inv_scale) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  float v = to_f32(src[idx]) * inv_scale;
  float r = nearbyintf(v);
  r = fminf(fmaxf(r, -128.0f), 127.0f);
  dst[idx] = static_cast<int8_t>(r);
}

// Main's atomicMaxFloat (CAS-based, works for all float values)
__device__ inline void atomicMaxFloat(float* addr, float val) {
  int* addr_as_i = reinterpret_cast<int*>(addr);
  int old = *addr_as_i;
  int assumed;
  int val_i = __float_as_int(val);
  while (val_i > old) {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, val_i);
  }
}

// Main's absmax_kernel (simple per-thread atomic)
__global__ void absmax_kernel(
    const cutlass::half_t* __restrict__ src,
    float* __restrict__ out,
    int64_t n) {
  float local_max = 0.0f;
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += (int64_t)blockDim.x * gridDim.x) {
    float v = fabsf(to_f32(src[idx]));
    if (v > local_max) local_max = v;
  }
  atomicMaxFloat(out, local_max);
}

// Turbo's fast atomic max for non-negative floats (uses atomicMax on uint).
__device__ inline void atomicMaxFloatNonNeg(float* addr, float val) {
  val = fmaxf(val, 0.0f);
  unsigned int* addr_as_ui = reinterpret_cast<unsigned int*>(addr);
  atomicMax(addr_as_ui, __float_as_uint(val));
}

// Turbo's warp-level reduction for absmax
__device__ __forceinline__ float warp_reduce_max(float v) {
  // Full warp mask
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(mask, v, offset);
    v = fmaxf(v, other);
  }
  return v;
}

// Turbo's optimized absmax kernel (warp + block reduction)
__global__ void absmax_block_kernel(
    const cutlass::half_t* __restrict__ src,
    float* __restrict__ out,
    int64_t n) {
  float local_max = 0.0f;
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += (int64_t)blockDim.x * gridDim.x) {
    float v = fabsf(to_f32(src[idx]));
    local_max = fmaxf(local_max, v);
  }

  // Reduce within warp
  local_max = warp_reduce_max(local_max);

  // Reduce warp maxima within block
  __shared__ float warp_max[8]; // supports up to 256 threads (8 warps)
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_max[warp] = local_max;
  __syncthreads();

  float block_max = 0.0f;
  if (warp == 0) {
    block_max = (threadIdx.x < (blockDim.x >> 5)) ? warp_max[lane] : 0.0f;
    block_max = warp_reduce_max(block_max);
    if (lane == 0) {
      atomicMaxFloatNonNeg(out, block_max);
    }
  }
}

// Main's compute_activation_scale (host-side, copies result back to host)
cudaError_t compute_activation_scale(
    const cutlass::half_t* src,
    int64_t num_elements,
    float* activation_scale_out,
    cudaStream_t stream) {
  if (activation_scale_out == nullptr || src == nullptr) {
    return cudaErrorInvalidValue;
  }
  float* d_max = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_max, sizeof(float), stream));
  CUDA_CHECK(cudaMemsetAsync(d_max, 0, sizeof(float), stream));
  int threads = 256;
  int64_t blocks = (num_elements + threads - 1) / threads;
  blocks = blocks > 1024 ? 1024 : blocks;
  absmax_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(src, d_max, num_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(activation_scale_out, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFreeAsync(d_max, stream));
  float act = (*activation_scale_out) / 127.0f;
  if (act < 1.0e-6f) act = 1.0e-6f;
  *activation_scale_out = act;
  return cudaSuccess;
}

// Turbo's compute_activation_scale_device (device-side, result stays on GPU)
cudaError_t compute_activation_scale_device(
    const cutlass::half_t* src,
    int64_t num_elements,
    float* activation_scale_out_dev,
    cudaStream_t stream) {
  if (activation_scale_out_dev == nullptr || src == nullptr) {
    return cudaErrorInvalidValue;
  }
  {
    cudaError_t e = cudaMemsetAsync(activation_scale_out_dev, 0, sizeof(float), stream);
    if (e != cudaSuccess) return e;
  }
  int threads = 256;
  int64_t blocks = (num_elements + threads - 1) / threads;
  blocks = blocks > 1024 ? 1024 : blocks;
  absmax_block_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(src, activation_scale_out_dev, num_elements);
  {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
  }
  return cudaGetLastError();
}

// Main's convert_half_to_int8 (host-side scale)
cudaError_t convert_half_to_int8(
    const cutlass::half_t* src,
    int8_t* dst,
    int64_t num_elements,
    cudaStream_t stream,
    float activation_scale) {
  int threads = 256;
  int64_t blocks = (num_elements + threads - 1) / threads;
  float inv_scale = activation_scale > 0 ? 1.0f / activation_scale : 1.0f;
  half_to_int8_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(src, dst, num_elements, inv_scale);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

// Turbo's device-scale quantization kernels
__global__ void half_to_int8_device_scale_kernel(
    const cutlass::half_t* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t n,
    const float* __restrict__ scale_dev) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  // scale_dev stores activation absmax. Convert to scale = amax/127 with clamp.
  float amax = *scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);
  float inv_scale = 127.0f / amax;
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  float v = to_f32(src[idx]) * inv_scale;
  int q = __float2int_rn(v);
  q = max(-128, min(127, q));
  dst[idx] = static_cast<int8_t>(q);
}

__global__ void half_to_int8_device_scale_kernel_vec4(
    const half* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t n,
    const float* __restrict__ scale_dev) {
  int64_t base = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (base >= n) return;
  // scale_dev stores activation absmax. Convert to inv_scale = 127/amax with clamp.
  float amax = *scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);
  float inv_scale = 127.0f / amax;

  // Vectorized path expects n % 4 == 0 and src/dst aligned by the caller.
  half2 a0 = *reinterpret_cast<const half2*>(src + base + 0);
  half2 a1 = *reinterpret_cast<const half2*>(src + base + 2);

  float2 f0 = __half22float2(a0);
  float2 f1 = __half22float2(a1);

  int q0 = __float2int_rn(f0.x * inv_scale);
  int q1 = __float2int_rn(f0.y * inv_scale);
  int q2 = __float2int_rn(f1.x * inv_scale);
  int q3 = __float2int_rn(f1.y * inv_scale);

  q0 = max(-128, min(127, q0));
  q1 = max(-128, min(127, q1));
  q2 = max(-128, min(127, q2));
  q3 = max(-128, min(127, q3));

  char4 packed;
  packed.x = (char)q0;
  packed.y = (char)q1;
  packed.z = (char)q2;
  packed.w = (char)q3;
  *reinterpret_cast<char4*>(dst + base) = packed;
}

// Turbo's convert_half_to_int8_device_scale (device pointer for scale)
cudaError_t convert_half_to_int8_device_scale(
    const cutlass::half_t* src,
    int8_t* dst,
    int64_t num_elements,
    cudaStream_t stream,
    const float* activation_scale_dev) {
  if (activation_scale_dev == nullptr) return cudaErrorInvalidValue;
  if (src == nullptr || dst == nullptr) return cudaErrorInvalidValue;
  int threads = 256;

  // Fast path: vectorize 4 elements per thread when aligned and divisible by 4.
  uintptr_t src_u = reinterpret_cast<uintptr_t>(src);
  uintptr_t dst_u = reinterpret_cast<uintptr_t>(dst);
  bool aligned4 = ((src_u & 3u) == 0u) && ((dst_u & 3u) == 0u);
  if (aligned4 && (num_elements % 4 == 0)) {
    int64_t blocks = ((num_elements / 4) + threads - 1) / threads;
    half_to_int8_device_scale_kernel_vec4<<<(unsigned int)blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src), dst, num_elements, activation_scale_dev);
    return cudaGetLastError();
  }

  // Fallback: scalar
  int64_t blocks = (num_elements + threads - 1) / threads;
  half_to_int8_device_scale_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(src, dst, num_elements, activation_scale_dev);
  return cudaGetLastError();
}

// Turbo's INT8 to half dequantization kernels (device pointer for scale)
__global__ void int8_to_half_device_scale_kernel(
    const int8_t* __restrict__ src,
    cutlass::half_t* __restrict__ dst,
    int64_t n,
    const float* __restrict__ scale_dev) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  // scale_dev stores activation absmax. Convert to scale = amax/127 with clamp.
  float amax = *scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);
  float scale = amax / 127.0f;
  float v = static_cast<float>(src[idx]) * scale;
  dst[idx] = cutlass::half_t(v);
}

__global__ void int8_to_half_device_scale_kernel_vec4(
    const int8_t* __restrict__ src,
    half* __restrict__ dst,
    int64_t n,
    const float* __restrict__ scale_dev) {
  int64_t base = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (base >= n) return;
  // scale_dev stores activation absmax. Convert to scale = amax/127 with clamp.
  float amax = *scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);
  float scale = amax / 127.0f;

  char4 packed = *reinterpret_cast<const char4*>(src + base);
  float f0 = (float)packed.x * scale;
  float f1 = (float)packed.y * scale;
  float f2 = (float)packed.z * scale;
  float f3 = (float)packed.w * scale;

  // Store as two half2.
  half2 h0 = __floats2half2_rn(f0, f1);
  half2 h1 = __floats2half2_rn(f2, f3);
  *reinterpret_cast<half2*>(dst + base + 0) = h0;
  *reinterpret_cast<half2*>(dst + base + 2) = h1;
}

cudaError_t convert_int8_to_half_device_scale(
    const int8_t* src,
    cutlass::half_t* dst,
    int64_t num_elements,
    cudaStream_t stream,
    const float* activation_scale_dev) {
  if (activation_scale_dev == nullptr) return cudaErrorInvalidValue;
  if (src == nullptr || dst == nullptr) return cudaErrorInvalidValue;
  int threads = 256;

  uintptr_t src_u = reinterpret_cast<uintptr_t>(src);
  uintptr_t dst_u = reinterpret_cast<uintptr_t>(dst);
  bool aligned4 = ((src_u & 3u) == 0u) && ((dst_u & 3u) == 0u);
  if (aligned4 && (num_elements % 4 == 0)) {
    int64_t blocks = ((num_elements / 4) + threads - 1) / threads;
    int8_to_half_device_scale_kernel_vec4<<<(unsigned int)blocks, threads, 0, stream>>>(
        src, reinterpret_cast<half*>(dst), num_elements, activation_scale_dev);
    return cudaGetLastError();
  }

  int64_t blocks = (num_elements + threads - 1) / threads;
  int8_to_half_device_scale_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(src, dst, num_elements, activation_scale_dev);
  return cudaGetLastError();
}

// ============================================================================
// INT8/FP16 POST-OPS (column-wise scale/bias)
// ============================================================================

// Column-wise scale + optional bias on row-major [N, out_features] - half2 vectorized
__global__ void col_scale_bias_kernel(
    cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    float scale_mul,
    int out_features,
    long long total_elems) {
  long long idx2 = (long long)blockIdx.x * blockDim.x + threadIdx.x;  // half2 index
  long long idx = idx2 * 2;
  if (idx >= total_elems) return;
  half2* data_h2 = reinterpret_cast<half2*>(data);
  const half2* scale_h2 = reinterpret_cast<const half2*>(scale);
  int col = (int)(idx % out_features);
  int col2 = col / 2;
  half2 v = data_h2[idx2];
  half2 s = scale_h2[col2];
  float v0 = __half2float(v.x) * (__half2float(s.x) * scale_mul);
  float v1 = __half2float(v.y) * (__half2float(s.y) * scale_mul);
  if (bias) {
    const half2* bias_h2 = reinterpret_cast<const half2*>(bias);
    half2 b = bias_h2[col2];
    v0 += __half2float(b.x);
    v1 += __half2float(b.y);
  }
  data_h2[idx2] = half2{__float2half(v0), __float2half(v1)};
}

__global__ void col_scale_bias_to_bf16_kernel(
    const cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    cutlass::bfloat16_t* __restrict__ output,
    float scale_mul,
    int out_features,
    long long total_elems) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  float x = to_f32(data[idx]) * (to_f32(scale[col]) * scale_mul);
  if (bias) {
    x += to_f32(bias[col]);
  }
  output[idx] = cutlass::bfloat16_t(x);
}

__global__ void col_scale_bias_to_bf16_vec2_kernel(
    const cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    cutlass::bfloat16_t* __restrict__ output,
    float scale_mul,
    int out_features,
    long long total_elems) {
  long long idx2 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long idx = idx2 * 2;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  int col2 = col / 2;
  const half2* data_h2 = reinterpret_cast<const half2*>(data);
  const half2* scale_h2 = reinterpret_cast<const half2*>(scale);
  half2 v = data_h2[idx2];
  half2 s = scale_h2[col2];
  float x0 = __half2float(v.x) * (__half2float(s.x) * scale_mul);
  float x1 = __half2float(v.y) * (__half2float(s.y) * scale_mul);
  if (bias) {
    const half2* bias_h2 = reinterpret_cast<const half2*>(bias);
    half2 b = bias_h2[col2];
    x0 += __half2float(b.x);
    x1 += __half2float(b.y);
  }
  reinterpret_cast<__nv_bfloat162*>(output)[idx2] =
      __floats2bfloat162_rn(x0, x1);
}

// half2-vectorized: col_scale + bias + GELU on row-major [N, out_features]
__global__ void col_scale_bias_gelu_kernel(
    cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    float scale_mul,
    int out_features,
    long long total_elems) {
  long long idx2 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long idx = idx2 * 2;
  if (idx >= total_elems) return;
  half2* data_h2 = reinterpret_cast<half2*>(data);
  const half2* scale_h2 = reinterpret_cast<const half2*>(scale);
  int col = (int)(idx % out_features);
  int col2 = col / 2;
  half2 v = data_h2[idx2];
  half2 s = scale_h2[col2];
  float x0 = __half2float(v.x) * (__half2float(s.x) * scale_mul);
  float x1 = __half2float(v.y) * (__half2float(s.y) * scale_mul);
  if (bias) {
    const half2* bias_h2 = reinterpret_cast<const half2*>(bias);
    half2 b = bias_h2[col2];
    x0 += __half2float(b.x);
    x1 += __half2float(b.y);
  }
  // GELU tanh approximation
  float c = 0.044715f;
  float y0 = 0.7978845608028654f * (x0 + c * x0 * x0 * x0);
  float y1 = 0.7978845608028654f * (x1 + c * x1 * x1 * x1);
  float g0 = 0.5f * x0 * (1.0f + tanhf(y0));
  float g1 = 0.5f * x1 * (1.0f + tanhf(y1));
  data_h2[idx2] = half2{__float2half(g0), __float2half(g1)};
}

// Fused: col_scale + bias + FP8 conversion for producers that feed another
// quantized layer directly.
__global__ void col_scale_bias_to_fp8_kernel(
    const cutlass::half_t* __restrict__ data,
    cutlass::float_e4m3_t* __restrict__ fp8_out,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    float scale_mul,
    float inv_output_scale,
    int out_features,
    long long total_elems) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  float x = to_f32(data[idx]) * (to_f32(scale[col]) * scale_mul);
  if (bias) {
    x += to_f32(bias[col]);
  }
  fp8_out[idx] = to_fp8(x * inv_output_scale);
}

// Fused: col_scale + bias + GELU + FP8 conversion -- eliminates intermediate FP16 write/read
// Reads half GEMM output, applies scale+bias+GELU, writes FP8 directly
__global__ void col_scale_bias_gelu_to_fp8_kernel(
    const cutlass::half_t* __restrict__ data,        // input: GEMM output [N, out_features]
    cutlass::float_e4m3_t* __restrict__ fp8_out,     // output: FP8 [N, out_features]
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    float scale_mul,
    float inv_output_scale,
    int out_features,
    long long total_elems) {
  long long idx2 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long idx = idx2 * 2;
  if (idx >= total_elems) return;
  const half2* data_h2 = reinterpret_cast<const half2*>(data);
  const half2* scale_h2 = reinterpret_cast<const half2*>(scale);
  int col = (int)(idx % out_features);
  int col2 = col / 2;
  half2 v = data_h2[idx2];
  half2 s = scale_h2[col2];
  float x0 = __half2float(v.x) * (__half2float(s.x) * scale_mul);
  float x1 = __half2float(v.y) * (__half2float(s.y) * scale_mul);
  if (bias) {
    const half2* bias_h2 = reinterpret_cast<const half2*>(bias);
    half2 b = bias_h2[col2];
    x0 += __half2float(b.x);
    x1 += __half2float(b.y);
  }
  // GELU tanh approximation
  float c = 0.044715f;
  float y0 = 0.7978845608028654f * (x0 + c * x0 * x0 * x0);
  float y1 = 0.7978845608028654f * (x1 + c * x1 * x1 * x1);
  float g0 = 0.5f * x0 * (1.0f + tanhf(y0));
  float g1 = 0.5f * x1 * (1.0f + tanhf(y1));
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  fp8_out[idx]     = to_fp8(g0 * inv_output_scale);
  fp8_out[idx + 1] = to_fp8(g1 * inv_output_scale);
}

__global__ void col_scale_bias_residual_kernel(
    const cutlass::half_t* __restrict__ data,
    cutlass::half_t* __restrict__ residual_inout,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    float scale_mul,
    int out_features,
    long long total_elems) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  float acc = __half2float(reinterpret_cast<const half*>(data)[idx]) *
              (__half2float(reinterpret_cast<const half*>(scale)[col]) * scale_mul);
  if (bias) {
    acc += __half2float(reinterpret_cast<const half*>(bias)[col]);
  }
  float res = __half2float(reinterpret_cast<half*>(residual_inout)[idx]);
  reinterpret_cast<half*>(residual_inout)[idx] = __float2half(res + acc);
}

__global__ void col_scale_residual_bf16_kernel(
    const cutlass::half_t* __restrict__ data,
    cutlass::bfloat16_t* __restrict__ residual_inout,
    const cutlass::half_t* __restrict__ scale,
    float scale_mul,
    int out_features,
    long long total_elems) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  cutlass::NumericConverter<float, cutlass::half_t> half_to_f32;
  cutlass::NumericConverter<float, cutlass::bfloat16_t> bf16_to_f32;
  float acc = half_to_f32(data[idx]) * (half_to_f32(scale[col]) * scale_mul);
  float res = bf16_to_f32(residual_inout[idx]);
  residual_inout[idx] = cutlass::bfloat16_t(res + acc);
}

__global__ void col_scale_residual_bf16_vec2_kernel(
    const cutlass::half_t* __restrict__ data,
    cutlass::bfloat16_t* __restrict__ residual_inout,
    const cutlass::half_t* __restrict__ scale,
    float scale_mul,
    int out_features,
    long long total_elems) {
  long long idx2 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long idx = idx2 * 2;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  int col2 = col / 2;
  const half2* data_h2 = reinterpret_cast<const half2*>(data);
  const half2* scale_h2 = reinterpret_cast<const half2*>(scale);
  half2 v = data_h2[idx2];
  half2 s = scale_h2[col2];
  float x0 = __half2float(v.x) * (__half2float(s.x) * scale_mul);
  float x1 = __half2float(v.y) * (__half2float(s.y) * scale_mul);
  __nv_bfloat162 r = reinterpret_cast<const __nv_bfloat162*>(residual_inout)[idx2];
  float2 rf = __bfloat1622float2(r);
  reinterpret_cast<__nv_bfloat162*>(residual_inout)[idx2] =
      __floats2bfloat162_rn(rf.x + x0, rf.y + x1);
}

static bool ptr_aligned_4(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

cudaError_t apply_col_scale_bias(
    cutlass::half_t* data,
    const cutlass::half_t* scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul) {
  if (scale == nullptr) return cudaErrorInvalidValue;
  long long total = (long long)N * out_features;
  long long total_h2 = total / 2;  // half2 elements
  int threads = 256;
  long long blocks = (total_h2 + threads - 1) / threads;
  col_scale_bias_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(data, scale, bias, scale_mul, out_features, total);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

cudaError_t apply_col_scale_bias_gelu(
    cutlass::half_t* data,
    const cutlass::half_t* scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul) {
  if (scale == nullptr) return cudaErrorInvalidValue;
  long long total = (long long)N * out_features;
  long long total_h2 = total / 2;  // half2 elements
  int threads = 256;
  long long blocks = (total_h2 + threads - 1) / threads;
  col_scale_bias_gelu_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(data, scale, bias, scale_mul, out_features, total);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

cudaError_t apply_col_scale_bias_to_fp8(
    const cutlass::half_t* data,
    cutlass::float_e4m3_t* fp8_out,
    const cutlass::half_t* scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul,
    float output_scale) {
  if (scale == nullptr || data == nullptr || fp8_out == nullptr) return cudaErrorInvalidValue;
  if (!(output_scale > 0.0f) || !std::isfinite(output_scale)) return cudaErrorInvalidValue;
  long long total = (long long)N * out_features;
  int threads = 256;
  long long blocks = (total + threads - 1) / threads;
  float inv_output_scale = 1.0f / output_scale;
  col_scale_bias_to_fp8_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
      data, fp8_out, scale, bias, scale_mul, inv_output_scale, out_features, total);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

cudaError_t apply_col_scale_bias_gelu_to_fp8(
    const cutlass::half_t* data,
    cutlass::float_e4m3_t* fp8_out,
    const cutlass::half_t* scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul,
    float output_scale) {
  if (scale == nullptr) return cudaErrorInvalidValue;
  if (!(output_scale > 0.0f) || !std::isfinite(output_scale)) return cudaErrorInvalidValue;
  long long total = (long long)N * out_features;
  long long total_h2 = total / 2;
  int threads = 256;
  long long blocks = (total_h2 + threads - 1) / threads;
  float inv_output_scale = 1.0f / output_scale;
  col_scale_bias_gelu_to_fp8_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
      data, fp8_out, scale, bias, scale_mul, inv_output_scale, out_features, total);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

// GELU + bias kernel (no scaling) - used for per-block INT8 quantization
__global__ void gelu_bias_kernel(
    cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ bias,
    int out_features,
    long long total_elems) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  float x = __half2float(reinterpret_cast<half*>(data)[idx]);
  if (bias) {
    x += __half2float(reinterpret_cast<const half*>(bias)[col]);
  }
  // GELU tanh approximation
  float c = 0.044715f;
  float y = 0.7978845608028654f * (x + c * x * x * x);
  float gelu = 0.5f * x * (1.0f + tanhf(y));
  // Clamp to prevent half overflow
  gelu = fminf(fmaxf(gelu, -65504.0f), 65504.0f);
  reinterpret_cast<half*>(data)[idx] = __float2half(gelu);
}

cudaError_t apply_gelu_bias(
    cutlass::half_t* data,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream) {
  long long total = (long long)N * out_features;
  int threads = 256;
  long long blocks = (total + threads - 1) / threads;
  gelu_bias_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(data, bias, out_features, total);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

cudaError_t apply_col_scale_bias_residual(
    const cutlass::half_t* data,
    cutlass::half_t* residual_inout,
    const cutlass::half_t* scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    float scale_mul) {
  if (scale == nullptr || residual_inout == nullptr) return cudaErrorInvalidValue;
  long long total = (long long)N * out_features;
  int threads = 256;
  long long blocks = (total + threads - 1) / threads;
  col_scale_bias_residual_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(data, residual_inout, scale, bias, scale_mul, out_features, total);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

// Simple in-place add kernel: dest += src
__global__ void add_inplace_kernel(
    cutlass::half_t* __restrict__ dest,
    const cutlass::half_t* __restrict__ src,
    long long total_elems) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  float d = __half2float(reinterpret_cast<half*>(dest)[idx]);
  float s = __half2float(reinterpret_cast<const half*>(src)[idx]);
  float result = d + s;
  // Clamp to prevent half overflow
  result = fminf(fmaxf(result, -65504.0f), 65504.0f);
  reinterpret_cast<half*>(dest)[idx] = __float2half(result);
}

cudaError_t add_inplace_half_rr(
    cutlass::half_t* dest,
    const cutlass::half_t* src,
    long long numel,
    cudaStream_t stream) {
  if (dest == nullptr || src == nullptr) return cudaErrorInvalidValue;
  int threads = 256;
  long long blocks = (numel + threads - 1) / threads;
  add_inplace_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(dest, src, numel);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

// ============================================================================
// INT8 SCALAR-AWARE POST-OPS
// ============================================================================

__global__ void col_scale_bias_scalar_kernel(
    cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ weight_scale,
    const cutlass::half_t* __restrict__ bias,
    int out_features,
    long long total_elems,
    const float* __restrict__ act_scale_dev) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  // act_scale_dev stores activation absmax. Convert to scale = amax/127 with clamp.
  // NOTE: GEMM output is scaled by 1/127^2 to prevent half overflow, so we need
  // to compensate by multiplying by 127^2 here. Combined with act_scale and
  // weight_scale (both are amax/127), we get: 127^2 * (amax_a/127) * (amax_b/127) = amax_a * amax_b.
  float amax = *act_scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);
  // Multiply by amax directly (not amax/127) to compensate for GEMM's 1/127^2 scaling
  float ws = weight_scale ? __half2float(reinterpret_cast<const half*>(weight_scale)[col]) * 127.0f : 1.0f;
  float b = bias ? __half2float(reinterpret_cast<const half*>(bias)[col]) : 0.0f;
  float v = __half2float(reinterpret_cast<half*>(data)[idx]);
  float result = v * (amax * ws) + b;
  // Clamp to half range to prevent overflow
  result = fminf(fmaxf(result, -65504.0f), 65504.0f);
  reinterpret_cast<half*>(data)[idx] = __float2half(result);
}

__global__ void col_scale_bias_scalar_kernel_h2(
    cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ weight_scale,
    const cutlass::half_t* __restrict__ bias,
    int out_features,
    long long total_pairs,
    const float* __restrict__ act_scale_dev) {
  long long pair_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) return;

  // out_features is even for this kernel (enforced by caller), so half2 pairs never cross rows.
  int col0 = int((pair_idx * 2) % out_features);

  // NOTE: GEMM output is scaled by 1/127^2 to prevent half overflow, so we need
  // to compensate by multiplying by 127^2 here. Combined with act_scale and
  // weight_scale (both are amax/127), we get: 127^2 * (amax_a/127) * (amax_b/127) = amax_a * amax_b.
  float amax = *act_scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);
  // Use amax directly (not amax/127) to compensate for GEMM's 1/127^2 scaling

  half2 v2 = reinterpret_cast<half2*>(data)[pair_idx];
  float2 vf = __half22float2(v2);

  float2 wf;
  if (weight_scale) {
    half2 w2 = *reinterpret_cast<const half2*>(reinterpret_cast<const half*>(weight_scale) + col0);
    wf = __half22float2(w2);
    // Multiply weight_scale by 127 to compensate for GEMM's 1/127^2 scaling
    wf.x *= 127.0f;
    wf.y *= 127.0f;
  } else {
    wf.x = 1.0f; wf.y = 1.0f;
  }

  float2 bf;
  if (bias) {
    half2 b2 = *reinterpret_cast<const half2*>(reinterpret_cast<const half*>(bias) + col0);
    bf = __half22float2(b2);
  } else {
    bf.x = 0.0f; bf.y = 0.0f;
  }

  vf.x = vf.x * (amax * wf.x) + bf.x;
  vf.y = vf.y * (amax * wf.y) + bf.y;
  // Clamp to half range to prevent overflow
  vf.x = fminf(fmaxf(vf.x, -65504.0f), 65504.0f);
  vf.y = fminf(fmaxf(vf.y, -65504.0f), 65504.0f);

  reinterpret_cast<half2*>(data)[pair_idx] = __floats2half2_rn(vf.x, vf.y);
}

cudaError_t apply_col_scale_bias_scalar(
    cutlass::half_t* data,
    const cutlass::half_t* weight_scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    const float* act_scale_dev) {
  if (data == nullptr || act_scale_dev == nullptr) return cudaErrorInvalidValue;
  long long total = (long long)N * out_features;
  int threads = 256;

  // Fast path: half2 on data/scale/bias when out_features is even (no row-crossing) and pointers are 4-byte aligned.
  bool even_out = ((out_features & 1) == 0);
  uintptr_t data_u = reinterpret_cast<uintptr_t>(data);
  bool aligned = ((data_u & 3u) == 0u);
  if (even_out && aligned) {
    long long total_pairs = total / 2;
    long long blocks = (total_pairs + threads - 1) / threads;
    col_scale_bias_scalar_kernel_h2<<<(unsigned int)blocks, threads, 0, stream>>>(
        data, weight_scale, bias, out_features, total_pairs, act_scale_dev);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
  }

  long long blocks = (total + threads - 1) / threads;
  col_scale_bias_scalar_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
      data, weight_scale, bias, out_features, total, act_scale_dev);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

__global__ void col_scale_bias_gelu_scalar_kernel(
    cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ weight_scale,
    const cutlass::half_t* __restrict__ bias,
    int out_features,
    long long total_elems,
    const float* __restrict__ act_scale_dev) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  // NOTE: GEMM output is scaled by 1/127^2 to prevent half overflow.
  // Compensate by multiplying weight_scale by 127 and using amax directly.
  float amax = *act_scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);
  float ws = weight_scale ? __half2float(reinterpret_cast<const half*>(weight_scale)[col]) * 127.0f : 1.0f;
  float b = bias ? __half2float(reinterpret_cast<const half*>(bias)[col]) : 0.0f;
  float v = __half2float(reinterpret_cast<half*>(data)[idx]);
  float x = v * (amax * ws) + b;
  float c = 0.044715f;
  float y = 0.7978845608028654f * (x + c * x * x * x);
  float gelu = 0.5f * x * (1.0f + tanhf(y));
  // Clamp to half range to prevent overflow
  gelu = fminf(fmaxf(gelu, -65504.0f), 65504.0f);
  reinterpret_cast<half*>(data)[idx] = __float2half(gelu);
}

__global__ void col_scale_bias_gelu_scalar_kernel_h2(
    cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ weight_scale,
    const cutlass::half_t* __restrict__ bias,
    int out_features,
    long long total_pairs,
    const float* __restrict__ act_scale_dev) {
  long long pair_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) return;
  int col0 = int((pair_idx * 2) % out_features);

  // NOTE: GEMM output is scaled by 1/127^2 to prevent half overflow.
  // Compensate by multiplying weight_scale by 127 and using amax directly.
  float amax = *act_scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);

  half2 v2 = reinterpret_cast<half2*>(data)[pair_idx];
  float2 vf = __half22float2(v2);

  float2 wf;
  if (weight_scale) {
    half2 w2 = *reinterpret_cast<const half2*>(reinterpret_cast<const half*>(weight_scale) + col0);
    wf = __half22float2(w2);
    wf.x *= 127.0f;
    wf.y *= 127.0f;
  } else {
    wf.x = 1.0f; wf.y = 1.0f;
  }

  float2 bf;
  if (bias) {
    half2 b2 = *reinterpret_cast<const half2*>(reinterpret_cast<const half*>(bias) + col0);
    bf = __half22float2(b2);
  } else {
    bf.x = 0.0f; bf.y = 0.0f;
  }

  // Apply scale+bias then GELU elementwise.
  float x0 = vf.x * (amax * wf.x) + bf.x;
  float x1 = vf.y * (amax * wf.y) + bf.y;

  float c = 0.044715f;
  float y0 = 0.7978845608028654f * (x0 + c * x0 * x0 * x0);
  float y1 = 0.7978845608028654f * (x1 + c * x1 * x1 * x1);
  float g0 = 0.5f * x0 * (1.0f + tanhf(y0));
  float g1 = 0.5f * x1 * (1.0f + tanhf(y1));
  // Clamp to half range to prevent overflow
  g0 = fminf(fmaxf(g0, -65504.0f), 65504.0f);
  g1 = fminf(fmaxf(g1, -65504.0f), 65504.0f);

  reinterpret_cast<half2*>(data)[pair_idx] = __floats2half2_rn(g0, g1);
}

cudaError_t apply_col_scale_bias_gelu_scalar(
    cutlass::half_t* data,
    const cutlass::half_t* weight_scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    const float* act_scale_dev) {
  if (data == nullptr || act_scale_dev == nullptr) return cudaErrorInvalidValue;
  long long total = (long long)N * out_features;
  int threads = 256;

  bool even_out = ((out_features & 1) == 0);
  uintptr_t data_u = reinterpret_cast<uintptr_t>(data);
  bool aligned = ((data_u & 3u) == 0u);
  if (even_out && aligned) {
    long long total_pairs = total / 2;
    long long blocks = (total_pairs + threads - 1) / threads;
    col_scale_bias_gelu_scalar_kernel_h2<<<(unsigned int)blocks, threads, 0, stream>>>(
        data, weight_scale, bias, out_features, total_pairs, act_scale_dev);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
  }

  long long blocks = (total + threads - 1) / threads;
  col_scale_bias_gelu_scalar_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
      data, weight_scale, bias, out_features, total, act_scale_dev);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

__global__ void col_scale_bias_residual_scalar_kernel(
    const cutlass::half_t* __restrict__ data,
    cutlass::half_t* __restrict__ residual_inout,
    const cutlass::half_t* __restrict__ weight_scale,
    const cutlass::half_t* __restrict__ bias,
    int out_features,
    long long total_elems,
    const float* __restrict__ act_scale_dev) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = (int)(idx % out_features);
  // NOTE: GEMM output is scaled by 1/127^2 to prevent half overflow.
  // Compensate by multiplying weight_scale by 127 and using amax directly.
  float amax = *act_scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);
  float ws = weight_scale ? __half2float(reinterpret_cast<const half*>(weight_scale)[col]) * 127.0f : 1.0f;
  float b = bias ? __half2float(reinterpret_cast<const half*>(bias)[col]) : 0.0f;
  float v = __half2float(reinterpret_cast<const half*>(data)[idx]);
  float acc = v * (amax * ws) + b;
  float res = __half2float(reinterpret_cast<half*>(residual_inout)[idx]);
  float result = res + acc;
  // Clamp to half range to prevent overflow
  result = fminf(fmaxf(result, -65504.0f), 65504.0f);
  reinterpret_cast<half*>(residual_inout)[idx] = __float2half(result);
}

__global__ void col_scale_bias_residual_scalar_kernel_h2(
    const cutlass::half_t* __restrict__ data,
    cutlass::half_t* __restrict__ residual_inout,
    const cutlass::half_t* __restrict__ weight_scale,
    const cutlass::half_t* __restrict__ bias,
    int out_features,
    long long total_pairs,
    const float* __restrict__ act_scale_dev) {
  long long pair_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) return;
  int col0 = int((pair_idx * 2) % out_features);

  // NOTE: GEMM output is scaled by 1/127^2 to prevent half overflow.
  // Compensate by multiplying weight_scale by 127 and using amax directly.
  float amax = *act_scale_dev;
  amax = fmaxf(amax, 127.0f * 1.0e-6f);

  half2 v2 = reinterpret_cast<const half2*>(data)[pair_idx];
  float2 vf = __half22float2(v2);

  half2 r2 = reinterpret_cast<half2*>(residual_inout)[pair_idx];
  float2 rf = __half22float2(r2);

  float2 wf;
  if (weight_scale) {
    half2 w2 = *reinterpret_cast<const half2*>(reinterpret_cast<const half*>(weight_scale) + col0);
    wf = __half22float2(w2);
    wf.x *= 127.0f;
    wf.y *= 127.0f;
  } else {
    wf.x = 1.0f; wf.y = 1.0f;
  }

  float2 bf;
  if (bias) {
    half2 b2 = *reinterpret_cast<const half2*>(reinterpret_cast<const half*>(bias) + col0);
    bf = __half22float2(b2);
  } else {
    bf.x = 0.0f; bf.y = 0.0f;
  }

  rf.x = rf.x + (vf.x * (amax * wf.x) + bf.x);
  rf.y = rf.y + (vf.y * (amax * wf.y) + bf.y);
  // Clamp to half range to prevent overflow
  rf.x = fminf(fmaxf(rf.x, -65504.0f), 65504.0f);
  rf.y = fminf(fmaxf(rf.y, -65504.0f), 65504.0f);
  reinterpret_cast<half2*>(residual_inout)[pair_idx] = __floats2half2_rn(rf.x, rf.y);
}

cudaError_t apply_col_scale_bias_residual_scalar(
    const cutlass::half_t* data,
    cutlass::half_t* residual_inout,
    const cutlass::half_t* weight_scale,
    const cutlass::half_t* bias,
    int N, int out_features,
    cudaStream_t stream,
    const float* act_scale_dev) {
  if (data == nullptr || residual_inout == nullptr || act_scale_dev == nullptr) return cudaErrorInvalidValue;
  long long total = (long long)N * out_features;
  int threads = 256;

  bool even_out = ((out_features & 1) == 0);
  uintptr_t data_u = reinterpret_cast<uintptr_t>(data);
  uintptr_t res_u = reinterpret_cast<uintptr_t>(residual_inout);
  bool aligned = ((data_u & 3u) == 0u) && ((res_u & 3u) == 0u);
  if (even_out && aligned) {
    long long total_pairs = total / 2;
    long long blocks = (total_pairs + threads - 1) / threads;
    col_scale_bias_residual_scalar_kernel_h2<<<(unsigned int)blocks, threads, 0, stream>>>(
        data, residual_inout, weight_scale, bias, out_features, total_pairs, act_scale_dev);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
  }

  long long blocks = (total + threads - 1) / threads;
  col_scale_bias_residual_scalar_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
      data, residual_inout, weight_scale, bias, out_features, total, act_scale_dev);
  CUDA_CHECK(cudaGetLastError());
  return cudaSuccess;
}

// ============================================================================
// INT8 FUSED GEMM WITH SCALE+BIAS VIA CUSTOM EPILOGUE
// ============================================================================

namespace {

// Epilogue output operator for broadcast GEMM that applies:
//   out = (act * weight_scale) * accum + beta * C
// where:
// - act is derived from a device scalar holding activation absmax (amax): act = max(amax/127, 1e-6)
// - weight_scale is provided via the broadcast vector V
// - C is optionally used for bias via stride trick (ldc=0 broadcasts bias)
template <typename ElementOutput_, int ElementsPerAccess, typename ElementAccumulator_, typename ElementCompute_>
class LinearCombinationActScaleAmaxPtrTimesVectorPlusC {
public:
  using ElementOutput = ElementOutput_;
  using ElementD = ElementOutput;
  using ElementC = cutlass::half_t;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;

  // Broadcast vector element (weight_scale)
  using ElementVector = cutlass::half_t;

  // Secondary tensor type unused (required by DefaultEpilogueWithBroadcastTensorOp)
  using ElementZ = ElementOutput;
  using ElementT = ElementOutput;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;
  // EpilogueWithBroadcast has separate implementations for single-source and dual-source.
  // We only use a single source tensor (C) plus one broadcast vector (V).
  static bool const kIsSingleSource = true;
  static bool const kStoreZ = true;
  static bool const kStoreT = false;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
  using FragmentVector = cutlass::Array<ElementVector, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;

  struct Params {
    ElementCompute beta;             // scales source tensor C (bias)
    ElementCompute const* amax_ptr;  // device pointer to activation absmax (amax)

    CUTLASS_HOST_DEVICE
    Params() : beta(ElementCompute(0)), amax_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute beta_, ElementCompute const* amax_ptr_)
        : beta(beta_), amax_ptr(amax_ptr_) {}
  };

private:
  ElementCompute beta_;
  ElementCompute const* amax_ptr_;

public:
  CUTLASS_HOST_DEVICE
  LinearCombinationActScaleAmaxPtrTimesVectorPlusC(Params const& params)
      : beta_(params.beta), amax_ptr_(params.amax_ptr) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return beta_ != ElementCompute(0); }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& frag_Z,
      FragmentT&,
      FragmentAccumulator const& AB,
      FragmentC const& frag_C,
      FragmentCompute const& V) const {
    // amax_ptr_ points to device scalar, but CUTLASS will dereference it on device.
    ElementCompute amax = (amax_ptr_ ? *amax_ptr_ : ElementCompute(0));
    amax = amax < ElementCompute(127.0f * 1.0e-6f) ? ElementCompute(127.0f * 1.0e-6f) : amax;
    // The INT8 GEMM accumulator is A_int8 @ B_int8, where:
    //   A_int8 = A_half * 127 / amax_a, B_int8 = B_half * 127 / amax_b
    // So: A_half @ B_half = accum * amax_a * amax_b / 127^2 = accum * amax * V / 127
    // where V = weight_scale = amax_b / 127
    //
    // IMPORTANT: Scale the accumulator FIRST to prevent overflow in intermediate results.
    // The accumulator can be very large (up to 127*127*K ≈ 144M for K=8960).
    // Computing (accum * amax * V) before dividing would overflow.
    // Instead: (accum / 127) * amax * V keeps values in range.
    constexpr ElementCompute kInv127 = ElementCompute(1.0f / 127.0f);

    FragmentCompute tmp_Accum =
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C =
        cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);

    FragmentCompute result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      // V[i] is weight_scale = amax_b / 127
      // Scale accumulator first to prevent overflow: (accum / 127) * amax * V
      ElementCompute scaled_accum = tmp_Accum[i] * kInv127;
      ElementCompute y = (scaled_accum * amax * V[i]) + (beta_ * tmp_C[i]);
      result[i] = y;
    }

    cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result);
  }

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& frag_Z,
      FragmentT& frag_T,
      FragmentAccumulator const& AB,
      FragmentCompute const& V) const {
    // No source C
    FragmentC zeros;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      zeros[i] = ElementC(0);
    }
    (*this)(frag_Z, frag_T, AB, zeros, V);
  }
};

} // namespace

cudaError_t cutlass_linear_layer_rcr_int8_scale_bias_dev_amax(
    const int8_t* input_row,
    const int8_t* weight_int8,
    const cutlass::half_t* weight_scale,
    const cutlass::half_t* bias,
    cutlass::half_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream,
    const float* activation_amax_dev) {

  if (input_row == nullptr || weight_int8 == nullptr || weight_scale == nullptr || output_row == nullptr ||
      activation_amax_dev == nullptr) {
    return cudaErrorInvalidValue;
  }

  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  constexpr int kElementsPerAccess = 8;

  using EpilogueOp = LinearCombinationActScaleAmaxPtrTimesVectorPlusC<
      ElementOutput, kElementsPerAccess, ElementAccumulator, ElementCompute>;

  using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;

  using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
      ElementInputA, LayoutA,
      ElementInputB, LayoutB,
      ElementOutput, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 64>,
      DefaultConfig::WarpShape,
      DefaultConfig::InstructionShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      DefaultConfig::kStages,
      DefaultConfig::kAlignmentA,
      DefaultConfig::kAlignmentB>;

  Gemm gemm_op;

  // Bias via stride trick: C points to bias vector, ldc=0 broadcasts across rows.
  const cutlass::half_t* c_ptr = bias ? bias : output_row;
  int64_t ldc = bias ? int64_t(0) : int64_t(out_features);
  float beta = bias ? 1.0f : 0.0f;

  typename EpilogueOp::Params epilogue_params(beta, activation_amax_dev);

  Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, out_features, in_features},
      1,
      epilogue_params,
      input_row,
      weight_int8,
      c_ptr,
      output_row,
      const_cast<cutlass::half_t*>(weight_scale),
      nullptr,
      int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
      int64_t(in_features), int64_t(in_features),
      ldc,
      int64_t(out_features),
      int64_t(0),  // ldr=0 broadcasts weight_scale across rows
      int64_t(0));

  cutlass::Status status = gemm_op.initialize(args, nullptr);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  status = gemm_op(stream);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

// FP8 A/B with fused residual + optional bias addition, Row-major output (half)

// INT8 A/B with fused residual + optional bias addition, Row-major output (half)
cudaError_t cutlass_linear_layer_rcr_int8_fused_residual(
    const int8_t* input_row,              // Row-major [N, in_features] INT8
    const int8_t* weight_int8,            // Column-major [in_features, out_features] INT8
    const cutlass::half_t* bias,          // [out_features] half or nullptr
    cutlass::half_t* residual_inout,      // Row-major [N, out_features] half
    int N, int in_features, int out_features,
    cudaStream_t stream,
    float activation_scale) {

    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    constexpr int kElementsPerAccess = 8;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
        ElementOutput, ElementAccumulator, ElementCompute, ElementOutput,
        kElementsPerAccess,
        cutlass::epilogue::thread::Identity,
        cutlass::plus,
        cutlass::epilogue::thread::Identity
    >;

    using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ElementInputA, ElementInputB, ElementOutput,
        ElementAccumulator>;

    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        DefaultConfig::WarpShape,
        DefaultConfig::InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        DefaultConfig::kStages,
        DefaultConfig::kAlignmentA,
        DefaultConfig::kAlignmentB
    >;

    Gemm gemm_op;

    Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, out_features, in_features},
        1,
        {ElementCompute(activation_scale), ElementCompute(1.0f)},
        input_row,
        weight_int8,
        residual_inout,
        residual_inout,
        const_cast<cutlass::half_t*>(bias),
        nullptr,
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        int64_t(in_features), int64_t(in_features),
        int64_t(out_features),
        int64_t(out_features),
        int64_t(0),
        int64_t(0)
    );

    cutlass::Status status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

    return cudaSuccess;
}

// Row-major A/B/C path with stride-trick bias fusion
cudaError_t cutlass_linear_layer_rrr(
    const half* input_row,         // Row-major [N, in_features]
    const half* weight_row,        // Row-major [in_features, out_features]
    const half* bias,              // [out_features] or nullptr
    half* output_row,              // Row-major [N, out_features]
    int N, int in_features, int out_features,
    cudaStream_t stream) {
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    const cutlass::half_t* c_ptr = bias ? reinterpret_cast<const cutlass::half_t*>(bias) : reinterpret_cast<const cutlass::half_t*>(output_row);
    int ldc = bias ? 0 : out_features;
    cutlass::half_t beta = bias ? cutlass::half_t(1.0f) : cutlass::half_t(0.0f);

    const cutlass::half_t* input_ptr = reinterpret_cast<const cutlass::half_t*>(input_row);
    const cutlass::half_t* weight_ptr = reinterpret_cast<const cutlass::half_t*>(weight_row);
    cutlass::half_t* output_ptr = reinterpret_cast<cutlass::half_t*>(output_row);

    cutlass::Status status;

    // Choose tile size based on M dimension for better occupancy
    // Threshold is configurable via set_gemm_tile_threshold()
    if (N < g_gemm_tile_threshold) {
        // Small-M: 128x128x32 for reasonable tile coverage
        using Gemm = cutlass::gemm::device::Gemm<
            ElementInputA, LayoutA, ElementInputB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>>;
        Gemm gemm_op;
        Gemm::Arguments args({N, out_features, in_features}, {input_ptr, in_features}, {weight_ptr, out_features},
                             {c_ptr, ldc}, {output_ptr, out_features}, {cutlass::half_t(1.0f), beta});
        status = gemm_op.initialize(args, nullptr);
        if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
        status = gemm_op(stream);
    } else {
        // Large-M: 256x128x32 matches PyTorch/cuBLAS optimal configuration
        // This produces 4x fewer threadblocks than 128x64x64, improving occupancy
        using Gemm = cutlass::gemm::device::Gemm<
            ElementInputA, LayoutA, ElementInputB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<256, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>>;
        Gemm gemm_op;
        Gemm::Arguments args({N, out_features, in_features}, {input_ptr, in_features}, {weight_ptr, out_features},
                             {c_ptr, ldc}, {output_ptr, out_features}, {cutlass::half_t(1.0f), beta});
        status = gemm_op.initialize(args, nullptr);
        if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
        status = gemm_op(stream);
    }

    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    return cudaSuccess;
}

// Row-major GEMM with explicit output leading dimension (row-major stride)
cudaError_t cutlass_linear_layer_rrr_strided(
    const half* input_row,         // Row-major [N, in_features]
    const half* weight_row,        // Row-major [in_features, out_features]
    const half* bias,              // [out_features] or nullptr
    half* output_row,              // Row-major with leading dim ld_out
    int N, int in_features, int out_features,
    int ld_out,
    cudaStream_t stream) {
    if (N <= 0) return cudaSuccess;
    if (ld_out < out_features) return cudaErrorInvalidValue;

    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    const cutlass::half_t* c_ptr = bias ? reinterpret_cast<const cutlass::half_t*>(bias)
                                        : reinterpret_cast<const cutlass::half_t*>(output_row);
    int ldc = bias ? 0 : ld_out;
    cutlass::half_t beta = bias ? cutlass::half_t(1.0f) : cutlass::half_t(0.0f);

    const cutlass::half_t* input_ptr = reinterpret_cast<const cutlass::half_t*>(input_row);
    const cutlass::half_t* weight_ptr = reinterpret_cast<const cutlass::half_t*>(weight_row);
    cutlass::half_t* output_ptr = reinterpret_cast<cutlass::half_t*>(output_row);

    cutlass::Status status;

    // Match the tile selection heuristic from cutlass_linear_layer_rrr()
    if (N < 1024) {
        using Gemm = cutlass::gemm::device::Gemm<
            ElementInputA, LayoutA, ElementInputB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>>;
        Gemm gemm_op;
        Gemm::Arguments args({N, out_features, in_features},
                             {input_ptr, in_features},
                             {weight_ptr, out_features},
                             {c_ptr, ldc},
                             {output_ptr, ld_out},
                             {cutlass::half_t(1.0f), beta});
        status = gemm_op.initialize(args, nullptr);
        if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
        status = gemm_op(stream);
    } else {
        using Gemm = cutlass::gemm::device::Gemm<
            ElementInputA, LayoutA, ElementInputB, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 64, 64>, cutlass::gemm::GemmShape<64, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>>;
        Gemm gemm_op;
        Gemm::Arguments args({N, out_features, in_features},
                             {input_ptr, in_features},
                             {weight_ptr, out_features},
                             {c_ptr, ldc},
                             {output_ptr, ld_out},
                             {cutlass::half_t(1.0f), beta});
        status = gemm_op.initialize(args, nullptr);
        if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
        status = gemm_op(stream);
    }

    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    return cudaSuccess;
}

// Row-major GEMM with fused residual + bias addition
// Computes: residual = (A @ B + bias) + residual
// Uses CUTLASS GemmUniversalWithBroadcast to fuse bias and residual in epilogue
cudaError_t cutlass_linear_layer_rrr_fused_residual(
    const half* input_row,         // Row-major [N, in_features]
    const half* weight_row,        // Row-major [in_features, out_features]
    const half* bias,              // [out_features] or nullptr
    half* residual_inout,          // Row-major [N, out_features] - input and output
    int N, int in_features, int out_features,
    cudaStream_t stream) {

    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    // LinearCombinationResidualBlock epilogue computes: output = (AB + residual) + bias
    // Uses Identity functors (no activation) and plus for residual addition.
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
        ElementOutput, ElementAccumulator, ElementCompute, ElementOutput,
        kElementsPerAccess,
        cutlass::epilogue::thread::Identity,
        cutlass::plus,
        cutlass::epilogue::thread::Identity
    >;

    // Use sm_80 configuration, because it's the only one defined that supports tensor operations.
    using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ElementInputA, ElementInputB, ElementOutput,
        ElementAccumulator>;

    // Use 256x128x32 tiles to match PyTorch/cuBLAS optimal configuration
    // Note: 128x256x64 caused launch errors, but 256x128x32 should work
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,

        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        DefaultConfig::kStages,
        DefaultConfig::kAlignmentA,
        DefaultConfig::kAlignmentB
    >;

    Gemm gemm_op;

    Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, out_features, in_features},
        1,  // batch_count
        {ElementCompute(1.0f), ElementCompute(1.0f)},   // {alpha, beta}
        input_row,
        weight_row,
        residual_inout,                                 // C = residual
        residual_inout,                                 // D = residual (in-place)
        const_cast<half*>(bias),
        nullptr,                                        // ptr_Tensor
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        int64_t(in_features), int64_t(out_features), int64_t(out_features),
        int64_t(out_features),
        int64_t(0),                                     // ldr: stride=0 broadcasts bias
        int64_t(0)
    );

    cutlass::Status status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

    return cudaSuccess;
}

// Row-major GEMM with fused SiLU activation
// Computes: D = SiLU(A @ B + bias)
cudaError_t cutlass_linear_layer_rrr_silu(
    const half* input_row,         // Row-major [N, in_features]
    const half* weight_row,        // Row-major [in_features, out_features]
    const half* bias,              // [out_features] or nullptr
    half* output_row,              // Row-major [N, out_features]
    int N, int in_features, int out_features,
    cudaStream_t stream) {

    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementOutput,
        kElementsPerAccess,
        ElementAccumulator,
        float
    >;

    // Use 256x128x32 tiles to match PyTorch/cuBLAS optimal configuration
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        EpilogueOp
    >;

    Gemm gemm_op;

    // Use stride-trick for bias: provide bias as C with stride 0
    typename Gemm::Arguments args(
        {N, out_features, in_features},
        {reinterpret_cast<const cutlass::half_t*>(input_row), in_features},
        {reinterpret_cast<const cutlass::half_t*>(weight_row), out_features},
        {reinterpret_cast<const cutlass::half_t*>(bias), 0},  // Bias with stride 0
        {reinterpret_cast<cutlass::half_t*>(output_row), out_features},
        {1.0f, 1.0f}  // alpha=1, beta=1 for bias addition before SiLU
    );

    cutlass::Status status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

    return cudaSuccess;
}

/**
 * Timestep MLP: Linear1 -> SiLU -> Linear2
 * REFERENCE: TimestepEmbedding in embeddings.py
 */
cudaError_t timestep_mlp_projection(
    const half* input,              // [N, freq_dim] row-major
    half* output,                   // [N, proj_dim] row-major
    const half* w1_col, const half* b1,  // w1: [freq_dim, hidden_dim] ROW-major (PyTorch tensor)
    const half* w2_col, const half* b2,  // w2: [hidden_dim, proj_dim] ROW-major (PyTorch tensor)
    half* hidden_buffer,            // [N, hidden_dim] workspace buffer
    int N, int freq_dim, int hidden_dim, int proj_dim,
    cudaStream_t stream) {

    if (N <= 0) {
        return cudaSuccess;
    }

    // Stage 1: RowMajor GEMM with fused SiLU activation -> hidden_buffer
    cudaError_t err = cutlass_linear_layer_rrr_silu(
        input,
        w1_col,
        b1,
        hidden_buffer,
        N,
        freq_dim,
        hidden_dim,
        stream);
    if (err != cudaSuccess) {
        return err;
    }

    // Stage 2: RowMajor GEMM with stride-trick bias directly into output
    err = cutlass_linear_layer_rrr(
        hidden_buffer,
        w2_col,
        b2,
        output,
        N,
        hidden_dim,
        proj_dim,
        stream);

    if (err != cudaSuccess) {
        return err;
    }

    return cudaSuccess;
}

// ============================================================================
// TEXT PROJECTION
// REFERENCE: HuggingFace embeddings.py - PixArtAlphaTextProjection
// Structure: Linear (+ optional LayerNorm + act_fn)
// ============================================================================

/**
 * LayerNorm kernel (FP32 compute, FP16 output)
 * Reuses existing implementation pattern from wan_transformer_block.cu
 */
__global__ void layernorm_text_kernel(
    const half* input,
    half* output,
    const half* gamma,
    const half* beta,
    int N, int seq_len, int dim,
    float eps) {

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    int total = N * seq_len;
    if (idx >= total) return;

    extern __shared__ float smem[];
    float* ssum = smem;
    float* ssum2 = smem + blockDim.x;
    int tid = threadIdx.x;

    // Compute mean and variance
    float acc1 = 0.f, acc2 = 0.f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = __half2float(input[idx * dim + d]);
        acc1 += v;
        acc2 += v * v;
    }
    ssum[tid] = acc1;
    ssum2[tid] = acc2;
    __syncthreads();

    // Reduction
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            ssum[tid] += ssum[tid + off];
            ssum2[tid] += ssum2[tid + off];
        }
        __syncthreads();
    }

    float mean = ssum[0] / float(dim);
    float var = ssum2[0] / float(dim) - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = __half2float(input[idx * dim + d]);
        v = (v - mean) * inv_std;
        if (gamma) v = v * __half2float(gamma[d]);
        if (beta) v = v + __half2float(beta[d]);
        output[idx * dim + d] = __float2half(v);
    }
}

cudaError_t text_projection(
    const half* input,
    half* output,
    const half* weight,
    const half* bias,
    const half* ln_weight,
    const half* ln_bias,
    int N, int seq_len, int text_dim, int model_dim,
    float eps,
    cudaStream_t stream) {

    int M = N * seq_len;

    // CUTLASS GEMM: [M, text_dim] x [text_dim, model_dim] -> [M, model_dim]
    // Mixed layouts: row-major input, column-major weight (from transpose kernel), row-major output
    {
        using ElementInputA = cutlass::half_t;
        using ElementInputB = cutlass::half_t;
        using ElementOutput = cutlass::half_t;
        using ElementAccumulator = float;
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::RowMajor;
        using LayoutC = cutlass::layout::RowMajor;
        using Gemm = cutlass::gemm::device::Gemm<
            ElementInputA, LayoutA,
            ElementInputB, LayoutB,
            ElementOutput, LayoutC,
            ElementAccumulator>;
        Gemm gemm_op;
        // Use stride trick when bias is present: beta=1, ldc=0, C points to bias vector
        const cutlass::half_t* c_ptr = bias ? reinterpret_cast<const cutlass::half_t*>(bias) : nullptr;
        int ldc = bias ? 0 : model_dim;
        cutlass::half_t beta = bias ? cutlass::half_t(1.0f) : cutlass::half_t(0.0f);
        Gemm::Arguments args({M, model_dim, text_dim},
                            {reinterpret_cast<const cutlass::half_t*>(input), text_dim},
                            {reinterpret_cast<const cutlass::half_t*>(weight), model_dim},
                            {c_ptr, ldc},
                            {reinterpret_cast<cutlass::half_t*>(output), model_dim},
                            {cutlass::half_t(1.0f), beta});
        cutlass::Status st = gemm_op.initialize(args, nullptr);
        if (st != cutlass::Status::kSuccess) return cudaErrorUnknown;
        st = gemm_op(stream);
        if (st != cutlass::Status::kSuccess) return cudaErrorUnknown;
    }

    // Apply LayerNorm if requested
    if (ln_weight) {
        dim3 block(256, 1);
        dim3 grid((M + block.y - 1) / block.y);
        size_t smem = 2 * block.x * sizeof(float);

        layernorm_text_kernel<<<grid, block, smem, stream>>>(
            output, output, ln_weight, ln_bias, N, seq_len, model_dim, eps);

        CUDA_CHECK(cudaGetLastError());
    }

    return cudaSuccess;
}

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
    cudaStream_t stream) {

    if (!encoder_hidden_states_img || !out_img_k || N <= 0 || img_seq <= 0 || img_dim <= 0 || K <= 0) {
        return cudaSuccess;
    }

    // 1) norm1 on raw CLIP embeds: [N*img_seq, img_dim]
    if (norm1_gamma || norm1_beta) {
        int M_ln = N * img_seq;
        dim3 block(256, 1);
        dim3 grid((M_ln + block.y - 1) / block.y);
        size_t smem = 2 * block.x * sizeof(float);
        layernorm_text_kernel<<<grid, block, smem, stream>>>(
            encoder_hidden_states_img,
            scratch_ln1,
            norm1_gamma, norm1_beta,
            N, img_seq, img_dim, 1e-6f);
        CUDA_CHECK(cudaGetLastError());
    } else {
        size_t bytes = sizeof(half) * (size_t)N * img_seq * img_dim;
        cudaMemcpyAsync(scratch_ln1, encoder_hidden_states_img, bytes, cudaMemcpyDeviceToDevice, stream);
    }

    // 2) MLP fc_in: img_dim -> img_dim
    CUDA_CHECK(cutlass_linear_layer_rrr(
        scratch_ln1,
        ff0_w, ff0_b,
        scratch_ff0,
        N * img_seq, img_dim, img_dim, stream));

    // 3) GELU exact (erf): matches nn.GELU(approximate="none")
    {
        long long numel = (long long)N * img_seq * img_dim;
        int threads = 256;
        long long blocks = (numel + threads - 1) / threads;
        gelu_erf_inplace_kernel<<<blocks, threads, 0, stream>>>(scratch_ff0, numel);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4) MLP fc_out: img_dim -> K
    CUDA_CHECK(cutlass_linear_layer_rrr(
        scratch_ff0,
        ff2_w, ff2_b,
        out_img_k,
        N * img_seq, img_dim, K, stream));

    // 5) norm2 on K
    if (norm2_gamma || norm2_beta) {
        int M_ln = N * img_seq;
        dim3 block(256, 1);
        dim3 grid((M_ln + block.y - 1) / block.y);
        size_t smem = 2 * block.x * sizeof(float);
        layernorm_text_kernel<<<grid, block, smem, stream>>>(
            out_img_k, out_img_k, norm2_gamma, norm2_beta,
            N, img_seq, K, 1e-6f);
        CUDA_CHECK(cudaGetLastError());
    }

    // 6) Optional pos_embed: [1, img_seq, K]
    if (pos_embed) {
        long long total = (long long)N * img_seq * K;
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        add_pos_embed_img_kernel<<<blocks, threads, 0, stream>>>(out_img_k, pos_embed, N, img_seq, K);
        CUDA_CHECK(cudaGetLastError());
    }

    return cudaSuccess;
}

// ============================================================================
// PATCHIFY OPERATION
// REFERENCE: Inverse of unpatchify in transformer_wan.py
// NOTE: In practice, Conv3d with stride=patch_size IS the patchify operation
//       (it extracts patches AND projects channels in one operation)
//       This kernel is provided for completeness but not currently used.
// ============================================================================

/**
 * Patchify 3D video into non-overlapping patches
 * Input:  [N, C_in, T, H, W]
 * Output: [N, post_T*post_H*post_W, C_in*pt*ph*pw]
 *
 * Extracts patches with stride=patch_size (non-overlapping)
 * If needed, can replace Conv3d with: patchify + CUTLASS GEMM for projection
 */
__global__ void patchify_3d_kernel(
    const half* __restrict__ input,   // [N, C_in, T, H, W]
    half* __restrict__ output,        // [N, M, C_in*patch_vol]
    int N, int C_in,
    int T, int H, int W,
    int pt, int ph, int pw,
    int post_T, int post_H, int post_W) {

    int n = blockIdx.z;
    int patch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int c_local = blockIdx.x * blockDim.x + threadIdx.x;

    int M = post_T * post_H * post_W;
    int patch_vol = pt * ph * pw;
    int C_patch = C_in * patch_vol;

    if (n >= N || patch_idx >= M || c_local >= C_patch) return;

    // Decompose patch_idx to spatial patch position
    int pw_idx = patch_idx % post_W;
    int ph_idx = (patch_idx / post_W) % post_H;
    int pt_idx = patch_idx / (post_H * post_W);

    // Decompose c_local to (channel, t_local, h_local, w_local)
    int w_local = c_local % pw;
    int h_local = (c_local / pw) % ph;
    int t_local = (c_local / (pw * ph)) % pt;
    int c_in = c_local / patch_vol;

    // Source position in input
    int t_src = pt_idx * pt + t_local;
    int h_src = ph_idx * ph + h_local;
    int w_src = pw_idx * pw + w_local;

    // Input: [N, C_in, T, H, W] layout
    long long in_idx = ((long long)n * C_in + c_in) * (T * H * W) +
                       t_src * (H * W) + h_src * W + w_src;

    // Output: [N, M, C_patch] layout
    long long out_idx = ((long long)n * M + patch_idx) * C_patch + c_local;

    output[out_idx] = input[in_idx];
}

cudaError_t patchify_3d(
    const half* input,
    half* output,
    int N, int C_in,
    int T, int H, int W,
    int pt, int ph, int pw,
    int post_T, int post_H, int post_W,
    cudaStream_t stream) {

    int M = post_T * post_H * post_W;
    int patch_vol = pt * ph * pw;
    int C_patch = C_in * patch_vol;

    // Launch configuration
    dim3 block(32, 16);  // x: C_patch elements, y: patches
    dim3 grid((C_patch + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              N);

    patchify_3d_kernel<<<grid, block, 0, stream>>>(
        input, output,
        N, C_in, T, H, W,
        pt, ph, pw,
        post_T, post_H, post_W
    );

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Concatenate two tensors along sequence dimension
 * Used to combine image and text embeddings
 */
__global__ void concatenate_seq_kernel(
    const half* input1, int seq1,
    const half* input2, int seq2,
    half* output,
    int N, int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (seq1 + seq2) * dim;

    if (idx >= total) return;

    // Decompose index
    int remaining = idx;
    int d = remaining % dim; remaining /= dim;
    int s = remaining % (seq1 + seq2); remaining /= (seq1 + seq2);
    int n = remaining;

    // Copy from appropriate source
    if (s < seq1) {
        // From input1
        output[idx] = input1[(n * seq1 + s) * dim + d];
    } else {
        // From input2
        output[idx] = input2[(n * seq2 + (s - seq1)) * dim + d];
    }
}

cudaError_t concatenate_seq_dim(
    const half* input1, int seq1,
    const half* input2, int seq2,
    half* output,
    int N, int dim,
    cudaStream_t stream) {

    int total = N * (seq1 + seq2) * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    concatenate_seq_kernel<<<blocks, threads, 0, stream>>>(
        input1, seq1, input2, seq2, output, N, dim);

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

/**
 * Unflatten: [N, M*K] -> [N, M, K]
 * This is just a memory copy with reshape (no data movement needed)
 */
cudaError_t unflatten_dimension(
    const half* input,
    half* output,
    int N, int M, int K,
    cudaStream_t stream) {

    // Unflatten is a view operation - no kernel needed
    // Just ensure pointers are correct
    if (input != output) {
        size_t total_bytes = sizeof(half) * N * M * K;
        cudaMemcpyAsync(output, input, total_bytes, cudaMemcpyDeviceToDevice, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    return cudaSuccess;
}

// ============================================================================
// FORMAT CONVERSION KERNELS
// Support conversion between PyTorch (NCTHW) and CUTLASS (NDHWC) formats
// ============================================================================

/**
 * Convert NCTHW -> NDHWC (PyTorch format -> CUTLASS format)
 * Input:  [N, C, T, H, W] - channels first
 * Output: [N, T, H, W, C] - channels last
 */
__global__ void ncthw_to_ndhwc_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N, int C, int T, int H, int W) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * T * H * W * C;
    if (idx >= total) return;

    // Decompose flat index to NDHWC position
    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int t = (idx / (C * W * H)) % T;
    int n = idx / (C * W * H * T);

    // Read from NCTHW: [n, c, t, h, w]
    long long in_idx = ((((long long)n * C + c) * T + t) * H + h) * W + w;
    output[idx] = input[in_idx];
}

/**
 * Convert KCTRS -> KTRSC (PyTorch weight format -> CUTLASS weight format)
 * Input:  [K, C, T, R, S] - channels second
 * Output: [K, T, R, S, C] - channels last
 */
__global__ void kctrs_to_ktrsc_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int K, int C, int T, int R, int S) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)K * T * R * S * C;
    if (idx >= total) return;

    // Decompose flat index to KTRSC position
    int c = idx % C;
    int s = (idx / C) % S;
    int r = (idx / (C * S)) % R;
    int t = (idx / (C * S * R)) % T;
    int k = idx / (C * S * R * T);

    // Read from KCTRS: [k, c, t, r, s]
    long long in_idx = ((((long long)k * C + c) * T + t) * R + r) * S + s;
    output[idx] = input[in_idx];
}

/**
 * Flatten and transpose NDHWC to ColumnMajor
 * Input:  [N, T, H, W, C] NDHWC (channels last)
 * Output: [N*T*H*W, C] ColumnMajor
 *
 * This is used after patch embedding to prepare data for transformer blocks.
 */
__global__ void flatten_transpose_ndhwc_to_colmajor_kernel(
    const half* __restrict__ in_ndhwc,  // [N, T, H, W, C]
    half* __restrict__ out_col,          // [M, C] ColumnMajor where M=N*T*H*W
    int N, int C, int T, int H, int W) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long M = (long long)N * T * H * W;
    if (idx >= M) return;

    // Input is already in NDHWC: [N, T, H, W, C]
    // We just need to copy it to ColumnMajor layout
    // ColumnMajor [M, C]: element at row i, col c is at i + c*M

    // Copy all channels for this spatial position
    for (int c = 0; c < C; ++c) {
        // Read from NDHWC: idx*C + c (sequential in memory)
        // Write to ColumnMajor: idx + c*M
        out_col[idx + c * M] = in_ndhwc[idx * C + c];
    }
}

cudaError_t convert_ncthw_to_ndhwc(
    const half* input,
    half* output,
    int N, int C, int T, int H, int W,
    cudaStream_t stream) {

    int threads = 256;
    long long total = (long long)N * C * T * H * W;
    long long blocks = (total + threads - 1) / threads;

    ncthw_to_ndhwc_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, T, H, W);

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t convert_kctrs_to_ktrsc(
    const half* input,
    half* output,
    int K, int C, int T, int R, int S,
    cudaStream_t stream) {

    int threads = 256;
    long long total = (long long)K * C * T * R * S;
    long long blocks = (total + threads - 1) / threads;

    kctrs_to_ktrsc_kernel<<<blocks, threads, 0, stream>>>(
        input, output, K, C, T, R, S);

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t flatten_transpose_ndhwc_to_colmajor(
    const half* input,
    half* output,
    int N, int C, int T, int H, int W,
    cudaStream_t stream) {

    int threads = 256;
    long long M = (long long)N * T * H * W;
    long long blocks = (M + threads - 1) / threads;

    flatten_transpose_ndhwc_to_colmajor_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, T, H, W);

    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

// ============================================================================
// 2D CONVOLUTION HELPER
// Used when depth dimension is 1
// ============================================================================

static cudaError_t patch_embedding_conv2d_impl(
    const half* input,        // [N, H, W, C_in] NHWC format
    const half* weight,       // [C_out, K_h, K_w, C_in] KRSC format
    half* output,             // [N, H_out, W_out, C_out] NHWC format
    int N, int C_in, int C_out,
    int H, int W,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream) {

    // Calculate output dimensions
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

    // Define CUTLASS 2D convolution kernel (channels-last native)
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;

    using Conv2dKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInputA, cutlass::layout::TensorNHWC,
        ElementInputB, cutlass::layout::TensorNHWC,
        ElementOutput, cutlass::layout::TensorNHWC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,  // Stages
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided,
        4,  // AlignmentA
        4   // AlignmentB
    >::Kernel;

    using Conv2dOp = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernel>;

    // Create problem size
    cutlass::conv::Conv2dProblemSize problem_size(
        N,                              // N (batch)
        H, W,                          // H, W (input spatial)
        C_in,                          // C (input channels)
        C_out,                         // K (output channels)
        K_h, K_w,                      // R, S (filter spatial)
        H_out, W_out,                  // P, Q (output spatial)
        pad_h, pad_w,                  // padding
        stride_h, stride_w,            // stride
        1, 1,                          // dilation
        cutlass::conv::Mode::kCrossCorrelation
    );

    // Configure CUTLASS arguments
    Conv2dOp conv_op;

    // Create layout objects (must be lvalues for TensorRef constructor)
    auto layout_A = PACK_INPUT_LAYOUT_2D(problem_size);
    auto layout_B = PACK_WEIGHT_LAYOUT_2D(problem_size);
    auto layout_C = PACK_OUTPUT_LAYOUT_2D(problem_size);

    // Create tensor references using kernel-defined types
    typename Conv2dKernel::TensorRefA ref_A(
        const_cast<cutlass::half_t*>(reinterpret_cast<const cutlass::half_t*>(input)),
        layout_A
    );

    typename Conv2dKernel::TensorRefB ref_B(
        const_cast<cutlass::half_t*>(reinterpret_cast<const cutlass::half_t*>(weight)),
        layout_B
    );

    typename Conv2dKernel::TensorRefC ref_C(
        reinterpret_cast<cutlass::half_t*>(output),
        layout_C
    );

    typename Conv2dKernel::TensorRefC ref_D(
        reinterpret_cast<cutlass::half_t*>(output),
        layout_C
    );

    typename Conv2dKernel::Epilogue::OutputOp::Params epilogue_params(
        ElementAccumulator(1.0f),  // alpha
        ElementAccumulator(0.0f)   // beta
    );

    typename Conv2dOp::Arguments arguments(
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        epilogue_params,
        cutlass::conv::SplitKMode::kSerial
    );

    // Check if kernel can implement this problem
    cutlass::Status status = conv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }

    // Initialize and run kernel
    status = conv_op.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInitializationError;
    }

    status = conv_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

// ============================================================================
// 3D CONVOLUTION FOR PATCH EMBEDDING
// REFERENCE: HuggingFace transformer_wan.py - patch_embedding layer
// ============================================================================

/**
 * 3D Convolution with channels-last (NDHWC) format
 * This is the native CUTLASS format - no internal conversions needed
 *
 * @param input [N, T, H, W, C_in] NDHWC format (channels last)
 * @param weight [C_out, K_d, K_h, K_w, C_in] KTRSC format (channels last)
 * @param output [N, T_out, H_out, W_out, C_out] NDHWC format (channels last)
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
    cudaStream_t stream) {

    // Optimization / robustness: if the depth dimension is trivial, run a 2D conv per-frame.
    //
    // WAN patch embedding almost always uses K_d=1 and stride_d=1. For I2V, C_in can be 36,
    // and CUTLASS Conv3d kernels may reject some (C_in, C_out, kernel) configurations.
    //
    // Because NDHWC is contiguous with channels-last, we can reinterpret:
    //   [N, T, H, W, C]  as  [N*T, H, W, C]
    // without any data movement, and run the existing NHWC Conv2d implementation.
    if (K_d == 1 && stride_d == 1 && pad_d == 0) {
        int NT = N * T;
        return patch_embedding_conv2d_impl(
            input, weight, output,
            NT, C_in, C_out,
            H, W,
            K_h, K_w,
            stride_h, stride_w,
            pad_h, pad_w,
            stream
        );
    }

    // Calculate output dimensions
    int T_out = (T + 2 * pad_d - K_d) / stride_d + 1;
    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;

    // Define CUTLASS 3D convolution kernel (channels-last native)
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;

    using Conv3dKernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
        ElementInputA, cutlass::layout::TensorNDHWC,
        ElementInputB, cutlass::layout::TensorNDHWC,
        ElementOutput, cutlass::layout::TensorNDHWC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,  // Stages
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided
    >::Kernel;

    using Conv3dOp = cutlass::conv::device::ImplicitGemmConvolution<Conv3dKernel>;

    // Create problem size
    cutlass::conv::Conv3dProblemSize problem_size(
        N,                                    // N (batch)
        T, H, W,                             // D, H, W (input spatial)
        C_in,                                // C (input channels)
        C_out,                               // K (output channels)
        K_d, K_h, K_w,                       // T, R, S (filter spatial)
        T_out, H_out, W_out,                 // Z, P, Q (output spatial)
        pad_d, pad_h, pad_w,                 // padding
        stride_d, stride_h, stride_w,        // stride
        1, 1, 1,                             // dilation
        cutlass::conv::Mode::kCrossCorrelation
    );

    // Configure CUTLASS arguments (all in channels-last format)
    Conv3dOp conv_op;

    // Create layout objects (must be lvalues for TensorRef constructor)
    auto layout_A = PACK_INPUT_LAYOUT(problem_size);
    auto layout_B = PACK_WEIGHT_LAYOUT(problem_size);
    auto layout_C = PACK_OUTPUT_LAYOUT(problem_size);

    // Create tensor references using kernel-defined types
    typename Conv3dKernel::TensorRefA ref_A(
        const_cast<cutlass::half_t*>(reinterpret_cast<const cutlass::half_t*>(input)),
        layout_A
    );

    typename Conv3dKernel::TensorRefB ref_B(
        const_cast<cutlass::half_t*>(reinterpret_cast<const cutlass::half_t*>(weight)),
        layout_B
    );

    typename Conv3dKernel::TensorRefC ref_C(
        reinterpret_cast<cutlass::half_t*>(output),
        layout_C
    );

    typename Conv3dKernel::TensorRefC ref_D(
        reinterpret_cast<cutlass::half_t*>(output),
        layout_C
    );

    typename Conv3dKernel::Epilogue::OutputOp::Params epilogue_params(
        ElementAccumulator(1.0f),  // alpha
        ElementAccumulator(0.0f)   // beta
    );

    typename Conv3dOp::Arguments arguments(
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        epilogue_params,
        cutlass::conv::SplitKMode::kSerial
    );

    // Check if kernel can implement this problem
    cutlass::Status status = conv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }

    // Initialize and run kernel
    status = conv_op.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInitializationError;
    }

    status = conv_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

// ============================================================================
// Additional helpers for full-model orchestrator
// ============================================================================

// NDHWC -> row-major [N*M, K] where M=T*H*W
__global__ void flatten_ndhwc_to_row_kernel(
    const half* __restrict__ in_ndhwc,
    half* __restrict__ out_row,
    int N, int T, int H, int W, int K) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_positions = (long long)N * T * H * W;
    if (idx >= total_positions) return;
    int w = idx % W; int tmp = idx / W;
    int h = tmp % H; tmp /= H;
    int t = tmp % T; int n = tmp / T;
    long long in_base = ((((long long)n * T + t) * H + h) * W + w) * K;
    long long row = ((long long)n * T * H * W) + (t * H * W + h * W + w);
    for (int c = 0; c < K; ++c) {
        out_row[row * K + c] = in_ndhwc[in_base + c];
    }
}

// Add bias along last dimension for a contiguous NDHWC tensor
__global__ void add_bias_lastdim_kernel(half* __restrict__ data,
                                        const half* __restrict__ bias,
                                        long long total_elems,
                                        int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elems) return;
    int c = (int)(i % K);
    float v = __half2float(data[i]);
    float b = __half2float(bias[c]);
    data[i] = __float2half(v + b);
}

// Add per-token positional embedding for image tokens.
// data: [N, img_seq, K] row-major, pos: [1, img_seq, K] row-major (broadcast over batch)
__global__ void add_pos_embed_img_kernel(half* __restrict__ data,
                                        const half* __restrict__ pos,
                                        int N, int img_seq, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * img_seq * K;
    if (i >= total) return;
    int k = (int)(i % K);
    long long tmp = i / K;
    int s = (int)(tmp % img_seq);  // token index (within image seq)
    float v = __half2float(data[i]);
    float p = __half2float(pos[(long long)s * K + k]);  // broadcast N dimension
    data[i] = __float2half(v + p);
}

// GELU tanh approximation in-place
__global__ void gelu_tanh_inplace_kernel(half* data, long long numel) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float x = __half2float(data[i]);
    float c = 0.044715f;
    float y = 0.7978845608028654f * (x + c * x * x * x);
    float out = 0.5f * x * (1.0f + tanhf(y));
    data[i] = __float2half(out);
}

// Exact GELU (erf) in-place: matches torch.nn.GELU(approximate="none")
__global__ void gelu_erf_inplace_kernel(half* data, long long numel) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float x = __half2float(data[i]);
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    float out = 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
    data[i] = __float2half(out);
}

// Expand [N,6,K] to [N*M,6,K] by repeating per sequence token
__global__ void expand_temb_to_tokens_kernel(const half* __restrict__ temb,
                                             int N, int M, int K,
                                             half* __restrict__ temb_out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; // over N*M*6*K
    long long total = (long long)N * M * 6 * K;
    if (i >= total) return;
    int k = i % K; long long tmp = i / K;
    int s = tmp % 6; tmp /= 6;
    int n = tmp / M;
    long long in_idx = ((long long)n * 6 + s) * K + k;
    temb_out[i] = temb[in_idx];
}

// Build per-token scale/shift from table [2,K] and temb [N,K] -> [N*M,K]
__global__ void build_final_scale_shift_kernel(const half* __restrict__ sst_2k,
                                               const half* __restrict__ temb_nk,
                                               float* __restrict__ scale_rm,
                                               float* __restrict__ shift_rm,
                                               int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, N*M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, K)
    if (row >= N * M || col >= K) return;
    int n = row / (M);
    float shift_base = __half2float(sst_2k[col]);
    float scale_base = __half2float(sst_2k[K + col]);
    float shift_add = __half2float(temb_nk[n * K + col]);
    float scale_add = __half2float(temb_nk[n * K + col]);
    shift_rm[row * K + col] = shift_base + shift_add;
    scale_rm[row * K + col] = scale_base + scale_add;
}

// Row-major LayerNorm without affine; optional modulation; output row-major half
__global__ void row_layernorm_modulate_half_kernel(
    const half* __restrict__ X_row, int M, int K, float eps,
    const float* __restrict__ scale_rm, const float* __restrict__ shift_rm,
    half* __restrict__ Y_row) {
    int row = blockIdx.x; if (row >= M) return;
    extern __shared__ float s[]; float* ssum = s; float* ssum2 = s + blockDim.x; int tid = threadIdx.x;
    float acc1 = 0.f, acc2 = 0.f;
    for (int j = tid; j < K; j += blockDim.x) {
        float v = __half2float(X_row[(size_t)row * K + j]); acc1 += v; acc2 += v * v;
    }
    ssum[tid] = acc1; ssum2[tid] = acc2; __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) { if (tid < off) { ssum[tid] += ssum[tid+off]; ssum2[tid] += ssum2[tid+off]; } __syncthreads(); }
    float mean = ssum[0] / float(K); float var = ssum2[0] / float(K) - mean * mean; float inv = rsqrtf(var + eps);
    for (int j = tid; j < K; j += blockDim.x) {
        float v = (__half2float(X_row[(size_t)row * K + j]) - mean) * inv;
        if (scale_rm && shift_rm) {
            float s_val = scale_rm[(size_t)row * K + j]; float t_val = shift_rm[(size_t)row * K + j];
            v = v * (1.f + s_val) + t_val;
        }
        Y_row[(size_t)row * K + j] = __float2half(v);
    }
}

// Unpatchify: reverse patch embedding to spatial NCTHW
__global__ void unpatchify_3d_kernel(
    const half* __restrict__ input,  // [N, M, patch_vol*C]
    half* __restrict__ output,       // [N, C, T, H, W]
    int N, int C, int T, int H, int W,
    int pt, int ph, int pw,
    int post_T, int post_H, int post_W) {
    int n = blockIdx.z; int c = blockIdx.y; int thw = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || c >= C) return; int total_spatial = T * H * W; if (thw >= total_spatial) return;
    int t = thw / (H * W); int hw = thw % (H * W); int h = hw / W; int w = hw % W;
    int pt_idx = t / pt; int ph_idx = h / ph; int pw_idx = w / pw;
    int t_local = t % pt; int h_local = h % ph; int w_local = w % pw;
    int patch_idx = (pt_idx * post_H + ph_idx) * post_W + pw_idx;
    int in_c = ((t_local * ph + h_local) * pw + w_local) * C + c;
    long long in_offset = ((long long)n * (post_T * post_H * post_W) + patch_idx) * ((long long)pt * ph * pw * C) + in_c;
    long long out_offset = ((long long)n * C + c) * ((long long)T * H * W) + thw;
    output[out_offset] = input[in_offset];
}

// Debug helper to print tensor stats
static void debug_print_tensor_stats(const half* d_tensor, size_t numel, const char* name, cudaStream_t stream) {
    #ifdef WAN_DEBUG_FORWARD
    cudaStreamSynchronize(stream);
    std::vector<half> h_data(numel);
    cudaMemcpy(h_data.data(), d_tensor, numel * sizeof(half), cudaMemcpyDeviceToHost);
    float sum = 0, sum2 = 0, minv = 1e30f, maxv = -1e30f;
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < numel; ++i) {
        float v = __half2float(h_data[i]);
        if (std::isnan(v)) { nan_count++; continue; }
        if (std::isinf(v)) { inf_count++; continue; }
        sum += v; sum2 += v * v;
        minv = std::min(minv, v); maxv = std::max(maxv, v);
    }
    float mean = sum / numel;
    float var = sum2 / numel - mean * mean;
    float std = std::sqrt(var > 0 ? var : 0);
    printf("[DEBUG C++] %s: numel=%zu, mean=%.6f, std=%.6f, min=%.6f, max=%.6f, nan=%d, inf=%d\n",
           name, numel, mean, std, minv, maxv, nan_count, inf_count);
    #else
    (void)d_tensor; (void)numel; (void)name; (void)stream;
    #endif
}

template <typename WeightT, typename Arch>
cudaError_t transformer_forward(
    const TransformerConfig& cfg,
    const ModelWeightsT<WeightT>& w,
    const ModelIO& io,
    ts::Workspace* workspace,
    cudaStream_t stream) {
    int N = io.N, C_in = io.C_in, T = io.T, H = io.H, W = io.W;
    int pt = cfg.patch_t, ph = cfg.patch_h, pw = cfg.patch_w;
    int post_T = T / pt, post_H = H / ph, post_W = W / pw;
    int M = post_T * post_H * post_W;
    int K = cfg.num_attention_heads * cfg.attention_head_dim;
    int patch_vol = pt * ph * pw;

    if (!workspace) return cudaErrorInvalidValue;

    // Optional coarse profiling (CUDA-event timestamps across major phases)
    int prof_lvl = g_wan_profile_level.load(std::memory_order_relaxed);
    bool do_prof = (prof_lvl > 0);
    // NOTE: these events measure end-to-end GPU time between boundaries, including any idle gaps on the stream.
    enum {
        EV_START = 0,
        EV_AFTER_PATCH,       // after patch embed + flatten
        EV_AFTER_ROPE,        // after RoPE
        EV_AFTER_TEMB6K,      // after timestep -> 6K projection
        EV_AFTER_TEXT,        // after text projection
        EV_AFTER_IMAGE,       // after image embedder
        EV_AFTER_TEMB_EXPAND, // after temb expand
        EV_AFTER_BLOCKS,      // after transformer blocks chain
        EV_AFTER_FINAL_SHIFT, // after build_final_scale_shift
        EV_AFTER_SEQ_NORM,    // after final layernorm+modulate
        EV_AFTER_PROJ_OUT,    // after proj_out GEMM
        EV_AFTER_UNPATCH,     // after unpatchify
        EV_COUNT
    };
    cudaEvent_t ev[EV_COUNT] = {};
    if (do_prof) {
        for (int i = 0; i < EV_COUNT; ++i) {
            cudaEventCreate(&ev[i]);
        }
        cudaEventRecord(ev[EV_START], stream);
    }
    auto rec = [&](int idx) {
        if (do_prof) cudaEventRecord(ev[idx], stream);
    };

    // 1) Patch embedding
    half* dInputNDHWC = reinterpret_cast<half*>(workspace->input_ndhwc);
    CUDA_CHECK(convert_ncthw_to_ndhwc(io.hidden_states_ncthw, dInputNDHWC, N, C_in, T, H, W, stream));
    half* dPEWeightKTRSC = reinterpret_cast<half*>(workspace->pe_weight_ktrsc);
    if (w.patch_embedding_weight == nullptr) return cudaErrorInvalidDevicePointer;
    CUDA_CHECK(convert_kctrs_to_ktrsc(w.patch_embedding_weight, dPEWeightKTRSC, K, C_in, pt, ph, pw, stream));
    half* dPEOutNDHWC = reinterpret_cast<half*>(workspace->pe_out_ndhwc);
    CUDA_CHECK(patch_embedding_conv3d(dInputNDHWC, dPEWeightKTRSC, dPEOutNDHWC,
        N, C_in, K, T, H, W, pt, ph, pw, pt, ph, pw, 0, 0, 0, stream));

    if (w.patch_embedding_bias) {
        long long total_elems = (long long)N * post_T * post_H * post_W * K;
        int threads = 256; long long blocks = (total_elems + threads - 1) / threads;
        add_bias_lastdim_kernel<<<blocks, threads, 0, stream>>>(
            dPEOutNDHWC, w.patch_embedding_bias, total_elems, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // Flatten NDHWC -> row-major [N*M, K]
    half* dHiddenRM = reinterpret_cast<half*>(workspace->hidden_rm);
    {
        long long total_pos = (long long)N * post_T * post_H * post_W;
        int threads = 256; long long blocks = (total_pos + threads - 1) / threads;
        flatten_ndhwc_to_row_kernel<<<blocks, threads, 0, stream>>>(dPEOutNDHWC, dHiddenRM, N, post_T, post_H, post_W, K);
        CUDA_CHECK(cudaGetLastError());
    }
    debug_print_tensor_stats(dHiddenRM, (size_t)N * M * K, "1_patch_embed_out", stream);
    rec(EV_AFTER_PATCH);

    // 2) RoPE
    float* dCos = workspace->rope_cos;
    float* dSin = workspace->rope_sin;
    CUDA_CHECK(generate_rope_3d(dCos, dSin, post_T, post_H, post_W, cfg.attention_head_dim, 10000.0f, cfg.rope_max_seq_len,
        workspace->rope_cos_t, workspace->rope_sin_t,
        workspace->rope_cos_h, workspace->rope_sin_h,
        workspace->rope_cos_w, workspace->rope_sin_w,
        stream));
    rec(EV_AFTER_ROPE);

    // 3) Conditioning: timestep -> temb -> SiLU -> final proj to 6K
    half* dTimestepSin = reinterpret_cast<half*>(workspace->timestep_sin);
    CUDA_CHECK(timestep_sinusoidal_embedding(io.timesteps, dTimestepSin, N, cfg.freq_dim, 10000, stream));
    // MLP projection to time_proj_dim
    // Project sinusoidal -> temb (pre-activation output used later for final modulation)
    half* dTembPre = reinterpret_cast<half*>(workspace->temb_pre);
    half* dTembHidden = reinterpret_cast<half*>(workspace->temb_hidden);
    CUDA_CHECK(timestep_mlp_projection(dTimestepSin, dTembPre,
        w.time_proj_linear1_w, w.time_proj_linear1_b, w.time_proj_linear2_w, w.time_proj_linear2_b,
        dTembHidden,
        N, cfg.freq_dim, cfg.time_hidden_dim, cfg.time_proj_dim, stream));
    // SiLU applied BEFORE final projection per HF implementation (on a separate buffer)
    half* dTembAct = reinterpret_cast<half*>(workspace->temb_act);
    {
        size_t bytes = sizeof(half) * (size_t)N * cfg.time_proj_dim;
        cudaMemcpyAsync(dTembAct, dTembPre, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    CUDA_CHECK(apply_silu_inplace(dTembAct, (long long)N * cfg.time_proj_dim, stream));
    // Final projection to 6K
    half* dTproj6K = reinterpret_cast<half*>(workspace->temb_6k);
    CUDA_CHECK(cutlass_linear_layer_rrr(dTembAct, w.time_proj_final_w, w.time_proj_final_b,
        dTproj6K, N, cfg.time_proj_dim, 6 * K, stream));
    debug_print_tensor_stats(dTproj6K, (size_t)N * 6 * K, "3_temb_6k", stream);
    // Reshape to [N,6,K] already contiguous in row-major
    rec(EV_AFTER_TEMB6K);

    // Text projection to K
    int text_seq = io.text_seq_len;
    half* dEncTextK = nullptr;
    if (w.caption_proj_w) {
        dEncTextK = reinterpret_cast<half*>(workspace->enc_text_k);
            CUDA_CHECK(text_projection(io.encoder_hidden_states, dEncTextK, w.caption_proj_w, w.caption_proj_b,
                nullptr, nullptr, N, text_seq, io.text_dim, K, 1e-6f, stream));
    } else if (w.text_embedder_w1 && w.text_embedder_w2) {
        int hidden = cfg.text_hidden_dim;
        half* dH1 = reinterpret_cast<half*>(workspace->text_h1);
        CUDA_CHECK(cutlass_linear_layer_rrr(io.encoder_hidden_states, w.text_embedder_w1, w.text_embedder_b1,
            dH1, N * text_seq, io.text_dim, hidden, stream));
        {
            long long numel = (long long)N * text_seq * hidden; int threads = 256; long long blocks = (numel + threads - 1) / threads;
            gelu_tanh_inplace_kernel<<<blocks, threads, 0, stream>>>(dH1, numel);
            CUDA_CHECK(cudaGetLastError());
        }
        dEncTextK = reinterpret_cast<half*>(workspace->enc_text_k);
        CUDA_CHECK(cutlass_linear_layer_rrr(dH1, w.text_embedder_w2, w.text_embedder_b2,
            dEncTextK, N * text_seq, hidden, K, stream));
        debug_print_tensor_stats(dEncTextK, (size_t)N * text_seq * K, "4_text_embed_out", stream);
    } else {
        // Assume already in K
        dEncTextK = nullptr;
    }
    rec(EV_AFTER_TEXT);

    // Image projection: run HF-style image_embedder (norm1 -> ff0+GELU -> ff2 -> norm2 [+ pos_embed])
    int img_seq = io.img_seq_len;
    half* dEncImgK = nullptr;
    bool has_img = (io.encoder_hidden_states_img != nullptr && img_seq > 0);

    if (has_img) {
        dEncImgK = reinterpret_cast<half*>(workspace->enc_img_k); // [N*img_seq, K]
        half* img_ln1 = reinterpret_cast<half*>(workspace->enc_flat); // [N*img_seq, img_dim]
        half* img_ff0 = reinterpret_cast<half*>(workspace->enc_cat);  // [N*img_seq, img_dim]
        CUDA_CHECK(image_embedder_forward(
            io.encoder_hidden_states_img,
            dEncImgK,
            w.image_norm1_gamma, w.image_norm1_beta,
            w.image_ff0_w, w.image_ff0_b,
            w.image_ff2_w, w.image_ff2_b,
            w.image_norm2_gamma, w.image_norm2_beta,
            w.image_pos_embed,
            N, img_seq, io.img_dim, K,
            img_ln1, img_ff0,
            stream));
    }
    rec(EV_AFTER_IMAGE);

    // Prepare encoder pointers (no concatenation)
    const half* enc_text_ptr = dEncTextK ? dEncTextK : io.encoder_hidden_states;
    const half* enc_img_ptr = has_img ? dEncImgK : nullptr;

    // temb [N,6,K] -- no longer expanded to [N*M,6,K]; kernels broadcast via temb_row_stride=0
    const half* dTemb6K = reinterpret_cast<const half*>(dTproj6K);
    rec(EV_AFTER_TEMB_EXPAND);

    // 4) Transformer blocks chain
    half* dPing = reinterpret_cast<half*>(workspace->ping);
    half* dPong = reinterpret_cast<half*>(workspace->pong);
    // Copy hidden into dPing as current
    {
        size_t bytes = sizeof(half) * (size_t)N * M * K; cudaMemcpyAsync(dPing, dHiddenRM, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    const half* cur = dPing; half* next = dPong;

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const BlockWeightsT<WeightT>& bw = w.blocks[layer];
        TransformerBlockParamsT<WeightT> p{};
        p.B = N; p.Mq = M; p.K = K; p.H = cfg.num_attention_heads; p.D = cfg.attention_head_dim;
        p.layer_idx = layer;
        p.Mk = text_seq;
        p.Mk_img = has_img ? img_seq : 0;
        p.added_kv_proj_dim = has_img ? K : 0;
        p.FF = cfg.ffn_dim;
        p.encoder_batch_size = cfg.encoder_batch_size;
        p.hidden_states = reinterpret_cast<const cutlass::half_t*>(cur);
        p.encoder_hidden_states = reinterpret_cast<const cutlass::half_t*>(enc_text_ptr);
        p.encoder_hidden_states_img = reinterpret_cast<const cutlass::half_t*>(enc_img_ptr);
        p.rotary_cos = dCos; p.rotary_sin = dSin;
        p.temb_scale_shift_table = reinterpret_cast<const cutlass::half_t*>(bw.scale_shift_table);
        p.temb = reinterpret_cast<const cutlass::half_t*>(dTemb6K);
        p.temb_row_stride = 0;  // broadcast: all rows share the same temb
        p.sa_w_qkv = reinterpret_cast<const WeightT*>(bw.sa_w_qkv);
        p.sa_w_qkv_scale = reinterpret_cast<const cutlass::half_t*>(bw.sa_w_qkv_scale);
        p.sa_b_qkv = reinterpret_cast<const cutlass::half_t*>(bw.sa_b_qkv);
        p.sa_norm_q_gamma = reinterpret_cast<const cutlass::half_t*>(bw.sa_norm_q);
        p.sa_norm_k_gamma = reinterpret_cast<const cutlass::half_t*>(bw.sa_norm_k);
        p.sa_w_out = reinterpret_cast<const WeightT*>(bw.sa_w_out);
        p.sa_w_out_scale = reinterpret_cast<const cutlass::half_t*>(bw.sa_w_out_scale);
        p.sa_b_out = reinterpret_cast<const cutlass::half_t*>(bw.sa_b_out);
        p.ca_w_q = reinterpret_cast<const WeightT*>(bw.ca_w_q);
        p.ca_w_q_scale = reinterpret_cast<const cutlass::half_t*>(bw.ca_w_q_scale);
        p.ca_b_q = reinterpret_cast<const cutlass::half_t*>(bw.ca_b_q);
        p.ca_w_kv = reinterpret_cast<const WeightT*>(bw.ca_w_kv);
        p.ca_w_kv_scale = reinterpret_cast<const cutlass::half_t*>(bw.ca_w_kv_scale);
        p.ca_b_kv = reinterpret_cast<const cutlass::half_t*>(bw.ca_b_kv);
        p.ca_norm_q_gamma = reinterpret_cast<const cutlass::half_t*>(bw.ca_norm_q);
        p.ca_norm_k_gamma = reinterpret_cast<const cutlass::half_t*>(bw.ca_norm_k);
        p.ca_w_add_k = reinterpret_cast<const WeightT*>(bw.ca_w_add_k);
        p.ca_b_add_k = reinterpret_cast<const cutlass::half_t*>(bw.ca_b_add_k);
        p.ca_w_add_k_scale = reinterpret_cast<const cutlass::half_t*>(bw.ca_w_add_k_scale);
        p.ca_w_add_v = reinterpret_cast<const WeightT*>(bw.ca_w_add_v);
        p.ca_b_add_v = reinterpret_cast<const cutlass::half_t*>(bw.ca_b_add_v);
        p.ca_w_add_v_scale = reinterpret_cast<const cutlass::half_t*>(bw.ca_w_add_v_scale);
        p.ca_norm_added_k_gamma = reinterpret_cast<const cutlass::half_t*>(bw.ca_norm_added_k);
        p.ca_w_out = reinterpret_cast<const WeightT*>(bw.ca_w_out);
        p.ca_w_out_scale = reinterpret_cast<const cutlass::half_t*>(bw.ca_w_out_scale);
        p.ca_b_out = reinterpret_cast<const cutlass::half_t*>(bw.ca_b_out);
        p.ln1_gamma = reinterpret_cast<const cutlass::half_t*>(bw.ln1_gamma);
        p.ln2_gamma = reinterpret_cast<const cutlass::half_t*>(bw.ln2_gamma);
        p.ln2_bias = reinterpret_cast<const cutlass::half_t*>(bw.ln2_bias);
        p.ln3_gamma = reinterpret_cast<const cutlass::half_t*>(bw.ln3_gamma);
        p.ffn_w1 = reinterpret_cast<const WeightT*>(bw.ffn_w1);
        p.ffn_w1_scale = reinterpret_cast<const cutlass::half_t*>(bw.ffn_w1_scale);
        p.ffn_b1 = reinterpret_cast<const cutlass::half_t*>(bw.ffn_b1);
        p.ffn_w2 = reinterpret_cast<const WeightT*>(bw.ffn_w2);
        p.ffn_w2_scale = reinterpret_cast<const cutlass::half_t*>(bw.ffn_w2_scale);
        p.ffn_b2 = reinterpret_cast<const cutlass::half_t*>(bw.ffn_b2);
        // Per-block INT8 weight scales (for INT8 with per-block quantization)
        p.sa_w_qkv_block_scale = bw.sa_w_qkv_block_scale;
        p.sa_w_out_block_scale = bw.sa_w_out_block_scale;
        p.ca_w_q_block_scale = bw.ca_w_q_block_scale;
        p.ca_w_kv_block_scale = bw.ca_w_kv_block_scale;
        p.ca_w_out_block_scale = bw.ca_w_out_block_scale;
        p.ffn_w1_block_scale = bw.ffn_w1_block_scale;
        p.ffn_w2_block_scale = bw.ffn_w2_block_scale;
        // Enable per-block quantization mode for INT8
        p.use_perblock_quant = std::is_same<WeightT, int8_t>::value && bw.sa_w_qkv_block_scale != nullptr;
        p.out = reinterpret_cast<cutlass::half_t*>(next);
        p.workspace = workspace;
        cudaError_t err = run_transformer_block<WeightT, Arch>(p, stream); if (err != cudaSuccess) return err;
        if (layer < 3) {
            char buf[64]; snprintf(buf, sizeof(buf), "5_block_%d_out", layer);
            debug_print_tensor_stats(next, (size_t)N * M * K, buf, stream);
        }
        // swap
        const half* tmp = cur; cur = next; next = const_cast<half*>(tmp);
    }
    rec(EV_AFTER_BLOCKS);

    // 5) Final norm + modulation and projection
    float* dScale = workspace->final_scale;
    float* dShift = workspace->final_shift;
    {
        dim3 tb(32, 32); dim3 gb((K + tb.x - 1) / tb.x, (N * M + tb.y - 1) / tb.y);
        build_final_scale_shift_kernel<<<gb, tb, 0, stream>>>(w.final_scale_shift_table,
            dTembPre, dScale, dShift, N, M, K);
        CUDA_CHECK(cudaGetLastError());
    }
    rec(EV_AFTER_FINAL_SHIFT);
    half* dSeqNorm = reinterpret_cast<half*>(workspace->seq_norm);
    row_layernorm_modulate_half_kernel<<<N * M, 256, 2 * 256 * sizeof(float), stream>>>(cur, N * M, K, 1e-6f,
        dScale, dShift, dSeqNorm);
    CUDA_CHECK(cudaGetLastError());
    rec(EV_AFTER_SEQ_NORM);

    // Linear to patch_vol*C_out and unpatchify
    int out_dim = patch_vol * cfg.out_channels;
    half* dTokens = reinterpret_cast<half*>(workspace->tokens);
    CUDA_CHECK(cutlass_linear_layer_rrr(dSeqNorm, w.proj_out_w, w.proj_out_b, dTokens, N * M, K, out_dim, stream));
    rec(EV_AFTER_PROJ_OUT);
    // Reshape to [N, M, out_dim] and unpatchify
    {
        int total_spatial = T * H * W; int threads = 256; int blocks = (total_spatial + threads - 1) / threads;
        dim3 grid(blocks, cfg.out_channels, N);
        unpatchify_3d_kernel<<<grid, threads, 0, stream>>>(dTokens, io.out_ncthw, N, cfg.out_channels, T, H, W, pt, ph, pw, post_T, post_H, post_W);
        CUDA_CHECK(cudaGetLastError());
    }
    rec(EV_AFTER_UNPATCH);

    if (do_prof) {
        cudaEventSynchronize(ev[EV_AFTER_UNPATCH]);
        auto ms = [&](int a, int b) -> float {
            float out = 0.0f;
            cudaEventElapsedTime(&out, ev[a], ev[b]);
            return out;
        };
        float t_patch = ms(EV_START, EV_AFTER_PATCH);
        float t_rope = ms(EV_AFTER_PATCH, EV_AFTER_ROPE);
        float t_temb = ms(EV_AFTER_ROPE, EV_AFTER_TEMB6K);
        float t_text = ms(EV_AFTER_TEMB6K, EV_AFTER_TEXT);
        float t_img = ms(EV_AFTER_TEXT, EV_AFTER_IMAGE);
        float t_expand = ms(EV_AFTER_IMAGE, EV_AFTER_TEMB_EXPAND);
        float t_blocks = ms(EV_AFTER_TEMB_EXPAND, EV_AFTER_BLOCKS);
        float t_fshift = ms(EV_AFTER_BLOCKS, EV_AFTER_FINAL_SHIFT);
        float t_seq = ms(EV_AFTER_FINAL_SHIFT, EV_AFTER_SEQ_NORM);
        float t_proj = ms(EV_AFTER_SEQ_NORM, EV_AFTER_PROJ_OUT);
        float t_unpatch = ms(EV_AFTER_PROJ_OUT, EV_AFTER_UNPATCH);
        float t_total = ms(EV_START, EV_AFTER_UNPATCH);

        std::printf(
            "[transformer_forward][profile] N=%d T=%d H=%d W=%d post=(%d,%d,%d) M=%d K=%d text_seq=%d img_seq=%d | "
            "patch=%.3f rope=%.3f temb=%.3f text=%.3f img=%.3f expand=%.3f blocks=%.3f final_shift=%.3f seq_norm=%.3f proj_out=%.3f unpatch=%.3f total=%.3f ms\n",
            N, T, H, W, post_T, post_H, post_W, M, K, io.text_seq_len, io.img_seq_len,
            t_patch, t_rope, t_temb, t_text, t_img, t_expand, t_blocks, t_fshift, t_seq, t_proj, t_unpatch, t_total
        );

#ifdef OMNIDREAMS_SINGLEVIEW_PROFILE
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "patch_embed", t_patch);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "rope", t_rope);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "timestep_mlp", t_temb);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "text_proj", t_text);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "image_embed", t_img);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "temb_expand", t_expand);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "blocks_total", t_blocks);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "final_shift", t_fshift);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "seq_norm", t_seq);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "proj_out", t_proj);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "unpatchify", t_unpatch);
        OMNIDREAMS_SINGLEVIEW_PROF_RECORD("model_forward", "total", t_total);
#endif

        for (int i = 0; i < EV_COUNT; ++i) {
            cudaEventDestroy(ev[i]);
        }
    }

    return cudaSuccess;
}

// Explicit instantiations (WeightT x Arch combinations).
// Currently only WanArchTraits is instantiated; future models add their own lines.
template cudaError_t transformer_forward<cutlass::half_t, WanArchTraits>(
    const TransformerConfig& cfg,
    const ModelWeightsT<cutlass::half_t>& w,
    const ModelIO& io,
    ts::Workspace* workspace,
    cudaStream_t stream
);
template cudaError_t transformer_forward<cutlass::float_e4m3_t, WanArchTraits>(
    const TransformerConfig& cfg,
    const ModelWeightsT<cutlass::float_e4m3_t>& w,
    const ModelIO& io,
    ts::Workspace* workspace,
    cudaStream_t stream
);
template cudaError_t transformer_forward<int8_t, WanArchTraits>(
    const TransformerConfig& cfg,
    const ModelWeightsT<int8_t>& w,
    const ModelIO& io,
    ts::Workspace* workspace,
    cudaStream_t stream
);

} // namespace omnidreams_singleview
