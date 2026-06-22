// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// cosmos_block.cuh — declarations for the Cosmos DiT streaming transformer block.
//
// This header is independent of the WAN block (it does not include
// `transformer_block.cuh` / `attention.cuh` to keep compile-time clean and
// to avoid coupling the cosmos param struct to the WAN one). The cosmos
// orchestrator calls into:
//   - `cutlass_linear_layer_rrr*_bf16` -- bf16 CUTLASS GEMMs (this header)
//   - `cosmos_layernorm_modulate` / `cosmos_residual_gate` / `cosmos_rmsnorm_per_head`
//     -- elementwise primitives (this header)
//   - `cosmos_adaln_lora_split` -- adaln-LoRA helper (this header)
//   - `omnidreams_singleview::run_cudnn_fmha_packed_qkv` -- cuDNN SDPA (existing, attention.cuh)
//
// All buffers are externally provided by the bridge (no Workspace yet --
// Phase 1 sticks with at::empty per-call allocations to keep the diff small).

#include <cuda_runtime.h>
#include <cstdint>
#include <cutlass/numeric_types.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include "workspace.cuh"

namespace omnidreams_singleview {

enum class CosmosLinearBackend : int {
  BF16 = 0,
  FP8 = 1,
  MIXED = 2,
};

enum class CosmosAttentionBackend : int {
  CUDNN_BF16 = 0,
  FP8_DENSE_REF = 4,
  SAGE3 = 7,
  FP8_CUDNN = 8,
  SAGE3_FP8 = 9,
};

static constexpr int kCosmosFp8ActivationScaleSites = 10;
static constexpr int kCosmosFp8ActivationScaleFfn1Gelu = 9;

// ---------------------------------------------------------------------------
// 1) bf16 CUTLASS GEMMs (host launchers, defined in `cosmos_gemm_bf16.cu`)
//
// The existing `cutlass_linear_layer_rrr*` kernels are fp16-only and accept
// WAN's pre-transposed `[in_features, out_features]` weight layout (WAN
// does the transpose offline in `native/common/weight_utils.py`). Cosmos
// has no offline weight-extraction step, so it consumes raw PyTorch
// `nn.Linear.weight` (`[out_features, in_features]` row-major). To match
// PyTorch's `output = input @ weight.T` math without an offline transpose,
// these kernels use CUTLASS's **RCR** layout (Row-Col-Row): input
// row-major × weight column-major × output row-major. PyTorch's
// `[out, in]` row-major weight is byte-equivalent to `[in, out]`
// column-major -- so the same memory consumed by an RCR GEMM with
// ldb=in_features computes `output = input @ weight^T` directly.
//
// Tile shapes mirror the fp16 RRR kernels (256×128×32 SM80 tensor cores
// for large M, 128×128×32 for small M) so the bf16 path inherits the same
// large-M / small-M tuning the WAN team converged on.
//
// Naming note: kept the `_rrr_` suffix in the function names even though
// the underlying CUTLASS layout is RCR. The `_rrr_` here describes the
// CALLER's perception (input row-major, weight in PyTorch's natural
// row-major shape, output row-major), not the internal CUTLASS layout
// triple. This keeps the call-site code readable for someone coming from
// the Python side.
// ---------------------------------------------------------------------------

// D = input @ weight^T + bias  (PyTorch nn.Linear math)
//   input:  [N, in_features]              row-major
//   weight: [out_features, in_features]   row-major  (PyTorch nn.Linear.weight)
//   bias:   [out_features] or nullptr     (broadcast via stride-0 trick)
//   output: [N, out_features]             row-major
cudaError_t cutlass_linear_layer_rrr_bf16(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream);

// D = input @ prepared_weight  where prepared_weight is [in_features, out_features]
// row-major. This matches the native WAN/native GEMM layout and avoids the
// raw PyTorch [out, in] ColumnMajor reinterpretation in the hot path.
cudaError_t cutlass_linear_layer_rrr_bf16_prepared(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row_prepared,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream);

// D = GELU(input @ weight^T + bias)
cudaError_t cutlass_linear_layer_rrr_gelu_bf16(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream);

// D = GELU(input @ prepared_weight + bias)
cudaError_t cutlass_linear_layer_rrr_gelu_bf16_prepared(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row_prepared,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream);

// residual_inout = residual_inout + gate * (input @ prepared_weight)
// gate is a [out_features] vector broadcast across rows (B==1 streaming path).
cudaError_t cutlass_linear_layer_rrr_bf16_prepared_gated_residual(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row_prepared,
    const cutlass::bfloat16_t* gate,
    cutlass::bfloat16_t* residual_inout,
    int N, int in_features, int out_features,
    cudaStream_t stream);

// D = SiLU(input @ weight^T + bias)
cudaError_t cutlass_linear_layer_rrr_silu_bf16(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream);

// Strided-batched BF16 GEMM via cuBLASLt: D[b] = A[b] @ B[b]^T + bias_per_instance
//
// Each batch instance computes M x N x K matmul with B^T applied (i.e. PyTorch
// nn.Linear semantics: output = input @ weight.T). Strides advance pointers
// across the batch dimension; pass stride = 0 to broadcast a single matrix
// across all instances. Bias is added per-instance via cuBLASLt's epilogue
// when bias_stride > 0; pass bias = nullptr to skip the bias add.
//
// Used for the AdaLN-LoRA pre-stack path's batched up GEMM. With M = 1
// (B = 1 streaming) and batchCount = num_blocks * 3, the batch dim fills
// the GPU even though per-instance M is a vector.
cudaError_t cublaslt_strided_batched_bf16_gemm(
    const cutlass::bfloat16_t* a_row,           // [batchCount, M, K] row-major
    const cutlass::bfloat16_t* b_row,           // [batchCount, N, K] row-major (PyTorch nn.Linear weight)
    const cutlass::bfloat16_t* bias,            // [batchCount, N] or nullptr; broadcast over M rows
    cutlass::bfloat16_t* d_row,                 // [batchCount, M, N] row-major output
    int M, int K, int N,
    int batchCount,
    int64_t stride_a,                           // elements between consecutive A instances (0 = broadcast)
    int64_t stride_b,                           // elements between consecutive B instances (0 = broadcast)
    int64_t stride_d,                           // elements between consecutive D instances
    int64_t stride_bias,                        // elements between consecutive bias vectors (0 = broadcast)
    cudaStream_t stream);

// residual_inout = (A @ B) + residual   (no bias variant; cosmos out-projs are bias-free)
// Useful for fusing the SA/CA out projection's residual add into the GEMM
// epilogue. Cosmos's SA/CA `output_proj.weight` is `[D, D]` with no bias and
// the gated residual is `x + gate * out`, so we use the simpler
// linear+residual fusion and apply the gate via `cosmos_residual_gate`
// AFTER the GEMM (gate scaling is per-row from adaln-LoRA, not a constant
// the CUTLASS epilogue can broadcast).
//
// For Phase 1 we keep things simple: GEMM -> separate gated_residual kernel.
// (Future: write a custom epilogue that bakes gate*y + x into the GEMM.)

// ---------------------------------------------------------------------------
// 2) Element-wise primitives (defined in `cosmos_modulate.cu`)
// ---------------------------------------------------------------------------

// Y = norm(X) * (1 + scale) + shift
//   X      [M, K]    activations (no-affine LayerNorm input)
//   shift  [B, K]    per-row modulation shift; broadcast to (M / B) rows each
//   scale  [B, K]    per-row modulation scale
template <typename ElementT>
cudaError_t cosmos_layernorm_modulate(
    const ElementT* X,
    const ElementT* shift,
    const ElementT* scale,
    ElementT* Y,
    int M, int K,
    int B,
    float eps,
    cudaStream_t stream);

// Same as cosmos_layernorm_modulate, but also emits the modulated activation
// as raw E4M3 bytes for FP8 linears that immediately consume it.
template <typename ElementT>
cudaError_t cosmos_layernorm_modulate_to_fp8(
    const ElementT* X,
    const ElementT* shift,
    const ElementT* scale,
    ElementT* Y,
    cutlass::float_e4m3_t* Y_fp8,
    int M, int K,
    int B,
    float eps,
    cudaStream_t stream);

// Same normalization/modulation as above, but emits only raw E4M3 bytes. Use
// when the BF16 modulated activation is not consumed before being overwritten.
template <typename ElementT>
cudaError_t cosmos_layernorm_modulate_to_fp8_only(
    const ElementT* X,
    const ElementT* shift,
    const ElementT* scale,
    cutlass::float_e4m3_t* Y_fp8,
    int M, int K,
    int B,
    float eps,
    cudaStream_t stream);

// Fused col_scale + residual + LayerNorm + modulate -> FP8. Replaces the
// back-to-back pair (col_scale_residual_bf16 + cosmos_layernorm_modulate_to_fp8_only)
// that runs after FP8 SA-out / CA-out projections, saving one full BF16 read
// of x between the residual write and the LN reduce. `gemm_half` is the FP16
// scratch produced by the cuBLASLt FP8 GEMM; `alpha` is the per-column
// (weight_scale * gate * input_scale) precomputed by `cosmos_scale_gate_half`.
template <typename ElementT>
cudaError_t cosmos_col_scale_residual_layernorm_modulate_to_fp8_only(
    const cutlass::half_t* gemm_half,
    const cutlass::half_t* alpha,
    ElementT* residual_inout,
    const ElementT* ln_shift,
    const ElementT* ln_scale,
    cutlass::float_e4m3_t* fp8_out,
    float scale_mul,
    int M, int K,
    int B,
    float eps,
    cudaStream_t stream);

// dest = src + gate * y   (per-row gate, broadcast across head dim)
template <typename ElementT>
cudaError_t cosmos_residual_gate(
    ElementT* dest,
    const ElementT* src,
    const ElementT* y,
    const ElementT* gate,                   // [B, K]
    int M, int K,
    int B,
    cudaStream_t stream);

// In-place per-head RMSNorm on packed [B, M, H, D] data.
// gamma is [D] -- same scale per head.
template <typename ElementT>
cudaError_t cosmos_rmsnorm_per_head(
    ElementT* data,
    const ElementT* gamma,
    int B, int M, int H, int D,
    float eps,
    cudaStream_t stream);

// In-place per-head RMSNorm plus FP8 copy for quantized attention Q paths.
// `data` remains normalized in its original dtype; `fp8_out` receives the same
// BMHK values quantized as raw E4M3 bytes.
template <typename ElementT>
cudaError_t cosmos_rmsnorm_per_head_to_fp8(
    ElementT* data,
    const ElementT* gamma,
    cutlass::float_e4m3_t* fp8_out,
    cutlass::float_e4m3_t* fp8_out_bhmd,
    int B, int M, int H, int D,
    float eps,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// 3) adaln-LoRA helper (defined in `cosmos_adaln_lora.cu`)
//
// Computes:
//   h     = SiLU(t_emb @ down.T)            -- [B, lora_dim]
//   mods  = h @ up.T + adaln_lora_3D        -- [B, 3K]
//   shift = mods[:, 0:K]                    -- [B, K]
//   scale = mods[:, K:2K]                   -- [B, K]
//   gate  = mods[:, 2K:3K]                  -- [B, K]
//
// The output `mods_out` is a [B, 3K] buffer the caller carves into three
// [B, K] views (no extra copy). This replaces three separate ATen matmuls.
// ---------------------------------------------------------------------------
template <typename ElementT>
cudaError_t cosmos_adaln_lora_split(
    const ElementT* t_emb,                  // [B, K] -- caller has already applied SiLU
    const ElementT* down_weight,            // [lora_dim, K] row-major (PyTorch layout)
    const ElementT* up_weight,              // [3K, lora_dim] row-major
    const ElementT* adaln_lora_3D,          // [B, 3K] additive component
    ElementT* lora_hidden_buf,              // [B, lora_dim] scratch
    ElementT* mods_out,                     // [B, 3K]
    int B, int K, int lora_dim,
    cudaStream_t stream);

// In-place SiLU on a flat buffer. Used once per forward by the bridge to
// pre-compute SiLU(t_emb) before the per-sub-layer adaln-LoRA helpers
// (the SiLU happens BEFORE the down GEMM, not as a fused epilogue).
template <typename ElementT>
cudaError_t cosmos_silu_inplace(
    ElementT* data,
    int64_t numel,
    cudaStream_t stream);

// Split fused QKV projection output [M, 3K] into contiguous [M, K] Q/K/V rows.
template <typename ElementT>
cudaError_t cosmos_split_qkv(
    const ElementT* qkv,
    ElementT* q,
    ElementT* k,
    ElementT* v,
    int M, int K,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// 4) Streaming transformer block orchestrator (defined in `cosmos_block.cu`)
//
// All pointers are device-resident, all weights are bf16 row-major
// (PyTorch's default state_dict layout).
//
// Per-block streaming flow (mirrors `optimized_dit_forward` in the
// existing bridge but routes everything through CUTLASS GEMMs and the
// fused modulate/residual kernels):
//
//   1) adaln_lora_split  -> (shift_sa, scale_sa, gate_sa)
//      adaln_lora_split  -> (shift_ca, scale_ca, gate_ca)
//      adaln_lora_split  -> (shift_mlp, scale_mlp, gate_mlp)
//
//   2) Self-attention residual (in-place into x):
//       a) ln+modulate(x)              -> normed
//       b) QKV GEMM (3K, K)            -> qkv_row [M, 3K]
//       c) split + per-head RMSNorm Q, K
//       d) apply rotate-half RoPE to Q, K   (caller passes rope_emb)
//       e) write K_rot, V into KV cache at [write_start : write_start + M)
//       f) cuDNN FMHA over [0 : write_start + M)
//       g) out GEMM (K, K)             -> sa_out
//       h) gated_residual:  x = x + gate_sa * sa_out
//
//   3) Cross-attention residual (in-place into x):
//       a) ln+modulate(x)              -> normed
//       b) Q GEMM (K, K)               -> q_row
//       c) per-head RMSNorm Q
//       d) cuDNN FMHA against pre-cached k_cross / v_cross
//       e) out GEMM (K, K)             -> ca_out
//       f) gated_residual:  x = x + gate_ca * ca_out
//
//   4) FFN residual (in-place into x):
//       a) ln+modulate(x)              -> normed
//       b) GEMM1 + GELU (FF, K)        -> ffn1_out
//       c) GEMM2 (K, FF)               -> ffn2_out
//       d) gated_residual:  x = x + gate_mlp * ffn2_out
//
// All GEMMs are bf16 (cutlass_linear_layer_rrr*_bf16). FMHA stays bf16
// via `run_cudnn_fmha_packed_qkv`. KV-cache I/O is bf16 (FlashDreams'
// caches are bf16 tensors).

struct CosmosBlockWeights {
  // Self-attention
  const void* sa_w_qkv;                       // optional [3K, K] row-major raw E4M3 bytes
  const void* sa_w_q;                         // [K, K] row-major bf16 or raw E4M3 bytes
  const void* sa_w_k;                         // [K, K]
  const void* sa_w_v;                         // [K, K]
  const void* sa_w_out;                       // [K, K]
  const cutlass::half_t* sa_w_qkv_scale;      // [3K] for optional fused FP8 QKV
  const cutlass::half_t* sa_w_q_scale;        // [K] for FP8, nullptr for bf16
  const cutlass::half_t* sa_w_k_scale;
  const cutlass::half_t* sa_w_v_scale;
  const cutlass::half_t* sa_w_out_scale;
  const cutlass::bfloat16_t* sa_w_qkv_prepared;  // optional [K, 3K] row-major BF16
  const cutlass::bfloat16_t* sa_w_out_prepared;  // optional [K, K] row-major BF16
  const cutlass::bfloat16_t* sa_q_norm;       // [D]
  const cutlass::bfloat16_t* sa_k_norm;       // [D]

  // Cross-attention (Q only -- K/V come from pre-cached encoder buffers)
  const void* ca_w_q;                         // [K, K] row-major bf16 or raw E4M3 bytes
  const void* ca_w_out;                       // [K, K]
  const cutlass::half_t* ca_w_q_scale;
  const cutlass::half_t* ca_w_out_scale;
  const cutlass::bfloat16_t* ca_w_q_prepared;    // optional [K, K] row-major BF16
  const cutlass::bfloat16_t* ca_w_out_prepared;  // optional [K, K] row-major BF16
  const cutlass::bfloat16_t* ca_q_norm;       // [D]

  // FFN
  const void* ffn_w1;                         // [FF, K] row-major bf16 or raw E4M3 bytes
  const void* ffn_w2;                         // [K, FF]
  const cutlass::half_t* ffn_w1_scale;
  const cutlass::half_t* ffn_w2_scale;
  const cutlass::bfloat16_t* ffn_w1_prepared;    // optional [K, FF] row-major BF16
  const cutlass::bfloat16_t* ffn_w2_prepared;    // optional [FF, K] row-major BF16

  // adaln-LoRA (per sub-layer)
  const cutlass::bfloat16_t* adaln_sa_down;   // [lora_dim, K]
  const cutlass::bfloat16_t* adaln_sa_up;     // [3K, lora_dim]
  const cutlass::bfloat16_t* adaln_ca_down;
  const cutlass::bfloat16_t* adaln_ca_up;
  const cutlass::bfloat16_t* adaln_mlp_down;
  const cutlass::bfloat16_t* adaln_mlp_up;
};

struct CosmosBlockBuffers {
  // Per-call scratch (caller allocates via at::empty; sized below).
  // qkv_row is used only when optional pre-fused self-attention QKV FP8
  // weights are supplied. Split q_row/k_row/v_row remain the fallback.
  cutlass::bfloat16_t* qkv_row;           // [M, 3K]
  cutlass::bfloat16_t* q_row;             // [M, K]
  cutlass::bfloat16_t* k_row;             // [M, K]
  cutlass::bfloat16_t* v_row;             // [M, K]
  cutlass::bfloat16_t* q_bmhk;            // [M, H, D]
  cutlass::bfloat16_t* k_bmhk;            // [M, H, D]
  cutlass::bfloat16_t* v_bmhk;            // [M, H, D]
  cutlass::bfloat16_t* o_bmhk;            // [M, H, D]
  cutlass::bfloat16_t* attn_out_row;      // [M, K] (after re-flatten)
  cutlass::bfloat16_t* normed;            // [M, K] (pre-attn / pre-FFN normed input)
  cutlass::bfloat16_t* ffn_intermediate;  // [M, FF]

  // adaln-LoRA scratch (one per sub-layer).
  cutlass::bfloat16_t* lora_hidden_sa;    // [B, lora_dim]
  cutlass::bfloat16_t* lora_hidden_ca;
  cutlass::bfloat16_t* lora_hidden_mlp;
  cutlass::bfloat16_t* mods_sa;           // [B, 3K]
  cutlass::bfloat16_t* mods_ca;
  cutlass::bfloat16_t* mods_mlp;

  // Quantized path scratch, reused sequentially by per-block linears/attention.
  cutlass::float_e4m3_t* linear_fp8_scratch;  // [M, max(K, FF)]
  cutlass::half_t* linear_half_scratch;       // [M, max(K, FF)]
  cutlass::float_e4m3_t* attn_q_fp8;          // [B, M, H, D]
  cutlass::float_e4m3_t* attn_k_fp8;          // [B, max_attn_tokens, H, D]
  cutlass::float_e4m3_t* attn_v_fp8;          // [B, max_attn_tokens, H, D]
  cutlass::float_e4m3_t* attn_q_bhmd_fp8;     // [B, H, M, D] dense two-GEMM scratch
  cutlass::float_e4m3_t* attn_k_bhmd_fp8;     // [B, H, max_attn_tokens, D]
  cutlass::float_e4m3_t* attn_v_bhmd_fp8;     // [B, H, max_attn_tokens, D]
  cutlass::float_e4m3_t* attn_v_bhdm_fp8;     // [B, H, D, max_attn_tokens] SM120 TN PV scratch
  uint8_t* attn_q_sage3_fp4;                  // [B, H, round_up(M, 128), D / 2]
  cutlass::float_e4m3_t* attn_q_sage3_sf;     // [B, H, round_up(M, 128), D / 16]
  cutlass::half_t* attn_scores_half;          // [B * H, M, max_attn_tokens] for dense FP8 two-GEMM
  cutlass::bfloat16_t* attn_scores_bf16;      // [B * H, M, max_attn_tokens] for SM120 FP8 TC diagnostic
  cutlass::bfloat16_t* attn_score_c_bf16;     // [B * H, M, max_attn_tokens] C scratch for SM120 FP8 TC diagnostic
  cutlass::float_e4m3_t* attn_probs_fp8;      // [B * H, M, max_attn_tokens] for dense FP8 two-GEMM
  cutlass::half_t* attn_o_bhmd_half;          // [B, H, M, D] dense two-GEMM scratch
  cutlass::bfloat16_t* attn_o_bhmd_bf16;      // [B, H, M, D] SM120 FP8 TC diagnostic output scratch
  cutlass::bfloat16_t* attn_o_c_bf16;         // [B, H, M, D] C scratch for SM120 FP8 TC diagnostic
  cutlass::half_t* attn_o_half;               // [B, M, H, D]
  float* attn_tc_scale;                       // [scale_elems] all-ones block-scale scratch
  int64_t attn_tc_scale_elems;
  bool attn_tc_scale_is_ones;                 // true when caller initialized attn_tc_scale to 1.0f

};

struct CosmosBlockParams {
  // Geometry
  int B;             // batch size (typically 1; CFG could be 2)
  int M;             // L_new -- per-call query length
  int K;             // model_channels
  int H;             // num_heads
  int D;             // head_dim = K / H
  int FF;            // FFN inner dim (usually 4 * K)
  int lora_dim;      // adaln_lora_dim (256 for cosmos production)
  int Mk_cross;      // length of pre-cached cross-attn K/V (text tokens)
  int self_attn_cache_cap;   // total slots in self-attn KV cache
  int self_attn_write_start; // append new K/V at [write_start : write_start + M)
  CosmosLinearBackend linear_backend;         // bf16 or FP8 block linear weights
  CosmosAttentionBackend attention_backend;   // cuDNN bf16 or FP8 fused candidate
  bool fp8_kv_cache_enabled;                  // use externally managed FP8 shadow K/V caches for FP8 attention
  bool write_bf16_self_kv_cache;              // keep BF16 self-cache in sync when using FP8 KV caches

  // Optional trace outputs. Each pointer, when non-null, receives a device copy
  // of x after the corresponding residual update.
  cutlass::bfloat16_t* trace_sa_out;
  cutlass::bfloat16_t* trace_ca_out;
  cutlass::bfloat16_t* trace_ffn_out;
  cutlass::bfloat16_t* trace_block_out;
  int64_t trace_elems;

  // Optional activation calibration I/O. `fp8_activation_scales` is a host
  // pointer consumed as [kCosmosFp8ActivationScaleSites] scale values for this
  // block. `fp8_activation_amax_out` is a device pointer that receives BF16
  // amax values for sites that the runtime can observe.
  const float* fp8_activation_scales;
  float* fp8_activation_amax_out;

  // Inputs
  cutlass::bfloat16_t* x;          // [M, K] in/out (residual updated in-place)
  const cutlass::bfloat16_t* t_emb;             // [B, K]            timestep embedding (post-RMSNorm)
  const cutlass::bfloat16_t* adaln_lora_3D;     // [B, 3K]           per-block lora component (added inside helper)
  const cutlass::bfloat16_t* precomputed_mods_sa;  // [B, 3K] optional (shift, scale, gate)
  const cutlass::bfloat16_t* precomputed_mods_ca;  // [B, 3K] optional (shift, scale, gate)
  const cutlass::bfloat16_t* precomputed_mods_mlp; // [B, 3K] optional (shift, scale, gate)

  // Cross-attn pre-cached K/V (post k_norm, no RoPE; these are written by
  // FlashDreams' CrossAttention.initialize_cache and stay constant for the
  // full streaming run).
  const cutlass::bfloat16_t* k_cross;           // [B, Mk_cross, H, D]
  const cutlass::bfloat16_t* v_cross;           // [B, Mk_cross, H, D]

  // Self-attn KV ring buffer (mutated in place)
  cutlass::bfloat16_t* k_self_cache;            // [B, self_attn_cache_cap, H, D]
  cutlass::bfloat16_t* v_self_cache;            // [B, self_attn_cache_cap, H, D]
  cutlass::float_e4m3_t* k_self_cache_fp8;       // [B, self_attn_cache_cap, H, D] optional shadow cache
  cutlass::float_e4m3_t* v_self_cache_fp8;       // [B, self_attn_cache_cap, H, D] optional shadow cache
  const cutlass::float_e4m3_t* k_cross_fp8;      // [B, Mk_cross, H, D] optional pre-quantized cross cache
  const cutlass::float_e4m3_t* v_cross_fp8;      // [B, Mk_cross, H, D] optional pre-quantized cross cache
  cutlass::float_e4m3_t* k_self_cache_fp8_bhmd;  // [B, H, self_attn_cache_cap, D] optional TC-layout cache
  cutlass::float_e4m3_t* v_self_cache_fp8_bhmd;  // [B, H, self_attn_cache_cap, D] optional cuDNN input-layout cache
  cutlass::float_e4m3_t* v_self_cache_fp8_bhdm;  // [B, H, D, self_attn_cache_cap] optional custom TC-layout cache
  const cutlass::float_e4m3_t* k_cross_fp8_bhmd; // [B, H, Mk_cross, D] optional TC-layout cross cache
  const cutlass::float_e4m3_t* v_cross_fp8_bhmd; // [B, H, Mk_cross, D] optional cuDNN input-layout cross cache
  const cutlass::float_e4m3_t* v_cross_fp8_bhdm; // [B, H, D, Mk_cross] optional custom TC-layout cross cache
  int k_cross_fp8_bhmd_tokens;                   // physical token dimension for k_cross_fp8_bhmd
  int v_cross_fp8_bhmd_tokens;                   // physical token dimension for v_cross_fp8_bhmd
  const uint8_t* k_cross_sage3_fp4;            // [B, H, round_up(Mk_cross, 128), D / 2]
  const uint8_t* v_cross_sage3_fp4;            // [B, H, D, round_up(Mk_cross, 128) / 2]
  const cutlass::float_e4m3_t* k_cross_sage3_sf; // [B, H, round_up(Mk_cross, 128), D / 16]
  const cutlass::float_e4m3_t* v_cross_sage3_sf; // [B, H, D, round_up(Mk_cross, 128) / 16]
  int Mk_cross_sage3_padded;

  // Pre-computed RoPE cos/sin in q.dtype, broadcast as [1, M, 1, D].
  // (Cosmos uses rotate-half; the bridge precomputes cos/sin from rope_emb
  //  once per forward and reuses across blocks.)
  const cutlass::bfloat16_t* rope_cos;          // [M, D]
  const cutlass::bfloat16_t* rope_sin;          // [M, D]

  CosmosBlockWeights w;
  CosmosBlockBuffers buf;

  // When true, the bridge has already computed the (shift, scale, gate) bundles
  // for all 28 blocks and stored them at `buf.mods_sa / buf.mods_ca / buf.mods_mlp`.
  // The orchestrator will skip the three per-sub-layer cosmos_adaln_lora_split
  // calls and consume the pre-computed mods directly. See the
  // "AdaLN-LoRA pre-stack + strided-batched up GEMM" Phase 1 plan.
  bool adaln_precomputed;
};

// Runs one Cosmos transformer block in streaming mode. After this returns,
// `params.x` holds the post-block hidden state and the self-attn KV caches
// at slots [write_start : write_start + M) are populated with this block's
// post-RoPE / post-RMSNorm K and post-projection V.
cudaError_t cosmos_run_transformer_block_streaming(
    const CosmosBlockParams& params,
    cudaStream_t stream);

} // namespace omnidreams_singleview
