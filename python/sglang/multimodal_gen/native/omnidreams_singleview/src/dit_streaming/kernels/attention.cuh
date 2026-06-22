// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include "cutlass/numeric_types.h"
#include "arch_traits.h"

namespace ts {
// Forward declaration
struct Workspace;
}

namespace omnidreams_singleview {

// Attention backend selection
enum class AttnBackend {
    CUTLASS_FLASH,  // Default: fused Flash Attention implementation
    CUDNN,          // cuDNN Flash Attention implementation
    // Future backends can be added here (e.g., PYTORCH_REFERENCE, TRITON, etc.)
};

// Set the attention backend to use for subsequent attention operations
void set_attention_backend(AttnBackend backend);

// Get the currently selected attention backend
AttnBackend get_attention_backend();

// GEMM backend selection (FP16, FP8, or INT8 linear layers)
enum class GemmBackend { FP16, FP8, INT8 };
void set_gemm_backend(GemmBackend backend);
GemmBackend get_gemm_backend();

template <typename WeightT>
struct AttentionDeviceParamsT {
  // Dimensions
  // Batched attention contract:
  // - Query tokens are logically shaped [B, Mq, K] but are passed as a flattened
  //   contiguous row-major buffer [B*Mq, K].
  // - For self-attention: keys/values are the same sequence as queries (Mk ignored).
  // - For cross-attention: encoder tokens are [B, Mk, K] flattened to [B*Mk, K].
  //   When encoder_batch_size=1 and B>1, encoder KV is projected once and broadcast
  //   to all batch items (useful for CFG with shared/empty negative prompt).
  int B = 1;   // batch size
  int Mq = 0;  // query length per batch item
  int K; // embed dim (heads * head_dim)
  int H; // heads
  int D; // head_dim
  int Mk = 0;      // key/value length per batch item (text cross-attn); if 0, defaults to Mq
  int Mk_img = 0;  // key/value length for image added-KV branch (optional)
  int added_kv_proj_dim = 0; // dim of image encoder tokens before add_k/add_v
  int encoder_batch_size = 0; // 0 = same as B, else project KV for this many batches and broadcast

  // Input pointers (device)
  const cutlass::half_t* hidden_states;   // [B*Mq, K] RowMajor
  const WeightT* w_qkv;                   // [3K,K] row-major (PyTorch)
  const cutlass::half_t* w_qkv_scale = nullptr; // [3K] per-channel scale (for FP8/INT8) or nullptr
  const cutlass::half_t* b_qkv;           // [3K]
  // Cross-attention variants (optional): separate Q and KV
  const WeightT* w_q = nullptr;           // [K,K]
  const cutlass::half_t* w_q_scale = nullptr; // [K] or nullptr
  const cutlass::half_t* b_q = nullptr;   // [K]
  const WeightT* w_kv = nullptr;          // [2K,K] (text KV)
  const cutlass::half_t* b_kv = nullptr;  // [2K]    (text KV)
  const WeightT* w_add_k = nullptr;       // [K, added_kv_proj_dim] (image K)
  const cutlass::half_t* b_add_k = nullptr;  // [K] (image K)
  const cutlass::half_t* w_add_k_scale = nullptr; // [K] or nullptr
  const WeightT* w_add_v = nullptr;       // [K, added_kv_proj_dim] (image V)
  const cutlass::half_t* b_add_v = nullptr;  // [K] (image V)
  const cutlass::half_t* w_add_v_scale = nullptr; // [K] or nullptr
  const cutlass::half_t* norm_added_k_gamma = nullptr; // [K] RMSNorm for image K
  const cutlass::half_t* w_kv_scale = nullptr; // [2K] or nullptr
  // Cross-attn: [enc_B*Mk, K] RowMajor (text); for image add-KV, [enc_B*Mk_img, added_kv_proj_dim].
  const cutlass::half_t* encoder_hidden_states = nullptr;
  const cutlass::half_t* norm_q_gamma;    // [K]
  const cutlass::half_t* norm_k_gamma;    // [K]
  const WeightT* w_out;                   // [K,K] row-major
  const cutlass::half_t* w_out_scale = nullptr; // [K] or nullptr
  const cutlass::half_t* b_out;           // [K] (can be nullptr when bias applied externally)
  const float* rotary_cos;                // [Mq,D] row-major (shared across batch)
  const float* rotary_sin;                // [Mq,D] row-major (shared across batch)

  // Per-block INT8 weight scales (for per-block quantization mode)
  // Shape: [in_features/128, out_features/128] float32
  const float* w_qkv_block_scale = nullptr;   // [K/128, 3K/128] for self-attn
  const float* w_q_block_scale = nullptr;     // [K/128, K/128] for cross-attn Q
  const float* w_kv_block_scale = nullptr;    // [K/128, 2K/128] for cross-attn KV
  const float* w_out_block_scale = nullptr;   // [K/128, K/128] for output projection
  bool use_perblock_quant = false;            // Flag to enable per-block quantization

  // Fused gated residual params (optional, INT8 only)
  // When gate_sst != nullptr, the output projection GEMM fuses gate*bias + residual
  const cutlass::half_t* gate_sst = nullptr;        // scale_shift_table [6, K]
  const cutlass::half_t* gate_temb = nullptr;        // temporal embedding
  int gate_temb_row_stride = 0;                      // temb row stride (0 = broadcast)
  int gate_idx = -1;                                 // gate index in sst (-1 = disabled)
  const cutlass::half_t* residual_src = nullptr;     // source for residual add

  // Output pointer (device)
  // For cross-attn, treated as input/output residual when fuse_residual=true.
  cutlass::half_t* out_after_linear;      // [B*Mq, K] row-major (fp16)

  // Causal masking (self-attention only; ignored for cross-attention)
  bool is_causal = false;

  // Workspace
  ts::Workspace* workspace;
};

// Split orchestrators for cleaner flow
template <typename WeightT>
cudaError_t run_self_attention(const AttentionDeviceParamsT<WeightT>& p, cudaStream_t stream);
template <typename WeightT>
cudaError_t run_cross_attention(const AttentionDeviceParamsT<WeightT>& p,
                                    bool fuse_residual,
                                    cudaStream_t stream);

// FMHA-only helper: takes pre-packed Q/K/V in BMHK bfloat16 format, returns FMHA output.
// Useful when QKV projection and RoPE are computed externally (e.g. with a different
// RoPE convention from the built-in rmsnorm_pack_bmhk_rope_from_row_kernel).
// Uses cuDNN SDPA; preferred at large sequence lengths (≳8K tokens).
//
// Inputs  Q     [B*Mq, H, D]  bfloat16 (BMHK — batch items concatenated along M)
//         K, V  [B*Mk, H, D]  bfloat16
// Output  O     [B*Mq, H, D]  bfloat16 (caller-allocated)
// Params  B, Mq, Mk, H, D, causal (top-left mask), scale (0 → 1/sqrt(D))
cudaError_t run_cudnn_fmha_packed_qkv(
    const cutlass::bfloat16_t* Q,
    const cutlass::bfloat16_t* K,
    const cutlass::bfloat16_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream);

struct CosmosFp8SdpaSelection {
  std::string preset;
  std::string layout;
  std::string heuristics;
  std::string plan;
  std::string reason;
  bool layout_env_override = false;
  bool heuristics_env_override = false;
  bool plan_env_override = false;
};

CosmosFp8SdpaSelection select_cosmos_fp8_sdpa(
    int B, int Mq, int Mk,
    int H, int D);

// Standalone cuDNN FP8 SDPA probe: accepts pre-packed BMHK raw E4M3 Q/K/V and
// returns BMHK raw E4M3 O. This is intentionally not wired into Cosmos
// streaming until a local SM120 probe proves correctness and performance.
cudaError_t run_cudnn_fmha_packed_qkv_fp8(
    const cutlass::float_e4m3_t* Q,
    const cutlass::float_e4m3_t* K,
    const cutlass::float_e4m3_t* V,
    cutlass::float_e4m3_t* O,
    const float* descale_q,
    const float* descale_k,
    const float* descale_v,
    const float* descale_s,
    const float* scale_s,
    const float* scale_o,
    float* amax_s,
    float* amax_o,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream);

cudaError_t run_sage3_fmha_packed_qkv(
    const cutlass::bfloat16_t* Q,
    const cutlass::bfloat16_t* K,
    const cutlass::bfloat16_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream);

cudaError_t run_sage3_fmha_packed_qkv_fp8(
    const cutlass::float_e4m3_t* Q,
    const cutlass::float_e4m3_t* K,
    const cutlass::float_e4m3_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream);

bool sage3_is_built();
bool sage3_is_runtime_supported(int device);

cudaError_t run_sage3_fmha_packed_qfp4_kvfp8(
    const uint8_t* Q_fp4,
    const cutlass::float_e4m3_t* Q_sf,
    const cutlass::float_e4m3_t* K,
    const cutlass::float_e4m3_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    int padded_mq,
    cudaStream_t stream);

cudaError_t run_sage3_fmha_packed_qkv_fp4(
    const uint8_t* Q_fp4,
    const uint8_t* K_fp4,
    const uint8_t* V_fp4,
    const cutlass::float_e4m3_t* Q_sf,
    const cutlass::float_e4m3_t* K_sf,
    const cutlass::float_e4m3_t* V_sf,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    int padded_mq,
    int padded_mk,
    cudaStream_t stream);

cudaError_t sage3_quantize_q_bf16(
    const cutlass::bfloat16_t* Q,
    const cutlass::bfloat16_t* gamma,
    const cutlass::bfloat16_t* rope_cos,
    const cutlass::bfloat16_t* rope_sin,
    uint8_t* Q_fp4,
    cutlass::float_e4m3_t* Q_sf,
    int B, int Mq,
    int H, int D,
    int input_row_stride,
    int input_head_offset,
    bool apply_rope,
    int padded_mq,
    cudaStream_t stream);

// ============================================================================
// I2V fused cross-attention (text + image branches share Q)
// ============================================================================
//
// HF I2V cross-attn does:
//   q = to_q(x);  k,v = to_kv(text);  k_img,v_img = add_k/add_v(image)
//   out = attn(q,k,v) + attn(q,k_img,v_img)
//   out = to_out(out)
//
// The existing implementation ran text and image cross-attn as two separate calls,
// which duplicates Q projection + Q RMSNorm/packing. This fused path computes Q once
// and reuses it for both branches (CUTLASS backend).
template <typename WeightT>
struct CrossAttentionI2VParamsT {
  // Dimensions
  int B = 1; // batch size (B>1 supported for cuDNN backend)
  int M; // query length
  int K; // embed dim
  int H; // heads
  int D; // head dim
  int Mk_text; // text KV length
  int Mk_img;  // image KV length (0 disables image branch)
  int added_kv_proj_dim; // dim of image tokens before add_k/add_v

  // Inputs
  const cutlass::half_t* hidden_states;            // [M,K] row-major
  const cutlass::half_t* encoder_hidden_states_text; // [Mk_text, K] row-major
  const cutlass::half_t* encoder_hidden_states_img;  // [Mk_img, added_kv_proj_dim] row-major (optional)

  // Weights
  const WeightT* w_q;                   // [K,K] row-major
  const cutlass::half_t* b_q;           // [K]
  const cutlass::half_t* w_q_scale = nullptr; // [K] or nullptr
  const WeightT* w_kv;                  // [K,2K] row-major (text KV)
  const cutlass::half_t* b_kv;          // [2K]
  const cutlass::half_t* w_kv_scale = nullptr; // [2K] or nullptr
  const WeightT* w_add_k;               // [added_kv_proj_dim, K] row-major (image K)
  const cutlass::half_t* b_add_k;       // [K]
  const cutlass::half_t* w_add_k_scale = nullptr; // [K] or nullptr
  const WeightT* w_add_v;               // [added_kv_proj_dim, K] row-major (image V)
  const cutlass::half_t* b_add_v;       // [K]
  const cutlass::half_t* w_add_v_scale = nullptr; // [K] or nullptr
  const cutlass::half_t* norm_q_gamma;  // [K]
  const cutlass::half_t* norm_k_gamma;  // [K] (text K)
  const cutlass::half_t* norm_added_k_gamma; // [K] (image K)
  const WeightT* w_out;                 // [K,K] row-major
  const cutlass::half_t* b_out;         // [K] (can be nullptr when bias applied externally)
  const cutlass::half_t* w_out_scale = nullptr; // [K] or nullptr

  // Output/residual
  cutlass::half_t* out_after_linear;    // [M,K] row-major (if fuse_residual=true, this is residual_inout)

  // Workspace
  ts::Workspace* workspace;
};

template <typename WeightT>
cudaError_t run_cross_attention_i2v(const CrossAttentionI2VParamsT<WeightT>& p,
                                        bool fuse_residual,
                                        cudaStream_t stream);



// Transformer block orchestrator (T2V flavor)
template <typename WeightT>
struct TransformerBlockParamsT {
  // Dimensions
  int B = 1;   // batch size
  int Mq = 0;  // token length per batch item (patch tokens)
  int K; // model dim
  int H; // heads
  int D; // head dim
  int layer_idx = -1; // optional: which block (for profiling output)
  int Mk = 0; // encoder text seq len per batch item (if 0, defaults to Mq)
  int Mk_img = 0;  // encoder image seq len per batch item (optional)
  int added_kv_proj_dim = 0; // image encoder dim before add_k/add_v
  int FF; // ffn inner dim
  int encoder_batch_size = 0; // 0 = same as B, else project KV for this many batches and broadcast

  // Inputs (device)
  const cutlass::half_t* hidden_states;            // [B*Mq, K] RowMajor
  const cutlass::half_t* encoder_hidden_states;    // [enc_B*Mk, K] RowMajor (text)
  const cutlass::half_t* encoder_hidden_states_img; // [enc_B*Mk_img, added_kv_proj_dim] RowMajor (optional)
  const float* rotary_cos;                         // [Mq, D] row-major (self-attn only)
  const float* rotary_sin;                         // [Mq, D] row-major (self-attn only)
  const cutlass::half_t* temb_scale_shift_table;   // [6,K] RowMajor
  const cutlass::half_t* temb;                     // [N,6,K] or [1,6,K] RowMajor (broadcast via temb_row_stride)
  int temb_row_stride = 0;                         // 0 = broadcast (all rows share temb[0]), 6*K = per-batch-item

  // Self-attn weights
  const WeightT* sa_w_qkv;   // [3K,K] row-major
  const cutlass::half_t* sa_w_qkv_scale = nullptr; // [3K] per-channel scale or nullptr
  const cutlass::half_t* sa_b_qkv;   // [3K]
  const cutlass::half_t* sa_norm_q_gamma; // [K]
  const cutlass::half_t* sa_norm_k_gamma; // [K]
  const WeightT* sa_w_out;   // [K,K] row-major
  const cutlass::half_t* sa_w_out_scale = nullptr; // [K] per-channel scale or nullptr
  const cutlass::half_t* sa_b_out;   // [K]

  // Cross-attn weights (separate Q and KV)
  const WeightT* ca_w_q;     // [K,K] row-major
  const cutlass::half_t* ca_w_q_scale = nullptr; // [K] per-channel scale or nullptr
  const cutlass::half_t* ca_b_q;     // [K]
  const WeightT* ca_w_kv;    // [2K,K] row-major
  const cutlass::half_t* ca_w_kv_scale = nullptr; // [2K] per-channel scale or nullptr
  const cutlass::half_t* ca_b_kv;    // [2K]
  const cutlass::half_t* ca_norm_q_gamma; // [K]
  const cutlass::half_t* ca_norm_k_gamma; // [K]
  const WeightT* ca_w_add_k;   // [K, added_kv_proj_dim] (image)
  const cutlass::half_t* ca_b_add_k;   // [K] (image)
  const cutlass::half_t* ca_w_add_k_scale; // [K] or nullptr
  const WeightT* ca_w_add_v;   // [K, added_kv_proj_dim] (image)
  const cutlass::half_t* ca_b_add_v;   // [K] (image)
  const cutlass::half_t* ca_w_add_v_scale; // [K] or nullptr
  const cutlass::half_t* ca_norm_added_k_gamma; // [K] (image)
  const WeightT* ca_w_out;   // [K,K] row-major
  const cutlass::half_t* ca_w_out_scale = nullptr; // [K] per-channel scale or nullptr
  const cutlass::half_t* ca_b_out;   // [K]

  // Norms
  const cutlass::half_t* ln1_gamma; // [K] (FP32LayerNorm elementwise_affine=False -> gamma only)
  const cutlass::half_t* ln3_gamma; // [K]
  const cutlass::half_t* ln2_gamma; // [K] optional if cross_attn_norm
  const cutlass::half_t* ln2_bias;  // [K] optional

  // FFN weights
  const WeightT* ffn_w1; // [ffn_dim, K] row-major
  const cutlass::half_t* ffn_w1_scale = nullptr; // [ffn_dim] per-channel scale or nullptr
  const cutlass::half_t* ffn_b1; // [ffn_dim]
  const WeightT* ffn_w2; // [K, ffn_dim] row-major
  const cutlass::half_t* ffn_w2_scale = nullptr; // [K] per-channel scale or nullptr
  const cutlass::half_t* ffn_b2; // [K]

  // Per-block INT8 weight scales (for per-block quantization mode)
  // These are used when use_perblock_quant=true
  // Shape: [in_features/128, out_features/128] float32
  const float* sa_w_qkv_block_scale = nullptr;   // [K/128, 3K/128]
  const float* sa_w_out_block_scale = nullptr;   // [K/128, K/128]
  const float* ca_w_q_block_scale = nullptr;     // [K/128, K/128]
  const float* ca_w_kv_block_scale = nullptr;    // [K/128, 2K/128]
  const float* ca_w_out_block_scale = nullptr;   // [K/128, K/128]
  const float* ffn_w1_block_scale = nullptr;     // [K/128, FF/128]
  const float* ffn_w2_block_scale = nullptr;     // [FF/128, K/128]

  // Flag to enable per-block quantization mode
  bool use_perblock_quant = false;

  // Output
  cutlass::half_t* out; // [B*Mq, K] RowMajor (half)

  // Workspace
  ts::Workspace* workspace;
};

// Runs the full transformer block forward pass.
// Arch selects the model-specific compute pattern (norm type, activation, gate style).
// Defaults to WanArchTraits; future models define their own traits struct.
template <typename WeightT, typename Arch = WanArchTraits>
cudaError_t run_transformer_block(const TransformerBlockParamsT<WeightT>& p, cudaStream_t stream);

} // namespace omnidreams_singleview
