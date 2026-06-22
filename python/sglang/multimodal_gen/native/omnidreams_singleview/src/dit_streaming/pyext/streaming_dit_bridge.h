// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// streaming_dit_bridge.h — bridge declarations for OmniDreams single-view native extension CUDA extension.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <string>

#include "../kernels/attention.cuh"

namespace omnidreams_singleview {
bool sage3_is_built();
bool sage3_is_runtime_supported(int device);
std::vector<torch::Tensor> sage3_quantize_cross_kv_bf16(
    torch::Tensor k_bmhd,
    torch::Tensor v_bmhd);
}

// Full model forward pass using native CUTLASS FMHA for attention blocks.
//
// Args (matching CosmosCausalHdmapDiT._forward_train):
//   x              — [B, 16, T, H, W] latent video input (bf16 or fp16)
//   condition_mask — [B, 1, T, H, W] conditioning video mask
//   padding_mask   — [B, 1, H, W] spatial padding mask (all-ones = fully valid)
//   timesteps      — [B, T] per-frame diffusion timesteps (float32 or model dtype)
//   crossattn_emb  — [B, Lc, 1024] cross-attention context (text)
//   hdmap          — [B, 16, T, H, W] HDmap control input
//   weights        — dict[str, Tensor] model parameters by name
//   config         — dict with integer/float hyper-parameters
//
// Returns: [B, 16, T, H, W] denoised output (same dtype as x).
torch::Tensor cosmos_forward(
    torch::Tensor x,
    torch::Tensor condition_mask,
    torch::Tensor padding_mask,
    torch::Tensor timesteps,
    torch::Tensor crossattn_emb,
    torch::Tensor hdmap,
    py::dict weights,
    py::dict config
);


// ---------------------------------------------------------------------------
// Streaming forward — mirrors FlashDreams' CosmosDiTNetwork.forward.
//
// Unlike `cosmos_forward` above (full-sequence DDPM, causal SA over Mq==Mk),
// this entry point runs streaming autoregressive inference with a KV cache
// that is fed fresh Q tokens (L_new) attending over a larger cached K/V
// buffer (up to `sink_size + local_attn_size * tokens_per_frame`). Causality
// is enforced implicitly by only caching past tokens, so the FMHA call runs
// with `causal=false` and Mq != Mk.
//
// Inputs (all CUDA tensors; dtype = bf16 unless noted):
//   x_new                    [B, V, T, HW, D_in]       pre-patchified tokens
//                            (output of FlashDreams patchify_and_maybe_split_cp)
//   condition_mask_patched   [B, V, T, HW, D_cond]     pre-patchified condition mask
//   hdmap_patched            [B, V, T, HW, D_hdmap]    pre-patchified HD-map bbox control
//                            (pass `torch::Tensor()` to disable)
//   timesteps                [B] (bf16/fp32)           ONE timestep per batch item
//                            (FlashDreams uses a scalar timestep for the whole block,
//                             unlike cosmos_forward which takes [B, T] per-frame)
//   rope_emb                 [L_new, 1, 1, Dh] fp32    RoPE angles, pre-shifted to
//                            `chunk_idx * num_latents_per_block` by the caller
//   k_cross_caches           list[num_blocks] of [B*V, Lc, H, Dh]
//                            pre-computed text K (post-k_norm, no RoPE)
//   v_cross_caches           list[num_blocks] of [B*V, Lc, H, Dh]
//                            pre-computed text V (raw v_proj, no norm)
//   k_self_caches            list[num_blocks] of [B*V, cache_cap, H, Dh]  MUTABLE
//                            self-attn K ring buffer (post-RoPE + k_norm)
//   v_self_caches            list[num_blocks] of [B*V, cache_cap, H, Dh]  MUTABLE
//                            self-attn V ring buffer (raw)
//   self_attn_write_start    int — write new K/V at cache[:, write_start:write_start+L_new]
//                            (caller is responsible for rolling the ring buffer via
//                             BlockKVCache.before_update(...) before this call;
//                             write_start == _n_cached in non-steady state, or
//                             cache_cap - L_new once the cache saturates)
//   weights                  dict  model parameters, same keys as FlashDreams state_dict
//                            (x_embedder must be post-`_fuse_padding_mask_into_patch_embed`,
//                            final_layer.linear must be post-`_fuse_shuffle_op_into_last_layer`)
//   config                   dict  num_blocks, num_heads, model_channels,
//                                  timestep_scale, (no padding / rope ratios — those
//                                  are baked into the pre-computed rope_emb and weights).
//                                  Optional invariant-cache tensors include
//                                  cosmos_t_emb/cosmos_t_emb_silu/cosmos_adaln_lora,
//                                  cosmos_final_shift/cosmos_final_scale,
//                                  cosmos_rope_cos/cosmos_rope_sin, and
//                                  cosmos_block_mods_sa/ca/mlp, and
//                                  cosmos_hdmap_embed.
//
// Returns: [B, V, T, HW, D_out] (pre-unpatchify); caller applies
//          FlashDreams unpatchify_and_maybe_gather_cp afterwards.
torch::Tensor optimized_dit_forward(
    torch::Tensor x_new,
    torch::Tensor condition_mask_patched,
    torch::Tensor hdmap_patched,
    torch::Tensor timesteps,
    torch::Tensor rope_emb,
    std::vector<torch::Tensor> k_cross_caches,
    std::vector<torch::Tensor> v_cross_caches,
    std::vector<torch::Tensor> k_self_caches,
    std::vector<torch::Tensor> v_self_caches,
    int64_t self_attn_write_start,
    py::dict weights,
    py::dict config
);

// Test/bring-up hook for the Cosmos FP8 RCR linear contract:
//   input [N, in] fp16
//   weight_fp8_u8 [out, in] raw E4M3 bytes in PyTorch Linear layout
//   weight_scale [out] optional per-output-channel dequant scale
// Returns [N, out] fp16.
torch::Tensor cosmos_test_linear_fp8(
    torch::Tensor input,
    torch::Tensor weight_fp8_u8,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> bias,
    bool gelu
);

torch::Tensor cosmos_test_linear_fp8_out_fp8(
    torch::Tensor input_bf16,
    torch::Tensor weight_fp8_u8,
    torch::Tensor weight_scale
);

torch::Tensor cosmos_test_linear_fp8_gelu_out_fp8(
    torch::Tensor input_bf16,
    torch::Tensor weight_fp8_u8,
    torch::Tensor weight_scale,
    double output_scale = 1.0,
    bool alias_output = false
);

torch::Tensor cosmos_test_linear_fp8_scaled_bf16(
    torch::Tensor input_bf16,
    torch::Tensor weight_fp8_u8,
    torch::Tensor weight_scale
);

torch::Tensor cosmos_test_linear_fp8_residual_scaled_bf16(
    torch::Tensor input_bf16,
    torch::Tensor weight_fp8_u8,
    torch::Tensor alpha,
    torch::Tensor residual_bf16
);

py::dict cosmos_test_fp8_linear_tile_selection(
    std::string op_kind,
    int64_t rows,
    int64_t in_features,
    int64_t out_features
);

py::dict cosmos_test_fp8_sdpa_selection(
    int64_t B,
    int64_t Mq,
    int64_t Mk,
    int64_t H,
    int64_t D
);

torch::Tensor cosmos_test_fp8_dense_ref_sdpa(
    torch::Tensor q_fp8_u8,
    torch::Tensor k_fp8_u8,
    torch::Tensor v_fp8_u8,
    bool causal
);

torch::Tensor cosmos_test_fp8_cudnn_sdpa(
    torch::Tensor q_fp8_u8,
    torch::Tensor k_fp8_u8,
    torch::Tensor v_fp8_u8,
    bool causal
);

torch::Tensor cosmos_test_fp8_tc_probe_qk(
    torch::Tensor q_fp8_u8,
    torch::Tensor k_fp8_u8
);

torch::Tensor cosmos_test_fp8_tc_probe_pv(
    torch::Tensor probs_fp8_u8,
    torch::Tensor v_fp8_u8
);

std::vector<torch::Tensor> cosmos_test_fp8_attention_backend(
    torch::Tensor q_fp8_u8,
    torch::Tensor k_fp8_u8,
    torch::Tensor v_fp8_u8,
    bool causal,
    std::string backend
);
