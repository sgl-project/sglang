// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "common/workspace_alloc.h"

// cuDNN SDPA workspace is backend/shape dependent. We allocate it dynamically in Workspace.

namespace ts {

// CUTLASS Flash Attention workspace pointers
// Contains buffers specific to the CUTLASS Flash Attention implementation
struct CutlassFlashAttentionWorkspace {
    // Self-attention workspace (BMHK packed format for Flash Attention)
    cutlass::half_t* sa_qkv_row;        // M×3K - QKV projection output
    cutlass::half_t* sa_q_bmhk;         // M×H×D - Packed Q
    cutlass::half_t* sa_k_bmhk;         // M×H×D - Packed K
    cutlass::half_t* sa_v_bmhk;         // M×H×D - Packed V
    cutlass::half_t* sa_o_bmhk;         // M×H×D - FMHA output (reuses sa_q_bmhk)
    cutlass::half_t* sa_out_row;        // M×K - Unpacked output (reuses portion of sa_qkv_row)

    // Cross-attention workspace (BMHK packed format for Flash Attention)
    cutlass::half_t* ca_q_bmhk;         // M×H×D - Packed Q
    cutlass::half_t* ca_k_bmhk;         // Mk×H×D - Packed K
    cutlass::half_t* ca_v_bmhk;         // Mk×H×D - Packed V
    cutlass::half_t* ca_o_bmhk;         // M×H×D - FMHA output (reuses ca_q_bmhk)
    cutlass::half_t* ca_q_row;          // M×K - Q projection (reuses portion of sa_qkv_row)
    cutlass::half_t* ca_kv_row;         // Mk×2K - KV projection

    static size_t calculate_size(int M, int K, int H, int D, int Mk_max) {
        size_t mhd = size_t(M) * H * D;
        size_t mk_max_hd = size_t(Mk_max) * H * D;
        size_t mk_max_2k = size_t(Mk_max) * 2 * K;

        size_t total = 0;
        total += size_t(M) * 3 * K; // sa_qkv_row
        total += mhd;               // sa_q_bmhk
        total += mhd;               // sa_k_bmhk
        total += mhd;               // sa_v_bmhk
        total += mhd;               // ca_q_bmhk
        total += mk_max_hd;         // ca_k_bmhk
        total += mk_max_hd;         // ca_v_bmhk
        total += mk_max_2k;         // ca_kv_row

        return total * sizeof(cutlass::half_t);
    }

    void initialize(cutlass::half_t* ptr, int M, int K, int H, int D, int Mk_max) {
        size_t budget = calculate_size(M, K, H, D, Mk_max);
        ts::WorkspaceAllocator ws(ptr, budget, "cutlass_flash");

        sa_qkv_row = ws.alloc<cutlass::half_t>(size_t(M) * 3 * K);
        sa_out_row = sa_qkv_row;   // alias: reuses beginning of sa_qkv_row
        ca_q_row   = sa_qkv_row;   // alias: reuses beginning of sa_qkv_row

        sa_q_bmhk = ws.alloc<cutlass::half_t>(size_t(M) * H * D);
        sa_o_bmhk = sa_q_bmhk;     // alias: output overwrites Q in-place

        sa_k_bmhk = ws.alloc<cutlass::half_t>(size_t(M) * H * D);
        sa_v_bmhk = ws.alloc<cutlass::half_t>(size_t(M) * H * D);

        ca_q_bmhk = ws.alloc<cutlass::half_t>(size_t(M) * H * D);
        ca_o_bmhk = ca_q_bmhk;     // alias: output overwrites Q in-place

        ca_k_bmhk = ws.alloc<cutlass::half_t>(size_t(Mk_max) * H * D);
        ca_v_bmhk = ws.alloc<cutlass::half_t>(size_t(Mk_max) * H * D);
        ca_kv_row = ws.alloc<cutlass::half_t>(size_t(Mk_max) * 2 * K);
    }
};

// cuDNN Attention workspace pointers
// Contains buffers specific to the cuDNN attention implementation
struct CudnnAttentionWorkspace {
    // Self-attention workspace
    cutlass::half_t* sa_qkv_row;        // M×3K - QKV projection output
    cutlass::half_t* sa_q_bmhk;         // 1×H×M×D - Q in BHSD format
    cutlass::half_t* sa_k_bmhk;         // 1×H×M×D - K in BHSD format
    cutlass::half_t* sa_v_bmhk;         // 1×H×M×D - V in BHSD format
    cutlass::half_t* sa_o_bmhk;         // 1×H×M×D - Output in BHSD format

    // Cross-attention workspace
    cutlass::half_t* ca_q_row;          // M×K - Q projection
    cutlass::half_t* ca_kv_row;         // Mk×2K - KV projection
    cutlass::half_t* ca_q_bmhk;         // 1×H×M×D - Q in BHSD format
    cutlass::half_t* ca_k_bmhk;         // 1×H×Mk×D - K in BHSD format
    cutlass::half_t* ca_v_bmhk;         // 1×H×Mk×D - V in BHSD format
    cutlass::half_t* ca_o_bmhk;         // 1×H×M×D - Output in BHSD format

    // cuDNN workspace (for cuDNN internal use)
    void* cudnn_workspace;

    static size_t calculate_size(int M, int K, int H, int D, int Mk_max) {
        size_t total = 0;

        // Self-attention buffers
        total += size_t(M) * 3 * K;     // sa_qkv_row
        total += size_t(M) * K;         // sa_q_bmhk (reinterpreted as M×K)
        total += size_t(M) * K;         // sa_k_bmhk
        total += size_t(M) * K;         // sa_v_bmhk
        total += size_t(M) * K;         // sa_o_bmhk

        // Cross-attention buffers
        total += size_t(M) * K;         // ca_q_row
        total += size_t(Mk_max) * 2 * K; // ca_kv_row
        total += size_t(M) * K;         // ca_q_bmhk
        total += size_t(Mk_max) * K;    // ca_k_bmhk
        total += size_t(Mk_max) * K;    // ca_v_bmhk
        total += size_t(M) * K;         // ca_o_bmhk

        // Convert to bytes (cuDNN SDPA workspace is allocated dynamically elsewhere)
        return total * sizeof(cutlass::half_t);
    }

    void initialize(cutlass::half_t* ptr, int M, int K, int H, int D, int Mk_max) {
        size_t budget = calculate_size(M, K, H, D, Mk_max);
        ts::WorkspaceAllocator ws(ptr, budget, "cudnn_attn");

        sa_qkv_row = ws.alloc<cutlass::half_t>(size_t(M) * 3 * K);
        sa_q_bmhk  = ws.alloc<cutlass::half_t>(size_t(M) * K);
        sa_k_bmhk  = ws.alloc<cutlass::half_t>(size_t(M) * K);
        sa_v_bmhk  = ws.alloc<cutlass::half_t>(size_t(M) * K);
        sa_o_bmhk  = ws.alloc<cutlass::half_t>(size_t(M) * K);

        ca_q_row   = ws.alloc<cutlass::half_t>(size_t(M) * K);
        ca_kv_row  = ws.alloc<cutlass::half_t>(size_t(Mk_max) * 2 * K);
        ca_q_bmhk  = ws.alloc<cutlass::half_t>(size_t(M) * K);
        ca_k_bmhk  = ws.alloc<cutlass::half_t>(size_t(Mk_max) * K);
        ca_v_bmhk  = ws.alloc<cutlass::half_t>(size_t(Mk_max) * K);
        ca_o_bmhk  = ws.alloc<cutlass::half_t>(size_t(M) * K);

        cudnn_workspace = nullptr;
    }
};

// Unified workspace for WAN model and transformer blocks
// Supports both full model forward pass and individual block testing
struct Workspace {
    // === Dimensions ===

    // Block-level dimensions (always used)
    int M;           // Sequence length (for blocks: N*post_T*post_H*post_W)
    int K;           // Inner dimension
    int H;           // Number of heads
    int D;           // Head dimension
    int FF;          // FFN intermediate dimension
    int Mk_max;      // Maximum encoder sequence length

    // Model-level dimensions (0 if block-only mode)
    int N;           // Batch size
    int C_in;        // Input channels
    int T, H_in, W;  // Input spatial dimensions
    int pt, ph, pw;  // Patch sizes
    int post_T, post_H, post_W;  // Post-patch spatial dimensions
    int freq_dim;    // Timestep embedding frequency dimension
    int time_hidden_dim;  // Timestep MLP hidden dimension
    int time_proj_dim;    // Timestep projection dimension
    int text_hidden_dim;  // Text embedder hidden dimension (if used)
    int text_dim;    // Text input dimension
    int img_dim;     // Image input dimension (if used)
    int img_seq;     // Image sequence length (if used)

    // Monolithic allocation
    cutlass::DeviceAllocation<uint8_t> memory_pool;
    // Separate cuDNN SDPA workspace (safe to grow without invalidating other pointers)
    cutlass::DeviceAllocation<uint8_t> cudnn_workspace_pool;
    size_t cudnn_workspace_capacity_bytes = 0;

    // === Transformer Block Buffers (always allocated) ===

    cutlass::half_t* hidden_persistent;  // M×K - persistent hidden state
    cutlass::half_t* scratch_mk_a;      // M×K - reused for pre-norm inputs
    cutlass::half_t* scratch_mk_b;      // M×K - reused for attention outputs
    cutlass::half_t* ffn_intermediate;  // M×FF - FFN GEMM1 output

    // INT8 quantization scratch buffers (allocated when quant_linear is enabled)
    int8_t* int8_scratch;               // M×max(K,FF) - INT8 activation scratch
    float* int8_act_amax;               // Device scalar for activation amax (per-tensor mode)

    // Per-block INT8 quantization buffers (allocated when quant_linear_perblock is enabled)
    // These enable per-128x128-block quantization for better precision
    float* int8_act_block_scales;       // [M/128, max(K,FF)/128] - per-block activation scales
    int int8_act_block_scales_m_blocks; // Number of M blocks for current allocation
    int int8_act_block_scales_k_blocks; // Number of K blocks for current allocation

    // Attention-specific workspace (backend-dependent)
    // Both backends can coexist - they store pointers to the same underlying memory
    CutlassFlashAttentionWorkspace cutlass_flash;
    CudnnAttentionWorkspace cudnn;
    // === Model-Level Buffers (nullptr if block-only mode) ===

    // Patch embedding
    cutlass::half_t* input_ndhwc;           // N×T×H×W×C_in
    cutlass::half_t* pe_weight_ktrsc;       // K×pt×ph×pw×C_in
    cutlass::half_t* pe_out_ndhwc;          // N×post_T×post_H×post_W×K
    cutlass::half_t* hidden_rm;             // N×M×K - hidden states (row-major)

    // RoPE buffers (float precision)
    float* rope_cos;                        // M×D - RoPE cosine table
    float* rope_sin;                        // M×D - RoPE sine table
    float* rope_cos_t;                      // post_T×D_t - temporal RoPE cosine
    float* rope_sin_t;                      // post_T×D_t - temporal RoPE sine
    float* rope_cos_h;                      // post_H×D_h - height RoPE cosine
    float* rope_sin_h;                      // post_H×D_h - height RoPE sine
    float* rope_cos_w;                      // post_W×D_w - width RoPE cosine
    float* rope_sin_w;                      // post_W×D_w - width RoPE sine

    // Timestep embedding buffers
    cutlass::half_t* timestep_sin;          // N×freq_dim
    cutlass::half_t* temb_hidden;           // N×time_hidden_dim - intermediate MLP buffer
    cutlass::half_t* temb_pre;              // N×time_proj_dim
    cutlass::half_t* temb_act;              // N×time_proj_dim
    cutlass::half_t* temb_6k;               // N×6×K
    // temb_expanded removed: kernels now broadcast via temb_row_stride=0 (saves N*M*6*K*2 bytes)

    // Text encoder buffers
    cutlass::half_t* enc_text_k;            // N×text_seq×K
    cutlass::half_t* text_h1;               // N×text_seq×text_hidden_dim

    // Image encoder buffers (optional)
    cutlass::half_t* enc_img_k;             // N×img_seq×K
    cutlass::half_t* enc_cat;               // N×enc_seq×K
    cutlass::half_t* enc_flat;              // N×enc_seq×K

    // Transformer block ping-pong buffers
    cutlass::half_t* ping;                  // N×M×K
    cutlass::half_t* pong;                  // N×M×K

    // Final projection buffers
    float* final_scale;                     // N×M×K
    float* final_shift;                     // N×M×K
    cutlass::half_t* seq_norm;              // N×M×K
    cutlass::half_t* tokens;                // N×M×(patch_vol×C_in)
    cutlass::half_t* output_ndhwc;          // N×T×H×W×C_in

    // Helper to calculate transformer block workspace size.
    static size_t calculate_block_size(
        int M,
        int K,
        int H,
        int D,
        int FF,
        int Mk_max,
        bool quant_linear = false) {
        size_t mk = size_t(M) * K;
        size_t mff = size_t(M) * FF;

        size_t total = 0;
        total += mk;              // hidden_persistent
        total += mk;              // scratch_mk_a
        total += mk;              // scratch_mk_b
        total += mff;             // ffn_intermediate

        // Convert to bytes before adding attention workspace size
        size_t total_bytes = total * sizeof(cutlass::half_t);

        // INT8 quantization buffers (when quant_linear is enabled)
        if (quant_linear) {
            size_t max_features = std::max(size_t(K), size_t(FF));
            total_bytes += size_t(M) * max_features * sizeof(int8_t);  // int8_scratch
            total_bytes = (total_bytes + 255) & ~255;  // Align for int8_act_amax
            total_bytes += sizeof(float);  // int8_act_amax (device scalar for per-tensor mode)
            total_bytes = (total_bytes + 255) & ~255;  // Align for block scales

            // Per-block INT8 quantization scales
            // Shape: [M/128, max(K,FF)/128] float32
            int num_m_blocks = (M + 127) / 128;
            int num_k_blocks = (max_features + 127) / 128;
            total_bytes += size_t(num_m_blocks) * num_k_blocks * sizeof(float);  // int8_act_block_scales
            total_bytes = (total_bytes + 255) & ~255;  // Align for attention workspace
        }

        // CUTLASS and cuDNN can share memory (mutual exclusion at runtime)
        size_t cutlass_size = CutlassFlashAttentionWorkspace::calculate_size(M, K, H, D, Mk_max);
        size_t cudnn_size = CudnnAttentionWorkspace::calculate_size(M, K, H, D, Mk_max);
        size_t shared_attn_size = std::max(cutlass_size, cudnn_size);
        total_bytes += shared_attn_size;

        return total_bytes;
    }

    // Helper to calculate full model workspace size
    static size_t calculate_model_size(
            int N, int M, int K, int H, int D, int FF, int enc_seq,
            int C_in, int T, int H_in, int W, int pt, int ph, int pw,
            int post_T, int post_H, int post_W,
            int freq_dim, int time_hidden_dim, int time_proj_dim,
            int text_hidden_dim, int text_dim, int text_seq,
            int img_dim, int img_seq,
            bool quant_linear = false) {

        size_t total_half = 0;
        size_t total_float = 0;
        size_t total_bytes = 0;  // For mixed-type allocations (block-level)

        int patch_vol = pt * ph * pw;

        // Block-level buffers (includes INT8 scratch when quant_linear=true)
        // For true batched attention, cross-attn KV workspace must accommodate all batch items:
        // total_kv_rows = N * enc_seq (not just enc_seq).
        // NOTE: calculate_block_size returns bytes, so add directly to total_bytes
        total_bytes += calculate_block_size(N * M, K, H, D, FF, N * enc_seq, quant_linear);

        // Patch embedding
        total_half += size_t(N) * T * H_in * W * C_in;
        total_half += size_t(K) * pt * ph * pw * C_in;
        total_half += size_t(N) * post_T * post_H * post_W * K;
        total_half += size_t(N) * M * K;

        // RoPE buffers
        total_float += size_t(M) * D * 2;  // cos + sin
        size_t rope_temp = std::max({size_t(post_T) * D, size_t(post_H) * D, size_t(post_W) * D});
        total_float += rope_temp * 6;

        // Timestep embedding
        total_half += size_t(N) * freq_dim;
        total_half += size_t(N) * time_hidden_dim;  // temb_hidden
        total_half += size_t(N) * time_proj_dim * 2;  // temb_pre, temb_act
        total_half += size_t(N) * 6 * K;
        // temb_expanded removed (was N*M*6*K)

        // Text encoder
        total_half += size_t(N) * text_seq * K;
        if (text_hidden_dim > 0) {
            total_half += size_t(N) * text_seq * text_hidden_dim;
        }

        // Image encoder
        if (img_seq > 0) {
            total_half += size_t(N) * img_seq * K;
            total_half += size_t(N) * enc_seq * K * 2;
        }

        // Ping-pong + final projection
        total_half += size_t(N) * M * K * 2;
        total_float += size_t(N) * M * K * 2;
        total_half += size_t(N) * M * K;
        total_half += size_t(N) * M * patch_vol * C_in;
        total_half += size_t(N) * T * H_in * W * C_in;

        return total_bytes + total_half * sizeof(cutlass::half_t) + total_float * sizeof(float);
    }

    // Initialize for block-only testing (no model-level buffers)
    void initialize_for_blocks(int M_, int K_, int H_, int D_, int FF_, int Mk_max_, bool quant_linear_ = false) {
        M = M_; K = K_; H = H_; D = D_; FF = FF_; Mk_max = Mk_max_;
        N = 0;  // Indicates block-only mode
        quant_linear = quant_linear_;

        size_t total_size = calculate_block_size(M, K, H, D, FF, Mk_max, quant_linear);
        memory_pool.reset(total_size);

        WorkspaceAllocator ws(memory_pool.get(), total_size, "block_workspace");
        initialize_block_pointers(ws, M);

        // Null out model-level pointers
        input_ndhwc = pe_weight_ktrsc = pe_out_ndhwc = hidden_rm = nullptr;
        rope_cos = rope_sin = rope_cos_t = rope_sin_t = nullptr;
        rope_cos_h = rope_sin_h = rope_cos_w = rope_sin_w = nullptr;
        timestep_sin = temb_hidden = temb_pre = temb_act = temb_6k = nullptr;
        enc_text_k = text_h1 = enc_img_k = enc_cat = enc_flat = nullptr;
        ping = pong = seq_norm = tokens = output_ndhwc = nullptr;
        final_scale = final_shift = nullptr;
    }

    // Initialize for full model forward pass
    void initialize_for_model(
            int N_, int M_, int K_, int H_, int D_, int FF_, int enc_seq_,
            int C_in_, int T_, int H_in_, int W_, int pt_, int ph_, int pw_,
            int post_T_, int post_H_, int post_W_,
            int freq_dim_, int time_hidden_dim_, int time_proj_dim_,
            int text_hidden_dim_, int text_dim_, int text_seq,
            int img_dim_, int img_seq_,
            bool quant_linear_ = false) {

        N = N_; M = M_; K = K_; H = H_; D = D_; FF = FF_;
        // For true batched attention, Mk_max is the total encoder tokens across batch.
        Mk_max = N_ * enc_seq_;
        C_in = C_in_; T = T_; H_in = H_in_; W = W_;
        pt = pt_; ph = ph_; pw = pw_;
        post_T = post_T_; post_H = post_H_; post_W = post_W_;
        freq_dim = freq_dim_; time_hidden_dim = time_hidden_dim_; time_proj_dim = time_proj_dim_;
        text_hidden_dim = text_hidden_dim_; text_dim = text_dim_;
        img_dim = img_dim_; img_seq = img_seq_;
        quant_linear = quant_linear_;

        size_t total_bytes = calculate_model_size(N, M, K, H, D, FF, enc_seq_,
                C_in, T, H_in, W, pt, ph, pw, post_T, post_H, post_W,
                freq_dim, time_hidden_dim, time_proj_dim, text_hidden_dim,
                text_dim, text_seq, img_dim, img_seq, quant_linear);

        memory_pool.reset(total_bytes);
        WorkspaceAllocator ws(memory_pool.get(), total_bytes, "model_workspace");
        int patch_vol = pt * ph * pw;

        // Patch embedding
        input_ndhwc     = ws.alloc<cutlass::half_t>(size_t(N) * T * H_in * W * C_in);
        pe_weight_ktrsc = ws.alloc<cutlass::half_t>(size_t(K) * pt * ph * pw * C_in);
        pe_out_ndhwc    = ws.alloc<cutlass::half_t>(size_t(N) * post_T * post_H * post_W * K);
        hidden_rm       = ws.alloc<cutlass::half_t>(size_t(N) * M * K);

        // RoPE buffers
        rope_cos   = ws.alloc<float>(size_t(M) * D);
        rope_sin   = ws.alloc<float>(size_t(M) * D);
        rope_cos_t = ws.alloc<float>(size_t(post_T) * D);
        rope_sin_t = ws.alloc<float>(size_t(post_T) * D);
        rope_cos_h = ws.alloc<float>(size_t(post_H) * D);
        rope_sin_h = ws.alloc<float>(size_t(post_H) * D);
        rope_cos_w = ws.alloc<float>(size_t(post_W) * D);
        rope_sin_w = ws.alloc<float>(size_t(post_W) * D);

        // Timestep embedding
        timestep_sin = ws.alloc<cutlass::half_t>(size_t(N) * freq_dim);
        temb_hidden  = ws.alloc<cutlass::half_t>(size_t(N) * time_hidden_dim);
        temb_pre     = ws.alloc<cutlass::half_t>(size_t(N) * time_proj_dim);
        temb_act     = ws.alloc<cutlass::half_t>(size_t(N) * time_proj_dim);
        temb_6k      = ws.alloc<cutlass::half_t>(size_t(N) * 6 * K);

        // Text encoder
        enc_text_k = ws.alloc<cutlass::half_t>(size_t(N) * text_seq * K);
        if (text_hidden_dim > 0) {
            text_h1 = ws.alloc<cutlass::half_t>(size_t(N) * text_seq * text_hidden_dim);
        } else {
            text_h1 = nullptr;
        }

        // Image encoder
        if (img_seq > 0) {
            enc_img_k = ws.alloc<cutlass::half_t>(size_t(N) * img_seq * K);
            enc_cat   = ws.alloc<cutlass::half_t>(size_t(N) * enc_seq_ * K);
            enc_flat  = ws.alloc<cutlass::half_t>(size_t(N) * enc_seq_ * K);
        } else {
            enc_img_k = enc_cat = enc_flat = nullptr;
        }

        // Ping-pong buffers
        ping = ws.alloc<cutlass::half_t>(size_t(N) * M * K);
        pong = ws.alloc<cutlass::half_t>(size_t(N) * M * K);

        // Final projection
        final_scale  = ws.alloc<float>(size_t(N) * M * K);
        final_shift  = ws.alloc<float>(size_t(N) * M * K);
        seq_norm     = ws.alloc<cutlass::half_t>(size_t(N) * M * K);
        tokens       = ws.alloc<cutlass::half_t>(size_t(N) * M * patch_vol * C_in);
        output_ndhwc = ws.alloc<cutlass::half_t>(size_t(N) * T * H_in * W * C_in);

        // Initialize block-level pointers (model mode uses N*M for effective batch*sequence)
        //
        // IMPORTANT: The transformer block implementation operates on a flattened row dimension:
        //   M_block = B * Mq = N * (post_T * post_H * post_W).
        // All block scratch/attention workspace buffers must be sized for M_block rows.
        initialize_block_pointers(ws, N * M);
    }

    // Release workspace
    void release() {
        memory_pool.release();
        cudnn_workspace_pool.release();
        cudnn_workspace_capacity_bytes = 0;
    }

    // Get total allocated size in bytes
    size_t size_bytes() const {
        return memory_pool.size();
    }

    // Get total allocated size in MB
    float size_mb() const {
        return size_bytes() / (1024.0f * 1024.0f);
    }

    // INT8 linear quantization flag (set before calling initialize)
    bool quant_linear = false;

    // Validate INT8 buffers are properly allocated (returns true if valid)
    bool validate_int8_buffers() const {
        if (!quant_linear) return true;  // Not using INT8, no validation needed
        bool valid = true;
        if (int8_scratch == nullptr) {
            printf("[ERROR] Workspace::validate_int8_buffers: int8_scratch is nullptr despite quant_linear=true\n");
            valid = false;
        }
        if (int8_act_amax == nullptr) {
            printf("[ERROR] Workspace::validate_int8_buffers: int8_act_amax is nullptr despite quant_linear=true\n");
            valid = false;
        }
        if (int8_act_block_scales == nullptr) {
            printf("[ERROR] Workspace::validate_int8_buffers: int8_act_block_scales is nullptr despite quant_linear=true\n");
            valid = false;
        }
        if (valid) {
            printf("[OK] Workspace::validate_int8_buffers: All INT8 buffers valid\n");
        }
        return valid;
    }

    // Ensure cuDNN SDPA workspace exists with at least `bytes` capacity.
    // This can be called between kernel launches and does not invalidate pointers in `memory_pool`.
    void ensure_cudnn_workspace(size_t bytes) {
        if (bytes == 0) {
            // Still keep pointer null; callers should handle zero-size workspace.
            cudnn_workspace_capacity_bytes = 0;
            cudnn.cudnn_workspace = nullptr;
            return;
        }
        if (cudnn_workspace_capacity_bytes >= bytes && cudnn_workspace_pool.get() != nullptr) {
            cudnn.cudnn_workspace = cudnn_workspace_pool.get();
            return;
        }
        // Round up to reduce realloc churn (256KB granularity)
        size_t rounded = (bytes + (256 * 1024 - 1)) & ~(size_t(256 * 1024 - 1));
        cudnn_workspace_pool.reset(rounded);
        cudnn_workspace_capacity_bytes = rounded;
        cudnn.cudnn_workspace = cudnn_workspace_pool.get();
    }

private:
    // Helper to initialize block-level pointers using bounds-checked allocator.
    // effective_M is the actual number of rows to process (N*M for model forward, M for block-only).
    void initialize_block_pointers(WorkspaceAllocator& ws, int effective_M) {
        hidden_persistent = ws.alloc<cutlass::half_t>(size_t(effective_M) * K);
        scratch_mk_a      = ws.alloc<cutlass::half_t>(size_t(effective_M) * K);
        scratch_mk_b      = ws.alloc<cutlass::half_t>(size_t(effective_M) * K);
        ffn_intermediate   = ws.alloc<cutlass::half_t>(size_t(effective_M) * FF);

        // INT8 quantization buffers (when enabled)
        if (quant_linear) {
            size_t max_features = std::max(size_t(K), size_t(FF));
            int8_scratch = ws.alloc<int8_t>(size_t(effective_M) * max_features);
            ws.align_to(256);
            int8_act_amax = ws.alloc<float>(1);
            ws.align_to(256);

            int num_m_blocks = (effective_M + 127) / 128;
            int num_k_blocks = (max_features + 127) / 128;
            int8_act_block_scales = ws.alloc<float>(size_t(num_m_blocks) * num_k_blocks);
            int8_act_block_scales_m_blocks = num_m_blocks;
            int8_act_block_scales_k_blocks = num_k_blocks;
            ws.align_to(256);
        } else {
            int8_scratch = nullptr;
            int8_act_amax = nullptr;
            int8_act_block_scales = nullptr;
            int8_act_block_scales_m_blocks = 0;
            int8_act_block_scales_k_blocks = 0;
        }

        // CUTLASS and cuDNN share the same underlying buffer (mutual exclusion at runtime).
        // Carve out the larger of the two as a shared region.
        size_t cutlass_size = CutlassFlashAttentionWorkspace::calculate_size(effective_M, K, H, D, Mk_max);
        size_t cudnn_size = CudnnAttentionWorkspace::calculate_size(effective_M, K, H, D, Mk_max);
        size_t shared_attn_size = std::max(cutlass_size, cudnn_size);
        WorkspaceAllocator attn_sub = ws.sub(shared_attn_size, "shared_attn");
        auto* attn_base = reinterpret_cast<cutlass::half_t*>(attn_sub.current());
        cutlass_flash.initialize(attn_base, effective_M, K, H, D, Mk_max);
        cudnn.initialize(attn_base, effective_M, K, H, D, Mk_max);

    }
};

} // namespace ts
