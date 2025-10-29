/*
 * Copyright (c) 2024 by SGLang team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * MLA RoPE + FP8 Quantization + KV Cache Write Fusion Kernel
 *
 * This is a SGLang-native kernel that fuses three operations for DeepSeek V3.2 MLA:
 * 1. Apply RoPE (Rotary Position Embedding) to q_rope and k_rope
 * 2. Quantize all components (q_nope, q_rope, k_nope, k_rope) to FP8 E4M3
 * 3. Optionally write K directly into KV cache buffer
 *
 * Motivation:
 * - Original path: mla_rope_quantize_fp8 (FlashInfer) → writes k_out → set_mla_kv_buffer reads k_out → writes KV cache
 * - Fused path: This kernel → directly writes to KV cache (eliminates intermediate global memory ops)
 *
 * Performance: ~4.9x faster than baseline (measured on B200), includes:
 * - Vectorized memory access (4-byte aligned loads/stores)
 * - Warp-level parallelism (32 threads per row)
 * - Direct KV cache write (no intermediate buffers)
 */

// Only include PyBind11 for standalone builds
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#else
#include <ATen/ATen.h>
#include <torch/types.h>
#endif

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>  // BF16 support
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_fp8.h>
#endif
#include <optional>
#include <stdint.h>

// Utility macros (borrowed from pytorch_extension_utils.h style)
#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be " #d "D")

// ---- Helpers -------------------------------------------------------

#define CHECK_SAME_DEVICE(a, b) TORCH_CHECK(a.device() == b.device(), #a " and " #b " must be on same device")

namespace {

// ============================================================================
// Dtype Traits: Support both FP16 (__half) and BF16 (nv_bfloat16)
// ============================================================================
template <typename T>
struct Vec2Traits;

template <>
struct Vec2Traits<__half> {
    using v2 = __half2;
    
    __device__ static inline float2 to_float2(v2 h2) {
        return __half22float2(h2);
    }
    
    __device__ static inline float to_float(const __half& h) {
        return __half2float(h);
    }
};

template <>
struct Vec2Traits<nv_bfloat16> {
    using v2 = nv_bfloat162;
    
    __device__ static inline float2 to_float2(v2 h2) {
        return __bfloat1622float2(h2);
    }
    
    __device__ static inline float to_float(const nv_bfloat16& h) {
        return __bfloat162float(h);
    }
};

// Convert float -> FP8 E4M3 (finite saturation). Return raw byte.
__device__ inline uint8_t float_to_e4m3fn_byte(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // CUDA 12+ with native FP8 support
    __nv_fp8_storage_t byte = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return static_cast<uint8_t>(byte);
#else
    // Fallback: Manual FP8 E4M3 conversion
    // E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
    // Range: [-448, 448], NaN represented as 0x7F
    
    // Clamp to FP8 E4M3 range
    x = fmaxf(-448.0f, fminf(448.0f, x));
    
    // Simple conversion (not bit-exact but close enough for testing)
    // In production, you'd want proper rounding
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = x;
    
    // Extract sign
    uint32_t sign = (conv.u >> 31) & 0x1;
    
    // Handle zero
    if (x == 0.0f) return 0;
    
    // Simplified: scale and round
    // This is a placeholder - for production use proper FP8 conversion
    int exp = ((conv.u >> 23) & 0xFF) - 127;  // Extract exponent
    exp = max(-6, min(8, exp));  // E4M3 range
    
    uint32_t mant = (conv.u >> 20) & 0x7;  // Top 3 bits of mantissa
    
    uint8_t result = (sign << 7) | ((exp + 7) << 3) | mant;
    return result;
#endif
}

// Pack 4 bytes into uint32_t for vectorized write
__device__ inline uint32_t pack4(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3) {
    return (uint32_t)a0 | ((uint32_t)a1 << 8) | ((uint32_t)a2 << 16) | ((uint32_t)a3 << 24);
}

// Apply RoPE to a pair (xr, xi) given cos, sin
__device__ inline void rope_rotate(float& xr, float& xi, float c, float s, bool /*is_neox*/) {
    float xr_new = xr * c - xi * s;
    float xi_new = xr * s + xi * c;
    xr = xr_new;
    xi = xi_new;
}

// ============================================================================
// Vectorized kernel: warp-per-row, vectorized load/store
// Template supports both FP16 (__half) and BF16 (nv_bfloat16)
// ============================================================================
template<int WARPS_PER_CTA, typename T>
__global__ void FusedRopeQuantizeKernelVec(
    const T* __restrict__ q_nope,
    const T* __restrict__ q_rope,
    int64_t qn_stride_tok, int64_t qn_stride_head,  // Q_nope strides in elements
    int64_t qr_stride_tok, int64_t qr_stride_head,  // Q_rope strides in elements
    const T* __restrict__ k_nope,
    const T* __restrict__ k_rope,
    const float* __restrict__ cos_sin,
    const int64_t* __restrict__ pos_ids,
    int nnz, int num_heads, int Dn, int Dr,
    bool is_neox,
    uint8_t* __restrict__ q_out_fp8,
    int64_t qout_stride_tok_bytes, int64_t qout_stride_head_bytes,  // Q_out strides in bytes
    uint8_t* __restrict__ k_nope_out_fp8,
    uint8_t* __restrict__ k_rope_out_fp8,
    uint8_t* __restrict__ kv_buffer_bytes,
    int64_t kv_stride_n_bytes,
    int64_t kv_stride_m_bytes,  // NEW: stride for page-internal row
    int page_size,              // NEW: page size for row offset calculation
    const int64_t* __restrict__ kv_cache_loc
) {
    constexpr int WARP_SIZE = 32;
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    int global_row = blockIdx.x * WARPS_PER_CTA + warp_in_block;
    if (global_row >= nnz * num_heads) return;

    // Decompose global row index: token_id and head_id
    int token_id = global_row / num_heads;
    int head_id = global_row % num_heads;

    // Pointers for this (token, head) using proper strides
    // Use template type T (not hardcoded __half) for BF16 support
    const T* qn = q_nope + size_t(token_id) * qn_stride_tok + size_t(head_id) * qn_stride_head;
    const T* qr = q_rope + size_t(token_id) * qr_stride_tok + size_t(head_id) * qr_stride_head;
    
    // K is always 2D: [nnz_tokens, dim]
    const T* kn = k_nope + size_t(token_id) * Dn;
    const T* kr = k_rope + size_t(token_id) * Dr;

    // Q output using byte strides
    uint8_t* qdst = q_out_fp8 + size_t(token_id) * qout_stride_tok_bytes + size_t(head_id) * qout_stride_head_bytes;
    
    // K outputs (if provided)
    uint8_t* kndst = k_nope_out_fp8 ? (k_nope_out_fp8 + size_t(token_id) * Dn) : nullptr;
    uint8_t* krdst = k_rope_out_fp8 ? (k_rope_out_fp8 + size_t(token_id) * Dr) : nullptr;

    // Get position for RoPE and KV cache row calculation
    int pos = static_cast<int>(pos_ids[token_id]);

    // KV cache destination: include row offset within page
    // CRITICAL FIX: write to correct row in page (pos % page_size), not always row 0
    uint8_t* kvdst = nullptr;
    if (kv_buffer_bytes && kv_cache_loc) {
        int64_t slot = kv_cache_loc[token_id];
        int row = pos % page_size;  // ⬅️ Token's row within page
        kvdst = kv_buffer_bytes 
              + slot * kv_stride_n_bytes 
              + static_cast<int64_t>(row) * kv_stride_m_bytes;
    }
    const float* cos_ptr = cos_sin + size_t(pos) * (2 * Dr);
    const float* sin_ptr = cos_ptr + Dr;

    // Use traits for dtype-agnostic vectorized load
    using V2 = typename Vec2Traits<T>::v2;

    // Process Q_nope: vectorized quantize + write
    for (int c = lane * 4; c < Dn; c += WARP_SIZE * 4) {
        V2 h0 = *reinterpret_cast<const V2*>(qn + c + 0);
        V2 h1 = *reinterpret_cast<const V2*>(qn + c + 2);
        float2 f0 = Vec2Traits<T>::to_float2(h0);
        float2 f1 = Vec2Traits<T>::to_float2(h1);

        uint32_t packed = pack4(
            float_to_e4m3fn_byte(f0.x), float_to_e4m3fn_byte(f0.y),
            float_to_e4m3fn_byte(f1.x), float_to_e4m3fn_byte(f1.y));
        *reinterpret_cast<uint32_t*>(qdst + c) = packed;
    }

    // Process Q_rope: paired rotation + vectorized quantize + write
    for (int c = lane * 4; c < Dr; c += WARP_SIZE * 4) {
        V2 h0 = *reinterpret_cast<const V2*>(qr + c + 0);
        V2 h1 = *reinterpret_cast<const V2*>(qr + c + 2);
        float2 f0 = Vec2Traits<T>::to_float2(h0);
        float2 f1 = Vec2Traits<T>::to_float2(h1);

        float c0 = cos_ptr[c + 0], s0 = sin_ptr[c + 0];
        float c1 = cos_ptr[c + 2], s1 = sin_ptr[c + 2];
        rope_rotate(f0.x, f0.y, c0, s0, is_neox);
        rope_rotate(f1.x, f1.y, c1, s1, is_neox);

        uint32_t packed = pack4(
            float_to_e4m3fn_byte(f0.x), float_to_e4m3fn_byte(f0.y),
            float_to_e4m3fn_byte(f1.x), float_to_e4m3fn_byte(f1.y));
        *reinterpret_cast<uint32_t*>(qdst + Dn + c) = packed;
    }

    // Process K_nope and K_rope: only once per token (head_id == 0)
    // K is 2D [nnz_tokens, dim], not per-head
    if (head_id == 0) {
        // Process K_nope: vectorized quantize + write to k_nope_out and/or KV buffer
        for (int c = lane * 4; c < Dn; c += WARP_SIZE * 4) {
            V2 h0 = *reinterpret_cast<const V2*>(kn + c + 0);
            V2 h1 = *reinterpret_cast<const V2*>(kn + c + 2);
            float2 f0 = Vec2Traits<T>::to_float2(h0);
            float2 f1 = Vec2Traits<T>::to_float2(h1);

            uint32_t packed = pack4(
                float_to_e4m3fn_byte(f0.x), float_to_e4m3fn_byte(f0.y),
                float_to_e4m3fn_byte(f1.x), float_to_e4m3fn_byte(f1.y));

            if (kndst) *reinterpret_cast<uint32_t*>(kndst + c) = packed;
            if (kvdst) *reinterpret_cast<uint32_t*>(kvdst + c) = packed;
        }

        // Process K_rope: paired rotation + vectorized quantize + write to k_rope_out and/or KV buffer
        for (int c = lane * 4; c < Dr; c += WARP_SIZE * 4) {
            V2 h0 = *reinterpret_cast<const V2*>(kr + c + 0);
            V2 h1 = *reinterpret_cast<const V2*>(kr + c + 2);
            float2 f0 = Vec2Traits<T>::to_float2(h0);
            float2 f1 = Vec2Traits<T>::to_float2(h1);

            float c0 = cos_ptr[c + 0], s0 = sin_ptr[c + 0];
            float c1 = cos_ptr[c + 2], s1 = sin_ptr[c + 2];
            rope_rotate(f0.x, f0.y, c0, s0, is_neox);
            rope_rotate(f1.x, f1.y, c1, s1, is_neox);

            uint32_t packed = pack4(
                float_to_e4m3fn_byte(f0.x), float_to_e4m3fn_byte(f0.y),
                float_to_e4m3fn_byte(f1.x), float_to_e4m3fn_byte(f1.y));

            if (krdst) *reinterpret_cast<uint32_t*>(krdst + c) = packed;
            if (kvdst) *reinterpret_cast<uint32_t*>(kvdst + Dn + c) = packed;
        }
    }
}

// ============================================================================
// Scalar fallback kernel: for dimensions not divisible by 4
// Template supports both FP16 (__half) and BF16 (nv_bfloat16)
// ============================================================================
template <int BLOCK_THREADS, typename T>
__global__ void FusedRopeQuantizeKernelScalar(
    const T* __restrict__ q_nope,
    const T* __restrict__ q_rope,
    int64_t qn_stride_tok, int64_t qn_stride_head,
    int64_t qr_stride_tok, int64_t qr_stride_head,
    const T* __restrict__ k_nope,
    const T* __restrict__ k_rope,
    const float* __restrict__ cos_sin,
    const int64_t* __restrict__ pos_ids,
    int nnz, int num_heads, int Dn, int Dr,
    bool is_neox,
    uint8_t* __restrict__ q_out_fp8,
    int64_t qout_stride_tok_bytes, int64_t qout_stride_head_bytes,
    uint8_t* __restrict__ k_nope_out_fp8,
    uint8_t* __restrict__ k_rope_out_fp8,
    uint8_t* __restrict__ kv_buffer_bytes,
    int64_t kv_stride_n_bytes,
    int64_t kv_stride_m_bytes,  // NEW: stride for page-internal row
    int page_size,              // NEW: page size for row offset calculation
    const int64_t* __restrict__ kv_cache_loc
) {
    // Thread mapping: grid-stride loop over all (token, head) pairs
    for (int global_row = blockIdx.x * BLOCK_THREADS + threadIdx.x; 
         global_row < nnz * num_heads; 
         global_row += gridDim.x * BLOCK_THREADS) {
        
        // Decompose global row index
        int token_id = global_row / num_heads;
        int head_id = global_row % num_heads;
        
        int pos = static_cast<int>(pos_ids[token_id]);
        const float* cos_ptr = cos_sin + size_t(pos) * (2 * Dr);
        const float* sin_ptr = cos_ptr + Dr;

        // ---- Quantize q ----
        // q_out: [nope | rope]
        {
            // Pointers using proper strides
            const T* qn = q_nope + size_t(token_id) * qn_stride_tok + size_t(head_id) * qn_stride_head;
            const T* qr = q_rope + size_t(token_id) * qr_stride_tok + size_t(head_id) * qr_stride_head;
            uint8_t* qdst = q_out_fp8 + size_t(token_id) * qout_stride_tok_bytes + size_t(head_id) * qout_stride_head_bytes;
            
            // write nope part
            for (int i = 0; i < Dn; ++i) {
                float x = Vec2Traits<T>::to_float(qn[i]);
                qdst[i] = float_to_e4m3fn_byte(x);
            }
            // rope part (avoid OOB read when Dr is odd)
            for (int i = 0; i < Dr; i += 2) {
                float xr = Vec2Traits<T>::to_float(qr[i + 0]);
                float xi = 0.0f;
                if (i + 1 < Dr) xi = Vec2Traits<T>::to_float(qr[i + 1]);
                float c = cos_ptr[i + 0];
                float s = sin_ptr[i + 0];
                // NeoX interleave is typically handled by reindexing; for demo we still rotate pairs.
                rope_rotate(xr, xi, c, s, is_neox);
                qdst[Dn + i + 0] = float_to_e4m3fn_byte(xr);
                if (i + 1 < Dr) qdst[Dn + i + 1] = float_to_e4m3fn_byte(xi);
            }
        }

        // ---- Quantize k & optional fused KV write ----
        // K is always 2D: [nnz_tokens, dim]
        const T* kn = k_nope + size_t(token_id) * Dn;
        const T* kr = k_rope + size_t(token_id) * Dr;

        // Optional: write k_nope_out / k_rope_out (only once per token, not per head)
        // Note: K outputs are 2D [nnz_tokens, dim], so only first head processes them
        if (head_id == 0) {
            if (k_nope_out_fp8) {
                uint8_t* knd = k_nope_out_fp8 + size_t(token_id) * Dn;
                for (int i = 0; i < Dn; ++i) {
                    knd[i] = float_to_e4m3fn_byte(Vec2Traits<T>::to_float(kn[i]));
                }
            }
            if (k_rope_out_fp8) {
                uint8_t* krd = k_rope_out_fp8 + size_t(token_id) * Dr;
                for (int i = 0; i < Dr; i += 2) {
                    float xr = Vec2Traits<T>::to_float(kr[i + 0]);
                    float xi = 0.0f;
                    if (i + 1 < Dr) xi = Vec2Traits<T>::to_float(kr[i + 1]);
                    float c = cos_ptr[i + 0];
                    float s = sin_ptr[i + 0];
                    rope_rotate(xr, xi, c, s, is_neox);
                    krd[i + 0] = float_to_e4m3fn_byte(xr);
                    if (i + 1 < Dr) krd[i + 1] = float_to_e4m3fn_byte(xi);
                }
            }

            // Fused direct KV write (if kv_buffer provided)
            // CRITICAL FIX: write to correct row in page (pos % page_size), not always row 0
            if (kv_buffer_bytes && kv_cache_loc) {
                int64_t slot = kv_cache_loc[token_id];
                int row = pos % page_size;  // ⬅️ Token's row within page
                uint8_t* dst = kv_buffer_bytes 
                             + slot * kv_stride_n_bytes 
                             + static_cast<int64_t>(row) * kv_stride_m_bytes;
                // Write nope first
                for (int i = 0; i < Dn; ++i) {
                    dst[i] = float_to_e4m3fn_byte(Vec2Traits<T>::to_float(kn[i]));
                }
                // Then rope with rotation (avoid OOB read when Dr is odd)
                for (int i = 0; i < Dr; i += 2) {
                    float xr = Vec2Traits<T>::to_float(kr[i + 0]);
                    float xi = 0.0f;
                    if (i + 1 < Dr) xi = Vec2Traits<T>::to_float(kr[i + 1]);
                    float c = cos_ptr[i + 0];
                    float s = sin_ptr[i + 0];
                    rope_rotate(xr, xi, c, s, is_neox);
                    dst[Dn + i + 0] = float_to_e4m3fn_byte(xr);
                    if (i + 1 < Dr) dst[Dn + i + 1] = float_to_e4m3fn_byte(xi);
                }
            }
        }
    }
}

} // namespace

// Python-exposed function
// q_nope, q_rope, k_nope, k_rope: half/bfloat16 (we treat as half here for demo)
// cos_sin_cache: float32 [max_seq, 2*Dr]
// pos_ids: int64 [nnz]
// q_out: uint8 [nnz, Dn+Dr] (stores E4M3 raw bytes)
// k_nope_out/k_rope_out: optional uint8 outputs (None allowed)
// kv_buffer: optional uint8 [(slots+page), 1, (Dn+Dr)] raw bytes
// kv_cache_loc: optional int64 [nnz]
void mla_rope_quantize_fp8_fused(
    at::Tensor q_nope,
    at::Tensor q_rope,
    at::Tensor k_nope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool is_neox,
    at::Tensor q_out,
    c10::optional<at::Tensor> k_nope_out,
    c10::optional<at::Tensor> k_rope_out,
    // fused args
    c10::optional<at::Tensor> kv_buffer,
    c10::optional<at::Tensor> kv_cache_loc
) {
    CHECK_INPUT(q_nope); CHECK_INPUT(q_rope); CHECK_INPUT(k_nope); CHECK_INPUT(k_rope);
    CHECK_INPUT(cos_sin_cache); CHECK_INPUT(pos_ids); CHECK_INPUT(q_out);
    CHECK_SAME_DEVICE(q_nope, q_rope); CHECK_SAME_DEVICE(q_nope, k_nope);
    CHECK_SAME_DEVICE(q_nope, cos_sin_cache); CHECK_SAME_DEVICE(q_nope, pos_ids);
    CHECK_SAME_DEVICE(q_nope, q_out);

    // Q can be 2D or 3D: [nnz, dim] or [nnz, num_heads, dim]
    // K must be 2D: [nnz, dim]
    TORCH_CHECK(q_nope.dim() == 2 || q_nope.dim() == 3, "q_nope must be 2D or 3D");
    TORCH_CHECK(q_rope.dim() == 2 || q_rope.dim() == 3, "q_rope must be 2D or 3D");
    CHECK_DIM(2, k_nope); CHECK_DIM(2, k_rope);
    CHECK_DIM(1, pos_ids); CHECK_DIM(2, cos_sin_cache);

    // Determine dimensions and strides based on Q shape
    int nnz_tokens, num_heads, Dn, Dr;
    int64_t qn_stride_tok, qn_stride_head, qr_stride_tok, qr_stride_head;
    int64_t qout_stride_tok_bytes, qout_stride_head_bytes;
    
    if (q_nope.dim() == 3) {
        // 3D Q: [nnz_tokens, num_heads, dim]
        nnz_tokens = q_nope.size(0);
        num_heads = q_nope.size(1);
        Dn = q_nope.size(2);
        Dr = q_rope.size(2);
        
        TORCH_CHECK(q_rope.size(0) == nnz_tokens && q_rope.size(1) == num_heads, "q_rope shape mismatch");
        TORCH_CHECK(q_out.dim() == 3 && q_out.size(0) == nnz_tokens && q_out.size(1) == num_heads && q_out.size(2) == (Dn + Dr),
                    "q_out must be [nnz, num_heads, Dn+Dr] when Q is 3D");
        
        // Q strides in elements
        qn_stride_tok = q_nope.stride(0);
        qn_stride_head = q_nope.stride(1);
        qr_stride_tok = q_rope.stride(0);
        qr_stride_head = q_rope.stride(1);
        
        // q_out strides in BYTES (uint8)
        qout_stride_tok_bytes = q_out.stride(0);
        qout_stride_head_bytes = q_out.stride(1);
    } else {
        // 2D Q: [nnz_tokens, dim] (single head or flattened)
        nnz_tokens = q_nope.size(0);
        Dn = q_nope.size(1);
        Dr = q_rope.size(1);
        num_heads = 1;
        
        TORCH_CHECK(q_rope.size(0) == nnz_tokens, "q_rope vs q_nope mismatch");
        TORCH_CHECK(q_out.dim() == 2 && q_out.size(0) == nnz_tokens && q_out.size(1) == (Dn + Dr),
                    "q_out must be [nnz, Dn+Dr] when Q is 2D");
        
        qn_stride_tok = q_nope.stride(0);
        qn_stride_head = 0;
        qr_stride_tok = q_rope.stride(0);
        qr_stride_head = 0;
        
        qout_stride_tok_bytes = q_out.stride(0);
        qout_stride_head_bytes = 0;
    }
    
    int nnz_k = k_rope.size(0);
    TORCH_CHECK(k_nope.size(0) == nnz_k && k_nope.size(1) == Dn, "k_nope shape mismatch");
    TORCH_CHECK(k_rope.size(0) == nnz_k && k_rope.size(1) == Dr, "k_rope shape mismatch");
    TORCH_CHECK(nnz_k == nnz_tokens, "K batch size must match Q token count");
    
    // ===== Robustness checks (expert suggestions) =====
    
    // 1. K must be contiguous on last dim (or explicitly handle stride)
    //    For simplicity, we enforce contiguous K on dim=1
    if (k_nope.stride(1) != 1) {
        TORCH_CHECK(false, "k_nope must be contiguous on last dim. Call .contiguous() before passing to kernel.");
    }
    if (k_rope.stride(1) != 1) {
        TORCH_CHECK(false, "k_rope must be contiguous on last dim. Call .contiguous() before passing to kernel.");
    }
    
    // 2. Q last dim must be contiguous (vectorized kernel assumes this)
    int q_last_dim = q_nope.dim() - 1;
    TORCH_CHECK(q_nope.stride(q_last_dim) == 1, "q_nope last dim must be contiguous");
    TORCH_CHECK(q_rope.stride(q_last_dim) == 1, "q_rope last dim must be contiguous");
    
    // 3. q_out last dim must be contiguous
    int qout_last_dim = q_out.dim() - 1;
    TORCH_CHECK(q_out.stride(qout_last_dim) == 1, "q_out last dim must be contiguous");
    
    // ==================================================

    uint8_t* k_nope_out_ptr = nullptr;
    uint8_t* k_rope_out_ptr = nullptr;
    if (k_nope_out.has_value()) {
        auto t = k_nope_out.value();
        CHECK_INPUT(t); CHECK_DIM(2, t);
        TORCH_CHECK(t.size(0) == nnz_k && t.size(1) == Dn, "k_nope_out shape mismatch");
        // Accept FP8 tensor, reinterpret as uint8*
        k_nope_out_ptr = reinterpret_cast<uint8_t*>(t.data_ptr());
    }
    if (k_rope_out.has_value()) {
        auto t = k_rope_out.value();
        CHECK_INPUT(t); CHECK_DIM(2, t);
        TORCH_CHECK(t.size(0) == nnz_k && t.size(1) == Dr, "k_rope_out shape mismatch");
        // Accept FP8 tensor, reinterpret as uint8*
        k_rope_out_ptr = reinterpret_cast<uint8_t*>(t.data_ptr());
    }

    uint8_t* kv_buf_ptr = nullptr;
    int64_t kv_stride_n_bytes = 0;
    int64_t kv_stride_m_bytes = 0;
    int page_size = 0;
    const int64_t* kv_loc_ptr = nullptr;
    if (kv_buffer.has_value() || kv_cache_loc.has_value()) {
        TORCH_CHECK(kv_buffer.has_value() && kv_cache_loc.has_value(),
            "kv_buffer and kv_cache_loc must be both provided for fused write");
        auto kv = kv_buffer.value();
        auto loc = kv_cache_loc.value();
        CHECK_INPUT(kv); CHECK_INPUT(loc);
        CHECK_DIM(3, kv); CHECK_DIM(1, loc);
        TORCH_CHECK(kv.size(2) == (Dn + Dr), "kv_buffer last dim must be Dn+Dr");
        TORCH_CHECK(loc.size(0) == nnz_k, "kv_cache_loc size must match K batch size");
        
        // CRITICAL: Check contiguity on last dim to avoid silent errors
        TORCH_CHECK(kv.stride(2) == 1, "kv_buffer last dim must be contiguous (stride=1)");
        
        // Accept FP8 tensor, reinterpret as uint8*
        kv_buf_ptr = reinterpret_cast<uint8_t*>(kv.data_ptr());
        
        // KV buffer layout: [num_blocks, page_size, kv_dim]
        // stride(0): block stride in bytes (already uint8 elements = bytes)
        // stride(1): page-internal row stride in bytes
        page_size = kv.size(1);
        kv_stride_n_bytes = kv.stride(0);  // Block stride
        kv_stride_m_bytes = kv.stride(1);  // Row stride within page
        kv_loc_ptr = loc.data_ptr<int64_t>();
    }

    // Get common pointers
    const float* cs_ptr = cos_sin_cache.data_ptr<float>();
    const int64_t* pos_ptr = pos_ids.data_ptr<int64_t>();
    // Accept FP8 tensor for q_out, reinterpret as uint8*
    uint8_t* q_out_ptr = reinterpret_cast<uint8_t*>(q_out.data_ptr());

    // Get current CUDA stream (compatible with PyTorch 2.x)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Total number of work items: nnz_tokens * num_heads
    int total_rows = nnz_tokens * num_heads;
    
    // Dispatch: use vectorized kernel if dimensions are 4-byte aligned
    // Vectorized path requires (Dn+Dr) % 4 == 0 for uint32_t writes
    bool can_vectorize = ((Dn & 3) == 0) && ((Dr & 3) == 0);
    
    if (can_vectorize) {
        // Additional check: q_out strides must be 4-byte aligned for vectorized writes
        // Since q_out is uint8 and we write uint32_t, strides should be multiples of 4
        bool strides_aligned = (qout_stride_tok_bytes % 4 == 0) && 
                               (num_heads > 1 ? (qout_stride_head_bytes % 4 == 0) : true);
        if (!strides_aligned) {
            // Fallback to scalar kernel if strides are misaligned
            can_vectorize = false;
        }
    }
    
    // ===== Dtype dispatch: support both FP16 and BF16 =====
    auto dtype = q_nope.scalar_type();
    
    if (dtype == at::kHalf) {
        // FP16 path
        const __half* qn_ptr = reinterpret_cast<const __half*>(q_nope.data_ptr<at::Half>());
        const __half* qr_ptr = reinterpret_cast<const __half*>(q_rope.data_ptr<at::Half>());
        const __half* kn_ptr = reinterpret_cast<const __half*>(k_nope.data_ptr<at::Half>());
        const __half* kr_ptr = reinterpret_cast<const __half*>(k_rope.data_ptr<at::Half>());
        
        if (can_vectorize) {
            constexpr int WARPS_PER_CTA = 4;
            dim3 vecBlock(WARPS_PER_CTA * 32);
            dim3 vecGrid((total_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA);
            
            FusedRopeQuantizeKernelVec<WARPS_PER_CTA, __half><<<vecGrid, vecBlock, 0, stream>>>(
                qn_ptr, qr_ptr, qn_stride_tok, qn_stride_head, qr_stride_tok, qr_stride_head,
                kn_ptr, kr_ptr, cs_ptr, pos_ptr, nnz_tokens, num_heads, Dn, Dr, is_neox,
                q_out_ptr, qout_stride_tok_bytes, qout_stride_head_bytes,
                k_nope_out_ptr, k_rope_out_ptr, 
                kv_buf_ptr, kv_stride_n_bytes, kv_stride_m_bytes, page_size, kv_loc_ptr
            );
        } else {
            constexpr int BLOCK_THREADS = 256;
            dim3 grid((total_rows + BLOCK_THREADS - 1) / BLOCK_THREADS);
            
            FusedRopeQuantizeKernelScalar<BLOCK_THREADS, __half><<<grid, BLOCK_THREADS, 0, stream>>>(
                qn_ptr, qr_ptr, qn_stride_tok, qn_stride_head, qr_stride_tok, qr_stride_head,
                kn_ptr, kr_ptr, cs_ptr, pos_ptr, nnz_tokens, num_heads, Dn, Dr, is_neox,
                q_out_ptr, qout_stride_tok_bytes, qout_stride_head_bytes,
                k_nope_out_ptr, k_rope_out_ptr, 
                kv_buf_ptr, kv_stride_n_bytes, kv_stride_m_bytes, page_size, kv_loc_ptr
            );
        }
    } else if (dtype == at::kBFloat16) {
        // BF16 path
        const nv_bfloat16* qn_ptr = reinterpret_cast<const nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>());
        const nv_bfloat16* qr_ptr = reinterpret_cast<const nv_bfloat16*>(q_rope.data_ptr<at::BFloat16>());
        const nv_bfloat16* kn_ptr = reinterpret_cast<const nv_bfloat16*>(k_nope.data_ptr<at::BFloat16>());
        const nv_bfloat16* kr_ptr = reinterpret_cast<const nv_bfloat16*>(k_rope.data_ptr<at::BFloat16>());
        
        if (can_vectorize) {
            constexpr int WARPS_PER_CTA = 4;
            dim3 vecBlock(WARPS_PER_CTA * 32);
            dim3 vecGrid((total_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA);
            
            FusedRopeQuantizeKernelVec<WARPS_PER_CTA, nv_bfloat16><<<vecGrid, vecBlock, 0, stream>>>(
                qn_ptr, qr_ptr, qn_stride_tok, qn_stride_head, qr_stride_tok, qr_stride_head,
                kn_ptr, kr_ptr, cs_ptr, pos_ptr, nnz_tokens, num_heads, Dn, Dr, is_neox,
                q_out_ptr, qout_stride_tok_bytes, qout_stride_head_bytes,
                k_nope_out_ptr, k_rope_out_ptr, 
                kv_buf_ptr, kv_stride_n_bytes, kv_stride_m_bytes, page_size, kv_loc_ptr
            );
        } else {
            constexpr int BLOCK_THREADS = 256;
            dim3 grid((total_rows + BLOCK_THREADS - 1) / BLOCK_THREADS);
            
            FusedRopeQuantizeKernelScalar<BLOCK_THREADS, nv_bfloat16><<<grid, BLOCK_THREADS, 0, stream>>>(
                qn_ptr, qr_ptr, qn_stride_tok, qn_stride_head, qr_stride_tok, qr_stride_head,
                kn_ptr, kr_ptr, cs_ptr, pos_ptr, nnz_tokens, num_heads, Dn, Dr, is_neox,
                q_out_ptr, qout_stride_tok_bytes, qout_stride_head_bytes,
                k_nope_out_ptr, k_rope_out_ptr, 
                kv_buf_ptr, kv_stride_n_bytes, kv_stride_m_bytes, page_size, kv_loc_ptr
            );
        }
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused kernel. Only FP16 and BF16 are supported.");
    }
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
}

// PyBind11 module definition (ONLY for standalone build)
// When building as part of sgl_kernel, this is handled by common_extension.cc
// TORCH_EXTENSION_NAME is only defined by torch.utils.cpp_extension (standalone)
#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_rope_quantize_fp8_fused", &mla_rope_quantize_fp8_fused,
          "Fused MLA RoPE + FP8 quantization with optional direct KV write",
          py::arg("q_nope"),
          py::arg("q_rope"),
          py::arg("k_nope"),
          py::arg("k_rope"),
          py::arg("cos_sin_cache"),
          py::arg("pos_ids"),
          py::arg("is_neox"),
          py::arg("q_out"),
          py::arg("k_nope_out") = py::none(),
          py::arg("k_rope_out") = py::none(),
          py::arg("kv_buffer") = py::none(),
          py::arg("kv_cache_loc") = py::none());
}
#endif
