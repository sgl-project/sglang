// Host helpers shared by mmq_dense.cu and mmq_moe.cu (ggml types; not part of sgl_llama_mmq.h).
#pragma once

#include "common.cuh"
#include "mmq.cuh"
#include "sgl_llama_mmq.h"

namespace sgl_llama_mmq {

// Stream of the launch in progress on this thread; ggml_cuda_pool allocations are made on it (ggml_shim.cu).
extern thread_local cudaStream_t g_cur_stream;

// The current device, checked against GGML_CUDA_MAX_DEVICES.
int current_device();

// One ggml_backend_cuda_context per device (its memory pool is the torch caching allocator).
ggml_backend_cuda_context& device_context(int device);

// Tile width along the columns (tokens) as chosen by ggml_cuda_mul_mat_q for ncols columns.
int select_tile_J(ggml_type type, int64_t rows, int64_t ncols, int device);

// mul_mat_q_case<type, GGML_PREC_Q8> for the types SGLang compiles.
void mul_mat_q_switch_type(ggml_backend_cuda_context& ctx, const mmq_args& args, cudaStream_t stream);

// MoE variants of quantize_q8_1_mmq() (quantize_mmq_q8_1.cu), both writing a q8_1 buffer of ncols columns:
// gather: column c < *n_valid from input row ids_src[c] (ncols = n_tokens * n_used);
// scatter: input row t (one per token) to columns slot_col[t * n_used + s] of its slots with a valid expert
// (slot_expert != INT_MAX), so that a token shared by its n_used slots is quantized once.
void quantize_q8_1_mmq_gather(
    const void* x,
    XType xtype,
    int64_t ncols,
    int64_t K,
    int64_t ldx,
    int64_t type,
    const int32_t* ids_src,
    const int32_t* n_valid,
    void* yq,
    cudaStream_t stream);
void quantize_q8_1_mmq_scatter(
    const void* x,
    XType xtype,
    int64_t n_tokens,
    int n_used,
    int64_t K,
    int64_t ldx,
    int64_t type,
    const int32_t* slot_col,
    const int32_t* slot_expert,
    void* yq,
    cudaStream_t stream);

}  // namespace sgl_llama_mmq
