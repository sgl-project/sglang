// Interface between SGLang's GGUF ops and the llama.cpp MMQ kernels vendored under ggml/.
//
// Neither torch nor ggml headers are included here: the ggml headers (ggml-common.h, ...) define the
// same block structs as SGLang's own csrc/quantization/gguf/ggml-common.h, so the two code bases live in
// separate translation units and only talk through this header.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sgl_llama_mmq {

using alloc_fn = void* (*)(size_t nbytes, cudaStream_t stream);
using free_fn = void (*)(void* ptr);

// Scratch memory of the kernels (stream-k fixup buffer) is taken from this allocator; the torch side
// installs the c10 CUDA caching allocator. Must be called before mul_mat().
void set_allocator(alloc_fn alloc, free_fn free);

// True if MMQ kernels for this GGML type are compiled into this build.
bool type_supported(int64_t type);

// Values per block and bytes per block of a GGML type supported by MMQ; false for other types.
bool type_info(int64_t type, int64_t* qk, int64_t* block_bytes);

// Bytes the kernel may read past the end of the last weight row for a row length of K values:
// ggml pads the last row to a multiple of MATRIX_ROW_PADDING (512) values. These bytes must be
// readable and finite (zeros); 0 when K % 512 == 0.
int64_t weight_tail_bytes(int64_t type, int64_t K);

// Bytes of the q8_1 activation buffer for M rows of K values, including the over-read padding of the kernel.
int64_t q8_buffer_bytes(int64_t M, int64_t K);

enum class XType : int { F32 = 0, F16 = 1, BF16 = 2 };

// Quantizes x [M, K] (row stride ldx elements, unit column stride) to the block_q8_1_mmq layout that
// mul_mat() expects for weights of `type`.
void quantize_q8_1_mmq(
    const void* x, XType xtype, int64_t M, int64_t K, int64_t ldx, int64_t type, void* yq, cudaStream_t stream);

// dst[M, rows] (row stride ld_dst elements of dst_type) = yq[M, K] @ W[rows, K]^T, accumulated in fp32 and
// rounded once to dst_type (round to nearest even). W rows are raw GGUF blocks with a row stride of ldw_bytes
// (a multiple of the block size); the weight storage must hold weight_tail_bytes() past the last row.
void mul_mat(
    const void* w,
    int64_t rows,
    int64_t K,
    int64_t ldw_bytes,
    int64_t type,
    const void* yq,
    int64_t M,
    void* dst,
    XType dst_type,
    int64_t ld_dst,
    cudaStream_t stream);

// ---------------------------------------------------------------------------------------------
// MoE (llama.cpp's MUL_MAT_ID path, mmq_moe.cu): dst[t, s, :] = W[expert(t, s)] @ x(t, s) for the n_used slots s of
// each token t, with x(t, s) = x[t] (x_per_slot = false, the gate/up projection: each token is quantized once) or
// x[t * n_used + s] (x_per_slot = true, the down projection).

// Largest number of tokens per moe_mul_mat() call: mm_ids_helper keeps 4 bytes per token in shared memory
// (smpbo / 4), and the kernels index the output and the q8_1 buffer with 32-bit ints.
int64_t moe_max_tokens_per_call(int64_t n_used, int64_t rows, int64_t K);

// topk_ids [n_slots / n_used, n_used] (int32 or int64) -> ids32 [n_slots] int32 with masked / out-of-range ids
// (< 0 or >= n_experts) and repeats of an expert within a token replaced by INT_MAX, which matches no expert; the
// dst rows [n_slots, rows] of those slots are zeroed (no tile writes them).
void moe_prepare_ids(
    const void* topk_ids,
    bool ids_int64,
    int64_t n_slots,
    int64_t n_used,
    int64_t n_experts,
    int32_t* ids32,
    float* dst,
    int64_t rows,
    cudaStream_t stream);

// One call for n_tokens <= moe_max_tokens_per_call() tokens. W: n_experts x rows x row blocks with row and expert
// strides in bytes (multiples of the block size); the storage must hold weight_tail_bytes() past the last row of
// the last expert. x: [n_tokens, K] or [n_tokens * n_used, K] (x_per_slot), row stride ldx elements. ids32: this
// call's [n_tokens, n_used] slice of moe_prepare_ids(). Scratch: ids_src1 [n_tokens * n_used], ids_dst
// [n_tokens * n_used + 128], expert_bounds [n_experts + 1] int32, yq q8_buffer_bytes(n_tokens * n_used, K).
// dst: [n_tokens, n_used, rows] fp32; rows of masked slots are not written.
void moe_mul_mat(
    const void* w,
    int64_t n_experts,
    int64_t rows,
    int64_t K,
    int64_t ldw_row_bytes,
    int64_t ldw_expert_bytes,
    int64_t type,
    const void* x,
    XType xtype,
    int64_t ldx,
    bool x_per_slot,
    const int32_t* ids32,
    int64_t n_tokens,
    int64_t n_used,
    int32_t* ids_src1,
    int32_t* ids_dst,
    int32_t* expert_bounds,
    void* yq,
    float* dst,
    cudaStream_t stream);

}  // namespace sgl_llama_mmq
