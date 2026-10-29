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

}  // namespace sgl_llama_mmq
