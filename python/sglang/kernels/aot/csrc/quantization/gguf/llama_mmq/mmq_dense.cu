// Host entry for the vendored llama.cpp MMQ kernels: a plain 2D matmul on raw GGUF blocks.
// Mirrors the dense (non-ids) path of ggml_cuda_mul_mat_q() in llama.cpp's ggml/src/ggml-cuda/mmq.cu
// (commit in ggml/LLAMA_CPP_COMMIT) with ne02 = ne03 = ne12 = ne13 = 1 and the activations always
// quantized to q8_1 (GGML_PREC_Q8). The mul_mat_q_case<type> instances are compiled from the vendored
// ggml/src/ggml-cuda/template-instances/mmq-instance-*.cu files, built from a copy of ggml/ with the patches in
// patches/ applied (0001: dst in f32, f16 or bf16).
// Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors, see ggml/LICENSE).

#include <climits>
#include <mutex>

#include "common.cuh"
#include "mmq.cuh"
#include "sgl_llama_mmq.h"

namespace sgl_llama_mmq {
extern thread_local cudaStream_t g_cur_stream;
}

// The GGML types SGLang dispatches to MMQ; one template-instances/mmq-instance-<type>.cu each.
#define SGL_LLAMA_MMQ_TYPES(X) \
  X(Q4_0)                      \
  X(Q4_1)                      \
  X(Q5_0)                      \
  X(Q5_1)                      \
  X(Q8_0)                      \
  X(Q2_K)                      \
  X(Q3_K)                      \
  X(Q4_K)                      \
  X(Q5_K)                      \
  X(Q6_K)                      \
  X(IQ1_S)                     \
  X(IQ1_M)                     \
  X(IQ2_XXS)                   \
  X(IQ2_XS)                    \
  X(IQ2_S)                     \
  X(IQ3_XXS)                   \
  X(IQ3_S)                     \
  X(IQ4_NL)                    \
  X(IQ4_XS)

namespace sgl_llama_mmq {

bool type_supported(int64_t type) {
  switch (type) {
#define X(T) case GGML_TYPE_##T:
    SGL_LLAMA_MMQ_TYPES(X)
#undef X
    return true;
    default:
      return false;
  }
}

bool type_info(int64_t type, int64_t* qk, int64_t* block_bytes) {
  switch (type) {
#define X(T)                                                 \
  case GGML_TYPE_##T:                                        \
    *qk = ggml_cuda_type_traits<GGML_TYPE_##T>::qk;          \
    *block_bytes = ggml_cuda_type_traits<GGML_TYPE_##T>::bs; \
    return true;
    SGL_LLAMA_MMQ_TYPES(X)
#undef X
    default:
      return false;
  }
}

int64_t weight_tail_bytes(int64_t type, int64_t K) {
  int64_t qk, bs;
  GGML_ASSERT(type_info(type, &qk, &bs));
  GGML_ASSERT(K % qk == 0);
  // ggml_backend_cuda_buffer_type_get_alloc_size(): row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)
  if (K % MATRIX_ROW_PADDING == 0) {
    return 0;
  }
  return (MATRIX_ROW_PADDING - K % MATRIX_ROW_PADDING) / qk * bs;
}

// Largest over-read of the q8_1 buffer: the kernel copies J = 128 columns of block_q8_1_mmq in chunks of
// nthreads (at most 512) ints, as in the src1 padding of ggml_cuda_mul_mat_q().
static constexpr int64_t q8_tail_bytes() {
  constexpr int64_t chunk = 512 * sizeof(int);
  return ((128 * (int64_t)sizeof(block_q8_1_mmq) + chunk - 1) / chunk) * chunk;
}

int64_t q8_buffer_bytes(int64_t M, int64_t K) {
  return M * GGML_PAD(K, MATRIX_ROW_PADDING) * (int64_t)sizeof(block_q8_1_mmq) / QK8_1_MMQ + q8_tail_bytes();
}

static ggml_backend_cuda_context& device_context(int device) {
  static std::mutex mtx;
  static ggml_backend_cuda_context* ctxs[GGML_CUDA_MAX_DEVICES] = {nullptr};
  std::lock_guard<std::mutex> lock(mtx);
  if (ctxs[device] == nullptr) {
    ctxs[device] = new ggml_backend_cuda_context(device);  // never freed: must outlive torch teardown
  }
  return *ctxs[device];
}

static void mul_mat_q_switch_type(ggml_backend_cuda_context& ctx, const mmq_args& args, cudaStream_t stream) {
  switch (args.type_x) {
#define X(T)                                                        \
  case GGML_TYPE_##T:                                               \
    mul_mat_q_case<GGML_TYPE_##T, GGML_PREC_Q8>(ctx, args, stream); \
    break;
    SGL_LLAMA_MMQ_TYPES(X)
#undef X
    default:
      GGML_ABORT("GGML type %d is not supported by MMQ", (int)args.type_x);
  }
}

void mul_mat(
    const void* w,
    int64_t rows,
    int64_t K,
    int64_t ldw_bytes,
    int64_t type_i,
    const void* yq,
    int64_t M,
    void* dst,
    XType dst_type,
    int64_t ld_dst,
    cudaStream_t stream) {
  const ggml_type type = (ggml_type)type_i;
  int64_t qk, bs;
  GGML_ASSERT(type_info(type, &qk, &bs));
  GGML_ASSERT(K % qk == 0);
  GGML_ASSERT(ldw_bytes % bs == 0);
  GGML_ASSERT(ld_dst >= rows);

  const int id = ggml_cuda_get_device();
  if (id >= ggml_cuda_info().device_count) {
    GGML_ABORT("device %d: llama.cpp MMQ supports at most %d devices", id, GGML_CUDA_MAX_DEVICES);
  }
  const int cc = ggml_cuda_info().devices[id].cc;
  const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

  constexpr ggml_prec prec_src1 = GGML_PREC_Q8;
  const bool fallback = ggml_cuda_mmq_needs_fallback(rows);

  // Tile width along M: the J with the fewest tiles that fits in shared memory (ggml_cuda_mul_mat_q).
  int J_best = 0;
  {
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
      const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
      if (config.type == GGML_TYPE_COUNT) {
        continue;
      }
      if (mmq_get_nbytes_shared(config, cc) > smpbo) {
        continue;
      }
      const int ntiles_x = (M + config.J - 1) / config.J;
      if (ntiles_x < ntiles_J_best) {
        J_best = J;
        ntiles_J_best = ntiles_x;
      }
    }
  }
  GGML_ASSERT(J_best > 0);

  const int64_t K_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
  const int64_t s01 = ldw_bytes / bs;  // weight row stride in blocks
  const int64_t s02 = rows * s01;
  const int64_t s2 = M * ld_dst;
  const int64_t s12 = M * K_padded * (int64_t)sizeof(block_q8_1) / (QK8_1 * (int64_t)sizeof(int));

  const mmq_args args = {/*x=*/(const char*)w,
                         /*type_x=*/type,
                         /*y=*/(const int*)yq,
                         /*ids_dst=*/nullptr,
                         /*expert_bounds=*/nullptr,
                         /*dst=*/(float*)dst,
                         /*y_scale=*/nullptr,
                         /*ncols_x=*/K,
                         /*nrows_x=*/rows,
                         /*ncols_dst=*/M,
                         /*stride_row_x=*/s01,
                         /*ncols_y=*/M,
                         /*nrows_dst=*/ld_dst,
                         /*nchannels_x=*/1,
                         /*nchannels_y=*/1,
                         /*stride_channel_x=*/s02,
                         /*stride_channel_y=*/s12,
                         /*stride_channel_dst=*/s2,
                         /*nsamples_x=*/1,
                         /*nsamples_y=*/1,
                         /*stride_sample_x=*/s02,
                         /*stride_sample_y=*/s12,
                         /*stride_sample_dst=*/s2,
                         /*ncols_max=*/M,
                         /*J_best=*/J_best,
                         /*type_dst=*/dst_type == XType::F32 ? GGML_TYPE_F32
                         : dst_type == XType::F16            ? GGML_TYPE_F16
                                                             : GGML_TYPE_BF16};

  g_cur_stream = stream;
  mul_mat_q_switch_type(device_context(id), args, stream);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgl_llama_mmq
