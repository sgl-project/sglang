// Host entry for the MoE path of the vendored llama.cpp MMQ kernels: the MUL_MAT_ID branch of
// ggml_cuda_mul_mat_q() in llama.cpp's ggml/src/ggml-cuda/mmq.cu (commit in ggml/LLAMA_CPP_COMMIT), with the
// expert ids sorted on the GPU by the vendored mm_ids_helper (ggml/src/ggml-cuda/mmid.cu): no host sync, CUDA
// graph capturable. Masked routes (id < 0, e.g. expert parallelism) are skipped and their output rows zeroed;
// so is a route that repeats an expert of the same token (mm_ids_helper sorts distinct experts per token).
// Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors, see ggml/LICENSE).

#include <algorithm>
#include <climits>

#include "common.cuh"
#include "mmid.cuh"
#include "mmq.cuh"
#include "mmq_internal.cuh"
#include "sgl_llama_mmq.h"

namespace sgl_llama_mmq {

int64_t moe_max_tokens_per_call(int64_t n_used, int64_t rows, int64_t K) {
  const int id = current_device();
  const int64_t smpbo = (int64_t)ggml_cuda_info().devices[id].smpbo;
  int64_t t = smpbo / (int64_t)sizeof(uint32_t);  // mm_ids_helper: one mm_ids_helper_store per token
  t = std::min<int64_t>(t, (1 << 22) - 1);        // 22 bits for the token index in mm_ids_helper_store
  // dst offsets (slot * rows) and q8_1 offsets (about slot * K_padded / 4 ints) are 32-bit in the kernels.
  const int64_t width = std::max<int64_t>(rows, GGML_PAD(K, MATRIX_ROW_PADDING));
  t = std::min<int64_t>(t, (int64_t)(INT_MAX - 1024) / (std::max<int64_t>(n_used, 1) * width));
  GGML_ASSERT(t > 0);
  return t;
}

template <typename id_t>
static __global__ void moe_prepare_ids_kernel(
    const id_t* __restrict__ ids,
    int32_t* __restrict__ ids32,
    float* __restrict__ dst,
    const int64_t rows,
    const int64_t n_experts,
    const int n_used) {
  const int64_t slot = blockIdx.x;
  const int64_t e = (int64_t)ids[slot];
  bool valid = e >= 0 && e < n_experts;
  // mm_ids_helper assumes distinct experts per token (as top-k gives); a repeated expert is masked.
  const int64_t first = slot - slot % n_used;
  for (int64_t j = first; valid && j < slot; ++j) {
    valid = (int64_t)ids[j] != e;
  }
  if (threadIdx.x == 0) {
    ids32[slot] = valid ? (int32_t)e : INT_MAX;
  }
  if (!valid) {
    for (int64_t i = threadIdx.x; i < rows; i += blockDim.x) {
      dst[slot * rows + i] = 0.0f;
    }
  }
}

void moe_prepare_ids(
    const void* topk_ids,
    bool ids_int64,
    int64_t n_slots,
    int64_t n_used,
    int64_t n_experts,
    int32_t* ids32,
    float* dst,
    int64_t rows,
    cudaStream_t stream) {
  if (n_slots == 0) {
    return;
  }
  GGML_ASSERT(n_slots <= INT_MAX && n_used > 0 && n_slots % n_used == 0);
  if (ids_int64) {
    moe_prepare_ids_kernel<int64_t>
        <<<n_slots, 128, 0, stream>>>((const int64_t*)topk_ids, ids32, dst, rows, n_experts, (int)n_used);
  } else {
    moe_prepare_ids_kernel<int32_t>
        <<<n_slots, 128, 0, stream>>>((const int32_t*)topk_ids, ids32, dst, rows, n_experts, (int)n_used);
  }
  CUDA_CHECK(cudaGetLastError());
}

void moe_mul_mat(
    const void* w,
    int64_t n_experts,
    int64_t rows,
    int64_t K,
    int64_t ldw_row_bytes,
    int64_t ldw_expert_bytes,
    int64_t type_i,
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
    cudaStream_t stream) {
  const ggml_type type = (ggml_type)type_i;
  int64_t qk, bs;
  GGML_ASSERT(type_info(type, &qk, &bs));
  GGML_ASSERT(K % qk == 0);
  GGML_ASSERT(ldw_row_bytes % bs == 0 && ldw_expert_bytes % bs == 0);
  GGML_ASSERT(n_tokens > 0 && n_used > 0 && n_experts > 0);
  GGML_ASSERT(n_tokens <= moe_max_tokens_per_call(n_used, rows, K));

  const int id = current_device();
  const int64_t n_slots = n_tokens * n_used;

  // Sort the slots by expert: expert_bounds, ids_dst (compact column -> output slot) and ids_src1 (compact
  // column -> input row, or with write_inverse the slot -> compact column map for the scatter quantizer).
  ggml_cuda_launch_mm_ids_helper(
      ids32,
      ids_src1,
      ids_dst,
      expert_bounds,
      (int)n_experts,
      (int)n_tokens,
      (int)n_used,
      /*nchannels_y=*/x_per_slot ? (int)n_used : 1,
      /*si1=*/(int)n_used,
      /*sis1=*/x_per_slot ? (int)n_used : 1,
      /*write_inverse=*/!x_per_slot,
      stream);
  CUDA_CHECK(cudaGetLastError());

  if (x_per_slot) {
    quantize_q8_1_mmq_gather(x, xtype, n_slots, K, ldx, type, ids_src1, expert_bounds + n_experts, yq, stream);
  } else {
    quantize_q8_1_mmq_scatter(x, xtype, n_tokens, (int)n_used, K, ldx, type, ids_src1, ids32, yq, stream);
  }

  // As in ggml_cuda_mul_mat_q for MUL_MAT_ID: ne02 experts as channels, the compact columns sorted by expert.
  const int J_best = select_tile_J(type, rows, n_tokens, id);
  const int64_t K_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
  const int64_t ne11 = x_per_slot ? n_used : 1;
  const int64_t s01 = ldw_row_bytes / bs;
  const int64_t s02 = ldw_expert_bytes / bs;
  const int64_t s12 = ne11 * K_padded * (int64_t)sizeof(block_q8_1) / (QK8_1 * (int64_t)sizeof(int));
  const int64_t s2 = n_used * rows;

  const mmq_args args = {/*x=*/(const char*)w,
                         /*type_x=*/type,
                         /*y=*/(const int*)yq,
                         /*ids_dst=*/ids_dst,
                         /*expert_bounds=*/expert_bounds,
                         /*dst=*/dst,
                         /*y_scale=*/nullptr,
                         /*ncols_x=*/K,
                         /*nrows_x=*/rows,
                         /*ncols_dst=*/n_slots,
                         /*stride_row_x=*/s01,
                         /*ncols_y=*/n_slots,
                         /*nrows_dst=*/rows,
                         /*nchannels_x=*/n_experts,
                         /*nchannels_y=*/n_experts,
                         /*stride_channel_x=*/s02,
                         /*stride_channel_y=*/s12,
                         /*stride_channel_dst=*/s2,
                         /*nsamples_x=*/1,
                         /*nsamples_y=*/1,
                         /*stride_sample_x=*/n_experts * s02,
                         /*stride_sample_y=*/n_tokens * s12,
                         /*stride_sample_dst=*/n_tokens * s2,
                         /*ncols_max=*/n_tokens,
                         /*J_best=*/J_best};

  g_cur_stream = stream;
  mul_mat_q_switch_type(device_context(id), args, stream);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgl_llama_mmq
