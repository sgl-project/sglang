/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
#include <cuda.h>

#include <vector>

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
  using index_t = int64_t;
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;

  // The number of heads.
  int h, h_k;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
  // different from nheads (query).
  int h_h_k_ratio;  // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
  // The O matrix (output).
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the P matrix.
  void* __restrict__ p_ptr;

  // The pointer to the softmax sum.
  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ softmax_lseaccum_ptr;

  // The dimensions.
  int b, seqlen_q, seqlen_k, seqlen_v, seqlen_knew, seqlen_vnew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded,
      rotary_dim, total_q;

  // The scaling factors for the kernel.
  float scale_softmax;
  float scale_softmax_log2;

  // array of length b+1 holding starting offset of each sequence.
  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;
  int* __restrict__ cu_seqlens_v;
  int* __restrict__ leftpad_k;
  int* __restrict__ leftpad_v;

  // If provided, the actual length of each k sequence.
  int* __restrict__ seqused_k;
  int* __restrict__ seqused_v;
  uint64_t* __restrict__ blockmask;
  int m_block_dim, n_block_dim, num_k_heads;
  int num_blocks_m, num_blocks_n;

  // The K_new and V_new matrices.
  void* __restrict__ knew_ptr;
  void* __restrict__ vnew_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t knew_batch_stride;
  index_t vnew_batch_stride;
  index_t knew_row_stride;
  index_t vnew_row_stride;
  index_t knew_head_stride;
  index_t vnew_head_stride;

  // The cos and sin matrices for rotary embedding.
  void* __restrict__ rotary_cos_ptr;
  void* __restrict__ rotary_sin_ptr;

  // The indices to index into the KV cache.
  int* __restrict__ cache_batch_idx;

  // Paged KV cache
  int* __restrict__ block_table;
  index_t block_table_batch_stride;
  int page_block_size;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint32_t p_dropout_in_uint;
  // uint16_t p_dropout_in_uint16_t;
  uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;
  float scale_softmax_rp_dropout;

  // Local window size
  int window_size_left, window_size_right;
  float softcap;

  // Random state.
  at::PhiloxCudaState philox_args;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;

  bool is_bf16;
  bool is_causal;

  // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
  // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
  bool is_seqlens_k_cumulative;

  // If is_seqlens_v_cumulative, then seqlen_v is cu_seqlens_v[bidb + 1] - cu_seqlens_v[bidb].
  // Otherwise it's cu_seqlens_v[bidb], i.e., we use cu_seqlens_v to store the sequence lengths of V.
  bool is_seqlens_v_cumulative;

  bool is_rotary_interleaved;

  int num_splits;  // For split-KV version

  void* __restrict__ alibi_slopes_ptr;
  index_t alibi_slopes_batch_stride;

  bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
  bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv,
                                 // d).
};

template <typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params& params, cudaStream_t stream);
