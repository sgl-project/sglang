/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Varlen = true>
struct BlockInfo {
  template <typename Params>
  __device__ BlockInfo(const Params& params, const int bidb)
      : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb]),
        sum_s_k(
            !Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1
                                                                                         : params.cu_seqlens_k[bidb]),
        sum_s_v(
            !Varlen || params.cu_seqlens_v == nullptr || !params.is_seqlens_v_cumulative ? -1
                                                                                         : params.cu_seqlens_v[bidb]),
        actual_seqlen_q(
            !Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        ,
        leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]),
        seqlen_k_cache(
            (!Varlen || params.cu_seqlens_k == nullptr
                 ? params.seqlen_k
                 : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k
                                                   : params.cu_seqlens_k[bidb])) -
            leftpad_k),
        actual_seqlen_k(
            params.seqused_k ? params.seqused_k[bidb] - leftpad_k
                             : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        // If is_seqlens_v_cumulative, then seqlen_v is cu_seqlens_v[bidb + 1] - cu_seqlens_v[bidb].
        // Otherwise it's cu_seqlens_v[bidb], i.e., we use cu_seqlens_v to store the sequence lengths of V.
        ,
        leftpad_v(params.leftpad_v == nullptr ? 0 : params.leftpad_v[bidb]),
        seqlen_v_cache(
            (!Varlen || params.cu_seqlens_v == nullptr
                 ? params.seqlen_v
                 : (params.is_seqlens_v_cumulative ? params.cu_seqlens_v[bidb + 1] - sum_s_v
                                                   : params.cu_seqlens_v[bidb])) -
            leftpad_v),
        actual_seqlen_c(
            params.seqused_v ? params.seqused_v[bidb] - leftpad_v
                             : seqlen_v_cache + (params.vnew_ptr == nullptr ? 0 : params.seqlen_vnew)) {}

  template <typename index_t>
  __forceinline__ __device__ index_t
  q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
    return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t
  k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
    return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t
  v_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
    return sum_s_v == -1 ? bidb * batch_stride + leftpad_v * row_stride : uint32_t(sum_s_v + leftpad_v) * row_stride;
  }

  template <typename index_t>
  inline __device__ index_t blockmask_q_offset(const index_t m_block_dim, const int bidb) const {
    return sum_s_q == -1 ? bidb * (actual_seqlen_q / m_block_dim) : uint32_t(sum_s_q) / m_block_dim;
  }

  const int sum_s_q;
  const int sum_s_k;
  const int sum_s_v;
  const int actual_seqlen_q;
  // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
  const int leftpad_k;
  const int seqlen_k_cache;
  const int actual_seqlen_k;
  // We have to have seqlen_v_cache declared before actual_seqlen_c, otherwise actual_seqlen_c is set to 0.
  const int leftpad_v;
  const int seqlen_v_cache;
  const int actual_seqlen_c;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
