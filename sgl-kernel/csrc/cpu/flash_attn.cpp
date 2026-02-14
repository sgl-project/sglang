/*****************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************************/
#include "flash_attn.h"

#include "common.h"
#include "gemm.h"

// [NOTE]: flash attention interface for CPU

namespace {

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
void flash_attn_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    void* __restrict__ buffer,
    int seqlen_q,
    int seqlen_k,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float sm_scale,
    int buffer_size_per_thread,
    bool causal) {
  // strides
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // we use same buffer for packed key and value
  const int ldb_tmp = std::max(head_size, head_size_v);

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  // number of super locks along M
  int MB = div_up(seqlen_q, BLOCK_M);

  // parallel on [batches, num_heads, MB]
  parallel_for(batches * num_heads * MB, [&](int begin, int end) {
    int bs{0}, head_id{0}, mb{0};
    data_index_init(begin, bs, batches, head_id, num_heads, mb, MB);

    int tid = get_thread_num();
    // s_i and s_delta: [BLOCK_M, BLOCK_N]
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    scalar_t* __restrict__ s_delta = reinterpret_cast<scalar_t*>(s_i);

    // v_prime: [BLOCK_M, head_size_v]
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

    // Btmp: [BLOCK_N, max(head_size, head_size_v)]
    scalar_t* __restrict__ Btmp = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);

    // init Btmp and Btmp2 just once for each thread to prevent NaN
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      int seq_q_start_loc = bs * seqlen_q;
      int seq_k_start_loc = bs * seqlen_k;

      // offset and size in MB
      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, seqlen_q - m);

      assert(m_size > 0);

      int head_kv_id = head_id / num_groups;

      // get query
      const scalar_t* __restrict__ q_ptr = q + (seq_q_start_loc + m) * q_strideM + head_id * q_strideH;

      // init v', s' and m'
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);

      int num_keys = causal ? std::min(m + m_size, seqlen_k) : seqlen_k;
      for (int n = 0; n < num_keys; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, num_keys - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t>(
            /*    dst */ Btmp,
            /*    src */ k + (seq_k_start_loc + n) * k_strideN + head_kv_id * k_strideH,
            /*     N  */ n_size,
            /*     K  */ head_size,
            /* ld_src */ k_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        // apply causal mask
        if (causal && num_keys - n <= BLOCK_N) {
          for (int row = 0; row < m_size; ++row) {
            int last_col = m + row - n;
            // fill [last_col + 1, n_size) to -inf
            float* row_ptr = s_i + row * BLOCK_N;
            fill_stub(row_ptr + last_col + 1, -std::numeric_limits<float>::infinity(), n_size - last_col - 1);
          }
        }

        flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
            s_i, s_delta, v_prime, s_prime, m_prime, m_size, n_size, padded_n_size, head_size_v, sm_scale);

        // get value and pack
        pack_vnni2<scalar_t>(
            /*    dst */ Btmp,
            /*    src */ v + (seq_k_start_loc + n) * v_strideN + head_kv_id * v_strideH,
            /*     K  */ n_size,
            /*     N  */ head_size_v,
            /* ld_src */ v_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,  // n_size
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seqlen_k

      scalar_t* __restrict__ out_ptr = out + (seq_q_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
      }

      // move to the next index
      data_index_step(bs, batches, head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
void flash_attn_varlen_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ cu_seqlens_k,
    void* __restrict__ buffer,
    int32_t* __restrict__ indices,
    int max_seqlen_q,
    int max_seqlen_k,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float sm_scale,
    int buffer_size_per_thread,
    bool causal) {
  // strides
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // compute index (bs, mb_offset) for Query blocks
  // do this sequentially as usually problem size won't be big
  int idx = 0;
  for (int32_t bs = 0; bs < batches; ++bs) {
    int32_t seqlen_q = cu_seqlens_q[bs + 1] - cu_seqlens_q[bs];
    int32_t seqlen_k = cu_seqlens_k[bs + 1] - cu_seqlens_k[bs];
    TORCH_CHECK(seqlen_q <= max_seqlen_q && seqlen_k <= max_seqlen_k);

    int32_t blocks = div_up(seqlen_q, BLOCK_M);
    for (int32_t offset = 0; offset < blocks; ++offset) {
      indices[idx * 2 + 0] = bs;
      indices[idx * 2 + 1] = offset;
      idx++;
    }
  }
  // number of query blocks
  int MB = idx;

  // we use same buffer for packed key and value
  const int ldb_tmp = std::max(head_size, head_size_v);

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  // parallel on [MB, num_heads]
  parallel_for(num_heads * MB, [&](int begin, int end) {
    int head_id{0}, mb{0};
    data_index_init(begin, head_id, num_heads, mb, MB);

    int tid = get_thread_num();
    // s_i and s_delta: [BLOCK_M, BLOCK_N]
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    scalar_t* __restrict__ s_delta = reinterpret_cast<scalar_t*>(s_i);

    // v_prime: [BLOCK_M, head_size_v]
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

    // Btmp: [BLOCK_N, max(head_size, head_size_v)]
    scalar_t* __restrict__ Btmp = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);

    // init Btmp just once for each thread to prevent NaN
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      int32_t bs = indices[mb * 2 + 0];
      int32_t seq_q_start_loc = cu_seqlens_q[bs];
      int32_t seq_k_start_loc = cu_seqlens_k[bs];
      int32_t seqlen_q = cu_seqlens_q[bs + 1] - cu_seqlens_q[bs];

      // offset and size in MB
      int m = indices[mb * 2 + 1] * BLOCK_M;
      int m_size = std::min(BLOCK_M, seqlen_q - m);

      assert(m_size > 0);

      int head_kv_id = head_id / num_groups;

      // get query
      const scalar_t* __restrict__ q_ptr = q + (seq_q_start_loc + m) * q_strideM + head_id * q_strideH;

      // init v', s' and m'
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);

      int seqlen_k = cu_seqlens_k[bs + 1] - cu_seqlens_k[bs];
      int num_keys = causal ? std::min(m + m_size, seqlen_k) : seqlen_k;
      for (int n = 0; n < num_keys; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, num_keys - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t>(
            /*    dst */ Btmp,
            /*    src */ k + (seq_k_start_loc + n) * k_strideN + head_kv_id * k_strideH,
            /*     N  */ n_size,
            /*     K  */ head_size,
            /* ld_src */ k_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        // apply causal mask
        if (causal && num_keys - n <= BLOCK_N) {
          for (int row = 0; row < m_size; ++row) {
            int last_col = m + row - n;
            // fill [last_col + 1, n_size) to -inf
            float* row_ptr = s_i + row * BLOCK_N;
            fill_stub(row_ptr + last_col + 1, -std::numeric_limits<float>::infinity(), n_size - last_col - 1);
          }
        }

        flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
            s_i, s_delta, v_prime, s_prime, m_prime, m_size, n_size, padded_n_size, head_size_v, sm_scale);

        // get value and pack
        pack_vnni2<scalar_t>(
            /*    dst */ Btmp,
            /*    src */ v + (seq_k_start_loc + n) * v_strideN + head_kv_id * v_strideH,
            /*     K  */ n_size,
            /*     N  */ head_size_v,
            /* ld_src */ v_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,  // n_size
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seqlen_k

      scalar_t* __restrict__ out_ptr = out + (seq_q_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
      }

      // move to the next index
      data_index_step(head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

template <typename index_t>
inline bool has_varlen_sequences(
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int batches,
    index_t max_seqlen_q,
    index_t max_seqlen_k) {
  const index_t* cu_seqlens_q_data = cu_seqlens_q.data_ptr<index_t>();
  const index_t* cu_seqlens_k_data = cu_seqlens_k.data_ptr<index_t>();

  for (int bs = 0; bs < batches; ++bs) {
    index_t seqlen_q = cu_seqlens_q_data[bs + 1] - cu_seqlens_q_data[bs];
    index_t seqlen_k = cu_seqlens_k_data[bs + 1] - cu_seqlens_k_data[bs];
    if (seqlen_q != max_seqlen_q || seqlen_k != max_seqlen_k) {
      return true;
    }
  }
  return false;
}

template <int BLOCK_M, int BLOCK_N>
inline int resize_buffer(at::Tensor& buffer, int num_threads, int head_size, int head_size_v) {
  static_assert(BLOCK_M <= BLOCK_N, "Make sure BLOCK_M <= BLOCK_N to prevent buffer overflows during causal masking");
  const int size_per_thread =
      /* s_i     */ BLOCK_M * BLOCK_N * sizeof(float) +
      /* v_prime */ BLOCK_M * head_size_v * sizeof(float) +
      /* Btmp    */ BLOCK_N * std::max(head_size, head_size_v) * sizeof(uint16_t);

  buffer.resize_({num_threads, size_per_thread});
  return size_per_thread;
}

template <int BLOCK_M>
inline void resize_indices(at::Tensor& indices, int num_seqs, int max_seqlen_q) {
  // we allocate memory based on max seqlen
  indices.resize_({num_seqs, div_up(max_seqlen_q, BLOCK_M), 2});
}

// [NOTE]: `flash_attn_varlen_func` AMX kernel
//
//   q: [num_tokens, num_heads, head_size]
//   k: [num_tokens, num_heads_kv, head_size]
//   v: [num_tokens, num_heads_kv, head_size_v]
//   cu_seqlens_q: [num_seqs + 1]
//   cu_seqlens_k: [num_seqs + 1]
//   out: [num_tokens, num_heads, head_size_v]
//
at::Tensor flash_attn_varlen_func(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool causal) {
  RECORD_FUNCTION(
      "sgl_kernel::flash_attn_varlen_func",
      std::vector<c10::IValue>({q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v);
  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_INPUT(cu_seqlens_q);
  CHECK_INPUT(cu_seqlens_k);
  CHECK_EQ(cu_seqlens_q.scalar_type(), at::kInt);
  CHECK_EQ(cu_seqlens_k.scalar_type(), at::kInt);

  int num_seqs = cu_seqlens_q.size(0) - 1;
  int num_tokens = q.size(0);
  int num_heads = q.size(1);
  int num_heads_kv = k.size(1);
  int head_size = q.size(2);
  int head_size_v = v.size(2);

  // strides for q, k and v
  int q_strideM = q.stride(0);
  int q_strideH = q.stride(1);
  int k_strideN = k.stride(0);
  int k_strideH = k.stride(1);
  int v_strideN = v.stride(0);
  int v_strideH = v.stride(1);

  // check sizes
  CHECK_EQ(k.size(2), head_size);
  CHECK_EQ(v.size(1), num_heads_kv);
  CHECK_EQ(cu_seqlens_k.size(0), num_seqs + 1);

  // D and DV need to be even as we transpose by 512-bit
  TORCH_CHECK(head_size % 2 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 2 == 0, "invalid head_size_v ", head_size_v);

  // softmax scale
  double sm_scale = 1.0 / std::sqrt(static_cast<double>(head_size));

  // check whether the batch has variant lengths
  const bool is_varlen =
      has_varlen_sequences<int32_t>(cu_seqlens_q, cu_seqlens_k, num_seqs, max_seqlen_q, max_seqlen_k);

  int num_threads = at::get_num_threads();
  at::Tensor buffer = at::empty({}, q.options().dtype(at::kChar));
  at::Tensor indices = at::empty({}, q.options().dtype(at::kInt));
  at::Tensor out = at::empty({num_tokens, num_heads, head_size_v}, q.options());

  // TODO: tune the block size
  constexpr int BLOCK_M = 512;
  constexpr int BLOCK_N = 768;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "flash_attn_varlen_func", [&] {
    int sz = resize_buffer<BLOCK_M, BLOCK_N>(buffer, num_threads, head_size, head_size_v);

    if (is_varlen) {
      resize_indices<BLOCK_M>(indices, num_seqs, max_seqlen_q);
      flash_attn_varlen_kernel_impl<scalar_t, BLOCK_M, BLOCK_N>(
          out.data_ptr<scalar_t>(),
          q.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(),
          cu_seqlens_q.data_ptr<int32_t>(),
          cu_seqlens_k.data_ptr<int32_t>(),
          buffer.data_ptr(),
          indices.data_ptr<int32_t>(),
          max_seqlen_q,
          max_seqlen_k,
          num_seqs,
          num_heads,
          num_heads_kv,
          head_size,
          head_size_v,
          q_strideM,
          q_strideH,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          sm_scale,
          sz,
          causal);
    } else {
      flash_attn_kernel_impl<scalar_t, BLOCK_M, BLOCK_N>(
          out.data_ptr<scalar_t>(),
          q.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(),
          buffer.data_ptr(),
          max_seqlen_q,
          max_seqlen_k,
          num_seqs,
          num_heads,
          num_heads_kv,
          head_size,
          head_size_v,
          q_strideM,
          q_strideH,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          sm_scale,
          sz,
          causal);
    }
  });

  return out;
}
