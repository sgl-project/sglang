#include "common.h"
#include "flash_attn.h"
#include "gemm.h"

namespace {

// [NOTE]: extend attention for CPU
//   1. BLOCK_M and BLOCK_N tuned for various seq lengths
//   2. can handle non-contiguous k_extend and v_extend
//   3. computes attention for prefix and extend separately
//   4. TODO: apply head dimension blocking to optimize GQA
//   5. optional tree mask for speculative decoding TARGET_VERIFY (EAGLE topk > 1):
//      `tree_mask` is a flat [batches * qlen * qlen] bool tensor in
//      TreeMaskMode::QLEN_ONLY layout, where qlen == extend_seq_lens[bs] ==
//      max_len_extend (uniform across the batch, equal to draft_token_num).
//      Row i = query draft token, column j = key draft token; true means query i
//      may attend key j (each row marks self + ancestors + root). The committed
//      prefix (stage 1) is implicitly fully visible to every draft token, which
//      is why the mask only covers the qlen x qlen new-token block; the GPU
//      FULL_MASK layout carries the prefix columns explicitly but they are
//      all-true for EAGLE. When tree_mask is absent, stage 2 falls back to the
//      plain causal mask (correct for non-spec extend and topk == 1 chains).
//

inline int tree_row_depth(const bool* __restrict__ tree_mask_seq, int seq_len_extend, int row_idx) {
  const bool* __restrict__ row_ptr = tree_mask_seq + static_cast<int64_t>(row_idx) * seq_len_extend;
  int vis = 0;
  for (int col = 0; col <= row_idx && col < seq_len_extend; ++col) {
    vis += row_ptr[col] ? 1 : 0;
  }
  return vis > 0 ? (vis - 1) : row_idx;
}

inline void apply_rel_bias_and_scale(
    float* __restrict__ s_i_row,
    const void* __restrict__ rel_bias_ptr,
    at::ScalarType rb_dtype,
    int64_t rb_row_offset,
    int64_t rel_extent,
    int64_t base_dist,
    int n_size,
    float sm_scale) {
  const int col_start = static_cast<int>(std::max<int64_t>(0, base_dist - rel_extent + 1));
  const int col_end = static_cast<int>(std::min<int64_t>(n_size, base_dist + 1));
  const float neg_inf = -std::numeric_limits<float>::infinity();
  auto run = [&](const auto* rb) {
    for (int col = 0; col < col_start; ++col) {
      float v = s_i_row[col];
      if (v != neg_inf) {
        s_i_row[col] = v * sm_scale;
      }
    }
    for (int col = col_start; col < col_end; ++col) {
      float v = s_i_row[col];
      if (v != neg_inf) {
        s_i_row[col] = v * sm_scale + static_cast<float>(rb[rb_row_offset + (base_dist - col)]);
      }
    }
    for (int col = col_end; col < n_size; ++col) {
      float v = s_i_row[col];
      if (v != neg_inf) {
        s_i_row[col] = v * sm_scale;
      }
    }
  };
  if (rb_dtype == at::kFloat) {
    run(static_cast<const float*>(rel_bias_ptr));
  } else if (rb_dtype == at::kBFloat16) {
    run(static_cast<const at::BFloat16*>(rel_bias_ptr));
  } else {
    run(static_cast<const at::Half*>(rel_bias_ptr));
  }
}

template <typename scalar_t, typename packed_t, typename index_t, int BLOCK_M, int BLOCK_N>
void extend_attention_kernel_impl(
    void* __restrict__ o_extend_ptr,
    bool out_fp32,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const packed_t* __restrict__ k_buffer,
    const packed_t* __restrict__ v_buffer,
    const float* __restrict__ k_buf_scale,
    const float* __restrict__ v_buf_scale,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const int64_t* __restrict__ encoder_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    const void* __restrict__ buffer,
    const scalar_t* __restrict__ sinks,
    const bool* __restrict__ tree_mask,
    const void* __restrict__ rel_bias_ptr,
    at::ScalarType rel_bias_dtype,
    int64_t rb_strideM,
    int64_t rb_strideH,
    int64_t rel_extent,
    float* __restrict__ lse_out_ptr,
    int64_t lse_strideM,
    int64_t lse_strideH,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float sm_scale,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    int buffer_size_per_thread,
    int64_t sliding_window_size,
    bool is_prefix_skipped,
    bool is_cross_attn,
    bool is_causal,
    bool kv_from_cache,
    bool has_encoder_lens,
    bool has_sink,
    bool window_anchor_at_start) {
  // strides
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // we use same buffer for packed key and value
  const int ldb_tmp = std::max(head_size, head_size_v);

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  // number of blocks along M
  int MB = div_up(max_len_extend, BLOCK_M);

  // parallel on [batches, num_heads, BM]
  at::parallel_for(0, batches * num_heads * MB, 0, [&](int begin, int end) {
    int bs{0}, head_id{0}, mb{0};
    data_index_init(begin, bs, batches, head_id, num_heads, mb, MB);

    int tid = at::get_thread_num();
    // s_i: [BLOCK_M, BLOCK_N]
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);

    // v_prime: [BLOCK_M, head_size_v]
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

    // s_delta: [BLOCK_M, BLOCK_N]
    scalar_t* __restrict__ s_delta = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);

    // Btmp: [BLOCK_N, max(head_size, head_size_v)]
    scalar_t* __restrict__ Btmp = reinterpret_cast<scalar_t*>(s_delta + BLOCK_M * BLOCK_N);

    // init Btmp just once for each thread to prevent NaN
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);
    fill_stub(s_delta, 0.f, BLOCK_M * BLOCK_N);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      // seq_len = prefix + extend
      int head_kv_id = head_id / num_groups;
      int seq_len = seq_lens[bs];
      int seq_len_extend = extend_seq_lens[bs];
      int seq_len_prefix = seq_len - seq_len_extend;
      int seq_extend_start_loc = extend_start_loc[bs];

      int req_pool_id = req_pool_indices[bs];
      const int64_t req_to_token_row_offset = static_cast<int64_t>(req_pool_id) * max_context_len;
      int kv_offset = (has_encoder_lens && (!is_cross_attn)) ? encoder_lens[bs] : 0;
      TORCH_CHECK(seq_len_prefix >= 0, "prefix len < 0!");
      TORCH_CHECK(seq_len <= max_context_len, "seq_len out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      if (is_prefix_skipped) {
        TORCH_CHECK(seq_len_prefix == 0, "extend attention: expect seq_len_prefix to be 0, got ", seq_len_prefix);
      }

      if (tree_mask != nullptr) {
        // QLEN_ONLY layout assumes a uniform qlen across the batch (TARGET_VERIFY)
        TORCH_CHECK(
            seq_len_extend == max_len_extend,
            "extend attention: tree_mask requires uniform extend_seq_lens, got ",
            seq_len_extend,
            " vs ",
            max_len_extend);
      }

      // offset and size in MB
      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, seq_len_extend - m);

      if (m_size <= 0) {
        data_index_step(bs, batches, head_id, num_heads, mb, MB);
        continue;
      }

      // get query
      const scalar_t* __restrict__ q_ptr = q_extend + (seq_extend_start_loc + m) * q_strideM + head_id * q_strideH;

      // init v', s' and m'
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<float>::infinity(), m_size);
      const bool use_per_row_swa = sliding_window_size > 0 && !window_anchor_at_start;
      // stage 1: compute scores with prefix
      // kv_from_cache has no stage 2, so cover the extend range and stop at the diagonal
      int kv_start = (window_anchor_at_start && sliding_window_size > 0 && !is_cross_attn)
                         ? std::max(0, seq_len_prefix - (static_cast<int>(sliding_window_size) - 1))
                         : 0;
      int kv_end = is_cross_attn ? encoder_lens[bs] : (kv_from_cache ? seq_len_prefix + m + m_size : seq_len_prefix);
      for (int n = kv_start; n < kv_end; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, kv_end - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t, packed_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ k_buffer + head_kv_id * k_strideH,
            /* src_scale*/ k_buf_scale,
            /*    ind */ req_to_token + req_to_token_row_offset + n + kv_offset,
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

        for (int row = 0; row < m_size; ++row) {
          bool row_is_empty = false;
          if (kv_from_cache) {
            int first_future_col = seq_len_prefix + m + row + 1;
            if (n >= first_future_col) {
              row_is_empty = true;
            } else if (first_future_col < n + n_size) {
              fill_stub(
                  s_i + row * BLOCK_N + (first_future_col - n),
                  -std::numeric_limits<float>::infinity(),
                  n + n_size - first_future_col);
            }
          }
          const bool* __restrict__ tree_mask_seq =
              (tree_mask != nullptr) ? (tree_mask + static_cast<int64_t>(bs) * seq_len_extend * seq_len_extend)
                                     : nullptr;
          const int q_depth =
              (tree_mask_seq != nullptr) ? tree_row_depth(tree_mask_seq, seq_len_extend, m + row) : (m + row);
          if (!row_is_empty && use_per_row_swa) {
            int last_col = seq_len_prefix + q_depth - sliding_window_size + 1;
            if (last_col >= n + n_size) {
              row_is_empty = true;
            } else if (last_col > n) {
              fill_stub(s_i + row * BLOCK_N, -std::numeric_limits<float>::infinity(), last_col - n);
            }
          }
          if (row_is_empty) {
            // s_delta is reused across blocks - zero an empty row rather than
            // skip it, or P @ V below applies the previous block's weights here
            fill_stub(s_delta + row * BLOCK_N, 0.f, padded_n_size);
            continue;
          }
          float softmax_scale = sm_scale;
          if (rel_bias_ptr != nullptr) {
            int64_t q_tok_idx = static_cast<int64_t>(seq_extend_start_loc) + m + row;
            int64_t rb_row_offset = q_tok_idx * rb_strideM + static_cast<int64_t>(head_id) * rb_strideH;
            int64_t base_dist = static_cast<int64_t>(seq_len_prefix) + q_depth - n;
            apply_rel_bias_and_scale(
                s_i + row * BLOCK_N,
                rel_bias_ptr,
                rel_bias_dtype,
                rb_row_offset,
                rel_extent,
                base_dist,
                n_size,
                sm_scale);
            softmax_scale = 1.0f;
          }
          flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
              s_i, s_delta, v_prime, s_prime, m_prime, m_size, n_size, padded_n_size, head_size_v, softmax_scale, row);
        }

        // get value and pack
        pack_vnni2<scalar_t, packed_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ v_buffer + head_kv_id * v_strideH,
            /* src_scale*/ v_buf_scale,
            /*    ind */ req_to_token + req_to_token_row_offset + n + kv_offset,
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
      }  // loop with seq_len_prefix
      if (!is_cross_attn && !kv_from_cache) {
        // stage 2: compute the triangle part (or, when !is_causal, the full square)
        int num_keys = is_causal ? std::min(seq_len_extend, m + BLOCK_M) : seq_len_extend;
        for (int n = 0; n < num_keys; n += BLOCK_N) {
          int n_size = std::min(BLOCK_N, num_keys - n);

          // `n_size` is K in 2nd gemm, pad to TILE_K;
          const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

          // get key and pack
          pack_vnni<scalar_t>(
              /*    dst */ Btmp,
              /*    src */ k_extend + (seq_extend_start_loc + n) * ke_strideN + head_kv_id * ke_strideH,
              /*     N  */ n_size,
              /*     K  */ head_size,
              /* ld_src */ ke_strideN,
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

          alignas(64) bool stage2_empty[BLOCK_M];
          for (int row = 0; row < m_size; ++row) {
            stage2_empty[row] = false;
          }
          // apply tree mask (speculative TARGET_VERIFY) or causal mask
          if (tree_mask != nullptr) {
            // [Note] tree mask for EAGLE topk > 1 (TreeMaskMode::QLEN_ONLY).
            const bool* __restrict__ mask_base =
                tree_mask + (static_cast<int64_t>(bs) * seq_len_extend + m) * seq_len_extend + n;
            for (int row = 0; row < m_size; ++row) {
              float* __restrict__ row_ptr = s_i + row * BLOCK_N;
              const bool* __restrict__ mask_ptr = mask_base + static_cast<int64_t>(row) * seq_len_extend;
              bool any_visible = false;
              for (int col = 0; col < n_size; ++col) {
                if (!mask_ptr[col]) {
                  row_ptr[col] = -std::numeric_limits<float>::infinity();
                } else {
                  any_visible = true;
                }
              }
              if (!any_visible) {
                stage2_empty[row] = true;
              }
            }
          } else if (is_causal && n + n_size - 1 > m) {
            // apply causal mask
            for (int row = 0; row < m_size; ++row) {
              int last_col = m + row - n;
              if (last_col < 0) {
                stage2_empty[row] = true;
              } else if (last_col + 1 < n_size) {
                float* row_ptr = s_i + row * BLOCK_N;
                fill_stub(row_ptr + last_col + 1, -std::numeric_limits<float>::infinity(), n_size - last_col - 1);
              }
            }
          }

          const bool* __restrict__ tree_mask_seq =
              (tree_mask != nullptr) ? (tree_mask + static_cast<int64_t>(bs) * seq_len_extend * seq_len_extend)
                                     : nullptr;
          for (int row = 0; row < m_size; ++row) {
            bool row_is_empty = stage2_empty[row];
            const int q_depth =
                (tree_mask_seq != nullptr) ? tree_row_depth(tree_mask_seq, seq_len_extend, m + row) : (m + row);
            if (!row_is_empty && use_per_row_swa) {
              if (tree_mask_seq != nullptr) {
                const bool* __restrict__ mask_ptr = tree_mask_seq + static_cast<int64_t>(m + row) * seq_len_extend + n;
                float* __restrict__ row_ptr = s_i + row * BLOCK_N;
                bool any_vis = false;
                for (int col = 0; col < n_size; ++col) {
                  if (!mask_ptr[col]) {
                    continue;
                  }
                  int k_depth = tree_row_depth(tree_mask_seq, seq_len_extend, n + col);
                  if (q_depth - k_depth >= sliding_window_size) {
                    row_ptr[col] = -std::numeric_limits<float>::infinity();
                  } else {
                    any_vis = true;
                  }
                }
                if (!any_vis) {
                  row_is_empty = true;
                }
              } else {
                int left_end = m + row - sliding_window_size + 1 - n;
                if (left_end >= n_size) {
                  row_is_empty = true;
                } else if (left_end > 0) {
                  fill_stub(s_i + row * BLOCK_N, -std::numeric_limits<float>::infinity(), left_end);
                }
                if (!row_is_empty && !is_causal) {
                  int right_start = m + row + sliding_window_size - n;
                  if (right_start <= 0) {
                    row_is_empty = true;
                  } else if (right_start < n_size) {
                    fill_stub(
                        s_i + row * BLOCK_N + right_start,
                        -std::numeric_limits<float>::infinity(),
                        n_size - right_start);
                  }
                }
              }
            }
            if (row_is_empty) {
              fill_stub(s_delta + row * BLOCK_N, 0.f, padded_n_size);
              continue;
            }
            float softmax_scale = sm_scale;
            if (rel_bias_ptr != nullptr) {
              int64_t q_tok_idx = static_cast<int64_t>(seq_extend_start_loc) + m + row;
              int64_t rb_row_offset = q_tok_idx * rb_strideM + static_cast<int64_t>(head_id) * rb_strideH;
              if (tree_mask_seq != nullptr) {
                const bool* __restrict__ mask_ptr = tree_mask_seq + static_cast<int64_t>(m + row) * seq_len_extend + n;
                float* __restrict__ row_ptr = s_i + row * BLOCK_N;
                const float neg_inf = -std::numeric_limits<float>::infinity();
                for (int col = 0; col < n_size; ++col) {
                  if (!mask_ptr[col]) {
                    continue;
                  }
                  int64_t dist = static_cast<int64_t>(q_depth - tree_row_depth(tree_mask_seq, seq_len_extend, n + col));
                  float v = row_ptr[col] * sm_scale;
                  if (dist >= 0 && dist < rel_extent) {
                    if (rel_bias_dtype == at::kFloat) {
                      v += static_cast<const float*>(rel_bias_ptr)[rb_row_offset + dist];
                    } else if (rel_bias_dtype == at::kBFloat16) {
                      v += static_cast<float>(static_cast<const at::BFloat16*>(rel_bias_ptr)[rb_row_offset + dist]);
                    } else {
                      v += static_cast<float>(static_cast<const at::Half*>(rel_bias_ptr)[rb_row_offset + dist]);
                    }
                  }
                  row_ptr[col] = (row_ptr[col] == neg_inf) ? neg_inf : v;
                }
              } else {
                int64_t base_dist = static_cast<int64_t>(m + row) - n;
                apply_rel_bias_and_scale(
                    s_i + row * BLOCK_N,
                    rel_bias_ptr,
                    rel_bias_dtype,
                    rb_row_offset,
                    rel_extent,
                    base_dist,
                    n_size,
                    sm_scale);
              }
              softmax_scale = 1.0f;
            }
            flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
                s_i,
                s_delta,
                v_prime,
                s_prime,
                m_prime,
                m_size,
                n_size,
                padded_n_size,
                head_size_v,
                softmax_scale,
                row);
          }

          // get value and pack
          pack_vnni2<scalar_t>(
              /*    dst */ Btmp,
              /*    src */ v_extend + (seq_extend_start_loc + n) * ve_strideN + head_kv_id * ve_strideH,
              /*     K  */ n_size,
              /*     N  */ head_size_v,
              /* ld_src */ ve_strideN,
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
        }  // loop with seq_len_extend
      }
      const int64_t out_base = static_cast<int64_t>(seq_extend_start_loc + m) * o_strideM + head_id * o_strideH;
      scalar_t* __restrict__ out_ptr = out_fp32 ? nullptr : (static_cast<scalar_t*>(o_extend_ptr) + out_base);
      float* __restrict__ out_f32_ptr = out_fp32 ? (static_cast<float*>(o_extend_ptr) + out_base) : nullptr;
      for (int row = 0; row < m_size; ++row) {
        if (has_sink) {
          float sink_val = static_cast<float>(sinks[head_id]);
          if (m_prime[row] == -std::numeric_limits<float>::infinity()) {
            m_prime[row] = sink_val;
            s_prime[row] = 1.0f;
          } else if (sink_val > m_prime[row]) {
            float m_delta = std::exp(m_prime[row] - sink_val);
            for (int d = 0; d < head_size_v; ++d) {
              v_prime[row * head_size_v + d] *= m_delta;
            }
            s_prime[row] = s_prime[row] * m_delta + 1.0f;
            m_prime[row] = sink_val;
          } else {
            s_prime[row] += std::exp(sink_val - m_prime[row]);
          }
        }
        float s = s_prime[row] > 0.f ? (1.0f / s_prime[row]) : 0.0f;
        if (out_fp32) {
          copy_stub<float>(out_f32_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
        } else {
          copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
        }
        if (lse_out_ptr != nullptr) {
          float lse =
              (s_prime[row] > 0.f) ? (m_prime[row] + std::log(s_prime[row])) : -std::numeric_limits<float>::infinity();
          lse_out_ptr[(static_cast<int64_t>(seq_extend_start_loc) + m + row) * lse_strideM + head_id * lse_strideH] =
              lse;
        }
      }

      // move to the next index
      data_index_step(bs, batches, head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

template <int BLOCK_M, int BLOCK_N>
inline int resize_buffer(at::Tensor& buffer, int num_threads, int head_size, int head_size_v) {
  static_assert(BLOCK_M <= BLOCK_N, "Make sure BLOCK_M <= BLOCK_N to prevent buffer overflows during causal masking");
  const int size_per_thread =
      /* s_i     */ BLOCK_M * BLOCK_N * sizeof(float) +
      /* v_prime */ BLOCK_M * head_size_v * sizeof(float) +
      /* s_delta */ BLOCK_M * BLOCK_N * sizeof(uint16_t) +
      /* Btmp    */ BLOCK_N * std::max(head_size, head_size_v) * sizeof(uint16_t);

  buffer.resize_({num_threads, size_per_thread});
  return size_per_thread;
}

#define LAUNCH_EXTEND_ATTENTION_KERNEL(BLOCK_M, BLOCK_N)                                   \
  do {                                                                                     \
    int sz = resize_buffer<BLOCK_M, BLOCK_N>(buffer, num_threads, head_size, head_size_v); \
                                                                                           \
    extend_attention_kernel_impl<scalar_t, packed_t, index_t, BLOCK_M, BLOCK_N>(           \
        o_extend.data_ptr(),                                                               \
        out_fp32,                                                                          \
        q_extend.data_ptr<scalar_t>(),                                                     \
        k_extend.data_ptr<scalar_t>(),                                                     \
        v_extend.data_ptr<scalar_t>(),                                                     \
        k_buffer.data_ptr<packed_t>(),                                                     \
        v_buffer.data_ptr<packed_t>(),                                                     \
        k_buf_scale_ptr,                                                                   \
        v_buf_scale_ptr,                                                                   \
        req_to_token.data_ptr<index_t>(),                                                  \
        req_pool_indices.data_ptr<int64_t>(),                                              \
        seq_lens.data_ptr<int64_t>(),                                                      \
        encoder_lens_t.data_ptr<int64_t>(),                                                \
        extend_seq_lens.data_ptr<index_t>(),                                               \
        extend_start_loc.data_ptr<index_t>(),                                              \
        buffer.data_ptr(),                                                                 \
        sinks_tensor.data_ptr<scalar_t>(),                                                 \
        tree_mask_ptr,                                                                     \
        rel_bias_ptr,                                                                      \
        rel_bias_dtype,                                                                    \
        rb_strideM,                                                                        \
        rb_strideH,                                                                        \
        rel_extent,                                                                        \
        lse_out_ptr,                                                                       \
        lse_strideM,                                                                       \
        lse_strideH,                                                                       \
        num_seqs,                                                                          \
        num_heads,                                                                         \
        num_heads_kv,                                                                      \
        head_size,                                                                         \
        head_size_v,                                                                       \
        q_strideM,                                                                         \
        q_strideH,                                                                         \
        ke_strideN,                                                                        \
        ke_strideH,                                                                        \
        ve_strideN,                                                                        \
        ve_strideH,                                                                        \
        k_strideN,                                                                         \
        k_strideH,                                                                         \
        v_strideN,                                                                         \
        v_strideH,                                                                         \
        sm_scale,                                                                          \
        max_num_reqs,                                                                      \
        max_context_len,                                                                   \
        max_total_num_tokens,                                                              \
        max_len_extend,                                                                    \
        sz,                                                                                \
        sliding_window_size,                                                               \
        is_prefix_skipped,                                                                 \
        is_cross_attn,                                                                     \
        is_causal,                                                                         \
        kv_from_cache,                                                                     \
        has_encoder_lens,                                                                  \
        has_sink,                                                                          \
        window_anchor_at_start);                                                           \
  } while (0)

// q_extend, k_extend, v_extend, o_extend: contiguous tensors
// k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
//
// q_extend: [num_tokens, num_heads, head_size]
// k_extend: [num_extend_tokens, num_heads, head_size]
// v_extend: [num_extend_tokens, num_heads, head_size]
// o_extend: [num_tokens, num_heads, head_size]
// k_buffer: [max_total_num_tokens, num_heads, head_size]
// v_buffer: [max_total_num_tokens, num_heads, head_size]
// req_to_token: [max_num_reqs, max_context_len] int32 or int64
// req_pool_indices: [num_seqs] int64
// seq_lens: [num_seqs] int64
// extend_seq_lens: [num_seqs]
// extend_start_loc: [num_seqs]
// encoder_lens: [num_seqs] int64 or None
// sinks: [num_heads] or None
// tree_mask: [num_seqs * max_len_extend * max_len_extend] bool or None
//   TreeMaskMode::QLEN_ONLY tree mask for speculative TARGET_VERIFY; see [NOTE] 5 above.
void extend_attention_cpu(
    at::Tensor& q_extend,
    const std::optional<at::Tensor>& k_extend_opt,
    const std::optional<at::Tensor>& v_extend_opt,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    double k_buf_scale,
    double v_buf_scale,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap,
    bool is_cross_attn,
    int64_t sliding_window_size,
    std::optional<at::Tensor> encoder_lens,
    std::optional<at::Tensor> sinks,
    std::optional<at::Tensor> tree_mask,
    bool is_causal = true,
    const std::optional<at::Tensor>& rel_bias = std::nullopt,
    const std::optional<at::Tensor>& lse_out = std::nullopt,
    bool window_anchor_at_start = false) {
  TORCH_CHECK(k_extend_opt.has_value() == v_extend_opt.has_value(), "k_extend and v_extend must be given together");
  // A KV-shared layer (Gemma 4) passes no extend K/V - the layer it shares with
  // already wrote them to the cache, so this kernel masks causally itself. Cross
  // attention can also arrive without K/V but reads an encoder sequence that
  // carries no causal order, so it keeps its own path.
  const bool kv_from_cache = !is_cross_attn && !k_extend_opt.has_value();
  // unused when the range comes from the cache - bind them to the buffers
  if (max_len_extend <= 0) {
    max_len_extend = extend_seq_lens.max().item<int64_t>();
  }
  auto k_extend = k_extend_opt.has_value() ? k_extend_opt.value() : k_buffer;
  auto v_extend = v_extend_opt.has_value() ? v_extend_opt.value() : v_buffer;

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  const bool out_fp32 = o_extend.scalar_type() == at::kFloat;
  TORCH_CHECK(
      out_fp32 || o_extend.scalar_type() == q_extend.scalar_type(),
      "extend: expect o_extend dtype to match q_extend or be float32, got ",
      o_extend.scalar_type());

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = req_to_token.size(0);
  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = q_extend.size(1);
  int num_heads_kv = k_extend.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_extend.size(2);

  // strides for q_extend, k_extend and v_extend
  int q_strideM = q_extend.stride(0);
  int q_strideH = q_extend.stride(1);
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

  // strides for k_buffer and v_buffer
  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  // check sizes
  CHECK_EQ(req_pool_indices.size(0), num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), num_seqs);
  CHECK_EQ(extend_start_loc.size(0), num_seqs);
  CHECK_EQ(v_extend.size(1), num_heads_kv);
  CHECK_EQ(k_buffer.size(1), v_buffer.size(1));

  // MLA will skip prefix part
  const bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  // check index data types
  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "extend: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "extend: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "extend: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == index_dtype && extend_start_loc.scalar_type() == index_dtype,
      "extend: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

  // D and DV need to be 32x as we transpose by 512-bit
  TORCH_CHECK(head_size % 32 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 32 == 0, "invalid head_size_v ", head_size_v);
  auto kv_dtype = k_buffer.scalar_type();
  if (kv_dtype == at::ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(v_buffer.scalar_type() == kv_dtype, "k_buffer and v_buffer should have same data type");
    TORCH_CHECK(k_buf_scale > 0.0 && v_buf_scale > 0.0, "float8 static scales must be positive");
  }
  int num_threads = at::get_num_threads();
  float k_buf_scale_float = static_cast<float>(k_buf_scale);
  float v_buf_scale_float = static_cast<float>(v_buf_scale);
  const float* k_buf_scale_ptr = kv_dtype == at::ScalarType::Float8_e4m3fn ? &k_buf_scale_float : nullptr;
  const float* v_buf_scale_ptr = kv_dtype == at::ScalarType::Float8_e4m3fn ? &v_buf_scale_float : nullptr;
  auto buffer = at::empty({}, q_extend.options().dtype(at::kChar));

  bool has_encoder_lens = encoder_lens.has_value();
  // Since encoder_lens is not used when it is None, encoder_lens_t can be initialized as any tensor of int64_t dtype.
  at::Tensor encoder_lens_t = seq_lens;
  if (has_encoder_lens) {
    encoder_lens_t = encoder_lens.value();
    CHECK_EQ(encoder_lens_t.size(0), num_seqs);
  }
  bool has_sink = sinks.has_value();
  at::Tensor sinks_tensor = has_sink ? sinks.value() : at::empty({num_heads}, q_extend.options());
  CHECK_DIM(1, sinks_tensor);
  CHECK_EQ(sinks_tensor.size(0), num_heads);

  const bool* tree_mask_ptr = nullptr;
  if (tree_mask.has_value()) {
    const at::Tensor& tree_mask_t = tree_mask.value();
    CHECK_INPUT(tree_mask_t);
    TORCH_CHECK(
        tree_mask_t.scalar_type() == at::kBool, "extend: expect tree_mask to be bool, got ", tree_mask_t.scalar_type());
    TORCH_CHECK(
        tree_mask_t.numel() == static_cast<int64_t>(num_seqs) * max_len_extend * max_len_extend,
        "extend: expect tree_mask numel to be num_seqs * max_len_extend^2 = ",
        static_cast<int64_t>(num_seqs) * max_len_extend * max_len_extend,
        ", got ",
        tree_mask_t.numel());
    TORCH_CHECK(!is_cross_attn, "extend: tree_mask is not supported for cross attention");
    TORCH_CHECK(!kv_from_cache, "extend: tree_mask is not supported for KV-shared layers");

    tree_mask_ptr = tree_mask_t.data_ptr<bool>();
  }

  const void* rel_bias_ptr = nullptr;
  at::ScalarType rel_bias_dtype = at::kFloat;
  int64_t rb_strideM = 0;
  int64_t rb_strideH = 0;
  int64_t rel_extent = 0;
  if (rel_bias.has_value()) {
    const at::Tensor& rb = rel_bias.value();
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(rb);
    CHECK_DIM(3, rb);
    CHECK_EQ(rb.size(0), q_extend.size(0));
    CHECK_EQ(rb.size(1), num_heads);
    TORCH_CHECK(rb.size(2) > 0, "extend: rel_bias rel_extent must be positive");
    rel_bias_dtype = rb.scalar_type();
    TORCH_CHECK(
        rel_bias_dtype == at::kFloat || rel_bias_dtype == at::kBFloat16 || rel_bias_dtype == at::kHalf,
        "extend: unsupported rel_bias dtype ",
        rel_bias_dtype);
    rel_bias_ptr = rb.data_ptr();
    rb_strideM = rb.stride(0);
    rb_strideH = rb.stride(1);
    rel_extent = rb.size(2);
  }

  float* lse_out_ptr = nullptr;
  int64_t lse_strideM = 0;
  int64_t lse_strideH = 0;
  if (lse_out.has_value()) {
    const at::Tensor& lse_t = lse_out.value();
    CHECK_DIM(2, lse_t);
    TORCH_CHECK(lse_t.device().is_cpu(), "extend: lse_out must be a CPU tensor");
    TORCH_CHECK(lse_t.scalar_type() == at::kFloat, "extend: expect lse_out to be float32, got ", lse_t.scalar_type());
    CHECK_EQ(lse_t.size(0), q_extend.size(0));
    CHECK_EQ(lse_t.size(1), num_heads);
    lse_out_ptr = lse_t.data_ptr<float>();
    lse_strideM = lse_t.stride(0);
    lse_strideH = lse_t.stride(1);
  }

  if (q_extend.size(0) == 0 || num_seqs == 0) {
    return;
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_extend.scalar_type(), "extend_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_indices", [&] {
      CPU_DISPATCH_PACKED_TYPES(k_buffer.scalar_type(), "extend_attention_packed_types", [&] {
        if (max_len_extend <= 256) {
          LAUNCH_EXTEND_ATTENTION_KERNEL(32, 64);
        } else if (max_len_extend <= 1024) {
          LAUNCH_EXTEND_ATTENTION_KERNEL(128, 256);
        } else if (max_len_extend <= 4096) {
          LAUNCH_EXTEND_ATTENTION_KERNEL(256, 768);
        } else {  // max_len_extend > 4096
          LAUNCH_EXTEND_ATTENTION_KERNEL(512, 768);
        }
      });
    });
  });
}
