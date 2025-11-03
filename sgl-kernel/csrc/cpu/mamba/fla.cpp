#include "common.h"
#include "vec.h"
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>

namespace {

#if defined(CPU_CAPABILITY_AVX512)
// key: from [N, 32] to [32/2, N, 2]
template <typename scalar_t>
inline void pack_vnni_Nx32(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int N,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[16];

  int n = 0;
  for (; n < N; ++n) {
    vinputs[n] = _mm512_loadu_si512(src + n * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  // pack key
  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst + k * ld_dst * 2, vmask, vinputs[k]);
  }
}

// value: from [K, 32] to [K/2, 32, 2]
template <typename scalar_t>
inline void pack_vnni_Kx32(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int K,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[2];

  int k = 0;
  for (; k < K; ++k) {
    vinputs[k] = _mm512_loadu_si512(src + k * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  // pack value
  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2, d0);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2 + 32, d1);
}
#endif

// convert to vnni format
// from [N, K/2, 2] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int NB = div_up(N, 16);
  const int KB = K / 32;  // no remainder

  for (int nb = 0; nb < NB; ++nb) {
    for (int kb = 0; kb < KB; ++kb) {
      // handle 16x512bits each block
      int nb_size = std::min(N - nb * 16, 16);
      pack_vnni_Nx32<scalar_t>(
          /*    dst */ dst + ((kb * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          /*    src */ src + kb * 32 + nb * 16 * ld_src,
          /*      N */ nb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst[k * ld_dst * 2 + n * 2 + d] = src[n * ld_src + k * 2 + d];
      }
    }
  }
#endif
}

// convert to vnni format
// from [K/2, 2, N] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int KB = div_up(K, 2);
  const int NB = N / 32;  // no remainder

  for (int kb = 0; kb < KB; ++kb) {
    for (int nb = 0; nb < NB; ++nb) {
      // handle 2x512bits each block
      int kb_size = std::min(K - kb * 2, 2);
      pack_vnni_Kx32<scalar_t>(
          /*    dst */ dst + ((kb * 2) >> 1) * ld_dst * 2 + nb * 32 * 2,
          /*    src */ src + kb * 2 * ld_src + nb * 32,
          /*      K */ kb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  int k = 0;
  for (; k < (K >> 1) * 2; k += 2) {
    for (int n = 0; n < N; ++n) {
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = src[k * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = src[(k + 1) * ld_src + n];
    }
  }
  if (K % 2 != 0) {
    for (int n = 0; n < N; ++n) {
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 0] = src[(K - 1) * ld_src + n];
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 1] = 0;
    }
    k += 2;
  }
#endif
}

template <typename scalar_t>
void chunk_gated_delta_rule_kernel_impl(
        at::Tensor& output, // [B, T, HV, EV]
        at::Tensor& final_state, // [N, HV, EK, EV]
        at::Tensor& query, // [B, T, HK, EK]
        at::Tensor& key, // [B, T, HK, EK]
        at::Tensor& value, // [B, T, HV, EV]
        at::Tensor& g, // [B, T, HV] FP32
        at::Tensor& beta, // [B, T, HV]
        at::Tensor& cu_seqlens, // [N + 1] INT32
        int64_t chunk_size=64) {
    // query: [B, T, HK, EK] -> [B, HK, T, EK]
    // key: [B, T, HK, EK] -> [B, HK, T, EK]
    // value: [B, T, HV, EV] -> [B, HV, T, EV]
    // g: [B, T, HV] -> [B, HV, T]
    // beta: [B, T, HV] -> [B, HV, T]
    query = query.transpose(1, 2);
    key = key.transpose(1, 2);
    value = value.transpose(1, 2);
    g = g.transpose(1, 2).contiguous();
    beta = beta.transpose(1, 2).contiguous();

    // Sizes
    TORCH_CHECK(query.size(0) == 1);
    int64_t batch_size = final_state.size(0);
    int64_t global_seq_len = query.size(2);
    int64_t qk_num_head = query.size(1);
    int64_t v_num_head = value.size(1);
    int64_t qk_head_size = query.size(3);
    int64_t v_head_size = value.size(3);
    int64_t head_group = v_num_head / qk_num_head;
    float scale = 1.0 / std::sqrt(qk_head_size);
    const int32_t vec_size = at::vec::Vectorized<float>::size();
    const int32_t reduced_vec_size = at::vec::Vectorized<scalar_t>::size();

    // Strides
    int64_t oStrideT = output.stride(1);
    int64_t oStrideH = output.stride(2);
    int64_t qStrideH = query.stride(1);
    int64_t qStrideT = query.stride(2);
    int64_t kStrideH = key.stride(1);
    int64_t kStrideT = key.stride(2);
    int64_t vStrideH = value.stride(1);
    int64_t vStrideT = value.stride(2);
    int64_t gStrideH = g.stride(1);
    int64_t bStrideH = beta.stride(1);
    int64_t final_state_StrideN = final_state.stride(0);
    int64_t final_state_StrideH = final_state.stride(1);
    int64_t final_state_StrideE = final_state.stride(2);

    // Data pointers
    const scalar_t* q_orig = query.const_data_ptr<scalar_t>();
    const scalar_t* k_orig = key.const_data_ptr<scalar_t>();
    const scalar_t* v_orig = value.const_data_ptr<scalar_t>();
    const float* g_orig = g.const_data_ptr<float>();
    const scalar_t* b_orig = beta.const_data_ptr<scalar_t>();
    const int32_t* cu_seqlens_ptr = cu_seqlens.const_data_ptr<int32_t>();
    scalar_t* out = output.data_ptr<scalar_t>();
    float* final_state_data = final_state.data_ptr<float>();

    // Deduce the padded seq lengths
    std::vector<int64_t> pad_start_q(batch_size, 0);
    int64_t largest_total_seq_length = 0;
    int64_t s = 0;
    int64_t e = 0;
    int64_t s_pad = 0;
    int64_t e_pad = 0;
    for (int64_t n = 0; n < batch_size; n++) {
      e = cu_seqlens_ptr[n + 1];
      int64_t seq_len = e - s;
      int64_t pad_size = (chunk_size - seq_len % chunk_size) % chunk_size;
      int64_t total_seq_length = seq_len + pad_size;
      largest_total_seq_length = std::max(total_seq_length, largest_total_seq_length);
      e_pad = s_pad + total_seq_length;
      pad_start_q[n] = s_pad;
      s = e;
      s_pad = e_pad;
    }
    int64_t global_total_seq_length = e_pad;

    // Allocate buffer
    int64_t buff_size = v_num_head * global_total_seq_length // g_pad_data
          + v_num_head * largest_total_seq_length * chunk_size // attn_data
          + batch_size * v_num_head * global_total_seq_length * v_head_size // core_attn
          + v_num_head * largest_total_seq_length * chunk_size // decay_mask
          + v_num_head * largest_total_seq_length * v_head_size // v_beta_attn
          + v_num_head * largest_total_seq_length * qk_head_size; // k_cumdecay
    at::Tensor buff_data = at::empty({buff_size}, query.options().dtype(at::kFloat));
    float* buff = buff_data.data_ptr<float>();
    float* g_pad = buff;
    float* attn = g_pad + v_num_head * global_total_seq_length;
    float* core_attn_out = attn + v_num_head * largest_total_seq_length * chunk_size;
    float* decay_mask = core_attn_out + batch_size * v_num_head * global_total_seq_length * v_head_size;
    float* v_beta_attn = decay_mask + v_num_head * largest_total_seq_length * chunk_size;
    float* k_cumdecay = v_beta_attn + v_num_head * largest_total_seq_length * v_head_size;

    int64_t reduced_buff_size = qk_num_head * global_total_seq_length * qk_head_size // q_pad_data
          + qk_num_head * global_total_seq_length * qk_head_size // k_pad_data
          + v_num_head * global_total_seq_length * v_head_size // v_pad_data
          + v_num_head * global_total_seq_length * qk_head_size // k_beta_data
          + v_num_head * global_total_seq_length * v_head_size // v_beta_data
          + qk_num_head * largest_total_seq_length * qk_head_size // k_transpose
          + v_num_head * largest_total_seq_length * chunk_size // attn_reduced
          + v_num_head * largest_total_seq_length * qk_head_size // k_beta_g
          + v_num_head * largest_total_seq_length * qk_head_size; // k_cumdecay_reduced
    at::Tensor reduced_buff_data = at::empty({reduced_buff_size}, query.options());
    scalar_t* reduced_buff = reduced_buff_data.data_ptr<scalar_t>();
    scalar_t* q_pad = reduced_buff;
    scalar_t* k_pad = q_pad + qk_num_head * global_total_seq_length * qk_head_size;
    scalar_t* v_pad = k_pad + qk_num_head * global_total_seq_length * qk_head_size;
    scalar_t* k_beta = v_pad + v_num_head * global_total_seq_length * v_head_size;
    scalar_t* v_beta = k_beta + v_num_head * global_total_seq_length * qk_head_size;
    scalar_t* k_transpose = v_beta + v_num_head * global_total_seq_length * v_head_size;
    scalar_t* attn_reduced = k_transpose + qk_num_head * largest_total_seq_length * qk_head_size;
    scalar_t* k_beta_g = attn_reduced + v_num_head * largest_total_seq_length * chunk_size;
    scalar_t* k_cumdecay_reduced = k_beta_g + v_num_head * largest_total_seq_length * qk_head_size;

    int64_t num_thread = at::get_num_threads();
    int64_t buff_size_16bit_per_thread =
        /* v_pack */ chunk_size * v_head_size +
        /* k_beta_g_pack  */ chunk_size * qk_head_size +
        /* curr_last_recurrent_state_reduced  */ qk_head_size * v_head_size +
        /* curr_last_recurrent_state_pack_reduced   */ qk_head_size * v_head_size +
        /* k_transpose_i  */ qk_head_size * chunk_size +
        /* attn_i   */ chunk_size * chunk_size * 2 +
        /* attn_i_reduced     */ chunk_size * chunk_size +
        /* v_prime */ chunk_size * v_head_size * 2 +
        /* v_prime_reduced */ chunk_size * v_head_size +
        /* v_prime_pack_reduced */ chunk_size * v_head_size +
        /* qg */ chunk_size * qk_head_size +
        /* attn_inter */ chunk_size * v_head_size * 2 +
        /* kg */ chunk_size * qk_head_size +
        /* kg_transpose */ qk_head_size * chunk_size +
        /* kgv */ qk_head_size * v_head_size * 2;

    at::Tensor thread_buff_data = at::empty({num_thread, buff_size_16bit_per_thread}, query.options());
    scalar_t* thread_buff = thread_buff_data.data_ptr<scalar_t>();

    int64_t start_q = 0;
    int64_t end_q = 0;
    for (int64_t n = 0; n < batch_size; n++) {
        end_q = cu_seqlens_ptr[n + 1];
        auto q_orig_ptr = q_orig + start_q * qStrideT;
        auto k_orig_ptr = k_orig + start_q * kStrideT;
        auto v_orig_ptr = v_orig + start_q * vStrideT;
        auto g_orig_ptr = g_orig + start_q;
        auto b_orig_ptr = b_orig + start_q;
        auto out_ptr = out + start_q * oStrideT;
        auto final_state_ptr = final_state_data + n * final_state_StrideN;

        auto start_q_pad = pad_start_q[n];
        auto core_attn_out_ptr = core_attn_out + start_q_pad * v_head_size;
        auto q_pad_ptr = q_pad + start_q_pad * qk_head_size;
        auto k_pad_ptr = k_pad + start_q_pad * qk_head_size;
        auto v_pad_ptr = v_pad + start_q_pad * v_head_size;
        auto g_pad_ptr = g_pad + start_q_pad;
        auto k_beta_ptr = k_beta + start_q_pad * qk_head_size;
        auto v_beta_ptr = v_beta + start_q_pad * v_head_size;
        int64_t seq_len = end_q - start_q;
        int64_t pad_size = (chunk_size - seq_len % chunk_size) % chunk_size;
        int64_t total_seq_length = seq_len + pad_size;
        int64_t num_chunk = total_seq_length / chunk_size;

        // query = query * scale
        // k_beta = key * beta.unsqueeze(-1)
        // v_beta = value * beta.unsqueeze(-1)
        at::parallel_for(0, qk_num_head * seq_len, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h_qk = 0, l = 0;
            at::native::data_index_init(begin, h_qk, qk_num_head, l, seq_len);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_q_orig = q_orig_ptr + h_qk * qStrideH + l * qStrideT;
                auto curr_k_orig = k_orig_ptr + h_qk * kStrideH + l * kStrideT;
                auto curr_q_pad = q_pad_ptr + h_qk * global_total_seq_length * qk_head_size + l * qk_head_size;
                auto curr_k_pad = k_pad_ptr + h_qk * global_total_seq_length * qk_head_size + l * qk_head_size;

                int64_t i = 0;
                scalar_t scale_reduced = static_cast<scalar_t>(scale);
                auto vec_scale_reduced = at::vec::Vectorized<scalar_t>(scale_reduced);
                for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
                    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_q_orig + i);
                    auto tmp2 = tmp0 * vec_scale_reduced;
                    tmp2.store(curr_q_pad + i);
                    auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i);
                    tmp3.store(curr_k_pad + i);
                }
                if (i < qk_head_size) {
                    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_q_orig + i, qk_head_size - i);
                    auto tmp2 = tmp0 * vec_scale_reduced;
                    tmp2.store(curr_q_pad + i, qk_head_size - i);
                    auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i, qk_head_size - i);
                    tmp3.store(curr_k_pad + i, qk_head_size - i);
                }

                for (auto hi = 0; hi < head_group; hi++) {
                  int64_t h = h_qk * head_group + hi;
                  auto curr_v_orig = v_orig_ptr + h * vStrideH + l * vStrideT;
                  auto curr_b_orig = b_orig_ptr + h * bStrideH;
                  scalar_t b_orig_val_reduced = l < seq_len ? *(curr_b_orig + l) : static_cast<scalar_t>(0);
                  auto curr_v_pad = v_pad_ptr + h * global_total_seq_length * v_head_size + l * v_head_size;
                  auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + l * qk_head_size;
                  auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + l * v_head_size;

                  // query = query * scale
                  // k_beta = key * beta.unsqueeze(-1)
                  int64_t i = 0;
                  auto vec_b_reduced = at::vec::Vectorized<scalar_t>(b_orig_val_reduced);
                  for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
                      auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i);
                      auto tmp5 = tmp3 * vec_b_reduced;
                      tmp5.store(curr_k_beta + i);
                  }
                  if (i < qk_head_size) {
                      auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i, qk_head_size - i);
                      auto tmp5 = tmp3 * vec_b_reduced;
                      tmp5.store(curr_k_beta + i, qk_head_size - i);
                  }
                  // v_beta = value * beta.unsqueeze(-1)
                  i = 0;
                  for (; i < reduced_vec_size * (v_head_size / reduced_vec_size); i += reduced_vec_size) {
                      auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_v_orig + i);
                      tmp3.store(curr_v_pad + i);
                      auto tmp5 = tmp3 * vec_b_reduced;
                      tmp5.store(curr_v_beta + i);
                  }
                  if (i < v_head_size) {
                      auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_v_orig + i, v_head_size - i);
                      tmp3.store(curr_v_pad + i, v_head_size - i);
                      auto tmp5 = tmp3 * vec_b_reduced;
                      tmp5.store(curr_v_beta + i, v_head_size - i);
                  }
                }
                // Move to the next query
                at::native::data_index_step(h_qk, qk_num_head, l, seq_len);
            }
        });

        // Padding for q/k/v/beta
        at::parallel_for(0, qk_num_head * pad_size, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h_qk = 0, l = 0;
            at::native::data_index_init(begin, h_qk, qk_num_head, l, pad_size);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_q_pad = q_pad_ptr + h_qk * global_total_seq_length * qk_head_size + (seq_len + l) * qk_head_size;
                auto curr_k_pad = k_pad_ptr + h_qk * global_total_seq_length * qk_head_size + (seq_len + l) * qk_head_size;

                int64_t i = 0;
                auto vec_zero = at::vec::Vectorized<scalar_t>(0.0);
                for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
                    vec_zero.store(curr_q_pad + i);
                    vec_zero.store(curr_k_pad + i);
                }
                if (i < qk_head_size) {
                    vec_zero.store(curr_q_pad + i, qk_head_size - i);
                    vec_zero.store(curr_k_pad + i, qk_head_size - i);
                }

                for (auto hi = 0; hi < head_group; hi++) {
                  int64_t h = h_qk * head_group + hi;
                  auto curr_v_pad = v_pad_ptr + h * global_total_seq_length * v_head_size + (seq_len + l) * v_head_size;
                  auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + (seq_len + l) * qk_head_size;
                  auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + (seq_len + l) * v_head_size;

                  int64_t i = 0;
                  for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
                      vec_zero.store(curr_k_beta + i);
                  }
                  if (i < qk_head_size) {
                      vec_zero.store(curr_k_beta + i, qk_head_size - i);
                  }
                  i = 0;
                  for (; i < reduced_vec_size * (v_head_size / reduced_vec_size); i += reduced_vec_size) {
                      vec_zero.store(curr_v_pad + i);
                      vec_zero.store(curr_v_beta + i);
                  }
                  if (i < v_head_size) {
                      vec_zero.store(curr_v_pad + i, v_head_size - i);
                      vec_zero.store(curr_v_beta + i, v_head_size - i);
                  }
                }
                // Move to the next query
                at::native::data_index_step(h_qk, qk_num_head, l, pad_size);
            }
        });

        // Padding for g
        // g = g.cumsum(dim=-1)
        // g: [B, HV, num_chunk, chunk_size]
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_g_orig = g_orig_ptr + h * gStrideH + c * chunk_size;
                auto curr_g_pad = g_pad_ptr + h * global_total_seq_length + c * chunk_size;
                float acc_val = 0;
                for (int64_t i = 0; i < chunk_size; i++) {
                    if (c * chunk_size + i < seq_len) {
                        acc_val += curr_g_orig[i];
                    }
                    curr_g_pad[i] = acc_val;
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        // decay_mask: [B, HV, num_chunk, chunk_size, chunk_size]
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_g_pad = g_pad_ptr + h * global_total_seq_length + c * chunk_size;
                auto curr_decay_mask = decay_mask + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                for (int64_t i = 0; i < chunk_size; i++) {
                    float curr_g_pad_i = static_cast<float>(curr_g_pad[i]);
                    auto vec_curr_g_pad_i = at::vec::Vectorized<float>(curr_g_pad_i);
                    int64_t j = 0;
                    int64_t len = i + 1;
                    for (; j < vec_size * (len / vec_size); j += vec_size) {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(curr_g_pad + j);
                        auto tmp1 = vec_curr_g_pad_i - tmp0;
                        auto tmp2 = tmp1.exp();
                        tmp2.store(curr_decay_mask + i * chunk_size + j);
                    }
                    if (j < len) {
                      auto tmp0 = at::vec::Vectorized<float>::loadu(curr_g_pad + j, len - j);
                        auto tmp1 = vec_curr_g_pad_i - tmp0;
                        auto tmp2 = tmp1.exp();
                        tmp2.store(curr_decay_mask + i * chunk_size + j, len - j);
                    }
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // attn = k_beta @ key.transpose(-1, -2)
        // attn: [B, HV, num_chunk, chunk_size, chunk_size]
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                int64_t h_qk = h / head_group;
                auto curr_k_pad = k_pad_ptr + h_qk * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_transpose = k_transpose + h_qk * num_chunk * qk_head_size * chunk_size + c * qk_head_size * chunk_size;
                auto curr_attn = attn + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                auto curr_decay_mask = decay_mask + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                // transpose and pack for key
                pack_vnni<scalar_t>(
                    /*    dst */ curr_k_transpose,
                    /*    src */ curr_k_pad,
                    /*     N  */ chunk_size,
                    /*     K  */ qk_head_size,
                    /* ld_src */ qk_head_size,
                    /* ld_dst */ chunk_size);
                // k_beta @ key.transpose(-1, -2)
                at::native::cpublas::brgemm(
                    /* M */ chunk_size,
                    /* N */ chunk_size,
                    /* K */ qk_head_size,
                    /* lda */ qk_head_size,
                    /* ldb */ chunk_size,
                    /* ldc */ chunk_size,
                    /* add_C */ false,
                    /* A */ curr_k_beta,
                    /* B */ curr_k_transpose,
                    /* C */ curr_attn);
                // attn = attn * decay_mask
                for (int64_t m = 0; m < chunk_size; m++) {
                    at::vec::map2<float>(
                        [](at::vec::Vectorized<float> x, at::vec::Vectorized<float> y) { return at::vec::Vectorized<float>(0) - x * y; },
                        curr_attn + m * chunk_size,
                        curr_attn + m * chunk_size,
                        curr_decay_mask + m * chunk_size,
                        chunk_size);
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // chunk decay
        // attn: [B, HV, num_chunk, chunk_size, chunk_size]
        // mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
        // attn = -attn.masked_fill(mask, 0)
        // attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2) [B, HV, num_chunk, i]
        // attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_attn = attn + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                auto curr_attn_reduced = attn_reduced + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                // attn = -attn.masked_fill(mask, 0)
                for (int i = 0; i < chunk_size; i++) {
                    const auto vec_zero = at::vec::Vectorized<float>(0);
                    int64_t len = chunk_size - i;
                    int64_t front = len % vec_size;
                    int64_t j = i;
                    // first masked vec for alignment
                    if (front > 0) {
                        vec_zero.store(curr_attn + i * chunk_size + j, front);
                        j += front;
                    }
                    for (; j < vec_size * (chunk_size / vec_size); j += vec_size) {
                        vec_zero.store(curr_attn + i * chunk_size + j);
                    }
                }
                for (int i = 1; i < chunk_size; i++) {
                    // row = attn[..., i, :i] [B, HK, num_chunk, i]
                    at::Tensor row_data = at::empty({i}, query.options().dtype(at::kFloat));
                    float* row = row_data.data_ptr<float>();
                    int64_t j = 0;
                    int64_t len = i;
                    for (; j < vec_size * (len / vec_size); j += vec_size) {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(curr_attn + i * chunk_size + j);
                        tmp0.store(row + j);
                    }
                    if (j < len) {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(curr_attn + i * chunk_size + j, len - j);
                        tmp0.store(row + j, len - j);
                    }
                    // (row.unsqueeze(-1) * sub).sum(-2)
                    at::Tensor updated_data = at::zeros({i}, query.options().dtype(at::kFloat));
                    float* updated = updated_data.data_ptr<float>();
                    for (int k = 0; k < i; k++) {
                        float row_k = row[k];
                        auto vec_row_k = at::vec::Vectorized<float>(row_k);
                        int64_t j = 0;
                        int64_t len = i;
                        for (; j < vec_size * (len / vec_size); j += vec_size) {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(curr_attn + k * chunk_size + j);
                            auto tmp1 = vec_row_k * tmp0;
                            auto tmp2 = at::vec::Vectorized<float>::loadu(updated + j);
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(updated + j);
                        }
                        if (j < len) {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(curr_attn + k * chunk_size + j, len - j);
                            auto tmp1 = vec_row_k * tmp0;
                            auto tmp2 = at::vec::Vectorized<float>::loadu(updated + j);
                            auto tmp3 = tmp1 + tmp2;
                            tmp3.store(updated + j, len - j);
                        }
                    }
                    // attn[..., i, :i] = row + sum(...)
                    j = 0;
                    len = i;
                    for (; j < vec_size * (len / vec_size); j += vec_size) {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(row + j);
                        auto tmp1 = at::vec::Vectorized<float>::loadu(updated + j);
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(curr_attn + i * chunk_size + j);
                    }
                    if (j < len) {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(row + j, len - j);
                        auto tmp1 = at::vec::Vectorized<float>::loadu(updated + j, len - j);
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(curr_attn + i * chunk_size + j, len - j);
                    }
                }
                for (int i = 0; i < chunk_size; i++) {
                    curr_attn[i * chunk_size + i] += 1.0f;
                    at::vec::map<scalar_t>(
                            [](at::vec::Vectorized<float> x) { return x; },
                            curr_attn_reduced + i * chunk_size,
                            curr_attn + i * chunk_size,
                            chunk_size);
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // v_beta_attn = attn @ v_beta
        // k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        // v_beta_attn: [B, HV, num_chunk, chunk_size, EV]
        // k_beta_g = k_beta * g: [B, HV, num_chunk, chunk_size, EK]
        // k_cumdecay: [B, HV, num_chunk, chunk_size, EK]
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            int ompIdx = at::get_thread_num();
            int64_t offset = 0;
            scalar_t* thread_buff_ptr = thread_buff + ompIdx * buff_size_16bit_per_thread;
            scalar_t* v_pack = thread_buff_ptr + offset;
            offset += chunk_size * v_head_size;
            scalar_t* k_beta_g_pack = thread_buff_ptr + offset;
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_beta_g = k_beta_g + h * num_chunk * chunk_size * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_cumdecay = k_cumdecay + h * num_chunk * chunk_size * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_cumdecay_reduced = k_cumdecay_reduced + h * num_chunk * chunk_size * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_attn = attn + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                auto curr_attn_reduced = attn_reduced + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + c * chunk_size * v_head_size;
                auto curr_value = v_beta_attn + h * num_chunk * chunk_size * v_head_size + c * chunk_size * v_head_size;
                auto curr_g_pad = g_pad_ptr + h * global_total_seq_length + c * chunk_size;

                // pack for value
                pack_vnni2<scalar_t>(
                    /*    dst */ v_pack,
                    /*    src */ curr_v_beta,
                    /*     N  */ chunk_size,
                    /*     K  */ v_head_size,
                    /* ld_src */ v_head_size,
                    /* ld_dst */ v_head_size);
                // value = attn @ v_beta
                at::native::cpublas::brgemm(
                    /* M */ chunk_size,
                    /* N */ v_head_size,
                    /* K */ chunk_size,
                    /* lda */ chunk_size,
                    /* ldb */ v_head_size,
                    /* ldc */ v_head_size,
                    /* add_C */ false,
                    /* A */ curr_attn_reduced,
                    /* B */ v_pack,
                    /* C */ curr_value);
                // k_beta_g = k_beta * g.exp().unsqueeze(-1)
                for (int64_t j = 0; j < chunk_size; j++) {
                    int64_t i = 0;
                    float g_exp = std::exp(curr_g_pad[j]);
                    scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
                    auto vec_g_exp_reduced = at::vec::Vectorized<scalar_t>(g_exp_reduced);
                    for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
                        auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_k_beta + j * qk_head_size + i);
                        auto tmp1 = tmp0 * vec_g_exp_reduced;
                        tmp1.store(curr_k_beta_g + j * qk_head_size + i);
                    }
                    if (i < qk_head_size) {
                        auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_k_beta + j * qk_head_size + i, qk_head_size - i);
                        auto tmp1 = tmp0 * vec_g_exp_reduced;
                        tmp1.store(curr_k_beta_g + j * qk_head_size + i, qk_head_size - i);
                    }
                }
                // pack for k_beta_g
                pack_vnni2<scalar_t>(
                    /*    dst */ k_beta_g_pack,
                    /*    src */ curr_k_beta_g,
                    /*     N  */ chunk_size,
                    /*     K  */ qk_head_size,
                    /* ld_src */ qk_head_size,
                    /* ld_dst */ qk_head_size);
                // k_cumdecay = attn @ k_beta_g
                at::native::cpublas::brgemm(
                    /* M */ chunk_size,
                    /* N */ qk_head_size,
                    /* K */ chunk_size,
                    /* lda */ chunk_size,
                    /* ldb */ qk_head_size,
                    /* ldc */ qk_head_size,
                    /* add_C */ false,
                    /* A */ curr_attn_reduced,
                    /* B */ k_beta_g_pack,
                    /* C */ curr_k_cumdecay);
                for (int i = 0; i < chunk_size; i++) {
                    at::vec::map<scalar_t>(
                          [](at::vec::Vectorized<float> x) { return x; },
                          curr_k_cumdecay_reduced + i * qk_head_size,
                          curr_k_cumdecay + i * qk_head_size,
                          qk_head_size);
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // for each chunk
        at::parallel_for(0, v_num_head, 1, [&](int64_t begin, int64_t end) {
            int64_t h = 0;
            at::native::data_index_init(begin, h, v_num_head);
            int ompIdx = at::get_thread_num();
            int64_t offset = 0;
            scalar_t* thread_buff_ptr = thread_buff + ompIdx * buff_size_16bit_per_thread;
            scalar_t* curr_last_recurrent_state_reduced = thread_buff_ptr + offset;
            offset += qk_head_size * v_head_size;
            scalar_t* curr_last_recurrent_state_pack_reduced = thread_buff_ptr + offset;
            offset += qk_head_size * v_head_size;
            scalar_t* k_transpose_i = thread_buff_ptr + offset;
            offset += qk_head_size * chunk_size;
            float* attn_i = reinterpret_cast<float*>(thread_buff_ptr + offset);
            offset += chunk_size * chunk_size * 2;
            scalar_t* attn_i_reduced = thread_buff_ptr + offset;
            offset += chunk_size * chunk_size;
            float* v_prime = reinterpret_cast<float*>(thread_buff_ptr + offset);
            offset += chunk_size * v_head_size * 2;
            scalar_t* v_prime_reduced = thread_buff_ptr + offset;
            offset += chunk_size * v_head_size;
            scalar_t* v_prime_pack_reduced = thread_buff_ptr + offset;
            offset += chunk_size * v_head_size;
            scalar_t* qg = thread_buff_ptr + offset;
            offset += chunk_size * qk_head_size;
            float* attn_inter = reinterpret_cast<float*>(thread_buff_ptr + offset);
            offset += chunk_size * v_head_size * 2;
            scalar_t* kg = thread_buff_ptr + offset;
            offset += chunk_size * qk_head_size;
            scalar_t* kg_transpose = thread_buff_ptr + offset;
            offset += qk_head_size * chunk_size;
            float* kgv = reinterpret_cast<float*>(thread_buff_ptr + offset);
            offset += qk_head_size * v_head_size * 2;

            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                int64_t h_qk = h / head_group;
                auto curr_q = q_pad_ptr + h_qk * global_total_seq_length * qk_head_size; // [num_chunk, chunk_size, EK]
                auto curr_k = k_pad_ptr + h_qk * global_total_seq_length * qk_head_size; // [num_chunk, chunk_size, EK]
                auto curr_v = v_beta_attn + h * num_chunk * chunk_size * v_head_size; // [num_chunk, chunk_size, EV]
                auto curr_decay_mask = decay_mask + h * num_chunk * chunk_size * chunk_size; // [num_chunk, chunk_size, chunk_size]
                auto curr_k_cumdecay_reduced = k_cumdecay_reduced + h * num_chunk * chunk_size * qk_head_size; // [num_chunk, chunk_size, EK]
                auto curr_last_recurrent_state = final_state_ptr + h * final_state_StrideH; // [EK, EV]
                auto curr_g_pad = g_pad_ptr + h * global_total_seq_length; // [num_chunk, chunk_size]
                auto curr_core_attn_out = core_attn_out_ptr + h * global_total_seq_length * v_head_size; // [num_chunk, chunk_size, EV]
                for (int64_t c = 0; c < num_chunk; c++) {
                    for (int i = 0; i < qk_head_size; i++) {
                        at::vec::map<scalar_t>(
                              [](at::vec::Vectorized<float> x) { return x; },
                              curr_last_recurrent_state_reduced + i * v_head_size,
                              curr_last_recurrent_state + i * v_head_size,
                              v_head_size);
                    }
                    auto q_i = curr_q + c * chunk_size * qk_head_size; // [chunk_size, EK]
                    auto k_i = curr_k + c * chunk_size * qk_head_size; // [chunk_size, EK]
                    auto v_i = curr_v + c * chunk_size * v_head_size; // [chunk_size, EV]
                    auto decay_mask_i = curr_decay_mask + c * chunk_size * chunk_size; // [chunk_size, chunk_size]
                    auto k_cumdecay_i_reduced = curr_k_cumdecay_reduced + c * chunk_size * qk_head_size; // [chunk_size, EK]
                    auto g_pad_i = curr_g_pad + c * chunk_size; // [chunk_size]
                    auto core_attn_out_i = curr_core_attn_out + c * chunk_size * v_head_size; // [chunk_size, EV]

                    // attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
                    // k_transpose_i = k_i.transpose(-1, -2)
                    pack_vnni<scalar_t>(
                        /*    dst */ k_transpose_i,
                        /*    src */ k_i,
                        /*     N  */ chunk_size,
                        /*     K  */ qk_head_size,
                        /* ld_src */ qk_head_size,
                        /* ld_dst */ chunk_size);
                    // attn_i = q_i @ k_transpose_i
                    at::native::cpublas::brgemm(
                        /* M */ chunk_size,
                        /* N */ chunk_size,
                        /* K */ qk_head_size,
                        /* lda */ qk_head_size,
                        /* ldb */ chunk_size,
                        /* ldc */ chunk_size,
                        /* add_C */ false,
                        /* A */ q_i,
                        /* B */ k_transpose_i,
                        /* C */ attn_i);
                    // attn_i = attn_i * decay_mask_i
                    for (int64_t m = 0; m < chunk_size; m++) {
                        auto attn_i_m = attn_i + m * chunk_size;
                        auto attn_i_reduced_m = attn_i_reduced + m * chunk_size;
                        auto decay_mask_i_m = decay_mask_i + m * chunk_size;
                        int64_t n = 0;
                        for (; n < vec_size * (chunk_size / vec_size); n += vec_size) {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(attn_i_m + n);
                            auto tmp1 = at::vec::Vectorized<float>::loadu(decay_mask_i_m + n);
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
                            tmp3.store(attn_i_reduced_m + n, vec_size);
                        }
                        if (n < chunk_size) {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(attn_i_m + n, chunk_size - n);
                            auto tmp1 = at::vec::Vectorized<float>::loadu(decay_mask_i_m + n, chunk_size - n);
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
                            tmp3.store(attn_i_reduced_m + n, chunk_size - n);
                        }
                    }
                    // mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
                    // attn_i = attn_i.masked_fill_(mask, 0)
                    for (int i = 0; i < chunk_size - 1; i++) {
                        const auto vec_zero = at::vec::Vectorized<scalar_t>(0);
                        int64_t len = chunk_size - i - 1;
                        int64_t front = len % reduced_vec_size;
                        int64_t j = i + 1;
                        // first masked vec for alignment
                        if (front > 0) {
                            vec_zero.store(attn_i_reduced + i * chunk_size + j, front);
                            j += front;
                        }
                        for (; j < reduced_vec_size * (chunk_size / reduced_vec_size); j += reduced_vec_size) {
                            vec_zero.store(attn_i_reduced + i * chunk_size + j);
                        }
                    }

                    // pack for curr_last_recurrent_state
                    pack_vnni2<scalar_t>(
                        /*    dst */ curr_last_recurrent_state_pack_reduced,
                        /*    src */ curr_last_recurrent_state_reduced,
                        /*     N  */ qk_head_size,
                        /*     K  */ v_head_size,
                        /* ld_src */ v_head_size,
                        /* ld_dst */ v_head_size);

                    // v_prime = k_cumdecay_i @ curr_last_recurrent_state: [chunk_size, EV]
                    // k_cumdecay_i: [chunk_size, EK]
                    // curr_last_recurrent_state: [EK, EV]
                    at::native::cpublas::brgemm(
                        /* M */ chunk_size,
                        /* N */ v_head_size,
                        /* K */ qk_head_size,
                        /* lda */ qk_head_size,
                        /* ldb */ v_head_size,
                        /* ldc */ v_head_size,
                        /* add_C */ false,
                        /* A */ k_cumdecay_i_reduced,
                        /* B */ curr_last_recurrent_state_pack_reduced,
                        /* C */ v_prime);

                    // v_new = v_prime = v_i - v_prime
                    // v_i: [chunk_size, EV]
                    for (int64_t m = 0; m < chunk_size; m++) {
                        int64_t i = 0;
                        for (; i < vec_size * (v_head_size / vec_size); i += vec_size) {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(v_i + m * v_head_size + i);
                            auto tmp1 = at::vec::Vectorized<float>::loadu(v_prime + m * v_head_size + i);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
                            tmp3.store(v_prime_reduced + m * v_head_size + i, vec_size);
                        }
                        if (i < v_head_size) {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(v_i + m * v_head_size + i, v_head_size - i);
                            auto tmp1 = at::vec::Vectorized<float>::loadu(v_prime + m * v_head_size + i, v_head_size - i);
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
                            tmp3.store(v_prime_reduced + m * v_head_size + i, v_head_size - i);
                        }
                    }

                    // attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
                    // qg = q_i * g[:, :, i, :, None].exp(): [chunk_size, EK]
                    // q_i: [chunk_size, EK]
                    // g[:, :, i, :, None]: [chunk_size, 1]
                    for (int64_t m = 0; m < chunk_size; m++) {
                        auto g_pad_i_m = g_pad_i + m;
                        auto g_exp = std::exp(*g_pad_i_m);
                        int64_t i = 0;
                        scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
                        auto vec_g_exp_reduced = at::vec::Vectorized<scalar_t>(g_exp_reduced);
                        for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
                            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(q_i + m * qk_head_size + i);
                            auto tmp2 = tmp0 * vec_g_exp_reduced;
                            tmp2.store(qg + m * qk_head_size + i);
                        }
                        if (i < qk_head_size) {
                            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(q_i + m * qk_head_size + i, qk_head_size - i);
                            auto tmp2 = tmp0 * vec_g_exp_reduced;
                            tmp2.store(qg + m * qk_head_size + i, qk_head_size - i);
                        }
                    }
                    // attn_inter = qg @ curr_last_recurrent_state: [chunk_size, EV]
                    // curr_last_recurrent_state: [EK, EV]
                    at::native::cpublas::brgemm(
                        /* M */ chunk_size,
                        /* N */ v_head_size,
                        /* K */ qk_head_size,
                        /* lda */ qk_head_size,
                        /* ldb */ v_head_size,
                        /* ldc */ v_head_size,
                        /* add_C */ false,
                        /* A */ qg,
                        /* B */ curr_last_recurrent_state_pack_reduced,
                        /* C */ attn_inter);

                    // core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
                    // pack for v_prime
                    pack_vnni2<scalar_t>(
                        /*    dst */ v_prime_pack_reduced,
                        /*    src */ v_prime_reduced,
                        /*     N  */ chunk_size,
                        /*     K  */ v_head_size,
                        /* ld_src */ v_head_size,
                        /* ld_dst */ v_head_size);
                    // attn_inter = attn_inter + attn_i @ v_new: [chunk_size, EV]
                    // attn_i: [chunk_size, chunk_size]
                    // v_new: [chunk_size, EV]
                    at::native::cpublas::brgemm(
                        /* M */ chunk_size,
                        /* N */ v_head_size,
                        /* K */ chunk_size,
                        /* lda */ chunk_size,
                        /* ldb */ v_head_size,
                        /* ldc */ v_head_size,
                        /* add_C */ true,
                        /* A */ attn_i_reduced,
                        /* B */ v_prime_pack_reduced,
                        /* C */ attn_inter);

                    // core_attn_out[:, :, i] = attn_inter
                    for (int64_t m = 0; m < chunk_size; m++) {
                        at::vec::map<float>(
                            [](at::vec::Vectorized<float> x) { return x; },
                            core_attn_out_i + m * v_head_size,
                            attn_inter + m * v_head_size,
                            v_head_size);
                    }

                    // last_recurrent_state = (
                    //     last_recurrent_state * g[:, :, i, -1, None, None].exp()
                    //     + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
                    // )
                    // 1) last_recurrent_state * g[:, :, i, -1, None, None].exp()
                        // curr_last_recurrent_state: [EK, EV]
                        // g[:, :, i, -1, None, None]: [1, 1]
                        // last_recurrent_state * g[:, :, i, -1, None, None].exp(): [EK, EV]
                    auto g_pad_i_last = g_pad_i + chunk_size - 1;
                    auto g_exp_last = std::exp(g_pad_i_last[0]);
                    for (int64_t m = 0; m < qk_head_size; m++) {
                        int64_t i = 0;
                        auto vec_g_exp_last = at::vec::Vectorized<float>(g_exp_last);
                        for (; i < vec_size * (v_head_size / vec_size); i += vec_size) {
                            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_last_recurrent_state_reduced + m * v_head_size + i);
                            auto tmp1 = at::vec::convert<float>(tmp0);
                            auto tmp2 = tmp1 * vec_g_exp_last;
                            tmp2.store(curr_last_recurrent_state + m * v_head_size + i);
                        }
                        if (i < v_head_size) {
                            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_last_recurrent_state_reduced + m * v_head_size + i, v_head_size - i);
                            auto tmp1 = at::vec::convert<float>(tmp0);
                            auto tmp2 = tmp1 * vec_g_exp_last;
                            tmp2.store(curr_last_recurrent_state + m * v_head_size + i, v_head_size - i);
                        }
                    }
                    // 2) (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
                        // k_i: [chunk_size, EK]
                        // g[:, :, i, -1, None]: [1]
                        // g[:, :, i]: [chunk_size]
                        // (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]: [chunk_size, 1]
                        // kg = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]: [chunk_size, EK]
                        // (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2): [EK, chunk_size]
                        // v_new: [chunk_size, EV]
                        // (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new: [EK, EV]
                    // kg = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                    for (int64_t m = 0; m < chunk_size; m++) {
                        auto g_exp = std::exp((g_pad_i_last[0] - g_pad_i[m]));
                        int64_t i = 0;
                        scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
                        auto vec_g_exp_reduced = at::vec::Vectorized<scalar_t>(g_exp_reduced);
                        for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
                            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(k_i + m * qk_head_size + i);
                            auto tmp2 = tmp0 * vec_g_exp_reduced;
                            tmp2.store(kg + m * qk_head_size + i);
                        }
                        if (i < qk_head_size) {
                            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(k_i + m * qk_head_size + i, qk_head_size - i);
                            auto tmp2 = tmp0 * vec_g_exp_reduced;
                            tmp2.store(kg + m * qk_head_size + i, qk_head_size - i);
                        }
                    }
                    // kg.transpose(-1, -2): [EK, chunk_size]
                    at::native::utils::transpose<scalar_t>(
                        /* M */ chunk_size,
                        /* N */ qk_head_size, 
                        /* src */ kg,
                        /* ld_src */ qk_head_size,
                        /* dst */ kg_transpose,
                        /* ld_dst */ chunk_size);
                    // kgv = kg.transpose(-1, -2) @ v_new
                    // v_new: [chunk_size, EV]
                    at::native::cpublas::brgemm(
                        /* M */ qk_head_size,
                        /* N */ v_head_size,
                        /* K */ chunk_size,
                        /* lda */ chunk_size,
                        /* ldb */ v_head_size,
                        /* ldc */ v_head_size,
                        /* add_C */ false,
                        /* A */ kg_transpose,
                        /* B */ v_prime_pack_reduced,
                        /* C */ kgv);
                    // last_recurrent_state = 1) + 2)
                    for (int64_t m = 0; m < qk_head_size; m++) {
                        at::vec::map2<float>(
                            [](at::vec::Vectorized<float> x, at::vec::Vectorized<float> y) { return x + y; },
                            curr_last_recurrent_state + m * v_head_size,
                            curr_last_recurrent_state + m * v_head_size,
                            kgv + m * v_head_size,
                            v_head_size);
                    }
                }

                // core_attn_out -> output
                // output: [B, T, HV, EV]
                // core_attn_out: [B, HV, padded_T, EV]
                auto curr_out = out_ptr + h * oStrideH;
                for (int64_t m = 0; m < seq_len; m++) {
                    at::vec::map<scalar_t>(
                        [](at::vec::Vectorized<float> x) { return x; },
                        curr_out + m * oStrideT,
                        curr_core_attn_out + m * v_head_size,
                        v_head_size);
                }

                // Move to the next query
                at::native::data_index_step(h, v_num_head);
            }
        });

        start_q = end_q;
    }
}
}  // anonymous namespace

extern at::Tensor qwen3_next_l2norm_cpu(at::Tensor& input, double eps);


// query: [B, T, HK, EK]
// key: [B, T, HK, EK]
// value: [B, T, HV, EV]
// g: [B, T, HV] FP32
// beta: [B, T, HV]
// initial_state: [N, HV, EK, EV] FP32
// output_final_state: bool
// cu_seqlens: [N + 1] INT32
// head_first: bool
// use_qk_l2norm_in_kernel: bool
std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu(
        at::Tensor& query,
        at::Tensor& key,
        at::Tensor& value,
        at::Tensor& g,
        at::Tensor& beta,
        at::Tensor& initial_state,
        bool output_final_state,
        at::Tensor& cu_seqlens,
        bool head_first,
        bool use_qk_l2norm_in_kernel) {
    RECORD_FUNCTION("sgl-kernel::chunk_gated_delta_rule_cpu", std::vector<c10::IValue>({query, key, value, g, beta, initial_state}));

    TORCH_CHECK(head_first == false, "chunk_gated_delta_rule_cpu does not support head first");
    TORCH_CHECK(query.dtype() == at::kBFloat16 && query.dtype() == key.dtype()
        && query.dtype() == value.dtype() && query.dtype() == beta.dtype());
    TORCH_CHECK(g.dtype() == at::kFloat && g.dtype() == initial_state.dtype());
    TORCH_CHECK(cu_seqlens.dtype() == at::kInt);
    CHECK_DIM(4, query);
    CHECK_DIM(4, key);
    CHECK_DIM(4, value);
    CHECK_DIM(3, g);
    CHECK_DIM(3, beta);
    CHECK_DIM(1, cu_seqlens);
    CHECK_DIM(4, initial_state);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(value);
    CHECK_CONTIGUOUS(g);
    CHECK_CONTIGUOUS(beta);
    CHECK_CONTIGUOUS(initial_state);
    int64_t B = query.size(0);
    int64_t T = query.size(1);
    int64_t HK = query.size(2);
    int64_t EK = query.size(3);
    int64_t HV = value.size(2);
    int64_t EV = value.size(3);
    CHECK_EQ(B, 1);
    CHECK_EQ(key.size(0), B);
    CHECK_EQ(key.size(1), T);
    CHECK_EQ(key.size(2), HK);
    CHECK_EQ(key.size(3), EK);
    CHECK_EQ(value.size(0), B);
    CHECK_EQ(value.size(1), T);
    CHECK_EQ(g.size(0), B);
    CHECK_EQ(g.size(1), T);
    CHECK_EQ(g.size(2), HV);
    CHECK_EQ(beta.size(0), B);
    CHECK_EQ(beta.size(1), T);
    CHECK_EQ(beta.size(2), HV);
    CHECK_EQ(initial_state.size(1), HV);
    CHECK_EQ(initial_state.size(2), EK);
    CHECK_EQ(initial_state.size(3), EV);
    CHECK_EQ(HV % HK, 0);

    at::Tensor output = at::empty_like(value, value.options()); // [B, T, HV, EV]
    at::Tensor final_state = initial_state.to(at::kFloat); // [N, HV, EK, EV]
    at::Tensor query_ = query.contiguous();
    at::Tensor key_ = key.contiguous();
    if (use_qk_l2norm_in_kernel) {
        query_ = qwen3_next_l2norm_cpu(query_, 1e-6);
        key_ = qwen3_next_l2norm_cpu(key_, 1e-6);
    }

    AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "chunk_gated_delta_rule_kernel", [&] {
        chunk_gated_delta_rule_kernel_impl<scalar_t>(
            output,
            final_state,
            query_,
            key_,
            value,
            g,
            beta,
            cu_seqlens
        );
    });
    return std::make_tuple(std::move(output), std::move(final_state));
}
