#include "common.h"
#include "gemm.h"
#include "vec.h"
namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    bVec x = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * weight_vec;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

// acc from [topk, K] to [K]
template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  if (topk == 1) {
    // do copy for topk = 1
    copy_stub(out, input, K);
  } else {
    // do sum for topk != 1
    int64_t d;
#pragma GCC unroll 4
    for (d = 0; d <= K - kVecSize; d += kVecSize) {
      fVec sum_fvec0 = fVec(0.f);
      fVec sum_fvec1 = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        bVec x_bvec = bVec::loadu(input + t * K + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec0 += x_fvec0;
        sum_fvec1 += x_fvec1;
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(sum_fvec0, sum_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += static_cast<float>(input[t * K + d]);
      }
      out[d] = static_cast<scalar_t>(sum_val);
    }
  }
}

// out = input + input2 * scale
template <typename scalar_t>
inline void add_mul_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ input2,
    float scale,
    int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_vec = fVec(scale);

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    bVec x_bvec = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x_bvec);

    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

    x0 = x0 + y0 * s_vec;
    x1 = x1 + y1 * s_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + float(input2[d]) * scale);
  }
}

template <typename scalar_t>
inline void silu_and_mul_stub(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const scalar_t* __restrict__ input2, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);

  // no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += bVec::size()) {
    bVec x = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    bVec y = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y);
    x0 = x0 / (one + x0.neg().exp_u20());
    x1 = x1 / (one + x1.neg().exp_u20());
    x0 = x0 * y0;
    x1 = x1 * y1;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
}

}  // anonymous namespace

// TODO: stride access
template <int64_t N>
inline void copy_bias(const float* bias_ptr, float* y_buf, int64_t m, int64_t ldn) {
  if (bias_ptr) {
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 bias_vec = _mm512_loadu_ps(bias_ptr + j);
        _mm512_storeu_ps(y_buf + i * ldn + j, bias_vec);
      }
#endif
      for (; j < N; ++j) {
        y_buf[i * ldn + j] = bias_ptr[j];
      }
    }
  } else {  // initialize to zero
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 zero_vec = _mm512_setzero_ps();
        _mm512_storeu_ps(y_buf + i * ldn + j, zero_vec);
      }
#endif
      for (; j < N; ++j) {
        y_buf[i * ldn + j] = 0;
      }
    }
  }
}

template <typename scalar_t>
void fused_experts_int4_w4a8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    int32_t* __restrict__ Azp_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const uint8_t* __restrict__ packed_w1,
    const uint8_t* __restrict__ packed_w2,
    const int8_t* __restrict__ w1z,
    const int8_t* __restrict__ w2z,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int group_size,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  int num_threads = at::get_num_threads();
  // int64_t buffer_size_nbytes = M * topk * N * 2
  //                              M * topk * K * 2 +
  //                              num_threads * BLOCK_M * K +
  //                              num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float)  +
  //                              M * topk * 2 * N * 2 +
  //                              max(M * K, M * topk * N)  +
  //                              M * topk * sizeof(float);

  // intermediate_cache1 (scalar_t):     START + M * topk * N
  // intermediate_cache2 (scalar_t):     + M * topk * K
  // A_tmp (uint8_t):                    + num_threads * BLOCK_M * K
  // C_tmp (float):                      + num_threads * 2 * BLOCK_M * BLOCK_N
  // intermediate_cache0 (scalar_t):     + M * topk * 2 * N
  // Aq_tmp (uint8_t):                   + max(M * K, M * topk * N)
  // As_tmp (float):                     + M * topk

  // stage 0: quantize input to uint8, [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * K, As_tmp[m], input + m * K, K);
    }
  });

  auto Azp = at::ones({M * topk}).to(at::kInt).mul(128);
  auto Azp_ptr = Azp.data_ptr<int32_t>();
  // stage 1: intermediate_cache0 = hidden_states @ w1
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  int64_t block_per_group = group_size / BLOCK_K;
  int64_t Kc = K / BLOCK_K;
  int64_t num_groups = K / group_size;

  const int64_t stride_e = 2 * NB * Kc * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t)));
  const bool sym_quant_a = false;
  // weight + compensation shape = [E, Nc, Kc, block_n * BLOCK_K / 2 + block_n*sizeof(int32_t)]
  // scales/qzeros shape = [E, Nc, G, block_n]

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    // get local pointers
    int tid = at::get_thread_num();
    alignas(64) float As[BLOCK_M];
    uint8_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    float* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;
    bool is_brgemm_used = false;
    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB;
      int64_t nb = i % NB;
      int64_t nb1 = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);
      // B shape [K, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const uint8_t* __restrict__ B = packed_w1 + expert_id * stride_e;
      // Bz and Bs: [E, K/gs, 2N]
      const int8_t* __restrict__ Bz = w1z + expert_id * (num_groups) * (2 * N);
      const float* __restrict__ Bs = w1s + expert_id * (num_groups) * (2 * N);

      // 1.a load A
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      const bool use_brgemm = can_use_brgemm<int8_t>(m_size);
      is_brgemm_used = is_brgemm_used || use_brgemm;
      // copy to A [BLOCK_M, K]
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, Aq_tmp + index * K, K);
        As[m] = As_tmp[index];
      }
      const int64_t offset = offsets[mb];
      copy_bias<BLOCK_N>(nullptr, C0, m_size, BLOCK_N);
      copy_bias<BLOCK_N>(nullptr, C1, m_size, BLOCK_N);
      for (int kci = 0; kci < Kc; ++kci) {
        int32_t* compensation_ptr =
            sym_quant_a ? nullptr
                        : (int32_t*)(void*)(B + (nb * Kc + kci) * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t))) +
                                            BLOCK_K * BLOCK_N / 2) /*Bcomp*/;
        tinygemm_kernel<scalar_t>(
            ic0 + offset * 2 * N + nb * BLOCK_N,
            C0,
            A + kci * BLOCK_K,
            As,
            Azp_ptr,
            B + (nb * Kc + kci) * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t))) /*B*/,
            Bs + nb * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N /*scales_b*/,
            Bz + nb * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N /*qzeros_b*/,
            compensation_ptr,
            m_size,
            BLOCK_K,
            K,
            BLOCK_N,
            2 * N,
            kci == Kc - 1,
            use_brgemm);
      }

      for (int kci = 0; kci < Kc; ++kci) {
        int32_t* compensation_ptr =
            sym_quant_a ? nullptr
                        : (int32_t*)(void*)(B + (nb1 * Kc + kci) * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t))) +
                                            BLOCK_K * BLOCK_N / 2) /*Bcomp*/;
        tinygemm_kernel<scalar_t>(
            ic0 + offset * 2 * N + nb1 * BLOCK_N,
            C1,
            A + kci * BLOCK_K,
            As,
            Azp_ptr,
            B + (nb1 * Kc + kci) * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t))) /*B*/,
            Bs + nb1 * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N /*scales_b*/,
            Bz + nb1 * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N /*qzeros_b*/,
            compensation_ptr,
            m_size,
            BLOCK_K,
            K,
            BLOCK_N,
            2 * N,
            kci == Kc - 1,
            use_brgemm);
      }
    }

    if (is_brgemm_used) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 1.5: intermediate_cache1 = silu(intermediate_cache0)
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
    }
  });

  // stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * N, As_tmp[m], ic1 + m * N, N);
    }
  });
  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_oc = IC;
  num_groups = IC / group_size;
  Kc = IC / BLOCK_K;
  const int64_t stride_e2 = NB2 * Kc * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t)));
  // parallel on [MB2, NB2]
  at::parallel_for(0, MB2 * NB2, 0, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    float* __restrict__ C2 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    bool is_brgemm_used = false;
    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB2;
      int64_t nb = i % NB2;

      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);
      const bool use_brgemm = can_use_brgemm<int8_t>(m_size);
      is_brgemm_used = is_brgemm_used || use_brgemm;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const uint8_t* __restrict__ B = packed_w2 + expert_id * stride_e2;

      // Bz and Bs: [E, IC/gs, OC]
      const int8_t* __restrict__ Bz = w2z + expert_id * (num_groups)*OC;
      const float* __restrict__ Bs = w2s + expert_id * (num_groups)*OC;

      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const uint8_t* __restrict__ A = Aq_tmp + offsets[mb] * IC;
      const float* __restrict__ As = As_tmp + offsets[mb];
      copy_bias<BLOCK_N>(nullptr, C2, m_size, BLOCK_N);
      for (int kci = 0; kci < Kc; ++kci) {
        int32_t* compensation_ptr =
            sym_quant_a ? nullptr
                        : (int32_t*)(void*)(B + (nb * Kc + kci) * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t))) +
                                            BLOCK_K * BLOCK_N / 2) /*Bcomp*/;
        tinygemm_kernel<scalar_t>(
            nullptr,
            C2,
            A + kci * BLOCK_K,
            As,
            Azp_ptr,
            B + (nb * Kc + kci) * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t))),
            Bs + nb * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N /*scales_b*/,
            Bz + nb * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N /*zeros_b*/,
            compensation_ptr,
            m_size,
            BLOCK_K,
            IC,
            BLOCK_N,
            BLOCK_N,
            false,
            use_brgemm);
      }

      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C2 + m * BLOCK_N, weight, n_size);
      }
    }

    if (is_brgemm_used) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 3: out = intermediate_cache2.sum(dim=1)
  //   from [M, topk, K] to [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

#define INSTANTIATE_MOE_INT4_W4A8_TEMPLATE(TYPE)           \
  template void fused_experts_int4_w4a8_kernel_impl<TYPE>( \
      TYPE* __restrict__ output,                           \
      TYPE* __restrict__ ic0,                              \
      TYPE* __restrict__ ic1,                              \
      TYPE* __restrict__ ic2,                              \
      uint8_t* __restrict__ A_tmp,                         \
      uint8_t* __restrict__ Aq_tmp,                        \
      float* __restrict__ As_tmp,                          \
      int32_t* __restrict__ Azp_tmp,                       \
      float* __restrict__ C_tmp,                           \
      const TYPE* __restrict__ input,                      \
      const uint8_t* __restrict__ packed_w1,               \
      const uint8_t* __restrict__ packed_w2,               \
      const int8_t* __restrict__ w1z,                      \
      const int8_t* __restrict__ w2z,                      \
      const float* __restrict__ w1s,                       \
      const float* __restrict__ w2s,                       \
      int group_size,                                      \
      const float* __restrict__ topk_weights,              \
      const int32_t* __restrict__ sorted_ids,              \
      const int32_t* __restrict__ expert_ids,              \
      const int32_t* __restrict__ offsets,                 \
      int64_t M,                                           \
      int64_t N,                                           \
      int64_t K,                                           \
      int64_t E,                                           \
      int64_t topk,                                        \
      int64_t num_tokens_post_pad)

INSTANTIATE_MOE_INT4_W4A8_TEMPLATE(at::BFloat16);
INSTANTIATE_MOE_INT4_W4A8_TEMPLATE(at::Half);
