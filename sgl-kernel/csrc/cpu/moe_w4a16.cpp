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
inline void copy_mul_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    bVec x = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    x0 = x0 * weight_vec;
    x1 = x1 * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
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

template <typename scalar_t>
void fused_experts_int4_w4a16_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const at::quint4x2* __restrict__ packed_w1,
    const at::quint4x2* __restrict__ packed_w2,
    const uint8_t* __restrict__ w1z,
    const uint8_t* __restrict__ w2z,
    const scalar_t* __restrict__ w1s,
    const scalar_t* __restrict__ w2s,
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

  // stage 1: intermediate_cache0 = hidden_states @ w1
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(2 * N, BLOCK_N);

  const int64_t stride_e = 2 * N * K;
  const int64_t stride_n = K;

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    // get local pointers
    int tid = at::get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;

    bool is_brgemm_used = false;

    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB;
      int64_t nb = i % NB;

      int64_t n_size = std::min(2 * N - nb * BLOCK_N, BLOCK_N);

      // B shape [K, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const at::quint4x2* __restrict__ B = packed_w1 + (expert_id * stride_e + nb * BLOCK_N * stride_n) / 2;
      // Bz and Bs: [E, K/gs, 2N]
      const uint8_t* __restrict__ Bz = w1z + expert_id * (K / group_size) * (2 * N) + nb * BLOCK_N;
      const scalar_t* __restrict__ Bs = w1s + expert_id * (K / group_size) * (2 * N) + nb * BLOCK_N;

      // 1.a load A
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      const bool use_brgemm = can_use_brgemm<at::quint4x2>(m_size);
      is_brgemm_used = is_brgemm_used || use_brgemm;

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, input + index * K, K);
      }

      const int64_t offset = offsets[mb];
      tinygemm_kernel<scalar_t>(
          /*   A            */ A,
          /*   B            */ B,
          /*   C            */ ic0 + offset * 2 * N + nb * BLOCK_N,
          /*   Bz           */ Bz,
          /*   Bs           */ Bs,
          /*   Btmp         */ B_tmp + tid * BLOCK_N * std::max(K, N),
          /*   Ctmp         */ C_tmp + tid * 2 * BLOCK_M * BLOCK_N,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ K,
          /*   group_size   */ group_size,
          /*   lda          */ K,
          /*   ldb          */ n_size,
          /*   ldc          */ 2 * N,
          /*   strideBz     */ 2 * N,
          /*   strideBs     */ 2 * N,
          /*   brg          */ use_brgemm);
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

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * IC;
  const int64_t stride_oc = IC;

  // parallel on [MB2, NB2]
  at::parallel_for(0, MB2 * NB2, 0, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    alignas(64) scalar_t C[BLOCK_M * BLOCK_K];

    bool is_brgemm_used = false;

    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB2;
      int64_t nb = i % NB2;

      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      const bool use_brgemm = can_use_brgemm<at::quint4x2>(m_size);
      is_brgemm_used = is_brgemm_used || use_brgemm;

      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const scalar_t* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const at::quint4x2* __restrict__ B = packed_w2 + (expert_id * stride_e2 + nb * BLOCK_N * stride_oc) / 2;
      // Bz and Bs: [E, IC/gs, OC]
      const uint8_t* __restrict__ Bz = w2z + expert_id * (IC / group_size) * OC + nb * BLOCK_N;
      const scalar_t* __restrict__ Bs = w2s + expert_id * (IC / group_size) * OC + nb * BLOCK_N;

      tinygemm_kernel<scalar_t>(
          /*   A            */ A,
          /*   B            */ B,
          /*   C            */ C,
          /*   Bz           */ Bz,
          /*   Bs           */ Bs,
          /*   Btmp         */ B_tmp + tid * BLOCK_N * std::max(K, N),
          /*   Ctmp         */ C_tmp + tid * 2 * BLOCK_M * BLOCK_N,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ IC,
          /*   group_size   */ group_size,
          /*   lda          */ IC,
          /*   ldb          */ n_size,
          /*   ldc          */ BLOCK_N,
          /*   strideBz     */ OC,
          /*   strideBs     */ OC,
          /*   brg          */ use_brgemm);

      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
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

#define INSTANTIATE_MOE_INT4_W4A16_TEMPLATE(TYPE)           \
  template void fused_experts_int4_w4a16_kernel_impl<TYPE>( \
      TYPE* __restrict__ output,                            \
      TYPE* __restrict__ ic0,                               \
      TYPE* __restrict__ ic1,                               \
      TYPE* __restrict__ ic2,                               \
      TYPE* __restrict__ A_tmp,                             \
      TYPE* __restrict__ B_tmp,                             \
      float* __restrict__ C_tmp,                            \
      const TYPE* __restrict__ input,                       \
      const at::quint4x2* __restrict__ packed_w1,           \
      const at::quint4x2* __restrict__ packed_w2,           \
      const uint8_t* __restrict__ w1z,                      \
      const uint8_t* __restrict__ w2z,                      \
      const TYPE* __restrict__ w1s,                         \
      const TYPE* __restrict__ w2s,                         \
      int group_size,                                       \
      const float* __restrict__ topk_weights,               \
      const int32_t* __restrict__ sorted_ids,               \
      const int32_t* __restrict__ expert_ids,               \
      const int32_t* __restrict__ offsets,                  \
      int64_t M,                                            \
      int64_t N,                                            \
      int64_t K,                                            \
      int64_t E,                                            \
      int64_t topk,                                         \
      int64_t num_tokens_post_pad)

INSTANTIATE_MOE_INT4_W4A16_TEMPLATE(at::BFloat16);
INSTANTIATE_MOE_INT4_W4A16_TEMPLATE(at::Half);
