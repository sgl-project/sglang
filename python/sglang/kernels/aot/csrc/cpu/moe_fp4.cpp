#include "common.h"
#include "gemm.h"
#include "moe.h"

template <typename scalar_t>
void fused_experts_fp4_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const uint8_t* __restrict__ packed_w1,
    const uint8_t* __restrict__ packed_w2,
    const float* __restrict__ w1_bias,
    const float* __restrict__ w2_bias,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w2s,
    int64_t block_size_K,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad,
    float alpha,
    float limit,
    CPUActMethod act_func,
    bool with_bias) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 1: intermediate_cache0 = hidden_states @ w1
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(2 * N, BLOCK_N);
  int64_t scale_size_N = 2 * N;
  int64_t scale_size_K = div_up(K, block_size_K);

  const int64_t packed_K = get_row_size<uint8_t>(K);
  const int64_t stride_e = 2 * N * packed_K;
  const int64_t stride_n = packed_K;

  int64_t avg_M = std::max(int64_t(1), M * topk / E);
  const bool use_brgemm = can_use_brgemm<uint8_t>(avg_M);

  int64_t B_tmp_size_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * std::max(K, N);

  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;

    loop_2d<uint8_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t n_size = std::min(2 * N - nb * BLOCK_N, BLOCK_N);

      // B shape [K, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const uint8_t* __restrict__ B = packed_w1 + expert_id * stride_e + nb * BLOCK_N * stride_n;
      const uint8_t* __restrict__ Bs = w1s + expert_id * scale_size_N * scale_size_K + nb * BLOCK_N * scale_size_K;
      const float* __restrict__ B_bias = with_bias ? w1_bias + expert_id * 2 * N + nb * BLOCK_N : nullptr;

      // do unpacking for the first row or a new expert
      int32_t pre_expert_id = mb == 0 ? -1 : expert_ids[mb - 1];
      bool do_unpack = (mb == mb0) || (expert_id != pre_expert_id);

      int64_t m_size = offsets[mb + 1] - offsets[mb];

      if (nb_offset == 0) {
        // 1.a load A
        const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
        for (int64_t m = 0; m < m_size; ++m) {
          int32_t index = A_ids[m] / topk;
          copy_stub(A + m * K, input + index * K, K);
        }
      }

      const int64_t offset = offsets[mb];
      tinygemm_kernel<scalar_t>(
          /*   A            */ A,
          /*   B            */ B,
          /*   C            */ ic0 + offset * 2 * N + nb * BLOCK_N,
          /*   Btmp         */ B_tmp + tid * B_tmp_size_per_thread + nb_offset * BLOCK_N * K,
          /*   Ctmp         */ C_tmp + tid * 2 * BLOCK_M * BLOCK_N,
          /*   Bbias        */ B_bias,
          /*   scale        */ Bs,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ K,
          /*   lda          */ K,
          /*   ldb          */ n_size,
          /*   ldc          */ 2 * N,
          /*   brg          */ use_brgemm,
          /*   block_size_K */ block_size_K,
          /*   do_unpack    */ do_unpack);
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 1.5: intermediate_cache1 = activation(intermediate_cache0)
  if (act_func == CPUActMethod::silu_and_mul) {
    at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
      }
    });
  } else if (act_func == CPUActMethod::gelu_and_mul) {
    at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        gelu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
      }
    });
  } else if (act_func == CPUActMethod::swiglu) {
    at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        clamp_sigmoid_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, N / 2, alpha, limit);
        clamp_sigmoid_and_mul_stub(ic1 + m * N + N / 2, ic0 + m * 2 * N + N, N / 2, alpha, limit);
      }
    });
  }

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  scale_size_N = K;
  scale_size_K = div_up(N, block_size_K);
  const int64_t packed_IC = get_row_size<uint8_t>(IC);
  const int64_t stride_e2 = OC * packed_IC;
  const int64_t stride_oc = packed_IC;

  // parallel on [MB2, NB2]
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    alignas(64) scalar_t C[BLOCK_M * BLOCK_K];

    loop_2d<uint8_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const scalar_t* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const uint8_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
      const uint8_t* __restrict__ Bs = w2s + expert_id * scale_size_N * scale_size_K + nb * BLOCK_N * scale_size_K;
      const float* __restrict__ B_bias = with_bias ? w2_bias + expert_id * OC + nb * BLOCK_N : nullptr;

      // do unpacking for the first row or a new expert
      int32_t pre_expert_id = mb == 0 ? -1 : expert_ids[mb - 1];
      bool do_unpack = (mb == mb0) || (expert_id != pre_expert_id);

      tinygemm_kernel<scalar_t>(
          /*   A            */ A,
          /*   B            */ B,
          /*   C            */ C,
          /*   Btmp         */ B_tmp + tid * B_tmp_size_per_thread + nb_offset * BLOCK_N * IC,
          /*   Ctmp         */ C_tmp + tid * 2 * BLOCK_M * BLOCK_N,
          /*   Bbias        */ B_bias,
          /*   scale        */ Bs,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ IC,
          /*   lda          */ IC,
          /*   ldb          */ n_size,
          /*   ldc          */ BLOCK_N,
          /*   brg          */ use_brgemm,
          /*   block_size_K */ block_size_K,
          /*   do_unpack    */ do_unpack);

      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
      }
    });

    if (use_brgemm) {
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

#define INSTANTIATE_MOE_FP4_TEMPLATE(TYPE)           \
  template void fused_experts_fp4_kernel_impl<TYPE>( \
      TYPE* __restrict__ output,                     \
      TYPE* __restrict__ ic0,                        \
      TYPE* __restrict__ ic1,                        \
      TYPE* __restrict__ ic2,                        \
      TYPE* __restrict__ A_tmp,                      \
      TYPE* __restrict__ B_tmp,                      \
      float* __restrict__ C_tmp,                     \
      const TYPE* __restrict__ input,                \
      const uint8_t* __restrict__ packed_w1,         \
      const uint8_t* __restrict__ packed_w2,         \
      const float* __restrict__ w1_bias,             \
      const float* __restrict__ w2_bias,             \
      const uint8_t* __restrict__ w1s,               \
      const uint8_t* __restrict__ w2s,               \
      int64_t block_size_K,                          \
      const float* __restrict__ topk_weights,        \
      const int32_t* __restrict__ sorted_ids,        \
      const int32_t* __restrict__ expert_ids,        \
      const int32_t* __restrict__ offsets,           \
      int64_t M,                                     \
      int64_t N,                                     \
      int64_t K,                                     \
      int64_t E,                                     \
      int64_t topk,                                  \
      int64_t num_tokens_post_pad,                   \
      float alpha,                                   \
      float limit,                                   \
      CPUActMethod act_func,                         \
      bool with_bias)

INSTANTIATE_MOE_FP4_TEMPLATE(at::BFloat16);
INSTANTIATE_MOE_FP4_TEMPLATE(at::Half);
