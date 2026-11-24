#include "moe.h"

#include "common.h"
#include "gemm.h"

namespace {

// [NOTE]: Fused MoE kernel with AMX
//
//   This file contains implementations for
//     * `moe_align_block_size`
//     * `fused_moe`
//
//   The functionality is identical to triton kernel, excepts:
//     * fuse silu_and_mul with gemm1, therefore this kernel
//       allocates 2 intermediate_caches instead of 3
//     * add `offsets` in `moe_align_block_size` which keeps track
//       of starting offset for each M block. this is for keeping
//       output of silu_and_mul in sorted order, thus load_A for
//       the 2nd gemm would be contiguous, therefore we can directly
//       load A from intermediate_cache1.
//
//  TODO:
//     1. tune BLOCK_M and BLOCK_N (BLOCK_N * K fit L2)
//     2. add prefetch for load A which is indexed access
//     3. abstract at::native::cpublas::brgemm with WoQ gemm (M = 1 & M != 1)
//

template <int BLOCK_M>
int moe_align_block_size(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ total_cnts,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int num_experts,
    int numel,
    int num_threads) {
#define T_INDEX(tt) total_cnts + (tt) * num_experts

  // accumulate count of expert ids locally
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_cnts = T_INDEX(tid + 1);

    for (int i = begin; i < end; ++i) {
      local_cnts[topk_ids[i]]++;
    }
  });

  using iVec = at::vec::Vectorized<int32_t>;
  for (int t = 0; t < num_threads; ++t) {
    at::vec::map2<int32_t>(
        [](iVec x, iVec y) { return x + y; }, T_INDEX(t + 1), T_INDEX(t + 1), T_INDEX(t), num_experts);
  }

  // the last row holds sums of each experts
  int32_t* total_cnts_t_1 = T_INDEX(num_threads);

  cumsums[0] = 0;
  for (int e = 0; e < num_experts; ++e) {
    // accumulate `num_tokens_post_pad`, also as the expert offset
    cumsums[e + 1] = cumsums[e] + div_up(total_cnts_t_1[e], BLOCK_M) * BLOCK_M;

    for (int k = cumsums[e]; k < cumsums[e + 1]; k += BLOCK_M) {
      expert_ids[k / BLOCK_M] = e;
    }
  }
  int num_tokens_post_pad = cumsums[num_experts];

  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    // thread tid offsets in `total_cnts`
    int32_t* __restrict__ offsets = T_INDEX(tid);

    for (int i = begin; i < end; ++i) {
      int32_t expert_id = topk_ids[i];
      int32_t b_offset = cumsums[expert_id];
      int32_t t_offset = offsets[expert_id];
      sorted_ids[b_offset + t_offset] = i;
      offsets[expert_id]++;
    }
  });

  // debug: the offset for thread t_1 should be identical to t_2
  int32_t* total_cnts_t_2 = T_INDEX(num_threads - 1);
  for (int e = 0; e < num_experts; ++e) {
    TORCH_CHECK(total_cnts_t_1[e] == total_cnts_t_2[e]);
  }

  // padding value for sorted_ids: numel
  auto sorted_id_size = [=](const int32_t* sorted_ids_ptr) {
    for (int d = 0; d < BLOCK_M; ++d) {
      if (sorted_ids_ptr[d] == numel) {
        return d;
      }
    }
    return BLOCK_M;
  };

  // offsets holds starting offset for each valida M blocks
  //   shape : [num_token_blocks + 1]
  offsets[0] = 0;
  const int num_token_blocks = num_tokens_post_pad / BLOCK_M;
  at::parallel_for(0, num_token_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
    for (int mb = begin; mb < end; ++mb) {
      offsets[mb + 1] = sorted_id_size(sorted_ids + mb * BLOCK_M);
    }
  });
  // TODO: do we need to vecterize this ?
  for (int mb = 0; mb < num_token_blocks; ++mb) {
    offsets[mb + 1] += offsets[mb];
  }
  // debug: the last value of offsets should be `numel`
  TORCH_CHECK(offsets[num_token_blocks] == numel);

  return num_tokens_post_pad;
}

template <int BLOCK_M>
int inkling2_moe_align_block_size(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ total_cnts,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int num_experts,
    int numel,
    int num_threads) {
#define I2_T_INDEX(tt) (total_cnts + (tt) * num_experts)

  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_cnts = I2_T_INDEX(tid + 1);
    for (int i = begin; i < end; ++i) {
      int32_t eid = topk_ids[i];
      if (eid >= 0 && eid < num_experts) {
        local_cnts[eid]++;
      }
    }
  });

  using iVec = at::vec::Vectorized<int32_t>;
  for (int t = 0; t < num_threads; ++t) {
    at::vec::map2<int32_t>(
        [](iVec x, iVec y) { return x + y; }, I2_T_INDEX(t + 1), I2_T_INDEX(t + 1), I2_T_INDEX(t), num_experts);
  }

  int32_t* total_cnts_t_1 = I2_T_INDEX(num_threads);
  cumsums[0] = 0;
  for (int e = 0; e < num_experts; ++e) {
    cumsums[e + 1] = cumsums[e] + div_up(total_cnts_t_1[e], BLOCK_M) * BLOCK_M;
    for (int k = cumsums[e]; k < cumsums[e + 1]; k += BLOCK_M) {
      expert_ids[k / BLOCK_M] = e;
    }
  }
  int num_tokens_post_pad = cumsums[num_experts];
  if (num_tokens_post_pad == 0) {
    offsets[0] = 0;
    return 0;
  }

  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ tid_offsets = I2_T_INDEX(tid);
    for (int i = begin; i < end; ++i) {
      int32_t expert_id = topk_ids[i];
      if (expert_id >= 0 && expert_id < num_experts) {
        int32_t b_offset = cumsums[expert_id];
        int32_t t_offset = tid_offsets[expert_id];
        sorted_ids[b_offset + t_offset] = i;
        tid_offsets[expert_id]++;
      }
    }
  });

  auto sorted_id_size = [=](const int32_t* sorted_ids_ptr) {
    for (int d = 0; d < BLOCK_M; ++d) {
      if (sorted_ids_ptr[d] == numel) {
        return d;
      }
    }
    return BLOCK_M;
  };

  offsets[0] = 0;
  const int num_token_blocks = num_tokens_post_pad / BLOCK_M;
  at::parallel_for(0, num_token_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
    for (int mb = begin; mb < end; ++mb) {
      offsets[mb + 1] = sorted_id_size(sorted_ids + mb * BLOCK_M);
    }
  });
  for (int mb = 0; mb < num_token_blocks; ++mb) {
    offsets[mb + 1] += offsets[mb];
  }
  TORCH_CHECK(offsets[num_token_blocks] <= numel);

#undef I2_T_INDEX
  return num_tokens_post_pad;
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2 {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B0,
      const scalar_t* __restrict__ B1,
      scalar_t* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B0,
      const at::BFloat16* __restrict__ B1,
      at::BFloat16* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    static_assert(COLS >= 1);

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb0[COLS];
    __m512bh vb1[COLS];
    __m512 vc0[ROWS * COLS];
    __m512 vc1[ROWS * COLS];

    auto loadc = [&](auto i) {
      vc0[i] = _mm512_set1_ps(0.f);
      vc1[i] = _mm512_set1_ps(0.f);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b0_ptr = reinterpret_cast<const float*>(B0);
    const float* b1_ptr = reinterpret_cast<const float*>(B1);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb0[col] = (__m512bh)(_mm512_loadu_si512(b0_ptr + k * ldb2 + col * 16));
        vb1[col] = (__m512bh)(_mm512_loadu_si512(b1_ptr + k * ldb2 + col * 16));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b0_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          _mm_prefetch(b1_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc0[i] = _mm512_dpbf16_ps(vc0[i], va, vb0[col]);
      vc1[i] = _mm512_dpbf16_ps(vc1[i], va, vb1[col]);
    };
    for (int64_t k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        __m512 x0 = vc0[row * COLS + col + 0];
        __m512 x1 = vc0[row * COLS + col + 1];
        __m512 y0 = vc1[row * COLS + col + 0];
        __m512 y1 = vc1[row * COLS + col + 1];
        x0 = _mm512_mul_ps(_mm512_rcp14_silu_ps(x0), y0);
        x1 = _mm512_mul_ps(_mm512_rcp14_silu_ps(x1), y1);

        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(__m512(x1), __m512(x0))));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)       \
  tinygemm_kernel_nn2<scalar_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B0 + nb_start * 2, B1 + nb_start * 2, C + mb_start * ldc + nb_start, K, lda, ldb, ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B0,
    const scalar_t* __restrict__ B1,
    scalar_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  // pattern: 1-(2+2)-(8+8)
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 32;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        // mb_size = 2
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        // mb_size = 3
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        // mb_size = 4
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      float* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      float* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    static_assert(COLS >= 1);

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    auto loadc = [&](auto i) { vc[i] = _mm512_set1_ps(0.f); };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b_ptr = reinterpret_cast<const float*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb[col] = (__m512bh)(_mm512_loadu_si512(b_ptr + k * ldb2 + col * 16));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      _mm512_storeu_ps(reinterpret_cast<__m512*>(C + row * ldc + col * 16), vc[i]);
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN2(MB_SIZE, NB_SIZE)     \
  tinygemm_kernel_nn<scalar_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start, K, lda, ldb, ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  // pattern: 1-2-8
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 32;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 16);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 32);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NN2(2, 16);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN2(2, 32);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NN2(3, 16);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN2(3, 32);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NN2(4, 16);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN2(4, 32);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t>
void fused_experts_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ packed_w1,
    const scalar_t* __restrict__ packed_w2,
    const float* __restrict__ w1_bias,
    const float* __restrict__ w2_bias,
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
  // handle 2 tiles per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  // strides for w1: [E, 2N, K]
  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  const int64_t stride_e = 2 * N * K;
  const int64_t stride_n = K;

  int64_t avg_M = std::max(int64_t(1), M * topk / E);
  const bool use_brgemm = can_use_brgemm<scalar_t>(avg_M);

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    float* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K * 2, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      // nb_upper from top half and nb_lower from bottom half
      int64_t nb_upper = nb, nb_lower = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);

      // B shape [K, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const scalar_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb_upper * BLOCK_N * stride_n;
      const scalar_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + nb_lower * BLOCK_N * stride_n;
      const float* __restrict__ B0_bias = w1_bias + expert_id * 2 * N + nb_upper * BLOCK_N;
      const float* __restrict__ B1_bias = w1_bias + expert_id * 2 * N + nb_lower * BLOCK_N;

      int64_t m_size = offsets[mb + 1] - offsets[mb];

      if (nb_offset == 0) {
        // 1.a load A
        const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
        for (int64_t m = 0; m < m_size; ++m) {
          int32_t index = A_ids[m] / topk;
          copy_stub(A + m * K, input + index * K, K);
        }
      }

      if (use_brgemm) {
        // 1.b gemm: C0 = A @ B0
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B0,
            /* C     */ C0);

        // 1.c gemm: C1 = A @ B1
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B1,
            /* C     */ C1);

      } else {
        const int64_t offset = offsets[mb];
        // the fused path below hardcodes SiLU in its store step
        if (act_func != CPUActMethod::silu_and_mul) {
          tinygemm_kernel(
              /* A     */ A,
              /* B     */ B0,
              /* C     */ C0,
              /* M     */ m_size,
              /* N     */ n_size,
              /* K     */ K,
              /* lda   */ K,
              /* ldb   */ n_size,
              /* ldc   */ BLOCK_N);
          tinygemm_kernel(
              /* A     */ A,
              /* B     */ B1,
              /* C     */ C1,
              /* M     */ m_size,
              /* N     */ n_size,
              /* K     */ K,
              /* lda   */ K,
              /* ldb   */ n_size,
              /* ldc   */ BLOCK_N);
        } else {
          // fused 1.bcd: silu_and_mul(A @ B0, A @ B1)
          tinygemm_kernel(
              /* A     */ A,
              /* B0    */ B0,
              /* B1    */ B1,
              /* C     */ ic1 + offset * N + nb * BLOCK_N,
              /* M     */ m_size,
              /* N     */ n_size,
              /* K     */ K,
              /* lda   */ K,
              /* ldb   */ n_size,
              /* ldc   */ N);
        }
      }
      if (with_bias) {
        for (int64_t m = 0; m < m_size; ++m) {
          add_bias_stub(C0 + m * BLOCK_N, B0_bias, n_size);
          add_bias_stub(C1 + m * BLOCK_N, B1_bias, n_size);
        }
      }
      // 1.d activation and mul
      const int64_t offset = offsets[mb];
      if (act_func == CPUActMethod::silu_and_mul && use_brgemm) {
        for (int64_t m = 0; m < m_size; ++m) {
          silu_and_mul_stub(ic1 + (offset + m) * N + nb * BLOCK_N, C0 + m * BLOCK_N, C1 + m * BLOCK_N, BLOCK_N);
        }
      } else if (act_func == CPUActMethod::gelu_and_mul) {
        for (int64_t m = 0; m < m_size; ++m) {
          gelu_and_mul_stub(ic1 + (offset + m) * N + nb * BLOCK_N, C0 + m * BLOCK_N, C1 + m * BLOCK_N, BLOCK_N);
        }
      } else if (act_func == CPUActMethod::swiglu) {
        for (int64_t m = 0; m < m_size; ++m) {
          scalar_t* __restrict__ ic1_row = ic1 + (offset + m) * N;
          clamp_sigmoid_and_mul_stub(ic1_row + nb * BLOCK_N / 2, C0 + m * BLOCK_N, BLOCK_N / 2, alpha, limit);
          clamp_sigmoid_and_mul_stub(ic1_row + N / 2 + nb * BLOCK_N / 2, C1 + m * BLOCK_N, BLOCK_N / 2, alpha, limit);
        }
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
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
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    // we won't be using C1 for gemm2
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const scalar_t* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const scalar_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
      const float* __restrict__ B_bias = w2_bias + expert_id * OC + nb * BLOCK_N;

      // 2.a gemm: C = A @ B
      if (use_brgemm) {
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ IC,
            /* lda   */ IC,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B,
            /* C     */ C);
      } else {
        tinygemm_kernel(
            /* A     */ A,
            /* B     */ B,
            /* C     */ C,
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ IC,
            /* lda   */ IC,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N);
      }

      if (with_bias) {
        for (int64_t m = 0; m < m_size; ++m) {
          add_bias_stub(C + m * BLOCK_N, B_bias, n_size);
        }
      }
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

template <typename scalar_t>
void shared_expert_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    scalar_t* __restrict__ input,
    const scalar_t* __restrict__ packed_w1,
    const scalar_t* __restrict__ packed_w2,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K) {
  // handle 2 tiles per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);
  const int64_t stride_n = K;

  const bool use_brgemm = can_use_brgemm<scalar_t>(M);

  const bool apply_scaling_factor = fused_experts_out != nullptr;

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    float* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K * 2, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      // nb_upper from top half and nb_lower from bottom half
      int64_t nb_upper = nb, nb_lower = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);

      // A shape [m_size, K]
      const scalar_t* A = input + mb * BLOCK_M * K;

      // B shape [K, n_size] in vnni format
      const scalar_t* __restrict__ B0 = packed_w1 + nb_upper * BLOCK_N * stride_n;
      const scalar_t* __restrict__ B1 = packed_w1 + nb_lower * BLOCK_N * stride_n;

      if (use_brgemm) {
        // 1.b gemm: C0 = A @ B0
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B0,
            /* C     */ C0);

        // 1.c gemm: C1 = A @ B1
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B1,
            /* C     */ C1);

        // 1.d silu and mul
        for (int64_t m = 0; m < m_size; ++m) {
          silu_and_mul_stub(ic1 + (mb * BLOCK_M + m) * N + nb * BLOCK_N, C0 + m * BLOCK_N, C1 + m * BLOCK_N, BLOCK_N);
        }
      } else {
        // fused 1.bcd: silu_and_mul(A @ B0, A @ B1)
        tinygemm_kernel(
            /* A     */ A,
            /* B0    */ B0,
            /* B1    */ B1,
            /* C     */ ic1 + mb * BLOCK_M * N + nb * BLOCK_N,
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ N);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 2: output = intermediate_cache1 @ w2
  //   w2 : [K, N] as [OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_oc = IC;

  // parallel on [MB2, NB2]
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    // we won't be using C1 for gemm2
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A shape [m_size, IC]
      const scalar_t* __restrict__ A = ic1 + mb * BLOCK_M * N;

      // B shape [IC, n_size] in vnni format
      const scalar_t* __restrict__ B = packed_w2 + nb * BLOCK_N * stride_oc;

      // 2.a gemm: C = A @ B
      if (use_brgemm) {
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ IC,
            /* lda   */ IC,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ A,
            /* B     */ B,
            /* C     */ C);
      } else {
        tinygemm_kernel(
            /* A     */ A,
            /* B     */ B,
            /* C     */ C,
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ IC,
            /* lda   */ IC,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N);
      }

      // 2.b copy from C to output and add fused_experts_out
      scalar_t* __restrict__ out = output + mb * BLOCK_M * K + nb * BLOCK_N;
      const scalar_t* __restrict__ fused_out =
          apply_scaling_factor ? fused_experts_out + mb * BLOCK_M * K + nb * BLOCK_N : nullptr;
      for (int64_t m = 0; m < m_size; ++m) {
        const scalar_t* __restrict__ fused_out_row = apply_scaling_factor ? (fused_out + m * K) : nullptr;
        add_mul_stub(out + m * K, C + m * BLOCK_N, fused_out_row, routed_scaling_factor, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

template <typename scalar_t>
inline void inkling2_capped_swiglu_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ gate_up,
    int64_t n_size,
    float gamma,
    float gate_cap,
    float up_cap) {
  using fVec = at::vec::Vectorized<float>;
  const fVec gamma_v(gamma);
  const bool has_gate_cap = gate_cap > 0.f;
  const bool has_up_cap = up_cap > 0.f;
  const fVec gate_cap_v(gate_cap);
  const fVec inv_gate_cap_v(has_gate_cap ? (1.0f / gate_cap) : 0.0f);
  const fVec up_cap_v(up_cap);
  const fVec inv_up_cap_v(has_up_cap ? (1.0f / up_cap) : 0.0f);
  constexpr int64_t kVec2 = 2 * fVec::size();
  int64_t d = 0;
  for (; d <= n_size - kVec2; d += kVec2) {
    auto [x0_, y0_] = load_float_vec2(gate_up + d);
    auto [gate, up] = at::vec::deinterleave2<float>(x0_, y0_);
    fVec sig = fast_sigmoid(gate);
    if (has_gate_cap) {
      gate = gate_cap_v * (gate * inv_gate_cap_v).tanh();
    }
    if (has_up_cap) {
      up = up_cap_v * (up * inv_up_cap_v).tanh();
    }
    fVec res = gate * sig * up * gamma_v;
    store_from_float_ext(out + d / 2, res);
  }
  for (; d < n_size; d += 2) {
    float g = gate_up[d];
    float u = gate_up[d + 1];
    float sig = 1.0f / (1.0f + std::exp(-g));
    if (has_gate_cap) {
      g = gate_cap * std::tanh(g / gate_cap);
    }
    if (has_up_cap) {
      u = up_cap * std::tanh(u / up_cap);
    }
    out[d / 2] = static_cast<scalar_t>(g * sig * u * gamma);
  }
}

template <typename scalar_t>
void inkling2_fused_experts_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ packed_w13,
    const scalar_t* __restrict__ packed_w2,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ topk_ids,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad,
    float gate_softcap,
    float up_softcap) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB_w13 = div_up(2 * N, BLOCK_N);
  const int64_t stride_e = 2 * N * K;
  const int64_t stride_n = K;

  int64_t avg_M = std::max(int64_t(1), M * topk / std::max(int64_t(1), E));
  const bool use_brgemm = can_use_brgemm<scalar_t>(avg_M);

  // Stage 1: gate_up = hidden_states @ w13^T, then capped SwiGLU with per-row gamma
  parallel_2d(MB, NB_w13, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t n_size = std::min(2 * N - nb * BLOCK_N, BLOCK_N);
      int32_t expert_id = expert_ids[mb];
      const scalar_t* __restrict__ B = packed_w13 + expert_id * stride_e + nb * BLOCK_N * stride_n;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      if (nb_offset == 0) {
        const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
        for (int64_t m = 0; m < m_size; ++m) {
          int32_t index = A_ids[m] / topk;
          copy_stub(A + m * K, input + index * K, K);
        }
      }

      if (use_brgemm) {
        at::native::cpublas::brgemm(m_size, n_size, K, K, n_size, BLOCK_N, false, A, B, C0);
      } else {
        tinygemm_kernel(A, B, C0, m_size, n_size, K, K, n_size, BLOCK_N);
      }

      const int64_t offset = offsets[mb];
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      const int64_t out_col_offset = (nb * BLOCK_N) / 2;
      for (int64_t m = 0; m < m_size; ++m) {
        float gamma = topk_weights[A_ids[m]];
        inkling2_capped_swiglu_stub(
            ic1 + (offset + m) * N + out_col_offset, C0 + m * BLOCK_N, n_size, gamma, gate_softcap, up_softcap);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // Stage 2: down = ic1 @ w2^T (unit weight scatter to ic2)
  const int64_t OC = K;
  const int64_t IC = N;
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * IC;
  const int64_t stride_oc = IC;

  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      const scalar_t* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      int32_t expert_id = expert_ids[mb];
      const scalar_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;

      if (use_brgemm) {
        at::native::cpublas::brgemm(m_size, n_size, IC, IC, n_size, BLOCK_N, false, A, B, C);
      } else {
        tinygemm_kernel(A, B, C, m_size, n_size, IC, IC, n_size, BLOCK_N);
      }

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        copy_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // Stage 3: sum across valid local topk slots (skipping -1 remote EP ids)
  auto reduce_row = [&](int64_t m) {
    scalar_t* __restrict__ dst = output + m * K;
    const scalar_t* __restrict__ base = ic2 + m * topk * K;
    const int32_t* __restrict__ tids = topk_ids + m * topk;
    int64_t valid_idx[64];
    int n_valid = 0;
    for (int64_t k = 0; k < topk; ++k) {
      int32_t eid = tids[k];
      if (eid >= 0 && eid < E && n_valid < 64) {
        valid_idx[n_valid++] = k;
      }
    }
    if (n_valid == 0) {
      std::fill(dst, dst + K, static_cast<scalar_t>(0.0f));
    } else if (n_valid == 1) {
      const scalar_t* __restrict__ src = base + valid_idx[0] * K;
      std::copy(src, src + K, dst);
    } else {
      using bVec = at::vec::Vectorized<scalar_t>;
      using fVec = at::vec::Vectorized<float>;
      constexpr int kVecSize = bVec::size();
      int64_t d = 0;
      for (; d <= K - kVecSize; d += kVecSize) {
        fVec acc0 = fVec(0.f);
        fVec acc1 = fVec(0.f);
        for (int vi = 0; vi < n_valid; ++vi) {
          bVec bv = bVec::loadu(base + valid_idx[vi] * K + d);
          auto [v0, v1] = convert_to_float<scalar_t>(bv);
          acc0 += v0;
          acc1 += v1;
        }
        bVec out_v = convert_from_float_ext<scalar_t>(acc0, acc1);
        out_v.store(dst + d);
      }
      for (; d < K; ++d) {
        float acc = 0.f;
        for (int vi = 0; vi < n_valid; ++vi) {
          acc += static_cast<float>(base[valid_idx[vi] * K + d]);
        }
        dst[d] = static_cast<scalar_t>(acc);
      }
    }
  };
  if (M <= 4 && M * K <= 16384) {
    for (int64_t m = 0; m < M; ++m) {
      reduce_row(m);
    }
  } else {
    at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        reduce_row(m);
      }
    });
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <typename packed_t, int GROUP_K, int M_ROWS>
inline void inkling2_quant_gemm_small_m(
    const at::BFloat16* __restrict__ A,
    const packed_t* __restrict__ B,
    const float* __restrict__ Bs,
    float* __restrict__ C,
    int64_t IC) {
  static_assert(std::is_same_v<packed_t, uint8_t> || std::is_same_v<packed_t, at::Float8_e4m3fn>);
  static_assert(GROUP_K == 16 || GROUP_K == 32);
  constexpr int64_t BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);
  constexpr int kPrefetchK = 32;

  __m512 vc[M_ROWS][2];
  for (int r = 0; r < M_ROWS; ++r) {
    vc[r][0] = _mm512_setzero_ps();
    vc[r][1] = _mm512_setzero_ps();
  }

  const int64_t num_groups = IC / GROUP_K;
  const int64_t k2_per_group = GROUP_K >> 1;
  const int64_t lda2 = IC >> 1;
  const float* a_ptr = reinterpret_cast<const float*>(A);
  const __m512i zero = _mm512_setzero_si512();

  for (int64_t g = 0; g < num_groups; ++g) {
    __m512 vs0 = _mm512_loadu_ps(Bs + g * BLOCK_N);
    __m512 vs1 = _mm512_loadu_ps(Bs + g * BLOCK_N + 16);
    __m512 vsum[M_ROWS][2];
    for (int r = 0; r < M_ROWS; ++r) {
      vsum[r][0] = _mm512_setzero_ps();
      vsum[r][1] = _mm512_setzero_ps();
    }
    const int64_t k_base = g * k2_per_group;
    for (int64_t kp = 0; kp < k2_per_group; ++kp) {
      const int64_t k = k_base + kp;
      __m512bh vb0, vb1;
      if constexpr (std::is_same_v<packed_t, uint8_t>) {
        const uint8_t* b_row = B + k * BLOCK_N;
        if constexpr (kPrefetchK > 0) {
          _mm_prefetch(b_row + kPrefetchK * BLOCK_N, _MM_HINT_T0);
        }
        __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row));
        std::tie(vb0, vb1) = CVT_MXFP4_TO_BF16(b4, zero, zero);
      } else {
        const at::Float8_e4m3fn* b_row = B + k * (BLOCK_N * 2);
        if constexpr (kPrefetchK > 0) {
          _mm_prefetch(b_row + kPrefetchK * (BLOCK_N * 2), _MM_HINT_T0);
        }
        __m512i b8 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_row));
        vb0 = CVT_FP8_TO_BF16(_mm512_extracti32x8_epi32(b8, 0));
        vb1 = CVT_FP8_TO_BF16(_mm512_extracti32x8_epi32(b8, 1));
      }
      for (int r = 0; r < M_ROWS; ++r) {
        __m512bh va = (__m512bh)(_mm512_set1_ps(a_ptr[r * lda2 + k]));
        vsum[r][0] = _mm512_dpbf16_ps(vsum[r][0], va, vb0);
        vsum[r][1] = _mm512_dpbf16_ps(vsum[r][1], va, vb1);
      }
    }
    for (int r = 0; r < M_ROWS; ++r) {
      vc[r][0] = _mm512_fmadd_ps(vsum[r][0], vs0, vc[r][0]);
      vc[r][1] = _mm512_fmadd_ps(vsum[r][1], vs1, vc[r][1]);
    }
  }
  for (int r = 0; r < M_ROWS; ++r) {
    _mm512_storeu_ps(C + r * BLOCK_N, vc[r][0]);
    _mm512_storeu_ps(C + r * BLOCK_N + 16, vc[r][1]);
  }
}

template <typename packed_t, int GROUP_K>
inline void inkling2_unpack_quant_block_to_bf16(
    at::BFloat16* __restrict__ Btmp, const packed_t* __restrict__ B, const float* __restrict__ Bs, int64_t IC) {
  constexpr int64_t BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);
  const int64_t num_groups = IC / GROUP_K;
  const int64_t k2_per_group = GROUP_K >> 1;
  const __m512i zero = _mm512_setzero_si512();
  const __m512i idx_lo = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i idx_hi = _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);

  for (int64_t g = 0; g < num_groups; ++g) {
    __m512 vs0 = _mm512_loadu_ps(Bs + g * BLOCK_N);
    __m512 vs1 = _mm512_loadu_ps(Bs + g * BLOCK_N + 16);
    __m512 s0_lo = _mm512_permutexvar_ps(idx_lo, vs0);
    __m512 s0_hi = _mm512_permutexvar_ps(idx_hi, vs0);
    __m512 s1_lo = _mm512_permutexvar_ps(idx_lo, vs1);
    __m512 s1_hi = _mm512_permutexvar_ps(idx_hi, vs1);
    const int64_t k_base = g * k2_per_group;
    for (int64_t kp = 0; kp < k2_per_group; ++kp) {
      const int64_t k = k_base + kp;
      __m512bh vb0, vb1;
      if constexpr (std::is_same_v<packed_t, uint8_t>) {
        __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + k * BLOCK_N));
        std::tie(vb0, vb1) = CVT_MXFP4_TO_BF16(b4, zero, zero);
      } else {
        __m512i b8 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B + k * (BLOCK_N * 2)));
        vb0 = CVT_FP8_TO_BF16(_mm512_extracti32x8_epi32(b8, 0));
        vb1 = CVT_FP8_TO_BF16(_mm512_extracti32x8_epi32(b8, 1));
      }
      __m512 f0_lo = _mm512_mul_ps(CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)vb0, 0)), s0_lo);
      __m512 f0_hi = _mm512_mul_ps(CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)vb0, 1)), s0_hi);
      __m512 f1_lo = _mm512_mul_ps(CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)vb1, 0)), s1_lo);
      __m512 f1_hi = _mm512_mul_ps(CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)vb1, 1)), s1_hi);
      __m512bh q0 = _mm512_cvtne2ps_pbh(f0_hi, f0_lo);
      __m512bh q1 = _mm512_cvtne2ps_pbh(f1_hi, f1_lo);
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(Btmp + k * BLOCK_N * 2 + 0), (__m512i)q0);
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(Btmp + k * BLOCK_N * 2 + 32), (__m512i)q1);
    }
  }
}
#endif

template <typename packed_t, int GROUP_K>
void inkling2_fused_experts_quant_kernel_impl(
    at::BFloat16* __restrict__ output,
    at::BFloat16* __restrict__ ic1,
    at::BFloat16* __restrict__ ic2,
    at::BFloat16* __restrict__ A_tmp,
    at::BFloat16* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const at::BFloat16* __restrict__ input,
    const packed_t* __restrict__ packed_w13,
    const packed_t* __restrict__ packed_w2,
    const float* __restrict__ w13_scale,
    const float* __restrict__ w2_scale,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ topk_ids,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad,
    float gate_softcap,
    float up_softcap) {
#if !defined(CPU_CAPABILITY_AVX512)
  TORCH_CHECK(false, "inkling2_fused_experts_quant_kernel_impl requires AVX-512");
#else
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);

  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB_w13 = div_up(2 * N, BLOCK_N);
  const int64_t packed_K = get_row_size<packed_t>(K);
  const int64_t stride_e = 2 * N * packed_K;
  const int64_t groups_K = K / GROUP_K;
  const int64_t scale_stride_e = NB_w13 * groups_K * BLOCK_N;
  const int64_t btmp_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * std::max(K, N);

  auto run_tile = [&](const at::BFloat16* A_row,
                      const packed_t* B,
                      const float* Bs,
                      at::BFloat16* Btmp_slot,
                      float* C0,
                      int64_t m_size,
                      int64_t IC,
                      bool& reused_btmp,
                      int32_t& last_expert,
                      int32_t expert_id,
                      bool& used_brgemm) {
    if (m_size <= 4) {
      switch (m_size) {
        case 1:
          inkling2_quant_gemm_small_m<packed_t, GROUP_K, 1>(A_row, B, Bs, C0, IC);
          break;
        case 2:
          inkling2_quant_gemm_small_m<packed_t, GROUP_K, 2>(A_row, B, Bs, C0, IC);
          break;
        case 3:
          inkling2_quant_gemm_small_m<packed_t, GROUP_K, 3>(A_row, B, Bs, C0, IC);
          break;
        case 4:
          inkling2_quant_gemm_small_m<packed_t, GROUP_K, 4>(A_row, B, Bs, C0, IC);
          break;
      }
      return;
    }
    if (!reused_btmp || last_expert != expert_id) {
      inkling2_unpack_quant_block_to_bf16<packed_t, GROUP_K>(Btmp_slot, B, Bs, IC);
      last_expert = expert_id;
      reused_btmp = true;
    }
    at::native::cpublas::brgemm(m_size, BLOCK_N, IC, IC, BLOCK_N, BLOCK_N, false, A_row, Btmp_slot, C0);
    used_brgemm = true;
  };

  parallel_2d(MB, NB_w13, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    at::BFloat16* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    at::BFloat16* __restrict__ Btmp_tid = B_tmp + tid * btmp_per_thread;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    bool used_brgemm = false;
    int32_t unpacked_expert[MAX_CACHE_BLOCK_SIZE];
    bool unpacked_valid[MAX_CACHE_BLOCK_SIZE];
    for (int i = 0; i < MAX_CACHE_BLOCK_SIZE; ++i) {
      unpacked_expert[i] = -1;
      unpacked_valid[i] = false;
    }

    loop_2d<packed_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t n_size = std::min(2 * N - nb * BLOCK_N, BLOCK_N);
      int32_t expert_id = expert_ids[mb];
      const packed_t* __restrict__ B = packed_w13 + expert_id * stride_e + nb * BLOCK_N * packed_K;
      const float* __restrict__ Bs = w13_scale + expert_id * scale_stride_e + nb * groups_K * BLOCK_N;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      if (nb_offset == 0) {
        const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
        for (int64_t m = 0; m < m_size; ++m) {
          int32_t index = A_ids[m] / topk;
          copy_stub(A + m * K, input + index * K, K);
        }
      }

      run_tile(
          A,
          B,
          Bs,
          Btmp_tid + nb_offset * BLOCK_N * K,
          C0,
          m_size,
          K,
          unpacked_valid[nb_offset],
          unpacked_expert[nb_offset],
          expert_id,
          used_brgemm);

      const int64_t offset = offsets[mb];
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      const int64_t out_col_offset = (nb * BLOCK_N) / 2;
      for (int64_t m = 0; m < m_size; ++m) {
        float gamma = topk_weights[A_ids[m]];
        inkling2_capped_swiglu_stub(
            ic1 + (offset + m) * N + out_col_offset, C0 + m * BLOCK_N, n_size, gamma, gate_softcap, up_softcap);
      }
    });

    if (used_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  const int64_t OC = K;
  const int64_t IC = N;
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t packed_IC = get_row_size<packed_t>(IC);
  const int64_t stride_e2 = OC * packed_IC;
  const int64_t groups_IC = IC / GROUP_K;
  const int64_t scale_stride_e2 = NB2 * groups_IC * BLOCK_N;

  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    at::BFloat16* __restrict__ Btmp_tid = B_tmp + tid * btmp_per_thread;
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    bool used_brgemm = false;
    int32_t unpacked_expert[MAX_CACHE_BLOCK_SIZE];
    bool unpacked_valid[MAX_CACHE_BLOCK_SIZE];
    for (int i = 0; i < MAX_CACHE_BLOCK_SIZE; ++i) {
      unpacked_expert[i] = -1;
      unpacked_valid[i] = false;
    }

    loop_2d<packed_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);
      const at::BFloat16* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int32_t expert_id = expert_ids[mb];
      const packed_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * packed_IC;
      const float* __restrict__ Bs = w2_scale + expert_id * scale_stride_e2 + nb * groups_IC * BLOCK_N;

      run_tile(
          A,
          B,
          Bs,
          Btmp_tid + nb_offset * BLOCK_N * IC,
          C,
          m_size,
          IC,
          unpacked_valid[nb_offset],
          unpacked_expert[nb_offset],
          expert_id,
          used_brgemm);

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        copy_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, n_size);
      }
    });

    if (used_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  auto reduce_row = [&](int64_t m) {
    at::BFloat16* __restrict__ dst = output + m * K;
    const at::BFloat16* __restrict__ base = ic2 + m * topk * K;
    const int32_t* __restrict__ tids = topk_ids + m * topk;
    int64_t valid_idx[64];
    int n_valid = 0;
    for (int64_t k = 0; k < topk; ++k) {
      int32_t eid = tids[k];
      if (eid >= 0 && eid < E && n_valid < 64) {
        valid_idx[n_valid++] = k;
      }
    }
    if (n_valid == 0) {
      std::fill(dst, dst + K, static_cast<at::BFloat16>(0.0f));
    } else if (n_valid == 1) {
      const at::BFloat16* __restrict__ src = base + valid_idx[0] * K;
      std::copy(src, src + K, dst);
    } else {
      using bVec = at::vec::Vectorized<at::BFloat16>;
      using fVec = at::vec::Vectorized<float>;
      constexpr int kVecSize = bVec::size();
      int64_t d = 0;
      for (; d <= K - kVecSize; d += kVecSize) {
        fVec acc0 = fVec(0.f);
        fVec acc1 = fVec(0.f);
        for (int vi = 0; vi < n_valid; ++vi) {
          bVec bv = bVec::loadu(base + valid_idx[vi] * K + d);
          auto [v0, v1] = convert_to_float<at::BFloat16>(bv);
          acc0 += v0;
          acc1 += v1;
        }
        convert_from_float_ext<at::BFloat16>(acc0, acc1).store(dst + d);
      }
      for (; d < K; ++d) {
        float acc = 0.f;
        for (int vi = 0; vi < n_valid; ++vi) {
          acc += static_cast<float>(base[valid_idx[vi] * K + d]);
        }
        dst[d] = static_cast<at::BFloat16>(acc);
      }
    }
  };
  if (M <= 4 && M * K <= 16384) {
    for (int64_t m = 0; m < M; ++m) {
      reduce_row(m);
    }
  } else {
    at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        reduce_row(m);
      }
    });
  }
#endif
}

}  // anonymous namespace

// common checks
template <CPUQuantMethod quant>
static inline void check_moe_scales(
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size) {
  if constexpr (quant == CPUQuantMethod::INT8_W8A8) {
    TORCH_CHECK(w1_scale.has_value(), "missing w1_scale for int8 w8a8.");
    TORCH_CHECK(w2_scale.has_value(), "missing w2_scale for int8 w8a8.");
  } else if constexpr (quant == CPUQuantMethod::FP8_W8A16) {
    TORCH_CHECK(w1_scale.has_value(), "missing w1_scale for fp8 w8a16.");
    TORCH_CHECK(w2_scale.has_value(), "missing w2_scale for fp8 w8a16.");
    TORCH_CHECK(block_size.has_value(), "missing block_size for fp8 w8a16.");
    TORCH_CHECK(block_size.value().size() == 2, "expect block_size.size() to be 2.");
  } else if constexpr (quant == CPUQuantMethod::MXFP4) {
    TORCH_CHECK(w1_scale.has_value(), "missing w1_scale for mxfp4.");
    TORCH_CHECK(w2_scale.has_value(), "missing w2_scale for mxfp4.");
    TORCH_CHECK(w1_scale.value().scalar_type() == at::kByte, "expect w1_scale to be uint8.");
    TORCH_CHECK(w2_scale.value().scalar_type() == at::kByte, "expect w2_scale to be uint8.");
  }
}

static inline void check_moe_scales(
    int64_t moe_comp_method,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size) {
  if (moe_comp_method == CPUQuantMethod::INT8_W8A8) {
    check_moe_scales<CPUQuantMethod::INT8_W8A8>(w1_scale, w2_scale, block_size);
  } else if (moe_comp_method == CPUQuantMethod::FP8_W8A16) {
    check_moe_scales<CPUQuantMethod::FP8_W8A16>(w1_scale, w2_scale, block_size);
  } else if (moe_comp_method == CPUQuantMethod::MXFP4) {
    check_moe_scales<CPUQuantMethod::MXFP4>(w1_scale, w2_scale, block_size);
  }
}

#define CHECK_MOE_SCALES_FP8(DIM0, DIM1)                      \
  auto w1s = w1_scale.value();                                \
  auto w2s = w2_scale.value();                                \
  auto block_size_val = block_size.value();                   \
  int64_t block_size_N = block_size_val[0];                   \
  int64_t block_size_K = block_size_val[1];                   \
  TORCH_CHECK(w1s.size(DIM0) == div_up(2 * N, block_size_N)); \
  TORCH_CHECK(w1s.size(DIM1) == div_up(K, block_size_K));     \
  TORCH_CHECK(w2s.size(DIM0) == div_up(K, block_size_N));     \
  TORCH_CHECK(w2s.size(DIM1) == div_up(N, block_size_K))

// hidden_states: [M, K]
// w1: [E, 2N, K] or [E, 2N, K / 2] for uint8
// w2: [E, K, N] or [E, K, N / 2] for uint8
// topk_weights: [M, topk]
// topk_ids: [M, topk] (int32_t)
//

at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    int64_t moe_comp_method,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w1_zero,
    const std::optional<at::Tensor>& w2_zero,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    const std::optional<double>& alpha,
    const std::optional<double>& limit,
    bool is_vnni,
    const std::optional<std::string>& activation) {
  const CPUActMethod act_func = act_method_from_string(activation, alpha.has_value() && limit.has_value());
  // The int8 kernel only supports SiLU; int4 supports SiLU and GELU.
  TORCH_CHECK(
      !(moe_comp_method == CPUQuantMethod::INT8_W8A8) || act_func == CPUActMethod::silu_and_mul,
      "fused_experts_cpu: INT8_W8A8 supports activation='silu' only");
  TORCH_CHECK(
      moe_comp_method != CPUQuantMethod::INT4_W4A8 || act_func == CPUActMethod::silu_and_mul ||
          act_func == CPUActMethod::gelu_and_mul,
      "fused_experts_cpu: INT4_W4A8 supports activation='silu' or 'gelu'");

  auto packed_w1 = is_vnni ? w1 : convert_weight_packed(w1);
  auto packed_w2 = is_vnni ? w2 : convert_weight_packed(w2);

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  const auto st = hidden_states.scalar_type();
  // TODO: fused_topk_torch_native (CPU fallback for models like MiniMax)
  // returns int64 topk_ids; fused_experts_cpu requires int32. Remove the typecast after topk kernel is provided
  auto topk_ids_ = topk_ids.scalar_type() == at::kInt ? topk_ids : topk_ids.to(at::kInt);

  CHECK_INPUT(hidden_states);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_EQ(topk_weights.sizes(), topk_ids_.sizes());
  CHECK_DIM(2, hidden_states);
  if (moe_comp_method == CPUQuantMethod::INT4_W4A8 && is_vnni) {
    CHECK_DIM(4, w1);
    CHECK_DIM(4, w2);
  } else {
    CHECK_DIM(3, w1);
    CHECK_DIM(3, w2);
  }
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids_);

  CHECK_EQ(topk_ids_.scalar_type(), at::kInt);

  // TODO: support topk_weights to be bf16 or fp16 in the kernel.
  // The topk_weights of llama4 is computed via Llama4MoE:custom_routing_function and is bf16/fp16
  // while the kernel currently only supports it to be float32
  auto topk_weights_ = topk_weights.to(at::kFloat);
  CHECK_EQ(topk_weights_.scalar_type(), at::kFloat);

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N = moe_comp_method == CPUQuantMethod::INT4_W4A8 ? w1_scale.value().size(1) * w1_scale.value().size(3) / 2
                                                           : w1.size(1) / 2;
  int64_t E = w1.size(0);
  int64_t topk = topk_weights_.size(1);

  // we use int32_t compensation for int8 w8a8
  int64_t packed_K = get_row_size(static_cast<CPUQuantMethod>(moe_comp_method), K);
  int64_t packed_N = get_row_size(static_cast<CPUQuantMethod>(moe_comp_method), N);

  // check weight shapes
  CHECK_EQ(w2.size(0), E);
  if (!(moe_comp_method == CPUQuantMethod::INT4_W4A8)) {
    CHECK_EQ(w2.size(1), K);
    CHECK_EQ(packed_w1.size(2), packed_K / (moe_comp_method == CPUQuantMethod::INT4_W4A8 ? 2 : 1));
    CHECK_EQ(packed_w2.size(2), packed_N / (moe_comp_method == CPUQuantMethod::INT4_W4A8 ? 2 : 1));
  }
  // check scales
  check_moe_scales(moe_comp_method, w1_scale, w2_scale, block_size);

  at::Tensor out_hidden_states = inplace ? hidden_states : at::empty_like(hidden_states);

  // NB: worst case is each expert holds a block with remainder of 1
  //   1. sorted_ids : [M * topk + E * (BLOCK_M - 1)]
  //   2. expert_ids : [max_num_blocks]
  //   3. total_cnts : [T + 1, E]
  //   4. cumsums    : [E + 1]
  //   5. offsets    : [max_num_blocks + 1]
  //
  int num_threads = at::get_num_threads();
  int64_t max_num_tokens_padded = M * topk + E * (BLOCK_M - 1);
  int64_t max_num_blocks = div_up(max_num_tokens_padded, BLOCK_M);
  auto buffer = at::empty(
      {max_num_tokens_padded + max_num_blocks + (num_threads + 1) * E + (E + 1) + (max_num_blocks + 1)},
      topk_ids_.options());

  int32_t* __restrict__ sorted_ids = buffer.data_ptr<int32_t>();
  int32_t* __restrict__ expert_ids = sorted_ids + max_num_tokens_padded;
  int32_t* __restrict__ total_cnts = expert_ids + max_num_blocks;
  int32_t* __restrict__ cumsums = total_cnts + (num_threads + 1) * E;
  int32_t* __restrict__ offsets = cumsums + (E + 1);

  // init sorted_ids with `numel` as the padding number
  // init expert_ids with `num_experts`
  int64_t numel = M * topk;
  at::parallel_for(0, max_num_blocks, GRAIN_SIZE / BLOCK_M, [&](int64_t begin, int64_t end) {
    int64_t m_start = begin * BLOCK_M;
    int64_t m_size = std::min((end - begin) * BLOCK_M, max_num_tokens_padded - m_start);
    fill_stub(sorted_ids + m_start, (int32_t)numel, m_size);
    fill_stub(expert_ids + begin, (int32_t)E, end - begin);
  });
  // zero total_cnts and cumsums
  at::parallel_for(0, (num_threads + 1) * E + (E + 1), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    fill_stub(total_cnts + begin, 0, end - begin);
  });

  // align experts index
  int64_t num_tokens_post_pad = moe_align_block_size<BLOCK_M>(
      sorted_ids, expert_ids, topk_ids_.data_ptr<int32_t>(), total_cnts, cumsums, offsets, E, numel, num_threads);

  // unlike triton kernel, we fuse silu with gemm1 so only need 2 intermediate_caches:
  //   1. intermediate_cache1 : [M * topk, N]
  //   2. intermediate_cache2 : [M * topk, K]
  //   3. A_tmp : [T, BLOCK_M * K]
  //   4. C_tmp : [T, 2 * BLOCK_M * BLOCK_N]
  //
  // for int8 w8a8:
  //   5. Aq_tmp : [M, K] or [M * topk, N]
  //   6. As_tmp : [M * topk]
  //
  // for fp8 w8a16 and mxfp4:
  //   7. intermediate_cache0 : [M * topk, 2N]
  //   8. B_tmp : [T, MAX_CACHE_BLOCK_SIZE, BLOCK_N, std::max(K, N)]
  //
  int64_t buffer_size_nbytes =
      M * topk * N * 2 + M * topk * K * 2 +
      num_threads * BLOCK_M * K *
          (moe_comp_method == CPUQuantMethod::INT8_W8A8 | moe_comp_method == CPUQuantMethod::INT4_W4A8 ? 1 : 2) +
      num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float);

  if (moe_comp_method == CPUQuantMethod::INT8_W8A8) {
    buffer_size_nbytes += std::max(M * K, M * topk * N) + M * topk * sizeof(float);
  }
  if (moe_comp_method == CPUQuantMethod::FP8_W8A16 || moe_comp_method == CPUQuantMethod::MXFP4) {
    buffer_size_nbytes += M * topk * 2 * N * 2 + num_threads * MAX_CACHE_BLOCK_SIZE * BLOCK_N * std::max(K, N) * 2;
  }
  if (moe_comp_method == CPUQuantMethod::INT4_W4A8) {
    buffer_size_nbytes += M * topk * 2 * N * 2 + std::max(M * K, M * topk * N) + M * topk * sizeof(float) +
                          num_threads * 2 * get_4bit_block_k_size(K / w1_scale.value().size(2)) * BLOCK_N;
  }
  auto buffer2 = at::empty({buffer_size_nbytes}, hidden_states.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_experts_kernel_impl", [&] {
    scalar_t* __restrict__ intermediate_cache1 = (scalar_t*)((void*)(buffer2.data_ptr<int8_t>()));
    scalar_t* __restrict__ intermediate_cache2 = intermediate_cache1 + M * topk * N;

    if (moe_comp_method == CPUQuantMethod::INT8_W8A8) {
      uint8_t* __restrict__ A_tmp = (uint8_t*)((void*)(intermediate_cache2 + M * topk * K));
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
      uint8_t* __restrict__ Aq_tmp = (uint8_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      float* __restrict__ As_tmp = (float*)((void*)(Aq_tmp + std::max(M * K, M * topk * N)));

      auto w1s = w1_scale.value();
      auto w2s = w2_scale.value();
      TORCH_CHECK(w1s.numel() == E * 2 * N);
      TORCH_CHECK(w2s.numel() == E * K);

      fused_experts_int8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          C_tmp,
          Aq_tmp,
          As_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<int8_t>(),
          packed_w2.data_ptr<int8_t>(),
          w1s.data_ptr<float>(),
          w2s.data_ptr<float>(),
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad);
    } else if (moe_comp_method == CPUQuantMethod::FP8_W8A16) {
      // here we just ignore C_tmp as it is not used
      scalar_t* __restrict__ A_tmp = (scalar_t*)((void*)(intermediate_cache2 + M * topk * K));
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
      scalar_t* __restrict__ intermediate_cache0 = (scalar_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      scalar_t* __restrict__ B_tmp = (scalar_t*)((void*)(intermediate_cache0 + M * topk * 2 * N));
      bool with_bias = w1_bias.has_value();

      CHECK_MOE_SCALES_FP8(1, 2);
      fused_experts_fp_kernel_impl<scalar_t, at::Float8_e4m3fn, float, false>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache0,
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          B_tmp,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<at::Float8_e4m3fn>(),
          packed_w2.data_ptr<at::Float8_e4m3fn>(),
          with_bias ? w1_bias.value().data_ptr<float>() : nullptr,
          with_bias ? w2_bias.value().data_ptr<float>() : nullptr,
          w1s.data_ptr<float>(),
          w2s.data_ptr<float>(),
          block_size_N,
          block_size_K,
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad,
          alpha.has_value() ? float(alpha.value()) : 0,
          limit.has_value() ? float(limit.value()) : 0,
          act_func,
          with_bias);
    } else if (moe_comp_method == CPUQuantMethod::MXFP4) {
      scalar_t* __restrict__ A_tmp = (scalar_t*)((void*)(intermediate_cache2 + M * topk * K));
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
      scalar_t* __restrict__ intermediate_cache0 = (scalar_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      scalar_t* __restrict__ B_tmp = (scalar_t*)((void*)(intermediate_cache0 + M * topk * 2 * N));
      bool with_bias = w1_bias.has_value();

      // mxfp4 supports only group size of 32 (2^5)
      constexpr int64_t group_size = 32;
      auto w1s = w1_scale.value();
      auto w2s = w2_scale.value();
      TORCH_CHECK(w1s.numel() == (E * 2 * N * K >> 5), "mxfp4 w1_scale must hold one scale per 32 weights.");
      TORCH_CHECK(w2s.numel() == (E * K * N >> 5), "mxfp4 w2_scale must hold one scale per 32 weights.");
      fused_experts_fp_kernel_impl<scalar_t, uint8_t, uint8_t, true>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache0,
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          B_tmp,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<uint8_t>(),
          packed_w2.data_ptr<uint8_t>(),
          with_bias ? w1_bias.value().data_ptr<float>() : nullptr,
          with_bias ? w2_bias.value().data_ptr<float>() : nullptr,
          w1s.data_ptr<uint8_t>(),
          w2s.data_ptr<uint8_t>(),
          /*block_size_N*/ 1,
          /*block_size_K*/ group_size,
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad,
          alpha.has_value() ? float(alpha.value()) : 0,
          limit.has_value() ? float(limit.value()) : 0,
          act_func,
          with_bias);
    } else if (moe_comp_method == CPUQuantMethod::INT4_W4A8) {
      uint8_t* __restrict__ A_tmp = (uint8_t*)((void*)(intermediate_cache2 + M * topk * K));
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
      scalar_t* __restrict__ intermediate_cache0 = (scalar_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      uint8_t* __restrict__ Aq_tmp = (uint8_t*)((void*)(intermediate_cache0 + M * topk * 2 * N));
      float* __restrict__ As_tmp = (float*)((void*)(Aq_tmp + std::max(M * K, M * topk * N)));
      int8_t* __restrict__ dqB_tmp = (int8_t*)((void*)(As_tmp + M * topk));

      // weight + compensation shape = [Nc, Kc, block_n * block_k / 2 + block_n*sizeof(int32_t)]
      // scales/qzeros shape = [E, Nc, G, block_n]
      int64_t num_groups = w1_scale.value().size(2);
      const int group_size = K / num_groups;
      // TODO: check scales and zeros
      fused_experts_int4_w4a8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache0,
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          Aq_tmp,
          As_tmp,
          nullptr,
          C_tmp,
          dqB_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<uint8_t>(),
          packed_w2.data_ptr<uint8_t>(),
          w1_zero.value().data_ptr<int8_t>(),
          w2_zero.value().data_ptr<int8_t>(),
          w1_scale.value().data_ptr<float>(),
          w2_scale.value().data_ptr<float>(),
          group_size,
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad,
          act_func);
    } else {
      scalar_t* __restrict__ A_tmp = intermediate_cache2 + M * topk * K;
      float* __restrict__ C_tmp = (float*)((void*)(A_tmp + num_threads * BLOCK_M * K));
      bool with_bias = w1_bias.has_value();

      fused_experts_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<scalar_t>(),
          packed_w2.data_ptr<scalar_t>(),
          with_bias ? w1_bias.value().data_ptr<float>() : nullptr,
          with_bias ? w2_bias.value().data_ptr<float>() : nullptr,
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad,
          alpha.has_value() ? float(alpha.value()) : 0,
          limit.has_value() ? float(limit.value()) : 0,
          act_func,
          with_bias);
    }
  });
  return out_hidden_states;
}

// shared expert kernel
//
// hidden_states: [M, K]
// w1: [2N, K]
// w2: [K, N]
// fused_experts_out
at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    const std::optional<at::Tensor>& fused_experts_out,
    const std::optional<double> routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    bool is_vnni) {
  auto packed_w1 = is_vnni ? w1 : convert_weight_packed(w1);
  auto packed_w2 = is_vnni ? w2 : convert_weight_packed(w2);

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  double routed_scaling_factor_value = 0;
  if (routed_scaling_factor.has_value()) {
    TORCH_CHECK(fused_experts_out.has_value(), "shared_expert_cpu: expect fused_experts_out.");
    const auto fused_experts_out_tensor = fused_experts_out.value();
    routed_scaling_factor_value = routed_scaling_factor.value();
    CHECK_INPUT(fused_experts_out_tensor);
    CHECK_EQ(hidden_states.sizes(), fused_experts_out_tensor.sizes());
  }

  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, w1);
  CHECK_DIM(2, w2);
  CHECK_EQ(hidden_states.scalar_type(), st);

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N = w1.size(0) / 2;

  // we use int32_t compensation for int8 w8a8
  int64_t packed_K = get_row_size(K, use_int8_w8a8);
  int64_t packed_N = get_row_size(N, use_int8_w8a8);

  // check weight shapes
  CHECK_EQ(w2.size(0), K);
  CHECK_EQ(packed_w1.size(1), packed_K);
  CHECK_EQ(packed_w2.size(1), packed_N);

  // check scales
  if (use_int8_w8a8) {
    check_moe_scales<CPUQuantMethod::INT8_W8A8>(w1_scale, w2_scale, block_size);
  } else if (use_fp8_w8a16) {
    check_moe_scales<CPUQuantMethod::FP8_W8A16>(w1_scale, w2_scale, block_size);
  }

  at::Tensor out_hidden_states = inplace ? hidden_states : at::empty_like(hidden_states);

  // unlike triton kernel, we fuse silu with gemm1 so only need 2 intermediate_caches:
  //   1. intermediate_cache1 : [M, N]
  //   2. C_tmp : [T, 2 * BLOCK_M * BLOCK_N]
  //
  // for int8 w8a8:
  //   3. Aq_tmp : [M, K] or [M, N]
  //   4. As_tmp : [M]
  //
  // for fp8 w8a16:
  //   5. intermediate_cache0 : [M, 2N]
  //   6. B_tmp: [T, MAX_CACHE_BLOCK_SIZE, BLOCK_M, max(K, N)]
  //
  int num_threads = at::get_num_threads();
  int64_t buffer_size_nbytes = M * N * 2 + num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float);

  if (use_int8_w8a8) {
    buffer_size_nbytes += std::max(M * K, M * N) + M * sizeof(float);
  }
  if (use_fp8_w8a16) {
    buffer_size_nbytes += M * 2 * N * 2 + num_threads * MAX_CACHE_BLOCK_SIZE * BLOCK_M * std::max(K, N) * 2;
  }

  auto buffer = at::empty({buffer_size_nbytes}, hidden_states.options().dtype(at::kChar));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "share_experts_kernel_impl", [&] {
    scalar_t* __restrict__ intermediate_cache1 = (scalar_t*)((void*)(buffer.data_ptr<int8_t>()));
    float* __restrict__ C_tmp = (float*)((void*)(intermediate_cache1 + M * N));

    if (use_int8_w8a8) {
      uint8_t* __restrict__ Aq_tmp = (uint8_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      float* __restrict__ As_tmp = (float*)((void*)(Aq_tmp + std::max(M * K, M * N)));

      auto w1s = w1_scale.value();
      auto w2s = w2_scale.value();
      TORCH_CHECK(w1s.numel() == 2 * N);
      TORCH_CHECK(w2s.numel() == K);

      shared_expert_int8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          C_tmp,
          Aq_tmp,
          As_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<int8_t>(),
          packed_w2.data_ptr<int8_t>(),
          w1s.data_ptr<float>(),
          w2s.data_ptr<float>(),
          conditional_data_ptr<scalar_t>(fused_experts_out),
          routed_scaling_factor_value,
          M,
          N,
          K);
    } else if (use_fp8_w8a16) {
      scalar_t* __restrict__ intermediate_cache0 = (scalar_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
      scalar_t* __restrict__ B_tmp = (scalar_t*)((void*)(intermediate_cache0 + M * 2 * N));

      CHECK_MOE_SCALES_FP8(0, 1);
      shared_expert_fp8_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache0,
          intermediate_cache1,
          B_tmp,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<at::Float8_e4m3fn>(),
          packed_w2.data_ptr<at::Float8_e4m3fn>(),
          w1s.data_ptr<float>(),
          w2s.data_ptr<float>(),
          block_size_N,
          block_size_K,
          conditional_data_ptr<scalar_t>(fused_experts_out),
          routed_scaling_factor_value,
          M,
          N,
          K);
    } else {
      shared_expert_kernel_impl<scalar_t>(
          out_hidden_states.data_ptr<scalar_t>(),
          intermediate_cache1,
          C_tmp,
          hidden_states.data_ptr<scalar_t>(),
          packed_w1.data_ptr<scalar_t>(),
          packed_w2.data_ptr<scalar_t>(),
          conditional_data_ptr<scalar_t>(fused_experts_out),
          routed_scaling_factor_value,
          M,
          N,
          K);
    }
  });
  return out_hidden_states;
}

at::Tensor inkling2_fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w13,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    double gate_softcap,
    double up_softcap,
    bool inplace,
    bool is_vnni,
    const std::optional<at::Tensor>& w13_scale = std::nullopt,
    const std::optional<at::Tensor>& w2_scale = std::nullopt) {
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(w13);
  CHECK_INPUT(w2);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(3, w13);
  CHECK_DIM(3, w2);
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);
  CHECK_EQ(topk_weights.sizes(), topk_ids.sizes());

  const auto st = hidden_states.scalar_type();
  const bool has_quant_scales = w13_scale.has_value() || w2_scale.has_value();
  if (has_quant_scales) {
    TORCH_CHECK(w13_scale.has_value() && w2_scale.has_value(), "w13_scale and w2_scale must be given together");
    TORCH_CHECK(st == at::kBFloat16, "Quantized inkling2_fused_experts_cpu requires bfloat16 hidden_states");
    TORCH_CHECK(is_vnni, "Quantized inkling2_fused_experts_cpu requires VNNI-packed weights (is_vnni=true)");
    TORCH_CHECK(w13.scalar_type() == w2.scalar_type(), "w13 and w2 must have matching quantized dtypes");
    TORCH_CHECK(
        w13.scalar_type() == at::kByte || w13.scalar_type() == at::kFloat8_e4m3fn,
        "Unsupported quantized weight dtype for inkling2_fused_experts_cpu: ",
        w13.scalar_type());
  } else {
    TORCH_CHECK(w13.scalar_type() == st && w2.scalar_type() == st, "inkling2_fused_experts_cpu: dtype mismatch");
  }

  auto topk_ids_ =
      (topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous()) ? topk_ids : topk_ids.to(at::kInt).contiguous();
  auto topk_weights_ = (topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous())
                           ? topk_weights
                           : topk_weights.to(at::kFloat).contiguous();

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t E = w13.size(0);
  int64_t two_N = w13.size(1);
  TORCH_CHECK(two_N % 2 == 0, "inkling2_fused_experts_cpu: w13.size(1) must be even");
  int64_t N = two_N / 2;
  int64_t topk = topk_weights_.size(1);
  const int64_t k_packed_factor = (has_quant_scales && w13.scalar_type() == at::kByte) ? 2 : 1;

  CHECK_EQ(w13.size(2) * k_packed_factor, K);
  CHECK_EQ(w2.size(0), E);
  CHECK_EQ(w2.size(1), K);
  CHECK_EQ(w2.size(2) * k_packed_factor, N);
  CHECK_EQ(topk_weights_.size(0), M);

  at::Tensor out_hidden_states = inplace ? hidden_states : at::empty_like(hidden_states);
  if (M == 0) {
    return out_hidden_states;
  }

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  int num_threads = at::get_num_threads();
  int64_t max_num_tokens_padded = M * topk + E * (BLOCK_M - 1);
  int64_t max_num_blocks = div_up(max_num_tokens_padded, BLOCK_M);
  int64_t req_i32 = max_num_tokens_padded + max_num_blocks + (num_threads + 1) * E + (E + 1) + (max_num_blocks + 1);
  static at::Tensor ws_int32;
  if (!ws_int32.defined() || ws_int32.numel() < req_i32) {
    ws_int32 = at::empty({req_i32}, topk_ids_.options());
  }

  int32_t* __restrict__ sorted_ids = ws_int32.data_ptr<int32_t>();
  int32_t* __restrict__ expert_ids = sorted_ids + max_num_tokens_padded;
  int32_t* __restrict__ total_cnts = expert_ids + max_num_blocks;
  int32_t* __restrict__ cumsums = total_cnts + (num_threads + 1) * E;
  int32_t* __restrict__ offsets = cumsums + (E + 1);

  int64_t numel = M * topk;
  if (max_num_blocks <= 512) {
    std::fill(sorted_ids, sorted_ids + max_num_tokens_padded, (int32_t)numel);
    std::fill(expert_ids, expert_ids + max_num_blocks, (int32_t)E);
  } else {
    at::parallel_for(0, max_num_blocks, GRAIN_SIZE / BLOCK_M, [&](int64_t begin, int64_t end) {
      int64_t m_start = begin * BLOCK_M;
      int64_t m_size = std::min((end - begin) * BLOCK_M, max_num_tokens_padded - m_start);
      fill_stub(sorted_ids + m_start, (int32_t)numel, m_size);
      fill_stub(expert_ids + begin, (int32_t)E, end - begin);
    });
  }
  const int64_t cnt_elems = (num_threads + 1) * E + (E + 1);
  if (cnt_elems <= 8192) {
    std::fill(total_cnts, total_cnts + cnt_elems, 0);
  } else {
    at::parallel_for(
        0, cnt_elems, GRAIN_SIZE, [&](int64_t begin, int64_t end) { fill_stub(total_cnts + begin, 0, end - begin); });
  }

  int64_t num_tokens_post_pad = inkling2_moe_align_block_size<BLOCK_M>(
      sorted_ids, expert_ids, topk_ids_.data_ptr<int32_t>(), total_cnts, cumsums, offsets, E, numel, num_threads);
  if (num_tokens_post_pad == 0) {
    out_hidden_states.zero_();
    return out_hidden_states;
  }

  int64_t buffer_size_nbytes = M * topk * N * sizeof(uint16_t) + M * topk * K * sizeof(uint16_t) +
                               num_threads * BLOCK_M * K * sizeof(uint16_t) +
                               num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float);
  if (has_quant_scales) {
    buffer_size_nbytes += num_threads * MAX_CACHE_BLOCK_SIZE * BLOCK_N * std::max(K, N) * sizeof(uint16_t);
  }
  static at::Tensor ws_bytes;
  if (!ws_bytes.defined() || ws_bytes.numel() < buffer_size_nbytes) {
    ws_bytes = at::empty({buffer_size_nbytes}, hidden_states.options().dtype(at::kChar));
  }

  if (has_quant_scales) {
    TORCH_CHECK(two_N % BLOCK_N == 0 && K % 32 == 0 && N % 32 == 0, "Quantized inkling2 MoE requires 32-aligned dims");
    auto w13_s = w13_scale.value().contiguous();
    auto w2_s = w2_scale.value().contiguous();
    TORCH_CHECK(w13_s.scalar_type() == at::kFloat && w2_s.scalar_type() == at::kFloat, "MoE scales must be float32");
    at::BFloat16* __restrict__ intermediate_cache1 = reinterpret_cast<at::BFloat16*>(ws_bytes.data_ptr<int8_t>());
    at::BFloat16* __restrict__ intermediate_cache2 = intermediate_cache1 + M * topk * N;
    at::BFloat16* __restrict__ A_tmp = intermediate_cache2 + M * topk * K;
    float* __restrict__ C_tmp = reinterpret_cast<float*>(A_tmp + num_threads * BLOCK_M * K);
    at::BFloat16* __restrict__ B_tmp = reinterpret_cast<at::BFloat16*>(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N);
    if (w13.scalar_type() == at::kByte) {
      inkling2_fused_experts_quant_kernel_impl<uint8_t, 16>(
          out_hidden_states.data_ptr<at::BFloat16>(),
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          B_tmp,
          C_tmp,
          hidden_states.data_ptr<at::BFloat16>(),
          w13.data_ptr<uint8_t>(),
          w2.data_ptr<uint8_t>(),
          w13_s.data_ptr<float>(),
          w2_s.data_ptr<float>(),
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          topk_ids_.data_ptr<int32_t>(),
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad,
          static_cast<float>(gate_softcap),
          static_cast<float>(up_softcap));
    } else {
      inkling2_fused_experts_quant_kernel_impl<at::Float8_e4m3fn, 32>(
          out_hidden_states.data_ptr<at::BFloat16>(),
          intermediate_cache1,
          intermediate_cache2,
          A_tmp,
          B_tmp,
          C_tmp,
          hidden_states.data_ptr<at::BFloat16>(),
          w13.data_ptr<at::Float8_e4m3fn>(),
          w2.data_ptr<at::Float8_e4m3fn>(),
          w13_s.data_ptr<float>(),
          w2_s.data_ptr<float>(),
          topk_weights_.data_ptr<float>(),
          sorted_ids,
          expert_ids,
          offsets,
          topk_ids_.data_ptr<int32_t>(),
          M,
          N,
          K,
          E,
          topk,
          num_tokens_post_pad,
          static_cast<float>(gate_softcap),
          static_cast<float>(up_softcap));
    }
    return out_hidden_states;
  }

  auto packed_w13 = is_vnni ? w13 : convert_weight_packed(w13);
  auto packed_w2 = is_vnni ? w2 : convert_weight_packed(w2);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "inkling2_fused_experts_kernel_impl", [&] {
    scalar_t* __restrict__ intermediate_cache1 = reinterpret_cast<scalar_t*>(ws_bytes.data_ptr<int8_t>());
    scalar_t* __restrict__ intermediate_cache2 = intermediate_cache1 + M * topk * N;
    scalar_t* __restrict__ A_tmp = intermediate_cache2 + M * topk * K;
    float* __restrict__ C_tmp = reinterpret_cast<float*>(A_tmp + num_threads * BLOCK_M * K);

    inkling2_fused_experts_kernel_impl<scalar_t>(
        out_hidden_states.data_ptr<scalar_t>(),
        intermediate_cache1,
        intermediate_cache2,
        A_tmp,
        C_tmp,
        hidden_states.data_ptr<scalar_t>(),
        packed_w13.data_ptr<scalar_t>(),
        packed_w2.data_ptr<scalar_t>(),
        topk_weights_.data_ptr<float>(),
        sorted_ids,
        expert_ids,
        offsets,
        topk_ids_.data_ptr<int32_t>(),
        M,
        N,
        K,
        E,
        topk,
        num_tokens_post_pad,
        static_cast<float>(gate_softcap),
        static_cast<float>(up_softcap));
  });
  return out_hidden_states;
}
