#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// mxfp4
inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const uint8_t* __restrict__ packed_B,
    int64_t N,
    int64_t K,
    int64_t ldb,
    int64_t ldb_tmp,
    const uint8_t* __restrict__ scale) {
#if defined(CPU_CAPABILITY_AVX512)
  // [K/2, N, 2]
  const int64_t K2 = K >> 1;
  const int64_t ldb2 = ldb;                                           // ldb * 2 >> 1;
  const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(packed_B);  // 2 * 4 bit = 8 bit

  constexpr int BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);

  // prefetch distance
  constexpr int PREFETCH_SIZE_K = 64;

  // exponent bias 127
  const __m512i off = _mm512_set1_epi16(0x7F);

  // load 32 bytes only once for each block
  __m256i s8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scale));
  __m512i s16 = _mm512_slli_epi16(_mm512_sub_epi16(_mm512_cvtepu8_epi16(s8), off), 0x7);

  // holds Nx2(64) scales, interleaved as 2 belongs to K dimension
  // e.g. vs0: { s0,  s0,  s1,  s1, ..., s15, s15}
  //      vs1: {s16, s16, s17, s17, ..., s31, s31}
  auto [vscale0, vscale1] = transpose_2x32_16bit(s16, s16);

#pragma GCC unroll 4
  for (int64_t k = 0; k < K2; ++k) {
    __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + k * ldb2));
    if constexpr (PREFETCH_SIZE_K > 0) {
      _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2, _MM_HINT_T0);
    }
    auto [vb0, vb1] = CVT_MXFP4_TO_BF16(b4, vscale0, vscale1);

    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 0, (__m512i)vb0);
    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 32, (__m512i)vb1);
  }
#else
  TORCH_CHECK(false, "unpack_B: scalar path not implemented!");
#endif
}

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const uint8_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ bias,
      const uint8_t* __restrict__ scale,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const uint8_t* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const float* __restrict__ bias,
      const uint8_t* __restrict__ scale,
      int K,
      int lda,
      int ldb,
      int ldc,
      int64_t block_size_K) {
    // mxfp4 supports only group size of 32
    // expect weight packed in 32-way, vnni2 format Nx2(64)
    assert(block_size_K == 32);
    assert(BLOCK_N == 32);

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 64;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    // holds Nx2(64) scales, interleaved as 2 belongs to K dimension
    // e.g. vs0: { s0,  s0,  s1,  s1, ..., s15, s15}
    //      vs1: {s16, s16, s17, s17, ..., s31, s31}
    __m512i vscale[COLS];

    // exponent bias 127
    const __m512i off = _mm512_set1_epi16(0x7F);

    auto loadc = [&](auto i) {
      constexpr int col = i % COLS;
      if constexpr (has_bias) {
        vc[i] = _mm512_loadu_ps(bias + col * 16);
      } else {
        vc[i] = _mm512_setzero_ps();
      }
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(a_ptr + row * lda2 + k + PREFETCH_SIZE_K, _MM_HINT_T0);
        }
      }
      if constexpr (row == 0) {
        // load 32 * 2 (64) int4 at a time
        if constexpr (col % 2 == 0) {
          __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + k * ldb2 + col * 16));
          if constexpr (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          std::tie(vb[col + 0], vb[col + 1]) = CVT_MXFP4_TO_BF16(b4, vscale[col + 0], vscale[col + 1]);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };

    for (int64_t k = 0; k < K2; ++k) {
      // update scales every 16x2 K
      if ((k & 15) == 0) {
        __m256i s8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scale + (k >> 4) * 32));
        __m512i s16 = _mm512_slli_epi16(_mm512_sub_epi16(_mm512_cvtepu8_epi16(s8), off), 0x7);
        std::tie(vscale[0], vscale[1]) = transpose_2x32_16bit(s16, s16);
      }
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2,4 use 512bit store
      if constexpr (col % 2 == 0) {
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                          \
      B + nb_start * 2,                                            \
      C + mb_start * ldc + nb_start,                               \
      has_bias ? bias + nb_start : nullptr,                        \
      scale,                                                       \
      K,                                                           \
      lda,                                                         \
      ldb,                                                         \
      ldc,                                                         \
      block_size_K);

template <typename scalar_t, bool has_bias>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const uint8_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const uint8_t* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      bool do_unpack = true) {
    TORCH_CHECK(false, "struct brgemm: primary template not implemented!");
  }
};

template <bool has_bias>
struct brgemm<at::BFloat16, has_bias> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const uint8_t* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const uint8_t* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      bool do_unpack = true) {
    constexpr int BLOCK_N = block_size_n();

    // [K, BLOCK_N] -> [K / 2, BLOCK_N * 2]
    const int ldb_tmp = BLOCK_N;

    if (do_unpack) {
      // group size 32 for mxfp4
      for (int k = 0; k < K; k += 32) {
        unpack_B(Btmp + k * ldb_tmp, B + k * (ldb >> 1), N, 32, ldb, ldb_tmp, scale + (k >> 5) * BLOCK_N);
      }
    }

    at::native::cpublas::brgemm(M, N, K, lda, ldb_tmp, BLOCK_N, /* add_C */ false, A, Btmp, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};

template <typename scalar_t, bool has_bias>
void tinygemm_kernel_impl(
    const scalar_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const uint8_t* __restrict__ scale,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K,
    bool do_unpack = true) {
  if (brg) {
    brgemm<scalar_t, has_bias>::apply(A, B, C, Btmp, Ctmp, bias, scale, M, N, K, lda, ldb, ldc, do_unpack);
    return;
  }

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
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", nb_size);
      }
    }
  }
}

template <typename scalar_t>
void mxfp4_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const uint8_t* __restrict__ mat2,
    const uint8_t* __restrict__ scales2,
    const float* __restrict__ bias,
    scalar_t* __restrict__ buffer,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    int64_t block_size_K,
    int64_t buffer_size_per_thread) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  const bool use_brgemm = can_use_brgemm<uint8_t>(M);
  const int64_t packed_K = get_row_size<uint8_t>(K);
  // mxfp4 block gemm scale shape: [N/BLOCK_N, K/32, BLOCK_N]
  const int64_t s_strideN = (K >> 5) * BLOCK_N;

  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
      int tid = get_thread_num();
      scalar_t* __restrict__ Btmp = buffer + tid * buffer_size_per_thread;
      float* __restrict__ Ctmp = (float*)((void*)(Btmp + MAX_CACHE_BLOCK_SIZE * BLOCK_N * K));

      loop_2d<uint8_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
        const uint8_t* scale_ptr = scales2 + nb * s_strideN;

        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        // only do unpacking for the first row
        bool do_unpack = (mb == mb0);

        tinygemm_kernel_impl<scalar_t, has_bias>(
            /*   A            */ mat1 + mb_start * mat1_strideM,
            /*   B            */ mat2 + nb_start * packed_K,
            /*   C            */ out + mb_start * out_strideM + nb_start,
            /*   Btmp         */ Btmp + nb_offset * BLOCK_N * K,
            /*   Ctmp         */ Ctmp,
            /*   scale        */ scale_ptr,
            /*   bias         */ bias + nb_start,
            /*   M            */ mb_size,
            /*   N            */ nb_size,
            /*   K            */ K,
            /*   lda          */ mat1_strideM,
            /*   ldb          */ nb_size,
            /*   ldc          */ out_strideM,
            /*   brg          */ use_brgemm,
            /*   block_size_K */ block_size_K,
            /*   do_unpack    */ do_unpack);
      });

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

}  // anonymous namespace

// tinygemm interface for mxfp4
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ Bbias,
    const uint8_t* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K,
    bool do_unpack) {
  if (Bbias != nullptr) {
    tinygemm_kernel_impl<scalar_t, true>(
        A, B, C, Btmp, Ctmp, scale, Bbias, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
    return;
  }
  tinygemm_kernel_impl<scalar_t, false>(
      A, B, C, Btmp, Ctmp, scale, nullptr, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
}

#define INSTANTIATE_TINYGEMM_MXFP4_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(            \
      const TYPE* __restrict__ A,                 \
      const uint8_t* __restrict__ B,              \
      TYPE* __restrict__ C,                       \
      TYPE* __restrict__ Btmp,                    \
      float* __restrict__ Ctmp,                   \
      const float* __restrict__ Bbias,            \
      const uint8_t* __restrict__ scale,          \
      int64_t M,                                  \
      int64_t N,                                  \
      int64_t K,                                  \
      int64_t lda,                                \
      int64_t ldb,                                \
      int64_t ldc,                                \
      bool brg,                                   \
      int64_t block_size_K,                       \
      bool do_unpack)

INSTANTIATE_TINYGEMM_MXFP4_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_MXFP4_TEMPLATE(at::Half);

// mat1 : [M, K] bfloat16
// mat2 : [N, K / 2] uint8, actual layout: [N / BLOCK_N, K / 2, BLOCK_N, 2]
// scales2: [N, K / G], actual layout: [N / BLOCK_N, K / G, BLOCK_N]
at::Tensor mxfp4_scaled_mm_cpu(
    at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2, const std::optional<at::Tensor>& bias, bool is_vnni) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat2.size(1) * 2;

  // mxfp4 supports only group size of 32 (2^5)
  constexpr int64_t group_size = 32;

  CHECK_EQ(mat1.size(1), K);
  CHECK_EQ(scales2.numel(), N * K >> 5);

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "mxfp4_scaled_mm_cpu: expect A to be bfloat16 or half.");
  TORCH_CHECK(mat2.scalar_type() == at::kByte, "mxfp4_scaled_mm_cpu: expect mat2 to be uint8.");
  TORCH_CHECK(scales2.scalar_type() == at::kByte, "mxfp4_scaled_mm_cpu: expect scales to be uint8.");
  auto out = at::empty({M, N}, mat1.options());

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  int num_threads = at::get_num_threads();
  int64_t size_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * K + BLOCK_M * BLOCK_N * 2;
  auto buffer = at::empty({num_threads, size_per_thread}, mat1.options());
  const float* bias_ptr = nullptr;
  if (bias.has_value()) {
    CHECK_EQ(bias.value().size(0), N);
    bias_ptr = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "mxfp4_scaled_mm_kernel_impl", [&] {
    mxfp4_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<uint8_t>(),
        scales2.data_ptr<uint8_t>(),
        bias_ptr,
        buffer.data_ptr<scalar_t>(),
        M,
        N,
        K,
        mat1.stride(0),
        out.stride(0),
        /* block_size_K */ group_size,
        buffer.size(-1));
  });

  return out;
}
