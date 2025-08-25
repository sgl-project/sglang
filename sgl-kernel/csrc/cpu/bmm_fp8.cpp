#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}
template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int size, float scale) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec vscale = fVec(scale);

  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * vscale;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * vscale;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * scale);
  }
}

template <typename scalar_t, typename packed_t, int BLOCK_M, int BLOCK_N>
struct fp8_tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      float scale,
      int K,
      int lda,
      int ldb,
      int ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)

template <int BLOCK_M, int BLOCK_N>
struct fp8_tinygemm_kernel_nn<at::BFloat16, at::Float8_e4m3fn, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      float scale,
      int K,
      int lda,
      int ldb,
      int ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    const __m512 vscale = _mm512_set1_ps(scale);

    auto loadc = [&](auto i) { vc[i] = _mm512_set1_ps(0.f); };
    Unroll<ROWS * COLS>{}(loadc);

    const int K2 = K >> 1;
    const int lda2 = lda >> 1;
    const int ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + col * 16);
          if constexpr (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          __m256i b8_0 = _mm512_extracti32x8_epi32(b8, 0);
          __m256i b8_1 = _mm512_extracti32x8_epi32(b8, 1);
          __m512bh bf16_0 = CVT_FP8_TO_BF16(b8_0);
          __m512bh bf16_1 = CVT_FP8_TO_BF16(b8_1);
          vb[col + 0] = bf16_0;
          vb[col + 1] = bf16_1;
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        __m512 vc0 = _mm512_mul_ps(vc[row * COLS + col + 0], vscale);
        __m512 vc1 = _mm512_mul_ps(vc[row * COLS + col + 1], vscale);
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)), (__m512i)(_mm512_cvtne2ps_pbh(vc1, vc0)));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_FP8TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                 \
  fp8_tinygemm_kernel_nn<scalar_t, packed_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start, scale, K, lda, ldb, ldc);

template <typename scalar_t, typename packed_t>
struct fp8_brgemm {};

template <typename scalar_t>
struct fp8_brgemm<scalar_t, scalar_t> {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      float scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    UNUSED(scale);

    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
    }
  }
};

inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int N,
    int K,
    int ldb,
    int ldb_tmp) {
  // [K/2, N, 2]
  const int K2 = K >> 1;
  const int ldb2 = ldb;  // ldb * 2 >> 1;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(packed_B);

  for (int k = 0; k < K2; ++k) {
    for (int n = 0; n < N; n += 64) {
      __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + n);

      __m256i b8_0 = _mm512_extracti32x8_epi32(b8, 0);
      __m256i b8_1 = _mm512_extracti32x8_epi32(b8, 1);

      __m512bh bf16_0 = CVT_FP8_TO_BF16(b8_0);
      __m512bh bf16_1 = CVT_FP8_TO_BF16(b8_1);
      _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n * 2 + 0, (__m512i)bf16_0);
      _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n * 2 + 32, (__m512i)bf16_1);
    }
  }
}
template <>
struct fp8_brgemm<at::BFloat16, at::Float8_e4m3fn> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      float scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    constexpr int BLOCK_N = block_size_n();

    // [BLOCK_K, BLOCK_N] -> [BLOCK_K / 2, BLOCK_N * 2]
    const int ldb_tmp = block_size_n();

    // accumulate across K per BLOCK_K
    for (int k = 0; k < K; k += BLOCK_K) {
      int kb_size = std::min(BLOCK_K, K - k);
      unpack_B(Btmp, B + k * ldb, N, kb_size, ldb, ldb_tmp);

      const bool add_C = (k != 0);
      at::native::cpublas::brgemm(M, N, kb_size, lda, ldb_tmp, BLOCK_N, add_C, A + k, Btmp, Ctmp);
    }

    // copy from Ctmp to C and mul scale
    for (int m = 0; m < M; ++m) {
      copy_mul_stub(C + m * ldc, Ctmp + m * BLOCK_N, N, scale);
    }
  }
};

template <typename scalar_t, typename packed_t>
void fp8_tinygemm_kernel_impl(
    const scalar_t* __restrict__ A,
    const packed_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    float scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  if (brg) {
    fp8_brgemm<scalar_t, packed_t>::apply(A, B, C, Btmp, Ctmp, scale, M, N, K, lda, ldb, ldc);
    return;
  }

  // pattern: 1-8-8
  if (M == 1) {
    constexpr int64_t BLOCK_N = 128;
    const int64_t NB = div_up(N, BLOCK_N);
    int64_t mb_start = 0;

    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (nb_size >> 4) {
        case 2:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 4:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(1, 64);
          break;
        case 6:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(1, 96);
          break;
        case 8:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(1, 128);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, 1x", "nb_size");
      }
    }
    return;
  }

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x14:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(1, 64);
          break;
        // mb_size = 2
        case 0x22:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x24:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(2, 64);
          break;
        // mb_size = 3
        case 0x32:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x34:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(3, 64);
          break;
        // mb_size = 4
        case 0x42:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(4, 32);
          break;
        case 0x44:
          LAUNCH_FP8TINYGEMM_KERNEL_NN(4, 64);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}
}  // namespace

template <typename scalar_t, typename packed_t>
void fp8_tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const packed_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    float scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  fp8_tinygemm_kernel_impl<scalar_t, packed_t>(A, B, C, Btmp, Ctmp, scale, M, N, K, lda, ldb, ldc, brg);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE2(TYPE1, TYPE2) \
  template void fp8_tinygemm_kernel<TYPE1, TYPE2>(   \
      const TYPE1* __restrict__ A,                   \
      const TYPE2* __restrict__ B,                   \
      TYPE1* __restrict__ C,                         \
      TYPE1* __restrict__ Btmp,                      \
      float* __restrict__ Ctmp,                      \
      float scale,                                   \
      int64_t M,                                     \
      int64_t N,                                     \
      int64_t K,                                     \
      int64_t lda,                                   \
      int64_t ldb,                                   \
      int64_t ldc,                                   \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE2(at::BFloat16, at::Float8_e4m3fn);
