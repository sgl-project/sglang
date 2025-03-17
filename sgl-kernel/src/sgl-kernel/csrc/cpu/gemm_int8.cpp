#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {

// Adapted from https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
template <typename scalar_t>
void quant_A(
    uint8_t* __restrict__ Atmp, float* __restrict__ As,
    const scalar_t* __restrict__ A, int64_t M, int64_t K, int64_t lda,
    float eps = 1e-7) {

  for (int64_t m = 0; m < M; ++m) {
    float amax = 0.f; // absolute max
    for (int64_t k = 0; k < K; ++k) {
      const float val = static_cast<float>(A[m * lda + k]);
      amax = std::max(amax, std::abs(val));
    }

    amax = std::max(amax, eps);
    const float scale = amax / 127;
    const float inv_scale = 127 / amax;

    for (int64_t k = 0; k < K; ++k) {
      const float val = static_cast<float>(A[m * lda + k]) * inv_scale;
      Atmp[m * K + k] = (uint8_t)(std::round(val)) + 128;
    }
    As[m] = scale;
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
void quant_A<at::BFloat16>(
    uint8_t* __restrict__ Atmp, float* __restrict__ As,
    const at::BFloat16* __restrict__ A, int64_t M, int64_t K, int64_t lda,
    float eps) {

  const __m512 signBit = _mm512_set1_ps(-0.0f);
  const __m512i off = _mm512_set1_epi32(128);

  // K is 32x, no remainder
  for (int64_t m = 0; m < M; ++m) {
    float amax = 0.f;
    __m512 vamax0 = _mm512_set1_ps(0.f);
    __m512 vamax1 = _mm512_set1_ps(0.f);
    for (int64_t k = 0; k < K; k += 32) {
      __m512i va = _mm512_loadu_si512((void*)(A + m * lda + k));
      __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
      __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
      vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(signBit, va0));
      vamax1 = _mm512_max_ps(vamax1, _mm512_andnot_ps(signBit, va1));
    }
    amax = _mm512_reduce_max_ps(_mm512_max_ps(vamax0, vamax1));
    amax = std::max(amax, eps);
    const float scale = amax / 127;
    const float inv_scale = 127 / amax;
    const __m512 vd = _mm512_set1_ps(inv_scale);

    for (int64_t k = 0; k < K; k += 32) {
      __m512i va = _mm512_loadu_si512((void*)(A + m * lda + k));
      __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
      __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
      va0 = _mm512_mul_ps(va0, vd);
      va1 = _mm512_mul_ps(va1, vd);
      va0 = _mm512_roundscale_ps(va0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      va1 = _mm512_roundscale_ps(va1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i i0 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va0), off));
      __m128i i1 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va1), off));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(Atmp + m * K + k), _mm256_set_m128i(i1, i0));
    }
    As[m] = scale;
  }
}
#endif

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const uint8_t* __restrict__ A, const int8_t* __restrict__ B, scalar_t* __restrict__ C,
      const float* __restrict__ As, const float* __restrict__ Bs, const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const uint8_t* __restrict__ A, const int8_t* __restrict__ B, at::BFloat16* __restrict__ C,
      const float* __restrict__ As, const float* __restrict__ Bs, const int32_t* __restrict__ Bcomp,
      const float* __restrict__ bias, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512i va;
    __m512i vb[COLS];
    __m512i vc[ROWS * COLS];
    __m512i vcomp[COLS];
    __m512  vd0;
    __m512  vd1[COLS];

    // oops! 4x4 spills but luckly we use 4x2
    __m512 vbias[COLS];

    // [NOTE]: s8s8 igemm compensation in avx512-vnni
    //
    // avx512-vnni has no s8s8, so we need to change s8s8 to u8s8 with compensate:
    //
    //   a * b = (a + 128) * b - 128 * b
    //   s   s       u       s    u    s
    //
    // 1) 128 * b is pre-computed when packing B to vnni formats
    // 2) a + 128 is fused when dynamically quantize A
    //
    auto loadc = [&](auto i) {
      vc[i] = _mm512_set1_epi32(0);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K4 = K >> 2;
    const int64_t lda4 = lda >> 2;
    const int64_t ldb4 = ldb; // ldb * 4 >> 2;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A);
    const int32_t* b_ptr = reinterpret_cast<const int32_t*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
      }
      if constexpr (row == 0) {
        vb[col] = _mm512_loadu_si512(b_ptr + k * ldb4 + col * 16);
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb4 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K4; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      // load a scale
      if constexpr(col == 0) {
        vd0 = _mm512_set1_ps(As[row]);
      }
      // load b scale and vcomp per 2 vectors
      // also load bias if any
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          vd1[col + 0] = _mm512_loadu_ps(Bs + col * 16);
          vd1[col + 1] = _mm512_loadu_ps(Bs + col * 16 + 16);
          vcomp[col + 0] = _mm512_loadu_si512(Bcomp + col * 16);
          vcomp[col + 1] = _mm512_loadu_si512(Bcomp + col * 16 + 16);
          if constexpr (has_bias) {
            vbias[col + 0] = _mm512_loadu_ps(bias + col * 16);
            vbias[col + 1] = _mm512_loadu_ps(bias + col * 16 + 16);
          }
        }
      }

      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        __m512 vc0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[row * COLS + col + 0], vcomp[col + 0]));
        __m512 vc1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[row * COLS + col + 1], vcomp[col + 1]));
        if constexpr (has_bias) {
          vc0 = _mm512_fmadd_ps(_mm512_mul_ps(vc0, vd0), vd1[col + 0], vbias[col + 0]);
          vc1 = _mm512_fmadd_ps(_mm512_mul_ps(vc1, vd0), vd1[col + 1], vbias[col + 1]);
        } else {
          vc0 = _mm512_mul_ps(_mm512_mul_ps(vc0, vd0), vd1[col + 0]);
          vc1 = _mm512_mul_ps(_mm512_mul_ps(vc1, vd0), vd1[col + 1]);
        }

        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(vc1, vc0)));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply(         \
        Atmp + mb_start * lda, B + nb_start * 4, C + mb_start * ldc + nb_start, \
        As + mb_start, Bs + nb_start, Bcomp + nb_start,                      \
        has_bias ? bias + nb_start : nullptr, K, lda, ldb, ldc);

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    uint8_t* __restrict__ Atmp,
    int32_t* __restrict__ Ctmp,
    const float* __restrict__ Bs,
    const float* __restrict__ bias,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    bool brg) {

  // A scales
  float* As = reinterpret_cast<float*>(Atmp + M * K);
  // B compensation
  const int32_t* Bcomp = reinterpret_cast<const int32_t*>(B + block_size_n() * K);

  // quant A to int8 and add 128
  quant_A(Atmp, As, A, M, K, lda);

  // pattern: 1-4-16
  constexpr int BLOCK_M = 4;
  constexpr int BLOCK_N = 64;
  const int MB = div_up(M, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int mb_start = mb * BLOCK_M;
    int mb_size = std::min(BLOCK_M, M - mb_start);
    for (int nb = 0; nb < NB; ++nb) {
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);

      switch(mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 0x14: LAUNCH_TINYGEMM_KERNEL_NN(1, 64); break;
        // mb_size = 2
        case 0x22: LAUNCH_TINYGEMM_KERNEL_NN(2, 32); break;
        case 0x24: LAUNCH_TINYGEMM_KERNEL_NN(2, 64); break;
        // mb_size = 3
        case 0x32: LAUNCH_TINYGEMM_KERNEL_NN(3, 32); break;
        case 0x34: LAUNCH_TINYGEMM_KERNEL_NN(3, 64); break;
        // mb_size = 4
        case 0x42: LAUNCH_TINYGEMM_KERNEL_NN(4, 32); break;
        case 0x44: LAUNCH_TINYGEMM_KERNEL_NN(4, 64); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template<typename scalar_t>
void int8_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    uint8_t* __restrict__ buffer,
    int M,
    int N,
    int K,
    int mat1_strideM,
    int out_strideM) {

  constexpr int BLOCK_M = block_size_m();
  constexpr int BLOCK_N = block_size_n();
  const int MB = div_up(M, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);

  // TODO: brgemm u8s8 depends on PyTorch 2.7 release.
  const bool use_brgemm = false;

  // K + 4 after compensation
  const int packed_row_size = get_row_size<int8_t>(K);

  // {M * K : unit8_t} + {M : float32}
  const int buffer_size_per_thread = std::min(M, BLOCK_M) * (K + sizeof(float));

  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    at::parallel_for(0, MB * NB, 0, [&](int begin, int end) {
      int mb{0}, nb{0};
      data_index_init(begin, mb, MB, nb, NB);

      // get local pointers
      int tid = at::get_thread_num();

      // for dequant A
      uint8_t* __restrict__ Atmp = buffer + tid * buffer_size_per_thread;

      // for brgemm, use int32_t for accumulate
      alignas(64) int32_t Ctmp[BLOCK_M * BLOCK_N];

      for (int i = begin; i < end; ++i) {
        UNUSED(i);
        int mb_start = mb * BLOCK_M;
        int mb_size = std::min(M - mb_start, BLOCK_M);
        int nb_start = nb * BLOCK_N;
        int nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            /*   A */ mat1 + mb_start * mat1_strideM,
            /*   B */ mat2 + nb_start * packed_row_size /* nb * BLOCK_N * (K + 4) */,
            /*   C */ out + mb_start * out_strideM + nb_start,
            /* Atmp*/ Atmp,
            /* Ctmp*/ Ctmp,
            /*  Bs */ scales + nb_start,
            /* bias*/ bias + nb_start,
            /*   M */ mb_size,
            /*   N */ nb_size,
            /*   K */ K,
            /* lda */ mat1_strideM,
            /* ldb */ nb_size,
            /* ldc */ out_strideM,
            /* brg */ use_brgemm);

        // move to the next index
        data_index_step(mb, MB, nb, NB);
      }
 
      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

} // anonymous namespace

// weight     :  static, per-channel, symmetric
// activation : dynamic,   per-token, symmetric
//
// mat1   : [M, K]
// mat2   : [N, K]
// scales : [N]
// bias   : [N]
// out    : [M, N]
//
at::Tensor int8_scaled_mm_cpu(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales,
    std::optional<at::Tensor>& bias, bool is_vnni) {

  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  int M = mat1.size(0);
  int N = mat2.size(0);
  int K = mat1.size(1);

  // see [NOTE]: s8s8 igemm compensation in avx512-vnni
  CHECK_EQ(mat2.size(1), (int64_t)(is_vnni ? K + sizeof(int32_t) : K));
  CHECK_EQ(scales.numel(), N);

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf,
      "int8_scaled_mm: expect mat1 to be bfloat16 or half.");
  TORCH_CHECK(mat2.scalar_type() == at::kChar,
      "int8_scaled_mm: expect mat2 to be int8.");
  TORCH_CHECK(scales.scalar_type() == at::kFloat,
      "int8_scaled_mm: expect scales to be float32.");

  auto out = at::empty({M, N}, mat1.options());

  // strides
  int mat1_strideM = mat1.stride(0);
  int out_strideM = out.stride(0);

  // temp buffer for dynamically quantize A
  int num_threads = at::get_num_threads();
  int size_per_thread = std::min(M, block_size_m()) * (K + sizeof(float));
  auto buffer = at::empty({num_threads, size_per_thread}, mat1.options().dtype(at::kByte));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "int8_scaled_mm_kernel_impl", [&] {
    int8_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        bias_data,
        buffer.data_ptr<uint8_t>(),
        M,
        N,
        K,
        mat1_strideM,
        out_strideM);
  });

  return out;
}
