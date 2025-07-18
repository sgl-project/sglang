#include "vec.h"
#include <torch/all.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <c10/util/Unroll.h>

namespace {

#define BLOCK_N 32

static bool cpublas_checked = false;
static bool cpublas_can_pack = false;

bool cpublas_could_pack() {
  // the could_pack check requires AMX support implicitly
  if (cpublas_checked) {
    return cpublas_can_pack;
  }
  cpublas_can_pack = at::native::cpublas::could_pack(at::kByte);
  cpublas_checked = true;
  return cpublas_can_pack;
}

template<bool sym_quant_a>
struct ActDtype;
template<>
struct ActDtype<true> {
  using type = int8_t;
};

template<>
struct ActDtype<false> {
  using type = uint8_t;
};


#if defined(CPU_CAPABILITY_AVX512)
inline std::array<__m256i, 2> load_zps_4vnni(const int8_t* __restrict__ zps) {
  // broadcast 01234567 to
  // 01234567012345670123456701234567
  __m256i vzps_low = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps));
  __m256i vzps_high = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps + 8));
  // shuffle from
  // 01234567012345670123456701234567
  // to
  // 00001111222233334444555566667777
  __m256i shuffle_mask = _mm256_set_epi8(
      7,
      7,
      7,
      7,
      6,
      6,
      6,
      6,
      5,
      5,
      5,
      5,
      4,
      4,
      4,
      4,
      3,
      3,
      3,
      3,
      2,
      2,
      2,
      2,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0);
  vzps_low = _mm256_shuffle_epi8(vzps_low, shuffle_mask);
  vzps_high = _mm256_shuffle_epi8(vzps_high, shuffle_mask);
  return {vzps_low, vzps_high};
}

inline std::array<__m256i, 2> load_uint4_as_int8(const uint8_t* __restrict__ qB) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qB));
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i high = _mm256_srli_epi16(packed, 4);
  high = _mm256_and_si256(high, low_mask);
  __m256i low = _mm256_and_si256(packed, low_mask);
  return {low, high};
}

template <int64_t N, int64_t ldb>
void _dequant_weight_zp_only(
    const uint8_t* __restrict__ B,
    int8_t* dqB,
    const int8_t* __restrict__ qzeros,
    int64_t K) {
  // unpack weight int8 -> two int4
  // subtract zero point
  // B shape = [K, ldb] = [K, N / 2], actual shape = [K / 4, N / 2, 4]
  // dqB shape = [K, N], actual shape = [K / 4, N, 4]
#pragma GCC unroll 2
  for (int n = 0; n < N; n += 16) {
    auto [zps_low, zps_high] = load_zps_4vnni(&qzeros[n]);
    for (int k = 0; k < K; k += 4) {
      auto [vb_low, vb_high] = load_uint4_as_int8(B + ldb * k + n / 2 * 4);
      vb_high = _mm256_sub_epi8(vb_high, zps_high);
      vb_low = _mm256_sub_epi8(vb_low, zps_low);
      // store vb to B
      _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(dqB + N * k + n * 4), vb_low);
      _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(dqB + N * k + (n + 8) * 4), vb_high);
    }
  }
}

template <bool accum, int64_t N, bool sym_quant_a>
void _dequant_and_store(
    float* __restrict__ output,
    const int32_t* __restrict__ input,
    const float* __restrict__ scale_a,
    const int32_t* __restrict__ zp_a,
    const float* __restrict__ scale_b,
    const int32_t* __restrict__ comp_b,
    int M,
    int ldi,
    int ldo,
    int ldsa = 1) {
  for (int m = 0; m < M; ++m) {
    float a_scale = *(scale_a + m * ldsa);
    __m512 va_scale = _mm512_set1_ps(a_scale);
    int32_t a_zp;
    __m512i va_zp;
    if constexpr (!sym_quant_a) {
      a_zp = *(zp_a + m * ldsa);
      va_zp = _mm512_set1_epi32(a_zp);
    }
    int n = 0;
#pragma GCC unroll 2
    for (; n < N; n += 16) {
      __m512i vc = _mm512_loadu_si512(input + m * ldi + n);
      if constexpr (!sym_quant_a) {
        __m512i vb_comp = _mm512_loadu_si512(comp_b + n);
        vc = _mm512_sub_epi32(vc, _mm512_mullo_epi32(vb_comp, va_zp));
      }
      __m512 vc_f = _mm512_cvtepi32_ps(vc);
      __m512 vc_f_mul = _mm512_mul_ps(vc_f, va_scale);
      __m512 vb_s = _mm512_loadu_ps(scale_b + n);
      vc_f_mul = _mm512_mul_ps(vc_f_mul, vb_s);
      if constexpr (accum) {
        __m512 vo = _mm512_loadu_ps(output + m * ldo + n);
        _mm512_storeu_ps(output + m * ldo + n, _mm512_add_ps(vo, vc_f_mul));
      } else {
        _mm512_storeu_ps(output + m * ldo + n, vc_f_mul);
      }
    }
    for (; n < N; ++n) {
      float dq_val;
      if constexpr (sym_quant_a) {
        dq_val = (float)input[m * ldi + n] * a_scale * scale_b[n];
      } else {
        dq_val =
          (float)(input[m * ldi + n] - a_zp * comp_b[n]) * a_scale * scale_b[n];
      }
      if constexpr (accum) {
        output[m * ldo + n] += dq_val;
      } else {
        output[m * ldo + n] = dq_val;
      }
    }
  }
}

#else
template<int64_t N, int64_t ldb>
void _dequant_weight_zp_only(
    const uint8_t* B,
    int8_t* dqB,
    const int8_t* qzeros,
    int64_t K) {
  // B shape = [K, N / 2]
  // dqB shape = [K, N]
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N / 2; ++n) {
      int32_t b = (int32_t)B[k * ldb + n];
      dqB[k * N + n * 2] = (b & 0xf) - qzeros[n];
      dqB[k * N + n * 2 + 1] = (b >> 4) - qzeros[n];
    }
  }
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
inline __m512i combine_m256i(__m256i a, __m256i b) {
  __m512i c = _mm512_castsi256_si512(a);
  return _mm512_inserti64x4(c, b, 1);
}

inline __m512i combine_m256i(std::array<__m256i, 2> two_256) {
  return combine_m256i(two_256[0], two_256[1]);
}

// negate elements in a according to b's sign
static inline __m512i _mm512_sign_epi8(__m512i a, __m512i b) {
  __m512i zero = _mm512_setzero_si512();
  __mmask64 blt0 = _mm512_movepi8_mask(b);
  return _mm512_mask_sub_epi8(a, blt0, zero, a);
}

template <int64_t M, int64_t N, int64_t ldb, bool sym_quant_a>
void _dequant_gemm_accum_small_M(
    float* __restrict__ C,
    const uint8_t* A,
    const float* scales_a,
    const int32_t* qzeros_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  // if sym_quant_a is true, A pointer type is passed in as uint8_t* but actually int8_t*.

  constexpr int COLS = N / 16;
  // Computing compensation is faster than loading it for small M
  // because it's memory bound.
  __m512i ones = _mm512_set1_epi8(1); // used for computing compensation
  __m512i va;
  __m512i vb[COLS];
  __m512i vc[M * COLS];
  __m512 vscales[COLS];
  __m512i vzps[COLS];
  __m512i vcompensate[COLS];

  // Load scales and zps
  c10::ForcedUnroll<COLS>{}([&](auto i) {
      vscales[i] = _mm512_loadu_ps(scales_b + i * 16);
      vzps[i] = combine_m256i(load_zps_4vnni(qzeros_b + i * 16));
      if constexpr (!sym_quant_a) {
        vcompensate[i] = _mm512_setzero_epi32();
      }
    });
  c10::ForcedUnroll<M * COLS>{}(
      [&](auto i) { vc[i] = _mm512_setzero_epi32(); });

  auto compute = [&](auto i, int k) {
    constexpr const int row = i / COLS;
    constexpr const int col = i % COLS;

    if constexpr (col == 0) {
      va = _mm512_set1_epi32(*(int32_t*)(A + row * lda + k));
    }

    if constexpr (row == 0) {
      int B_offset = k * ldb + col * 16 * 2;
      vb[col] = combine_m256i(load_uint4_as_int8(B + B_offset));
      vb[col] = _mm512_sub_epi8(vb[col], vzps[col]);
      if constexpr (!sym_quant_a) {
          vcompensate[col] =
              _mm512_dpbusd_epi32(vcompensate[col], ones, vb[col]);
      }
      _mm_prefetch(B + B_offset + 128 * ldb, _MM_HINT_T0);
    }
    if constexpr (sym_quant_a) {
        auto vsb = _mm512_sign_epi8(vb[col], va);
        auto vabsa = _mm512_sign_epi8(va, va);
        vc[i] = _mm512_dpbusds_epi32(vc[i], vabsa, vsb);
    } else {
      vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    }
  };

  // Accumulate along k
  constexpr const int unroll = 4;
  int k = 0;
  for (; k < K / 4 / unroll; k++) {
    c10::ForcedUnroll<unroll>{}([&](auto i) {
      c10::ForcedUnroll<M * COLS>{}(compute, 4 * (k * unroll + i));
    });
  }
  k *= 4 * unroll;
  for (; k < K; k += 4) {
    c10::ForcedUnroll<M * COLS>{}(compute, k);
  }

  // Store to C
  auto store = [&](auto i) {
    constexpr const int row = i / COLS;
    constexpr const int col = i % COLS;
    // compute (qC - compensate * zp_a) * scale_a * scale_b
    __m512 vc_float;
    if constexpr (!sym_quant_a) {
      vc[i] = _mm512_sub_epi32(
          vc[i],
          _mm512_mullo_epi32(
              vcompensate[col], _mm512_set1_epi32(*(qzeros_a + row))));
    }
    vc_float = _mm512_cvtepi32_ps(vc[i]);
    vc_float = _mm512_mul_ps(vc_float, _mm512_set1_ps(*(scales_a + row)));

    vc_float = _mm512_mul_ps(vc_float, vscales[col]);
    auto vc_old = _mm512_loadu_ps(C + row * ldc + col * 16);
    vc_float = _mm512_add_ps(vc_float, vc_old);
    _mm512_storeu_ps(C + row * ldc + col * 16, vc_float);
  };
  c10::ForcedUnroll<M * COLS>{}(store);

}

#define call_dequant_gemm_accum_small_M(M) \
  _dequant_gemm_accum_small_M<M, N, ldb, sym_quant_a>( \
      C, \
      A, \
      scales_a, \
      qzeros_a, \
      B, \
      scales_b, \
      qzeros_b, \
      K, \
      lda, \
      ldc);
#endif

template <bool cpublas_can_pack, int64_t N, int64_t ldb, bool sym_quant_a>
void _dequant_gemm_accum(
    float* C,
    const uint8_t* A,
    const float* scales_a,
    const int32_t* qzeros_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    const int32_t* compensation,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  // Compute GEMM int8 * int8 -> int32
  // dequant result to float by applying scales/qzeros
#if defined(CPU_CAPABILITY_AVX512)
  if (M <= 4 && cpublas_can_pack) {
    switch (M) {
      case 1:
        call_dequant_gemm_accum_small_M(1);
        return;
      case 2:
        call_dequant_gemm_accum_small_M(2);
        return;
      case 3:
        call_dequant_gemm_accum_small_M(3);
        return;
      case 4:
        call_dequant_gemm_accum_small_M(4);
        return;
    }
  }
#endif

  int8_t dqB[K * N];
  _dequant_weight_zp_only<N, ldb>(B, dqB, qzeros_b, K);
  using Tin = typename ActDtype<sym_quant_a>::type;
  Tin* A_ptr = (Tin*)A;
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (cpublas_can_pack) {
    int32_t C_i32[M * N];
    at::native::cpublas::brgemm(
        M,
        N,
        K,
        lda,
        N /*ldb*/,
        N /*ldc*/,
        false /* add_C */,
        A_ptr,
        dqB,
        C_i32,
        true /* is_vnni */);
    _mm_prefetch(B + N * K / 2, _MM_HINT_T0);
    _mm_prefetch(A + K, _MM_HINT_T0);
    _dequant_and_store<true, N, sym_quant_a>(
        C,
        C_i32,
        scales_a,
        qzeros_a,
        scales_b,
        compensation,
        M,
        N /*ldi*/,
        ldc,
        1 /*ldsa*/);
  } else
#endif
  {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0;
        for (int64_t k = 0; k < K; ++k) {
          if constexpr (sym_quant_a) {
            sum += ((int32_t)A_ptr[i * lda + k] * dqB[k * N + j]);
          } else {
            sum += ((int32_t)A_ptr[i * lda + k] - qzeros_a[i]) * (int32_t)dqB[k * N + j];
          }
        }
        C[i * ldc + j] += sum * scales_a[i] * scales_b[j];
      }
    }
  }
}

template<int64_t N>
inline void copy_bias(const float* bias_ptr, float* y_buf, int64_t m) {
  if (bias_ptr) {
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 bias_vec = _mm512_loadu_ps(bias_ptr + j);
        _mm512_storeu_ps(y_buf + i * N + j, bias_vec);
      }
#endif
      for (; j < N; ++j) {
        y_buf[i * N + j] = bias_ptr[j];
      }
    }
  } else { // initialize to zero
    for (int i = 0; i < m; ++i) {
      int j = 0;
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 zero_vec = _mm512_setzero_ps();
        _mm512_storeu_ps(y_buf + i * N + j, zero_vec);
      }
#endif
      for (; j < N; ++j) {
        y_buf[i * N + j] = 0;
      }
    }
  }
}

template<typename out_dtype, int64_t N>
inline void store_out(const float* y_buf, out_dtype* c_ptr, int64_t m, /* int64_t n, */ int64_t lda) {
  for (int i = 0; i < m; ++i) {
    int j = 0;
    if constexpr (std::is_same<out_dtype, float>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        _mm512_storeu_ps(c_ptr + i * lda + j, y_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = y_buf[i * N + j];
      }
    } else if constexpr (std::is_same<out_dtype, at::BFloat16>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        __m256i y_bf16_vec = at::vec::cvtfp32_bf16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_bf16_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = at::BFloat16(y_buf[i * N + j]);
      }
    } else if constexpr (std::is_same<out_dtype, at::Half>::value) {
#if defined(CPU_CAPABILITY_AVX512)
#pragma GCC unroll 2
      for (; j < N; j += 16) {
        __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
        __m256i y_fp16_vec = at::vec::cvtfp32_fp16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_fp16_vec);
      }
#endif
      for (; j < N; ++j) {
        c_ptr[i * lda + j] = at::Half(y_buf[i * N + j]);
      }
    } else {
      TORCH_CHECK(false, "Unsupported output dtype");
    }
  }
}


// https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
template <typename scalar_t>
inline void quantize_row_int8_sym(int8_t* __restrict__ Aq, float& As,
    const scalar_t* __restrict__ A, int64_t K, float eps = 1e-7) {

  float amax = 0.f; // absolute max
  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]);
    amax = std::max(amax, std::abs(val));
  }

  amax = std::max(amax, eps);
  const float scale = amax / 127;
  const float inv_scale = 127 / amax;

  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]) * inv_scale;
    Aq[k] = (int8_t)(std::round(val));
  }
  As = scale;
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void quantize_row_int8_sym<at::BFloat16>(int8_t* __restrict__ Aq, float& As,
    const at::BFloat16* __restrict__ A, int64_t K, float eps) {

  const __m512 signBit = _mm512_set1_ps(-0.0f);
  // K is 32x, no remainder
  float amax = 0.f;
  __m512 vamax0 = _mm512_set1_ps(0.f);
  __m512 vamax1 = _mm512_set1_ps(0.f);
  for (int64_t k = 0; k < K; k += 32) {
    __m512i va = _mm512_loadu_si512((void*)(A + k));
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
    __m512i va = _mm512_loadu_si512((void*)(A + k));
    __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
    __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
    va0 = _mm512_mul_ps(va0, vd);
    va1 = _mm512_mul_ps(va1, vd);
    va0 = _mm512_roundscale_ps(va0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    va1 = _mm512_roundscale_ps(va1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m128i i0 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(va0));
    __m128i i1 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(va1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Aq + k), _mm256_set_m128i(i1, i0));
  }
  As = scale;
}
#endif

std::tuple<at::Tensor, at::Tensor> per_token_quant_int8_cpu_sym(at::Tensor& A) {
  RECORD_FUNCTION("sgl-kernel::per_token_quant_int8_cpu", std::vector<c10::IValue>({A}));

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t lda = A.stride(0);

  const auto st = A.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf,
      "per_token_quant_int8: expect A to be bfloat16 or half.");

  auto Aq = at::empty({M, K}, A.options().dtype(c10::kChar));
  auto As = at::empty({M}, A.options().dtype(at::kFloat));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "per_token_quant_int8", [&] {
    int8_t* __restrict__ Aq_data = Aq.data_ptr<int8_t>();
    float* __restrict__ As_data = As.data_ptr<float>();
    const scalar_t* __restrict__ A_data = A.data_ptr<scalar_t>();

    at::parallel_for(0, M, 0, [&] (int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        quantize_row_int8_sym<scalar_t>(
            Aq_data + m * K,
            As_data[m],
            A_data + m * lda,
            K);
      }
    });
  });
  return std::make_tuple(Aq, As);
}

template<typename out_dtype, bool cpublas_can_pack, bool sym_quant_a>
void _da8w4_linear_impl(
    const at::Tensor& input,
    const at::Tensor& input_scales,
    const at::Tensor& input_qzeros,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_qzeros,
    const at::Tensor& compensation,
    const std::optional<at::Tensor>& bias,
    at::Tensor& output) {
  // input shape = [..., K]
  // input is per token quantized
  int64_t K = input.size(-1);
  auto input_view = input.view({-1, K});
  int64_t M = input_view.size(0);
  TORCH_CHECK(input_scales.numel() == M, "DA8W4: unexpected input scales shape");
  if(not sym_quant_a){
    TORCH_CHECK(input_scales.sizes() == input_qzeros.sizes(), "DA8W4: unexpected input qzeros shape");
  }

  // weight shape = [Nc, Kc, block_k, block_n/2]
  // scales/qzeros shape = [Nc, G, block_n]
  // compensation shape = [Nc, Kc, block_n]
  int64_t Nc = weight.size(0);
  int64_t Kc = weight.size(1);
  int64_t block_k = weight.size(2);
  constexpr int64_t block_n = BLOCK_N;
  TORCH_CHECK(weight.size(3) * 2 == block_n, "DA8W4: unexpected weight shape");
  int64_t N = Nc * block_n;
  TORCH_CHECK(K == Kc * block_k, "DA8W4: weight and input shapes mismatch");
  int64_t block_m = [&]() -> long {
    if (M <= 48) {
      return M;
    } else if (M < 64) {
      return 32;
    } else if (M < 96) {
      return 48;
    } else {
      return 64;
    }
  }();
  int64_t Mc = (M + block_m - 1) / block_m;
  bool parallel_on_M = M > 128;
  int64_t num_blocks = parallel_on_M ? Mc * Nc : Nc;

  // scales/qzeros shape = [Nc, G, block_n]
  int64_t num_groups = weight_scales.size(1);
  int64_t group_size = K / num_groups;
  TORCH_CHECK(group_size % block_k == 0,
              "DA8W4 CPU: group_size should be divisible by block_k");
  int64_t block_per_group = group_size / block_k;

  using Tin = typename ActDtype<sym_quant_a>::type;
  const Tin* a_ptr = input_view.data_ptr<Tin>();
  const float* a_scales_ptr = input_scales.data_ptr<float>();
  const int32_t* a_qzeros_ptr = sym_quant_a ? nullptr : input_qzeros.data_ptr<int32_t>();
  const uint8_t* b_ptr = weight.data_ptr<uint8_t>();
  const float* b_scales_ptr = weight_scales.data_ptr<float>();
  const int8_t* b_qzeros_ptr = weight_qzeros.data_ptr<int8_t>();
  const int32_t* compensation_ptr = sym_quant_a ? nullptr : compensation.data_ptr<int32_t>();
  out_dtype* c_ptr = output.data_ptr<out_dtype>();
  const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

  at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t mc = parallel_on_M ? i / Nc : 0;
      int64_t nc = parallel_on_M ? i % Nc : i;
      int64_t mc_end = parallel_on_M ? mc + 1 : Mc;

      for (int mci = mc; mci < mc_end; ++mci) {
        int64_t m_size = mci * block_m + block_m > M ? M - mci * block_m : block_m;
        alignas(64) float y_buf[m_size][block_n];
        // copy bias to y_buf if bias is not None
        auto bias_data = bias_ptr ? bias_ptr + nc * block_n : nullptr;
        copy_bias<block_n>(bias_data, y_buf[0], m_size);
        for (int kci = 0; kci < Kc; ++kci) {
          _dequant_gemm_accum<cpublas_can_pack, block_n, block_n / 2, sym_quant_a>(
            y_buf[0] /*C*/,
            (uint8_t*)a_ptr + mci * block_m * K + kci * block_k /*A*/,
            a_scales_ptr + mci * block_m /*scales_a*/,
            a_qzeros_ptr + mci * block_m /*qzeros_a*/,
            b_ptr + (nc * Kc + kci) * block_n * block_k / 2 /*B*/,
            b_scales_ptr + nc * block_n * num_groups + kci / block_per_group * block_n /*scales_b*/,
            b_qzeros_ptr + nc * block_n * num_groups + kci / block_per_group * block_n /*qzeros_b*/,
            compensation_ptr + nc * block_n * Kc + kci * block_n /*compensation*/,
            m_size /*M*/,
            block_k /*K*/,
            K /*lda*/,
            block_n /*ldc*/);
        }
        // store y_buf to output with dtype conversion
        store_out<out_dtype, block_n>(
          y_buf[0],
          c_ptr + mci * block_m * N + nc * block_n,
          m_size,
          N /*lda*/);
      }
    }
    if constexpr (cpublas_can_pack) {
      at::native::cpublas::brgemm_release();
    }
  });
}



} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("per_token_quant_int8_cpu_sym(Tensor act) -> (Tensor,Tensor)");
  m.impl("per_token_quant_int8_cpu_sym", torch::kCPU, &per_token_quant_int8_cpu_sym); 
}

/*
return: packed_weight, packed_scales, packed_qzeros, compensation
*/
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
da8w4_linear_prepack_cpu(
    const at::Tensor& weight,
    const at::Tensor& scales,
    const at::Tensor& qzeros) {
  // weight shape = [N, K]
  // scales shape = [N, G]
  // qzeros shape = [N, G]
  TORCH_CHECK(weight.dim() == 2,
              "DA8W4 CPU: Weight should be a 2D tensor for packing");
  TORCH_CHECK(weight.size(1) % 2 == 0,
              "DA8W4 CPU: Weight should have even number of columns for packing");

  auto new_scales = scales;
  auto new_qzeros = qzeros;
  if (new_scales.dim() == 1) {
    new_scales.unsqueeze_(1);
  }
  new_scales = new_scales.to(at::kFloat);
  if (new_qzeros.dim() == 1) {
    new_qzeros.unsqueeze_(1);
  }
  new_qzeros = new_qzeros.to(at::kChar);
  int N = weight.size(0);
  int K = weight.size(1);
  int G = scales.size(1);
  int group_size = K / G;
  int block_k = group_size > 128 ? 128 : group_size;
  constexpr int block_n = BLOCK_N;
  int Nc = N / block_n;
  int Kc = K / block_k;

  // Reorder weight to [N/block_n, K/block_k, block_k, block_n]
  // Reorder scales/qzeros to [N/block_n, G, block_n]
  auto weight_view = weight.view({Nc, block_n, Kc, block_k});
  at::Tensor weight_reordered = weight_view.permute({0, 2, 3, 1}).contiguous();
  at::Tensor blocked_weight;
  at::Tensor blocked_scales = new_scales.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();
  at::Tensor blocked_qzeros = new_qzeros.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();
  // Compensation = Σ(k)(W[k][n] - ZP[n]) for each block.
  auto weight_sub_qzero = weight.view({Nc, block_n, G, -1}).to(at::kInt) - new_qzeros.view({Nc, block_n, G, -1});
  weight_sub_qzero = weight_sub_qzero.view({Nc, block_n, Kc, block_k});
  at::Tensor compensation = weight_sub_qzero.sum(-1);
  compensation = compensation.permute({0, 2, 1}).contiguous().to(at::kInt);

  if (cpublas_could_pack()) {
    blocked_weight = at::empty({Nc, Kc, block_k, block_n / 2}, weight.options());
    auto weight_ptr = weight_reordered.data_ptr<uint8_t>();
    auto blocked_weight_ptr = blocked_weight.data_ptr<uint8_t>();
    int64_t num_blocks = Nc * Kc;
    at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        auto in_ptr = weight_ptr + i * block_k * block_n;
        auto out_ptr = blocked_weight_ptr + i * block_k * block_n / 2;

        // Reorder weight block to VNNI4 and pack two lanes along N
        // N=16 viewed as two lanes: a0, ...a7, b0, ...b7
        // pack two lanes: [a0, b0], ..., [a7, b7]
        // plain shape = [block_k, block_n]
        // packed shape = [block_k / 4, block_n / 2, 4] viewed as [block_k, block_n / 2]
        constexpr int n_group_size = 8;
        constexpr int vnni_size = 4;
        constexpr int n_group = block_n / n_group_size; // 4
        for (int nb = 0; nb < n_group; nb += 2) {
          for (int k = 0; k < block_k; k += vnni_size) {
            for (int ni = 0; ni < n_group_size; ++ni) {
              for (int ki = 0; ki < vnni_size; ++ki) {
                int src_idx_1 = nb * n_group_size + ni + (k + ki) * block_n;
                int src_idx_2 = (nb + 1) * n_group_size + ni + (k + ki) * block_n;
                int dst_idx = (nb / 2 * n_group_size + ni) * vnni_size + k * block_n / 2 + ki;
                uint8_t src_1 = *(in_ptr + src_idx_1);
                uint8_t src_2 = *(in_ptr + src_idx_2);
                uint8_t dst = (src_1 & 0x0f) | ((src_2 & 0x0f) << 4);
                *(out_ptr + dst_idx) = dst;
              }
            }
          }
        }
      }
    });
  } else {
    // Pack weight: two int4 -> one int8
    using namespace at::indexing;
    at::Tensor even_columns =
        weight_reordered.index({Slice(), Slice(), Slice(), Slice(1, None, 2)});
    even_columns = even_columns.bitwise_left_shift(4);
    at::Tensor odd_columns =
        weight_reordered.index({Slice(), Slice(), Slice(), Slice(None, None, 2)});
    blocked_weight = even_columns.bitwise_or(odd_columns);
  }

  return std::make_tuple(std::move(blocked_weight), std::move(blocked_scales), std::move(blocked_qzeros), std::move(compensation));
}

at::Tensor da8w4_linear_cpu(
  const at::Tensor& input,
  const at::Tensor& input_scales,
  const at::Tensor& input_qzeros,
  const at::Tensor& weight,
  const at::Tensor& weight_scales,
  const at::Tensor& weight_qzeros,
  const at::Tensor& compensation,
  const std::optional<at::Tensor>& bias,
  at::ScalarType output_dtype) {
    RECORD_FUNCTION(
      "sgl-kernel::da8w4_linear_cpu", std::vector<c10::IValue>({input, weight}));
static bool cpublas_can_pack = cpublas_could_pack();
bool sym_quant_a = input.scalar_type() == c10::kChar;
auto out_sizes = input.sizes().vec();
int64_t N = weight.size(0) * weight.size(-1) * 2;
out_sizes.back() = N;
auto output = at::empty(out_sizes, input.options().dtype(output_dtype));

#define call__da8w4_linear_impl(cpublas_can_pack, sym_quant_act) \
  AT_DISPATCH_FLOATING_TYPES_AND2( \
      at::ScalarType::BFloat16, at::ScalarType::Half, output_dtype, "da8w4_linear_cpu", [&] { \
        _da8w4_linear_impl<scalar_t, cpublas_can_pack, sym_quant_act>( \
            input, \
            input_scales, \
            input_qzeros, \
            weight, \
            weight_scales, \
            weight_qzeros, \
            compensation, \
            bias, \
            output); \
      });

if (cpublas_can_pack) {
  if (sym_quant_a) {
    call__da8w4_linear_impl(true, true);
  } else {
    call__da8w4_linear_impl(true, false);
  }
} else {
  if (sym_quant_a) {
    call__da8w4_linear_impl(false, true);
  } else {
    call__da8w4_linear_impl(false, false);
  }
}
return output;
}

at::Tensor da8w4_linear_cpu_with_quant(
  const at::Tensor& input,
  const at::Tensor& weight,
  const at::Tensor& weight_scales,
  const at::Tensor& weight_qzeros,
  const at::Tensor& compensation,
  const std::optional<at::Tensor>& bias,
  at::ScalarType output_dtype) {
    RECORD_FUNCTION(
      "sgl-kernel::da8w4_linear_cpu_with_quant", std::vector<c10::IValue>({input, weight}));

int64_t M_a = input.size(0);
int64_t K_a = input.size(1);
int64_t lda = input.stride(0);

const auto st = input.scalar_type();
TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf,
    "per_token_quant_int8: expect A to be bfloat16 or half.");

auto Aq = at::empty({M_a, K_a}, input.options().dtype(c10::kChar));
auto As = at::empty({M_a}, input.options().dtype(at::kFloat));
auto dummy_Azp = at::empty({M_a}, input.options().dtype(at::kInt));
static bool cpublas_can_pack = cpublas_could_pack();
bool sym_quant_a = true;
auto out_sizes = input.sizes().vec();
int64_t N = weight.size(0) * weight.size(-1) * 2;
out_sizes.back() = N;
auto output = at::empty(out_sizes, input.options().dtype(output_dtype));

#define call__da8w4_linear_with_quant_impl(cpublas_can_pack, sym_quant_act) \
  AT_DISPATCH_FLOATING_TYPES_AND2( \
      at::ScalarType::BFloat16, at::ScalarType::Half, output_dtype, "da8w4_linear_cpu_with_quant", [&] { \
        int8_t* __restrict__ Aq_data = Aq.data_ptr<int8_t>(); \
        float* __restrict__ As_data = As.data_ptr<float>();\
        const scalar_t* __restrict__ A_data = input.data_ptr<scalar_t>();\
        at::parallel_for(0, M_a, 0, [&] (int64_t begin, int64_t end) {\
          for (int64_t m = begin; m < end; ++m) {\
            quantize_row_int8_sym<scalar_t>(\
                Aq_data + m * K_a,\
                As_data[m],\
                A_data + m * lda,\
                K_a);\
          }\
        });\
        _da8w4_linear_impl<scalar_t, cpublas_can_pack, sym_quant_act>( \
            Aq, \
            As, \
            dummy_Azp, \
            weight, \
            weight_scales, \
            weight_qzeros, \
            compensation, \
            bias, \
            output); \
      });

if (cpublas_can_pack) {
  if (sym_quant_a) {
    call__da8w4_linear_with_quant_impl(true, true);
  } else {
    call__da8w4_linear_with_quant_impl(true, false);
  }
} else {
  if (sym_quant_a) {
    call__da8w4_linear_with_quant_impl(false, true);
  } else {
    call__da8w4_linear_with_quant_impl(false, false);
  }
}
return output;
}


