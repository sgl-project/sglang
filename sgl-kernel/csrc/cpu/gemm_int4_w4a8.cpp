#include <ATen/native/CPUBlas.h>
#include <c10/util/Unroll.h>
#include <torch/all.h>

#include "gemm.h"
#include "vec.h"
namespace {

#define BLOCK_N block_size_n()

template <bool sym_quant_a>
struct ActDtype;
template <>
struct ActDtype<true> {
  using type = int8_t;
};
template <>
struct ActDtype<false> {
  using type = uint8_t;
};

struct alignas(32) m256i_wrapper {
  __m256i data;
};

#if defined(CPU_CAPABILITY_AVX512)
inline std::array<m256i_wrapper, 2> load_zps_4vnni(const int8_t* __restrict__ zps) {
  // broadcast 01234567 to
  // 01234567012345670123456701234567
  __m256i vzps_low = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps));
  __m256i vzps_high = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps + 8));
  // shuffle from
  // 01234567012345670123456701234567
  // to
  // 00001111222233334444555566667777
  __m256i shuffle_mask =
      _mm256_set_epi8(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  vzps_low = _mm256_shuffle_epi8(vzps_low, shuffle_mask);
  vzps_high = _mm256_shuffle_epi8(vzps_high, shuffle_mask);
  m256i_wrapper vzps_low_wp, vzps_high_wp;
  vzps_low_wp.data = vzps_low;
  vzps_high_wp.data = vzps_high;
  return {vzps_low_wp, vzps_high_wp};
}

inline std::array<m256i_wrapper, 2> load_uint4_as_int8(const uint8_t* __restrict__ qB) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qB));
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i high = _mm256_srli_epi16(packed, 4);
  high = _mm256_and_si256(high, low_mask);
  __m256i low = _mm256_and_si256(packed, low_mask);
  m256i_wrapper low_wp, high_wp;
  low_wp.data = low;
  high_wp.data = high;
  return {low_wp, high_wp};
}

template <int64_t N, int64_t ldb>
void _dequant_weight_zp_only(const uint8_t* __restrict__ B, int8_t* dqB, const int8_t* __restrict__ qzeros, int64_t K) {
  // unpack weight int8 -> two int4
  // subtract zero point
  // B shape = [K, ldb] = [K, N / 2], actual shape = [K / 4, N / 2, 4]
  // dqB shape = [K, N], actual shape = [K / 4, N, 4]
#pragma GCC unroll 2
  for (int n = 0; n < N; n += 16) {
    auto [zps_low_wp, zps_high_wp] = load_zps_4vnni(&qzeros[n]);
    auto zps_low = zps_low_wp.data;
    auto zps_high = zps_high_wp.data;
    for (int k = 0; k < K; k += 4) {
      auto [vb_low_wp, vb_high_wp] = load_uint4_as_int8(B + ldb * k + n / 2 * 4);
      auto vb_low = vb_low_wp.data;
      auto vb_high = vb_high_wp.data;
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
        dq_val = (float)(input[m * ldi + n] - a_zp * comp_b[n]) * a_scale * scale_b[n];
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
template <int64_t N, int64_t ldb>
void _dequant_weight_zp_only(const uint8_t* B, int8_t* dqB, const int8_t* qzeros, int64_t K) {
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

inline __m512i combine_m256i(std::array<m256i_wrapper, 2> two_256) {
  return combine_m256i(two_256[0].data, two_256[1].data);
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
  __m512i ones = _mm512_set1_epi8(1);  // used for computing compensation
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
  c10::ForcedUnroll<M * COLS>{}([&](auto i) { vc[i] = _mm512_setzero_epi32(); });

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
        vcompensate[col] = _mm512_dpbusd_epi32(vcompensate[col], ones, vb[col]);
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
    c10::ForcedUnroll<unroll>{}([&](auto i) { c10::ForcedUnroll<M * COLS>{}(compute, 4 * (k * unroll + i)); });
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
      vc[i] = _mm512_sub_epi32(vc[i], _mm512_mullo_epi32(vcompensate[col], _mm512_set1_epi32(*(qzeros_a + row))));
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
  _dequant_gemm_accum_small_M<M, N, ldb, sym_quant_a>(C, A, scales_a, qzeros_a, B, scales_b, qzeros_b, K, lda, ldc);
#endif

template <int64_t N, int64_t ldb, bool sym_quant_a>
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
    int64_t ldc,
    bool use_brgemm) {
  // Compute GEMM int8 * int8 -> int32
  // dequant result to float by applying scales/qzeros
#if defined(CPU_CAPABILITY_AVX512)
  if (!use_brgemm) {
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
      default:
        TORCH_CHECK(false, "tinygemm_kernel: unexpected M for AVX path!");
    }
  }

  int8_t dqB[K * N];
  _dequant_weight_zp_only<N, ldb>(B, dqB, qzeros_b, K);
  using Tin = typename ActDtype<sym_quant_a>::type;
  Tin* A_ptr = (Tin*)A;
  if (use_brgemm) {
    int32_t C_i32[M * N];
    at::native::cpublas::brgemm(
        M, N, K, lda, N /*ldb*/, N /*ldc*/, false /* add_C */, A_ptr, dqB, C_i32, true /* is_vnni */);
    _mm_prefetch(B + N * K / 2, _MM_HINT_T0);
    _mm_prefetch(A + K, _MM_HINT_T0);
    _dequant_and_store<true, N, sym_quant_a>(
        C, C_i32, scales_a, qzeros_a, scales_b, compensation, M, N /*ldi*/, ldc, 1 /*ldsa*/);
  } else
#endif
  {
    TORCH_CHECK(false, "tinygemm_kernel: scalar path not implemented!");
  }
}

template <int64_t N>
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
  } else {  // initialize to zero
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

template <typename out_dtype, int64_t N>
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

template <typename out_dtype, bool sym_quant_a>
void _da8w4_linear_impl(
    const at::Tensor& input,
    const at::Tensor& input_scales,
    const at::Tensor& input_qzeros,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_qzeros,
    const std::optional<at::Tensor>& bias,
    at::Tensor& output) {
  // input shape = [..., K]
  // input is per token quantized
  int64_t K = input.size(-1);
  auto input_view = input.view({-1, K});
  int64_t M = input_view.size(0);
  TORCH_CHECK(input_scales.numel() == M, "DA8W4: unexpected input scales shape");
  if (not sym_quant_a) {
    TORCH_CHECK(input_scales.sizes() == input_qzeros.sizes(), "DA8W4: unexpected input qzeros shape");
  }

  // weight + compensation shape = [Nc, Kc, BLOCK_N * BLOCK_K / 2 + BLOCK_N*sizeof(int32_t)]
  // scales/qzeros shape = [Nc, G, BLOCK_N]
  const bool use_brgemm = can_use_brgemm<int8_t>(M);
  int64_t Nc = weight.size(0);
  int64_t Kc = weight.size(1);
  int64_t N = Nc * BLOCK_N;
  TORCH_CHECK(K == Kc * BLOCK_K, "DA8W4: weight and input shapes mismatch");
  int64_t block_m = [&]() -> long {
    if (M <= 48) {
      return M;
    } else if (M < 64) {
      return 32;
    } else if (M < 96) {
      return 64;
    } else {
      return 128;
    }
  }();
  int64_t Mc = (M + block_m - 1) / block_m;
  bool parallel_on_M = M > 128;
  int64_t num_blocks = parallel_on_M ? Mc * Nc : Nc;

  // scales/qzeros shape = [Nc, G, BLOCK_N]
  int64_t num_groups = weight_scales.size(1);
  int64_t group_size = K / num_groups;
  TORCH_CHECK(group_size % BLOCK_K == 0, "DA8W4 CPU: group_size should be divisible by BLOCK_K");
  int64_t block_per_group = group_size / BLOCK_K;

  using Tin = typename ActDtype<sym_quant_a>::type;
  const Tin* a_ptr = input_view.data_ptr<Tin>();
  const float* a_scales_ptr = input_scales.data_ptr<float>();
  const int32_t* a_qzeros_ptr = sym_quant_a ? nullptr : input_qzeros.data_ptr<int32_t>();
  const uint8_t* b_ptr = weight.data_ptr<uint8_t>();
  const float* b_scales_ptr = weight_scales.data_ptr<float>();
  const int8_t* b_qzeros_ptr = weight_qzeros.data_ptr<int8_t>();
  out_dtype* c_ptr = output.data_ptr<out_dtype>();
  const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

  at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t mc = parallel_on_M ? i / Nc : 0;
      int64_t nc = parallel_on_M ? i % Nc : i;
      int64_t mc_end = parallel_on_M ? mc + 1 : Mc;

      for (int mci = mc; mci < mc_end; ++mci) {
        int64_t m_size = mci * block_m + block_m > M ? M - mci * block_m : block_m;
        alignas(64) float y_buf[m_size][BLOCK_N];
        // copy bias to y_buf if bias is not None
        auto bias_data = bias_ptr ? bias_ptr + nc * BLOCK_N : nullptr;
        copy_bias<BLOCK_N>(bias_data, y_buf[0], m_size);
        for (int kci = 0; kci < Kc; ++kci) {
          int32_t* compensation_ptr =
              sym_quant_a ? nullptr
                          : (int32_t*)(void*)(b_ptr + (nc * Kc + kci) * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t))) +
                                              BLOCK_K * BLOCK_N / 2) /*Bcomp*/;
          _dequant_gemm_accum<BLOCK_N, BLOCK_N / 2, sym_quant_a>(
              y_buf[0] /*C*/,
              (uint8_t*)a_ptr + mci * block_m * K + kci * BLOCK_K /*A*/,
              a_scales_ptr + mci * block_m /*scales_a*/,
              a_qzeros_ptr + mci * block_m /*qzeros_a*/,
              b_ptr + (nc * Kc + kci) * (BLOCK_N * (BLOCK_K / 2 + sizeof(int32_t))),
              b_scales_ptr + nc * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N /*scales_b*/,
              b_qzeros_ptr + nc * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N /*qzeros_b*/,
              compensation_ptr /*Bcomp*/,
              m_size /*M*/,
              BLOCK_K /*K*/,
              K /*lda*/,
              BLOCK_N /*ldc*/,
              use_brgemm);
        }
        // store y_buf to output with dtype conversion
        store_out<out_dtype, BLOCK_N>(y_buf[0], c_ptr + mci * block_m * N + nc * BLOCK_N, m_size, N /*lda*/);
      }
    }
    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

}  // anonymous namespace

at::Tensor int4_scaled_mm_cpu_with_quant(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_qzeros,
    const std::optional<at::Tensor>& bias,
    at::ScalarType output_dtype) {
  RECORD_FUNCTION("sgl-kernel::int4_scaled_mm_cpu_with_quant", std::vector<c10::IValue>({input, weight}));

  int64_t M_a = input.size(0);
  int64_t K_a = input.size(1);
  int64_t lda = input.stride(0);

  const auto st = input.scalar_type();
  TORCH_CHECK(
      st == at::kBFloat16 || st == at::kHalf, "int4_scaled_mm_cpu_with_quant: expect A to be bfloat16 or half.");

  auto Aq = at::empty({M_a, K_a}, input.options().dtype(c10::kByte));
  auto As = at::empty({M_a}, input.options().dtype(at::kFloat));
  auto Azp = at::ones({M_a}, input.options().dtype(at::kInt)).mul(128);
  bool sym_quant_a = false;  // sym_a s8s8 is unified to u8s8 with compensation (128)

  auto out_sizes = input.sizes().vec();
  int64_t N = weight_scales.size(0) * weight_scales.size(-1);
  out_sizes.back() = N;
  auto output = at::empty(out_sizes, input.options().dtype(output_dtype));

#define call__da8w4_linear_with_quant_impl(sym_quant_act)                                                             \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                                                                    \
      at::ScalarType::BFloat16, at::ScalarType::Half, output_dtype, "int4_scaled_mm_cpu_with_quant", [&] {            \
        uint8_t* __restrict__ Aq_data = Aq.data_ptr<uint8_t>();                                                       \
        float* __restrict__ As_data = As.data_ptr<float>();                                                           \
        int32_t* __restrict__ Azp_data = Azp.data_ptr<int32_t>();                                                     \
        const scalar_t* __restrict__ A_data = input.data_ptr<scalar_t>();                                             \
        at::parallel_for(0, M_a, 0, [&](int64_t begin, int64_t end) {                                                 \
          for (int64_t m = begin; m < end; ++m) {                                                                     \
            quantize_row_int8<scalar_t>(Aq_data + m * K_a, As_data[m], A_data + m * lda, K_a);                        \
          }                                                                                                           \
        });                                                                                                           \
        _da8w4_linear_impl<scalar_t, sym_quant_act>(Aq, As, Azp, weight, weight_scales, weight_qzeros, bias, output); \
      });

  call__da8w4_linear_with_quant_impl(false);

  return output;
}
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    fVec x0 = fVec::loadu(input + d);
    fVec x1 = fVec::loadu(input + d + fVec::size());
    Vec res = convert_from_float_ext<scalar_t>(x0, x1);
    res.store(out + d);
  }
}

template <typename scalar_t>
void tinygemm_kernel(
    scalar_t* C,
    float* C_temp,
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
    int64_t ldc_f,
    int64_t ldc_s,
    bool store_out,
    bool use_brgemm) {
  // TODO: add sym quant act, now only asym
  _dequant_gemm_accum<BLOCK_N, BLOCK_N / 2, false>(
      C_temp, A, scales_a, qzeros_a, B, scales_b, qzeros_b, compensation, M, K, lda, ldc_f, use_brgemm);
  if (store_out) {
    // copy from Ctmp to C
    for (int64_t m = 0; m < M; ++m) {
      copy_stub<scalar_t>(C + m * ldc_s, C_temp + m * ldc_f, BLOCK_N);
    }
  }
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(      \
      TYPE * C,                             \
      float* C_temp,                        \
      const uint8_t* A,                     \
      const float* scales_a,                \
      const int32_t* qzeros_a,              \
      const uint8_t* B,                     \
      const float* scales_b,                \
      const int8_t* qzeros_b,               \
      const int32_t* compensation,          \
      int64_t M,                            \
      int64_t K,                            \
      int64_t lda,                          \
      int64_t ldc_f,                        \
      int64_t ldc_s,                        \
      bool store_out,                       \
      bool use_brgemm)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

// int4 gemm dispatch api register
at::Tensor int4_scaled_mm_cpu(
    at::Tensor& x, at::Tensor& w, at::Tensor& w_zeros, at::Tensor& w_scales, std::optional<at::Tensor> bias) {
  return int4_scaled_mm_cpu_with_quant(x, w, w_scales, w_zeros, bias, x.scalar_type());
}
