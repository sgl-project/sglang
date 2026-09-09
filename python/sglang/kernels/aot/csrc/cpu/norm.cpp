#include "common.h"
#include "vec.h"

namespace {

struct NormParams {
  // Treat all tensors as [B, H, T, D]:
  //   2D -> [B, 1, 1, D]
  //   3D -> [B, 1, T, D]
  //   4D -> [B, H, T, D]
  //
  // Input: last dimension contiguous.
  // Output: contiguous.

  int ndim{0};
  int64_t B{1}, H{1}, T{1}, D{1};
  int64_t i_strideB{0}, i_strideH{0}, i_strideT{0};
  float eps{1e-5f};
  float shift{0.f};

  const void* weight{nullptr};
  const void* bias{nullptr};

  explicit NormParams(const at::Tensor& input, float eps_) : ndim(input.dim()), eps(eps_) {
    TORCH_CHECK(ndim >= 2 && ndim <= 4, "Expected a 2D/3D/4D tensor, got ", ndim, "D.");

    B = input.size(0);
    D = input.size(ndim - 1);
    i_strideB = input.stride(0);
    switch (ndim) {
      case 2:
        break;
      case 3:
        T = input.size(1);
        i_strideT = input.stride(1);
        break;
      case 4:
        H = input.size(1);
        T = input.size(2);
        i_strideH = input.stride(1);
        i_strideT = input.stride(2);
        break;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
  }

  inline int64_t rows() const {
    return B * H * T;
  }
  inline int64_t input_offset(int64_t b, int64_t h, int64_t t) const {
    return b * i_strideB + h * i_strideH + t * i_strideT;
  }
  inline int64_t output_offset(int64_t b, int64_t h, int64_t t) const {
    return ((b * H + h) * T + t) * D;
  }
};

enum class NormMode {
  L2Norm,        // y = x / sqrt(mean(x^2) + eps)
  RMSNorm,       // y = x * weight / sqrt(mean(x^2) + eps)
  GemmaNorm,     // y = x * (weight + scale_shift) / sqrt(mean(x^2) + eps)
  LayerNorm,     // y = (x - mean(x)) * weight / sqrt(var(x) + eps) + bias
  RMSNormGated,  // y = x * weight / sqrt(mean(x^2) + eps) * SiLU(gate)
};

struct NormTraitsBase {
  static constexpr bool has_weight = false;
  static constexpr bool has_bias = false;
  static constexpr bool has_shift = false;
  static constexpr bool has_mean = false;
  static constexpr bool has_gate = false;

  template <typename VT>
  static inline VT apply_weight(VT x, VT w) {
    return x * w;
  }
#if defined(CPU_CAPABILITY_AVX512)
  static inline __m512 apply_weight(__m512 x, __m512 w) {
    return _mm512_mul_ps(x, w);
  }
#endif
};

template <NormMode M>
struct NormTraits : NormTraitsBase {};

template <>
struct NormTraits<NormMode::RMSNorm> : NormTraitsBase {
  static constexpr bool has_weight = true;
};

template <>
struct NormTraits<NormMode::GemmaNorm> : NormTraitsBase {
  static constexpr bool has_weight = true;
  static constexpr bool has_shift = true;

  template <typename VT>
  static inline VT apply_shift(VT w, VT shift) {
    return w + shift;
  }
#if defined(CPU_CAPABILITY_AVX512)
  static inline __m512 apply_shift(__m512 w, __m512 shift) {
    return _mm512_add_ps(w, shift);
  }
#endif
};

// LayerNorm: Var(X) = E(X^2) - (E(X))^2, refer to FlashInfer impl:
//   https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/norm.cuh#L552
template <>
struct NormTraits<NormMode::LayerNorm> : NormTraitsBase {
  static constexpr bool has_weight = true;
  static constexpr bool has_bias = true;
  static constexpr bool has_mean = true;

  template <typename VT>
  static inline VT apply_bias(VT x, VT bias) {
    return x + bias;
  }
#if defined(CPU_CAPABILITY_AVX512)
  static inline __m512 apply_bias(__m512 x, __m512 bias) {
    return _mm512_add_ps(x, bias);
  }
#endif
};

template <>
struct NormTraits<NormMode::RMSNormGated> : NormTraitsBase {
  static constexpr bool has_weight = true;
  static constexpr bool has_gate = true;

  static inline float apply_gate(float x, float gate) {
    return x * (gate / (1.f + std::exp(-gate)));
  }
  static inline at::vec::Vectorized<float> apply_gate(at::vec::Vectorized<float> x, at::vec::Vectorized<float> gate) {
    const auto one = at::vec::Vectorized<float>(1.f);
    return x * (gate / (one + gate.neg().exp_u20()));
  }
#if defined(CPU_CAPABILITY_AVX512)
  static inline __m512 apply_gate(__m512 x, __m512 gate) {
    __m512 minus_gate = _mm512_xor_ps(_mm512_set1_ps(-0.f), gate);
    __m512 denom = _mm512_add_ps(_mm512_exp_u20_ps(minus_gate), _mm512_set1_ps(1.0f));
    // NOTE: avoid vdivps -> use reciprocal
    __m512 sigmoid = _mm512_mul_ps(gate, _mm512_rcp14_ps(denom));
    return _mm512_mul_ps(x, sigmoid);
  }
#endif
};

template <NormMode M, typename scalar_t, int D>
struct NormReduce;

#if defined(CPU_CAPABILITY_AVX512)
template <NormMode M, int D>
struct NormReduce<M, at::BFloat16, D> {
  static inline void apply(
      at::BFloat16* __restrict__ out,
      const at::BFloat16* __restrict__ input,
      const at::BFloat16* __restrict__ gate,
      const NormParams& params) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 32;

    const bool use_bias = params.bias != nullptr;

    __m512bh va[COLS];
    __m512 vmean, vrscale;
    const __m512 vshift = _mm512_set1_ps(params.shift);

    // step 1: load input and do reduce with avx512-bf16
    __m512 vsum = _mm512_set1_ps(0.f);
    __m512 vsum2 = _mm512_set1_ps(0.f);
    Unroll<COLS>{}([&](auto col) {
      va[col] = (__m512bh)(_mm512_loadu_si512(input + col * 32));
      if constexpr (NormTraits<M>::has_mean) {
        vsum = _mm512_add_ps(vsum, CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)va[col], 0)));
        vsum = _mm512_add_ps(vsum, CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)va[col], 1)));
      }
      vsum2 = _mm512_dpbf16_ps(vsum2, va[col], va[col]);
    });

    // compute mean (if has_mean) and rscale
    float sum2 = _mm512_reduce_add_ps(vsum2);
    float variance = sum2 / D;
    if constexpr (NormTraits<M>::has_mean) {
      float sum = _mm512_reduce_add_ps(vsum);
      float mean = sum / D;
      variance -= mean * mean;
      vmean = _mm512_set1_ps(mean);
    }
    float rscale = 1.f / std::sqrt(variance + params.eps);
    vrscale = _mm512_set1_ps(rscale);

    // step 2: apply scale to output
    Unroll<COLS>{}([&](auto col) {
      __m512i a16 = (__m512i)va[col];
      __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 0));
      __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 1));
      if constexpr (NormTraits<M>::has_mean) {
        va0 = _mm512_sub_ps(va0, vmean);
        va1 = _mm512_sub_ps(va1, vmean);
      }
      va0 = _mm512_mul_ps(va0, vrscale);
      va1 = _mm512_mul_ps(va1, vrscale);
      if constexpr (NormTraits<M>::has_weight) {
        // TODO: need to block B to hide weight reload
        const at::BFloat16* weight = static_cast<const at::BFloat16*>(params.weight);
        __m512i w16 = (__m512i)(_mm512_loadu_si512(weight + col * 32));
        __m512 w0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(w16, 0));
        __m512 w1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(w16, 1));
        if constexpr (NormTraits<M>::has_shift) {
          w0 = NormTraits<M>::apply_shift(w0, vshift);
          w1 = NormTraits<M>::apply_shift(w1, vshift);
        }
        va0 = NormTraits<M>::apply_weight(va0, w0);
        va1 = NormTraits<M>::apply_weight(va1, w1);
      }
      if constexpr (NormTraits<M>::has_bias) {
        if (use_bias) {
          const at::BFloat16* bias = static_cast<const at::BFloat16*>(params.bias);
          __m512i b16 = (__m512i)(_mm512_loadu_si512(bias + col * 32));
          __m512 vbias0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 0));
          __m512 vbias1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 1));
          va0 = NormTraits<M>::apply_bias(va0, vbias0);
          va1 = NormTraits<M>::apply_bias(va1, vbias1);
        }
      }
      if constexpr (NormTraits<M>::has_gate) {
        __m512i g16 = (__m512i)(_mm512_loadu_si512(gate + col * 32));
        __m512 vgate0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(g16, 0));
        __m512 vgate1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(g16, 1));
        va0 = NormTraits<M>::apply_gate(va0, vgate0);
        va1 = NormTraits<M>::apply_gate(va1, vgate1);
      }
      _mm512_storeu_si512(out + col * 32, (__m512i)(_mm512_cvtne2ps_pbh(va1, va0)));
    });
  }
};
#endif

template <NormMode M, typename scalar_t, bool has_residual>
struct NormReduceGeneric {
  static inline void apply(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ gate,
      scalar_t* __restrict__ residual,
      const NormParams& params,
      int D) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int kVecSize = bVec::size();

    const bool use_bias = params.bias != nullptr;
    fVec sum_fvec{0.f}, sum2_fvec{0.f};
    float sum_val{0.f}, sum2_val{0.f};

    int d;
#pragma GCC unroll 4
    for (d = 0; d <= D - kVecSize; d += kVecSize) {
      auto [x_fvec0, x_fvec1] = load_float_vec2(input + d);
      if constexpr (has_residual) {
        auto [r_fvec0, r_fvec1] = load_float_vec2(residual + d);
        x_fvec0 += r_fvec0;
        x_fvec1 += r_fvec1;
      }
      sum2_fvec += x_fvec0 * x_fvec0;
      sum2_fvec += x_fvec1 * x_fvec1;
      if constexpr (NormTraits<M>::has_mean) {
        sum_fvec += x_fvec0;
        sum_fvec += x_fvec1;
      }
    }
#pragma GCC unroll 4
    for (; d < D; ++d) {
      float x_val = static_cast<float>(input[d]);
      if constexpr (has_residual) {
        x_val += static_cast<float>(residual[d]);
      }
      sum2_val += x_val * x_val;
      if constexpr (NormTraits<M>::has_mean) {
        sum_val += x_val;
      }
    }

    float mean = 0.f;
    float variance = sum2_val + vec_reduce_sum(sum2_fvec);
    variance /= D;
    if constexpr (NormTraits<M>::has_mean) {
      sum_val += vec_reduce_sum(sum_fvec);
      mean = sum_val / D;
      variance -= mean * mean;
    }

    float rsqrt_var = float(1) / std::sqrt(variance + params.eps);
    const fVec mean_fvec = fVec(mean);
    const fVec scale_fvec = fVec(rsqrt_var);
    const fVec shift_fvec = fVec(params.shift);

#pragma GCC unroll 4
    for (d = 0; d <= D - kVecSize; d += kVecSize) {
      auto [x_fvec0, x_fvec1] = load_float_vec2(input + d);
      if constexpr (has_residual) {
        auto [r_fvec0, r_fvec1] = load_float_vec2(residual + d);
        x_fvec0 += r_fvec0;
        x_fvec1 += r_fvec1;
        convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1).store(residual + d);
      }
      if constexpr (NormTraits<M>::has_mean) {
        x_fvec0 = x_fvec0 - mean_fvec;
        x_fvec1 = x_fvec1 - mean_fvec;
      }
      x_fvec0 = x_fvec0 * scale_fvec;
      x_fvec1 = x_fvec1 * scale_fvec;
      if constexpr (NormTraits<M>::has_weight) {
        auto [w_fvec0, w_fvec1] = load_float_vec2(static_cast<const scalar_t*>(params.weight) + d);
        if constexpr (NormTraits<M>::has_shift) {
          w_fvec0 = NormTraits<M>::apply_shift(w_fvec0, shift_fvec);
          w_fvec1 = NormTraits<M>::apply_shift(w_fvec1, shift_fvec);
        }
        x_fvec0 = NormTraits<M>::apply_weight(x_fvec0, w_fvec0);
        x_fvec1 = NormTraits<M>::apply_weight(x_fvec1, w_fvec1);
      }
      if constexpr (NormTraits<M>::has_bias) {
        if (use_bias) {
          auto [b_fvec0, b_fvec1] = load_float_vec2(static_cast<const scalar_t*>(params.bias) + d);
          x_fvec0 = NormTraits<M>::apply_bias(x_fvec0, b_fvec0);
          x_fvec1 = NormTraits<M>::apply_bias(x_fvec1, b_fvec1);
        }
      }
      if constexpr (NormTraits<M>::has_gate) {
        auto [g_fvec0, g_fvec1] = load_float_vec2(static_cast<const scalar_t*>(gate) + d);
        x_fvec0 = NormTraits<M>::apply_gate(x_fvec0, g_fvec0);
        x_fvec1 = NormTraits<M>::apply_gate(x_fvec1, g_fvec1);
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
      out_bvec.store(out + d);
    }
#pragma GCC unroll 4
    for (; d < D; ++d) {
      float x_val = static_cast<float>(input[d]);
      if constexpr (has_residual) {
        x_val += static_cast<float>(residual[d]);
        residual[d] = static_cast<scalar_t>(x_val);
      }
      if constexpr (NormTraits<M>::has_mean) {
        x_val -= mean;
      }
      x_val *= rsqrt_var;
      if constexpr (NormTraits<M>::has_weight) {
        float w_val = static_cast<float>(static_cast<const scalar_t*>(params.weight)[d]);
        if constexpr (NormTraits<M>::has_shift) {
          w_val = NormTraits<M>::apply_shift(w_val, params.shift);
        }
        x_val = NormTraits<M>::apply_weight(x_val, w_val);
      }
      if constexpr (NormTraits<M>::has_bias) {
        if (use_bias) {
          float b_val = static_cast<float>(static_cast<const scalar_t*>(params.bias)[d]);
          x_val = NormTraits<M>::apply_bias(x_val, b_val);
        }
      }
      if constexpr (NormTraits<M>::has_gate) {
        float g_val = static_cast<float>(static_cast<const scalar_t*>(gate)[d]);
        x_val = NormTraits<M>::apply_gate(x_val, g_val);
      }
      out[d] = static_cast<scalar_t>(x_val);
    }
  }
};

// TODO: add generic avx512-bf16 path here

#define LAUNCH_PARALLEL_LOOP(...)                                    \
  at::parallel_for(0, p.rows(), 0, [&](int64_t begin, int64_t end) { \
    int64_t b{0}, h{0}, t{0};                                        \
    data_index_init(begin, b, p.B, h, p.H, t, p.T);                  \
    for (int64_t i = begin; i < end; ++i) {                          \
      __VA_ARGS__;                                                   \
      data_index_step(b, p.B, h, p.H, t, p.T);                       \
    }                                                                \
  })

#define LAUNCH_PARALLEL_LOOP_HD(DIM)                                                              \
  case DIM:                                                                                       \
    LAUNCH_PARALLEL_LOOP(                                                                         \
        const scalar_t* __restrict__ gate_ptr{nullptr}; if constexpr (NormTraits<M>::has_gate) {  \
          gate_ptr = gate + p.output_offset(b, h, t);                                             \
        } NormReduce<M, scalar_t, DIM>::                                                          \
            apply(out + p.output_offset(b, h, t), input + p.input_offset(b, h, t), gate_ptr, p)); \
    return

template <NormMode M, typename scalar_t>
void norm4d_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const NormParams& p,
    const scalar_t* __restrict__ gate = nullptr) {
#if defined(CPU_CAPABILITY_AVX512)
  // fast path only applies to bfloat16 when D in {32, 64, 128, 256, 512}
  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    switch (p.D) {
      LAUNCH_PARALLEL_LOOP_HD(32);
      LAUNCH_PARALLEL_LOOP_HD(64);
      LAUNCH_PARALLEL_LOOP_HD(128);
      LAUNCH_PARALLEL_LOOP_HD(256);
      LAUNCH_PARALLEL_LOOP_HD(512);
      default:
        break;
    }
  }
#endif

  // generic path
  LAUNCH_PARALLEL_LOOP(
      const scalar_t* __restrict__ gate_ptr{nullptr}; if constexpr (NormTraits<M>::has_gate) {
        gate_ptr = gate + p.output_offset(b, h, t);
      } NormReduceGeneric<M, scalar_t, false>::
          apply(out + p.output_offset(b, h, t), input + p.input_offset(b, h, t), gate_ptr, nullptr, p, p.D));
}

template <NormMode M, typename scalar_t>
void fused_add_norm4d_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const NormParams& p,
    bool output_uses_input_stride = false) {
  LAUNCH_PARALLEL_LOOP(
      const int64_t out_offset = output_uses_input_stride ? p.input_offset(b, h, t) : p.output_offset(b, h, t);
      scalar_t* __restrict__ residual_ptr = residual + p.output_offset(b, h, t);
      NormReduceGeneric<M, scalar_t, true>::apply(
          out + out_offset, input + p.input_offset(b, h, t), nullptr, residual_ptr, p, p.D));
}

template <NormMode M, typename scalar_t, bool copy_gate = false>
void fused_qk_norm4d_kernel_impl(
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ gate_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const NormParams& params_q,
    const NormParams& params_k) {
  at::parallel_for(0, params_q.B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      for (int64_t h = 0; h < params_q.H /*num_head*/; ++h) {
        const int64_t q_offset = params_q.input_offset(b, h, /*t*/ 0);
        const int64_t out_offset = params_q.output_offset(b, h, /*t*/ 0);
        NormReduceGeneric<M, scalar_t, false>::apply(
            q_out + out_offset, q + q_offset, nullptr, nullptr, params_q, params_q.D);
        if constexpr (copy_gate) {
          std::memcpy(gate_out + out_offset, q + q_offset + params_q.D, params_q.D * sizeof(scalar_t));
        }
      }
      for (int64_t h = 0; h < params_k.H /*num_head_kv*/; ++h) {
        NormReduceGeneric<M, scalar_t, false>::apply(
            k_out + params_k.output_offset(b, h, /*t*/ 0),
            k + params_k.input_offset(b, h, /*t*/ 0),
            nullptr,
            nullptr,
            params_k,
            params_k.D);
      }
    }
  });
}

template <typename scalar_t>
float sum_squares(const scalar_t* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  fVec sum_fvec{0.f};
  float sum_val{0.f};
  int64_t i = 0;
  for (; i <= size - kVecSize; i += kVecSize) {
    auto [input_fvec0, input_fvec1] = load_float_vec2(input + i);
    sum_fvec += input_fvec0 * input_fvec0;
    sum_fvec += input_fvec1 * input_fvec1;
  }
  for (; i < size; ++i) {
    const float input_val = static_cast<float>(input[i]);
    sum_val += input_val * input_val;
  }
  return sum_val + vec_reduce_sum(sum_fvec);
}

template <typename scalar_t>
void fused_qk_norm_sumsq_kernel_impl(
    float* __restrict__ sum_sq,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const NormParams& q_params,
    const NormParams& k_params) {
  at::parallel_for(0, q_params.B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      sum_sq[b * 2] = sum_squares(q + q_params.input_offset(b, 0, 0), q_params.D);
      sum_sq[b * 2 + 1] = sum_squares(k + k_params.input_offset(b, 0, 0), k_params.D);
    }
  });
}

template <NormMode M, typename scalar_t>
void apply_norm_from_stats(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const NormParams& params,
    int64_t size,
    float sum,
    float sum_sq,
    int64_t tp_world_size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  const bool use_bias = params.bias != nullptr;
  float mean = 0.f;
  float variance = 0.f;
  const float global_size = static_cast<float>(size * tp_world_size);

  if constexpr (NormTraits<M>::has_mean) {
    mean = sum / global_size;
    variance = (sum_sq / global_size) - (mean * mean);
  } else {
    variance = sum_sq / global_size;
  }
  const float scale = 1.f / std::sqrt(variance + params.eps);
  const fVec scale_fvec{scale};
  const fVec mean_fvec{mean};
  const fVec shift_fvec{params.shift};

  int64_t i = 0;
  for (; i <= size - kVecSize; i += kVecSize) {
    auto [x_fvec0, x_fvec1] = load_float_vec2(input + i);
    if constexpr (NormTraits<M>::has_mean) {
      x_fvec0 = (x_fvec0 - mean_fvec) * scale_fvec;
      x_fvec1 = (x_fvec1 - mean_fvec) * scale_fvec;
    } else {
      x_fvec0 = x_fvec0 * scale_fvec;
      x_fvec1 = x_fvec1 * scale_fvec;
    }
    if constexpr (NormTraits<M>::has_weight) {
      auto [w_fvec0, w_fvec1] = load_float_vec2(static_cast<const scalar_t*>(params.weight) + i);
      if constexpr (NormTraits<M>::has_shift) {
        w_fvec0 = NormTraits<M>::apply_shift(w_fvec0, shift_fvec);
        w_fvec1 = NormTraits<M>::apply_shift(w_fvec1, shift_fvec);
      }
      x_fvec0 = NormTraits<M>::apply_weight(x_fvec0, w_fvec0);
      x_fvec1 = NormTraits<M>::apply_weight(x_fvec1, w_fvec1);
    }
    if constexpr (NormTraits<M>::has_bias) {
      if (use_bias) {
        auto [b_fvec0, b_fvec1] = load_float_vec2(static_cast<const scalar_t*>(params.bias) + i);
        x_fvec0 = NormTraits<M>::apply_bias(x_fvec0, b_fvec0);
        x_fvec1 = NormTraits<M>::apply_bias(x_fvec1, b_fvec1);
      }
    }

    convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1).store(output + i);
  }

  for (; i < size; ++i) {
    float x_val = static_cast<float>(input[i]);
    if constexpr (NormTraits<M>::has_mean) {
      x_val = (x_val - mean) * scale;
    } else {
      x_val = x_val * scale;
    }
    if constexpr (NormTraits<M>::has_weight) {
      float w_val = static_cast<float>(static_cast<const scalar_t*>(params.weight)[i]);
      if constexpr (NormTraits<M>::has_shift) {
        w_val = NormTraits<M>::apply_shift(w_val, params.shift);
      }
      x_val = NormTraits<M>::apply_weight(x_val, w_val);
    }
    if constexpr (NormTraits<M>::has_bias) {
      if (use_bias) {
        const float b_val = static_cast<float>(static_cast<const scalar_t*>(params.bias)[i]);
        x_val = NormTraits<M>::apply_bias(x_val, b_val);
      }
    }
    output[i] = static_cast<scalar_t>(x_val);
  }
}

template <NormMode M, typename scalar_t>
void fused_qk_norm_apply_from_stats_kernel_impl(
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const float* __restrict__ sum,
    const float* __restrict__ sum_sq,
    const NormParams& q_params,
    const NormParams& k_params,
    int64_t tp_world_size) {
  if constexpr (NormTraits<M>::has_mean) {
    TORCH_INTERNAL_ASSERT(sum != nullptr);
  }
  at::parallel_for(0, q_params.B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      float q_sum = 0.f;
      float k_sum = 0.f;
      if constexpr (NormTraits<M>::has_mean) {
        q_sum = sum[b * 2];
        k_sum = sum[b * 2 + 1];
      }
      apply_norm_from_stats<M>(
          q_out + q_params.output_offset(b, 0, 0),
          q + q_params.input_offset(b, 0, 0),
          q_params,
          q_params.D,
          q_sum,
          sum_sq[b * 2],
          tp_world_size);
      apply_norm_from_stats<M>(
          k_out + k_params.output_offset(b, 0, 0),
          k + k_params.input_offset(b, 0, 0),
          k_params,
          k_params.D,
          k_sum,
          sum_sq[b * 2 + 1],
          tp_world_size);
    }
  });
}

using fVec = at::vec::Vectorized<float>;
template <typename T>
struct ModulationParam {
  const T* data{nullptr};
  int64_t stride_b{0};
  int64_t stride_s{0};
  int64_t stride_c{0};

  ModulationParam() = default;

  explicit ModulationParam(const at::Tensor& tensor)
      : data(tensor.data_ptr<T>()),
        stride_b(tensor.stride(0)),
        stride_s(tensor.stride(1)),
        stride_c(tensor.stride(2)) {}

  inline const T* row(int64_t b, int64_t s) const {
    if (data == nullptr) {
      return nullptr;
    }
    return data + b * stride_b + s * stride_s;
  }
};

template <typename T>
inline void load_param_vec2(fVec& v0, fVec& v1, const T* __restrict__ p, int64_t stride_c, int64_t d) {
  if (stride_c == 0) {
    v0 = v1 = fVec(static_cast<float>(p[0]));
  } else {
    std::tie(v0, v1) = load_float_vec2(p + d);
  }
}

template <typename param_t>
inline void apply_scale_shift_vec(
    fVec& x0,
    fVec& x1,
    const param_t* __restrict__ scale,
    const param_t* __restrict__ shift,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    int64_t d,
    float scale_constant = 1.0f) {
  fVec scale0, scale1;
  fVec shift0, shift1;

  load_param_vec2(scale0, scale1, scale, scale_stride_c, d);
  load_param_vec2(shift0, shift1, shift, shift_stride_c, d);

  x0 = x0 * (fVec(scale_constant) + scale0) + shift0;
  x1 = x1 * (fVec(scale_constant) + scale1) + shift1;
}
template <typename scalar_t>
inline void apply_residual_gate_vec(
    fVec& x0,
    fVec& x1,
    const fVec& r0,
    const fVec& r1,
    const scalar_t* __restrict__ gate,
    const float* __restrict__ gate_fp32,
    int64_t gate_stride_c,
    int64_t d) {
  fVec g0, g1;

  if (gate_fp32 != nullptr) {
    load_param_vec2(g0, g1, gate_fp32, gate_stride_c, d);
  } else if (gate != nullptr) {
    load_param_vec2(g0, g1, gate, gate_stride_c, d);
  } else {
    g0 = g1 = fVec(1.0f);
  }

  x0 = r0 + x0 * g0;
  x1 = r1 + x1 * g1;
}

template <NormMode M, typename scalar_t, typename param_t>
inline void apply_norm_modulate_row(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const param_t* __restrict__ scale,
    const param_t* __restrict__ shift,
    int64_t D,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    const fVec& sum_vec,
    const fVec& sum_sq_vec,
    float sum,
    float sum_sq,
    float eps) {
  sum_sq += vec_reduce_sum(sum_sq_vec);

  float mean = 0.0f;
  float variance = sum_sq / static_cast<float>(D);

  if constexpr (NormTraits<M>::has_mean) {
    sum += vec_reduce_sum(sum_vec);
    mean = sum / static_cast<float>(D);
    variance -= mean * mean;
  }

  const float rstd = 1.0f / std::sqrt(variance + eps);

  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int64_t kVecSize = bVec::size();

  const fVec mean_vec(mean);
  const fVec rstd_vec(rstd);

  int64_t d = 0;

#pragma GCC unroll 4
  for (; d <= D - kVecSize; d += kVecSize) {
    auto [x0, x1] = load_float_vec2(input + d);

    if constexpr (NormTraits<M>::has_mean) {
      x0 -= mean_vec;
      x1 -= mean_vec;
    }

    x0 *= rstd_vec;
    x1 *= rstd_vec;

    if (weight != nullptr) {
      auto [w0, w1] = load_float_vec2(weight + d);
      x0 *= w0;
      x1 *= w1;
    }

    if constexpr (NormTraits<M>::has_bias) {
      if (bias != nullptr) {
        auto [b0, b1] = load_float_vec2(bias + d);
        x0 += b0;
        x1 += b1;
      }
    }

    // Match CUDA/CuTe activation-dtype boundary:
    // norm FP32 -> activation dtype -> scale/shift.
    const bVec norm_value = convert_from_float_ext<scalar_t>(x0, x1);
    std::tie(x0, x1) = at::vec::convert_to_float(norm_value);

    apply_scale_shift_vec(x0, x1, scale, shift, scale_stride_c, shift_stride_c, d);
    convert_from_float_ext<scalar_t>(x0, x1).store(output + d);
  }

#pragma GCC unroll 4
  for (; d < D; ++d) {
    float x = static_cast<float>(input[d]);

    if constexpr (NormTraits<M>::has_mean) {
      x -= mean;
    }

    x *= rstd;

    if (weight != nullptr) {
      x *= weight[d];
    }

    if constexpr (NormTraits<M>::has_bias) {
      if (bias != nullptr) {
        x += bias[d];
      }
    }

    // Match CUDA/CuTe activation-dtype boundary.
    x = static_cast<float>(static_cast<scalar_t>(x));

    x = x * (1.0f + static_cast<float>(scale[d * scale_stride_c])) + static_cast<float>(shift[d * shift_stride_c]);
    output[d] = static_cast<scalar_t>(x);
  }
}
template <typename scalar_t, typename param_t>
inline void fused_scale_shift_row(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const param_t* __restrict__ scale,
    const param_t* __restrict__ shift,
    int64_t D,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    float scale_constant) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int64_t kVecSize = bVec::size();
  int64_t d = 0;

#pragma GCC unroll 4
  for (; d <= D - kVecSize; d += kVecSize) {
    auto [x0, x1] = load_float_vec2(input + d);
    apply_scale_shift_vec(x0, x1, scale, shift, scale_stride_c, shift_stride_c, d, scale_constant);
    convert_from_float_ext<scalar_t>(x0, x1).store(output + d);
  }

#pragma GCC unroll 4
  for (; d < D; ++d) {
    const float x = static_cast<float>(input[d]);
    const float scale_value = static_cast<float>(scale[d * scale_stride_c]);
    const float shift_value = static_cast<float>(shift[d * shift_stride_c]);
    output[d] = static_cast<scalar_t>(x * (scale_constant + scale_value) + shift_value);
  }
}

template <NormMode M, typename scalar_t, typename param_t>
inline void fused_norm_scale_shift_row(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const param_t* __restrict__ scale,
    const param_t* __restrict__ shift,
    int64_t D,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    float eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int64_t kVecSize = bVec::size();

  fVec sum_vec{0.0f};
  fVec sum_sq_vec{0.0f};
  float sum = 0.0f;
  float sum_sq = 0.0f;

  int64_t d = 0;

#pragma GCC unroll 4
  for (; d <= D - kVecSize; d += kVecSize) {
    auto [x0, x1] = load_float_vec2(input + d);
    sum_sq_vec += x0 * x0 + x1 * x1;
    if constexpr (NormTraits<M>::has_mean) {
      sum_vec += x0 + x1;
    }
  }

#pragma GCC unroll 4
  for (; d < D; ++d) {
    const float x = static_cast<float>(input[d]);
    sum_sq += x * x;
    if constexpr (NormTraits<M>::has_mean) {
      sum += x;
    }
  }
  apply_norm_modulate_row<M>(
      output,
      input,
      weight,
      bias,
      scale,
      shift,
      D,
      scale_stride_c,
      shift_stride_c,
      sum_vec,
      sum_sq_vec,
      sum,
      sum_sq,
      eps);
}

template <NormMode M, typename scalar_t, typename param_t>
inline void fused_scale_residual_norm_scale_shift_row(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_output,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual_gate,
    const float* __restrict__ residual_gate_fp32,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const param_t* __restrict__ scale,
    const param_t* __restrict__ shift,
    int64_t D,
    int64_t gate_stride_c,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    float eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int64_t kVecSize = bVec::size();

  fVec sum_vec{0.0f};
  fVec sum_sq_vec{0.0f};
  float sum = 0.0f;
  float sum_sq = 0.0f;

  int64_t d = 0;

#pragma GCC unroll 4
  for (; d <= D - kVecSize; d += kVecSize) {
    auto [x0, x1] = load_float_vec2(input + d);
    auto [r0, r1] = load_float_vec2(residual + d);

    apply_residual_gate_vec(x0, x1, r0, r1, residual_gate, residual_gate_fp32, gate_stride_c, d);

    // Match CUDA: residual + gate * input is rounded to activation dtype
    // before normalization.
    const bVec residual_value = convert_from_float_ext<scalar_t>(x0, x1);

    residual_value.store(residual_output + d);

    std::tie(x0, x1) = at::vec::convert_to_float(residual_value);

    sum_sq_vec += x0 * x0 + x1 * x1;

    if constexpr (NormTraits<M>::has_mean) {
      sum_vec += x0 + x1;
    }
  }

#pragma GCC unroll 4
  for (; d < D; ++d) {
    float x = static_cast<float>(input[d]);
    if (residual_gate_fp32 != nullptr) {
      x *= residual_gate_fp32[d * gate_stride_c];
    } else if (residual_gate != nullptr) {
      x *= static_cast<float>(residual_gate[d * gate_stride_c]);
    }

    x += static_cast<float>(residual[d]);

    const scalar_t residual_value = static_cast<scalar_t>(x);

    residual_output[d] = residual_value;

    x = static_cast<float>(residual_value);

    sum_sq += x * x;

    if constexpr (NormTraits<M>::has_mean) {
      sum += x;
    }
  }

  apply_norm_modulate_row<M>(
      output,
      residual_output,
      weight,
      bias,
      scale,
      shift,
      D,
      scale_stride_c,
      shift_stride_c,
      sum_vec,
      sum_sq_vec,
      sum,
      sum_sq,
      eps);
}

template <typename scalar_t, typename param_t>
void launch_fused_scale_shift(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const ModulationParam<param_t>& scale,
    const ModulationParam<param_t>& shift,
    int64_t B,
    int64_t S,
    int64_t D,
    float scale_constant) {
  const int64_t rows = B * S;

  at::parallel_for(0, rows, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t b = row / S;
      const int64_t s = row % S;
      const int64_t offset = row * D;
      fused_scale_shift_row(
          output + offset,
          input + offset,
          scale.row(b, s),
          shift.row(b, s),
          D,
          scale.stride_c,
          shift.stride_c,
          scale_constant);
    }
  });
}

template <bool HasResidual, typename scalar_t, typename param_t>
void launch_fused_norm(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_output,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ input,
    const ModulationParam<scalar_t>& residual_gate,
    const ModulationParam<float>& residual_gate_fp32,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const ModulationParam<param_t>& scale,
    const ModulationParam<param_t>& shift,
    int64_t B,
    int64_t S,
    int64_t D,
    const std::string& norm_type,
    float eps) {
  auto launch = [&](auto mode_tag) {
    constexpr NormMode M = decltype(mode_tag)::value;
    at::parallel_for(0, B * S, 0, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        const int64_t b = row / S;
        const int64_t s = row % S;
        const int64_t offset = row * D;

        const param_t* scale_ptr = scale.row(b, s);
        const param_t* shift_ptr = shift.row(b, s);

        if constexpr (HasResidual) {
          const scalar_t* gate_ptr = residual_gate.row(b, s);
          const float* gate_fp32_ptr = residual_gate_fp32.row(b, s);

          const int64_t gate_stride_c = gate_fp32_ptr != nullptr ? residual_gate_fp32.stride_c : residual_gate.stride_c;
          fused_scale_residual_norm_scale_shift_row<M, scalar_t, param_t>(
              output + offset,
              residual_output + offset,
              residual + offset,
              input + offset,
              gate_ptr,
              gate_fp32_ptr,
              weight,
              bias,
              scale_ptr,
              shift_ptr,
              D,
              gate_stride_c,
              scale.stride_c,
              shift.stride_c,
              eps);
        } else {
          fused_norm_scale_shift_row<M, scalar_t, param_t>(
              output + offset,
              input + offset,
              weight,
              bias,
              scale_ptr,
              shift_ptr,
              D,
              scale.stride_c,
              shift.stride_c,
              eps);
        }
      }
    });
  };
  if (norm_type == "rms") {
    launch(std::integral_constant<NormMode, NormMode::RMSNorm>{});
  } else {
    launch(std::integral_constant<NormMode, NormMode::LayerNorm>{});
  }
}

#undef LAUNCH_PARALLEL_LOOP
#undef LAUNCH_PARALLEL_LOOP_HD
}  // anonymous namespace

template <int... Dims>
inline void CHECK_INPUT_ND(const at::Tensor& tensor) {
  static_assert(sizeof...(Dims) > 0);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(tensor);
  const int64_t dim = tensor.dim();
  const bool dim_ok = ((dim == Dims) || ...);
  TORCH_CHECK(dim_ok, "Expected input dim to match template constraints, got ", dim);
}

// input : {batch_size, hidden_size}
at::Tensor l2norm_cpu(at::Tensor& input, double eps) {
  const auto st = input.scalar_type();
  CHECK_INPUT_ND<2>(input);
  NormParams p{input, static_cast<float>(eps)};

  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "l2norm_kernel", [&] {
    norm4d_kernel_impl<NormMode::L2Norm, scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p);
  });
  return output;
}

// input : {batch_size, hidden_size} or {batch_size, seq_len, hidden_size}
// weight: {hidden_size}
at::Tensor rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps) {
  const auto st = input.scalar_type();
  CHECK_INPUT_ND<2, 3>(input);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {input.size(-1)}, st);

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();

  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "rmsnorm_kernel", [&] {
    norm4d_kernel_impl<NormMode::RMSNorm, scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p);
  });
  return output;
}

// input : {batch_size, hidden_size}
// weight: {hidden_size}
at::Tensor gemma_rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps) {
  CHECK_INPUT_ND<2>(input);
  const auto st = input.scalar_type();
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {input.size(-1)}, st);

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();
  p.shift = 1.f;

  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "gemma_rmsnorm_kernel", [&] {
    norm4d_kernel_impl<NormMode::GemmaNorm, scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p);
  });
  return output;
}

// input : {batch_size, hidden_size} or {batch_size, num_head, seq_len, head_dim}
// weight: {hidden_size}
at::Tensor gemma3_rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps) {
  const auto st = input.scalar_type();
  CHECK_INPUT_ND<2, 4>(input);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {input.size(-1)}, st);

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();
  p.shift = 1.f;

  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "gemma3_rmsnorm_kernel", [&] {
    norm4d_kernel_impl<NormMode::GemmaNorm, scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p);
  });
  return output;
}

// Gemma4RMSNorm: with_scale ? norm(x) * (weight + scale_shift) : norm(x)
// input : {batch_size, hidden_size} or {batch_size, seq_len, hidden_size}
// weight: {hidden_size}
at::Tensor gemma4_rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps, double scale_shift, bool with_scale) {
  const auto st = input.scalar_type();
  CHECK_INPUT_ND<2, 3>(input);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {input.size(-1)}, st);

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();
  p.shift = static_cast<float>(scale_shift);

  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "gemma4_rmsnorm_kernel", [&] {
    if (with_scale) {
      norm4d_kernel_impl<NormMode::GemmaNorm, scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p);
    } else {
      norm4d_kernel_impl<NormMode::L2Norm, scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p);
    }
  });
  return output;
}

// input : {batch_size, hidden_size} or {batch_size, seq_len, hidden_size}
// weight: {hidden_size}
// bias  : {hidden_size}
at::Tensor
layernorm_cpu(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias, double eps) {
  const auto st = input.scalar_type();
  const int64_t hidden_size = input.size(-1);
  CHECK_INPUT_ND<2, 3>(input);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {hidden_size}, st);
  if (bias.has_value()) {
    CHECK_INPUT_SHAPE_DTYPE<false>(bias.value(), {hidden_size}, st);
  }

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();
  p.bias = bias.has_value() ? bias.value().data_ptr() : nullptr;

  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "layernorm_kernel", [&] {
    norm4d_kernel_impl<NormMode::LayerNorm, scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p);
  });
  return output;
}

// input : {batch_size, hidden_size}
// weight: {hidden_size}
// gate: {batch_size, hidden_size}
at::Tensor fused_rmsnorm_gated_cpu(at::Tensor& input, at::Tensor& weight, at::Tensor& gate, double eps) {
  const auto st = input.scalar_type();
  const int64_t batch_size = input.size(0);
  const int64_t hidden_size = input.size(-1);
  CHECK_INPUT_ND<2>(input);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {hidden_size}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(gate, {batch_size, hidden_size}, st);

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();

  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_rmsnorm_gated_kernel", [&] {
    norm4d_kernel_impl<NormMode::RMSNormGated, scalar_t>(
        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p, gate.data_ptr<scalar_t>());
  });
  return output;
}

// input   : {batch_size, hidden_size} or {batch_size, seq_len, hidden_size}
// residual: {batch_size, hidden_size} or {batch_size, seq_len, hidden_size}
// weight  : {hidden_size}
void fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps) {
  const auto st = input.scalar_type();
  CHECK_INPUT_ND<2, 3>(input);
  CHECK_EQ(input.sizes(), residual.sizes());
  CHECK_EQ(st, residual.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {input.size(-1)}, st);

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_add_rmsnorm_kernel", [&] {
    fused_add_norm4d_kernel_impl<NormMode::RMSNorm, scalar_t>(
        input.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        p,
        /*output_uses_input_stride=*/true);
  });
}

// input   : {batch_size, hidden_size}
// residual: {batch_size, hidden_size}
// weight  : {hidden_size}
void gemma_fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps) {
  const auto st = input.scalar_type();
  CHECK_INPUT_ND<2>(input);
  CHECK_EQ(input.sizes(), residual.sizes());
  CHECK_EQ(st, residual.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {input.size(-1)}, st);

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();
  p.shift = 1.f;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "gemma_fused_add_rmsnorm_kernel", [&] {
    fused_add_norm4d_kernel_impl<NormMode::GemmaNorm, scalar_t>(
        input.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        p,
        /*output_uses_input_stride=*/true);
  });
}

// input   : {batch_size, hidden_size} or {batch_size, seq_len, hidden_size}
// residual: {batch_size, hidden_size} or {batch_size, seq_len, hidden_size}
// weight  : {hidden_size}
// bias    : {hidden_size}
at::Tensor fused_add_layernorm_cpu(
    const at::Tensor& input,
    at::Tensor& residual,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    double eps) {
  const auto st = input.scalar_type();
  const int64_t hidden_size = input.size(-1);
  CHECK_INPUT_ND<2, 3>(input);
  CHECK_EQ(input.sizes(), residual.sizes());
  CHECK_EQ(st, residual.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {hidden_size}, st);
  if (bias.has_value()) {
    CHECK_INPUT_SHAPE_DTYPE<false>(bias.value(), {hidden_size}, st);
  }

  NormParams p{input, static_cast<float>(eps)};
  p.weight = weight.data_ptr();
  p.bias = bias.has_value() ? bias.value().data_ptr() : nullptr;

  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_add_layernorm_kernel", [&] {
    fused_add_norm4d_kernel_impl<NormMode::LayerNorm, scalar_t>(
        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), p);
  });
  return output;
}

// q: {batch_size, q_hidden_size} 2D
// k: {batch_size, k_hidden_size} 2D
std::tuple<at::Tensor, at::Tensor> fused_qk_rmsnorm_cpu(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& q_weight, const at::Tensor& k_weight, double eps) {
  const auto st = q.scalar_type();
  CHECK_INPUT_ND<2>(q);
  CHECK_INPUT_ND<2>(k);

  CHECK_EQ(k.size(0), q.size(0));
  CHECK_EQ(k.scalar_type(), st);
  CHECK_INPUT_SHAPE_DTYPE<false>(q_weight, {q.size(1)}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(k_weight, {k.size(1)}, st);

  NormParams q_params{q, static_cast<float>(eps)};
  q_params.weight = q_weight.data_ptr();

  NormParams k_params{k, static_cast<float>(eps)};
  k_params.weight = k_weight.data_ptr();

  at::Tensor q_out = at::empty_like(q);
  at::Tensor k_out = at::empty_like(k);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_qk_rmsnorm_kernel", [&] {
    fused_qk_norm4d_kernel_impl<NormMode::RMSNorm, scalar_t, false>(
        q_out.data_ptr<scalar_t>(),
        k_out.data_ptr<scalar_t>(),
        nullptr,
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        q_params,
        k_params);
  });
  return std::make_tuple(q_out, k_out);
}

// q: {batch_size, local_q_hidden_size} 2D
// k: {batch_size, local_k_hidden_size} 2D
// output: local Q/K squared sums, {batch_size, 2} FP32
at::Tensor fused_qk_rmsnorm_sumsq_cpu(const at::Tensor& q, const at::Tensor& k) {
  const auto st = q.scalar_type();
  CHECK_INPUT_ND<2>(q);
  CHECK_INPUT_ND<2>(k);
  CHECK_EQ(k.size(0), q.size(0));
  CHECK_EQ(k.scalar_type(), st);

  NormParams q_params{q, 0.f};
  NormParams k_params{k, 0.f};
  at::Tensor sum_sq = at::empty({q.size(0), 2}, q.options().dtype(at::kFloat));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_qk_rmsnorm_sumsq_kernel", [&] {
    fused_qk_norm_sumsq_kernel_impl<scalar_t>(
        sum_sq.data_ptr<float>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), q_params, k_params);
  });
  return sum_sq;
}

// q: {batch_size, local_q_hidden_size} 2D
// k: {batch_size, local_k_hidden_size} 2D
// sum_sq: globally reduced Q/K squared sums, {batch_size, 2} FP32
std::tuple<at::Tensor, at::Tensor> fused_qk_rmsnorm_apply_from_stats_cpu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& sum_sq,
    int64_t tp_world_size,
    double eps) {
  const auto st = q.scalar_type();
  CHECK_INPUT_ND<2>(q);
  CHECK_INPUT_ND<2>(k);
  CHECK_EQ(k.size(0), q.size(0));
  CHECK_EQ(k.scalar_type(), st);
  CHECK_INPUT_SHAPE_DTYPE<false>(q_weight, {q.size(1)}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(k_weight, {k.size(1)}, st);
  CHECK_INPUT_SHAPE_DTYPE<true>(sum_sq, {q.size(0), 2}, at::kFloat);
  TORCH_CHECK(tp_world_size > 0, "tp_world_size must be positive, got ", tp_world_size);

  NormParams q_params{q, static_cast<float>(eps)};
  q_params.weight = q_weight.data_ptr();
  NormParams k_params{k, static_cast<float>(eps)};
  k_params.weight = k_weight.data_ptr();
  at::Tensor q_out = at::empty_like(q);
  at::Tensor k_out = at::empty_like(k);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_qk_rmsnorm_apply_kernel", [&] {
    fused_qk_norm_apply_from_stats_kernel_impl<NormMode::RMSNorm, scalar_t>(
        q_out.data_ptr<scalar_t>(),
        k_out.data_ptr<scalar_t>(),
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        nullptr,
        sum_sq.data_ptr<float>(),
        q_params,
        k_params,
        tp_world_size);
  });
  return std::make_tuple(q_out, k_out);
}

// q : {batch_size, num_head * head_dim} 2D
// k : {batch_size, num_head_kv * head_dim} 2D
std::tuple<at::Tensor, at::Tensor> fused_qk_gemma_rmsnorm_cpu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    double eps,
    int64_t head_dim) {
  const auto st = q.scalar_type();
  CHECK_INPUT_ND<2>(q);
  CHECK_INPUT_ND<2>(k);

  int64_t batch_size = q.size(0);
  int64_t num_head = q.size(1) / head_dim;
  int64_t num_head_kv = k.size(1) / head_dim;
  CHECK_EQ(k.size(0), batch_size);
  CHECK_EQ(k.scalar_type(), st);
  CHECK_INPUT_SHAPE_DTYPE<false>(q_weight, {head_dim}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(k_weight, {head_dim}, st);

  NormParams q_params{q, static_cast<float>(eps)};
  q_params.H = num_head;
  q_params.D = head_dim;
  q_params.i_strideH = head_dim;
  q_params.weight = q_weight.data_ptr();
  q_params.shift = 1.f;

  NormParams k_params{k, static_cast<float>(eps)};
  k_params.H = num_head_kv;
  k_params.D = head_dim;
  k_params.i_strideH = head_dim;
  k_params.weight = k_weight.data_ptr();
  k_params.shift = 1.f;

  at::Tensor q_out = at::empty_like(q);
  at::Tensor k_out = at::empty_like(k);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_qk_gemma_rmsnorm_kernel", [&] {
    fused_qk_norm4d_kernel_impl<NormMode::GemmaNorm, scalar_t, false>(
        q_out.data_ptr<scalar_t>(),
        k_out.data_ptr<scalar_t>(),
        nullptr,
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        q_params,
        k_params);
  });
  return std::make_tuple(q_out, k_out);
}

// q_gate : {batch_size, num_head * head_dim * 2} 2D, interleaved per head as [q_h, gate_h]
// k      : {batch_size, num_head_kv * head_dim} 2D
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_qk_gemma_rmsnorm_with_gate_cpu(
    const at::Tensor& q_gate,
    const at::Tensor& k,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    double eps,
    int64_t head_dim,
    int64_t num_head) {
  const auto st = q_gate.scalar_type();
  CHECK_INPUT_ND<2>(q_gate);
  CHECK_INPUT_ND<2>(k);

  int64_t batch_size = q_gate.size(0);
  int64_t num_head_kv = k.size(1) / head_dim;
  CHECK_EQ(q_gate.size(1), num_head * head_dim * 2);
  CHECK_EQ(k.size(0), batch_size);
  CHECK_EQ(k.scalar_type(), st);
  CHECK_INPUT_SHAPE_DTYPE<false>(q_weight, {head_dim}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(k_weight, {head_dim}, st);

  NormParams q_params{q_gate, static_cast<float>(eps)};
  q_params.H = num_head;
  q_params.D = head_dim;
  q_params.i_strideH = head_dim * 2;
  q_params.weight = q_weight.data_ptr();
  q_params.shift = 1.f;

  NormParams k_params{k, static_cast<float>(eps)};
  k_params.H = num_head_kv;
  k_params.D = head_dim;
  k_params.i_strideH = head_dim;
  k_params.weight = k_weight.data_ptr();
  k_params.shift = 1.f;

  at::Tensor q_out = at::empty({batch_size * num_head, head_dim}, q_gate.options());
  at::Tensor k_out = at::empty({batch_size * num_head_kv, head_dim}, k.options());
  at::Tensor gate_out = at::empty_like(q_out);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_qk_gemma_rmsnorm_with_gate_kernel", [&] {
    fused_qk_norm4d_kernel_impl<NormMode::GemmaNorm, scalar_t, true>(
        q_out.data_ptr<scalar_t>(),
        k_out.data_ptr<scalar_t>(),
        gate_out.data_ptr<scalar_t>(),
        q_gate.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        q_params,
        k_params);
  });
  return std::make_tuple(q_out, k_out, gate_out);
}

inline void check_modulation_param(const at::Tensor& param, const at::Tensor& input, const char* name) {
  CHECK_CPU(param);
  CHECK_DIM(3, param);
  CHECK_EQ(param.sizes(), input.sizes());
  TORCH_CHECK(param.stride(2) == 0 || param.stride(2) == 1, name, " hidden-dimension stride must be 0 or 1.");
}

inline const float* get_norm_param_ptr(const std::optional<at::Tensor>& param, int64_t D, const char* name) {
  if (!param.has_value()) {
    return nullptr;
  }

  const auto& tensor = param.value();

  CHECK_INPUT(tensor);
  CHECK_DIM(1, tensor);
  CHECK_EQ(tensor.size(0), D);

  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Float, "CPU fused diffusion norm only supports FP32 norm ", name, ".");
  return tensor.data_ptr<float>();
}
at::Tensor fused_scale_shift_cpu(
    const at::Tensor& input, const at::Tensor& scale, const at::Tensor& shift, double scale_constant) {
  CHECK_INPUT_ND<3>(input);

  check_modulation_param(scale, input, "scale");
  check_modulation_param(shift, input, "shift");

  CHECK_EQ(scale.scalar_type(), shift.scalar_type());

  const auto st = input.scalar_type();

  const int64_t B = input.size(0);
  const int64_t S = input.size(1);
  const int64_t D = input.size(2);

  at::Tensor output = at::empty_like(input);

  if (input.numel() == 0) {
    return output;
  }

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(st, scale.scalar_type(), "fused_scale_shift_cpu", [&] {
    const ModulationParam<param_t> scale_param(scale);
    const ModulationParam<param_t> shift_param(shift);
    launch_fused_scale_shift<scalar_t, param_t>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        scale_param,
        shift_param,
        B,
        S,
        D,
        static_cast<float>(scale_constant));
  });

  return output;
}
at::Tensor fused_norm_scale_shift_cpu(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& scale,
    const at::Tensor& shift,
    const std::string& norm_type,
    double eps) {
  CHECK_INPUT_ND<3>(input);

  TORCH_CHECK(norm_type == "rms" || norm_type == "layer", "norm_type must be \"rms\" or \"layer\".");

  check_modulation_param(scale, input, "scale");
  check_modulation_param(shift, input, "shift");

  CHECK_EQ(scale.scalar_type(), shift.scalar_type());

  const int64_t B = input.size(0);
  const int64_t S = input.size(1);
  const int64_t D = input.size(2);

  const float* weight_ptr = get_norm_param_ptr(weight, D, "weight");

  TORCH_CHECK(!bias.has_value() || norm_type == "layer", "bias is only supported for LayerNorm.");

  const float* bias_ptr = get_norm_param_ptr(bias, D, "bias");

  at::Tensor output = at::empty_like(input);

  if (input.numel() == 0) {
    return output;
  }

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(input.scalar_type(), scale.scalar_type(), "fused_norm_scale_shift_cpu", [&] {
    const ModulationParam<param_t> scale_param(scale);

    const ModulationParam<param_t> shift_param(shift);

    launch_fused_norm<false, scalar_t, param_t>(
        output.data_ptr<scalar_t>(),
        /*residual_output=*/nullptr,
        /*residual=*/nullptr,
        input.data_ptr<scalar_t>(),
        /*residual_gate=*/{},
        /*residual_gate_fp32=*/{},
        weight_ptr,
        bias_ptr,
        scale_param,
        shift_param,
        B,
        S,
        D,
        norm_type,
        static_cast<float>(eps));
  });

  return output;
}

std::tuple<at::Tensor, at::Tensor> fused_scale_residual_norm_scale_shift_cpu(
    const at::Tensor& residual,
    const at::Tensor& input,
    const std::optional<at::Tensor>& residual_gate,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& scale,
    const at::Tensor& shift,
    const std::string& norm_type,
    double eps) {
  CHECK_INPUT_ND<3>(input);
  CHECK_INPUT_ND<3>(residual);
  CHECK_EQ(residual.sizes(), input.sizes());
  CHECK_EQ(residual.scalar_type(), input.scalar_type());
  TORCH_CHECK(norm_type == "rms" || norm_type == "layer", "norm_type must be \"rms\" or \"layer\".");

  check_modulation_param(scale, input, "scale");
  check_modulation_param(shift, input, "shift");

  CHECK_EQ(scale.scalar_type(), shift.scalar_type());

  if (residual_gate.has_value()) {
    check_modulation_param(residual_gate.value(), input, "residual_gate");

    TORCH_CHECK(
        residual_gate->scalar_type() == input.scalar_type() || residual_gate->scalar_type() == at::ScalarType::Float,
        "residual_gate must have the same dtype as "
        "input or be FP32.");
  }

  const int64_t B = input.size(0);
  const int64_t S = input.size(1);
  const int64_t D = input.size(2);

  const float* weight_ptr = get_norm_param_ptr(weight, D, "weight");

  TORCH_CHECK(!bias.has_value() || norm_type == "layer", "bias is only supported for LayerNorm.");

  const float* bias_ptr = get_norm_param_ptr(bias, D, "bias");

  at::Tensor output = at::empty_like(input);
  at::Tensor residual_output = at::empty_like(input);

  if (input.numel() == 0) {
    return {output, residual_output};
  }

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(
      input.scalar_type(), scale.scalar_type(), "fused_scale_residual_norm_scale_shift_cpu", [&] {
        const ModulationParam<param_t> scale_param(scale);
        const ModulationParam<param_t> shift_param(shift);
        ModulationParam<scalar_t> gate_param{};
        ModulationParam<float> gate_fp32_param{};

        if (residual_gate.has_value()) {
          if (residual_gate->scalar_type() == at::ScalarType::Float) {
            gate_fp32_param = ModulationParam<float>(residual_gate.value());
          } else {
            gate_param = ModulationParam<scalar_t>(residual_gate.value());
          }
        }

        launch_fused_norm<true, scalar_t, param_t>(
            output.data_ptr<scalar_t>(),
            residual_output.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            gate_param,
            gate_fp32_param,
            weight_ptr,
            bias_ptr,
            scale_param,
            shift_param,
            B,
            S,
            D,
            norm_type,
            static_cast<float>(eps));
      });

  return {output, residual_output};
}
