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

template <typename scale_t, typename shift_t>
inline void apply_scale_shift_vec(
    at::vec::Vectorized<float>& x0,
    at::vec::Vectorized<float>& x1,
    const scale_t* __restrict__ scale,
    const shift_t* __restrict__ shift,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    int64_t c,
    float scale_constant = 1.0f);

template <typename scale_t, typename shift_t>
inline float apply_scale_shift_scalar(
    float x,
    const scale_t* __restrict__ scale,
    const shift_t* __restrict__ shift,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    int64_t c,
    float scale_constant = 1.0f);

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

template <NormMode M, typename scalar_t, bool has_residual, bool has_scale_shift = false, typename param_t = scalar_t>
struct NormReduceGeneric {
  static inline void apply(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ norm_gate,
      scalar_t* __restrict__ residual,
      const NormParams& params,
      int D,
      const param_t* __restrict__ scale = nullptr,
      const param_t* __restrict__ shift = nullptr,
      int64_t scale_stride_c = 0,
      int64_t shift_stride_c = 0,
      const param_t* __restrict__ residual_gate = nullptr,
      int64_t residual_gate_stride_c = 0,
      scalar_t* __restrict__ residual_output = nullptr) {
    scalar_t* __restrict__ residual_store = nullptr;
    if constexpr (has_residual) {
      residual_store = residual_output != nullptr ? residual_output : residual;
    }
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int kVecSize = bVec::size();

    const bool use_bias = params.bias != nullptr;
    const bool use_weight = params.weight != nullptr;
    fVec sum_fvec{0.f}, sum2_fvec{0.f};
    float sum_val{0.f}, sum2_val{0.f};
    int64_t d;
#pragma GCC unroll 4
    for (d = 0; d <= D - kVecSize; d += kVecSize) {
      auto [x_fvec0, x_fvec1] = load_float_vec2(input + d);
      if constexpr (has_residual) {
        if (residual_gate != nullptr) {
          apply_scale_shift_vec(
              x_fvec0,
              x_fvec1,
              residual_gate,
              residual,
              residual_gate_stride_c,
              /*shift_stride_c=*/1,
              d,
              /*scale_constant=*/0.0f);
        } else {
          // gate=None means gate=1.
          auto [r_fvec0, r_fvec1] = load_float_vec2(residual + d);
          x_fvec0 += r_fvec0;
          x_fvec1 += r_fvec1;
        }
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
        if (residual_gate != nullptr) {
          x_val = apply_scale_shift_scalar(
              x_val,
              residual_gate,
              residual,
              residual_gate_stride_c,
              /*shift_stride_c=*/1,
              d,
              /*scale_constant=*/0.0f);
        } else {
          x_val += static_cast<float>(residual[d]);
        }
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
        if (residual_gate != nullptr) {
          apply_scale_shift_vec(
              x_fvec0,
              x_fvec1,
              residual_gate,
              residual,
              residual_gate_stride_c,
              /*shift_stride_c=*/1,
              d,
              /*scale_constant=*/0.0f);
        } else {
          auto [r_fvec0, r_fvec1] = load_float_vec2(residual + d);

          x_fvec0 += r_fvec0;
          x_fvec1 += r_fvec1;
        }
        // Write residual + gate * x to the second output.
        convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1).store(residual_store + d);
      }
      if constexpr (NormTraits<M>::has_mean) {
        x_fvec0 = x_fvec0 - mean_fvec;
        x_fvec1 = x_fvec1 - mean_fvec;
      }
      x_fvec0 = x_fvec0 * scale_fvec;
      x_fvec1 = x_fvec1 * scale_fvec;
      if constexpr (NormTraits<M>::has_weight) {
        if (use_weight) {
          auto [w_fvec0, w_fvec1] = load_float_vec2(static_cast<const scalar_t*>(params.weight) + d);
          if constexpr (NormTraits<M>::has_shift) {
            w_fvec0 = NormTraits<M>::apply_shift(w_fvec0, shift_fvec);
            w_fvec1 = NormTraits<M>::apply_shift(w_fvec1, shift_fvec);
          }
          x_fvec0 = NormTraits<M>::apply_weight(x_fvec0, w_fvec0);
          x_fvec1 = NormTraits<M>::apply_weight(x_fvec1, w_fvec1);
        }
      }
      if constexpr (NormTraits<M>::has_bias) {
        if (use_bias) {
          auto [b_fvec0, b_fvec1] = load_float_vec2(static_cast<const scalar_t*>(params.bias) + d);
          x_fvec0 = NormTraits<M>::apply_bias(x_fvec0, b_fvec0);
          x_fvec1 = NormTraits<M>::apply_bias(x_fvec1, b_fvec1);
        }
      }
      if constexpr (NormTraits<M>::has_gate) {
        auto [g_fvec0, g_fvec1] = load_float_vec2(static_cast<const scalar_t*>(norm_gate) + d);
        x_fvec0 = NormTraits<M>::apply_gate(x_fvec0, g_fvec0);
        x_fvec1 = NormTraits<M>::apply_gate(x_fvec1, g_fvec1);
      }
      if constexpr (has_scale_shift) {
        apply_scale_shift_vec(x_fvec0, x_fvec1, scale, shift, scale_stride_c, shift_stride_c, d);
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
      out_bvec.store(out + d);
    }
#pragma GCC unroll 4
    for (; d < D; ++d) {
      float x_val = static_cast<float>(input[d]);
      if constexpr (has_residual) {
        if (residual_gate != nullptr) {
          x_val = apply_scale_shift_scalar(
              x_val,
              residual_gate,
              residual,
              residual_gate_stride_c,
              /*shift_stride_c=*/1,
              d,
              /*scale_constant=*/0.0f);
        } else {
          x_val += static_cast<float>(residual[d]);
        }
        residual_store[d] = static_cast<scalar_t>(x_val);
      }
      if constexpr (NormTraits<M>::has_mean) {
        x_val -= mean;
      }
      x_val *= rsqrt_var;
      if constexpr (NormTraits<M>::has_weight) {
        if (use_weight) {
          float w_val = static_cast<float>(static_cast<const scalar_t*>(params.weight)[d]);
          if constexpr (NormTraits<M>::has_shift) {
            w_val = NormTraits<M>::apply_shift(w_val, params.shift);
          }
          x_val = NormTraits<M>::apply_weight(x_val, w_val);
        }
      }
      if constexpr (NormTraits<M>::has_bias) {
        if (use_bias) {
          float b_val = static_cast<float>(static_cast<const scalar_t*>(params.bias)[d]);
          x_val = NormTraits<M>::apply_bias(x_val, b_val);
        }
      }
      if constexpr (NormTraits<M>::has_gate) {
        float g_val = static_cast<float>(static_cast<const scalar_t*>(norm_gate)[d]);
        x_val = NormTraits<M>::apply_gate(x_val, g_val);
      }
      if constexpr (has_scale_shift) {
        x_val = apply_scale_shift_scalar(x_val, scale, shift, scale_stride_c, shift_stride_c, d);
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
struct ScaleShiftBroadcastParams {
  // Normal 1D/2D/3D broadcast:
  //   offset = b * stride_b + l * stride_l + c * stride_c
  //
  // Frame-based 4D broadcast:
  //   tensor shape [B, F, 1, C]
  //   offset = b * stride_b + (l / frame_seqlen) * stride_f + c * stride_c

  int64_t stride_b{0};
  int64_t stride_l{0};
  int64_t stride_c{0};

  int64_t stride_f{0};
  int64_t frame_seqlen{1};
  bool frame_broadcast{false};

  inline int64_t row_offset(int64_t b, int64_t l) const {
    if (frame_broadcast) {
      return b * stride_b + (l / frame_seqlen) * stride_f;
    }
    return b * stride_b + l * stride_l;
  }
};

ScaleShiftBroadcastParams make_scale_shift_broadcast_params(
    const at::Tensor& tensor, int64_t batch_size, int64_t seq_len, int64_t hidden_size, const char* tensor_name) {
  ScaleShiftBroadcastParams params;

  // Scalar, regardless of whether its shape is [], [1], [1, 1], etc.
  if (tensor.numel() == 1) {
    return params;
  }

  TORCH_CHECK(
      tensor.dim() >= 1 && tensor.dim() <= 4,
      tensor_name,
      " must be a scalar or a 1D/2D/3D/4D tensor, got ",
      tensor.dim(),
      "D.");

  switch (tensor.dim()) {
    case 1: {
      // [C] or [1]
      TORCH_CHECK(tensor.size(0) == hidden_size, tensor_name, " must match the hidden dimension.");
      params.stride_c = tensor.size(0) == 1 ? 0 : tensor.stride(0);
      break;
    }

    case 2: {
      // [B, C], [1, C], [B, 1], [1, 1]
      TORCH_CHECK(tensor.size(0) == 1 || tensor.size(0) == batch_size, tensor_name, " must match the batch dimension.");
      TORCH_CHECK(
          tensor.size(1) == 1 || tensor.size(1) == hidden_size, tensor_name, " must match the hidden dimension.");
      params.stride_b = tensor.size(0) == 1 ? 0 : tensor.stride(0);
      params.stride_c = tensor.size(1) == 1 ? 0 : tensor.stride(1);
      break;
    }

    case 3: {
      // Broadcastable to [B, L, C]
      TORCH_CHECK(tensor.size(0) == 1 || tensor.size(0) == batch_size, tensor_name, " must match the batch dimension.");
      TORCH_CHECK(tensor.size(1) == 1 || tensor.size(1) == seq_len, tensor_name, " must match the sequence dimension.");
      TORCH_CHECK(
          tensor.size(2) == 1 || tensor.size(2) == hidden_size, tensor_name, " must match the hidden dimension.");
      params.stride_b = tensor.size(0) == 1 ? 0 : tensor.stride(0);
      params.stride_l = tensor.size(1) == 1 ? 0 : tensor.stride(1);
      params.stride_c = tensor.size(2) == 1 ? 0 : tensor.stride(2);
      break;
    }

    case 4: {
      TORCH_CHECK(tensor.size(0) == 1 || tensor.size(0) == batch_size, tensor_name, " must match the batch dimension.");
      TORCH_CHECK(tensor.size(2) == 1, tensor_name, " must have a singleton sequence dimension.");
      TORCH_CHECK(
          tensor.size(3) == 1 || tensor.size(3) == hidden_size, tensor_name, " must match the hidden dimension.");
      // [B, F, 1, C], used by frame-based diffusion models.
      const int64_t num_frames = tensor.size(1);
      TORCH_CHECK(seq_len % num_frames == 0, tensor_name, " frame dimension must divide the sequence length.");
      params.frame_broadcast = true;
      params.frame_seqlen = seq_len / num_frames;
      params.stride_b = tensor.size(0) == 1 ? 0 : tensor.stride(0);
      params.stride_f = tensor.stride(1);
      params.stride_c = tensor.size(3) == 1 ? 0 : tensor.stride(3);
      break;
    }

    default:
      TORCH_INTERNAL_ASSERT(false);
  }
  TORCH_CHECK(
      params.stride_c == 0 || params.stride_c == 1, tensor_name, " hidden dimension must be contiguous or broadcast.");

  return params;
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

template <typename scalar_t, typename param_t>
void fused_scale_shift_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const param_t* __restrict__ scale,
    const param_t* __restrict__ shift,
    int64_t batch_size,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t input_stride_b,
    int64_t input_stride_l,
    const ScaleShiftBroadcastParams& scale_params,
    const ScaleShiftBroadcastParams& shift_params,
    float scale_constant) {
  using bVec = at::vec::Vectorized<scalar_t>;

  constexpr int64_t kVecSize = bVec::size();

  const int64_t num_rows = batch_size * seq_len;

  at::parallel_for(0, num_rows, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t b = row / seq_len;
      const int64_t l = row % seq_len;
      const scalar_t* __restrict__ input_ptr = input + b * input_stride_b + l * input_stride_l;

      const param_t* __restrict__ scale_ptr = scale + scale_params.row_offset(b, l);

      const param_t* __restrict__ shift_ptr = shift + shift_params.row_offset(b, l);

      // output is newly allocated and contiguous.
      scalar_t* __restrict__ output_ptr = output + row * hidden_size;

      int64_t d = 0;
#pragma GCC unroll 4
      for (; d <= hidden_size - kVecSize; d += kVecSize) {
        auto [x0, x1] = load_float_vec2(input_ptr + d);
        apply_scale_shift_vec(
            x0, x1, scale_ptr, shift_ptr, scale_params.stride_c, shift_params.stride_c, d, scale_constant);
        convert_from_float_ext<scalar_t>(x0, x1).store(output_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        const float x = static_cast<float>(input_ptr[d]);
        const float result = apply_scale_shift_scalar(
            x, scale_ptr, shift_ptr, scale_params.stride_c, shift_params.stride_c, d, scale_constant);
        output_ptr[d] = static_cast<scalar_t>(result);
      }
    }
  });
}

template <NormMode M, typename scalar_t, typename param_t>
void fused_norm_scale_shift_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const param_t* __restrict__ scale,
    const param_t* __restrict__ shift,
    const NormParams& p,
    const ScaleShiftBroadcastParams& scale_params,
    const ScaleShiftBroadcastParams& shift_params) {
  // This kernel only supports logical [B, S, D].
  TORCH_INTERNAL_ASSERT(p.H == 1);

  const int64_t B = p.B;
  const int64_t S = p.T;
  const int64_t D = p.D;
  const int64_t num_rows = B * S;

  at::parallel_for(0, num_rows, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t b = row / S;
      const int64_t s = row % S;

      const int64_t input_offset = p.input_offset(b, /*h=*/0, s);

      // out is newly allocated and contiguous.
      const int64_t output_offset = row * D;

      const param_t* __restrict__ scale_ptr = scale + scale_params.row_offset(b, s);

      const param_t* __restrict__ shift_ptr = shift + shift_params.row_offset(b, s);

      NormReduceGeneric<M, scalar_t, false, true, param_t>::apply(
          out + output_offset,
          input + input_offset,
          /*norm_gate=*/nullptr,
          /*residual=*/nullptr,
          p,
          D,
          scale_ptr,
          shift_ptr,
          scale_params.stride_c,
          shift_params.stride_c);
    }
  });
}

template <NormMode M, typename scalar_t, typename param_t>
void fused_scale_residual_norm_scale_shift_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_output,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ input,
    const param_t* __restrict__ gate,
    const param_t* __restrict__ scale,
    const param_t* __restrict__ shift,
    int64_t residual_stride_b,
    int64_t residual_stride_s,
    const NormParams& p,
    const ScaleShiftBroadcastParams& gate_params,
    const ScaleShiftBroadcastParams& scale_params,
    const ScaleShiftBroadcastParams& shift_params) {
  const int64_t B = p.B;
  const int64_t S = p.T;
  const int64_t D = p.D;
  const int64_t num_rows = B * S;

  at::parallel_for(0, num_rows, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t b = row / S;
      const int64_t s = row % S;
      const scalar_t* __restrict__ input_ptr = input + p.input_offset(b, 0, s);
      scalar_t* __restrict__ residual_ptr =
          const_cast<scalar_t*>(residual + b * residual_stride_b + s * residual_stride_s);
      scalar_t* __restrict__ residual_output_ptr = residual_output + row * D;
      scalar_t* __restrict__ output_ptr = output + row * D;

      const param_t* __restrict__ gate_ptr = gate == nullptr ? nullptr : gate + gate_params.row_offset(b, s);

      const param_t* __restrict__ scale_ptr = scale + scale_params.row_offset(b, s);

      const param_t* __restrict__ shift_ptr = shift + shift_params.row_offset(b, s);

      NormReduceGeneric<M, scalar_t, true, true, param_t>::apply(
          output_ptr,
          input_ptr,
          /*norm_gate=*/nullptr,
          residual_ptr,
          p,
          D,
          scale_ptr,
          shift_ptr,
          scale_params.stride_c,
          shift_params.stride_c,
          gate_ptr,
          gate_params.stride_c,
          residual_output_ptr);
    }
  });
}
#undef LAUNCH_PARALLEL_LOOP
#undef LAUNCH_PARALLEL_LOOP_HD
template <typename scale_t, typename shift_t>
inline void apply_scale_shift_vec(
    at::vec::Vectorized<float>& x0,
    at::vec::Vectorized<float>& x1,
    const scale_t* __restrict__ scale,
    const shift_t* __restrict__ shift,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    int64_t c,
    float scale_constant) {
  using fVec = at::vec::Vectorized<float>;
  const fVec scale_constant_vec(scale_constant);
  fVec scale0;
  fVec scale1;

  if (scale_stride_c == 0) {
    const fVec scale_vec(static_cast<float>(scale[0]));
    scale0 = scale_vec;
    scale1 = scale_vec;
  } else {
    std::tie(scale0, scale1) = load_float_vec2(scale + c);
  }

  fVec shift0;
  fVec shift1;

  if (shift_stride_c == 0) {
    const fVec shift_vec(static_cast<float>(shift[0]));
    shift0 = shift_vec;
    shift1 = shift_vec;
  } else {
    std::tie(shift0, shift1) = load_float_vec2(shift + c);
  }
  x0 = x0 * (scale_constant_vec + scale0) + shift0;
  x1 = x1 * (scale_constant_vec + scale1) + shift1;
}

template <typename scale_t, typename shift_t>
inline float apply_scale_shift_scalar(
    float x,
    const scale_t* __restrict__ scale,
    const shift_t* __restrict__ shift,
    int64_t scale_stride_c,
    int64_t shift_stride_c,
    int64_t c,
    float scale_constant) {
  const float scale_value = static_cast<float>(scale[c * scale_stride_c]);
  const float shift_value = static_cast<float>(shift[c * shift_stride_c]);
  return x * (scale_constant + scale_value) + shift_value;
}
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

at::Tensor fused_scale_shift_cpu(
    const at::Tensor& input, const at::Tensor& scale, const at::Tensor& shift, double scale_constant) {
  CHECK_INPUT_ND<3>(input);
  const auto dtype = input.scalar_type();
  TORCH_CHECK(scale.scalar_type() == shift.scalar_type(), "scale dtype must match shift dtype.");
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.size(1);
  const int64_t hidden_size = input.size(2);

  at::Tensor output = at::empty(input.sizes(), input.options());

  if (input.numel() == 0) {
    return output;
  }

  const auto scale_params = make_scale_shift_broadcast_params(scale, batch_size, seq_len, hidden_size, "scale");

  const auto shift_params = make_scale_shift_broadcast_params(shift, batch_size, seq_len, hidden_size, "shift");

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(dtype, scale.scalar_type(), "fused_scale_shift_cpu", [&] {
    fused_scale_shift_kernel_impl<scalar_t, param_t>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        scale.data_ptr<param_t>(),
        shift.data_ptr<param_t>(),
        batch_size,
        seq_len,
        hidden_size,
        input.stride(0),
        input.stride(1),
        scale_params,
        shift_params,
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
  const auto st = input.scalar_type();

  TORCH_CHECK(
      norm_type == "rms" || norm_type == "layer", "norm_type must be \"rms\" or \"layer\", got ", norm_type, ".");

  TORCH_CHECK(scale.scalar_type() == shift.scalar_type(), "scale dtype must match shift dtype.");

  const int64_t B = input.size(0);
  const int64_t S = input.size(1);
  const int64_t D = input.size(2);

  if (weight.has_value()) {
    CHECK_INPUT_SHAPE_DTYPE<false>(weight.value(), {D}, st);
  }

  if (bias.has_value()) {
    TORCH_CHECK(norm_type == "layer", "bias is only supported for LayerNorm.");
    CHECK_INPUT_SHAPE_DTYPE<false>(bias.value(), {D}, st);
  }

  const auto scale_params = make_scale_shift_broadcast_params(scale, B, S, D, "scale");
  const auto shift_params = make_scale_shift_broadcast_params(shift, B, S, D, "shift");

  NormParams p{input, static_cast<float>(eps)};

  p.weight = weight.has_value() ? weight.value().data_ptr() : nullptr;

  p.bias = bias.has_value() ? bias.value().data_ptr() : nullptr;

  at::Tensor output = at::empty(input.sizes(), input.options());

  if (input.numel() == 0) {
    return output;
  }

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(st, scale.scalar_type(), "fused_norm_scale_shift_cpu", [&] {
    if (norm_type == "rms") {
      fused_norm_scale_shift_kernel_impl<NormMode::RMSNorm, scalar_t, param_t>(
          output.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          scale.data_ptr<param_t>(),
          shift.data_ptr<param_t>(),
          p,
          scale_params,
          shift_params);
    } else {
      fused_norm_scale_shift_kernel_impl<NormMode::LayerNorm, scalar_t, param_t>(
          output.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          scale.data_ptr<param_t>(),
          shift.data_ptr<param_t>(),
          p,
          scale_params,
          shift_params);
    }
  });

  return output;
}
std::tuple<at::Tensor, at::Tensor> fused_scale_residual_norm_scale_shift_cpu(
    const at::Tensor& residual,
    const at::Tensor& input,
    const std::optional<at::Tensor>& gate,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& scale,
    const at::Tensor& shift,
    const std::string& norm_type,
    double eps) {
  CHECK_INPUT_ND<3>(input);
  CHECK_INPUT_ND<3>(residual);

  const auto st = input.scalar_type();

  TORCH_CHECK(
      residual.sizes() == input.sizes(),
      "residual shape must match input shape, got ",
      residual.sizes(),
      " and ",
      input.sizes(),
      ".");

  TORCH_CHECK(residual.scalar_type() == st, "residual dtype must match input dtype.");

  TORCH_CHECK(
      norm_type == "rms" || norm_type == "layer", "norm_type must be \"rms\" or \"layer\", got ", norm_type, ".");

  const int64_t B = input.size(0);
  const int64_t S = input.size(1);
  const int64_t D = input.size(2);

  TORCH_CHECK(scale.scalar_type() == shift.scalar_type(), "scale dtype must match shift dtype.");

  if (weight.has_value()) {
    CHECK_INPUT_SHAPE_DTYPE<false>(weight.value(), {D}, st);
  }

  if (bias.has_value()) {
    TORCH_CHECK(norm_type == "layer", "bias is only supported for LayerNorm.");
    CHECK_INPUT_SHAPE_DTYPE<false>(bias.value(), {D}, st);
  }

  const ScaleShiftBroadcastParams scale_params = make_scale_shift_broadcast_params(scale, B, S, D, "scale");
  const ScaleShiftBroadcastParams shift_params = make_scale_shift_broadcast_params(shift, B, S, D, "shift");
  ScaleShiftBroadcastParams gate_params;

  if (gate.has_value()) {
    TORCH_CHECK(gate->scalar_type() == scale.scalar_type(), "gate dtype must match scale dtype.");
    gate_params = make_scale_shift_broadcast_params(gate.value(), B, S, D, "gate");
  }
  NormParams p{input, static_cast<float>(eps)};

  p.weight = weight.has_value() ? weight.value().data_ptr() : nullptr;

  p.bias = bias.has_value() ? bias.value().data_ptr() : nullptr;

  at::Tensor output = at::empty(input.sizes(), input.options());

  at::Tensor residual_output = at::empty(input.sizes(), input.options());

  if (input.numel() == 0) {
    return {
        output,
        residual_output,
    };
  }

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(st, scale.scalar_type(), "fused_scale_residual_norm_scale_shift_cpu", [&] {
    const param_t* gate_ptr = gate.has_value() ? gate->data_ptr<param_t>() : nullptr;
    if (norm_type == "rms") {
      fused_scale_residual_norm_scale_shift_kernel_impl<NormMode::RMSNorm, scalar_t, param_t>(
          output.data_ptr<scalar_t>(),
          residual_output.data_ptr<scalar_t>(),
          residual.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          gate_ptr,
          scale.data_ptr<param_t>(),
          shift.data_ptr<param_t>(),
          residual.stride(0),
          residual.stride(1),
          p,
          gate_params,
          scale_params,
          shift_params);
    } else {
      fused_scale_residual_norm_scale_shift_kernel_impl<NormMode::LayerNorm, scalar_t, param_t>(
          output.data_ptr<scalar_t>(),
          residual_output.data_ptr<scalar_t>(),
          residual.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          gate_ptr,
          scale.data_ptr<param_t>(),
          shift.data_ptr<param_t>(),
          residual.stride(0),
          residual.stride(1),
          p,
          gate_params,
          scale_params,
          shift_params);
    }
  });

  return {output, residual_output};
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
