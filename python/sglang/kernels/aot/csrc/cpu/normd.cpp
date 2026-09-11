#include "common.h"
#include "vec.h"

/*
 * [Note]: Fused norm kernels for diffusion models
 *
 * This file contains CPU kernels for fused normalization and modulation
 * operations used by diffusion models:
 *
 *   - fused_scale_shift_cpu:
 *       Applies scale-shift modulation:
 *         output = input * (scale_constant + scale) + shift.
 *
 *   - fused_norm_scale_shift_cpu:
 *       Applies RMSNorm or LayerNorm followed by scale-shift modulation.
 *
 *   - fused_scale_residual_norm_scale_shift_cpu:
 *       Fuses optional gated residual accumulation, normalization, and
 *       scale-shift modulation.
 */

namespace {

enum class DiffusionNormMode {
  RMSNorm,
  LayerNorm,
};

#define DISPATCH_DIFFUSION_NORM_TYPE(norm_type, name, ...)                                                \
  [&] {                                                                                                   \
    if ((norm_type) == "rms") {                                                                           \
      using norm_mode_t = std::integral_constant<DiffusionNormMode, DiffusionNormMode::RMSNorm>;          \
      return __VA_ARGS__(norm_mode_t{});                                                                  \
    }                                                                                                     \
    TORCH_CHECK((norm_type) == "layer", name, ": norm_type must be 'rms' or 'layer', got ", (norm_type)); \
    using norm_mode_t = std::integral_constant<DiffusionNormMode, DiffusionNormMode::LayerNorm>;          \
    return __VA_ARGS__(norm_mode_t{});                                                                    \
  }()

template <DiffusionNormMode M>
struct DiffusionNormTraits;

template <>
struct DiffusionNormTraits<DiffusionNormMode::RMSNorm> {
  static constexpr bool has_mean = false;
  static constexpr bool has_bias = false;
};

template <>
struct DiffusionNormTraits<DiffusionNormMode::LayerNorm> {
  static constexpr bool has_mean = true;
  static constexpr bool has_bias = true;
};

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

template <typename RowFn>
inline void parallel_for_rows(int64_t B, int64_t S, int64_t D, RowFn&& row_fn) {
  at::parallel_for(0, B * S, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      row_fn(row / S, row % S, row * D);
    }
  });
}

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

template <DiffusionNormMode M, typename scalar_t, typename param_t>
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

  if constexpr (DiffusionNormTraits<M>::has_mean) {
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

    if constexpr (DiffusionNormTraits<M>::has_mean) {
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

    if constexpr (DiffusionNormTraits<M>::has_bias) {
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

    if constexpr (DiffusionNormTraits<M>::has_mean) {
      x -= mean;
    }

    x *= rstd;

    if (weight != nullptr) {
      x *= weight[d];
    }

    if constexpr (DiffusionNormTraits<M>::has_bias) {
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

template <DiffusionNormMode M, typename scalar_t, typename param_t>
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
    if constexpr (DiffusionNormTraits<M>::has_mean) {
      sum_vec += x0 + x1;
    }
  }

#pragma GCC unroll 4
  for (; d < D; ++d) {
    const float x = static_cast<float>(input[d]);
    sum_sq += x * x;
    if constexpr (DiffusionNormTraits<M>::has_mean) {
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

template <DiffusionNormMode M, typename scalar_t, typename param_t>
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

    if constexpr (DiffusionNormTraits<M>::has_mean) {
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

    if constexpr (DiffusionNormTraits<M>::has_mean) {
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
}  // anonymous namespace
at::Tensor fused_scale_shift_cpu(
    const at::Tensor& input, const at::Tensor& scale, const at::Tensor& shift, double scale_constant) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_DIM(3, input);

  check_modulation_param(scale, input, "scale");
  check_modulation_param(shift, input, "shift");

  CHECK_EQ(scale.scalar_type(), shift.scalar_type());

  const int64_t B = input.size(0);
  const int64_t S = input.size(1);
  const int64_t D = input.size(2);

  // Output is contiguous even if input is only last-dim contiguous.
  at::Tensor output = at::empty(input.sizes(), input.options());

  if (input.numel() == 0) {
    return output;
  }

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(input.scalar_type(), scale.scalar_type(), "fused_scale_shift_cpu", [&] {
    const ModulationParam<param_t> scale_param(scale);
    const ModulationParam<param_t> shift_param(shift);

    const scalar_t* input_ptr = input.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();

    const int64_t input_stride_b = input.stride(0);
    const int64_t input_stride_s = input.stride(1);

    parallel_for_rows(B, S, D, [&](int64_t b, int64_t s, int64_t offset) {
      const scalar_t* input_row = input_ptr + b * input_stride_b + s * input_stride_s;
      fused_scale_shift_row<scalar_t, param_t>(
          output_ptr + offset,
          input_row,
          scale_param.row(b, s),
          shift_param.row(b, s),
          D,
          scale_param.stride_c,
          shift_param.stride_c,
          static_cast<float>(scale_constant));
    });
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
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_DIM(3, input);

  check_modulation_param(scale, input, "scale");
  check_modulation_param(shift, input, "shift");

  CHECK_EQ(scale.scalar_type(), shift.scalar_type());

  const int64_t B = input.size(0);
  const int64_t S = input.size(1);
  const int64_t D = input.size(2);

  const float* weight_ptr = get_norm_param_ptr(weight, D, "weight");
  const float* bias_ptr = get_norm_param_ptr(bias, D, "bias");

  at::Tensor output = at::empty(input.sizes(), input.options());

  if (input.numel() == 0) {
    return output;
  }

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(input.scalar_type(), scale.scalar_type(), "fused_norm_scale_shift_cpu", [&] {
    const ModulationParam<param_t> scale_param(scale);
    const ModulationParam<param_t> shift_param(shift);

    const scalar_t* input_ptr = input.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();

    const int64_t input_stride_b = input.stride(0);
    const int64_t input_stride_s = input.stride(1);

    DISPATCH_DIFFUSION_NORM_TYPE(norm_type, "fused_norm_scale_shift_cpu", [&](auto mode_tag) {
      constexpr DiffusionNormMode M = decltype(mode_tag)::value;

      if constexpr (!DiffusionNormTraits<M>::has_bias) {
        TORCH_CHECK(!bias.has_value(), "bias is only supported for LayerNorm.");
      }

      parallel_for_rows(B, S, D, [&](int64_t b, int64_t s, int64_t offset) {
        const scalar_t* input_row = input_ptr + b * input_stride_b + s * input_stride_s;

        fused_norm_scale_shift_row<M, scalar_t, param_t>(
            output_ptr + offset,
            input_row,
            weight_ptr,
            bias_ptr,
            scale_param.row(b, s),
            shift_param.row(b, s),
            D,
            scale_param.stride_c,
            shift_param.stride_c,
            static_cast<float>(eps));
      });
    });
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
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_DIM(3, input);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_DIM(3, residual);

  CHECK_EQ(residual.sizes(), input.sizes());
  CHECK_EQ(residual.scalar_type(), input.scalar_type());

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
  const float* bias_ptr = get_norm_param_ptr(bias, D, "bias");

  at::Tensor output = at::empty(input.sizes(), input.options());

  at::Tensor residual_output = at::empty(input.sizes(), input.options());

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

        const scalar_t* input_ptr = input.data_ptr<scalar_t>();

        const scalar_t* residual_ptr = residual.data_ptr<scalar_t>();

        scalar_t* output_ptr = output.data_ptr<scalar_t>();

        scalar_t* residual_output_ptr = residual_output.data_ptr<scalar_t>();

        const int64_t input_stride_b = input.stride(0);
        const int64_t input_stride_s = input.stride(1);

        const int64_t residual_stride_b = residual.stride(0);
        const int64_t residual_stride_s = residual.stride(1);

        DISPATCH_DIFFUSION_NORM_TYPE(norm_type, "fused_scale_residual_norm_scale_shift_cpu", [&](auto mode_tag) {
          constexpr DiffusionNormMode M = decltype(mode_tag)::value;
          if constexpr (!DiffusionNormTraits<M>::has_bias) {
            TORCH_CHECK(!bias.has_value(), "bias is only supported for LayerNorm.");
          }
          parallel_for_rows(B, S, D, [&](int64_t b, int64_t s, int64_t offset) {
            const scalar_t* input_row = input_ptr + b * input_stride_b + s * input_stride_s;

            const scalar_t* residual_row = residual_ptr + b * residual_stride_b + s * residual_stride_s;

            const scalar_t* gate_ptr = gate_param.row(b, s);

            const float* gate_fp32_ptr = gate_fp32_param.row(b, s);

            const int64_t gate_stride_c = gate_fp32_ptr != nullptr ? gate_fp32_param.stride_c : gate_param.stride_c;

            fused_scale_residual_norm_scale_shift_row<M, scalar_t, param_t>(
                output_ptr + offset,
                residual_output_ptr + offset,
                residual_row,
                input_row,
                gate_ptr,
                gate_fp32_ptr,
                weight_ptr,
                bias_ptr,
                scale_param.row(b, s),
                shift_param.row(b, s),
                D,
                gate_stride_c,
                scale_param.stride_c,
                shift_param.stride_c,
                static_cast<float>(eps));
          });
        });
      });

  return {output, residual_output};
}
#undef DISPATCH_DIFFUSION_NORM_TYPE
