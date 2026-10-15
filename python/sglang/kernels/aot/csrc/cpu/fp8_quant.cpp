#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <type_traits>

#include "common.h"
#include "vec.h"

#if defined(CPU_CAPABILITY_AVX512)
constexpr int64_t kFP8QuantVecSize = 16;

inline __m512 abs_ordered_fp8_quant_avx512(__m512 value) {
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
  const __m512 zero = _mm512_setzero_ps();
  __m512 abs_value = _mm512_andnot_ps(sign_mask, value);
  return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(abs_value, abs_value, _CMP_ORD_Q), zero, abs_value);
}

inline void store_fp8_quantized_avx512(
    at::Float8_e4m3fn* __restrict__ output, int64_t offset, __m512 value, __m512 quant_min_vec, __m512 quant_max_vec) {
  alignas(64) float values[kFP8QuantVecSize];
  const __mmask16 ordered = _mm512_cmp_ps_mask(value, value, _CMP_ORD_Q);
  const __m512 clipped = _mm512_min_ps(_mm512_max_ps(value, quant_min_vec), quant_max_vec);
  value = _mm512_mask_mov_ps(value, ordered, clipped);
  _mm512_store_ps(values, value);
  for (int64_t i = 0; i < kFP8QuantVecSize; ++i) {
    output[offset + i] = static_cast<at::Float8_e4m3fn>(values[i]);
  }
}
#endif

void quantize_groups_to_fp8_kernel(
    const at::BFloat16* __restrict__ input,
    at::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    int64_t num_groups,
    int64_t group_size,
    float eps) {
  constexpr float quant_max = 448.0f;
#if defined(CPU_CAPABILITY_AVX512)
  const __m512 quant_min_vec = _mm512_set1_ps(-quant_max);
  const __m512 quant_max_vec = _mm512_set1_ps(quant_max);

  at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t group = begin; group < end; ++group) {
      const int64_t offset = group * group_size;
      __m512 absmax_vec = _mm512_set1_ps(eps);
      int64_t i = 0;
      for (; i <= group_size - kFP8QuantVecSize; i += kFP8QuantVecSize) {
        __m512 abs_vec = abs_ordered_fp8_quant_avx512(__m512(load_float_vec(input + offset + i)));
        absmax_vec = _mm512_max_ps(absmax_vec, abs_vec);
      }
      float absmax = std::max(eps, _mm512_reduce_max_ps(absmax_vec));
      for (; i < group_size; ++i) {
        absmax = std::max(absmax, std::abs(static_cast<float>(input[offset + i])));
      }

      const float scale = absmax / quant_max;
      scales[group] = scale;
      const __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale);
      i = 0;
      for (; i <= group_size - kFP8QuantVecSize; i += kFP8QuantVecSize) {
        store_fp8_quantized_avx512(
            output,
            offset + i,
            _mm512_mul_ps(__m512(load_float_vec(input + offset + i)), inv_scale_vec),
            quant_min_vec,
            quant_max_vec);
      }
      for (; i < group_size; ++i) {
        const float value = std::clamp(static_cast<float>(input[offset + i]) / scale, -quant_max, quant_max);
        output[offset + i] = static_cast<at::Float8_e4m3fn>(value);
      }
    }
  });
#else
  at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t group = begin; group < end; ++group) {
      const int64_t offset = group * group_size;
      float absmax = eps;
      for (int64_t i = 0; i < group_size; ++i) {
        absmax = std::max(absmax, std::abs(static_cast<float>(input[offset + i])));
      }
      const float scale = absmax / quant_max;
      scales[group] = scale;
      for (int64_t i = 0; i < group_size; ++i) {
        const float value = std::clamp(static_cast<float>(input[offset + i]) / scale, -quant_max, quant_max);
        output[offset + i] = static_cast<at::Float8_e4m3fn>(value);
      }
    }
  });
#endif
}

template <typename scalar_t>
void static_quant_to_fp8_kernel(
    const scalar_t* __restrict__ input, at::Float8_e4m3fn* __restrict__ output, int64_t numel, float scale) {
  constexpr float quant_max = 448.0f;
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    const __m512 quant_min_vec = _mm512_set1_ps(-quant_max);
    const __m512 quant_max_vec = _mm512_set1_ps(quant_max);
    const __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale);

    at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
      int64_t i = begin;
      for (; i <= end - kFP8QuantVecSize; i += kFP8QuantVecSize) {
        store_fp8_quantized_avx512(
            output, i, _mm512_mul_ps(__m512(load_float_vec(input + i)), inv_scale_vec), quant_min_vec, quant_max_vec);
      }
      for (; i < end; ++i) {
        const float value = std::clamp(static_cast<float>(input[i]) / scale, -quant_max, quant_max);
        output[i] = static_cast<at::Float8_e4m3fn>(value);
      }
    });
  } else {
    at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const float value = std::clamp(static_cast<float>(input[i]) / scale, -quant_max, quant_max);
        output[i] = static_cast<at::Float8_e4m3fn>(value);
      }
    });
  }
#else
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const float value = std::clamp(static_cast<float>(input[i]) / scale, -quant_max, quant_max);
      output[i] = static_cast<at::Float8_e4m3fn>(value);
    }
  });
#endif
}

template <typename scalar_t>
void dynamic_tensor_quant_to_fp8_kernel(
    const scalar_t* __restrict__ input,
    at::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scale,
    int64_t numel) {
  constexpr float quant_max = 448.0f;
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    __m512 absmax_vec = _mm512_set1_ps(1.0e-12f);
    int64_t i = 0;
    for (; i <= numel - kFP8QuantVecSize; i += kFP8QuantVecSize) {
      __m512 abs_vec = abs_ordered_fp8_quant_avx512(__m512(load_float_vec(input + i)));
      absmax_vec = _mm512_max_ps(absmax_vec, abs_vec);
    }
    float absmax = std::max(1.0e-12f, _mm512_reduce_max_ps(absmax_vec));
    for (; i < numel; ++i) {
      absmax = std::max(absmax, std::abs(static_cast<float>(input[i])));
    }
    scale[0] = absmax / quant_max;
    static_quant_to_fp8_kernel(input, output, numel, scale[0]);
  } else {
    float absmax = 1.0e-12f;
    for (int64_t i = 0; i < numel; ++i) {
      absmax = std::max(absmax, std::abs(static_cast<float>(input[i])));
    }
    scale[0] = absmax / quant_max;
    static_quant_to_fp8_kernel(input, output, numel, scale[0]);
  }
#else
  float absmax = 1.0e-12f;
  for (int64_t i = 0; i < numel; ++i) {
    absmax = std::max(absmax, std::abs(static_cast<float>(input[i])));
  }
  scale[0] = absmax / quant_max;
  static_quant_to_fp8_kernel(input, output, numel, scale[0]);
#endif
}

std::tuple<at::Tensor, at::Tensor>
per_token_group_quant_fp8_cpu(const at::Tensor& input, int64_t group_size, double eps) {
  TORCH_CHECK(input.device().is_cpu(), "per_token_group_quant_fp8_cpu: input must be a CPU tensor");
  TORCH_CHECK(input.is_contiguous(), "per_token_group_quant_fp8_cpu: input must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "per_token_group_quant_fp8_cpu: input must have at least 2 dimensions");
  TORCH_CHECK(group_size > 0, "per_token_group_quant_fp8_cpu: group_size must be positive");
  TORCH_CHECK(
      input.size(-1) % group_size == 0,
      "per_token_group_quant_fp8_cpu: input last dimension must be divisible by group_size");
  TORCH_CHECK(eps > 0.0, "per_token_group_quant_fp8_cpu: eps must be positive");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(input_dtype == at::kBFloat16, "per_token_group_quant_fp8_cpu: input must be bfloat16");

  const int64_t num_groups = input.numel() / group_size;
  auto quantized = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
  auto scale_sizes = input.sizes().vec();
  scale_sizes.back() = input.size(-1) / group_size;
  auto scales = at::empty(scale_sizes, input.options().dtype(at::kFloat));

  quantize_groups_to_fp8_kernel(
      input.data_ptr<at::BFloat16>(),
      quantized.data_ptr<at::Float8_e4m3fn>(),
      scales.data_ptr<float>(),
      num_groups,
      group_size,
      static_cast<float>(eps));

  return std::make_tuple(quantized, scales);
}

std::tuple<at::Tensor, at::Tensor> scaled_fp8_quant_cpu(
    const at::Tensor& input,
    const std::optional<at::Tensor>& scale_opt,
    int64_t num_token_padding,
    bool use_per_token_if_dynamic) {
  TORCH_CHECK(input.device().is_cpu(), "scaled_fp8_quant_cpu: input must be a CPU tensor");
  TORCH_CHECK(input.is_contiguous(), "scaled_fp8_quant_cpu: input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "scaled_fp8_quant_cpu: input must be 2D");
  TORCH_CHECK(num_token_padding >= 0, "scaled_fp8_quant_cpu: num_token_padding must be non-negative");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(
      input_dtype == at::kBFloat16 || input_dtype == at::kHalf || input_dtype == at::kFloat,
      "scaled_fp8_quant_cpu: input must be bfloat16, float16, or float32");

  const int64_t num_rows = input.size(0);
  const int64_t row_size = input.size(1);
  const int64_t output_rows = std::max(num_rows, num_token_padding);
  at::Tensor scale;
  at::Tensor quantized_input;

  if (scale_opt.has_value()) {
    scale = scale_opt.value();
    TORCH_CHECK(scale.device().is_cpu(), "scaled_fp8_quant_cpu: scale must be a CPU tensor");
    TORCH_CHECK(scale.numel() == 1, "scaled_fp8_quant_cpu: static scale must be scalar");
    quantized_input = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
    const float scale_value = scale.item<float>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input_dtype, "scaled_fp8_quant_cpu_static", [&] {
      static_quant_to_fp8_kernel<scalar_t>(
          input.data_ptr<scalar_t>(), quantized_input.data_ptr<at::Float8_e4m3fn>(), input.numel(), scale_value);
    });
  } else if (use_per_token_if_dynamic) {
    auto result = per_token_group_quant_fp8_cpu(input, row_size, 1.0e-12);
    quantized_input = std::get<0>(result).reshape({num_rows, row_size});
    scale = std::get<1>(result).reshape({num_rows, 1});
  } else {
    scale = at::empty({1}, input.options().dtype(at::kFloat));
    quantized_input = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input_dtype, "scaled_fp8_quant_cpu_dynamic", [&] {
      dynamic_tensor_quant_to_fp8_kernel<scalar_t>(
          input.data_ptr<scalar_t>(),
          quantized_input.data_ptr<at::Float8_e4m3fn>(),
          scale.data_ptr<float>(),
          input.numel());
    });
  }

  if (output_rows == num_rows) {
    return std::make_tuple(quantized_input, scale);
  }

  auto output = at::zeros({output_rows, row_size}, input.options().dtype(at::kFloat8_e4m3fn));
  output.narrow(0, 0, num_rows).copy_(quantized_input);
  if (!scale_opt.has_value() && use_per_token_if_dynamic) {
    auto padded_scale = at::zeros({output_rows, 1}, input.options().dtype(at::kFloat));
    padded_scale.narrow(0, 0, num_rows).copy_(scale);
    scale = padded_scale;
  }
  return std::make_tuple(output, scale);
}

inline uint8_t ceil_to_ue8m0_byte(float value) {
  if (!(value > 0.0f) || !std::isfinite(value)) {
    return 1;
  }
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t exponent = (bits >> 23) & 0xFF;
  const uint32_t mantissa = bits & 0x7FFFFF;
  exponent += mantissa != 0;
  exponent = std::min<uint32_t>(std::max<uint32_t>(exponent, 1), 254);
  return static_cast<uint8_t>(exponent);
}

inline float ue8m0_byte_to_float(uint8_t scale) {
  const uint32_t bits = static_cast<uint32_t>(scale) << 23;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename scalar_t>
void mxfp8_quantize_groups_kernel(
    const scalar_t* __restrict__ input,
    at::Float8_e4m3fn* __restrict__ output,
    uint8_t* __restrict__ scale,
    int64_t num_groups) {
  constexpr int64_t group_size = 32;
  constexpr float quant_max = 448.0f;
#if defined(CPU_CAPABILITY_AVX512)
  if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 quant_min_vec = _mm512_set1_ps(-quant_max);
    const __m512 quant_max_vec = _mm512_set1_ps(quant_max);

    at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
      for (int64_t group = begin; group < end; ++group) {
        const int64_t offset = group * group_size;
        __m512 absmax_vec = zero;
        for (int64_t i = 0; i < group_size; i += kFP8QuantVecSize) {
          __m512 abs_vec = abs_ordered_fp8_quant_avx512(__m512(load_float_vec(input + offset + i)));
          absmax_vec = _mm512_max_ps(absmax_vec, abs_vec);
        }
        const float absmax = _mm512_reduce_max_ps(absmax_vec);
        const uint8_t scale_byte = ceil_to_ue8m0_byte(absmax / quant_max);
        const float scale_float = ue8m0_byte_to_float(scale_byte);
        scale[group] = scale_byte;
        const __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale_float);
        for (int64_t i = 0; i < group_size; i += kFP8QuantVecSize) {
          store_fp8_quantized_avx512(
              output,
              offset + i,
              _mm512_mul_ps(__m512(load_float_vec(input + offset + i)), inv_scale_vec),
              quant_min_vec,
              quant_max_vec);
        }
      }
    });
  } else {
    at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
      for (int64_t group = begin; group < end; ++group) {
        const int64_t offset = group * group_size;
        float absmax = 0.0f;
        for (int64_t i = 0; i < group_size; ++i) {
          absmax = std::max(absmax, std::abs(static_cast<float>(input[offset + i])));
        }
        const uint8_t scale_byte = ceil_to_ue8m0_byte(absmax / quant_max);
        const float scale_float = ue8m0_byte_to_float(scale_byte);
        scale[group] = scale_byte;
        for (int64_t i = 0; i < group_size; ++i) {
          const float value = std::clamp(static_cast<float>(input[offset + i]) / scale_float, -quant_max, quant_max);
          output[offset + i] = static_cast<at::Float8_e4m3fn>(value);
        }
      }
    });
  }
#else
  at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t group = begin; group < end; ++group) {
      const int64_t offset = group * group_size;
      float absmax = 0.0f;
      for (int64_t i = 0; i < group_size; ++i) {
        absmax = std::max(absmax, std::abs(static_cast<float>(input[offset + i])));
      }
      const uint8_t scale_byte = ceil_to_ue8m0_byte(absmax / quant_max);
      const float scale_float = ue8m0_byte_to_float(scale_byte);
      scale[group] = scale_byte;
      for (int64_t i = 0; i < group_size; ++i) {
        const float value = std::clamp(static_cast<float>(input[offset + i]) / scale_float, -quant_max, quant_max);
        output[offset + i] = static_cast<at::Float8_e4m3fn>(value);
      }
    }
  });
#endif
}

std::tuple<at::Tensor, at::Tensor> mxfp8_group_quantize_cpu(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu(), "mxfp8_group_quantize_cpu: input must be a CPU tensor");
  TORCH_CHECK(input.is_contiguous(), "mxfp8_group_quantize_cpu: input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "mxfp8_group_quantize_cpu: input must be 2D");
  TORCH_CHECK(input.size(1) % 32 == 0, "mxfp8_group_quantize_cpu: input K dimension must be divisible by 32");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(
      input_dtype == at::kBFloat16 || input_dtype == at::kHalf || input_dtype == at::kFloat,
      "mxfp8_group_quantize_cpu: input must be bfloat16, float16, or float32");

  constexpr int64_t group_size = 32;
  const int64_t num_groups = input.numel() / group_size;
  auto scale = at::empty({input.size(0), input.size(1) / group_size}, input.options().dtype(at::kByte));
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input_dtype, "mxfp8_group_quantize_cpu", [&] {
    mxfp8_quantize_groups_kernel<scalar_t>(
        input.data_ptr<scalar_t>(), output.data_ptr<at::Float8_e4m3fn>(), scale.data_ptr<uint8_t>(), num_groups);
  });
  return std::make_tuple(output, scale);
}
