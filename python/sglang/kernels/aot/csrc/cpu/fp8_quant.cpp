#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <type_traits>

#include "common.h"
#include "vec.h"

namespace {

constexpr float kFP8E4M3Max = 448.0f;
constexpr int64_t kMXFP8GroupSize = 32;

#if defined(CPU_CAPABILITY_AVX512)
constexpr int64_t kFP8QuantVecSize = 16;

template <typename scalar_t>
inline __m512 load_fp32_vec16(const scalar_t* __restrict__ data) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return _mm512_loadu_ps(data);
  } else {
    return __m512(load_float_vec(data));
  }
}

// NaN lanes become 0 so the vector absmax matches the scalar std::max path.
inline __m512 abs_ordered_fp8_quant_avx512(__m512 value) {
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
  const __m512 zero = _mm512_setzero_ps();
  __m512 abs_value = _mm512_andnot_ps(sign_mask, value);
  return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(abs_value, abs_value, _CMP_ORD_Q), zero, abs_value);
}

inline void store_fp8_quantized_avx512(at::Float8_e4m3fn* __restrict__ output, __m512 value) {
  const __m512 quant_min_vec = _mm512_set1_ps(-kFP8E4M3Max);
  const __m512 quant_max_vec = _mm512_set1_ps(kFP8E4M3Max);
  alignas(64) float values[kFP8QuantVecSize];
  const __mmask16 ordered = _mm512_cmp_ps_mask(value, value, _CMP_ORD_Q);
  const __m512 clipped = _mm512_min_ps(_mm512_max_ps(value, quant_min_vec), quant_max_vec);
  value = _mm512_mask_mov_ps(value, ordered, clipped);
  _mm512_store_ps(values, value);
  for (int64_t i = 0; i < kFP8QuantVecSize; ++i) {
    output[i] = static_cast<at::Float8_e4m3fn>(values[i]);
  }
}
#endif

template <typename scalar_t>
inline float absmax_row(const scalar_t* __restrict__ input, int64_t size, float init) {
  float absmax = init;
  int64_t i = 0;
#if defined(CPU_CAPABILITY_AVX512)
  __m512 absmax_vec = _mm512_set1_ps(init);
  for (; i <= size - kFP8QuantVecSize; i += kFP8QuantVecSize) {
    absmax_vec = _mm512_max_ps(absmax_vec, abs_ordered_fp8_quant_avx512(load_fp32_vec16(input + i)));
  }
  absmax = _mm512_reduce_max_ps(absmax_vec);
#endif
  for (; i < size; ++i) {
    absmax = std::max(absmax, std::abs(static_cast<float>(input[i])));
  }
  return absmax;
}

template <typename scalar_t>
inline void quantize_row_fp8(
    at::Float8_e4m3fn* __restrict__ output, const scalar_t* __restrict__ input, int64_t size, float scale) {
  int64_t i = 0;
#if defined(CPU_CAPABILITY_AVX512)
  const __m512 scale_vec = _mm512_set1_ps(scale);
  for (; i <= size - kFP8QuantVecSize; i += kFP8QuantVecSize) {
    store_fp8_quantized_avx512(output + i, _mm512_div_ps(load_fp32_vec16(input + i), scale_vec));
  }
#endif
  for (; i < size; ++i) {
    const float value = std::clamp(static_cast<float>(input[i]) / scale, -kFP8E4M3Max, kFP8E4M3Max);
    output[i] = static_cast<at::Float8_e4m3fn>(value);
  }
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
void per_token_group_quant_fp8_kernel_impl(
    at::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    const scalar_t* __restrict__ input,
    int64_t num_groups,
    int64_t group_size,
    float eps) {
  at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t group = begin; group < end; ++group) {
      const int64_t offset = group * group_size;
      const float scale = absmax_row(input + offset, group_size, eps) / kFP8E4M3Max;
      scales[group] = scale;
      quantize_row_fp8(output + offset, input + offset, group_size, scale);
    }
  });
}

template <typename scalar_t>
void static_per_tensor_quant_fp8_kernel_impl(
    at::Float8_e4m3fn* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t num_rows,
    int64_t row_size,
    float scale) {
  at::parallel_for(0, num_rows, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      quantize_row_fp8(output + row * row_size, input + row * row_size, row_size, scale);
    }
  });
}

template <typename scalar_t>
void dynamic_per_tensor_quant_fp8_kernel_impl(
    at::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scale,
    const scalar_t* __restrict__ input,
    int64_t num_rows,
    int64_t row_size) {
  constexpr float eps = 1.0e-12f;
  const float absmax = at::parallel_reduce(
      0,
      num_rows,
      0,
      eps,
      [&](int64_t begin, int64_t end, float local) {
        for (int64_t row = begin; row < end; ++row) {
          local = std::max(local, absmax_row(input + row * row_size, row_size, eps));
        }
        return local;
      },
      [](float a, float b) { return std::max(a, b); });
  scale[0] = absmax / kFP8E4M3Max;
  static_per_tensor_quant_fp8_kernel_impl(output, input, num_rows, row_size, scale[0]);
}

template <typename scalar_t>
void mxfp8_group_quant_kernel_impl(
    at::Float8_e4m3fn* __restrict__ output,
    uint8_t* __restrict__ scales,
    const scalar_t* __restrict__ input,
    int64_t num_groups) {
  at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t group = begin; group < end; ++group) {
      const int64_t offset = group * kMXFP8GroupSize;
      const uint8_t scale_byte = ceil_to_ue8m0_byte(absmax_row(input + offset, kMXFP8GroupSize, 0.0f) / kFP8E4M3Max);
      scales[group] = scale_byte;
      quantize_row_fp8(output + offset, input + offset, kMXFP8GroupSize, ue8m0_byte_to_float(scale_byte));
    }
  });
}

inline void check_fp8_quant_input(const at::Tensor& input, const char* op_name) {
  TORCH_CHECK(input.device().is_cpu(), op_name, ": input must be a CPU tensor");
  TORCH_CHECK(input.is_contiguous(), op_name, ": input must be contiguous");
  TORCH_CHECK(input.dim() == 2, op_name, ": input must be 2D");
  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(
      input_dtype == at::kBFloat16 || input_dtype == at::kHalf || input_dtype == at::kFloat,
      op_name,
      ": input must be bfloat16, float16, or float32");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor>
sgl_per_token_group_quant_8bit_cpu(const at::Tensor& input, int64_t group_size, double eps) {
  TORCH_CHECK(input.device().is_cpu(), "sgl_per_token_group_quant_8bit_cpu: input must be a CPU tensor");
  TORCH_CHECK(input.is_contiguous(), "sgl_per_token_group_quant_8bit_cpu: input must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "sgl_per_token_group_quant_8bit_cpu: input must have at least 2 dimensions");
  TORCH_CHECK(group_size > 0, "sgl_per_token_group_quant_8bit_cpu: group_size must be positive");
  TORCH_CHECK(
      input.size(-1) % group_size == 0,
      "sgl_per_token_group_quant_8bit_cpu: input last dimension must be divisible by group_size");
  TORCH_CHECK(eps > 0.0, "sgl_per_token_group_quant_8bit_cpu: eps must be positive");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(input_dtype == at::kBFloat16, "sgl_per_token_group_quant_8bit_cpu: input must be bfloat16");

  const int64_t num_groups = input.numel() / group_size;
  auto quantized = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
  auto scale_sizes = input.sizes().vec();
  scale_sizes.back() = input.size(-1) / group_size;
  auto scales = at::empty(scale_sizes, input.options().dtype(at::kFloat));

  per_token_group_quant_fp8_kernel_impl<at::BFloat16>(
      quantized.data_ptr<at::Float8_e4m3fn>(),
      scales.data_ptr<float>(),
      input.data_ptr<at::BFloat16>(),
      num_groups,
      group_size,
      static_cast<float>(eps));

  return std::make_tuple(quantized, scales);
}

std::tuple<at::Tensor, at::Tensor> sgl_per_token_quant_fp8_cpu(const at::Tensor& input) {
  check_fp8_quant_input(input, "sgl_per_token_quant_fp8_cpu");

  const int64_t num_rows = input.size(0);
  const int64_t row_size = input.size(1);
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
  auto scale = at::empty({num_rows, 1}, input.options().dtype(at::kFloat));

  AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(at::kFloat, input.scalar_type(), "sgl_per_token_quant_fp8_cpu", [&] {
    per_token_group_quant_fp8_kernel_impl<scalar_t>(
        output.data_ptr<at::Float8_e4m3fn>(),
        scale.data_ptr<float>(),
        input.data_ptr<scalar_t>(),
        num_rows,
        row_size,
        1.0e-12f);
  });
  return std::make_tuple(output, scale);
}

// A given `scale` means static quantization; otherwise the per-tensor scale is computed.
std::tuple<at::Tensor, at::Tensor>
sgl_per_tensor_quant_fp8_cpu(const at::Tensor& input, const std::optional<at::Tensor>& scale_opt) {
  check_fp8_quant_input(input, "sgl_per_tensor_quant_fp8_cpu");

  const int64_t num_rows = input.size(0);
  const int64_t row_size = input.size(1);
  const bool is_static = scale_opt.has_value();
  at::Tensor scale;
  float static_scale = 0.0f;
  if (is_static) {
    scale = scale_opt.value();
    TORCH_CHECK(scale.device().is_cpu(), "sgl_per_tensor_quant_fp8_cpu: scale must be a CPU tensor");
    TORCH_CHECK(scale.numel() == 1, "sgl_per_tensor_quant_fp8_cpu: static scale must be scalar");
    static_scale = scale.item<float>();
  } else {
    scale = at::empty({1}, input.options().dtype(at::kFloat));
  }
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));

  AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(at::kFloat, input.scalar_type(), "sgl_per_tensor_quant_fp8_cpu", [&] {
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<at::Float8_e4m3fn>();
    if (is_static) {
      static_per_tensor_quant_fp8_kernel_impl<scalar_t>(output_data, input_data, num_rows, row_size, static_scale);
    } else {
      dynamic_per_tensor_quant_fp8_kernel_impl<scalar_t>(
          output_data, scale.data_ptr<float>(), input_data, num_rows, row_size);
    }
  });
  return std::make_tuple(output, scale);
}

std::tuple<at::Tensor, at::Tensor> mxfp8_group_quantize_cpu(const at::Tensor& input) {
  check_fp8_quant_input(input, "mxfp8_group_quantize_cpu");
  TORCH_CHECK(
      input.size(1) % kMXFP8GroupSize == 0, "mxfp8_group_quantize_cpu: input K dimension must be divisible by 32");

  const int64_t num_groups = input.numel() / kMXFP8GroupSize;
  auto scale = at::empty({input.size(0), input.size(1) / kMXFP8GroupSize}, input.options().dtype(at::kByte));
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));

  AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(at::kFloat, input.scalar_type(), "mxfp8_group_quantize_cpu", [&] {
    mxfp8_group_quant_kernel_impl<scalar_t>(
        output.data_ptr<at::Float8_e4m3fn>(), scale.data_ptr<uint8_t>(), input.data_ptr<scalar_t>(), num_groups);
  });
  return std::make_tuple(output, scale);
}
