#include "common.h"
#include "vec.h"

namespace {

template <CpuIsa arch>
inline __attribute__((always_inline)) void quantize_fp8_rows(
    at::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    const at::BFloat16* __restrict__ input,
    int64_t begin,
    int64_t end,
    int64_t columns,
    int64_t stride) {
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
  const __m512 upper = _mm512_set1_ps(448.f);
  const __m512 lower = _mm512_set1_ps(-448.f);
  for (int64_t row = begin; row < end; ++row) {
    const at::BFloat16* row_input = input + row * stride;
    at::Float8_e4m3fn* row_output = output + row * columns;
    __m512 maximum0 = _mm512_setzero_ps();
    __m512 maximum1 = _mm512_setzero_ps();
    for (int64_t column = 0; column < columns; column += 32) {
      __m512i packed = _mm512_loadu_si512(row_input + column);
      __m512 values0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 0));
      __m512 values1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 1));
      maximum0 = _mm512_max_ps(maximum0, _mm512_andnot_ps(sign_mask, values0));
      maximum1 = _mm512_max_ps(maximum1, _mm512_andnot_ps(sign_mask, values1));
    }
    float maximum = _mm512_reduce_max_ps(_mm512_max_ps(maximum0, maximum1));
    float scale = std::max(maximum / 448.f, std::numeric_limits<float>::epsilon());
    scales[row] = scale;
    __m512 inverse_scale = _mm512_set1_ps(1.f / scale);
    for (int64_t column = 0; column < columns; column += 32) {
      __m512i packed = _mm512_loadu_si512(row_input + column);
      __m512 values0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 0));
      __m512 values1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 1));
      values0 = _mm512_min_ps(upper, _mm512_max_ps(lower, _mm512_mul_ps(values0, inverse_scale)));
      values1 = _mm512_min_ps(upper, _mm512_max_ps(lower, _mm512_mul_ps(values1, inverse_scale)));
      __m128i quantized0 = CVT_FP32_TO_FP8(arch, values0);
      __m128i quantized1 = CVT_FP32_TO_FP8(arch, values1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(row_output + column), _mm256_set_m128i(quantized1, quantized0));
    }
  }
}

template <CpuIsa arch>
void per_token_quant_fp8_kernel_impl(
    at::Float8_e4m3fn* output,
    float* scales,
    const at::BFloat16* input,
    int64_t rows,
    int64_t columns,
    int64_t stride) {
  at::parallel_for(0, rows, 0, [&](int64_t begin, int64_t end) {
    quantize_fp8_rows<arch>(output, scales, input, begin, end, columns, stride);
  });
}

#if defined(CPU_CAPABILITY_AVX10_2)
template <>
void per_token_quant_fp8_kernel_impl<CpuIsa::AVX10_2>(
    at::Float8_e4m3fn* output,
    float* scales,
    const at::BFloat16* input,
    int64_t rows,
    int64_t columns,
    int64_t stride) {
  at::parallel_for(0, rows, 0, [&](int64_t begin, int64_t end) __attribute__((target("avx10.2"))) {
    quantize_fp8_rows<CpuIsa::AVX10_2>(output, scales, input, begin, end, columns, stride);
  });
}
#endif

}  // namespace

std::tuple<at::Tensor, at::Tensor> per_token_quant_fp8_cpu(at::Tensor& input) {
  CHECK_CPU(input);
  CHECK_DIM(2, input);
  CHECK_LAST_DIM_CONTIGUOUS(input);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "per_token_quant_fp8: expect bfloat16 input.");
  const int64_t rows = input.size(0);
  const int64_t columns = input.size(1);
  TORCH_CHECK(columns > 0 && columns % 32 == 0, "per_token_quant_fp8: K must be a positive multiple of 32.");

  auto output = at::empty({rows, columns}, input.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({rows}, input.options().dtype(at::kFloat));
  CPU_DISPATCH_ARCH([&] {
    per_token_quant_fp8_kernel_impl<arch>(
        output.data_ptr<at::Float8_e4m3fn>(),
        scales.data_ptr<float>(),
        input.data_ptr<at::BFloat16>(),
        rows,
        columns,
        input.stride(0));
  });
  return std::make_tuple(output, scales);
}