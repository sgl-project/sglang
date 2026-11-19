#include "common.h"
#include "vec.h"

namespace {

constexpr auto FP8_E4M3_MAX = std::numeric_limits<at::Float8_e4m3fn>::max();

template <CpuIsa isa>
void quantize_row_fp8(
    at::Float8_e4m3fn* __restrict__ out, float& scale, const at::BFloat16* __restrict__ input, int size) {
  const float fp8_max = static_cast<float>(FP8_E4M3_MAX);
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
  const __m512 upper = _mm512_set1_ps(fp8_max);
  const __m512 lower = _mm512_set1_ps(-fp8_max);
  __m512 vmax = _mm512_setzero_ps();
  for (int d = 0; d < size; d += 32) {
    __m512i packed = _mm512_loadu_si512(input + d);
    __m512 values0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 0));
    __m512 values1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 1));
    vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(sign_mask, values0));
    vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(sign_mask, values1));
  }
  float maximum = _mm512_reduce_max_ps(vmax);
  scale = maximum / fp8_max;
  __m512 rscale = _mm512_set1_ps(scale == 0.f ? 0.f : 1.f / scale);
  for (int d = 0; d < size; d += 32) {
    __m512i packed = _mm512_loadu_si512(input + d);
    __m512 values0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 0));
    __m512 values1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 1));
    values0 = _mm512_min_ps(upper, _mm512_max_ps(lower, _mm512_mul_ps(values0, rscale)));
    values1 = _mm512_min_ps(upper, _mm512_max_ps(lower, _mm512_mul_ps(values1, rscale)));
    __m128i quants0 = CVT_FP32_TO_FP8(values0);
    __m128i quants1 = CVT_FP32_TO_FP8(values1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + d), _mm256_set_m128i(quants1, quants0));
  }
}

template <CpuIsa isa>
void per_token_quant_fp8_kernel_impl(
    at::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    const at::BFloat16* __restrict__ input,
    int64_t M,
    int64_t K,
    int64_t stride) {
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_fp8<isa>(output + m * K, scales[m], input + m * stride, K);
    }
  });
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> per_token_quant_fp8_cpu(at::Tensor& input) {
  CHECK_CPU(input);
  CHECK_DIM(2, input);
  CHECK_LAST_DIM_CONTIGUOUS(input);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "per_token_quant_fp8: expect bfloat16 input.");
  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  TORCH_CHECK(K > 0 && K % 32 == 0, "per_token_quant_fp8: K must be a positive multiple of 32.");

  auto output = at::empty({M, K}, input.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({M}, input.options().dtype(at::kFloat));
  CPU_DISPATCH_ISA([&] {
    per_token_quant_fp8_kernel_impl<isa>(
        output.data_ptr<at::Float8_e4m3fn>(),
        scales.data_ptr<float>(),
        input.data_ptr<at::BFloat16>(),
        M,
        K,
        input.stride(0));
  });
  return std::make_tuple(output, scales);
}
