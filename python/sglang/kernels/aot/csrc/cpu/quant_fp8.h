#pragma once

#include <limits>

#include "common.h"
#include "vec.h"

constexpr auto FP8_E4M3_MAX = std::numeric_limits<at::Float8_e4m3fn>::max();

// out = clamp(input * rscale) in FP8 E4M3; size must be a multiple of 32.
template <CpuIsa isa>
void quantize_row_fp8_scaled(
    at::Float8_e4m3fn* __restrict__ out, float rscale, const at::BFloat16* __restrict__ input, int size) {
  const float fp8_max = static_cast<float>(FP8_E4M3_MAX);
  const __m512 upper = _mm512_set1_ps(fp8_max);
  const __m512 lower = _mm512_set1_ps(-fp8_max);
  const __m512 rscale_vec = _mm512_set1_ps(rscale);
  for (int d = 0; d < size; d += 32) {
    __m512i packed = _mm512_loadu_si512(input + d);
    __m512 values0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 0));
    __m512 values1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(packed, 1));
    values0 = _mm512_min_ps(upper, _mm512_max_ps(lower, _mm512_mul_ps(values0, rscale_vec)));
    values1 = _mm512_min_ps(upper, _mm512_max_ps(lower, _mm512_mul_ps(values1, rscale_vec)));
    __m128i quants0 = CVT_FP32_TO_FP8(values0);
    __m128i quants1 = CVT_FP32_TO_FP8(values1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + d), _mm256_set_m128i(quants1, quants0));
  }
}

template <CpuIsa isa>
void quantize_row_fp8(
    at::Float8_e4m3fn* __restrict__ out, float& scale, const at::BFloat16* __restrict__ input, int size) {
  const float fp8_max = static_cast<float>(FP8_E4M3_MAX);
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
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
  quantize_row_fp8_scaled<isa>(out, scale == 0.f ? 0.f : 1.f / scale, input, size);
}
