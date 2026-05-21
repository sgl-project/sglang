#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/all.h>

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "common.h"
#include "vec.h"

namespace {

constexpr float kFP8Max = 448.0f;
constexpr float kFP8Min = -448.0f;
constexpr float kMinScaleAmax = 1.0e-4f;

template <bool round_scale, typename scalar_t>
void act_quant_cpu_impl(
    at::Float8_e4m3fn* __restrict__ y_ptr,
    float* __restrict__ scale_ptr,
    const scalar_t* __restrict__ x_ptr,
    int64_t block_size,
    int64_t M,
    int64_t N,
    int64_t num_groups) {
  const int64_t total_blocks = M * num_groups;
  at::parallel_for(0, total_blocks, 0, [&](int64_t begin, int64_t end) {
    int64_t m = 0;
    int64_t group = 0;
    data_index_init(begin, m, M, group, num_groups);

    for (int64_t offset = begin; offset < end; ++offset) {
      const int64_t base = m * N + group * block_size;

      float amax = 0.0f;
      for (int64_t d = 0; d < block_size; ++d) {
        amax = std::max(amax, std::abs(static_cast<float>(x_ptr[base + d])));
      }
      amax = std::max(amax, kMinScaleAmax);

      float scale_value;
      if constexpr (round_scale) {
        scale_value = std::exp2(std::ceil(std::log2(amax / kFP8Max)));
      } else {
        scale_value = amax / kFP8Max;
      }
      scale_ptr[offset] = scale_value;

      const float inv_scale = 1.0f / scale_value;
      for (int64_t d = 0; d < block_size; ++d) {
        float value = static_cast<float>(x_ptr[base + d]) * inv_scale;
        value = std::min(std::max(value, kFP8Min), kFP8Max);
        y_ptr[base + d] = at::Float8_e4m3fn(value);
      }

      data_index_step(m, M, group, num_groups);
    }
  });
}

#if defined(CPU_CAPABILITY_AVX512)

template <bool round_scale>
void act_quant_cpu_impl(
    at::Float8_e4m3fn* __restrict__ y_ptr,
    float* __restrict__ scale_ptr,
    const at::BFloat16* __restrict__ x_ptr,
    int64_t block_size,
    int64_t M,
    int64_t N,
    int64_t num_groups) {
  const int64_t total_blocks = M * num_groups;
  at::parallel_for(0, total_blocks, 0, [&](int64_t begin, int64_t end) {
    const __m512 signBit = _mm512_set1_ps(-0.0f);
    const __m512 vfp8_min = _mm512_set1_ps(kFP8Min);
    const __m512 vfp8_max = _mm512_set1_ps(kFP8Max);

    int64_t m = 0;
    int64_t group = 0;
    data_index_init(begin, m, M, group, num_groups);

    for (int64_t offset = begin; offset < end; ++offset) {
      const int64_t base = m * N + group * block_size;
      const at::BFloat16* __restrict__ x_block = x_ptr + base;
      auto* __restrict__ y_block = y_ptr + base;

      // Phase 1: vectorized absolute max reduction
      __m512 vamax0 = _mm512_setzero_ps();
      __m512 vamax1 = _mm512_setzero_ps();
      int64_t d = 0;
      for (; d + 32 <= block_size; d += 32) {
        __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(x_block + d));
        __m512 v0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(raw, 0));
        __m512 v1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(raw, 1));
        vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(signBit, v0));
        vamax1 = _mm512_max_ps(vamax1, _mm512_andnot_ps(signBit, v1));
      }
      float amax = _mm512_reduce_max_ps(_mm512_max_ps(vamax0, vamax1));
      for (; d < block_size; ++d) {
        amax = std::max(amax, std::abs(static_cast<float>(x_block[d])));
      }
      amax = std::max(amax, kMinScaleAmax);

      // Phase 2: compute scale
      float scale_value;
      if constexpr (round_scale) {
        scale_value = std::exp2(std::ceil(std::log2(amax / kFP8Max)));
      } else {
        scale_value = amax / kFP8Max;
      }
      scale_ptr[offset] = scale_value;

      // Phase 3: vectorized scale + scalar FP8 conversion
      const float inv_scale = 1.0f / scale_value;
      const __m512 vinv_scale = _mm512_set1_ps(inv_scale);
      const __m512 vfp8_min_local = vfp8_min;
      const __m512 vfp8_max_local = vfp8_max;
      d = 0;
      // Use vectorized scaling, then scalar FP8 cast to match reference exactly
      alignas(64) float scaled_buf[32];
      for (; d + 32 <= block_size; d += 32) {
        __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(x_block + d));
        __m512 v0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(raw, 0));
        __m512 v1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(raw, 1));

        v0 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v0, vinv_scale), vfp8_min_local), vfp8_max_local);
        v1 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v1, vinv_scale), vfp8_min_local), vfp8_max_local);

        _mm512_store_ps(scaled_buf, v0);
        _mm512_store_ps(scaled_buf + 16, v1);

        for (int64_t i = 0; i < 32; ++i) {
          y_block[d + i] = at::Float8_e4m3fn(scaled_buf[i]);
        }
      }
      // Scalar remainder
      for (; d < block_size; ++d) {
        float value = static_cast<float>(x_block[d]) * inv_scale;
        value = std::min(std::max(value, kFP8Min), kFP8Max);
        y_block[d] = at::Float8_e4m3fn(value);
      }

      data_index_step(m, M, group, num_groups);
    }
  });
}

#endif  // CPU_CAPABILITY_AVX512

}  // namespace

std::tuple<at::Tensor, at::Tensor>
act_quant_cpu(at::Tensor& x, int64_t block_size, const std::optional<std::string>& scale_fmt) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int64_t N = x.size(-1);
  TORCH_CHECK(N % block_size == 0, "Last dimension size must be divisible by block_size");

  const int64_t M = x.numel() / N;
  const int64_t num_groups = N / block_size;

  at::Tensor y = at::empty_like(x, x.options().dtype(at::kFloat8_e4m3fn));

  std::vector<int64_t> scale_sizes = x.sizes().vec();
  scale_sizes.back() = num_groups;
  at::Tensor scale = at::empty(scale_sizes, x.options().dtype(at::kFloat));

  const bool round_scale = scale_fmt.has_value();
  CPU_DISPATCH_FLOATING_TYPES(x.scalar_type(), "act_quant_cpu", [&] {
    if (round_scale) {
      act_quant_cpu_impl<true>(
          y.data_ptr<at::Float8_e4m3fn>(),
          scale.data_ptr<float>(),
          x.data_ptr<scalar_t>(),
          block_size,
          M,
          N,
          num_groups);
    } else {
      act_quant_cpu_impl<false>(
          y.data_ptr<at::Float8_e4m3fn>(),
          scale.data_ptr<float>(),
          x.data_ptr<scalar_t>(),
          block_size,
          M,
          N,
          num_groups);
    }
  });

  return {y, scale};
}
