#include "common.h"
#include "gemm.h"

template <typename scalar_t>
void quantize_tensor_fp8(
    at::Float8_e4m3fn* __restrict__ Aq,
    float& As,
    const scalar_t* __restrict__ A,
    int64_t B,
    int64_t M,
    int64_t K,
    int64_t strideB_Aq,
    int64_t strideB_A,
    int64_t strideM_Aq,
    int64_t strideM_A,
    float eps) {
  int num_threads = at::get_num_threads();
  std::vector<float> amax_vec(num_threads);
  float* amax = amax_vec.data();
  at::parallel_for(0, B * M, 0, [&](int64_t begin, int64_t end) {
    int64_t b{0}, m{0};
    int tid = at::get_thread_num();
    data_index_init(begin, b, B, m, M);
    float local_amax = 0.0f;
    for (int64_t idx = begin; idx < end; ++idx) {
      for (int64_t k = 0; k < K; ++k) {
        const float val = static_cast<float>(A[b * strideB_A + m * strideM_A + k]);
        local_amax = std::max(local_amax, std::abs(val));
      }
      data_index_step(b, B, m, M);
    }
    amax[tid] = local_amax;
  });
  float global_max = 0.0f;
  for (int i = 0; i < num_threads; ++i) {
    global_max = std::max(global_max, amax[i]);
  }
  const float scale = std::max(global_max / FP8_MAX, eps);
  const float inv_scale = 1 / scale;
  at::parallel_for(0, B * M, 0, [&](int64_t begin, int64_t end) {
    int64_t b{0}, m{0};
    data_index_init(begin, b, B, m, M);
    for (int64_t idx = begin; idx < end; ++idx) {
      for (int64_t k = 0; k < K; ++k) {
        float val = static_cast<float>(A[b * strideB_A + m * strideM_A + k]) * inv_scale;
        if (val < FP8_MIN) {
          val = FP8_MIN;
        } else if (val > FP8_MAX) {
          val = FP8_MAX;
        }
        Aq[b * strideB_Aq + m * strideM_Aq + k] = static_cast<at::Float8_e4m3fn>(val);
      }
      data_index_step(b, B, m, M);
    }
  });
  As = scale;
}

#if defined(CPU_CAPABILITY_AVX512)

template <>
void quantize_tensor_fp8(
    at::Float8_e4m3fn* __restrict__ Aq,
    float& As,
    const at::BFloat16* __restrict__ A,
    int64_t B,
    int64_t M,
    int64_t K,
    int64_t strideB_Aq,
    int64_t strideB_A,
    int64_t strideM_Aq,
    int64_t strideM_A,
    float eps) {
  constexpr int64_t kVecSize = 32;
  const __m512 vabs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

  // Phase 1: Find global max_abs
  int num_threads = at::get_num_threads();
  std::vector<float> amax_vec(num_threads);
  float* amax = amax_vec.data();
  at::parallel_for(0, B * M, 0, [&](int64_t begin, int64_t end) {
    int64_t b{0}, m{0};
    data_index_init(begin, b, B, m, M);
    float scalar_max = 0.0f;
    int tid = at::get_thread_num();

    __m512 vmax0 = _mm512_setzero_ps();
    __m512 vmax1 = _mm512_setzero_ps();
    for (int64_t idx = begin; idx < end; ++idx) {
      const at::BFloat16* __restrict__ input_ptr = A + b * strideB_A + m * strideM_A;

      int64_t k = 0;

      for (; k <= K - kVecSize; k += kVecSize) {
        __m512i bf16_data = _mm512_loadu_si512((__m512i*)(input_ptr + k));
        __m512 f0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(bf16_data, 0));
        __m512 f1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(bf16_data, 1));

        f0 = _mm512_and_ps(f0, vabs_mask);
        f1 = _mm512_and_ps(f1, vabs_mask);

        vmax0 = _mm512_max_ps(vmax0, f0);
        vmax1 = _mm512_max_ps(vmax1, f1);
      }

      // Remainder
      for (; k < K; ++k) {
        float val = static_cast<float>(input_ptr[k]);
        scalar_max = std::max(scalar_max, std::abs(val));
      }

      // move to the next index
      data_index_step(b, B, m, M);
    }
    float vec_max = _mm512_reduce_max_ps(_mm512_max_ps(vmax0, vmax1));

    amax[tid] = std::max(vec_max, scalar_max);
  });

  int i = 0;
  __m512 vmax = _mm512_setzero_ps();
  for (; i + 16 <= num_threads; i += 16) {
    __m512 v = _mm512_loadu_ps(amax + i);
    vmax = _mm512_max_ps(vmax, v);
  }
  float global_max = _mm512_reduce_max_ps(vmax);
  for (; i < num_threads; ++i) {
    global_max = std::max(global_max, amax[i]);
  }
  const float scale = std::max(global_max / FP8_MAX, eps);
  const float inv_scale = 1.0f / scale;

  // Phase 2: Quantize
  const __m512 vinv_scale = _mm512_set1_ps(inv_scale);
  const __m512 vfp8_min = _mm512_set1_ps(FP8_MIN);
  const __m512 vfp8_max = _mm512_set1_ps(FP8_MAX);

  at::parallel_for(0, B * M, 0, [&](int64_t begin, int64_t end) {
    int64_t b{0}, m{0};
    data_index_init(begin, b, B, m, M);
    for (int64_t idx = begin; idx < end; ++idx) {
      const at::BFloat16* __restrict__ input_ptr = A + b * strideB_A + m * strideM_A;
      at::Float8_e4m3fn* __restrict__ output_ptr = Aq + b * strideB_Aq + m * strideM_Aq;

      int64_t k = 0;

      for (; k <= K - kVecSize; k += kVecSize) {
        __m512i bf16_data = _mm512_loadu_si512((__m512i*)(input_ptr + k));

        __m512 f0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(bf16_data, 0));
        __m512 f1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(bf16_data, 1));

        // Scale and clamp
        f0 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(f0, vinv_scale), vfp8_min), vfp8_max);
        f1 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(f1, vinv_scale), vfp8_min), vfp8_max);

        __m128i o0 = cvtfp32_fp8e4m3(f0);
        __m128i o1 = cvtfp32_fp8e4m3(f1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_ptr + k), _mm256_set_m128i(o1, o0));
      }

      // Remainder
      for (; k < K; ++k) {
        float val = static_cast<float>(input_ptr[k]) * inv_scale;
        val = std::clamp(val, FP8_MIN, FP8_MAX);
        output_ptr[k] = static_cast<at::Float8_e4m3fn>(val);
      }
      // move to the next index
      data_index_step(b, B, m, M);
    }
  });

  As = scale;
}
#endif

#define INSTANTIATE_QUANTIZE_TENSOR_FP8_TEMPLATE(TYPE) \
  template void quantize_tensor_fp8<TYPE>(             \
      at::Float8_e4m3fn* __restrict__ Aq,              \
      float& As,                                       \
      const TYPE* __restrict__ A,                      \
      int64_t B,                                       \
      int64_t M,                                       \
      int64_t K,                                       \
      int64_t strideB_Aq,                              \
      int64_t strideB_A,                               \
      int64_t strideM_Aq,                              \
      int64_t strideM_A,                               \
      float eps);

INSTANTIATE_QUANTIZE_TENSOR_FP8_TEMPLATE(at::BFloat16)
INSTANTIATE_QUANTIZE_TENSOR_FP8_TEMPLATE(at::Half)
