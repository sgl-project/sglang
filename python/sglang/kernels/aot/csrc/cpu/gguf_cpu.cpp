#include <torch/all.h>

#include <cstring>
#include <vector>

#include "common.h"
#include "vec.h"

namespace {

constexpr int64_t kQ4_0 = 2;
constexpr int64_t kQ8_0 = 8;
constexpr int64_t kBlockSize = 32;
constexpr int64_t kQ4_0TypeSize = 18;
constexpr int64_t kQ8_0TypeSize = 34;

#if defined(CPU_CAPABILITY_AVX512)

inline float load_fp16_to_fp32(const uint8_t* ptr) {
  uint16_t h;
  std::memcpy(&h, ptr, sizeof(h));
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

inline void q8_0_block_to_fp32(const uint8_t* block, float* out) {
  const __m512 scale = _mm512_set1_ps(load_fp16_to_fp32(block));
  const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2));
  const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 18));
  _mm512_storeu_ps(out, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), scale));
  _mm512_storeu_ps(out + 16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), scale));
}

inline void q4_0_block_to_fp32(const uint8_t* block, float* out) {
  const __m512 scale = _mm512_set1_ps(load_fp16_to_fp32(block));
  const __m512i offset = _mm512_set1_epi32(8);
  const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2));
  const __m256i packed16 = _mm256_cvtepu8_epi16(packed);
  const __m256i lo = _mm256_and_si256(packed16, _mm256_set1_epi16(0x0F));
  const __m256i hi = _mm256_srli_epi16(packed16, 4);
  const __m512i lo_i32 = _mm512_sub_epi32(_mm512_cvtepi16_epi32(lo), offset);
  const __m512i hi_i32 = _mm512_sub_epi32(_mm512_cvtepi16_epi32(hi), offset);
  _mm512_storeu_ps(out, _mm512_mul_ps(_mm512_cvtepi32_ps(lo_i32), scale));
  _mm512_storeu_ps(out + 16, _mm512_mul_ps(_mm512_cvtepi32_ps(hi_i32), scale));
}

template <bool IS_Q8>
void decode_row_to_bf16(const uint8_t* qrow, at::BFloat16* out, int64_t k) {
  constexpr int64_t block_bytes = IS_Q8 ? kQ8_0TypeSize : kQ4_0TypeSize;
  const int64_t num_blocks = k / kBlockSize;

  alignas(64) float tmp[kBlockSize];
  for (int64_t block = 0; block < num_blocks; ++block) {
    const uint8_t* block_ptr = qrow + block * block_bytes;
    if constexpr (IS_Q8) {
      q8_0_block_to_fp32(block_ptr, tmp);
    } else {
      q4_0_block_to_fp32(block_ptr, tmp);
    }

    const __m512 fp0 = _mm512_loadu_ps(tmp);
    const __m512 fp1 = _mm512_loadu_ps(tmp + 16);
    _mm512_storeu_si512(out + block * kBlockSize, (__m512i)_mm512_cvtne2ps_pbh(fp1, fp0));
  }
}

template <bool IS_Q8>
float gguf_dot_fp32(const uint8_t* qrow, const float* x, int64_t k) {
  constexpr int64_t block_bytes = IS_Q8 ? kQ8_0TypeSize : kQ4_0TypeSize;
  const int64_t num_blocks = k / kBlockSize;
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();

  if constexpr (IS_Q8) {
    for (int64_t block = 0; block < num_blocks; ++block) {
      const uint8_t* block_ptr = qrow + block * block_bytes;
      const __m512 scale = _mm512_set1_ps(load_fp16_to_fp32(block_ptr));
      const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block_ptr + 2));
      const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block_ptr + 18));
      const __m512 q0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), scale);
      const __m512 q1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), scale);
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + block * kBlockSize), q0, acc0);
      acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + block * kBlockSize + 16), q1, acc1);
    }
  } else {
    const __m512i offset = _mm512_set1_epi32(8);
    for (int64_t block = 0; block < num_blocks; ++block) {
      const uint8_t* block_ptr = qrow + block * block_bytes;
      const __m512 scale = _mm512_set1_ps(load_fp16_to_fp32(block_ptr));
      const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block_ptr + 2));
      const __m256i packed16 = _mm256_cvtepu8_epi16(packed);
      const __m256i lo = _mm256_and_si256(packed16, _mm256_set1_epi16(0x0F));
      const __m256i hi = _mm256_srli_epi16(packed16, 4);
      const __m512 q0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepi16_epi32(lo), offset)), scale);
      const __m512 q1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepi16_epi32(hi), offset)), scale);
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + block * kBlockSize), q0, acc0);
      acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + block * kBlockSize + 16), q1, acc1);
    }
  }

  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

template <bool IS_Q8>
at::Tensor gguf_mul_mat_impl(const at::Tensor& x, const at::Tensor& qweight, int64_t n, int64_t k) {
  const int64_t m = x.numel() / k;
  const int64_t packed_cols = qweight.size(1);
  const uint8_t* qweight_ptr = qweight.data_ptr<uint8_t>();
  const at::ScalarType output_dtype = x.scalar_type();
  std::vector<int64_t> output_shape(x.sizes().begin(), x.sizes().end());
  output_shape.back() = n;

  if (m == 1) {
    auto x_fp32 = x.reshape({k}).to(at::kFloat).contiguous();
    const float* x_ptr = x_fp32.data_ptr<float>();
    auto output = at::empty({n}, x.options().dtype(at::kFloat));
    float* output_ptr = output.data_ptr<float>();

    at::parallel_for(0, n, 0, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        output_ptr[row] = gguf_dot_fp32<IS_Q8>(qweight_ptr + row * packed_cols, x_ptr, k);
      }
    });

    return output.to(output_dtype).reshape(output_shape);
  }

  auto weight_bf16 = at::empty({n, k}, x.options().dtype(at::kBFloat16));
  at::BFloat16* weight_ptr = weight_bf16.data_ptr<at::BFloat16>();
  at::parallel_for(0, n, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      decode_row_to_bf16<IS_Q8>(qweight_ptr + row * packed_cols, weight_ptr + row * k, k);
    }
  });

  auto x_bf16 = x.reshape({m, k}).to(at::kBFloat16);
  return at::linear(x_bf16, weight_bf16).to(output_dtype).reshape(output_shape);
}

#endif

}  // namespace

at::Tensor gguf_mul_mat_cpu(const at::Tensor& x, const at::Tensor& qweight, int64_t qtype) {
  TORCH_CHECK(x.device().is_cpu(), "gguf_mul_mat_cpu: x must be a CPU tensor");
  TORCH_CHECK(qweight.device().is_cpu(), "gguf_mul_mat_cpu: qweight must be a CPU tensor");
  TORCH_CHECK(x.is_contiguous(), "gguf_mul_mat_cpu: x must be contiguous");
  TORCH_CHECK(qweight.is_contiguous(), "gguf_mul_mat_cpu: qweight must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == at::kByte, "gguf_mul_mat_cpu: qweight must be uint8");
  TORCH_CHECK(x.dim() >= 1, "gguf_mul_mat_cpu: x must have at least one dimension");
  TORCH_CHECK(qweight.dim() == 2, "gguf_mul_mat_cpu: qweight must be 2D");

  const int64_t qtype_size = qtype == kQ8_0 ? kQ8_0TypeSize : qtype == kQ4_0 ? kQ4_0TypeSize : 0;
  TORCH_CHECK(qtype_size != 0, "gguf_mul_mat_cpu: only Q4_0 and Q8_0 are supported, got qtype=", qtype);
  const int64_t n = qweight.size(0);
  const int64_t packed_cols = qweight.size(1);
  TORCH_CHECK(packed_cols % qtype_size == 0, "gguf_mul_mat_cpu: invalid packed columns for qtype");
  const int64_t k = packed_cols / qtype_size * kBlockSize;
  TORCH_CHECK(k % kBlockSize == 0, "gguf_mul_mat_cpu: K must be divisible by 32");
  TORCH_CHECK(x.size(-1) == k, "gguf_mul_mat_cpu: x last dimension must match unpacked qweight columns");

#if defined(CPU_CAPABILITY_AVX512)
  if (qtype == kQ8_0) {
    return gguf_mul_mat_impl<true>(x, qweight, n, k);
  }
  return gguf_mul_mat_impl<false>(x, qweight, n, k);
#else
  TORCH_CHECK(false, "gguf_mul_mat_cpu: AVX512 support is required");
#endif
}
