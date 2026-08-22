#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>

#include "common.h"
#include "vec.h"

namespace {

constexpr int64_t kBlockSize = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kHeadDimWithScaleBytes = 132;
constexpr int64_t kScaleOffsetBytes = kBlockSize * kHeadDim;
constexpr int64_t kBlockBytes = kBlockSize * kHeadDimWithScaleBytes;

inline float fp8_e4m3_to_float(uint8_t v) {
  c10::Float8_e4m3fn x;
  x.x = v;
  return static_cast<float>(x);
}

inline float dot_fp8_128_scalar(const uint8_t* k, const uint8_t* q) {
  float dot = 0.0f;
  for (int64_t d = 0; d < kHeadDim; ++d) {
    dot += fp8_e4m3_to_float(k[d]) * fp8_e4m3_to_float(q[d]);
  }
  return dot;
}

#if defined(CPU_CAPABILITY_AVX512)
inline float dot_fp8_128(const uint8_t* k, const uint8_t* q) {
  __m512 acc = _mm512_setzero_ps();
  for (int64_t d = 0; d < kHeadDim; d += 32) {
    const __m256i k8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + d));
    const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d));
    acc = _mm512_dpbf16_ps(acc, CVT_FP8_TO_BF16(k8), CVT_FP8_TO_BF16(q8));
  }
  return _mm512_reduce_add_ps(acc);
}
#else
inline float dot_fp8_128(const uint8_t* k, const uint8_t* q) {
  return dot_fp8_128_scalar(k, q);
}
#endif

template <typename T>
inline int64_t load_int(const T* ptr, int64_t idx) {
  return static_cast<int64_t>(ptr[idx]);
}

template <typename T>
inline float load_weight(const T* ptr, int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <typename seq_t, typename page_t, typename weight_t>
void fp8_paged_mqa_logits_cpu_impl(
    const at::Tensor& q_fp8,
    const at::Tensor& kvcache_fp8,
    const at::Tensor& weight,
    const at::Tensor& seq_lens,
    const at::Tensor& page_table,
    at::Tensor& logits,
    int64_t max_seq_len) {
  const int64_t batch_size = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(2);
  const int64_t num_blocks = kvcache_fp8.size(0);
  const int64_t pages_per_batch = page_table.size(1);

  const auto* q_ptr = reinterpret_cast<const uint8_t*>(q_fp8.const_data_ptr());
  const auto* cache_ptr = reinterpret_cast<const uint8_t*>(kvcache_fp8.const_data_ptr());
  const auto* weight_ptr = weight.const_data_ptr<weight_t>();
  const auto* seq_ptr = seq_lens.const_data_ptr<seq_t>();
  const auto* page_ptr = page_table.const_data_ptr<page_t>();
  auto* out_ptr = logits.data_ptr<float>();

  at::parallel_for(0, batch_size * max_seq_len, GRAIN_SIZE / kHeadDim, [&](int64_t begin, int64_t end) {
    int64_t b{0}, token{0};
    data_index_init(begin, b, batch_size, token, max_seq_len);
    for (int64_t i = begin; i < end; ++i) {
      const int64_t seq_len = load_int(seq_ptr, b);
      TORCH_CHECK(seq_len >= 0 && seq_len <= max_seq_len, "seq_lens must be in [0, max_seq_len]");

      if (token >= seq_len) {
        data_index_step(b, batch_size, token, max_seq_len);
        continue;
      }

      const int64_t q_batch_offset = ((b * 1 * num_heads) * kHeadDim);
      const int64_t weight_batch_offset = b * num_heads;
      const int64_t page_batch_offset = b * pages_per_batch;
      float* out_row = out_ptr + b * max_seq_len;

      const int64_t logical_page = token / kBlockSize;
      const int64_t token_in_page = token % kBlockSize;
      TORCH_CHECK(logical_page < pages_per_batch, "page_table does not cover seq_len");

      const int64_t physical_page = load_int(page_ptr, page_batch_offset + logical_page);
      TORCH_CHECK(physical_page >= 0 && physical_page < num_blocks, "page_table contains an invalid page index");

      const uint8_t* block = cache_ptr + physical_page * kBlockBytes;
      const uint8_t* k_token = block + token_in_page * kHeadDim;
      const float* scale_ptr = reinterpret_cast<const float*>(block + kScaleOffsetBytes);
      const float k_scale = scale_ptr[token_in_page];

      float score_sum = 0.0f;
      for (int64_t h = 0; h < num_heads; ++h) {
        const uint8_t* q_head = q_ptr + q_batch_offset + h * kHeadDim;
        float dot = dot_fp8_128(k_token, q_head);
        dot = std::max(dot, 0.0f);
        score_sum += dot * load_weight(weight_ptr, weight_batch_offset + h);
      }

      out_row[token] = score_sum * k_scale;

      data_index_step(b, batch_size, token, max_seq_len);
    }
  });
}

template <typename seq_t, typename page_t>
void dispatch_weight_type(
    const at::Tensor& q_fp8,
    const at::Tensor& kvcache_fp8,
    const at::Tensor& weight,
    const at::Tensor& seq_lens,
    const at::Tensor& page_table,
    at::Tensor& logits,
    int64_t max_seq_len) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "fp8_paged_mqa_logits_cpu_weight", [&] {
        fp8_paged_mqa_logits_cpu_impl<seq_t, page_t, scalar_t>(
            q_fp8, kvcache_fp8, weight, seq_lens, page_table, logits, max_seq_len);
      });
}

template <typename seq_t>
void dispatch_page_type(
    const at::Tensor& q_fp8,
    const at::Tensor& kvcache_fp8,
    const at::Tensor& weight,
    const at::Tensor& seq_lens,
    const at::Tensor& page_table,
    at::Tensor& logits,
    int64_t max_seq_len) {
  if (page_table.scalar_type() == at::kInt) {
    dispatch_weight_type<seq_t, int32_t>(q_fp8, kvcache_fp8, weight, seq_lens, page_table, logits, max_seq_len);
  } else if (page_table.scalar_type() == at::kLong) {
    dispatch_weight_type<seq_t, int64_t>(q_fp8, kvcache_fp8, weight, seq_lens, page_table, logits, max_seq_len);
  } else {
    TORCH_CHECK(false, "page_table must be int32 or int64");
  }
}

}  // namespace

at::Tensor fp8_paged_mqa_logits_cpu(
    at::Tensor& q_fp8,
    at::Tensor& kvcache_fp8,
    at::Tensor& weight,
    at::Tensor& seq_lens,
    at::Tensor& page_table,
    int64_t max_seq_len,
    bool clean_logits) {
  TORCH_CHECK(!clean_logits, "fp8_paged_mqa_logits_cpu only supports clean_logits == false");
  CHECK_INPUT(q_fp8);
  CHECK_INPUT(kvcache_fp8);
  CHECK_INPUT(weight);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(page_table);
  TORCH_CHECK(q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "q_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(kvcache_fp8.scalar_type() == at::kByte, "kvcache_fp8 must be torch.uint8 storage");

  // The checks are aligned with fp8_paged_mqa_logits_torch in indexer.py.
  TORCH_CHECK(q_fp8.dim() == 4, "q_fp8 must have shape [batch, 1, heads, 128]");
  TORCH_CHECK(q_fp8.size(1) == 1, "q_fp8 second dimension must be 1");
  TORCH_CHECK(q_fp8.size(3) == kHeadDim, "q_fp8 head_dim must be 128");
  TORCH_CHECK(kvcache_fp8.dim() == 4, "kvcache_fp8 must have shape [blocks, 64, 1, 132]");
  TORCH_CHECK(kvcache_fp8.size(1) == kBlockSize, "kvcache_fp8 block size must be 64");
  TORCH_CHECK(kvcache_fp8.size(2) == 1, "kvcache_fp8 num kv heads must be 1");
  TORCH_CHECK(kvcache_fp8.size(3) == kHeadDimWithScaleBytes, "kvcache_fp8 last dimension must be 132 bytes");

  const int64_t batch_size = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(2);
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({batch_size, num_heads}), "weight must have shape [batch, heads]");
  TORCH_CHECK(seq_lens.sizes() == at::IntArrayRef({batch_size}), "seq_lens must have shape [batch]");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == batch_size, "page_table must have shape [batch, pages]");
  TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative");

  auto logits = at::empty({batch_size, max_seq_len}, q_fp8.options().dtype(at::kFloat));

  if (seq_lens.scalar_type() == at::kInt) {
    dispatch_page_type<int32_t>(q_fp8, kvcache_fp8, weight, seq_lens, page_table, logits, max_seq_len);
  } else if (seq_lens.scalar_type() == at::kLong) {
    dispatch_page_type<int64_t>(q_fp8, kvcache_fp8, weight, seq_lens, page_table, logits, max_seq_len);
  } else {
    TORCH_CHECK(false, "seq_lens must be int32 or int64");
  }

  return logits;
}
