#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
float rms_rscale(const scalar_t* input, int64_t head_dim, float eps) {
  float sum = 0.0f;
  for (int64_t d = 0; d < head_dim; ++d) {
    const float value = static_cast<float>(input[d]);
    sum += value * value;
  }
  return 1.0f / std::sqrt(sum / static_cast<float>(head_dim) + eps);
}

template <typename scalar_t>
void rmsnorm_row(
    scalar_t* output,
    const scalar_t* input,
    const scalar_t* weight,
    int64_t head_dim,
    float eps) {
  const float rscale = rms_rscale(input, head_dim, eps);
  for (int64_t d = 0; d < head_dim; ++d) {
    output[d] = static_cast<scalar_t>(
        static_cast<float>(input[d]) * rscale * static_cast<float>(weight[d]));
  }
}

template <typename scalar_t>
void apply_neox_rope_inplace(
    scalar_t* output,
    const scalar_t* normalized,
    const scalar_t* cache,
    int64_t head_dim,
    int64_t rope_dim) {
  const int64_t half = rope_dim / 2;
  for (int64_t d = 0; d < half; ++d) {
    const float x = static_cast<float>(normalized[d]);
    const float y = static_cast<float>(normalized[half + d]);
    const float cos = static_cast<float>(cache[d]);
    const float sin = static_cast<float>(cache[half + d]);
    output[d] = static_cast<scalar_t>(x * cos - y * sin);
    output[half + d] = static_cast<scalar_t>(y * cos + x * sin);
  }
  for (int64_t d = rope_dim; d < head_dim; ++d) {
    output[d] = normalized[d];
  }
}

template <typename scalar_t>
void fused_inplace_qknorm_rope_kernel(
    scalar_t* q,
    scalar_t* k,
    const scalar_t* q_weight,
    const scalar_t* k_weight,
    const scalar_t* cache,
    const int64_t* positions,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t rope_dim,
    int64_t q_stride_token,
    int64_t q_stride_head,
    int64_t k_stride_token,
    int64_t k_stride_head,
    float eps) {
  const int64_t rows = num_tokens * num_heads;
  at::parallel_for(0, rows, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    scalar_t q_tmp[128];
    scalar_t k_tmp[128];
    for (int64_t row = begin; row < end; ++row) {
      const int64_t token = row / num_heads;
      const int64_t head = row % num_heads;
      scalar_t* q_row = q + token * q_stride_token + head * q_stride_head;
      scalar_t* k_row = k + token * k_stride_token + head * k_stride_head;
      const scalar_t* cache_row = cache + positions[token] * rope_dim;
      rmsnorm_row(q_tmp, q_row, q_weight, head_dim, eps);
      rmsnorm_row(k_tmp, k_row, k_weight, head_dim, eps);
      apply_neox_rope_inplace(q_row, q_tmp, cache_row, head_dim, rope_dim);
      apply_neox_rope_inplace(k_row, k_tmp, cache_row, head_dim, rope_dim);
    }
  });
}

}  // namespace

void fused_inplace_qknorm_rope_cpu(
    at::Tensor& q,
    at::Tensor& k,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& positions,
    double eps,
    int64_t head_dim,
    int64_t rope_dim,
    bool is_neox,
    bool round_norm_before_rope) {
  CHECK_CPU(q);
  CHECK_CPU(k);
  CHECK_CPU(q_weight);
  CHECK_CPU(k_weight);
  CHECK_CPU(cos_sin_cache);
  CHECK_CPU(positions);
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_EQ(q.scalar_type(), at::kBFloat16);
  CHECK_EQ(k.scalar_type(), q.scalar_type());
  CHECK_EQ(q_weight.scalar_type(), q.scalar_type());
  CHECK_EQ(k_weight.scalar_type(), q.scalar_type());
  CHECK_EQ(cos_sin_cache.scalar_type(), q.scalar_type());
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(2, cos_sin_cache);
  CHECK_DIM(1, positions);
  CHECK_DIM(1, q_weight);
  CHECK_DIM(1, k_weight);
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(1), k.size(1));
  CHECK_EQ(q.size(2), head_dim);
  CHECK_EQ(k.size(2), head_dim);
  CHECK_EQ(q_weight.size(0), head_dim);
  CHECK_EQ(k_weight.size(0), head_dim);
  CHECK_EQ(positions.numel(), q.size(0));
  TORCH_CHECK(is_neox, "CPU fused qknorm+rope only supports is_neox=True");
  TORCH_CHECK(round_norm_before_rope, "CPU fused qknorm+rope requires rounded norm");
  TORCH_CHECK(head_dim == 128, "CPU fused qknorm+rope only supports head_dim=128");
  TORCH_CHECK(rope_dim == 96, "CPU fused qknorm+rope only supports rope_dim=96");
  TORCH_CHECK(cos_sin_cache.size(1) == rope_dim, "cos_sin_cache width must equal rope_dim");
  TORCH_CHECK(
      positions.numel() == 0 || positions.max().item<int64_t>() < cos_sin_cache.size(0),
      "positions exceed cos_sin_cache rows");

  fused_inplace_qknorm_rope_kernel<at::BFloat16>(
      q.data_ptr<at::BFloat16>(), k.data_ptr<at::BFloat16>(),
      q_weight.data_ptr<at::BFloat16>(), k_weight.data_ptr<at::BFloat16>(),
      cos_sin_cache.data_ptr<at::BFloat16>(), positions.data_ptr<int64_t>(),
      q.size(0), q.size(1), head_dim, rope_dim, q.stride(0), q.stride(1),
      k.stride(0), k.stride(1), static_cast<float>(eps));
}