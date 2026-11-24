#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
void rmsnorm_residual_scalar_row(
    scalar_t* out,
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* residual,
    int size,
    float scalar,
    float eps) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();

  fVec sum(0.f);
  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [x_fvec0, x_fvec1] = load_float_vec2(input + d);
    sum += x_fvec0 * x_fvec0;
    sum += x_fvec1 * x_fvec1;
  }
  float sum_val = vec_reduce_sum(sum);
#pragma GCC unroll 4
  for (; d < size; ++d) {
    const float x_val = static_cast<float>(input[d]);
    sum_val += x_val * x_val;
  }
  const float inv = 1.f / std::sqrt(sum_val / size + eps);
  const fVec inv_fvec(inv), s_fvec(scalar);

#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [x_fvec0, x_fvec1] = load_float_vec2(input + d);
    auto [w_fvec0, w_fvec1] = load_float_vec2(weight + d);
    auto [r_fvec0, r_fvec1] = load_float_vec2(residual + d);
    const fVec out_fvec0 = (x_fvec0 * inv_fvec * w_fvec0 + r_fvec0) * s_fvec;
    const fVec out_fvec1 = (x_fvec1 * inv_fvec * w_fvec1 + r_fvec1) * s_fvec;
    store_float_vec2(out + d, out_fvec0, out_fvec1);
  }
#pragma GCC unroll 4
  for (; d < size; ++d) {
    const float norm_val = static_cast<float>(input[d]) * inv * static_cast<float>(weight[d]);
    const float out_val = (norm_val + static_cast<float>(residual[d])) * scalar;
    out[d] = static_cast<scalar_t>(out_val);
  }
}

template <typename scalar_t>
void gemma_rmsnorm_residual_scalar_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ residual,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t x_strideT,
    int64_t res_strideT,
    float scalar,
    float eps) {
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      rmsnorm_residual_scalar_row(
          out + row * hidden_size,
          input + row * x_strideT,
          weight,
          residual + row * res_strideT,
          hidden_size,
          scalar,
          eps);
    }
  });
}

template <typename scalar_t>
void fused_rmsnorm_residual_scalar_row(
    scalar_t* out,
    float* merged,
    const scalar_t* x1,
    const scalar_t* x2,
    const scalar_t* w1,
    const scalar_t* w2,
    const scalar_t* w3,
    const scalar_t* residual,
    int size,
    float scalar,
    float eps1,
    float eps2,
    float eps3) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();

  fVec sum1(0.f), sum2(0.f);
  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [x1_fvec0, x1_fvec1] = load_float_vec2(x1 + d);
    auto [x2_fvec0, x2_fvec1] = load_float_vec2(x2 + d);
    sum1 += x1_fvec0 * x1_fvec0;
    sum1 += x1_fvec1 * x1_fvec1;
    sum2 += x2_fvec0 * x2_fvec0;
    sum2 += x2_fvec1 * x2_fvec1;
  }
  float sum1_val = vec_reduce_sum(sum1);
  float sum2_val = vec_reduce_sum(sum2);
#pragma GCC unroll 4
  for (; d < size; ++d) {
    const float x1_val = static_cast<float>(x1[d]);
    const float x2_val = static_cast<float>(x2[d]);
    sum1_val += x1_val * x1_val;
    sum2_val += x2_val * x2_val;
  }
  const float inv1 = 1.f / std::sqrt(sum1_val / size + eps1);
  const float inv2 = 1.f / std::sqrt(sum2_val / size + eps2);
  const fVec inv1_fvec(inv1), inv2_fvec(inv2);

  fVec sum3(0.f);
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [x1_fvec0, x1_fvec1] = load_float_vec2(x1 + d);
    auto [x2_fvec0, x2_fvec1] = load_float_vec2(x2 + d);
    auto [w1_fvec0, w1_fvec1] = load_float_vec2(w1 + d);
    auto [w2_fvec0, w2_fvec1] = load_float_vec2(w2 + d);
    const fVec y_fvec0 = x1_fvec0 * inv1_fvec * w1_fvec0 + x2_fvec0 * inv2_fvec * w2_fvec0;
    const fVec y_fvec1 = x1_fvec1 * inv1_fvec * w1_fvec1 + x2_fvec1 * inv2_fvec * w2_fvec1;
    store_float_vec2(merged + d, y_fvec0, y_fvec1);
    sum3 += y_fvec0 * y_fvec0;
    sum3 += y_fvec1 * y_fvec1;
  }
  float sum3_val = vec_reduce_sum(sum3);
#pragma GCC unroll 4
  for (; d < size; ++d) {
    const float x1_val = static_cast<float>(x1[d]) * inv1 * static_cast<float>(w1[d]);
    const float x2_val = static_cast<float>(x2[d]) * inv2 * static_cast<float>(w2[d]);
    const float merged_val = x1_val + x2_val;
    merged[d] = merged_val;
    sum3_val += merged_val * merged_val;
  }
  const float inv3 = 1.f / std::sqrt(sum3_val / size + eps3);
  const fVec inv3_fvec(inv3), s_fvec(scalar);

#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [y_fvec0, y_fvec1] = load_float_vec2(merged + d);
    auto [w3_fvec0, w3_fvec1] = load_float_vec2(w3 + d);
    auto [r_fvec0, r_fvec1] = load_float_vec2(residual + d);
    const fVec out_fvec0 = (y_fvec0 * inv3_fvec * w3_fvec0 + r_fvec0) * s_fvec;
    const fVec out_fvec1 = (y_fvec1 * inv3_fvec * w3_fvec1 + r_fvec1) * s_fvec;
    store_float_vec2(out + d, out_fvec0, out_fvec1);
  }
#pragma GCC unroll 4
  for (; d < size; ++d) {
    const float norm_val = merged[d] * inv3 * static_cast<float>(w3[d]);
    const float out_val = (norm_val + static_cast<float>(residual[d])) * scalar;
    out[d] = static_cast<scalar_t>(out_val);
  }
}

template <typename scalar_t>
void gemma_dual_rmsnorm_residual_scalar_kernel_impl(
    scalar_t* __restrict__ out,
    float* __restrict__ merged,
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x2,
    const scalar_t* __restrict__ w1,
    const scalar_t* __restrict__ w2,
    const scalar_t* __restrict__ w3,
    const scalar_t* __restrict__ residual,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t x1_strideT,
    int64_t x2_strideT,
    int64_t res_strideT,
    float scalar,
    float eps1,
    float eps2,
    float eps3) {
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    float* __restrict__ buf_ptr = merged + at::get_thread_num() * hidden_size;
    for (int64_t row = begin; row < end; ++row) {
      fused_rmsnorm_residual_scalar_row(
          out + row * hidden_size,
          buf_ptr,
          x1 + row * x1_strideT,
          x2 + row * x2_strideT,
          w1,
          w2,
          w3,
          residual + row * res_strideT,
          hidden_size,
          scalar,
          eps1,
          eps2,
          eps3);
    }
  });
}

}  // anonymous namespace

// output = (RMSNorm(input, weight, eps) + residual) * scalar
// input, residual: [num_tokens, hidden_size]
// weight: [hidden_size]
// scalar: [1]
at::Tensor gemma_rmsnorm_residual_scalar_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& residual,
    const at::Tensor& scalar,
    double eps) {
  CHECK_DIM(2, input);
  const auto st = input.scalar_type();
  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = input.size(1);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT_SHAPE_DTYPE<true>(residual, input.sizes(), st);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {hidden_size}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(scalar, {1}, st);

  at::Tensor output = at::empty(input.sizes(), input.options());
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "gemma_rmsnorm_residual_scalar_cpu", [&] {
    gemma_rmsnorm_residual_scalar_kernel_impl(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        num_tokens,
        hidden_size,
        input.stride(0),
        residual.stride(0),
        static_cast<float>(*scalar.data_ptr<scalar_t>()),
        eps);
  });
  return output;
}

// merged = RMSNorm(x1, weight1, eps1) + RMSNorm(x2, weight2, eps2)
// output = (RMSNorm(merged, weight3, eps3) + residual) * scalar
// x1, x2, residual: [num_tokens, hidden_size]
// weight1, weight2, weight3: [hidden_size]
// scalar: [1]
at::Tensor gemma_dual_rmsnorm_residual_scalar_cpu(
    const at::Tensor& x1,
    const at::Tensor& weight1,
    const at::Tensor& x2,
    const at::Tensor& weight2,
    const at::Tensor& weight3,
    const at::Tensor& residual,
    const at::Tensor& scalar,
    double eps1,
    double eps2,
    double eps3) {
  CHECK_DIM(2, x1);
  const auto st = x1.scalar_type();
  const int64_t num_tokens = x1.size(0);
  const int64_t hidden_size = x1.size(1);
  CHECK_INPUT_SHAPE_DTYPE<true>(x1, x1.sizes(), st);
  CHECK_INPUT_SHAPE_DTYPE<true>(x2, x1.sizes(), st);
  CHECK_INPUT_SHAPE_DTYPE<true>(residual, x1.sizes(), st);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight1, {hidden_size}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight2, {hidden_size}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(weight3, {hidden_size}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(scalar, {1}, st);

  // buffer for intermediate merged results per thread
  int num_threads = at::get_num_threads();
  at::Tensor merged = at::empty({num_threads, hidden_size}, x1.options().dtype(at::kFloat));
  at::Tensor output = at::empty(x1.sizes(), x1.options());
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "gemma_dual_rmsnorm_residual_scalar_cpu", [&] {
    gemma_dual_rmsnorm_residual_scalar_kernel_impl(
        output.data_ptr<scalar_t>(),
        merged.data_ptr<float>(),
        x1.data_ptr<scalar_t>(),
        x2.data_ptr<scalar_t>(),
        weight1.data_ptr<scalar_t>(),
        weight2.data_ptr<scalar_t>(),
        weight3.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        num_tokens,
        hidden_size,
        x1.stride(0),
        x2.stride(0),
        residual.stride(0),
        static_cast<float>(*scalar.data_ptr<scalar_t>()),
        eps1,
        eps2,
        eps3);
  });
  return output;
}
