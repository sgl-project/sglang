#include <ATen/cpu/vec/vec.h>

#include <cmath>
#include <vector>

#include "common.h"

namespace {

using Vec = at::vec::Vectorized<float>;
void compute_freqs(float* __restrict__ freqs, int64_t half_dim, float neg_log_max_period) {
  const int64_t vec_size = Vec::size();

  const Vec neg_log_max_period_vec(neg_log_max_period);

  int64_t i = 0;
  for (; i + vec_size <= half_dim; i += vec_size) {
    const Vec index_vec = Vec::arange(static_cast<float>(i), 1.0f);

    const Vec exponent_vec = index_vec * neg_log_max_period_vec;

    const Vec freq_vec = exponent_vec.exp();

    freq_vec.store(freqs + i);
  }

  for (; i < half_dim; ++i) {
    freqs[i] = std::exp(neg_log_max_period * static_cast<float>(i));
  }
}

template <typename scalar_t, bool flip_sin_to_cos>
void timestep_embedding_kernel_impl(
    const scalar_t* __restrict__ timesteps,
    float* __restrict__ output,
    const float* __restrict__ freqs,
    int64_t batch_size,
    int64_t dim,
    float scale) {
  const int64_t half_dim = dim / 2;
  const int64_t vec_size = Vec::size();
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const float t_val = static_cast<float>(timesteps[row]);

      const float timestep_scale = scale * t_val;

      const Vec timestep_scale_vec(timestep_scale);

      float* output_row = output + row * dim;

      float* sin_dst;
      float* cos_dst;
      // flip == false: [sin | cos]
      // flip == true: [cos | sin]
      if constexpr (!flip_sin_to_cos) {
        sin_dst = output_row;
        cos_dst = output_row + half_dim;
      } else {
        cos_dst = output_row;
        sin_dst = output_row + half_dim;
      }

      int64_t i = 0;

      for (; i + vec_size <= half_dim; i += vec_size) {
        const Vec freq_vec = Vec::loadu(freqs + i);

        const Vec angle_vec = freq_vec * timestep_scale_vec;

        const Vec sin_vec = angle_vec.sin();
        const Vec cos_vec = angle_vec.cos();

        sin_vec.store(sin_dst + i);
        cos_vec.store(cos_dst + i);
      }
      // Scalar tail.
      for (; i < half_dim; ++i) {
        const float angle = timestep_scale * freqs[i];

        sin_dst[i] = std::sin(angle);
        cos_dst[i] = std::cos(angle);
      }
      if (dim % 2 == 1) {
        output_row[dim - 1] = 0.0f;
      }
    }
  });
}

template <typename scalar_t>
void timestep_embedding_dispatch_flip(
    const scalar_t* __restrict__ timesteps,
    float* __restrict__ output,
    const float* __restrict__ freqs,
    int64_t batch_size,
    int64_t dim,
    bool flip_sin_to_cos,
    float scale) {
  if (flip_sin_to_cos) {
    timestep_embedding_kernel_impl<scalar_t, true>(timesteps, output, freqs, batch_size, dim, scale);
  } else {
    timestep_embedding_kernel_impl<scalar_t, false>(timesteps, output, freqs, batch_size, dim, scale);
  }
}

}  // namespace

at::Tensor timestep_embedding_cpu(
    const at::Tensor& timesteps,
    int64_t dim,
    bool flip_sin_to_cos,
    double downscale_freq_shift,
    double scale,
    int64_t max_period) {
  CHECK_INPUT(timesteps);
  CHECK_DIM(1, timesteps);

  TORCH_CHECK(dim > 0, "dim must be greater than 0.");
  TORCH_CHECK(max_period > 0, "max_period must be greater than 0.");

  const int64_t batch_size = timesteps.size(0);

  const int64_t half_dim = dim / 2;

  const float denominator = static_cast<float>(half_dim) - static_cast<float>(downscale_freq_shift);

  TORCH_CHECK(denominator != 0.0f, "half_dim - downscale_freq_shift must not be zero.")
  const float neg_log_max_period = std::log(static_cast<float>(max_period)) * (-1.0f) / denominator;

  auto output = at::empty({batch_size, dim}, timesteps.options().dtype(at::kFloat));

  if (batch_size == 0) {
    return output;
  }

  std::vector<float> freqs(static_cast<size_t>(half_dim));
  compute_freqs(freqs.data(), half_dim, neg_log_max_period);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, timesteps.scalar_type(), "timestep_embedding_cpu", [&] {
        timestep_embedding_dispatch_flip<scalar_t>(
            timesteps.data_ptr<scalar_t>(),
            output.data_ptr<float>(),
            freqs.data(),
            batch_size,
            dim,
            flip_sin_to_cos,
            static_cast<float>(scale));
      });

  return output;
}
