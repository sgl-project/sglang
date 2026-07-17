#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t, typename func_t, typename vec_func_t>
void act_and_mul_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t num_tokens,
    int64_t dim,
    const func_t& f,
    const vec_func_t& vf) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t kVecSize = bVec::size();
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      // local ptrs
      const scalar_t* __restrict__ input_ptr = input + i * 2 * dim;
      const scalar_t* __restrict__ input_other_ptr = input_ptr + dim;
      scalar_t* __restrict__ output_ptr = output + i * dim;

      int64_t d;
#pragma GCC unroll 4
      for (d = 0; d <= dim - kVecSize; d += kVecSize) {
        auto [x_fvec0, x_fvec1] = load_float_vec2(input_ptr + d);
        auto [y_fvec0, y_fvec1] = load_float_vec2(input_other_ptr + d);

        x_fvec0 = vf(x_fvec0);
        x_fvec1 = vf(x_fvec1);

        x_fvec0 = x_fvec0 * y_fvec0;
        x_fvec1 = x_fvec1 * y_fvec1;

        convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1).store(output_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < dim; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        float y_val = static_cast<float>(input_other_ptr[d]);
        output_ptr[d] = f(x_val) * y_val;
      }
    }
  });
}

// input : [num_tokens, dim] contiguous
// gate : [num_tokens, num_heads, head_dim] 2d or 3d, maybe strided
template <typename scalar_t>
void fused_sigmoid_mul_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gate,
    int64_t num_tokens,
    int64_t dim,
    int64_t num_heads,
    int64_t head_dim,
    int64_t g_strideT,
    int64_t g_strideH) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t kVecSize = bVec::size();
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* __restrict__ i_ptr = input + i * dim;
      const scalar_t* __restrict__ g_ptr = gate + i * g_strideT;
      scalar_t* __restrict__ o_ptr = output + i * dim;

      for (int64_t h = 0; h < num_heads; ++h) {
        const scalar_t* __restrict__ attn_ptr = i_ptr + h * head_dim;
        const scalar_t* __restrict__ gate_ptr = g_ptr + h * g_strideH;
        scalar_t* __restrict__ out_ptr = o_ptr + h * head_dim;

        int64_t d = 0;
#pragma GCC unroll 4
        for (; d <= head_dim - kVecSize; d += kVecSize) {
          auto [x_fvec0, x_fvec1] = load_float_vec2(attn_ptr + d);
          auto [g_fvec0, g_fvec1] = load_float_vec2(gate_ptr + d);
          x_fvec0 = x_fvec0 * fast_sigmoid(g_fvec0);
          x_fvec1 = x_fvec1 * fast_sigmoid(g_fvec1);
          convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1).store(out_ptr + d);
        }
#pragma GCC unroll 4
        for (; d < head_dim; ++d) {
          float x_val = static_cast<float>(attn_ptr[d]);
          float g_val = static_cast<float>(gate_ptr[d]);
          out_ptr[d] = static_cast<scalar_t>(x_val / (1.f + std::exp(-g_val)));
        }
      }
    }
  });
}

}  // anonymous namespace

// input   : {num_tokens, 2 * d}
// output  : {num_tokens, d}
at::Tensor silu_and_mul_cpu(at::Tensor& input) {
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "silu_and_mul", [&] {
    using Vec = at::vec::Vectorized<float>;
    act_and_mul_kernel_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        d,
        [](float x) { return x / (1.f + std::exp(-x)); },
        [](Vec x) { return fast_silu(x); });
  });
  return out;
}

at::Tensor gelu_tanh_and_mul_cpu(const at::Tensor& input) {
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());
  const float sqrt_2_div_pi = std::sqrt(2.f / M_PI);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "gelu_tanh_and_mul", [&] {
    using Vec = at::vec::Vectorized<float>;
    act_and_mul_kernel_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        d,
        [sqrt_2_div_pi](float x) {
          float x3 = x * x * x;
          float tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x3);
          return 0.5f * x * (1.f + std::tanh(tanh_arg));
        },
        [sqrt_2_div_pi](Vec x) {
          Vec x3 = x * x * x;
          Vec tanh_arg = Vec(sqrt_2_div_pi) * (x + Vec(0.044715f) * x3);
          return Vec(0.5f) * x * (Vec(1.f) + tanh_arg.tanh());
        });
  });

  return out;
}

at::Tensor gelu_and_mul_cpu(const at::Tensor& input) {
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "gelu_and_mul", [&] {
    using Vec = at::vec::Vectorized<float>;
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    act_and_mul_kernel_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        d,
        [inv_sqrt2](float x) { return 0.5f * x * (1.f + std::erf(x * inv_sqrt2)); },
        [inv_sqrt2](Vec x) { return Vec(0.5f) * x * (Vec(1.f) + (x * Vec(inv_sqrt2)).erf()); });
  });

  return out;
}

at::Tensor fused_sigmoid_mul_cpu(at::Tensor& input, const at::Tensor& gate, bool inplace) {
  CHECK_DIM(2, input);
  const int64_t gate_dim = gate.dim();
  TORCH_CHECK(gate_dim == 2 || gate_dim == 3, "gate must be a 2D or 3D tensor");
  CHECK_CONTIGUOUS(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(gate);

  const auto st = input.scalar_type();
  CHECK_EQ(gate.scalar_type(), st);

  int64_t num_tokens = input.size(0);
  int64_t d = input.size(1);

  const bool is_gate_3d = gate_dim == 3;
  int64_t num_heads = is_gate_3d ? gate.size(1) : 1;
  int64_t head_dim = gate.size(-1);
  CHECK_EQ(gate.size(0), num_tokens);
  CHECK_EQ(d, num_heads * head_dim);

  int64_t g_strideT = gate.stride(0);
  int64_t g_strideH = is_gate_3d ? gate.stride(1) : 0;

  at::Tensor out = inplace ? input : at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_sigmoid_mul", [&] {
    fused_sigmoid_mul_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(),
        num_tokens,
        d,
        num_heads,
        head_dim,
        g_strideT,
        g_strideH);
  });
  return out;
}
