#include "common.h"
#include "vec.h"

namespace {

// NB: avoid using `at::vec::map<>` on bfloat16 or half
// Llama4TextL2Norm
template <typename scalar_t>
void l2norm_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t batch_size,
    int64_t hidden_size,
    float eps = 1e-5) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int kVecSize = bVec::size();
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      // local ptrs
      scalar_t* __restrict__ out_ptr = output + i * hidden_size;
      const scalar_t* __restrict__ input_ptr = input + i * hidden_size;

      fVec sum_fvec = fVec(float(0));
      float sum_val = float(0);

      int64_t d;
#pragma GCC unroll 4
      for (d = 0; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec += x_fvec0 * x_fvec0;
        sum_fvec += x_fvec1 * x_fvec1;
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        sum_val += x_val * x_val;
      }

      sum_val += vec_reduce_sum(sum_fvec);
      float rsqrt_var = float(1) / std::sqrt(sum_val / hidden_size + eps);
      const fVec scale_fvec = fVec(rsqrt_var);

#pragma GCC unroll 4
      for (d = 0; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        x_fvec0 = x_fvec0 * scale_fvec;
        x_fvec1 = x_fvec1 * scale_fvec;

        bVec out_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
        out_bvec.store(out_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        out_ptr[d] = static_cast<scalar_t>(x_val * rsqrt_var);
      }
    }
  });
}
template <typename scalar_t>
void rmsnorm_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int64_t batch_size,
    int64_t hidden_size,
    int64_t input_strideN,
    float eps = 1e-5) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int kVecSize = bVec::size();
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      // local ptrs
      scalar_t* __restrict__ out_ptr = output + i * hidden_size;
      const scalar_t* __restrict__ input_ptr = input + i * input_strideN;

      fVec sum_fvec = fVec(float(0));
      float sum_val = float(0);

      int64_t d;
#pragma GCC unroll 4
      for (d = 0; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec += x_fvec0 * x_fvec0;
        sum_fvec += x_fvec1 * x_fvec1;
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        sum_val += x_val * x_val;
      }

      sum_val += vec_reduce_sum(sum_fvec);
      float rsqrt_var = float(1) / std::sqrt(sum_val / hidden_size + eps);
      const fVec scale_fvec = fVec(rsqrt_var);

#pragma GCC unroll 4
      for (d = 0; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        bVec w_bvec = bVec::loadu(weight + d);
        fVec w_fvec0, w_fvec1;
        std::tie(w_fvec0, w_fvec1) = at::vec::convert_to_float(w_bvec);

        x_fvec0 = x_fvec0 * scale_fvec * w_fvec0;
        x_fvec1 = x_fvec1 * scale_fvec * w_fvec1;

        bVec out_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
        out_bvec.store(out_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        float w_val = static_cast<float>(weight[d]);
        out_ptr[d] = static_cast<scalar_t>(x_val * rsqrt_var * w_val);
      }
    }
  });
}

template <typename scalar_t>
void fused_add_rmsnorm_kernel_impl(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    float* __restrict__ buffer,
    int64_t batch_size,
    int64_t hidden_size,
    int64_t input_strideN,
    float eps = 1e-5) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int kVecSize = bVec::size();
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    float* __restrict__ buffer_ptr = buffer + tid * hidden_size;

    for (int64_t i = begin; i < end; ++i) {
      // local ptrs
      scalar_t* __restrict__ input_ptr = input + i * input_strideN;
      scalar_t* __restrict__ residual_ptr = residual + i * hidden_size;

      fVec sum_fvec = fVec(float(0));
      float sum_val = float(0);

      int64_t d;
#pragma GCC unroll 4
      for (d = 0; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        bVec r_bvec = bVec::loadu(residual_ptr + d);
        fVec r_fvec0, r_fvec1;
        std::tie(r_fvec0, r_fvec1) = at::vec::convert_to_float(r_bvec);

        x_fvec0 += r_fvec0;
        x_fvec1 += r_fvec1;

        bVec out_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
        out_bvec.store(residual_ptr + d);

        sum_fvec += x_fvec0 * x_fvec0;
        sum_fvec += x_fvec1 * x_fvec1;

        x_fvec0.store(buffer_ptr + d);
        x_fvec1.store(buffer_ptr + d + fVec::size());
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        float r_val = static_cast<float>(residual_ptr[d]);

        x_val += r_val;
        residual_ptr[d] = static_cast<scalar_t>(x_val);

        sum_val += x_val * x_val;
        buffer_ptr[d] = x_val;
      }

      sum_val += vec_reduce_sum(sum_fvec);
      float rsqrt_var = float(1) / std::sqrt(sum_val / hidden_size + eps);
      const fVec scale_fvec = fVec(rsqrt_var);

#pragma GCC unroll 4
      for (d = 0; d <= hidden_size - kVecSize; d += kVecSize) {
        fVec x_fvec0 = fVec::loadu(buffer_ptr + d);
        fVec x_fvec1 = fVec::loadu(buffer_ptr + d + fVec::size());

        bVec w_bvec = bVec::loadu(weight + d);
        fVec w_fvec0, w_fvec1;
        std::tie(w_fvec0, w_fvec1) = at::vec::convert_to_float(w_bvec);

        x_fvec0 = x_fvec0 * scale_fvec * w_fvec0;
        x_fvec1 = x_fvec1 * scale_fvec * w_fvec1;
        bVec x_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
        x_bvec.store(input_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = buffer_ptr[d] * rsqrt_var * static_cast<float>(weight[d]);
        input_ptr[d] = x_val;
      }
    }
  });
}

template <typename scalar_t>
void fused_rmsnorm_gated_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ gate,
    int64_t batch_size,
    int64_t hidden_size,
    int64_t input_strideN,
    float eps = 1e-5) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);

  constexpr int kVecSize = bVec::size();
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      // local ptrs
      scalar_t* __restrict__ out_ptr = output + i * hidden_size;
      const scalar_t* __restrict__ input_ptr = input + i * input_strideN;
      const scalar_t* __restrict__ gate_ptr = gate + i * hidden_size;

      fVec sum_fvec = fVec(float(0));
      float sum_val = float(0);

      int64_t d;
#pragma GCC unroll 4
      for (d = 0; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec += x_fvec0 * x_fvec0;
        sum_fvec += x_fvec1 * x_fvec1;
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        sum_val += x_val * x_val;
      }

      sum_val += vec_reduce_sum(sum_fvec);
      float rsqrt_var = float(1) / std::sqrt(sum_val / hidden_size + eps);
      const fVec scale_fvec = fVec(rsqrt_var);

#pragma GCC unroll 4
      for (d = 0; d <= hidden_size - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        bVec w_bvec = bVec::loadu(weight + d);
        fVec w_fvec0, w_fvec1;
        std::tie(w_fvec0, w_fvec1) = at::vec::convert_to_float(w_bvec);

        bVec g_bvec = bVec::loadu(gate_ptr + d);
        fVec g_fvec0, g_fvec1;
        std::tie(g_fvec0, g_fvec1) = at::vec::convert_to_float(g_bvec);
        g_fvec0 = g_fvec0 / (one + g_fvec0.neg().exp_u20());
        g_fvec1 = g_fvec1 / (one + g_fvec1.neg().exp_u20());

        x_fvec0 = x_fvec0 * scale_fvec * w_fvec0 * g_fvec0;
        x_fvec1 = x_fvec1 * scale_fvec * w_fvec1 * g_fvec1;

        bVec out_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
        out_bvec.store(out_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        float w_val = static_cast<float>(weight[d]);
        float g_val = static_cast<float>(gate_ptr[d]);

        out_ptr[d] = static_cast<scalar_t>(x_val * rsqrt_var * w_val * g_val / (1.f + std::exp(-g_val)));
      }
    }
  });
}

}  // anonymous namespace

// input : {batch_size, hidden_size}
at::Tensor l2norm_cpu(at::Tensor& input, double eps) {
  RECORD_FUNCTION("sgl-kernel::l2norm_cpu", std::vector<c10::IValue>({input}));

  CHECK_INPUT(input);
  CHECK_DIM(2, input);
  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  at::Tensor output = at::empty_like(input);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "l2norm_kernel", [&] {
    l2norm_kernel_impl<scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), batch_size, hidden_size, eps);
  });
  return output;
}

// input : {batch_size, hidden_size}
// weight: {hidden_size}
at::Tensor rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps) {
  RECORD_FUNCTION("sgl-kernel::rmsnorm_cpu", std::vector<c10::IValue>({input, weight}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  CHECK_EQ(input.size(1), weight.size(0));
  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  at::Tensor output = at::empty_like(input);
  int64_t input_strideN = input.stride(0);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "rmsnorm_kernel", [&] {
    rmsnorm_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        batch_size,
        hidden_size,
        input_strideN,
        eps);
  });
  return output;
}

// input : {batch_size, hidden_size}
// weight: {hidden_size}
// gate: {batch_size, hidden_size}
at::Tensor fused_rmsnorm_gated_cpu(at::Tensor& input, at::Tensor& weight, at::Tensor& gate, double eps) {
  RECORD_FUNCTION("sgl-kernel::fused_rmsnorm_gated_cpu", std::vector<c10::IValue>({input, weight, gate}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(gate);
  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  CHECK_DIM(2, gate);
  CHECK_EQ(input.size(1), weight.size(0));
  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  CHECK_EQ(input.size(0), gate.size(0));
  CHECK_EQ(input.size(1), gate.size(1));
  at::Tensor output = at::empty_like(input);
  int64_t input_strideN = input.stride(0);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fused_rmsnorm_gated_kernel", [&] {
    fused_rmsnorm_gated_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(),
        batch_size,
        hidden_size,
        input_strideN,
        eps);
  });
  return output;
}

// input   : {batch_size, hidden_size}
// residual: {batch_size, hidden_size}
// weight  : {hidden_size}
void fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps) {
  RECORD_FUNCTION("sgl-kernel::fused_add_rmsnorm_cpu", std::vector<c10::IValue>({input, residual, weight}));
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  CHECK_DIM(2, input);
  CHECK_DIM(2, residual);
  CHECK_DIM(1, weight);
  CHECK_EQ(input.size(0), residual.size(0));
  CHECK_EQ(input.size(1), residual.size(1));
  CHECK_EQ(input.size(1), weight.size(0));
  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  int64_t input_strideN = input.stride(0);

  // allocate temp buffer to store x in float32 per thread
  // TODO: implement a singleton for context
  int64_t num_threads = at::get_num_threads();
  at::Tensor buffer = at::empty({num_threads, hidden_size}, input.options().dtype(at::kFloat));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fused_add_rmsnorm_kernel", [&] {
    fused_add_rmsnorm_kernel_impl<scalar_t>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        buffer.data_ptr<float>(),
        batch_size,
        hidden_size,
        input_strideN,
        eps);
  });
}
