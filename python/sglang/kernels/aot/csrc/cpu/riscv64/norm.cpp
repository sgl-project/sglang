#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#if defined(CPU_CAPABILITY_RVV)

#include <cmath>
#include <cstdint>

#include "common.h"
#include "riscv64/vector_helpers.h"

namespace {

// L2 norm: out[d] = x[d] * rsqrt(sum(x^2) / N + eps)
template <typename scalar_t>
void l2norm_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t batch_size,
    int64_t hidden_size,
    float eps) {
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* in_ptr = input + i * hidden_size;
      scalar_t* out_ptr = output + i * hidden_size;

      // Pass 1: accumulate sum of squares
      float sum_sq = 0.0f;
      size_t vl = 0;
      vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        vfloat32m4_t vsq = __riscv_vfmul_vv_f32m4(vx, vx, vl);
        sum_sq += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsq, vzero, vl));
      }
      float scale = 1.0f / std::sqrt(sum_sq / hidden_size + eps);

      // Pass 2: scale and store
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        store_from_float_m4(out_ptr + j, __riscv_vfmul_vf_f32m4(vx, scale, vl), vl, scratch);
      }
    }
  });
}

// RMSNorm (plain and gemma variant via template bool):
//   plain:  out[d] = x[d] * rsqrt_var * w[d]
//   gemma:  out[d] = x[d] * rsqrt_var * (1 + w[d])
template <typename scalar_t, bool is_gemma>
void rmsnorm_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int64_t batch_size,
    int64_t hidden_size,
    int64_t input_strideN,
    float eps) {
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* in_ptr = input + i * input_strideN;
      scalar_t* out_ptr = output + i * hidden_size;

      // Pass 1: sum x^2
      float sum_sq = 0.0f;
      size_t vl = 0;
      vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        vfloat32m4_t vsq = __riscv_vfmul_vv_f32m4(vx, vx, vl);
        sum_sq += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsq, vzero, vl));
      }
      float rsqrt_var = 1.0f / std::sqrt(sum_sq / hidden_size + eps);

      // Pass 2: out = x * rsqrt_var * w (or (1+w) for gemma)
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        vfloat32m4_t vw = load_as_float_m4(weight + j, vl, scratch);
        if constexpr (is_gemma) {
          vw = __riscv_vfadd_vf_f32m4(vw, 1.0f, vl);
        }
        vfloat32m4_t vout = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(vx, rsqrt_var, vl), vw, vl);
        store_from_float_m4(out_ptr + j, vout, vl, scratch);
      }
    }
  });
}

// Gemma3 RMSNorm 4D: input [B, H, S, D] with arbitrary strides
// Same as gemma rmsnorm but iterates over (B, H, S) with data_index_step.
template <typename scalar_t>
void gemma3_rmsnorm_kernel_4d_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int64_t batch_size,
    int64_t num_head,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t input_strideB,
    int64_t input_strideH,
    int64_t input_strideS,
    int64_t output_strideB,
    int64_t output_strideH,
    int64_t output_strideS,
    float eps) {
  at::parallel_for(0, batch_size * num_head * seq_len, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    int64_t bi{0}, hi{0}, si{0};
    data_index_init(begin, bi, batch_size, hi, num_head, si, seq_len);
    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* in_ptr = input + bi * input_strideB + hi * input_strideH + si * input_strideS;
      scalar_t* out_ptr = output + bi * output_strideB + hi * output_strideH + si * output_strideS;

      // Pass 1: sum x^2
      float sum_sq = 0.0f;
      size_t vl = 0;
      vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        vfloat32m4_t vsq = __riscv_vfmul_vv_f32m4(vx, vx, vl);
        sum_sq += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsq, vzero, vl));
      }
      float rsqrt_var = 1.0f / std::sqrt(sum_sq / hidden_size + eps);

      // Pass 2: out = x * rsqrt_var * (1 + w)
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        vfloat32m4_t vw = load_as_float_m4(weight + j, vl, scratch);
        vw = __riscv_vfadd_vf_f32m4(vw, 1.0f, vl);
        vfloat32m4_t vout = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(vx, rsqrt_var, vl), vw, vl);
        store_from_float_m4(out_ptr + j, vout, vl, scratch);
      }
      data_index_step(bi, batch_size, hi, num_head, si, seq_len);
    }
  });
}

// Fused add-RMSNorm (+ gemma variant via template bool):
//   Pass 1: vs = input[d] + residual[d];
//           residual[d] = scalar_t(vs);   buffer[d] = float(vs);
//           sum_sq += vs^2
//   Pass 2: input[d] = buffer[d] * rsqrt_var * w[d] (or (1+w))
template <typename scalar_t, bool is_gemma>
void fused_add_rmsnorm_kernel_impl(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    float* __restrict__ buffer,
    int64_t batch_size,
    int64_t hidden_size,
    int64_t input_strideN,
    float eps) {
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    int tid = at::get_thread_num();
    float* __restrict__ buf_ptr = buffer + tid * hidden_size;

    for (int64_t i = begin; i < end; ++i) {
      scalar_t* __restrict__ in_ptr = input + i * input_strideN;
      scalar_t* __restrict__ res_ptr = residual + i * hidden_size;

      // Pass 1: fused add → residual; fill float buffer; accumulate sum-sq
      float sum_sq = 0.0f;
      size_t vl = 0;
      vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        vfloat32m4_t vr = load_as_float_m4(res_ptr + j, vl, scratch);
        vfloat32m4_t vs = __riscv_vfadd_vv_f32m4(vx, vr, vl);
        // Write x+r to residual (converted to scalar_t)
        store_from_float_m4(res_ptr + j, vs, vl, scratch);
        // Keep float32 copy in buffer for Pass 2
        __riscv_vse32_v_f32m4(buf_ptr + j, vs, vl);
        vfloat32m4_t vsq = __riscv_vfmul_vv_f32m4(vs, vs, vl);
        sum_sq += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsq, vzero, vl));
      }
      float rsqrt_var = 1.0f / std::sqrt(sum_sq / hidden_size + eps);

      // Pass 2: input[d] = buffer[d] * rsqrt_var * w[d] (or (1+w))
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vbuf = __riscv_vle32_v_f32m4(buf_ptr + j, vl);
        vfloat32m4_t vw = load_as_float_m4(weight + j, vl, scratch);
        if constexpr (is_gemma) {
          vw = __riscv_vfadd_vf_f32m4(vw, 1.0f, vl);
        }
        vfloat32m4_t vout = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(vbuf, rsqrt_var, vl), vw, vl);
        store_from_float_m4(in_ptr + j, vout, vl, scratch);
      }
    }
  });
}

// Fused RMSNorm+Gated: out = rms_norm(x) * w * silu(gate)
// silu(g) = g * rec(1 + exp(-g))  — reuses vrec_f32m4 from vector_math.h
template <typename scalar_t>
void fused_rmsnorm_gated_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ gate,
    int64_t batch_size,
    int64_t hidden_size,
    int64_t input_strideN,
    float eps) {
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* in_ptr = input + i * input_strideN;
      const scalar_t* gate_ptr = gate + i * hidden_size;
      scalar_t* out_ptr = output + i * hidden_size;

      // Pass 1: sum x^2
      float sum_sq = 0.0f;
      size_t vl = 0;
      vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        vfloat32m4_t vsq = __riscv_vfmul_vv_f32m4(vx, vx, vl);
        sum_sq += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsq, vzero, vl));
      }
      float rsqrt_var = 1.0f / std::sqrt(sum_sq / hidden_size + eps);

      // Pass 2: out = x * rsqrt_var * w * silu(g)
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        vfloat32m4_t vw = load_as_float_m4(weight + j, vl, scratch);
        vfloat32m4_t vg = load_as_float_m4(gate_ptr + j, vl, scratch);
        // silu(g) = g / (1 + exp(-g)) = g * approx_rec(1 + exp(-g))
        vfloat32m4_t vdenom = __riscv_vfadd_vf_f32m4(vfexp_f32m4(__riscv_vfneg_v_f32m4(vg, vl), vl), 1.0f, vl);
        vfloat32m4_t vsilu = __riscv_vfmul_vv_f32m4(vg, vrec_f32m4(vdenom, vl), vl);
        vfloat32m4_t vout = __riscv_vfmul_vv_f32m4(
            __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(vx, rsqrt_var, vl), vw, vl), vsilu, vl);
        store_from_float_m4(out_ptr + j, vout, vl, scratch);
      }
    }
  });
}

// Fused add-LayerNorm (residual may be nullptr for plain LayerNorm):
//   Pass 1: x = input[d] + residual[d] (if any); save to residual and buffer;
//           accumulate sum(x) and sum(x^2).
//   Compute: mean = sum/N; variance = sum_sq/N - mean^2.
//   Pass 2: input[d] = (buffer[d] - mean) * rsqrt(variance + eps) * weight[d].
template <typename scalar_t>
void fused_add_layernorm_kernel_impl(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,  // nullable: plain layernorm passes nullptr
    const scalar_t* __restrict__ weight,
    float* __restrict__ buffer,
    int64_t batch_size,
    int64_t hidden_size,
    int64_t input_strideN,
    float eps) {
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
    int tid = at::get_thread_num();
    float* __restrict__ buf_ptr = buffer + tid * hidden_size;

    for (int64_t i = begin; i < end; ++i) {
      scalar_t* __restrict__ in_ptr = input + i * input_strideN;
      scalar_t* __restrict__ res_ptr = (residual != nullptr) ? residual + i * hidden_size : nullptr;

      // Pass 1: optional fused add; fill buffer; accumulate sum and sum-sq
      float sum_val = 0.0f, sum_sq = 0.0f;
      size_t vl = 0;
      vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vx = load_as_float_m4(in_ptr + j, vl, scratch);
        if (res_ptr != nullptr) {
          vfloat32m4_t vr = load_as_float_m4(res_ptr + j, vl, scratch);
          vx = __riscv_vfadd_vv_f32m4(vx, vr, vl);
          store_from_float_m4(res_ptr + j, vx, vl, scratch);
        }
        sum_val += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vx, vzero, vl));
        vfloat32m4_t vsq = __riscv_vfmul_vv_f32m4(vx, vx, vl);
        sum_sq += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsq, vzero, vl));
        __riscv_vse32_v_f32m4(buf_ptr + j, vx, vl);
      }

      // LayerNorm statistics: Var(X) = E(X²) - E(X)²
      // Note: E(X²) - E(X)² can produce negative values due to FP cancellation when
      // activations are large; clamp to 0 before adding eps to avoid sqrt(negative).
      float mean = sum_val / hidden_size;
      float mean_sq = sum_sq / hidden_size;
      float variance = std::max(0.0f, mean_sq - mean * mean);
      float rsqrt_var = 1.0f / std::sqrt(variance + eps);

      // Pass 2: input[d] = (buffer[d] - mean) * rsqrt_var * weight[d]
      for (int64_t j = 0; j < hidden_size; j += vl) {
        vl = __riscv_vsetvl_e32m4(hidden_size - j);
        vfloat32m4_t vbuf = __riscv_vle32_v_f32m4(buf_ptr + j, vl);
        vfloat32m4_t vw = load_as_float_m4(weight + j, vl, scratch);
        vfloat32m4_t vx = __riscv_vfsub_vf_f32m4(vbuf, mean, vl);
        vfloat32m4_t vout = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(vx, rsqrt_var, vl), vw, vl);
        store_from_float_m4(in_ptr + j, vout, vl, scratch);
      }
    }
  });
}

}  // namespace

// Public API from sgl-kernel/csrc/cpu/norm.cpp
// input: {batch_size, hidden_size}
at::Tensor l2norm_cpu(at::Tensor& input, double eps) {
  RECORD_FUNCTION("sgl-kernel::l2norm_cpu", std::vector<c10::IValue>({input}));
  CHECK_INPUT(input);
  CHECK_DIM(2, input);
  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "l2norm_kernel", [&] {
    l2norm_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), batch_size, hidden_size, static_cast<float>(eps));
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
  int64_t input_strideN = input.stride(0);
  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "rmsnorm_kernel", [&] {
    rmsnorm_kernel_impl<scalar_t, false>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        batch_size,
        hidden_size,
        input_strideN,
        static_cast<float>(eps));
  });
  return output;
}

// input : {batch_size, hidden_size}
// weight: {hidden_size}
// bias  : {hidden_size} (optional)
at::Tensor
layernorm_cpu(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias, double eps) {
  RECORD_FUNCTION("sgl-kernel::layernorm_cpu", std::vector<c10::IValue>({input, weight}));
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  CHECK_EQ(input.size(1), weight.size(0));
  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  int64_t num_threads = at::get_num_threads();
  at::Tensor buffer = at::empty({num_threads, hidden_size}, input.options().dtype(at::kFloat));
  // Clone input into output: materializes any non-contiguous input as contiguous,
  // so the kernel always sees stride(0) == hidden_size (no separate strideN parameter needed).
  at::Tensor output = input.clone();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "layernorm_kernel", [&] {
    fused_add_layernorm_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        nullptr,
        weight.data_ptr<scalar_t>(),
        buffer.data_ptr<float>(),
        batch_size,
        hidden_size,
        output.stride(0),
        static_cast<float>(eps));
  });
  if (bias.has_value()) {
    output.add_(bias.value());
  }
  return output;
}

// input : {batch_size, hidden_size}
// weight: {hidden_size}  — scale = 1 + weight
at::Tensor gemma_rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps) {
  RECORD_FUNCTION("sgl-kernel::gemma_rmsnorm_cpu", std::vector<c10::IValue>({input, weight}));
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  CHECK_EQ(input.size(1), weight.size(0));
  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  int64_t input_strideN = input.stride(0);
  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "gemma_rmsnorm_kernel", [&] {
    rmsnorm_kernel_impl<scalar_t, true>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        batch_size,
        hidden_size,
        input_strideN,
        static_cast<float>(eps));
  });
  return output;
}

// input : {batch_size, hidden_size} or {batch_size, num_head, seq_len, head_dim}
// weight: {hidden_size}  — scale = 1 + weight
at::Tensor gemma3_rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps) {
  RECORD_FUNCTION("sgl-kernel::gemma3_rmsnorm_cpu", std::vector<c10::IValue>({input, weight}));
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT(weight);
  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 4, "gemma3_rmsnorm_cpu: input must be 2D or 4D, got ", input.dim(), "D");
  CHECK_DIM(1, weight);
  CHECK_EQ(input.size(-1), weight.size(0));
  int64_t batch_size = input.size(0);
  int64_t hidden_size = weight.size(0);
  at::Tensor output = at::empty_like(input);
  if (input.dim() == 2) {
    int64_t input_strideN = input.stride(0);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "gemma3_rmsnorm_kernel", [&] {
      rmsnorm_kernel_impl<scalar_t, true>(
          output.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          batch_size,
          hidden_size,
          input_strideN,
          static_cast<float>(eps));
    });
  } else {
    int64_t input_strideB = input.stride(0);
    int64_t input_strideH = input.stride(1);
    int64_t input_strideS = input.stride(2);
    int64_t output_strideB = output.stride(0);
    int64_t output_strideH = output.stride(1);
    int64_t output_strideS = output.stride(2);
    int64_t num_head = input.size(1);
    int64_t seq_len = input.size(2);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "gemma3_rmsnorm_kernel_4d", [&] {
      gemma3_rmsnorm_kernel_4d_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(),
          batch_size,
          num_head,
          seq_len,
          hidden_size,
          input_strideB,
          input_strideH,
          input_strideS,
          output_strideB,
          output_strideH,
          output_strideS,
          static_cast<float>(eps));
    });
  }
  return output;
}

// input : {batch_size, hidden_size}
// weight: {hidden_size}, gate: {batch_size, hidden_size}
at::Tensor fused_rmsnorm_gated_cpu(at::Tensor& input, at::Tensor& weight, at::Tensor& gate, double eps) {
  RECORD_FUNCTION("sgl-kernel::fused_rmsnorm_gated_cpu", std::vector<c10::IValue>({input, weight, gate}));
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(gate);
  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  CHECK_DIM(2, gate);
  CHECK_EQ(input.size(1), weight.size(0));
  CHECK_EQ(input.size(0), gate.size(0));
  CHECK_EQ(input.size(1), gate.size(1));
  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  int64_t input_strideN = input.stride(0);
  at::Tensor output = at::empty_like(input);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fused_rmsnorm_gated_kernel", [&] {
    fused_rmsnorm_gated_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(),
        batch_size,
        hidden_size,
        input_strideN,
        static_cast<float>(eps));
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
  int64_t num_threads = at::get_num_threads();
  at::Tensor buffer = at::empty({num_threads, hidden_size}, input.options().dtype(at::kFloat));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fused_add_rmsnorm_kernel", [&] {
    fused_add_rmsnorm_kernel_impl<scalar_t, false>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        buffer.data_ptr<float>(),
        batch_size,
        hidden_size,
        input_strideN,
        static_cast<float>(eps));
  });
}

void gemma_fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps) {
  RECORD_FUNCTION("sgl-kernel::gemma_fused_add_rmsnorm_cpu", std::vector<c10::IValue>({input, residual, weight}));
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
  int64_t num_threads = at::get_num_threads();
  at::Tensor buffer = at::empty({num_threads, hidden_size}, input.options().dtype(at::kFloat));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "gemma_fused_add_rmsnorm_kernel", [&] {
    fused_add_rmsnorm_kernel_impl<scalar_t, true>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        buffer.data_ptr<float>(),
        batch_size,
        hidden_size,
        input_strideN,
        static_cast<float>(eps));
  });
}

// input   : {batch_size, hidden_size}
// residual: {batch_size, hidden_size}
// weight  : {hidden_size}
// bias    : {hidden_size} (optional)
at::Tensor fused_add_layernorm_cpu(
    const at::Tensor& input,
    at::Tensor& residual,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    double eps) {
  RECORD_FUNCTION("sgl-kernel::fused_add_layernorm_cpu", std::vector<c10::IValue>({input, residual, weight}));
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
  int64_t num_threads = at::get_num_threads();
  at::Tensor buffer = at::empty({num_threads, hidden_size}, input.options().dtype(at::kFloat));
  // Clone input into output; kernel writes normalized result in-place to output
  at::Tensor output = input.clone();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "fused_add_layernorm_kernel", [&] {
    fused_add_layernorm_kernel_impl<scalar_t>(
        output.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        buffer.data_ptr<float>(),
        batch_size,
        hidden_size,
        output.stride(0),
        static_cast<float>(eps));
  });
  if (bias.has_value()) {
    output.add_(bias.value());
  }
  return output;
}

#endif  // CPU_CAPABILITY_RVV
