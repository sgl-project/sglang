#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#if defined(CPU_CAPABILITY_RVV)

#include <cstdint>

#include "common.h"
#include "riscv64/vector_helpers.h"

namespace {

// sqrt(2 / pi), used in GELU tanh approximation
constexpr float kSqrt2DivPi = 0.7978845608028654f;

// 1 / sqrt(2), used in GELU exact (erf) formula
constexpr float kInvSqrt2 = 0.7071067811865476f;

// Inner loop: out[j] = silu(x[j]) * y[j]
// SiLU(x) = x / (1 + exp(-x)) = x * rec(1 + exp(-x))

template <typename scalar_t>
void act_silu_inner(
    scalar_t* __restrict__ out_ptr, const scalar_t* __restrict__ x_ptr, const scalar_t* __restrict__ y_ptr, int64_t d) {
  const size_t max_vl = __riscv_vsetvlmax_e32m4();
  const int64_t max_vl_i = static_cast<int64_t>(max_vl);
  int64_t j = 0;

  // 2-way unrolled main loop (both chunks are full-width, no tail logic needed here)
  for (; j + 2 * max_vl_i <= d; j += 2 * max_vl_i) {
    // ---- chunk A: load + issue exp ----
    vfloat32m4_t vxA = load_as_float_m4(x_ptr + j, max_vl, nullptr);
    vfloat32m4_t vyA = load_as_float_m4(y_ptr + j, max_vl, nullptr);
    vfloat32m4_t vexpA = vfexp_f32m4(__riscv_vfneg_v_f32m4(vxA, max_vl), max_vl);
    // ---- chunk B: load while expA computes ----
    vfloat32m4_t vxB = load_as_float_m4(x_ptr + j + max_vl_i, max_vl, nullptr);
    vfloat32m4_t vyB = load_as_float_m4(y_ptr + j + max_vl_i, max_vl, nullptr);
    vfloat32m4_t vexpB = vfexp_f32m4(__riscv_vfneg_v_f32m4(vxB, max_vl), max_vl);

    vfloat32m4_t vdA = __riscv_vfadd_vf_f32m4(vexpA, 1.0f, max_vl);
    vfloat32m4_t voutA =
        __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(vxA, vrec_f32m4(vdA, max_vl), max_vl), vyA, max_vl);
    store_from_float_m4(out_ptr + j, voutA, max_vl, nullptr);

    vfloat32m4_t vdB = __riscv_vfadd_vf_f32m4(vexpB, 1.0f, max_vl);
    vfloat32m4_t voutB =
        __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(vxB, vrec_f32m4(vdB, max_vl), max_vl), vyB, max_vl);
    store_from_float_m4(out_ptr + j + max_vl_i, voutB, max_vl, nullptr);
  }

  // tail: handles remainder (including d < 2*max_vl entirely)
  size_t vl;
  for (; j < d; j += vl) {
    vl = __riscv_vsetvl_e32m4(d - j);
    vfloat32m4_t vx = load_as_float_m4(x_ptr + j, vl, nullptr);
    vfloat32m4_t vy = load_as_float_m4(y_ptr + j, vl, nullptr);
    vfloat32m4_t vd = __riscv_vfadd_vf_f32m4(vfexp_f32m4(__riscv_vfneg_v_f32m4(vx, vl), vl), 1.0f, vl);
    store_from_float_m4(
        out_ptr + j, __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(vx, vrec_f32m4(vd, vl), vl), vy, vl), vl, nullptr);
  }
}

// Inner loop: out[j] = gelu_tanh(x[j]) * y[j]
// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

template <typename scalar_t>
void act_gelu_tanh_inner(
    scalar_t* __restrict__ out_ptr, const scalar_t* __restrict__ x_ptr, const scalar_t* __restrict__ y_ptr, int64_t d) {
  size_t vl;
  for (int64_t j = 0; j < d; j += vl) {
    vl = __riscv_vsetvl_e32m4(d - j);
    vfloat32m4_t vx = load_as_float_m4(x_ptr + j, vl, nullptr);
    vfloat32m4_t vx2 = __riscv_vfmul_vv_f32m4(vx, vx, vl);
    vfloat32m4_t vx3 = __riscv_vfmul_vv_f32m4(vx2, vx, vl);
    vfloat32m4_t vinner = __riscv_vfmacc_vf_f32m4(vx, 0.044715f, vx3, vl);
    vfloat32m4_t vtanh_arg = __riscv_vfmul_vf_f32m4(vinner, kSqrt2DivPi, vl);
    vfloat32m4_t vy = load_as_float_m4(y_ptr + j, vl, nullptr);
    vfloat32m4_t vtanh = vftanh_f32m4(vtanh_arg, vl);
    // gelu = 0.5 * x * (1 + tanh) * vy
    vfloat32m4_t vgelu =
        __riscv_vfmul_vf_f32m4(__riscv_vfmul_vv_f32m4(vx, __riscv_vfadd_vf_f32m4(vtanh, 1.0f, vl), vl), 0.5f, vl);
    store_from_float_m4(out_ptr + j, __riscv_vfmul_vv_f32m4(vgelu, vy, vl), vl, nullptr);
  }
}

// Inner loop: out[j] = gelu(x[j]) * y[j]
// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

template <typename scalar_t>
void act_gelu_inner(
    scalar_t* __restrict__ out_ptr, const scalar_t* __restrict__ x_ptr, const scalar_t* __restrict__ y_ptr, int64_t d) {
  size_t vl;
  for (int64_t j = 0; j < d; j += vl) {
    vl = __riscv_vsetvl_e32m4(d - j);
    vfloat32m4_t vx = load_as_float_m4(x_ptr + j, vl, nullptr);
    vfloat32m4_t verf = vferf_f32m4(__riscv_vfmul_vf_f32m4(vx, kInvSqrt2, vl), vl);
    vfloat32m4_t vy = load_as_float_m4(y_ptr + j, vl, nullptr);
    // gelu = 0.5 * x * (1 + erf) * vy
    vfloat32m4_t vgelu =
        __riscv_vfmul_vf_f32m4(__riscv_vfmul_vv_f32m4(vx, __riscv_vfadd_vf_f32m4(verf, 1.0f, vl), vl), 0.5f, vl);
    store_from_float_m4(out_ptr + j, __riscv_vfmul_vv_f32m4(vgelu, vy, vl), vl, nullptr);
  }
}

}  // namespace

// input   : {num_tokens, 2 * d}
// output  : {num_tokens, d}
at::Tensor silu_and_mul_cpu(at::Tensor& input) {
  RECORD_FUNCTION("sgl-kernel::silu_and_mul_cpu", std::vector<c10::IValue>({input}));
  auto input_contig = input.contiguous();
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_contig.scalar_type(), "silu_and_mul", [&] {
    const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
    scalar_t* out_ptr = out.data_ptr<scalar_t>();
    at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const scalar_t* x_ptr = in_ptr + i * 2 * d;
        act_silu_inner(out_ptr + i * d, x_ptr, x_ptr + d, d);
      }
    });
  });
  return out;
}

at::Tensor gelu_tanh_and_mul_cpu(const at::Tensor& input) {
  RECORD_FUNCTION("sgl-kernel::gelu_tanh_and_mul_cpu", std::vector<c10::IValue>({input}));
  auto input_contig = input.contiguous();
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_contig.scalar_type(), "gelu_tanh_and_mul", [&] {
    const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
    scalar_t* out_ptr = out.data_ptr<scalar_t>();
    at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const scalar_t* x_ptr = in_ptr + i * 2 * d;
        act_gelu_tanh_inner(out_ptr + i * d, x_ptr, x_ptr + d, d);
      }
    });
  });
  return out;
}

at::Tensor gelu_and_mul_cpu(const at::Tensor& input) {
  RECORD_FUNCTION("sgl-kernel::gelu_and_mul_cpu", std::vector<c10::IValue>({input}));
  auto input_contig = input.contiguous();
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_contig.scalar_type(), "gelu_and_mul", [&] {
    const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
    scalar_t* out_ptr = out.data_ptr<scalar_t>();
    at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const scalar_t* x_ptr = in_ptr + i * 2 * d;
        act_gelu_inner(out_ptr + i * d, x_ptr, x_ptr + d, d);
      }
    });
  });
  return out;
}

#endif  // CPU_CAPABILITY_RVV
