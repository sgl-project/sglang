// Bit-exact FLUX.2 gated residual + LayerNorm + adaLN modulation.
//
// The D=6144 kernel reproduces both the eager bf16 residual update and
// PyTorch's 128-thread vectorized LayerNorm Welford tree. Unsupported shapes
// stay on the existing unfused model path.

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>

namespace sglang {

namespace flux2_gated_resnorm {

constexpr int kHidden = 6144;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / device::kWarpThreads;
constexpr int kVecElems = 4;
constexpr int kIterations = kHidden / (kThreads * kVecElems);

static_assert(kWarps == 4);
static_assert(kIterations == 12);

struct Params {
  void* output;
  void* residual_out;
  const void* update;
  const void* residual;
  const void* gate;
  const void* scale;
  const void* shift;
  float eps;
};

struct WelfordState {
  float mean;
  float m2;
  float count;
};

SGL_DEVICE float reciprocal_nr(float x) {
  float out;
  asm volatile(
      "{\n\t"
      ".reg .f32 e, e2;\n\t"
      "rcp.approx.f32 %0, %1;\n\t"
      "fma.rn.f32 e, %1, %0, 0fBF800000;\n\t"
      "sub.ftz.f32 e2, 0f80000000, e;\n\t"
      "fma.rn.f32 %0, %0, e2, %0;\n\t"
      "}"
      : "=&f"(out)
      : "f"(x));
  return out;
}

SGL_DEVICE float div_rn(float numerator, float denominator) {
  float out;
  asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(numerator), "f"(denominator));
  return out;
}

SGL_DEVICE WelfordState welford_push(WelfordState state, float value) {
  const float delta = __fsub_rn(value, state.mean);
  const float count = __fadd_rn(state.count, 1.0f);
  const float mean = __fmaf_rn(delta, reciprocal_nr(count), state.mean);
  const float centered = __fsub_rn(value, mean);
  return WelfordState{mean, __fmaf_rn(delta, centered, state.m2), count};
}

SGL_DEVICE WelfordState welford_combine(WelfordState lower, WelfordState upper) {
  const float count = __fadd_rn(upper.count, lower.count);
  if (count <= 0.0f) {
    return WelfordState{0.0f, 0.0f, count};
  }
  const float coefficient = reciprocal_nr(count);
  const float delta = __fsub_rn(lower.mean, upper.mean);
  const float lower_fraction = __fmul_rn(coefficient, lower.count);
  const float delta_squared = __fmul_rn(delta, delta);
  const float upper_fraction = __fmul_rn(upper.count, coefficient);
  const float m2_sum = __fadd_rn(upper.m2, lower.m2);
  const float lower_weighted_mean = __fmul_rn(lower_fraction, lower.mean);
  const float mean = __fmaf_rn(upper.mean, upper_fraction, lower_weighted_mean);
  const float upper_delta_squared = __fmul_rn(upper.count, delta_squared);
  const float m2 = __fmaf_rn(lower_fraction, upper_delta_squared, m2_sum);
  return WelfordState{mean, m2, count};
}

SGL_DEVICE WelfordState warp_welford(WelfordState state, int lane) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const WelfordState upper{
        __shfl_down_sync(0xffffffffu, state.mean, offset),
        __shfl_down_sync(0xffffffffu, state.m2, offset),
        __shfl_down_sync(0xffffffffu, state.count, offset),
    };
    if (lane < offset) {
      state = welford_combine(state, upper);
    }
  }
  return state;
}

__global__ void kernel(const Params __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<bf16_t, kVecElems>;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & int(kWarpThreads - 1);
  const int warp = tid >> 5;
  const int row_offset = row * kHidden;

  WelfordState state{0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const int offset = row_offset + iteration * 512 + tid * kVecElems;
    const int gate_offset = iteration * 512 + tid * kVecElems;
    Vec update;
    Vec residual;
    Vec gate;
    Vec residual_out;
    update.load(static_cast<const bf16_t*>(params.update) + offset);
    residual.load(static_cast<const bf16_t*>(params.residual) + offset);
    gate.load(static_cast<const bf16_t*>(params.gate) + gate_offset);
#pragma unroll
    for (int element = 0; element < kVecElems; ++element) {
      // Match residual_gate_add: bf16-round the product before the add.
      const bf16_t product =
          static_cast<bf16_t>(__fmul_rn(static_cast<float>(update[element]), static_cast<float>(gate[element])));
      const bf16_t updated =
          static_cast<bf16_t>(__fadd_rn(static_cast<float>(residual[element]), static_cast<float>(product)));
      residual_out[element] = updated;
      state = welford_push(state, static_cast<float>(updated));
    }
    residual_out.store(static_cast<bf16_t*>(params.residual_out) + offset);
  }

  // Match aten LayerNorm's four-warp tree: (0,2), (1,3), then (0,1).
  state = warp_welford(state, lane);
  __shared__ WelfordState warp_states[kWarps];
  __shared__ float shared_mean;
  __shared__ float shared_rstd;
  if (lane == 0) {
    warp_states[warp] = state;
  }
  __syncthreads();
  if (tid == 0) {
    const WelfordState pair02 = welford_combine(warp_states[0], warp_states[2]);
    const WelfordState pair13 = welford_combine(warp_states[1], warp_states[3]);
    const WelfordState total = welford_combine(pair02, pair13);
    shared_mean = total.mean;
    shared_rstd = rsqrtf(__fadd_rn(div_rn(total.m2, float(kHidden)), params.eps));
  }
  __syncthreads();

#pragma unroll
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const int offset = row_offset + iteration * 512 + tid * kVecElems;
    const int modulation_offset = iteration * 512 + tid * kVecElems;
    Vec scale;
    Vec shift;
    Vec residual_out;
    Vec output;
    scale.load(static_cast<const bf16_t*>(params.scale) + modulation_offset);
    shift.load(static_cast<const bf16_t*>(params.shift) + modulation_offset);
    residual_out.load(static_cast<const bf16_t*>(params.residual_out) + offset);
#pragma unroll
    for (int element = 0; element < kVecElems; ++element) {
      // Match aten's bf16 LayerNorm output, then eager adaLN's bf16 rounding
      // after 1+scale, multiply, and shift addition.
      const bf16_t normalized = static_cast<bf16_t>(
          __fmul_rn(__fsub_rn(static_cast<float>(residual_out[element]), shared_mean), shared_rstd));
      const bf16_t one_plus_scale = static_cast<bf16_t>(__fadd_rn(1.0f, static_cast<float>(scale[element])));
      const bf16_t scaled =
          static_cast<bf16_t>(__fmul_rn(static_cast<float>(normalized), static_cast<float>(one_plus_scale)));
      output[element] = static_cast<bf16_t>(__fadd_rn(static_cast<float>(scaled), static_cast<float>(shift[element])));
    }
    output.store(static_cast<bf16_t*>(params.output) + offset);
  }
}

struct Kernel {
  static void
  run(tvm::ffi::TensorView output,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView update,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({rows, kHidden})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output)
        .verify(residual_out)
        .verify(residual)
        .verify(update);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(gate).verify(scale).verify(shift);
    RuntimeCheck(rows.unwrap() > 0, "rows must be positive");
    RuntimeCheck(rows.unwrap() <= int64_t(UINT32_MAX), "rows out of range");

    const auto params = Params{
        .output = output.data_ptr(),
        .residual_out = residual_out.data_ptr(),
        .update = update.data_ptr(),
        .residual = residual.data_ptr(),
        .gate = gate.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(static_cast<uint32_t>(rows.unwrap()), kThreads, device.unwrap())(kernel, params);
  }
};

}  // namespace flux2_gated_resnorm

}  // namespace sglang
