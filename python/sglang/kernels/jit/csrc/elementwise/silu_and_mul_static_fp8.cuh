#pragma once

#include "activation.cuh"
#include <cuda_fp8.h>

namespace sglang {
template <typename T>
__global__ void silu_and_mul_static_fp8_kernel(
    const T* input, uint8_t* output, const float* scale, float* rows, uint32_t m, uint32_t n) {
  using namespace device;
  constexpr uint32_t V = 8;
  using in_vec = AlignedVector<T, V>;
  using out_vec = AlignedVector<uint8_t, V>;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t vectors = n / V;
  const uint32_t row = tid / vectors;
  if (row >= m) return;
  const uint32_t col = tid % vectors;
  const auto gate = load_as<in_vec>(input, row * vectors * 2 + col);
  const auto up = load_as<in_vec>(input, row * vectors * 2 + vectors + col);
  const float s = *scale;
  const float inv = 1.0f / s;
  out_vec q;
#pragma unroll
  for (int i = 0; i < V; ++i) {
    const float activated = apply_activation_f32<ActivationKind::kSiLU>(cast<fp32_t>(gate[i]));
    // Match the intermediate dtype of the unfused activation output.
    const T rounded = cast<T>(activated * cast<fp32_t>(up[i]));
    float value = cast<fp32_t>(rounded) * inv;
    // Match Triton's max/min NaN handling as well as finite saturation.
    value = fminf(448.0f, fmaxf(-448.0f, value));
    // Match the reference SM89 static-quant lowering: FP32 -> FP16
    // toward zero, then FP16 -> E4M3 round-to-nearest. Direct FP32 -> FP8
    // differs at values just above an FP8 rounding midpoint.
    q[i] = __nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(__float2half_rz(value)), __NV_SATFINITE, __NV_E4M3);
  }
  store_as<out_vec>(output, q, tid);
  if (col == 0) rows[row] = s;
}

template <typename T>
struct SiluAndMulStaticFP8 {
  static void
  run(tvm::ffi::TensorView input, tvm::ffi::TensorView output, tvm::ffi::TensorView scale, tvm::ffi::TensorView rows) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto K2 = SymbolicSize{"K2"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K2}).with_dtype<T>().with_device(device).verify(input);
    TensorMatcher({M, K}).with_dtype<uint8_t>().with_device(device).verify(output);
    TensorMatcher({1}).with_dtype<float>().with_device(device).verify(scale);
    TensorMatcher({M, 1}).with_dtype<float>().with_device(device).verify(rows);
    const auto m = M.unwrap(), n = K.unwrap();
    RuntimeCheck(n > 0 && n % 8 == 0 && K2.unwrap() == n * 2, "unsupported dimensions");
    RuntimeCheck(m * n <= std::numeric_limits<uint32_t>::max() / 2, "index overflow");
    if (m == 0) return;
    // A contiguous view may still start at an unaligned storage offset.
    RuntimeCheck(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0, "input must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(output.data_ptr()) % 8 == 0, "output must be 8-byte aligned");
    LaunchKernel(div_ceil(static_cast<uint32_t>(m * n / 8), 256u), 256u, device.unwrap())(
        silu_and_mul_static_fp8_kernel<T>,
        static_cast<const T*>(input.data_ptr()),
        static_cast<uint8_t*>(output.data_ptr()),
        static_cast<const float*>(scale.data_ptr()),
        static_cast<float*>(rows.data_ptr()),
        static_cast<uint32_t>(m),
        static_cast<uint32_t>(n));
  }
};
}  // namespace sglang
using sglang::SiluAndMulStaticFP8;
