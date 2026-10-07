// SPDX-License-Identifier: Apache-2.0
// Kimi-K3 MLA output gate fused with per-token FP8 quant.
//
// Split out of mla_output_gate.cuh so the plain bf16 gate kernel -- the only
// variant CUDA builds -- keeps its JIT module and does not recompile for a
// consumer that only the ROCm PTPC o_proj path takes.
#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/cta.cuh>    // For cta::reduce_max
#include <sgl_kernel/math.cuh>   // For math::abs, math::max, FP8_E4M3_MAX
#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>  // For pack_fp8

namespace sglang {

// Fuse the gate multiply with per-token FP8 quant so MLA o_proj can consume
// (fp8, scale) without a second launch. One block per token; hidden must be
// divisible by the vector width. Matches torch.sigmoid+mul bf16 rounding, then
// the Triton per-token-group (group=H) scale: max(absmax, 1e-10) / fp8_max.
struct MlaOutputGateFp8QuantParams {
  const bf16_t* __restrict__ x;     // [T, H]
  const bf16_t* __restrict__ gate;  // [T, H]
  fp8_e4m3_t* __restrict__ out_q;   // [T, H]
  float* __restrict__ out_s;        // [T]
  uint32_t hidden;
};

template <int kThreads, bool kUsePDL>
__global__ void mla_output_gate_fp8_quant_kernel(const MlaOutputGateFp8QuantParams __grid_constant__ params) {
  using namespace device;
  using deepseek_v4::fp8::pack_fp8;
  constexpr int kVecN = 8;
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;
  using vec_fp8x2_t = AlignedVector<fp8x2_e4m3_t, kVecN / 2>;

  const uint32_t token = blockIdx.x;
  const uint32_t num_vecs = params.hidden / kVecN;
  const bf16_t* x_row = params.x + static_cast<uint64_t>(token) * params.hidden;
  const bf16_t* g_row = params.gate + static_cast<uint64_t>(token) * params.hidden;
  fp8_e4m3_t* q_row = params.out_q + static_cast<uint64_t>(token) * params.hidden;

  PDLWaitPrimary<kUsePDL>();

  float max_value = 0.0f;
  for (uint32_t i = threadIdx.x; i < num_vecs; i += kThreads) {
    vec_bf16_t xv, gv;
    xv.load(x_row, i);
    gv.load(g_row, i);
#pragma unroll
    for (int j = 0; j < kVecN; ++j) {
      const float g = cast<fp32_t>(gv[j]);
      const bf16_t s = cast<bf16_t>(1.0f / (1.0f + expf(-g)));
      const float y = cast<fp32_t>(cast<bf16_t>(cast<fp32_t>(xv[j]) * cast<fp32_t>(s)));
      max_value = math::max(max_value, math::abs(y));
    }
  }

  __shared__ float reduction_smem[32];
  __shared__ float scale_smem;
  cta::reduce_max(max_value, reduction_smem);
  __syncthreads();
  if (threadIdx.x == 0) {
    const float absmax = math::max(reduction_smem[0], 1.0e-10f);
    scale_smem = absmax / math::FP8_E4M3_MAX;
    params.out_s[token] = scale_smem;
  }
  __syncthreads();
  const float scale_inv = 1.0f / scale_smem;

  for (uint32_t i = threadIdx.x; i < num_vecs; i += kThreads) {
    vec_bf16_t xv, gv;
    vec_fp8x2_t qv;
    xv.load(x_row, i);
    gv.load(g_row, i);
#pragma unroll
    for (int j = 0; j < kVecN / 2; ++j) {
      const int j0 = 2 * j;
      const int j1 = j0 + 1;
      const float g0 = cast<fp32_t>(gv[j0]);
      const float g1 = cast<fp32_t>(gv[j1]);
      const bf16_t s0 = cast<bf16_t>(1.0f / (1.0f + expf(-g0)));
      const bf16_t s1 = cast<bf16_t>(1.0f / (1.0f + expf(-g1)));
      const float y0 = cast<fp32_t>(cast<bf16_t>(cast<fp32_t>(xv[j0]) * cast<fp32_t>(s0)));
      const float y1 = cast<fp32_t>(cast<bf16_t>(cast<fp32_t>(xv[j1]) * cast<fp32_t>(s1)));
      qv[j] = pack_fp8(y0 * scale_inv, y1 * scale_inv);
    }
    qv.store(reinterpret_cast<fp8x2_e4m3_t*>(q_row + i * kVecN), 0);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
struct MlaOutputGateFp8QuantKernel {
  static constexpr auto kernel = mla_output_gate_fp8_quant_kernel<kThreads, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s) {
    using namespace host;

    auto T_ = SymbolicSize{"tokens"};
    auto H_ = SymbolicSize{"hidden"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(gate);
    TensorMatcher({T_, H_}).with_dtype<fp8_e4m3_t>().with_device(device).verify(out_q);
    TensorMatcher({T_}).with_dtype<float>().with_device(device).verify(out_s);

    const auto T = static_cast<uint32_t>(T_.unwrap());
    const auto H = static_cast<uint32_t>(H_.unwrap());
    RuntimeCheck(H % 8 == 0, "hidden must be divisible by 8");
    if (T == 0 || H == 0) return;

    const auto params = MlaOutputGateFp8QuantParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .gate = static_cast<const bf16_t*>(gate.data_ptr()),
        .out_q = static_cast<fp8_e4m3_t*>(out_q.data_ptr()),
        .out_s = static_cast<float*>(out_s.data_ptr()),
        .hidden = H,
    };
    LaunchKernel(T, kThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
