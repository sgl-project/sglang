// SPDX-License-Identifier: Apache-2.0
// Attention output gate and sink, equal to the Triton _gate_sink_kernel in
// attention.py: y = x * sigmoid(gate) * sigmoid(lse - sink), with x and lse read
// from padded head groups when PADDED_GROUP > GROUP.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang::inkling2_gate_sink {

constexpr int kVec = 4;
using Bits = uint2;

__device__ __forceinline__ float sigmoid(float x) {
  float e, r;
  float const t = __fmul_rn(-x, 1.4426950408889634f);
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(__fadd_rn(e, 1.0f)));
  return r;
}

template <int H, int D, int GROUP, int PADDED_GROUP>
__global__ void __launch_bounds__(H* D / kVec) gate_sink_kernel(
    float const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ gate,
    float const* __restrict__ lse,
    __nv_bfloat16 const* __restrict__ sink,
    __nv_bfloat16* __restrict__ y,
    int64_t sx,
    int64_t sg,
    int64_t sl0,
    int64_t sl1,
    int64_t ss) {
  int const row = blockIdx.x;
  int const col = threadIdx.x * kVec;
  int const head = col / D;
  int const source_head = head + (head / GROUP) * (PADDED_GROUP - GROUP);
  // The gate (qkvr projection) and sinks are written several kernels earlier.
  Bits const g = *reinterpret_cast<Bits const*>(gate + row * sg + col);
  float const s = __bfloat162float(sink[row * ss + head]);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  float const* xr = x + row * sx + source_head * D + col % D;
  float const scale = sigmoid(__fsub_rn(lse[row * sl0 + source_head * sl1], s));
  __nv_bfloat16 const* gh = reinterpret_cast<__nv_bfloat16 const*>(&g);
  alignas(8) __nv_bfloat16 out[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    out[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(xr[i], sigmoid(__bfloat162float(gh[i]))), scale));
  *reinterpret_cast<Bits*>(y + row * H * D + col) = *reinterpret_cast<Bits*>(out);
}

template <int H, int D, int GROUP, int PADDED_GROUP>
void run(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView lse,
    tvm::ffi::TensorView sink,
    tvm::ffi::TensorView y) {
  static_assert(D % kVec == 0 && H * D / kVec <= 1024);
  host::LaunchKernel(y.size(0), H * D / kVec, y.device())
      .enable_pdl(true)(
          gate_sink_kernel<H, D, GROUP, PADDED_GROUP>,
          static_cast<float const*>(x.data_ptr()),
          static_cast<__nv_bfloat16 const*>(gate.data_ptr()),
          static_cast<float const*>(lse.data_ptr()),
          static_cast<__nv_bfloat16 const*>(sink.data_ptr()),
          static_cast<__nv_bfloat16*>(y.data_ptr()),
          x.stride(0),
          gate.stride(0),
          lse.stride(0),
          lse.stride(1),
          sink.stride(0));
}

}  // namespace sglang::inkling2_gate_sink
