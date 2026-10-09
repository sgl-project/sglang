// SPDX-License-Identifier: Apache-2.0
#pragma once
#define SGLANG_MXFP8_GEMV_BODY_ONLY
#include "mxfp8_gemv_gfx95.cuh"
#undef SGLANG_MXFP8_GEMV_BODY_ONLY

namespace sglang {
namespace shared_router_gfx950 {
using u16x8 = uint16_t __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));

// Independent workgroups share a launch, not intermediate data. The first
// 72 groups run native MXFP8 shared gate/up; the remaining 30 groups produce
// ten FP32 split-K partials of the BF16 router projection.
template <int STEPS>
__global__ void __launch_bounds__(512) shared_router_projection(
    const uint8_t* SW, const uint8_t* SS, const uint16_t* X, bf16_t* Gate, const uint16_t* RW, float* Part, int M) {
#if defined(__gfx950__)
  if (blockIdx.x < 72) {
    mxfp8_gemv::mxfp8_gemv_body<8, STEPS, 16, 16, true>(
        SW, SS, reinterpret_cast<const uint8_t*>(X), nullptr, Gate, M, 1152, 5120);
    return;
  }
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  const int tile = (blockIdx.x - 72) * 8 + wave;
  const int column = tile % 24, split = tile / 24;
  const int row = lane % 16, group = lane / 16;
  f32x4 acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int step = 0; step < 16; ++step) {
    const int k = split * 512 + step * 32 + group * 8;
    const u16x8 w = *reinterpret_cast<const u16x8*>(RW + (column * 16 + row) * 5120 + k);
    u16x8 x = {};
    if (row < M) x = *reinterpret_cast<const u16x8*>(X + row * 5120 + k);
    acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(w, x, acc, 0, 0, 0);
  }
  if (row < M) {
#pragma unroll
    for (int r = 0; r < 4; ++r)
      Part[split * M * 384 + row * 384 + column * 16 + group * 4 + r] = acc[r];
  }
#elif defined(__HIP_DEVICE_COMPILE__)
#error "shared_router_gfx950 requires gfx950"
#endif
}
}  // namespace shared_router_gfx950

struct SharedRouterGfx950Kernel {
  static void
  run(tvm::ffi::TensorView sw,
      tvm::ffi::TensorView ss,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView rw,
      tvm::ffi::TensorView part) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto m = SymbolicSize{"M"};
    TensorMatcher({72, 40, 2048}).with_dtype<uint8_t>().with_device(device).verify(sw);
    TensorMatcher({36, 160}).with_dtype<uint8_t>().with_device(device).verify(ss);
    TensorMatcher({m, 5120}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({m, 1152}).with_dtype<bf16_t>().with_device(device).verify(gate);
    TensorMatcher({384, 5120}).with_dtype<bf16_t>().with_device(device).verify(rw);
    TensorMatcher({10, m, 384}).with_dtype<float>().with_device(device).verify(part);
    const int rows = static_cast<int>(m.unwrap());
    RuntimeCheck(rows == 6 || rows == 12, "Shared/router fusion supports M6/M12");
    auto kernel = rows == 6 ? shared_router_gfx950::shared_router_projection<1>
                            : shared_router_gfx950::shared_router_projection<2>;
    LaunchKernel(dim3(102), dim3(512), device.unwrap())(
        kernel,
        static_cast<const uint8_t*>(sw.data_ptr()),
        static_cast<const uint8_t*>(ss.data_ptr()),
        static_cast<const uint16_t*>(x.data_ptr()),
        static_cast<bf16_t*>(gate.data_ptr()),
        static_cast<const uint16_t*>(rw.data_ptr()),
        static_cast<float*>(part.data_ptr()),
        rows);
  }
};
}  // namespace sglang
