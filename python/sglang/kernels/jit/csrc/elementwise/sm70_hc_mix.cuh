// SPDX-License-Identifier: Apache-2.0
// Qwen3.8 Flash Next's FP16 hyperconnection projections on Volta.
#include <cuda_fp16.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

#include <sgl_kernel/utils.cuh>

namespace sglang::sm70_hc {

// A CTA owns one low-rank output. Keep independent FP32 accumulators until
// the final reduction, and fetch eight contiguous halves per memory access.
__global__ __launch_bounds__(256) void down_kernel(
    const half* __restrict__ x, const half* __restrict__ weight,
    half* __restrict__ output) {
  const int row = blockIdx.x;
  float accum[8] = {};
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const int offset = (threadIdx.x + j * 256) * 8;
    const int4 xv = *reinterpret_cast<const int4*>(x + offset);
    const int4 wv =
        *reinterpret_cast<const int4*>(weight + row * 10240 + offset);
    const half* hx = reinterpret_cast<const half*>(&xv);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[i] += __half2float(hx[i]) * __half2float(hw[i]);
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += accum[i];
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, delta);
  }
  __shared__ float partials[8];
  if ((threadIdx.x & 31) == 0) partials[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum += partials[i];
    // Preserve F.linear's FP16 output boundary before division and SiLU.
    sum = __half2float(__float2half_rn(sum)) * 0.25f;
    output[row] = __float2half_rn(sum / (1.0f + __expf(-sum)));
  }
}

// One warp owns a hidden coordinate. Its four eight-lane groups compute
// the four HC branches concurrently; eight warps share each CTA. This avoids
// the one-warp-per-CTA residency limit and four serial branch reductions.
__global__ __launch_bounds__(256) void up_kernel(
    const half* __restrict__ activated, const half* __restrict__ x,
    const half* __restrict__ weight, half* __restrict__ output) {
  const int lane = threadIdx.x & 31;
  const int sublane = lane & 7;
  const int branch = lane >> 3;
  const int hidden = blockIdx.x * 8 + threadIdx.x / 32;
  const int row = branch * 2560 + hidden;
  float accum[8] = {};
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const int k = (sublane + j * 8) * 8;
    const int4 av = *reinterpret_cast<const int4*>(activated + k);
    const int4 wv = *reinterpret_cast<const int4*>(weight + row * 320 + k);
    const half* ha = reinterpret_cast<const half*>(&av);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[i] += __half2float(ha[i]) * __half2float(hw[i]);
    }
  }
  float value = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) value += accum[i];
#pragma unroll
  for (int delta = 4; delta > 0; delta /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, delta, 8);
  }
  value = __half2float(__float2half_rn(value));
  value = __half2float(x[row]) / (1.0f + __expf(-value));
  const float branch1 = __shfl_sync(0xffffffffu, value, 8);
  const float branch2 = __shfl_sync(0xffffffffu, value, 16);
  const float branch3 = __shfl_sync(0xffffffffu, value, 24);
  if (lane == 0) {
    output[hidden] =
        __float2half_rn((((value + branch1) + branch2) + branch3) * 0.25f);
  }
}

inline void down(tvm::ffi::TensorView x, tvm::ffi::TensorView weight,
                 tvm::ffi::TensorView output) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({1, 10240}).with_dtype<half>().with_device(device).verify(x);
  TensorMatcher({320, 10240})
      .with_dtype<half>()
      .with_device(device)
      .verify(weight);
  TensorMatcher({1, 320}).with_dtype<half>().with_device(device).verify(output);
  RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
               "HC input must be 16-byte aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
               "HC weight must be 16-byte aligned");
  LaunchKernel(320, 256, device.unwrap())(
      down_kernel, static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(weight.data_ptr()),
      static_cast<half*>(output.data_ptr()));
}

inline void up(tvm::ffi::TensorView activated, tvm::ffi::TensorView x,
               tvm::ffi::TensorView weight, tvm::ffi::TensorView output) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({1, 320}).with_dtype<half>().with_device(device).verify(
      activated);
  TensorMatcher({1, 10240}).with_dtype<half>().with_device(device).verify(x);
  TensorMatcher({10240, 320})
      .with_dtype<half>()
      .with_device(device)
      .verify(weight);
  TensorMatcher({1, 2560}).with_dtype<half>().with_device(device).verify(
      output);
  RuntimeCheck(reinterpret_cast<uintptr_t>(activated.data_ptr()) % 16 == 0,
               "HC activation must be 16-byte aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
               "HC weight must be 16-byte aligned");
  LaunchKernel(320, 256, device.unwrap())(
      up_kernel, static_cast<const half*>(activated.data_ptr()),
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(weight.data_ptr()),
      static_cast<half*>(output.data_ptr()));
}

}  // namespace sglang::sm70_hc
