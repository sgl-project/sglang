// SPDX-License-Identifier: Apache-2.0
// Small-batch Qwen3.8 FP16 projection/activation fusions on Volta.
#include <cuda_fp16.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

#include <sgl_kernel/utils.cuh>
namespace sglang::sm70_qwen_fusions {
template <int THREADS>
__global__ void kernel(const half* __restrict__ x,
                       const half* __restrict__ weight,
                       const half* __restrict__ value, half* __restrict__ out) {
  x += blockIdx.x * 2560;
  value += blockIdx.x * 2560;
  out += blockIdx.x * 2560;
  float a[8] = {};
  for (int k = threadIdx.x * 8; k < 2560; k += THREADS * 8) {
    int4 xv = *reinterpret_cast<const int4*>(x + k),
         wv = *reinterpret_cast<const int4*>(weight + k);
    const half* hx = reinterpret_cast<const half*>(&xv);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      a[i] = fmaf(__half2float(hx[i]), __half2float(hw[i]), a[i]);
  }
  float sum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += a[i];
#pragma unroll
  for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
  __shared__ float parts[THREADS / 32];
  __shared__ float gate;
  if (threadIdx.x % 32 == 0) parts[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    sum = 0;
#pragma unroll
    for (int i = 0; i < THREADS / 32; ++i) sum += parts[i];
    sum = __half2float(__float2half_rn(sum));
    gate = __half2float(__float2half_rn(1.f / (1.f + __expf(-sum))));
  }
  __syncthreads();
  for (int n = threadIdx.x; n < 2560; n += THREADS)
    out[n] = __float2half_rn(__half2float(value[n]) * gate);
}
template <int THREADS>
void run(tvm::ffi::TensorView x, tvm::ffi::TensorView weight,
         tvm::ffi::TensorView value, tvm::ffi::TensorView out) {
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  auto rows = SymbolicSize{};
  TensorMatcher({rows, 2560})
      .with_dtype<half>()
      .with_device(dev)
      .verify(x)
      .verify(value)
      .verify(out);
  TensorMatcher({1, 2560}).with_dtype<half>().with_device(dev).verify(weight);
  RuntimeCheck(rows.unwrap() == 1 || rows.unwrap() == 2 ||
                   (rows.unwrap() >= 4 && rows.unwrap() <= 32 && rows.unwrap() % 4 == 0),
               "Shared gate supports one, two, or multiples of four rows up to 32");
  RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
               "Operands must be 16-byte aligned");
  LaunchKernel(rows.unwrap(), THREADS, dev.unwrap())(
      kernel<THREADS>, static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(weight.data_ptr()),
      static_cast<const half*>(value.data_ptr()),
      static_cast<half*>(out.data_ptr()));
}
}  // namespace sglang::sm70_qwen_fusions
namespace sglang::sm70_qwen_fusions {
__global__ void gate_up_kernel(const half* __restrict__ x,
                               const half* __restrict__ w,
                               half* __restrict__ out) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = blockIdx.x * 4 + warp / 2;
  const int row = n + (warp % 2) * 160;
  float a[8] = {};
#pragma unroll
  for (int j = 0; j < 10; ++j) {
    const int k = lane * 8 + j * 256;
    const int4 xv = *reinterpret_cast<const int4*>(x + k),
               wv = *reinterpret_cast<const int4*>(w + row * 2560 + k);
    const half* hx = reinterpret_cast<const half*>(&xv);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      a[i] = fmaf(__half2float(hx[i]), __half2float(hw[i]), a[i]);
  }
  float sum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += a[i];
#pragma unroll
  for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
  __shared__ float vals[8];
  if (lane == 0) vals[warp] = __half2float(__float2half_rn(sum));
  __syncthreads();
  if (lane == 0 && warp % 2 == 0) {
    float g = vals[warp], u = vals[warp + 1];
    out[n] = __float2half_rn(g / (1.f + __expf(-g)) * u);
  }
}
void gate_up(tvm::ffi::TensorView x, tvm::ffi::TensorView weight,
             tvm::ffi::TensorView out) {
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({1, 2560}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({320, 2560}).with_dtype<half>().with_device(dev).verify(weight);
  TensorMatcher({1, 160}).with_dtype<half>().with_device(dev).verify(out);
  RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
               "Operands must be 16-byte aligned");
  LaunchKernel(40, 256, dev.unwrap())(
      gate_up_kernel, static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(weight.data_ptr()),
      static_cast<half*>(out.data_ptr()));
}
}  // namespace sglang::sm70_qwen_fusions
