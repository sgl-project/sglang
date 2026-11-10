// SPDX-License-Identifier: Apache-2.0
// Fused GDN projections and split layout for two/four/eight verification rows.
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>
namespace sglang::sm70_qwen_gdn {
template <int M, int NT, int L, int V>
__global__ void kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ tail,
    half* __restrict__ qkv,
    half* __restrict__ z,
    half* __restrict__ b,
    half* __restrict__ a) {
  constexpr int N = 4120, K = 2560;
  int row = blockIdx.x * (NT / L) + threadIdx.x / L, lane = threadIdx.x % L;
  float acc[M][V] = {};
  if (row < N) {
    const half* wr = row < 4096 ? w + row * K : tail + (row - 4096) * K;
    for (int k = lane * V; k < K; k += L * V) {
      int4 v = *reinterpret_cast<const int4*>(wr + k);
      const half* hw = reinterpret_cast<const half*>(&v);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        int4 xv = *reinterpret_cast<const int4*>(x + m * K + k);
        const half* hx = reinterpret_cast<const half*>(&xv);
#pragma unroll
        for (int i = 0; i < V; ++i)
          acc[m][i] = fmaf(__half2float(hx[i]), __half2float(hw[i]), acc[m][i]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float sum = 0;
#pragma unroll
    for (int i = 0; i < V; ++i)
      sum += acc[m][i];
#pragma unroll
    for (int d = L / 2; d > 0; d /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, d, L);
    if (row < N && lane == 0) {
      half value = __float2half_rn(sum);
      if (row < 2560)
        qkv[m * 2560 + row] = value;
      else if (row < 4096)
        z[m * 1536 + row - 2560] = value;
      else if (row < 4108)
        b[m * 12 + row - 4096] = value;
      else
        a[m * 12 + row - 4108] = value;
    }
  }
}
template <int M, int NT>
void run(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView t,
    tvm::ffi::TensorView qkv,
    tvm::ffi::TensorView z,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView a) {
  static_assert(M == 2 || M == 4 || M == 8);
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({M, 2560}).with_dtype<half>().with_device(dev).verify(x).verify(qkv);
  TensorMatcher({4096, 2560}).with_dtype<half>().with_device(dev).verify(w);
  TensorMatcher({24, 2560}).with_dtype<half>().with_device(dev).verify(t);
  TensorMatcher({M, 1536}).with_dtype<half>().with_device(dev).verify(z);
  TensorMatcher({M, 12}).with_dtype<half>().with_device(dev).verify(b).verify(a);
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
      "Projection operands must be 16-byte aligned");
  LaunchKernel((4120 + NT / 32 - 1) / (NT / 32), NT, dev.unwrap())(
      kernel<M, NT, 32, 8>,
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(w.data_ptr()),
      static_cast<const half*>(t.data_ptr()),
      static_cast<half*>(qkv.data_ptr()),
      static_cast<half*>(z.data_ptr()),
      static_cast<half*>(b.data_ptr()),
      static_cast<half*>(a.data_ptr()));
}
}  // namespace sglang::sm70_qwen_gdn
