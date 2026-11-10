// SPDX-License-Identifier: Apache-2.0
// Batched HC mix; preserve the FP16 projection boundaries.
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>
namespace sglang::sm70_hc_batch {
template <int M, int NT, bool Inject = false>
__global__ void down(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ o,
    const half* __restrict__ inject = nullptr,
    float* __restrict__ partials = nullptr) {
  int row = blockIdx.x;
  if constexpr (Inject) {
    if (row >= 320) {
      int branch = row - 320, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
      float sum[M] = {};
#pragma unroll
      for (int j = 0; j < 5; ++j) {
        int k = (threadIdx.x + j * 256) * 8;
        int4 wv = *reinterpret_cast<const int4*>(inject + branch * 10240 + k);
        const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
        for (int m = 0; m < M; ++m) {
          int4 xv = *reinterpret_cast<const int4*>(x + m * 10240 + k);
          const half* hx = reinterpret_cast<const half*>(&xv);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            sum[m] += __half2float(hx[2 * i]) * __half2float(hw[2 * i]) +
                      __half2float(hx[2 * i + 1]) * __half2float(hw[2 * i + 1]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int d = 16; d > 0; d /= 2)
          sum[m] += __shfl_xor_sync(0xffffffff, sum[m], d);
        if (lane == 0) partials[(m * 8 + warp) * 4 + branch] = sum[m];
      }
      return;
    }
  }
  float acc[M][8] = {};
#pragma unroll
  for (int j = 0; j < (10240 + NT * 8 - 1) / (NT * 8); ++j) {
    int k = (threadIdx.x + j * NT) * 8;
    if (k >= 10240) continue;
    int4 wv = *reinterpret_cast<const int4*>(w + row * 10240 + k);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      int4 xv = *reinterpret_cast<const int4*>(x + m * 10240 + k);
      const half* hx = reinterpret_cast<const half*>(&xv);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[m][i] += __half2float(hx[i]) * __half2float(hw[i]);
    }
  }
  __shared__ float part[M][NT / 32];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sum += acc[m][i];
#pragma unroll
    for (int d = 16; d > 0; d /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, d);
    if (threadIdx.x % 32 == 0) part[m][threadIdx.x / 32] = sum;
  }
  __syncthreads();
  if (threadIdx.x < M) {
    int m = threadIdx.x;
    float sum = 0;
#pragma unroll
    for (int i = 0; i < NT / 32; ++i)
      sum += part[m][i];
    sum = __half2float(__float2half_rn(sum)) * .25f;
    o[m * 320 + row] = __float2half_rn(sum / (1.f + __expf(-sum)));
  }
}
template <int M, int NT>
__global__ void
up(const half* __restrict__ a, const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ o) {
  int lane = threadIdx.x % 32, sub = lane % 8, branch = lane / 8, h = blockIdx.x * (NT / 32) + threadIdx.x / 32,
      row = branch * 2560 + h;
  float acc[M][8] = {};
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    int k = (sub + j * 8) * 8;
    int4 wv = *reinterpret_cast<const int4*>(w + row * 320 + k);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      int4 av = *reinterpret_cast<const int4*>(a + m * 320 + k);
      const half* ha = reinterpret_cast<const half*>(&av);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[m][i] += __half2float(ha[i]) * __half2float(hw[i]);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v += acc[m][i];
#pragma unroll
    for (int d = 4; d > 0; d /= 2)
      v += __shfl_down_sync(0xffffffff, v, d, 8);
    v = __half2float(__float2half_rn(v));
    v = __half2float(x[m * 10240 + row]) / (1.f + __expf(-v));
    float b1 = __shfl_sync(0xffffffff, v, 8), b2 = __shfl_sync(0xffffffff, v, 16), b3 = __shfl_sync(0xffffffff, v, 24);
    if (lane == 0) o[m * 2560 + h] = __float2half_rn((((v + b1) + b2) + b3) * .25f);
  }
}
template <int M>
void down_run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView o) {
  static_assert(M == 2 || M == 4);
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({M, 10240}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({320, 10240}).with_dtype<half>().with_device(dev).verify(w);
  TensorMatcher({M, 320}).with_dtype<half>().with_device(dev).verify(o);
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
      "Input and weight must be 16-byte aligned");
  LaunchKernel(320, 256, dev.unwrap())(
      down<M, 256>,
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(w.data_ptr()),
      static_cast<half*>(o.data_ptr()),
      static_cast<const half*>(nullptr),
      static_cast<float*>(nullptr));
}
template <int M>
void up_run(tvm::ffi::TensorView a, tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView o) {
  static_assert(M == 2 || M == 4);
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({M, 320}).with_dtype<half>().with_device(dev).verify(a);
  TensorMatcher({M, 10240}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({10240, 320}).with_dtype<half>().with_device(dev).verify(w);
  TensorMatcher({M, 2560}).with_dtype<half>().with_device(dev).verify(o);
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
      "Activation and weight must be 16-byte aligned");
  LaunchKernel(640, 128, dev.unwrap())(
      up<M, 128>,
      static_cast<const half*>(a.data_ptr()),
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(w.data_ptr()),
      static_cast<half*>(o.data_ptr()));
}
}  // namespace sglang::sm70_hc_batch
