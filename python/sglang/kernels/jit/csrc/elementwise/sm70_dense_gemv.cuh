// SPDX-License-Identifier: Apache-2.0
// FP16 row-major GEMVs for Qwen3.8 on Volta. Accumulate in FP32.
#include <cuda_fp16.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

#include <sgl_kernel/utils.cuh>
namespace sglang::sm70_dense_gemv {
template <int THREADS, int LANES, int VEC, bool DUAL = false>
__global__ void gemv_kernel(const half* __restrict__ x,
                            const half* __restrict__ w, half* __restrict__ out,
                            int N, int K, const half* tail) {
  // A warp (or fixed subgroup) owns each output row; contiguous vector
  // loads amortize address/FP16-conversion work across independent
  // accumulators.
  int row = blockIdx.x * (THREADS / LANES) + threadIdx.x / LANES;
  int lane = threadIdx.x % LANES;
  float accum[VEC] = {};
  if (row < N) {
    const half* row_weight = w + row * K;
    if constexpr (DUAL) {
      if (row >= 4096) row_weight = tail + (row - 4096) * K;
    }
    for (int k = lane * VEC; k < K; k += LANES * VEC) {
      if constexpr (VEC == 8) {
        int4 xv = *reinterpret_cast<const int4*>(x + k);
        int4 wv = *reinterpret_cast<const int4*>(row_weight + k);
        const half* hx = reinterpret_cast<const half*>(&xv);
        const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
        for (int i = 0; i < VEC; ++i)
          accum[i] = fmaf(__half2float(hx[i]), __half2float(hw[i]), accum[i]);
      } else if constexpr (VEC == 4) {
        int2 xv = *reinterpret_cast<const int2*>(x + k);
        int2 wv = *reinterpret_cast<const int2*>(row_weight + k);
        const half* hx = reinterpret_cast<const half*>(&xv);
        const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
        for (int i = 0; i < VEC; ++i)
          accum[i] = fmaf(__half2float(hx[i]), __half2float(hw[i]), accum[i]);
      } else {
#pragma unroll
        for (int i = 0; i < VEC; ++i)
          accum[i] = fmaf(__half2float(x[k + i]),
                          __half2float(row_weight[k + i]), accum[i]);
      }
    }
  }
  float sum = 0;
#pragma unroll
  for (int i = 0; i < VEC; ++i) sum += accum[i];
#pragma unroll
  for (int off = LANES / 2; off > 0; off /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, off, LANES);
  if (lane == 0 && row < N) out[row] = __float2half_rn(sum);
}
template <int THREADS, int LANES, int VEC>
void gemv(tvm::ffi::TensorView x, tvm::ffi::TensorView w,
          tvm::ffi::TensorView out) {
  static_assert(LANES == 8 || LANES == 16 || LANES == 32);
  static_assert(VEC == 4 || VEC == 8);
  static_assert(THREADS % 32 == 0);
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto N = SymbolicSize{};
  auto K = SymbolicSize{};
  TensorMatcher({1, K}).with_dtype<half>().with_device(device).verify(x);
  TensorMatcher({N, K}).with_dtype<half>().with_device(device).verify(w);
  TensorMatcher({1, N}).with_dtype<half>().with_device(device).verify(out);
  RuntimeCheck(N.unwrap() > 0 && K.unwrap() > 0 && K.unwrap() % VEC == 0,
               "Invalid GEMV dimensions");
  RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
               "GEMV operands must be 16-byte aligned");
  LaunchKernel((N.unwrap() + THREADS / LANES - 1) / (THREADS / LANES), THREADS,
               device.unwrap())(
      gemv_kernel<THREADS, LANES, VEC>, static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(w.data_ptr()),
      static_cast<half*>(out.data_ptr()), int(N.unwrap()), int(K.unwrap()),
      static_cast<const half*>(nullptr));
}

inline void qkv_ba(tvm::ffi::TensorView x, tvm::ffi::TensorView weight,
                   tvm::ffi::TensorView tail, tvm::ffi::TensorView out) {
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({1, 2560}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({4096, 2560})
      .with_dtype<half>()
      .with_device(dev)
      .verify(weight);
  TensorMatcher({24, 2560}).with_dtype<half>().with_device(dev).verify(tail);
  TensorMatcher({1, 4120}).with_dtype<half>().with_device(dev).verify(out);
  RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(tail.data_ptr()) % 16 == 0,
               "GEMV operands must be 16-byte aligned");
  LaunchKernel(1030, 64, dev.unwrap())(
      gemv_kernel<64, 16, 8, true>, static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(weight.data_ptr()),
      static_cast<half*>(out.data_ptr()), 4120, 2560,
      static_cast<const half*>(tail.data_ptr()));
}
}  // namespace sglang::sm70_dense_gemv
