// SPDX-License-Identifier: Apache-2.0
// Reuse each vectorized weight load across two to four verification rows.
// With the same lanes per output, every row is bitwise the one-row
// sm70_dense_gemv result: same per-lane order, same shuffle reduction.
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>
namespace sglang::sm70_small_gemm {
template <int M, int NT, int L, int V>
__global__ void kernel(const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ o, int N, int K) {
  int row = blockIdx.x * (NT / L) + threadIdx.x / L, lane = threadIdx.x % L;
  float acc[M][V] = {};
  if (row < N) {
    for (int k = lane * V; k < K; k += L * V) {
      int4 v = *reinterpret_cast<const int4*>(w + row * K + k);
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
    if (row < N && lane == 0) o[m * N + row] = __float2half_rn(sum);
  }
}
template <int M, int NT, int L>
void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView o) {
  static_assert(M >= 2 && M <= 4);
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  auto N = SymbolicSize{}, K = SymbolicSize{};
  TensorMatcher({M, K}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({N, K}).with_dtype<half>().with_device(dev).verify(w);
  TensorMatcher({M, N}).with_dtype<half>().with_device(dev).verify(o);
  RuntimeCheck(K.unwrap() % 8 == 0, "K must be divisible by eight");
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
      "Input and weight must be 16-byte aligned");
  LaunchKernel((N.unwrap() + NT / L - 1) / (NT / L), NT, dev.unwrap())(
      kernel<M, NT, L, 8>,
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(w.data_ptr()),
      static_cast<half*>(o.data_ptr()),
      int(N.unwrap()),
      int(K.unwrap()));
}
}  // namespace sglang::sm70_small_gemm
