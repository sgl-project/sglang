// SPDX-License-Identifier: Apache-2.0
// Batched GEMV for one to four rows: out[b, m, n] = sum_k x[b, m, k] * w[b, n, k].
// L lanes per output split K in a fixed pattern and reduce in a fixed order, so
// row m does not depend on how many rows share the launch: decode (one row) and
// MTP verify (up to four) give bitwise the same output for the same input row.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace sglang::sm70_rows_gemv {

inline constexpr int kVec = 8;

SGL_DEVICE void load8(const __half* p, float (&f)[kVec]) {
  const int4 v = *reinterpret_cast<const int4*>(p);
  const __half* h = reinterpret_cast<const __half*>(&v);
#pragma unroll
  for (int i = 0; i < kVec; ++i) f[i] = __half2float(h[i]);
}

SGL_DEVICE void load8(const float* p, float (&f)[kVec]) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w;
  f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

SGL_DEVICE void store(__half* p, float v) {
  *p = __float2half_rn(v);
}
SGL_DEVICE void store(float* p, float v) {
  *p = v;
}

// Strides are in elements; the last dim of x, w and out is contiguous.
struct Params {
  const __half* x;
  const void* w;
  void* out;
  int64_t x_b, x_m, w_b, w_n, o_b, o_m;
  int N, K;
};

template <int M, int NT, int L, typename W>
__global__ void __launch_bounds__(NT) kernel(const Params p) {
  static_assert(L == 8 || L == 16 || L == 32 || L == 64 || L == 128 || L == 256);
  static_assert(NT % L == 0 && NT % 32 == 0);
  constexpr int kWarpsPerRow = L > 32 ? L / 32 : 1;
  __shared__ float partial[M][NT / 32];
  const int b = static_cast<int>(blockIdx.y);
  const int r = static_cast<int>(threadIdx.x) / L;
  const int lane = static_cast<int>(threadIdx.x) % L;
  const int n = static_cast<int>(blockIdx.x) * (NT / L) + r;
  float acc[M][kVec] = {};
  if (n < p.N) {
    const W* wr = static_cast<const W*>(p.w) + b * p.w_b + n * p.w_n;
    const __half* xb = p.x + b * p.x_b;
    for (int k = lane * kVec; k < p.K; k += L * kVec) {
      float wf[kVec];
      load8(wr + k, wf);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[kVec];
        load8(xb + m * p.x_m + k, xf);
#pragma unroll
        for (int i = 0; i < kVec; ++i) acc[m][i] = fmaf(xf[i], wf[i], acc[m][i]);
      }
    }
  }
  float sum[M];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float s = acc[m][0];
#pragma unroll
    for (int i = 1; i < kVec; ++i) s = __fadd_rn(s, acc[m][i]);
#pragma unroll
    for (int d = (L < 32 ? L : 32) / 2; d > 0; d >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, d));
    sum[m] = s;
  }
  if constexpr (kWarpsPerRow > 1) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) partial[m][warp] = sum[m];
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float s = partial[m][r * kWarpsPerRow];
#pragma unroll
      for (int w = 1; w < kWarpsPerRow; ++w) s = __fadd_rn(s, partial[m][r * kWarpsPerRow + w]);
      sum[m] = s;
    }
  }
  if (n < p.N && lane == 0) {
    using O = W;
    O* out = static_cast<O*>(p.out) + b * p.o_b + n;
#pragma unroll
    for (int m = 0; m < M; ++m) store(out + m * p.o_m, sum[m]);
  }
}

// x [B, M, K] fp16; w [B, N, K] fp16 or fp32; out [B, M, N] in w's dtype.
template <int M, int NT, int L>
void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView out) {
  using namespace host;
  static_assert(M >= 1 && M <= 4);
  SymbolicSize B = {"batch"}, N = {"N"}, K = {"K"};
  SymbolicDType wdtype;
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({B, M, K}).with_dtype<fp16_t>().with_strides({-1, -1, 1}).with_device(device_).verify(x);
  TensorMatcher({B, N, K}).with_dtype<fp16_t, fp32_t>(wdtype).with_strides({-1, -1, 1}).with_device(device_).verify(w);
  TensorMatcher({B, M, N}).with_dtype<fp16_t, fp32_t>(wdtype).with_strides({-1, -1, 1}).with_device(device_).verify(out);
  const bool w32 = wdtype.is_type<fp32_t>();
  const int64_t align = w32 ? 4 : 8;
  RuntimeCheck(K.unwrap() % kVec == 0, "K must be divisible by eight");
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
          x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 && w.stride(0) % align == 0 && w.stride(1) % align == 0,
      "x and w rows must be 16-byte aligned");
  if (B.unwrap() == 0 || N.unwrap() == 0) {
    return;
  }
  const Params p{
      .x = static_cast<const __half*>(x.data_ptr()),
      .w = w.data_ptr(),
      .out = out.data_ptr(),
      .x_b = x.stride(0),
      .x_m = x.stride(1),
      .w_b = w.stride(0),
      .w_n = w.stride(1),
      .o_b = out.stride(0),
      .o_m = out.stride(1),
      .N = static_cast<int>(N.unwrap()),
      .K = static_cast<int>(K.unwrap()),
  };
  const dim3 grid((N.unwrap() + NT / L - 1) / (NT / L), B.unwrap());
  if (w32) {
    LaunchKernel(grid, NT, device_.unwrap())(kernel<M, NT, L, float>, p);
  } else {
    LaunchKernel(grid, NT, device_.unwrap())(kernel<M, NT, L, __half>, p);
  }
}

}  // namespace sglang::sm70_rows_gemv
