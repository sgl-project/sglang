// SPDX-License-Identifier: Apache-2.0
// SM70 decode GEMV for dense MXFP8 e4m3fn + UE8M0 g32 (M<=4).
// marlin_v100 FP8 W8A16 instantiates group_size {-1, 128} only, so official
// DSV4.1-Flash g32 stays packed in HBM. Dequant uses fp32 UE8M0 (exp<<23)
// so bytes 109-112 survive; do not store those scales as fp16.
#include "sm70_dsv41_fp4.cuh"

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_dsv41_mxfp8 {

inline constexpr int32_t kGroupSize = 32;
inline constexpr int32_t kBlockThreads = 256;
inline constexpr int32_t kRowsPerBlock = kBlockThreads / 32;  // 8 warps
inline constexpr int32_t kLaneE4 = 8;  // 8 e4m3 + 8 fp16; consecutive lanes coalesce
inline constexpr int32_t kWarpK = 32 * kLaneE4;

/**
 * \brief E4M3FN -> fp32 via IEEE bits. Bias 7 -> 127 is +120; denorm is man*2^-9.
 *
 * Matches torch.float8_e4m3fn for finite values (official MXFP8 weights).
 */
SGL_DEVICE float e4m3fn_fast(uint8_t b) {
  const uint32_t u = static_cast<uint32_t>(b);
  const uint32_t exp = (u >> 3) & 0x0Fu;
  const uint32_t man = u & 0x07u;
  const uint32_t sign = (u & 0x80u) << 24;
  if (exp == 0u) {
    return copysignf(static_cast<float>(man) * 0.001953125f, __uint_as_float(sign));
  }
  return __uint_as_float(sign | ((exp + 120u) << 23) | (man << 20));
}

/**
 * \brief One output row per warp. Consecutive lanes take consecutive 8-wide
 * K chunks so fp16 x loads are 16-byte coalesced. Reuse dequant across M.
 *
 * \tparam M Batched GEMV rows; 1, 2, or 4.
 */
template <int M>
__global__ __launch_bounds__(kBlockThreads) void mxfp8_gemv_kernel(
    fp16_t* __restrict__ y,
    const fp16_t* __restrict__ x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ scales,
    int32_t N,
    int32_t K) {
  static_assert(M == 1 || M == 2 || M == 4);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid / 32;
  const int32_t lane = tid % 32;
  const int32_t row = static_cast<int32_t>(blockIdx.x) * kRowsPerBlock + warp;
  if (row >= N) {
    return;
  }
  const uint8_t* __restrict__ w_row = w + static_cast<int64_t>(row) * K;
  const uint8_t* __restrict__ s_row = scales + static_cast<int64_t>(row) * (K / kGroupSize);
  float acc[M] = {};
  for (int32_t k = lane * kLaneE4; k < K; k += kWarpK) {
    const uint64_t packed = *reinterpret_cast<const uint64_t*>(w_row + k);
    const uint8_t* wb = reinterpret_cast<const uint8_t*>(&packed);
    const float s = sm70_dsv41::ue8m0_to_float(s_row[k / kGroupSize]);
#pragma unroll
    for (int32_t m = 0; m < M; ++m) {
      const int4 xv = *reinterpret_cast<const int4*>(x + static_cast<int64_t>(m) * K + k);
      const fp16_t* hx = reinterpret_cast<const fp16_t*>(&xv);
#pragma unroll
      for (int32_t i = 0; i < kLaneE4; ++i) {
        acc[m] = fmaf(__half2float(hx[i]), e4m3fn_fast(wb[i]) * s, acc[m]);
      }
    }
  }
#pragma unroll
  for (int32_t m = 0; m < M; ++m) {
    float sum = acc[m];
#pragma unroll
    for (int32_t d = 16; d > 0; d >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, d);
    }
    if (lane == 0) {
      y[static_cast<int64_t>(m) * N + row] = __float2half_rn(sum);
    }
  }
}

/**
 * \brief y[M,N] = x[M,K] @ e4m3[N,K].T * UE8M0[N, K/32] for M in {1,2,4}.
 */
inline void linear(
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView scales) {
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  auto M = SymbolicSize{"M"};
  auto N = SymbolicSize{"N"};
  auto K = SymbolicSize{"K"};
  auto G = SymbolicSize{"G"};
  TensorMatcher({M, K}).with_dtype<fp16_t>().with_device(dev).verify(x);
  TensorMatcher({N, K}).with_dtype<uint8_t, fp8_e4m3_t>().with_device(dev).verify(w);
  TensorMatcher({N, G}).with_dtype<uint8_t>().with_device(dev).verify(scales);
  TensorMatcher({M, N}).with_dtype<fp16_t>().with_device(dev).verify(y);
  const int64_t m = M.unwrap();
  const int64_t n = N.unwrap();
  const int64_t k = K.unwrap();
  CHECK_HOST(m == 1 || m == 2 || m == 4) << "MXFP8 GEMV M must be 1, 2, or 4; got " << m;
  CHECK_HOST(k % kGroupSize == 0) << "MXFP8 K must be a multiple of 32; got " << k;
  CHECK_HOST(G.unwrap() == k / kGroupSize) << "MXFP8 scales must be [N, K/32]; got G=" << G.unwrap();
  CHECK_HOST(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16u == 0u &&
      reinterpret_cast<uintptr_t>(w.data_ptr()) % 16u == 0u)
      << "x and weight must be 16-byte aligned";
  if (n == 0 || k == 0) {
    return;
  }
  const DLDevice device = dev.unwrap();
  const int32_t n32 = static_cast<int32_t>(n);
  const int32_t k32 = static_cast<int32_t>(k);
  const uint32_t grid = static_cast<uint32_t>(div_ceil(n, static_cast<int64_t>(kRowsPerBlock)));
  auto launch = [&](auto kernel) {
    LaunchKernel(grid, kBlockThreads, device)(
        kernel,
        static_cast<fp16_t*>(y.data_ptr()),
        static_cast<const fp16_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()),
        n32,
        k32);
  };
  if (m == 1) {
    launch(mxfp8_gemv_kernel<1>);
  } else if (m == 2) {
    launch(mxfp8_gemv_kernel<2>);
  } else {
    launch(mxfp8_gemv_kernel<4>);
  }
}

}  // namespace sglang::sm70_dsv41_mxfp8
