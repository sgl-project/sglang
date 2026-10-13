#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

using namespace device;

// Single-token (M=1) per-tensor-scale FP8 GEMV for SM120 decode:
//   y[N] = (W_fp8[N,K] @ x_fp8[K]) * alpha,  alpha = scale_a * scale_b.
//
// cuBLAS serves these shapes with SM89 tiles and leaves 30-50% DRAM
// bandwidth on the table for mid-sized N (wave-quantization floor around
// 19us). Same design as the Hopper bf16 GEMV: one warp computes kRows
// consecutive rows, the fp8 activation vector is staged in shared memory,
// weights stream once with evict-first loads, and the reduction is a
// register + warp-shuffle tree.

constexpr uint32_t kFp8VecSize = 16;  // 16 fp8 values per 16B load

__device__ __forceinline__ float dot16_fp8_f32(const float4 wv, const float4 xv) {
  const __nv_fp8x2_e4m3* w2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wv);
  const __nv_fp8x2_e4m3* x2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 w01 = static_cast<float2>(w2[i]);
    const float2 x01 = static_cast<float2>(x2[i]);
    acc = fmaf(w01.x, x01.x, acc);
    acc = fmaf(w01.y, x01.y, acc);
  }
  return acc;
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps>
__global__ void __launch_bounds__(kNumWarps * 32) sm120_fp8_gemv_kernel(
    bf16_t* __restrict__ out,
    const uint8_t* __restrict__ x,
    const uint8_t* __restrict__ w,
    const float* __restrict__ alpha) {
  __shared__ uint8_t sx[K];

  const uint32_t tid = threadIdx.x;
  for (uint32_t i = tid * kFp8VecSize; i < K; i += kNumWarps * 32 * kFp8VecSize) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t r0 = (blockIdx.x * kNumWarps + warp) * kRows;
  if (r0 >= N) {
    return;
  }

  float acc[kRows];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
    acc[r] = 0.0f;
  }

  constexpr uint32_t kStep = 32 * kFp8VecSize * kUnroll;
  if (r0 + kRows <= N) {
    for (uint32_t k = lane * kFp8VecSize * kUnroll; k < K; k += kStep) {
      float4 xv[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        xv[u] = *reinterpret_cast<const float4*>(sx + k + u * kFp8VecSize);
      }
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint8_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
        float4 wv[kUnroll];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          wv[u] = __ldcs(reinterpret_cast<const float4*>(wr + u * kFp8VecSize));
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          acc[r] += dot16_fp8_f32(wv[u], xv[u]);
        }
      }
    }
  } else {
    for (uint32_t k = lane * kFp8VecSize * kUnroll; k < K; k += kStep) {
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        if (r0 + r < N) {
          const uint8_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
#pragma unroll
          for (uint32_t u = 0; u < kUnroll; ++u) {
            float4 wv = __ldcs(reinterpret_cast<const float4*>(wr + u * kFp8VecSize));
            float4 xv = *reinterpret_cast<const float4*>(sx + k + u * kFp8VecSize);
            acc[r] += dot16_fp8_f32(wv, xv);
          }
        }
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
      acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
    }
  }
  if (lane == 0) {
    const float a = *alpha;
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (r0 + r < N) {
        out[r0 + r] = cast<bf16_t>(acc[r] * a);
      }
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps>
struct Sm120Fp8GemvKernel {
  static_assert(K % (32 * kFp8VecSize * kUnroll) == 0, "K must cover full unrolled warp strides");
  static_assert(K <= 48 * 1024, "activation row must fit static shared memory");

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView alpha,
      const tvm::ffi::TensorView out) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(w);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(alpha);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    constexpr uint32_t kRowsPerBlock = kRows * kNumWarps;
    constexpr uint32_t kNumBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    LaunchKernel(kNumBlocks, kNumWarps * 32, device.unwrap())(
        sm120_fp8_gemv_kernel<N, K, kRows, kUnroll, kNumWarps>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const uint8_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const float*>(alpha.data_ptr()));
  }
};

}  // namespace sglang
