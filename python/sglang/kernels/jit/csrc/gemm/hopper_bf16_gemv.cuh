#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

// Single-token (M=1) bf16 GEMV tuned for Hopper decode: y[N] = W[N,K] @ x[K].
//
// Layout: one warp computes kRows consecutive output rows; the activation
// vector is staged once in static shared memory and reused by every warp;
// weights are streamed with evict-first loads (read exactly once). All
// reductions happen in registers + one warp shuffle tree, so there is no
// split-K fixup kernel. Weight traffic dominates (N*K*2 bytes), so the
// design goal is simply maximum sustained DRAM read bandwidth.

constexpr uint32_t kGemvVecSize = 16 / sizeof(bf16_t);  // 8 bf16 per 16B load

__device__ __forceinline__ float dot8_f32(const float4 wv, const float4 xv) {
  const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto [w0, w1] = cast<fp32x2_t>(w2[i]);
    const auto [x0, x1] = cast<fp32x2_t>(x2[i]);
    acc = fmaf(w0, x0, acc);
    acc = fmaf(w1, x1, acc);
  }
  return acc;
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps>
__global__ void __launch_bounds__(kNumWarps * 32)
    hopper_bf16_gemv_kernel(bf16_t* __restrict__ out, const bf16_t* __restrict__ x, const bf16_t* __restrict__ w) {
  __shared__ bf16_t sx[K];

  const uint32_t tid = threadIdx.x;
  for (uint32_t i = tid * kGemvVecSize; i < K; i += kNumWarps * 32 * kGemvVecSize) {
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

  constexpr uint32_t kStep = 32 * kGemvVecSize * kUnroll;
  if (r0 + kRows <= N) {
    for (uint32_t k = lane * kGemvVecSize * kUnroll; k < K; k += kStep) {
      float4 xv[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        xv[u] = *reinterpret_cast<const float4*>(sx + k + u * kGemvVecSize);
      }
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
        float4 wv[kUnroll];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          wv[u] = __ldcs(reinterpret_cast<const float4*>(wr + u * kGemvVecSize));
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          acc[r] += dot8_f32(wv[u], xv[u]);
        }
      }
    }
  } else {
    // Tail block: guard each row (only reached when N % (kRows*kNumWarps) != 0).
    for (uint32_t k = lane * kGemvVecSize * kUnroll; k < K; k += kStep) {
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        if (r0 + r < N) {
          const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
#pragma unroll
          for (uint32_t u = 0; u < kUnroll; ++u) {
            float4 wv = __ldcs(reinterpret_cast<const float4*>(wr + u * kGemvVecSize));
            float4 xv = *reinterpret_cast<const float4*>(sx + k + u * kGemvVecSize);
            acc[r] += dot8_f32(wv, xv);
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
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (r0 + r < N) {
        out[r0 + r] = cast<bf16_t>(acc[r]);
      }
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps>
struct HopperBf16GemvKernel {
  static_assert(K % (32 * kGemvVecSize * kUnroll) == 0, "K must cover full unrolled warp strides");
  static_assert(K * sizeof(bf16_t) <= 48 * 1024, "activation row must fit static shared memory");

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    constexpr uint32_t kRowsPerBlock = kRows * kNumWarps;
    constexpr uint32_t kNumBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    LaunchKernel(kNumBlocks, kNumWarps * 32, device.unwrap())(
        hopper_bf16_gemv_kernel<N, K, kRows, kUnroll, kNumWarps>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w.data_ptr()));
  }
};

}  // namespace sglang
