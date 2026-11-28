// SPDX-License-Identifier: Apache-2.0
// Decode-time bf16 GEMV for the YuE2 AR model on ROCm (gfx9/RDNA).
//
//   y[1, N] = x[1, K] @ W[N, K]^T      bf16 in/out, fp32 accumulation
//
// Design (batch 1 is the only regime this kernel is used in):
//   * the operator is memory bound: arithmetic intensity is 1 FLOP/byte, so
//     the kernel is written to stream W at the highest sustainable rate and
//     to keep everything else out of the critical path;
//   * one warp produces kRowsPerWarp rows, so the x vector is fetched once per
//     group instead of once per row;
//   * x is staged in shared memory once per block (K * 2 bytes, <= 12 KiB for
//     this model) and reused by all warps;
//   * W is read with 16-byte vector loads (8 bf16) coalesced across the 32
//     lanes, and every row gets a single warp reduction with fp32 accumulators;
//   * N is rounded up per block, rows past N exit immediately.
//
// The kernel intentionally owns no allocation: the caller passes `out`.
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kThreads = 128;              // 4 warps per block
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerWarp = 4;            // x reuse x4, 4 fp32 accumulators
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr int kVecElems = 8;               // 8 bf16 == 16 bytes

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down(v, off);
  }
  return v;
}

__device__ __forceinline__ void fma8(const float4& w, const float4& x, float* acc) {
  const __hip_bfloat162* w2 = reinterpret_cast<const __hip_bfloat162*>(&w);
  const __hip_bfloat162* x2 = reinterpret_cast<const __hip_bfloat162*>(&x);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 wf = __bfloat1622float2(w2[i]);
    const float2 xf = __bfloat1622float2(x2[i]);
    acc[0] += wf.x * xf.x + wf.y * xf.y;
  }
}

__global__ __launch_bounds__(kThreads) void gemv_bf16_kernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ W,
    __hip_bfloat16* __restrict__ y, int N, int K) {
  extern __shared__ __align__(16) __hip_bfloat16 s_x[];

  for (int i = threadIdx.x; i < K; i += kThreads) {
    s_x[i] = x[i];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kWarps + warp) * kRowsPerWarp;
  if (row0 >= N) {
    return;
  }
  const int rows = (row0 + kRowsPerWarp <= N) ? kRowsPerWarp : (N - row0);
  const int nvec = K / kVecElems;
  const int tail = K - nvec * kVecElems;

  float acc[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float4* x4 = reinterpret_cast<const float4*>(s_x);

  for (int i = lane; i < nvec; i += 32) {
    const float4 xv = x4[i];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      if (r < rows) {
        const float4* w4 =
            reinterpret_cast<const float4*>(W + (long)(row0 + r) * K);
        fma8(w4[i], xv, &acc[r]);
      }
    }
  }

  // K not divisible by 8 (not the case for this checkpoint, kept for safety).
  if (tail != 0 && lane < tail) {
    const int base = nvec * kVecElems + lane;
    const float xv = static_cast<float>(s_x[base]);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      if (r < rows) {
        acc[r] += static_cast<float>(W[(long)(row0 + r) * K + base]) * xv;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    if (r < rows) {
      const float total = warp_sum(acc[r]);
      if (lane == 0) {
        y[row0 + r] = static_cast<__hip_bfloat16>(total);
      }
    }
  }
}

// y: [1, N] or [N] (contiguous), x: [1, K] (contiguous), w: [N, K] (contiguous)
void gemv_bf16(at::Tensor y, at::Tensor x, at::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda(), "tensors must be on GPU");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
                  y.scalar_type() == at::kBFloat16,
              "bf16 tensors expected");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y.is_contiguous(),
              "contiguous tensors expected");
  TORCH_CHECK(w.dim() == 2 && x.dim() == 2 && x.size(0) == 1,
              "expect x[1, K] and w[N, K]");

  const int N = w.size(0);
  const int K = w.size(1);
  TORCH_CHECK(x.size(1) == K, "x/w K mismatch");
  TORCH_CHECK(y.numel() >= N, "output too small");

  hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const size_t smem = static_cast<size_t>(K) * sizeof(__hip_bfloat16);
  const int blocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
  gemv_bf16_kernel<<<blocks, kThreads, smem, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(w.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(y.data_ptr()), N, K);
  const auto rc = hipGetLastError();
  TORCH_CHECK(rc == hipSuccess, "gemv_bf16 launch failed: ", hipGetErrorString(rc));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &gemv_bf16, "bf16 GEMV y = x @ w^T (decode, batch 1)");
}
