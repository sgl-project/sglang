#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

// SM120 (RTX 5090 / RTX PRO 6000) bf16 GEMV for skinny decode, y[M,N] = x[M,K] @ w[N,K]^T.
//
// Layout: one warp computes kRows consecutive output rows (all M activations for
// those rows); the M activation rows are staged once in dynamic shared memory and
// reused by every warp; weights are streamed with evict-first loads (read exactly
// once). All reductions happen in registers + one warp shuffle tree, so there is
// no split-K fixup kernel. fp32 accumulation. Weight traffic (N*K*2 bytes)
// dominates, so the design goal is maximum sustained DRAM read bandwidth, which
// cuBLAS's M<16 tiles leave on the table (measured 0.2-2.3 TB/s vs 1.5-4.6 TB/s).

constexpr uint32_t kGemvVec = 16 / sizeof(bf16_t);  // 8 bf16 per 16B load

// Chunked accumulation: one fp32 chain per bf16x2 pair (a 16B fragment has
// kChunkAccum = 4 pairs), folded into a total before the warp shuffle tree.
// The original single serial FMA chain per lane accumulated K/8 roundings
// into every output, so the largest-K shapes landed on a different bf16 grid
// point than cuBLAS (e.g. |diff| = 1.0 at |y| ~ 150 > the 2e-2 rtol gate).
// Four independent chains (length K/128 for the largest gated K) bound the
// error to 1-2 ulp of cuBLAS's own deviation from fp64 on every gated shape,
// for 1 extra fp32 add per pair and no extra weight traffic (DRAM read
// bandwidth is the bottleneck, so perf is unchanged).
constexpr uint32_t kChunkAccum = 4;  // fp32 chains per lane (one per bf16x2 pair)

// One 16B fragment (8 bf16) of the w * x dot into the per-pair chains.
__device__ __forceinline__ void dot8_chunk_f32_gemv(
    float (&lane_acc)[kChunkAccum], const float4 wv, const float4 xv) {
  const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
#pragma unroll
  for (uint32_t c = 0; c < kChunkAccum; ++c) {
    const auto [w0, w1] = cast<fp32x2_t>(w2[c]);
    const auto [x0, x1] = cast<fp32x2_t>(x2[c]);
    lane_acc[c] = fmaf(w0, x0, lane_acc[c]);
    lane_acc[c] = fmaf(w1, x1, lane_acc[c]);
  }
}

template <uint32_t N, uint32_t K, uint32_t M, uint32_t kRows, uint32_t kNumWarps>
__global__ void __launch_bounds__(kNumWarps * 32) sm120_bf16_gemv_kernel(
    bf16_t* __restrict__ out, const bf16_t* __restrict__ x, const bf16_t* __restrict__ w) {
  extern __shared__ __align__(16) bf16_t sx[];  // M * K activations

  const uint32_t tid = threadIdx.x;
  constexpr uint32_t kThreads = kNumWarps * 32;
  // stage all M activation rows (x is [M, K] row-major, contiguous)
  for (uint32_t i = tid * kGemvVec; i < M * K; i += kThreads * kGemvVec) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const uint32_t warp = tid >> 5, lane = tid & 31;
  const uint32_t r0 = (blockIdx.x * kNumWarps + warp) * kRows;
  if (r0 >= N) {
    return;
  }

  float lane_acc[M][kRows][kChunkAccum];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
      for (uint32_t c = 0; c < kChunkAccum; ++c) {
        lane_acc[m][r][c] = 0.0f;
      }
    }
  }

  if (r0 + kRows <= N) {
    for (uint32_t k = lane * kGemvVec; k < K; k += 32 * kGemvVec) {
      float4 xv[M];
#pragma unroll
      for (uint32_t m = 0; m < M; ++m) {
        xv[m] = *reinterpret_cast<const float4*>(sx + m * K + k);
      }
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
        const float4 wv = __ldcs(reinterpret_cast<const float4*>(wr));
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
          dot8_chunk_f32_gemv(lane_acc[m][r], wv, xv[m]);
        }
      }
    }
  } else {
    // Tail block: guard each row (reached when N % (kRows*kNumWarps) != 0).
    for (uint32_t k = lane * kGemvVec; k < K; k += 32 * kGemvVec) {
      float4 xv[M];
#pragma unroll
      for (uint32_t m = 0; m < M; ++m) {
        xv[m] = *reinterpret_cast<const float4*>(sx + m * K + k);
      }
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        if (r0 + r < N) {
          const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
          const float4 wv = __ldcs(reinterpret_cast<const float4*>(wr));
#pragma unroll
          for (uint32_t m = 0; m < M; ++m) {
            dot8_chunk_f32_gemv(lane_acc[m][r], wv, xv[m]);
          }
        }
      }
    }
  }

  // Reduce across the warp with a shuffle tree, then fold the per-pair
  // chains into the fp32 total.
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      float total = 0.0f;
#pragma unroll
      for (uint32_t c = 0; c < kChunkAccum; ++c) {
        float acc = lane_acc[m][r][c];
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
          acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        total += acc;
      }
      if (lane == 0 && r0 + r < N) {
        out[static_cast<size_t>(m) * N + r0 + r] = cast<bf16_t>(total);
      }
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t M, uint32_t kRows, uint32_t kNumWarps>
struct Sm120Bf16GemvKernel {
  static_assert(K % (32 * kGemvVec) == 0, "K must cover a full warp stride (K % 256 == 0)");
  static_assert(M * K * sizeof(bf16_t) <= 200 * 1024, "activation tile must fit shared memory");
  static_assert(M == 1 || M == 2 || M == 4, "GEMV path is tuned for M <= 4");

  static constexpr uint32_t kRowsPerBlock = kRows * kNumWarps;
  static constexpr uint32_t kNumBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
  static constexpr std::size_t kSmem = M * K * sizeof(bf16_t);

  // One-time device-side setup: opting in to >48KB dynamic shared memory
  // (cudaFuncSetAttribute) is a module-level mutation that is illegal inside
  // CUDA graph capture. Callers that may capture this kernel in a graph must
  // call warmup() outside the capture first (run() lazily does it on the
  // first eager call, so an eager call is also a valid warmup).
  static void warmup() {
    if constexpr (kSmem > 48 * 1024) {
      // Request exactly kSmem bytes (never a fixed 220KB): SM120 (RTX 5090)
      // caps opt-in dynamic shared memory at 101376B/block, far below the
      // B200 figure this constant was copied from, and an over-limit request
      // fails cudaFuncSetAttribute with cudaErrorInvalidValue.
      host::RuntimeDeviceCheck(cudaFuncSetAttribute(
          sm120_bf16_gemv_kernel<N, K, M, kRows, kNumWarps>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(kSmem)));
    }
  }

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    auto stream = LaunchKernel::resolve_device(device.unwrap());
    static bool warmed_up = (warmup(), true);
    (void)warmed_up;
    LaunchKernel(dim3(kNumBlocks), dim3(kNumWarps * 32), stream, kSmem)(
        sm120_bf16_gemv_kernel<N, K, M, kRows, kNumWarps>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w.data_ptr()));
  }
};

}  // namespace sglang
