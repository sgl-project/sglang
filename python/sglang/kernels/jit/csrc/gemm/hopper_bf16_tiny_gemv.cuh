#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

// Tiny-N bf16 GEMM for decode: out[M, N] = x[M, K] @ w[N, K]^T with M <= kMaxM
// and N of a few dozen rows (per-rank KDA beta / gated-attention projections,
// N = 8 at TP8). cuBLAS serves these with a split-K GEMM plus a reduce kernel,
// ~6 us for a 64 KB weight; one block per output row with the K dimension
// spread over the block and a shuffle + shared-memory reduce finishes in the
// launch latency.

constexpr uint32_t kTinyThreads = 128;
constexpr uint32_t kTinyVec = 8;  // bf16 per 16-byte load

template <uint32_t kMaxM>
__global__ void __launch_bounds__(kTinyThreads) hopper_bf16_tiny_gemv_kernel(
    bf16_t* __restrict__ out,
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ w,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K) {
  __shared__ float sred[kTinyThreads / 32][kMaxM];
  const uint32_t n = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const bf16_t* wrow = w + static_cast<size_t>(n) * K;

  float acc[kMaxM];
#pragma unroll
  for (uint32_t m = 0; m < kMaxM; ++m) {
    acc[m] = 0.0f;
  }
  for (uint32_t k = tid * kTinyVec; k < K; k += kTinyThreads * kTinyVec) {
    const float4 wv = __ldcs(reinterpret_cast<const float4*>(wrow + k));
    const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
#pragma unroll
    for (uint32_t m = 0; m < kMaxM; ++m) {
      if (m < M) {
        const float4 xv = *reinterpret_cast<const float4*>(x + static_cast<size_t>(m) * K + k);
        const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const auto [w0, w1] = cast<fp32x2_t>(w2[i]);
          const auto [x0, x1] = cast<fp32x2_t>(x2[i]);
          p = fmaf(w0, x0, p);
          p = fmaf(w1, x1, p);
        }
        acc[m] += p;
      }
    }
  }
#pragma unroll
  for (uint32_t m = 0; m < kMaxM; ++m) {
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
      acc[m] += __shfl_xor_sync(0xffffffff, acc[m], off);
    }
  }
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  if (lane == 0) {
#pragma unroll
    for (uint32_t m = 0; m < kMaxM; ++m) {
      sred[warp][m] = acc[m];
    }
  }
  __syncthreads();
  if (tid < kMaxM && tid < M) {
    float s = 0.0f;
#pragma unroll
    for (uint32_t wi = 0; wi < kTinyThreads / 32; ++wi) {
      s += sred[wi][tid];
    }
    out[static_cast<size_t>(tid) * N + n] = cast<bf16_t>(s);
  }
}

template <uint32_t kMaxM>
struct HopperBf16TinyGemvKernel {
  static_assert(kMaxM >= 1 && kMaxM <= 32, "kMaxM out of range");

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"M"};
    auto N = SymbolicSize{"N"};
    auto K = SymbolicSize{"K"};
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto m = static_cast<uint32_t>(M.unwrap());
    const auto n = static_cast<uint32_t>(N.unwrap());
    const auto k = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(m >= 1 && m <= kMaxM, "M must be in [1, kMaxM]");
    RuntimeCheck(k % (kTinyThreads * kTinyVec) == 0, "K must be a multiple of 1024");

    LaunchKernel(n, kTinyThreads, device.unwrap())(
        hopper_bf16_tiny_gemv_kernel<kMaxM>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w.data_ptr()),
        m,
        n,
        k);
  }
};

}  // namespace sglang
