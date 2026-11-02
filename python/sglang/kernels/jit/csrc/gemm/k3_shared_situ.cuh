// BS1 K3 shared gate/up projection with a SiTU epilogue.
#pragma once
#include "../kimi_k3/situ_and_mul.cuh"
#include "tiny_gemm.cuh"

namespace sglang {

struct K3SharedSituParams {
  const bf16_t* x;
  const bf16_t* w;
  float* front;
  bf16_t* activated;
  float beta;
  float inv_beta;
  float linear_beta;
  float inv_linear_beta;
};

template <bool kHasLinearBeta, bool kUsePDL>
__global__ __launch_bounds__(448, 1) void k3_shared_situ_kernel(const K3SharedSituParams params) {
  using namespace device;
  using Trait = GEMMTraitN<1536, 7168, 16, 32>;
  constexpr uint32_t K = Trait::K;
  constexpr uint32_t N_SPLIT = Trait::N_SPLIT;
  constexpr uint32_t kBlockSize = Trait::kBlockSize;
  constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;
  constexpr uint32_t kUnroll = Trait::kUnroll;
  using vec_t = AlignedVector<bf16x2_t, Trait::kVecSize / 2>;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  // Each CTA owns eight gate rows followed by their eight paired up rows.
  vec_t wv[N_SPLIT][kUnroll];
#pragma unroll
  for (uint32_t n = 0; n < N_SPLIT; ++n) {
    const auto row = bx * 8 + n % 8 + (n / 8) * 768;
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      wv[n][u].load(params.w + row * K, tx + u * kBlockSize);
    }
  }

  PDLWaitPrimary<kUsePDL>();
  vec_t xv[kUnroll];
#pragma unroll
  for (uint32_t u = 0; u < kUnroll; ++u) {
    xv[u].load(params.x, tx + u * kBlockSize);
  }

  __shared__ float s_acc[kNumWarps][N_SPLIT];
  const auto warp_id = tx / kWarpThreads;
#pragma unroll
  for (uint32_t n = 0; n < N_SPLIT; ++n) {
    float acc = 0.0f;
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      dot_product_vec(xv[u], wv[n][u], acc);
    }
    s_acc[warp_id][n] = warp::reduce_sum(acc);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

  if (tx < N_SPLIT) {
    float acc[kNumWarps];
#pragma unroll
    for (uint32_t i = 0; i < kNumWarps; ++i)
      acc[i] = s_acc[i][tx];
#pragma unroll
    for (uint32_t i = 1; i < kNumWarps; ++i)
      acc[0] += acc[i];

    const auto row = bx * 8 + tx % 8 + (tx / 8) * 768;
    params.front[row] = acc[0];
    // Gate/up threads are the adjacent groups [0,8), [8,16) of warp 0.
    const float up = __shfl_sync(0x0000ffff, acc[0], (tx % 8) + 8);
    if (tx < 8) {
      params.activated[bx * 8 + tx] = cast<bf16_t>(kimi_k3::situ_activate<kHasLinearBeta>(
          acc[0], up, params.beta, params.inv_beta, params.linear_beta, params.inv_linear_beta));
    }
  }
}

template <bool kUsePDL>
struct K3SharedSituKernel {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView front,
      const tvm::ffi::TensorView activated,
      const double beta,
      const double linear_beta,
      const bool has_linear_beta) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, 7168}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({1536, 7168}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({1, 1536}).with_dtype<fp32_t>().with_device(device).verify(front);
    TensorMatcher({1, 768}).with_dtype<bf16_t>().with_device(device).verify(activated);
    const auto bf = static_cast<float>(beta);
    const auto lf = static_cast<float>(linear_beta);
    const K3SharedSituParams params{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .front = static_cast<float*>(front.data_ptr()),
        .activated = static_cast<bf16_t*>(activated.data_ptr()),
        .beta = bf,
        .inv_beta = 1.0f / bf,
        .linear_beta = lf,
        .inv_linear_beta = lf != 0.0f ? 1.0f / lf : 0.0f,
    };
    const auto kernel = has_linear_beta ? k3_shared_situ_kernel<true, kUsePDL> : k3_shared_situ_kernel<false, kUsePDL>;
    LaunchKernel(96, 448, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
