// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 mHC mix GEMV on Volta (SM70).
// Shapes: hidden=5120, hc_mult=4, hc_dim=20480, mix_hc=24.
// Qwen sm70_hc_*.cuh is a different geometry (hidden 2560 / down 320) -- do not call it.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_dsv41_hc {

/// DSV4.1-Flash mHC: hc_mult=4, hidden=5120.
inline constexpr int kHcMult = 4;
inline constexpr int kHidden = 5120;
inline constexpr int kHcDim = kHcMult * kHidden;          // 20480
inline constexpr int kMixHc = (2 + kHcMult) * kHcMult;    // 24
inline constexpr int kBlockSize = 256;
inline constexpr int kVecN = 8;                           // 16B fp16 loads
inline constexpr int kIters = kHcDim / (kBlockSize * kVecN);  // 10

static_assert(kHcDim % (kBlockSize * kVecN) == 0, "hc_dim must tile the CTA vector loop");
static_assert(kHidden % kVecN == 0, "hidden must be a multiple of the fp16 vector width");

/**
 * \brief CTA-wide sum via warp shuffle then 8-slot shared memory.
 *
 * Tree reduction -- not sequential. Fine for the 20480-wide mix GEMV / RMS;
 * Sinkhorn uses a sequential 4-wide loop instead (see sm70_dsv41_hc_sinkhorn.cuh).
 */
SGL_DEVICE float cta_reduce_sum(float value, float* smem) {
  value = device::warp::reduce_sum(value);
  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  const uint32_t lane = threadIdx.x % device::kWarpThreads;
  if (lane == 0) {
    smem[warp_id] = value;
  }
  __syncthreads();
  if (warp_id == 0) {
    const float partial = lane < (kBlockSize / device::kWarpThreads) ? smem[lane] : 0.0f;
    const float total = device::warp::reduce_sum(partial);
    if (lane == 0) {
      smem[0] = total;
    }
  }
  __syncthreads();
  return smem[0];
}

/**
 * \brief Fill smem mixes[24] for one token: RMS scale times 24-wide GEMV.
 *
 * All 256 threads participate. mix_s[] is written by lane 0 of the CTA.
 */
SGL_DEVICE void fill_mixes_smem(const fp16_t* __restrict__ x_row,
                                const fp32_t* __restrict__ hc_fn,
                                float rms_eps,
                                float* __restrict__ mix_s,
                                float* __restrict__ red_smem) {
  using xvec_t = device::AlignedVector<fp16_t, kVecN>;
  using wvec_t = device::AlignedVector<fp32_t, 4>;
  const uint32_t tid = threadIdx.x;

  float sqr = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kIters; ++iter) {
    const uint32_t vi = tid + static_cast<uint32_t>(iter) * kBlockSize;
    xvec_t xv;
    xv.load(x_row, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      const float v = static_cast<float>(xv[i]);
      sqr += v * v;
    }
  }
  sqr = cta_reduce_sum(sqr, red_smem);
  const float inv_rms = device::math::rsqrt(sqr / static_cast<float>(kHcDim) + rms_eps);

  for (int row = 0; row < kMixHc; ++row) {
    const fp32_t* w_row = hc_fn + static_cast<int64_t>(row) * kHcDim;
    float acc = 0.0f;
#pragma unroll
    for (int iter = 0; iter < kIters; ++iter) {
      const uint32_t vi = tid + static_cast<uint32_t>(iter) * kBlockSize;
      xvec_t xv;
      wvec_t w0;
      wvec_t w1;
      xv.load(x_row, vi);
      w0.load(w_row, static_cast<int64_t>(vi) * 2);
      w1.load(w_row, static_cast<int64_t>(vi) * 2 + 1);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc += static_cast<float>(xv[i]) * w0[i];
        acc += static_cast<float>(xv[i + 4]) * w1[i];
      }
    }
    acc = cta_reduce_sum(acc, red_smem);
    if (tid == 0) {
      mix_s[row] = acc * inv_rms;
    }
    __syncthreads();
  }
}

/**
 * \brief Per-token RMS scale of x (fp16) times GEMV against fp32 mix weights.
 *
 * \param mixes   [T, 24] fp32 output
 * \param x       [T, 20480] fp16 activations
 * \param hc_fn   [24, 20480] fp32 mix weights (row-major, F.linear layout)
 * \param rms_eps RMS denominator epsilon
 *
 * One CTA per token. Mix stats stay FP32. The 20480-wide reduction is a CTA
 * tree, so this is not bitwise vs torch.float32 F.linear.
 */
__global__ __launch_bounds__(kBlockSize) void mix_stats_kernel(
    fp32_t* __restrict__ mixes,
    const fp16_t* __restrict__ x,
    const fp32_t* __restrict__ hc_fn,
    float rms_eps) {
  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const fp16_t* x_row = x + static_cast<int64_t>(token) * kHcDim;

  __shared__ float red_smem[kBlockSize / device::kWarpThreads];
  __shared__ float mix_s[kMixHc];
  fill_mixes_smem(x_row, hc_fn, rms_eps, mix_s, red_smem);
  if (tid == 0) {
    fp32_t* mix_row = mixes + static_cast<int64_t>(token) * kMixHc;
#pragma unroll
    for (int row = 0; row < kMixHc; ++row) {
      mix_row[row] = mix_s[row];
    }
  }
}

/**
 * \brief Validate tensors and launch mix_stats_kernel.
 *
 * \param mixes   [T, 24] fp32
 * \param x       [T, 20480] fp16
 * \param hc_fn   [24, 20480] fp32
 * \param rms_eps RMS epsilon
 */
inline void mix_stats(tvm::ffi::TensorView mixes,
                      tvm::ffi::TensorView x,
                      tvm::ffi::TensorView hc_fn,
                      double rms_eps) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tokens, kHcDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(x);
  TensorMatcher({kMixHc, kHcDim})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hc_fn);
  TensorMatcher({n_tokens, kMixHc})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(mixes);

  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0) << "sm70_dsv41_hc mix_stats: num_tokens must be > 0, got " << n;
  CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc mix_stats: x must be 16-byte aligned";
  CHECK_HOST(reinterpret_cast<uintptr_t>(hc_fn.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc mix_stats: hc_fn must be 16-byte aligned";

  LaunchKernel(n, kBlockSize, device_.unwrap())(
      mix_stats_kernel,
      static_cast<fp32_t*>(mixes.data_ptr()),
      static_cast<const fp16_t*>(x.data_ptr()),
      static_cast<const fp32_t*>(hc_fn.data_ptr()),
      static_cast<float>(rms_eps));
}

}  // namespace sglang::sm70_dsv41_hc
