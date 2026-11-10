// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 mHC Sinkhorn on Volta (SM70). FP32 mixing stats, 20 iters.
// Reduction order matches `_hc_split_sinkhorn_torch` (sequential k=0..3).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_dsv41_hc {

inline constexpr int kHcMultS = 4;
inline constexpr int kMixHcS = (2 + kHcMultS) * kHcMultS;  // 24

SGL_DEVICE float sequential_sum4(float a0, float a1, float a2, float a3) {
  // Left-to-right: ((a0+a1)+a2)+a3. Same order as a 4-wide CPU torch sum.
  float s = a0 + a1;
  s = s + a2;
  s = s + a3;
  return s;
}

SGL_DEVICE float sequential_max4(float a0, float a1, float a2, float a3) {
  float m = a0;
  m = fmaxf(m, a1);
  m = fmaxf(m, a2);
  m = fmaxf(m, a3);
  return m;
}

/**
 * \brief Sinkhorn for one token from a 24-wide mix vector in registers/smem.
 *
 * Same 4-wide sequential reductions as `_hc_split_sinkhorn_torch`.
 * Thread-private: caller must be a single thread (typically tid 0).
 */
SGL_DEVICE void sinkhorn_one_token(const float* __restrict__ mix,
                                   const fp32_t* __restrict__ hc_scale,
                                   const fp32_t* __restrict__ hc_base,
                                   fp32_t* __restrict__ pre_row,
                                   fp32_t* __restrict__ post_row,
                                   fp32_t* __restrict__ comb_row,
                                   int32_t sinkhorn_iters,
                                   float eps) {
  const float scale0 = hc_scale[0];
  const float scale1 = hc_scale[1];
  const float scale2 = hc_scale[2];

#pragma unroll
  for (int j = 0; j < kHcMultS; ++j) {
    const float logit = mix[j] * scale0 + hc_base[j];
    pre_row[j] = 1.0f / (1.0f + expf(-logit)) + eps;
  }
#pragma unroll
  for (int j = 0; j < kHcMultS; ++j) {
    const float logit = mix[j + kHcMultS] * scale1 + hc_base[j + kHcMultS];
    post_row[j] = 2.0f * (1.0f / (1.0f + expf(-logit)));
  }

  float cm[kHcMultS][kHcMultS];
#pragma unroll
  for (int j = 0; j < kHcMultS; ++j) {
#pragma unroll
    for (int k = 0; k < kHcMultS; ++k) {
      const int idx = kHcMultS * 2 + j * kHcMultS + k;
      cm[j][k] = mix[idx] * scale2 + hc_base[idx];
    }
  }

  // Initial row softmax (numerically stabilized) then column normalize.
#pragma unroll
  for (int j = 0; j < kHcMultS; ++j) {
    const float row_max = sequential_max4(cm[j][0], cm[j][1], cm[j][2], cm[j][3]);
    cm[j][0] = expf(cm[j][0] - row_max);
    cm[j][1] = expf(cm[j][1] - row_max);
    cm[j][2] = expf(cm[j][2] - row_max);
    cm[j][3] = expf(cm[j][3] - row_max);
    const float row_sum = sequential_sum4(cm[j][0], cm[j][1], cm[j][2], cm[j][3]);
    // First row pass: divide then add eps (matches torch: x/sum + eps).
    cm[j][0] = cm[j][0] / row_sum + eps;
    cm[j][1] = cm[j][1] / row_sum + eps;
    cm[j][2] = cm[j][2] / row_sum + eps;
    cm[j][3] = cm[j][3] / row_sum + eps;
  }
#pragma unroll
  for (int k = 0; k < kHcMultS; ++k) {
    const float col_sum = sequential_sum4(cm[0][k], cm[1][k], cm[2][k], cm[3][k]);
    const float denom = col_sum + eps;
    cm[0][k] = cm[0][k] / denom;
    cm[1][k] = cm[1][k] / denom;
    cm[2][k] = cm[2][k] / denom;
    cm[3][k] = cm[3][k] / denom;
  }

  for (int32_t it = 0; it < sinkhorn_iters - 1; ++it) {
#pragma unroll
    for (int j = 0; j < kHcMultS; ++j) {
      const float row_sum = sequential_sum4(cm[j][0], cm[j][1], cm[j][2], cm[j][3]);
      const float denom = row_sum + eps;
      cm[j][0] = cm[j][0] / denom;
      cm[j][1] = cm[j][1] / denom;
      cm[j][2] = cm[j][2] / denom;
      cm[j][3] = cm[j][3] / denom;
    }
#pragma unroll
    for (int k = 0; k < kHcMultS; ++k) {
      const float col_sum = sequential_sum4(cm[0][k], cm[1][k], cm[2][k], cm[3][k]);
      const float denom = col_sum + eps;
      cm[0][k] = cm[0][k] / denom;
      cm[1][k] = cm[1][k] / denom;
      cm[2][k] = cm[2][k] / denom;
      cm[3][k] = cm[3][k] / denom;
    }
  }

#pragma unroll
  for (int j = 0; j < kHcMultS; ++j) {
#pragma unroll
    for (int k = 0; k < kHcMultS; ++k) {
      comb_row[j * kHcMultS + k] = cm[j][k];
    }
  }
}

/**
 * \brief One-thread-per-token Sinkhorn matching `_hc_split_sinkhorn_torch`.
 *
 * Layout of mixes[t, :]: pre (4), post (4), comb row-major 4x4 (16).
 * First comb pass is row-softmax then `x/sum + eps`, then column
 * `x / (sum + eps)`. Remaining iters use `x / (sum + eps)` on both axes.
 *
 * \param pre            [T, 4] fp32
 * \param post           [T, 4] fp32
 * \param comb           [T, 4, 4] fp32
 * \param mixes          [T, 24] fp32
 * \param hc_scale       [3] fp32
 * \param hc_base        [24] fp32
 * \param sinkhorn_iters iteration count (DSV4.1 default 20)
 * \param eps            additive / denominator epsilon
 */
__global__ void split_sinkhorn_kernel(fp32_t* __restrict__ pre,
                                      fp32_t* __restrict__ post,
                                      fp32_t* __restrict__ comb,
                                      const fp32_t* __restrict__ mixes,
                                      const fp32_t* __restrict__ hc_scale,
                                      const fp32_t* __restrict__ hc_base,
                                      int32_t sinkhorn_iters,
                                      float eps) {
  const uint32_t t = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }

  const fp32_t* mix = mixes + static_cast<int64_t>(t) * kMixHcS;
  sinkhorn_one_token(
      mix,
      hc_scale,
      hc_base,
      pre + static_cast<int64_t>(t) * kHcMultS,
      post + static_cast<int64_t>(t) * kHcMultS,
      comb + static_cast<int64_t>(t) * (kHcMultS * kHcMultS),
      sinkhorn_iters,
      eps);
}

/**
 * \brief Validate tensors and launch split_sinkhorn_kernel.
 */
inline void split_sinkhorn(tvm::ffi::TensorView pre,
                           tvm::ffi::TensorView post,
                           tvm::ffi::TensorView comb,
                           tvm::ffi::TensorView mixes,
                           tvm::ffi::TensorView hc_scale,
                           tvm::ffi::TensorView hc_base,
                           int64_t sinkhorn_iters,
                           double eps) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tokens, kMixHcS})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(mixes);
  TensorMatcher({n_tokens, kHcMultS})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(pre)
      .verify(post);
  TensorMatcher({n_tokens, kHcMultS, kHcMultS})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(comb);
  TensorMatcher({3})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hc_scale);
  TensorMatcher({kMixHcS})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hc_base);

  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0) << "sm70_dsv41_hc split_sinkhorn: num_tokens must be > 0";
  CHECK_HOST(sinkhorn_iters >= 1)
      << "sm70_dsv41_hc split_sinkhorn: sinkhorn_iters must be >= 1, got "
      << sinkhorn_iters;

  LaunchKernel(n, 32, device_.unwrap())(
      split_sinkhorn_kernel,
      static_cast<fp32_t*>(pre.data_ptr()),
      static_cast<fp32_t*>(post.data_ptr()),
      static_cast<fp32_t*>(comb.data_ptr()),
      static_cast<const fp32_t*>(mixes.data_ptr()),
      static_cast<const fp32_t*>(hc_scale.data_ptr()),
      static_cast<const fp32_t*>(hc_base.data_ptr()),
      static_cast<int32_t>(sinkhorn_iters),
      static_cast<float>(eps));
}

}  // namespace sglang::sm70_dsv41_hc
