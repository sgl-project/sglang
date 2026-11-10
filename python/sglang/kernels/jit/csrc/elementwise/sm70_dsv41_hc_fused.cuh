// SPDX-License-Identifier: Apache-2.0
// Fuse mHC mix_stats + Sinkhorn, and optionally combine, on SM70.
// Same numerics as the three unfused kernels (mix CTA tree, sequential Sinkhorn,
// sequential k=0..3 combine). Reads x once when combine is fused in.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_dsv41_hc {

static_assert(kBlockSize == kBlockC,
              "fused CTA size must match combine_one_token stride");
static_assert(kHidden == kHiddenC && kHcMult == kHcMultC,
              "fused kernel constants must match mix/combine headers");

/**
 * \brief One CTA per token: mix GEMV, Sinkhorn, optional 4-wide combine.
 *
 * \tparam kDoCombine If true, also write y = sum_k p_k * x[k].
 * \tparam kUseApplyPre If true, combine uses apply_pre; else the new pre.
 */
template <bool kDoCombine, bool kUseApplyPre>
__global__ __launch_bounds__(kBlockSize) void mix_sinkhorn_kernel(
    fp32_t* __restrict__ pre,
    fp32_t* __restrict__ post,
    fp32_t* __restrict__ comb,
    fp16_t* __restrict__ y,
    const fp16_t* __restrict__ x,
    const fp32_t* __restrict__ hc_fn,
    const fp32_t* __restrict__ hc_scale,
    const fp32_t* __restrict__ hc_base,
    const fp32_t* __restrict__ apply_pre,
    int32_t sinkhorn_iters,
    float rms_eps,
    float hc_eps) {
  const uint32_t t = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const fp16_t* x_row = x + static_cast<int64_t>(t) * kHcDim;

  __shared__ float red_smem[kBlockSize / device::kWarpThreads];
  __shared__ float mix_s[kMixHc];
  __shared__ float pre_s[kHcMult];
  fill_mixes_smem(x_row, hc_fn, rms_eps, mix_s, red_smem);

  fp32_t* pre_row = pre + static_cast<int64_t>(t) * kHcMult;
  fp32_t* post_row = post + static_cast<int64_t>(t) * kHcMult;
  fp32_t* comb_row = comb + static_cast<int64_t>(t) * (kHcMult * kHcMult);
  if (tid == 0) {
    sinkhorn_one_token(
        mix_s, hc_scale, hc_base, pre_row, post_row, comb_row, sinkhorn_iters, hc_eps);
    if constexpr (kDoCombine) {
      if constexpr (kUseApplyPre) {
        const fp32_t* ap = apply_pre + static_cast<int64_t>(t) * kHcMult;
        pre_s[0] = ap[0];
        pre_s[1] = ap[1];
        pre_s[2] = ap[2];
        pre_s[3] = ap[3];
      } else {
        pre_s[0] = pre_row[0];
        pre_s[1] = pre_row[1];
        pre_s[2] = pre_row[2];
        pre_s[3] = pre_row[3];
      }
    }
  }
  if constexpr (kDoCombine) {
    __syncthreads();
    combine_one_token(
        y + static_cast<int64_t>(t) * kHidden,
        x_row,
        pre_s[0],
        pre_s[1],
        pre_s[2],
        pre_s[3]);
  }
}

inline void mix_sinkhorn(tvm::ffi::TensorView pre,
                         tvm::ffi::TensorView post,
                         tvm::ffi::TensorView comb,
                         tvm::ffi::TensorView x,
                         tvm::ffi::TensorView hc_fn,
                         tvm::ffi::TensorView hc_scale,
                         tvm::ffi::TensorView hc_base,
                         int64_t sinkhorn_iters,
                         double rms_eps,
                         double hc_eps) {
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
  TensorMatcher({n_tokens, kHcMult})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(pre)
      .verify(post);
  TensorMatcher({n_tokens, kHcMult, kHcMult})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(comb);
  TensorMatcher({3})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hc_scale);
  TensorMatcher({kMixHc})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hc_base);

  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0) << "sm70_dsv41_hc mix_sinkhorn: num_tokens must be > 0";
  CHECK_HOST(sinkhorn_iters >= 1)
      << "sm70_dsv41_hc mix_sinkhorn: sinkhorn_iters must be >= 1, got "
      << sinkhorn_iters;
  CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc mix_sinkhorn: x must be 16-byte aligned";
  CHECK_HOST(reinterpret_cast<uintptr_t>(hc_fn.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc mix_sinkhorn: hc_fn must be 16-byte aligned";

  LaunchKernel(n, kBlockSize, device_.unwrap())(
      mix_sinkhorn_kernel<false, false>,
      static_cast<fp32_t*>(pre.data_ptr()),
      static_cast<fp32_t*>(post.data_ptr()),
      static_cast<fp32_t*>(comb.data_ptr()),
      static_cast<fp16_t*>(nullptr),
      static_cast<const fp16_t*>(x.data_ptr()),
      static_cast<const fp32_t*>(hc_fn.data_ptr()),
      static_cast<const fp32_t*>(hc_scale.data_ptr()),
      static_cast<const fp32_t*>(hc_base.data_ptr()),
      static_cast<const fp32_t*>(nullptr),
      static_cast<int32_t>(sinkhorn_iters),
      static_cast<float>(rms_eps),
      static_cast<float>(hc_eps));
}

inline void mix_sinkhorn_combine(tvm::ffi::TensorView y,
                                 tvm::ffi::TensorView pre,
                                 tvm::ffi::TensorView post,
                                 tvm::ffi::TensorView comb,
                                 tvm::ffi::TensorView x,
                                 tvm::ffi::TensorView hc_fn,
                                 tvm::ffi::TensorView hc_scale,
                                 tvm::ffi::TensorView hc_base,
                                 tvm::ffi::TensorView apply_pre,
                                 int64_t sinkhorn_iters,
                                 double rms_eps,
                                 double hc_eps,
                                 int64_t use_apply_pre) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tokens, kHcDim})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(x);
  TensorMatcher({n_tokens, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(y);
  TensorMatcher({kMixHc, kHcDim})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hc_fn);
  TensorMatcher({n_tokens, kHcMult})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(pre)
      .verify(post);
  TensorMatcher({n_tokens, kHcMult, kHcMult})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(comb);
  TensorMatcher({3})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hc_scale);
  TensorMatcher({kMixHc})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hc_base);

  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0) << "sm70_dsv41_hc mix_sinkhorn_combine: num_tokens must be > 0";
  CHECK_HOST(sinkhorn_iters >= 1)
      << "sm70_dsv41_hc mix_sinkhorn_combine: sinkhorn_iters must be >= 1";
  CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc mix_sinkhorn_combine: x must be 16-byte aligned";
  CHECK_HOST(reinterpret_cast<uintptr_t>(hc_fn.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc mix_sinkhorn_combine: hc_fn must be 16-byte aligned";

  const DLDevice dev = device_.unwrap();
  if (use_apply_pre != 0) {
    TensorMatcher({n_tokens, kHcMult})  //
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(apply_pre);
    CHECK_HOST(apply_pre.data_ptr() != pre.data_ptr())
        << "sm70_dsv41_hc mix_sinkhorn_combine: apply_pre must not alias pre";
    LaunchKernel(n, kBlockSize, dev)(
        mix_sinkhorn_kernel<true, true>,
        static_cast<fp32_t*>(pre.data_ptr()),
        static_cast<fp32_t*>(post.data_ptr()),
        static_cast<fp32_t*>(comb.data_ptr()),
        static_cast<fp16_t*>(y.data_ptr()),
        static_cast<const fp16_t*>(x.data_ptr()),
        static_cast<const fp32_t*>(hc_fn.data_ptr()),
        static_cast<const fp32_t*>(hc_scale.data_ptr()),
        static_cast<const fp32_t*>(hc_base.data_ptr()),
        static_cast<const fp32_t*>(apply_pre.data_ptr()),
        static_cast<int32_t>(sinkhorn_iters),
        static_cast<float>(rms_eps),
        static_cast<float>(hc_eps));
  } else {
    LaunchKernel(n, kBlockSize, dev)(
        mix_sinkhorn_kernel<true, false>,
        static_cast<fp32_t*>(pre.data_ptr()),
        static_cast<fp32_t*>(post.data_ptr()),
        static_cast<fp32_t*>(comb.data_ptr()),
        static_cast<fp16_t*>(y.data_ptr()),
        static_cast<const fp16_t*>(x.data_ptr()),
        static_cast<const fp32_t*>(hc_fn.data_ptr()),
        static_cast<const fp32_t*>(hc_scale.data_ptr()),
        static_cast<const fp32_t*>(hc_base.data_ptr()),
        static_cast<const fp32_t*>(nullptr),
        static_cast<int32_t>(sinkhorn_iters),
        static_cast<float>(rms_eps),
        static_cast<float>(hc_eps));
  }
}

}  // namespace sglang::sm70_dsv41_hc
