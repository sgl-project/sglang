// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4.1 mHC combine / post on Volta (SM70).
// fp16 activations, fp32 mix coefficients. No PDL / TMA / wgmma.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_dsv41_hc {

inline constexpr int kHcMultC = 4;
inline constexpr int kHiddenC = 5120;
inline constexpr int kHcDimC = kHcMultC * kHiddenC;
inline constexpr int kVecNC = 8;
inline constexpr int kBlockC = 256;
inline constexpr int kHiddenVecs = kHiddenC / kVecNC;  // 640

/**
 * \brief y[t, h] = sum_k pre[t, k] * x[t, k, h]  (fp32 acc, fp16 store).
 *
 * Sequential k=0..3, matching a 4-wide torch sum over the HC axis.
 * All threads in the CTA participate; `t` is blockIdx.x.
 */
SGL_DEVICE void combine_one_token(fp16_t* __restrict__ y_row,
                                  const fp16_t* __restrict__ x_row,
                                  float p0,
                                  float p1,
                                  float p2,
                                  float p3) {
  using vec_t = device::AlignedVector<fp16_t, kVecNC>;
  const uint32_t tid = threadIdx.x;
  for (uint32_t vi = tid; vi < kHiddenVecs; vi += kBlockC) {
    vec_t a;
    vec_t b;
    vec_t c;
    vec_t d;
    vec_t out;
    a.load(x_row + 0 * kHiddenC, vi);
    b.load(x_row + 1 * kHiddenC, vi);
    c.load(x_row + 2 * kHiddenC, vi);
    d.load(x_row + 3 * kHiddenC, vi);
#pragma unroll
    for (int i = 0; i < kVecNC; ++i) {
      float acc = p0 * static_cast<float>(a[i]);
      acc = acc + p1 * static_cast<float>(b[i]);
      acc = acc + p2 * static_cast<float>(c[i]);
      acc = acc + p3 * static_cast<float>(d[i]);
      out[i] = DTypeTrait<fp16_t>::from(acc);
    }
    out.store(y_row, vi);
  }
}

/**
 * \brief y[t, h] = sum_k pre[t, k] * x[t, k, h]  (fp32 acc, fp16 store).
 *
 * Sequential k=0..3, matching a 4-wide torch sum over the HC axis.
 */
__global__ __launch_bounds__(kBlockC) void combine_kernel(fp16_t* __restrict__ y,
                                                          const fp16_t* __restrict__ x,
                                                          const fp32_t* __restrict__ pre) {
  const uint32_t t = blockIdx.x;
  const fp16_t* x_row = x + static_cast<int64_t>(t) * kHcDimC;
  fp16_t* y_row = y + static_cast<int64_t>(t) * kHiddenC;
  const fp32_t* pre_row = pre + static_cast<int64_t>(t) * kHcMultC;
  combine_one_token(y_row, x_row, pre_row[0], pre_row[1], pre_row[2], pre_row[3]);
}

/**
 * \brief out[t, j, h] = post[t, j] * x[t, h] + sum_k comb[t, k, j] * residual[t, k, h]
 */
__global__ __launch_bounds__(kBlockC) void post_kernel(fp16_t* __restrict__ out,
                                                       const fp16_t* __restrict__ x,
                                                       const fp16_t* __restrict__ residual,
                                                       const fp32_t* __restrict__ post,
                                                       const fp32_t* __restrict__ comb) {
  using vec_t = device::AlignedVector<fp16_t, kVecNC>;
  const uint32_t t = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const fp16_t* x_row = x + static_cast<int64_t>(t) * kHiddenC;
  const fp16_t* res_row = residual + static_cast<int64_t>(t) * kHcDimC;
  fp16_t* out_row = out + static_cast<int64_t>(t) * kHcDimC;
  const fp32_t* post_row = post + static_cast<int64_t>(t) * kHcMultC;
  const fp32_t* comb_row = comb + static_cast<int64_t>(t) * (kHcMultC * kHcMultC);

  float post_v[kHcMultC];
  float comb_kj[kHcMultC][kHcMultC];
#pragma unroll
  for (int j = 0; j < kHcMultC; ++j) {
    post_v[j] = post_row[j];
#pragma unroll
    for (int k = 0; k < kHcMultC; ++k) {
      comb_kj[k][j] = comb_row[k * kHcMultC + j];
    }
  }

  for (uint32_t vi = tid; vi < kHiddenVecs; vi += kBlockC) {
    vec_t xv;
    vec_t rv[kHcMultC];
    xv.load(x_row, vi);
#pragma unroll
    for (int k = 0; k < kHcMultC; ++k) {
      rv[k].load(res_row + k * kHiddenC, vi);
    }
#pragma unroll
    for (int j = 0; j < kHcMultC; ++j) {
      vec_t ov;
#pragma unroll
      for (int i = 0; i < kVecNC; ++i) {
        float acc = post_v[j] * static_cast<float>(xv[i]);
        acc = acc + comb_kj[0][j] * static_cast<float>(rv[0][i]);
        acc = acc + comb_kj[1][j] * static_cast<float>(rv[1][i]);
        acc = acc + comb_kj[2][j] * static_cast<float>(rv[2][i]);
        acc = acc + comb_kj[3][j] * static_cast<float>(rv[3][i]);
        ov[i] = DTypeTrait<fp16_t>::from(acc);
      }
      ov.store(out_row + j * kHiddenC, vi);
    }
  }
}

/**
 * \brief Validate and launch combine_kernel.
 *
 * \param y    [T, 5120] fp16
 * \param x    [T, 20480] fp16
 * \param pre  [T, 4] fp32
 */
inline void combine(tvm::ffi::TensorView y,
                    tvm::ffi::TensorView x,
                    tvm::ffi::TensorView pre) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tokens, kHcDimC})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(x);
  TensorMatcher({n_tokens, kHiddenC})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(y);
  TensorMatcher({n_tokens, kHcMultC})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(pre);

  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0) << "sm70_dsv41_hc combine: num_tokens must be > 0";
  CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc combine: x must be 16-byte aligned";

  LaunchKernel(n, kBlockC, device_.unwrap())(
      combine_kernel,
      static_cast<fp16_t*>(y.data_ptr()),
      static_cast<const fp16_t*>(x.data_ptr()),
      static_cast<const fp32_t*>(pre.data_ptr()));
}

/**
 * \brief Validate and launch post_kernel.
 *
 * \param out       [T, 4, 5120] fp16
 * \param x         [T, 5120] fp16
 * \param residual  [T, 4, 5120] fp16
 * \param post      [T, 4] fp32
 * \param comb      [T, 4, 4] fp32
 */
inline void post(tvm::ffi::TensorView out,
                 tvm::ffi::TensorView x,
                 tvm::ffi::TensorView residual,
                 tvm::ffi::TensorView post_mix,
                 tvm::ffi::TensorView comb) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tokens, kHiddenC})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(x);
  TensorMatcher({n_tokens, kHcMultC, kHiddenC})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(residual)
      .verify(out);
  TensorMatcher({n_tokens, kHcMultC})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(post_mix);
  TensorMatcher({n_tokens, kHcMultC, kHcMultC})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(comb);

  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0) << "sm70_dsv41_hc post: num_tokens must be > 0";
  CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc post: x must be 16-byte aligned";
  CHECK_HOST(reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16 == 0)
      << "sm70_dsv41_hc post: residual must be 16-byte aligned";

  LaunchKernel(n, kBlockC, device_.unwrap())(
      post_kernel,
      static_cast<fp16_t*>(out.data_ptr()),
      static_cast<const fp16_t*>(x.data_ptr()),
      static_cast<const fp16_t*>(residual.data_ptr()),
      static_cast<const fp32_t*>(post_mix.data_ptr()),
      static_cast<const fp32_t*>(comb.data_ptr()));
}

}  // namespace sglang::sm70_dsv41_hc
