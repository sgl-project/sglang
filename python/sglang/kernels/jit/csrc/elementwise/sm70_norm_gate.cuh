// SPDX-License-Identifier: Apache-2.0
// Sigmoid-gated RMSNorm over 128-wide heads (KDA output norm) on Volta,
// bitwise equal to fla's Triton layer_norm_gated_fwd_kernel (IS_RMS_NORM,
// ACTIVATION="sigmoid", fp16 x/g/w/y). The Triton build (triton 3.x, sm_70,
// num_warps=4) lays a row over 16 lanes x 8 values and emits the PTX below;
// every rounding step here is the same PTX instruction in the same order.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_norm_gate {

inline constexpr int kDim = 128;
inline constexpr int kVec = 8;
inline constexpr int kLanesPerRow = kDim / kVec;
inline constexpr int kThreads = 128;
inline constexpr int kRowsPerBlock = kThreads / kLanesPerRow;
// log2(e) as Triton's tl.sigmoid spells it before ex2.approx.
inline constexpr float kLog2e = 1.44269502162933349609375f;  // 0x3FB8AA3B

SGL_DEVICE float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

SGL_DEVICE float sqrt_approx_ftz(float a) {
  float r;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

SGL_DEVICE float ex2_approx(float a) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

/// y[r, :] = x * rstd * w * sigmoid(g), rstd = 1 / sqrt(mean(x^2) + eps).
/// Row r sits on lanes (r % 2) * 16 .. + 15 of warp r / 2; lane l holds 8
/// consecutive values. y may alias x.
__global__ __launch_bounds__(kThreads) void norm_gate_kernel(fp16_t* y,
                                                             const fp16_t* x,
                                                             const fp16_t* __restrict__ g,
                                                             const fp16_t* __restrict__ w,
                                                             int32_t rows,
                                                             float eps) {
  using vec_t = device::AlignedVector<fp16_t, kVec>;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + static_cast<int>(threadIdx.x) / kLanesPerRow;
  const uint32_t vi = threadIdx.x % kLanesPerRow;
  const bool valid = row < rows;
  const int64_t base = static_cast<int64_t>(valid ? row : 0) * kDim;
  vec_t xv, gv, wv;
  xv.load(x + base, vi);
  float xf[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    xf[i] = valid ? static_cast<float>(xv[i]) : 0.0f;
  }
  // Triton seeds the chain with x1^2, then folds x0, x2..x7 in by fma.
  float sum_sq = __fmul_rn(xf[1], xf[1]);
  sum_sq = __fmaf_rn(xf[0], xf[0], sum_sq);
#pragma unroll
  for (int i = 2; i < kVec; ++i) {
    sum_sq = __fmaf_rn(xf[i], xf[i], sum_sq);
  }
#pragma unroll
  for (int off = kLanesPerRow / 2; off > 0; off >>= 1) {
    sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, off));
  }
  const float var = div_full(sum_sq, static_cast<float>(kDim));
  const float rstd = div_full(1.0f, sqrt_approx_ftz(__fadd_rn(eps, var)));
  if (!valid) {
    return;
  }
  gv.load(g + base, vi);
  wv.load(w, vi);
  vec_t out;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float normed = __fmul_rn(__fmul_rn(rstd, xf[i]), static_cast<float>(wv[i]));
    const float e = ex2_approx(__fmul_rn(__fsub_rn(0.0f, static_cast<float>(gv[i])), kLog2e));
    const float sig = div_full(1.0f, __fadd_rn(e, 1.0f));
    out[i] = DTypeTrait<fp16_t>::from(__fmul_rn(normed, sig));
  }
  out.store(y + base, vi);
}

/// y, x, g [rows, 128] fp16 contiguous (y may be x); w [128] fp16.
void norm_gate(tvm::ffi::TensorView y,
               tvm::ffi::TensorView x,
               tvm::ffi::TensorView g,
               tvm::ffi::TensorView w,
               double eps) {
  using namespace host;
  SymbolicSize n_rows = {"rows"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_rows, kDim}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(y).verify(x).verify(g);
  TensorMatcher({kDim}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(w);
  for (const auto* t : {&y, &x, &g, &w}) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0);
  }
  const int64_t rows = n_rows.unwrap();
  CHECK_HOST(rows > 0 && rows < (int64_t{1} << 31));
  const uint32_t blocks = static_cast<uint32_t>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  LaunchKernel(blocks, kThreads, device_.unwrap())(
      norm_gate_kernel,
      static_cast<fp16_t*>(y.data_ptr()),
      static_cast<const fp16_t*>(x.data_ptr()),
      static_cast<const fp16_t*>(g.data_ptr()),
      static_cast<const fp16_t*>(w.data_ptr()),
      static_cast<int32_t>(rows),
      static_cast<float>(eps));
}

}  // namespace sglang::sm70_norm_gate
