#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/quantization/fp8.cuh>

#ifndef USE_ROCM
#include <cuda_fp4.h>
#endif

// FP4 (e2m1) helpers: e2m1 conversion, the per-block UE8M0 scale and the e2m1 roundtrip.

namespace sglang {

namespace device::fp4 {

/// Largest finite e2m1 value.
constexpr float kMax = 6.0f;
/// `6 * 2^-126`, the amax floor `fp4_fake_quant.fake_quant_fp4` clamps to.
constexpr float kAmaxFloor = 6.0f * 1.1754943508222875e-38f;
#ifdef USE_ROCM
/// \brief `cvt.rn.satfinite.e2m1x2.f32` for one value, in software: RNE onto the
/// e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, saturated at 6, sign kept (also for -0).
SGL_DEVICE uint32_t e2m1_code(float x) {
  const float mag = fminf(fabsf(x), kMax);
  const float step = mag < 2.0f ? 0.5f : (mag < 4.0f ? 1.0f : 2.0f);
  const float q = rintf(mag / step) * step;
  // q in {0, .5, 1, 1.5} -> 2q; {2, 3} -> q + 2; {4, 6} -> q / 2 + 4.
  const uint32_t idx = q < 2.0f ? static_cast<uint32_t>(q * 2.0f)
                                : (q < 4.0f ? static_cast<uint32_t>(q) + 2u : static_cast<uint32_t>(q * 0.5f) + 4u);
  return idx | (__float_as_uint(x) >> 31 << 3);
}

/// \brief The e2m1 grid value a code stands for, exact in fp32.
SGL_DEVICE float e2m1_value(uint32_t code) {
  const uint32_t idx = code & 0x7u;
  const float mag = idx < 4u ? static_cast<float>(idx) * 0.5f
                             : (idx < 6u ? static_cast<float>(idx) - 2.0f : static_cast<float>(idx - 4u) * 2.0f);
  return (code & 0x8u) ? -mag : mag;
}

/// \brief `__nv_cvt_float2_to_fp4x2(.., __NV_E2M1, cudaRoundNearest)`: `.x` in
/// the low nibble, `.y` in the high one.
SGL_DEVICE uint32_t e2m1x2_code(fp32x2_t x) {
  return e2m1_code(x.x) | (e2m1_code(x.y) << 4);
}
#endif

/// \brief Per-block ue8m0 scale and its reciprocal, from the block's absmax.
SGL_DEVICE fp32x2_t block_scale(float amax) {
  const auto exponent = fp8::cast_to_ue8m0(fmaxf(amax, kAmaxFloor) * (1.0f / kMax));
  return {__uint_as_float(static_cast<uint32_t>(exponent) << 23), fp8::inv_scale_ue8m0(exponent)};
}

/// \brief Round a pair onto the e2m1 grid and back, through `scale`.
///
/// Every e2m1 value is exact in fp16, so the roundtrip is lossless. Adding `0.0f` during
/// scaling clears negative zero to match `torch.sign(0) == 0` in `fp4_fake_quant.round_fp4`.
SGL_DEVICE fp32x2_t fake_quant_x2(fp32x2_t x, float scale, float inv_scale) {
  const fp32x2_t scaled{__fmaf_rn(x.x, inv_scale, 0.0f), __fmaf_rn(x.y, inv_scale, 0.0f)};
#ifdef USE_ROCM
  const fp32x2_t grid{e2m1_value(e2m1_code(scaled.x)), e2m1_value(e2m1_code(scaled.y))};
#else
  const auto code = __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
  const auto grid = device::cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
#endif
  return {grid.x * scale, grid.y * scale};
}

}  // namespace device::fp4

}  // namespace sglang
