#pragma once

#include <sgl_kernel/deepseek_v4/fp8_e4m3.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#ifndef USE_ROCM
#include <cuda_fp8.h>
#elif defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
// Only on the arches that take the hardware branch below. hip_fp8.h is what defines
// HIP_FP8_TYPE_FNUZ, and nothing else in this include tree pulls it in, so gating on
// those macros instead would also flip the software cast's arch constants on gfx942 --
// it picks fn today because the macro is not visible there.
#include <hip/hip_fp8.h>
#define SGL_ROCM_FP8_HW_CVT 1
#endif

// Small helpers shared by the DeepSeek-V4 FP8/UE8M0 quantization kernels
// (silu_and_mul_masked_post_quant, store, mega_moe_pre_dispatch, ...).
// All functions are `SGL_DEVICE` (= `__forceinline__ __device__`) so
// including this header in multiple translation units is ODR-safe.

namespace sglang {

namespace deepseek_v4::fp8 {

// Round `x` to the nearest representable UE8M0 value. Returns the raw
// 8-bit biased exponent; the actual fp32 scale is `2^(exp - 127)`
// (i.e. `__uint_as_float(exp << 23)`).
SGL_DEVICE int32_t cast_to_ue8m0(float x) {
  uint32_t u = __float_as_uint(x);
  int32_t exp = int32_t((u >> 23) & 0xFF);
  uint32_t mant = u & 0x7FFFFF;
  return exp + (mant != 0);
}

// 1 / 2^(exp - 127) as fp32. Equivalent to `1.0f / __uint_as_float(exp << 23)`.
SGL_DEVICE float inv_scale_ue8m0(int32_t exp) {
  return __uint_as_float((127 + 127 - exp) << 23);
}

// Clamp to [-FP8_E4M3_MAX, FP8_E4M3_MAX].
// Uses platform-specific max from type.cuh (448 for E4M3FN, 224 for E4M3FNUZ).
SGL_DEVICE float fp8_e4m3_clip(float val) {
  return fmaxf(fminf(val, kFP8E4M3Max), -kFP8E4M3Max);
}

#ifndef USE_ROCM
// Pack two fp32 values into a single fp8x2_e4m3 with clamping.
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};
}
#else
#ifdef SGL_ROCM_FP8_HW_CVT
// gfx950/gfx12xx do both lanes in one v_cvt_pk_fp8_f32 (RNE), and the flavour it produces
// is the OCP one kFP8E4M3Max already assumes there. Clip first rather than passing
// __HIP_SATFINITE -- the x2 fast path converts the value it was handed, not the clamped
// one (ROCm 7.2).
//
// gfx942 keeps the software cast below: this instruction does not produce the fnuz
// flavour that arch needs.
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  const fp32x2_t v{fp8_e4m3_clip(x), fp8_e4m3_clip(y)};
  return __hip_cvt_float2_to_fp8x2(v, __HIP_NOSAT, __HIP_E4M3);
}
#else
// Software float -> FP8 E4M3 conversion for the archs the branch above skips: gfx942,
// plus any target with no native fp8 convert.
// Keep the quantization clamp (FN: 448, FNUZ: 224): NaN -> +max, +/-Inf -> +/-max.
// FNUZ can represent 240, but that does not change this path's scale/clamp convention.
SGL_DEVICE uint8_t cvt_float_to_fp8_e4m3(float val) {
  // Do not leave signaling-NaN behavior to the platform's fmin/fmax lowering.
  if ((__float_as_uint(val) & 0x7fffffffu) > 0x7f800000u) val = kFP8E4M3Max;
  val = fp8_e4m3_clip(val);
#if HIP_FP8_TYPE_FNUZ
  return f32_to_fp8_e4m3_bits<true>(__float_as_uint(val));
#else
  return f32_to_fp8_e4m3_bits<false>(__float_as_uint(val));
#endif
}

// Pack two fp32 values into a single fp8x2_e4m3 (uint16_t on HIP).
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  uint8_t x8 = cvt_float_to_fp8_e4m3(x);
  uint8_t y8 = cvt_float_to_fp8_e4m3(y);
  return static_cast<uint16_t>(x8) | (static_cast<uint16_t>(y8) << 8);
}
#endif  // HIP_FP8_TYPE_OCP && !HIP_FP8_TYPE_FNUZ
#endif

}  // namespace deepseek_v4::fp8

}  // namespace sglang
