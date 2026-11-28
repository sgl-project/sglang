#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#ifndef USE_ROCM
#include <cuda_fp8.h>
#elif defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
// Only on the arches that take the hardware branch below. The software cast's flavour
// comes from the JIT build flags instead: get_default_target_flags passes
// -DHIP_FP8_TYPE_FNUZ=1 on gfx942 and -DHIP_FP8_TYPE_E4M3=1 elsewhere.
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
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  const fp32x2_t v{fp8_e4m3_clip(x), fp8_e4m3_clip(y)};
  return __hip_cvt_float2_to_fp8x2(v, __HIP_NOSAT, __HIP_E4M3);
}
#else
// Software float -> FP8 E4M3 conversion for the archs the branch above skips: gfx942,
// plus any target with no native fp8 convert. Rounds to nearest even like torch's cast:
// the rescale by a power of two is exact, so rintf does the rounding.
SGL_DEVICE uint8_t cvt_float_to_fp8_e4m3(float val) {
  val = fp8_e4m3_clip(val);
  const uint8_t sign = static_cast<uint8_t>((__float_as_uint(val) >> 31) << 7);
  const float a = fabsf(val);

#if HIP_FP8_TYPE_FNUZ
  // E4M3FNUZ: bias=8, max=240 (exponent 15 is a normal binade), no negative zero, NaN=0x80
  constexpr int32_t kBias = 8;
  constexpr int32_t kMinNormExp = -7;
  constexpr int32_t kMinSubnormExp = -10;
  constexpr int32_t kMaxTopMant = 7;    // 0_1111_111 = 240.0
  constexpr uint8_t kSaturate = 0x7Fu;  // max normal = 240.0
  // An underflowing negative must flush to +0: byte 0x80 is NaN here, not -0.0.
  constexpr bool kHasNegZero = false;
#else
  // E4M3FN: bias=7, max=448, NaN=0x7F
  constexpr int32_t kBias = 7;
  constexpr int32_t kMinNormExp = -6;
  constexpr int32_t kMinSubnormExp = -9;
  constexpr int32_t kMaxTopMant = 6;    // 0_1111_110 = 448.0; 0_1111_111 is NaN
  constexpr uint8_t kSaturate = 0x7Eu;  // max normal = 448.0
  constexpr bool kHasNegZero = true;
#endif

  if (a < ldexpf(1.0f, kMinNormExp)) {
    // Subnormal (or zero): count min subnormals, ties to even. The binade under the min
    // subnormal rounds up to it past its midpoint; 8 of them are the min normal.
    const int32_t n = static_cast<int32_t>(rintf(ldexpf(a, -kMinSubnormExp)));
    if (n == 0) return kHasNegZero ? sign : 0;
    return sign | static_cast<uint8_t>(n);  // n == 8 is exponent 1, mantissa 0
  }
  int32_t exp32 = ilogbf(a);
  // 1.mmm in eighths, 8..16 after rounding to nearest even
  int32_t eighths = static_cast<int32_t>(rintf(ldexpf(a, 3 - exp32)));
  if (eighths == 16) {
    eighths = 8;
    ++exp32;
  }
  const int32_t exp8 = exp32 + kBias;
  const int32_t mant3 = eighths - 8;
  // Exponent 15 is a full normal binade; only past its last finite code does it saturate.
  if (exp8 > 15 || (exp8 == 15 && mant3 > kMaxTopMant)) return sign | kSaturate;
  return sign | static_cast<uint8_t>((exp8 << 3) | mant3);
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
