#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#ifndef USE_ROCM
#include <cuda_fp8.h>
#endif

// Small helpers shared by the DeepSeek-V4 FP8/UE8M0 quantization kernels
// (silu_and_mul_masked_post_quant, store, mega_moe_pre_dispatch, ...).
// All functions are `SGL_DEVICE` (= `__forceinline__ __device__`) so
// including this header in multiple translation units is ODR-safe.

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
// Software float -> FP8 E4M3 conversion for ROCm/HIP.
// Supports both E4M3FN (MI350X, gfx950) and E4M3FNUZ (MI300X, gfx942).
SGL_DEVICE uint8_t cvt_float_to_fp8_e4m3(float val) {
  val = fp8_e4m3_clip(val);
  if (val == 0.0f) return 0;

  uint32_t f32 = __float_as_uint(val);
  uint8_t sign = static_cast<uint8_t>((f32 >> 31) << 7);
  int32_t exp32 = static_cast<int32_t>((f32 >> 23) & 0xFF) - 127;
  uint32_t mant23 = f32 & 0x7FFFFF;

#if HIP_FP8_TYPE_FNUZ
  // E4M3FNUZ: bias=8, max=240, no negative zero, NaN=0x80
  constexpr int32_t kBias = 8;
  constexpr int32_t kMaxExp = 15;
  constexpr int32_t kMinSubnormExp = -10;  // min subnormal exponent
  constexpr int32_t kMinNormExp = -7;      // min normal exponent
  constexpr uint8_t kSaturate = 0x7Fu;     // max normal = 0_1111_111 = 240.0
#else
  // E4M3FN: bias=7, max=448, NaN=0x7F
  constexpr int32_t kBias = 7;
  constexpr int32_t kMaxExp = 15;
  constexpr int32_t kMinSubnormExp = -9;
  constexpr int32_t kMinNormExp = -6;
  constexpr uint8_t kSaturate = 0x7Eu;  // max normal = 0_1111_110 = 448.0
#endif

  int32_t exp8;
  uint8_t mant3;

  if (exp32 < kMinSubnormExp) {
    return sign;
  } else if (exp32 < kMinNormExp) {
    // Subnormal range
    int32_t shift = -(kBias - 1) - exp32;  // 1..3
    uint32_t subnorm_mant = (0x800000 | mant23) >> (shift + 20);
    uint32_t round_bit = ((0x800000 | mant23) >> (shift + 19)) & 1;
    subnorm_mant += round_bit;
    mant3 = static_cast<uint8_t>(subnorm_mant & 0x07);
    exp8 = 0;
    if (subnorm_mant > 7) {
      exp8 = 1;
      mant3 = 0;
    }
  } else {
    exp8 = exp32 + kBias;
    mant3 = static_cast<uint8_t>(mant23 >> 20);
    uint32_t round_bit = (mant23 >> 19) & 1;
    mant3 += round_bit;
    if (mant3 > 7) {
      mant3 = 0;
      exp8++;
    }
    if (exp8 >= kMaxExp) return sign | kSaturate;
  }
  return sign | (static_cast<uint8_t>(exp8) << 3) | mant3;
}

// Pack two fp32 values into a single fp8x2_e4m3 (uint16_t on HIP).
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  uint8_t x8 = cvt_float_to_fp8_e4m3(x);
  uint8_t y8 = cvt_float_to_fp8_e4m3(y);
  return static_cast<uint16_t>(x8) | (static_cast<uint16_t>(y8) << 8);
}
#endif

}  // namespace deepseek_v4::fp8
