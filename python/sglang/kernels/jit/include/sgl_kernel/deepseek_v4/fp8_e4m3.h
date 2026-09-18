#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SGL_FP8_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define SGL_FP8_HOST_DEVICE inline
#endif

namespace sglang::deepseek_v4::fp8 {

// Convert IEEE binary32 bits to E4M3 with round-to-nearest, ties-to-even.
// SATFINITE: finite overflow and infinities saturate, NaNs remain NaNs (FN
// preserves their sign). FN has signed zero, bias 7, max 448 (0x7e); FNUZ has
// unsigned zero, bias 8, max 240 (0x7f), and its only NaN is 0x80.
// This is the format conversion, not the caller's quantization clamp: the
// DeepSeek-V4 JIT path currently clamps FNUZ to 224 before calling this helper.
// Only integer arithmetic is used, so this exact device implementation can
// also be tested with an ordinary host C++ compiler (no CUDA/HIP dependency).
template <bool kFnuz>
SGL_FP8_HOST_DEVICE uint8_t f32_to_fp8_e4m3_bits(uint32_t bits) {
  constexpr int32_t kBias = kFnuz ? 8 : 7;
  constexpr uint8_t kMaxCode = kFnuz ? 0x7f : 0x7e;
  constexpr uint32_t kMaxBits = kFnuz ? 0x43700000u : 0x43e00000u;
  const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return kFnuz ? 0x80u : (sign | 0x7fu);
  if (magnitude >= kMaxBits) return sign | kMaxCode;

  const int32_t exp8 = static_cast<int32_t>(magnitude >> 23) - 127 + kBias;
  uint32_t significand = magnitude & 0x7fffffu;
  uint32_t base = 0;
  int32_t shift = 20;
  if (exp8 <= 0) {
    shift = 21 - exp8;
    // Below half the smallest FP8 subnormal, including all FP32 subnormals.
    // Bound the shift BEFORE shifting, avoiding undefined shifts by >= 32.
    if (shift > 24) return kFnuz ? 0 : sign;
    significand |= 0x800000u;
  } else {
    // Exponent 15 contains finite values in BOTH formats, not just saturation.
    base = static_cast<uint32_t>(exp8) << 3;
  }

  uint32_t rounded = significand >> shift;
  const uint32_t remainder = significand & ((1u << shift) - 1u);
  const uint32_t halfway = 1u << (shift - 1);
  rounded += (remainder > halfway) || ((remainder == halfway) && (rounded & 1u));
  // Add, rather than masking to three bits: carry may cross from subnormal to
  // normal, or into the next exponent. Pre-saturation bounds the result.
  const uint8_t code = static_cast<uint8_t>(base + rounded);
  if (kFnuz && code == 0) return 0;  // Never turn negative underflow into NaN.
  return sign | code;
}

}  // namespace sglang::deepseek_v4::fp8

#undef SGL_FP8_HOST_DEVICE
