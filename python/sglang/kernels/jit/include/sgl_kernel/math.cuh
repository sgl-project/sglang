/// \file math.cuh
/// \brief Device-side math helper functions and constants.
///
/// Provides type-generic wrappers around CUDA math intrinsics by
/// dispatching through `DTypeTrait<T>`. All functions are forced-inline
/// device functions.

#pragma once
#include <sgl_kernel/type.cuh>

namespace sglang {

namespace device::math {

/// \brief Constant: log2(e)
inline constexpr float log2e = 1.44269504088896340736f;
/// \brief Constant: ln(2)
inline constexpr float loge2 = 0.693147180559945309417f;
/// \brief Maximum representable value for FP8 E4M3 format.
/// Arch-aware: 448 on CUDA / AMD OCP e4m3fn (gfx950), 224 on AMD e4m3fnuz
/// (gfx942). Mirrors kFP8E4M3Max so fp8 quant scale divisors and clamps in
/// the dsv4 compute path (indexer Q-quant, MoE silu+mul / dispatch quant,
/// GEMM per-tensor quant) do not over-saturate fnuz hardware.
inline constexpr float FP8_E4M3_MAX = kFP8E4M3Max;
static_assert(log2e * loge2 == 1.0f, "log2e * loge2 must be 1");

/// \brief Returns the larger of `a` and `b`.
template <typename T>
SGL_DEVICE T max(T a, T b) {
  return DTypeTrait<T>::max(a, b);
}

/// \brief Returns the smaller of `a` and `b`.
template <typename T>
SGL_DEVICE T min(T a, T b) {
  return DTypeTrait<T>::min(a, b);
}

/// \brief Returns the absolute value of `a`.
template <typename T>
SGL_DEVICE T abs(T a) {
  return DTypeTrait<T>::abs(a);
}

/// \brief Returns the square root of `a`.
template <typename T>
SGL_DEVICE T sqrt(T a) {
  return DTypeTrait<T>::sqrt(a);
}

/// \brief Returns the reciprocal square root of `a` (i.e. 1 / sqrt(a)).
template <typename T>
SGL_DEVICE T rsqrt(T a) {
  return DTypeTrait<T>::rsqrt(a);
}

/// \brief Returns e^a.
template <typename T>
SGL_DEVICE T exp(T a) {
  return DTypeTrait<T>::exp(a);
}

/// \brief Fast approximate reciprocal for FP32 device code.
///
/// One `MUFU.RCP`. No intrinsic reaches that: `1.f / x` and `__frcp_rn` are correctly
/// rounded, so they expand to a Newton step plus a slow-path `CALL`, and `__fdividef`
/// still carries an overflow test and a branch. Accurate to about 1 ulp, flushes
/// denormals to zero.
SGL_DEVICE float rcp_fast(float x) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

/// \brief Fast approximate sigmoid for FP32 device code.
/// \tparam kFastRcp Take the reciprocal with `rcp_fast` rather than a division.
template <bool kFastRcp = false>
SGL_DEVICE float sigmoid_fast(float x) {
  const float denominator = 1.0f + __expf(-x);
  if constexpr (kFastRcp) {
    return rcp_fast(denominator);
  } else {
    return 1.0f / denominator;
  }
}

/// \brief Fast approximate SiLU for FP32 device code.
SGL_DEVICE float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

/// \brief Fast approximate softplus for FP32 device code.
///
/// Values above 20 use the asymptotic result directly, avoiding overflow and
/// an unnecessary exponential while matching common softplus kernels.
SGL_DEVICE float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

/// \brief Returns sin(a).
template <typename T>
SGL_DEVICE T sin(T a) {
  return DTypeTrait<T>::sin(a);
}

/// \brief Returns cos(a).
template <typename T>
SGL_DEVICE T cos(T a) {
  return DTypeTrait<T>::cos(a);
}

// bf16 x bf16 -> fp32 fused multiply-add The mixed-precision PTX
// instruction saves the explicit converts; the fallback is bit-identical (the
// bf16 -> f32 conversion is exact, both round once). Shared by tiny_gemm,
// gemm_ag and ar_fusion.
SGL_DEVICE float fma_f32_bf16(bf16_t a, bf16_t b, float c) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
  const uint16_t a_bits = __bfloat16_as_ushort(a);
  const uint16_t b_bits = __bfloat16_as_ushort(b);
  float result;
  asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(result) : "h"(a_bits), "h"(b_bits), "f"(c));
  return result;
#else
  return fmaf(cast<fp32_t>(a), cast<fp32_t>(b), c);
#endif
}

SGL_DEVICE float fma_chain2_f32_bf16(bf16x2_t a, bf16x2_t b, float c) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
  return fma_f32_bf16(a.x, b.x, fma_f32_bf16(a.y, b.y, c));
#else
  const auto [a0, a1] = cast<fp32x2_t>(a);
  const auto [b0, b1] = cast<fp32x2_t>(b);
  return fmaf(a0, b0, fmaf(a1, b1, c));
#endif
}

SGL_DEVICE float2 fma_f32x2_bf16x2(bf16x2_t a, bf16x2_t b, float2 c) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
  return {fma_f32_bf16(a.x, b.x, c.x), fma_f32_bf16(a.y, b.y, c.y)};
#else
  const auto [a0, a1] = cast<fp32x2_t>(a);
  const auto [b0, b1] = cast<fp32x2_t>(b);
  return {fmaf(a0, b0, c.x), fmaf(a1, b1, c.y)};
#endif
}

// bf16 x fp32 -> fp32 fused multiply-add: same one-rounding contract as the
// overload above (the bf16 -> f32 convert is exact).
SGL_DEVICE float fma_f32_bf16(bf16_t a, float b, float c) {
  return fmaf(cast<fp32_t>(a), b, c);
}

SGL_DEVICE float2 fma_f32x2(float2 a, float2 b, float2 c) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
  const auto a_bits = reinterpret_cast<const uint64_t&>(a);
  const auto b_bits = reinterpret_cast<const uint64_t&>(b);
  const auto c_bits = reinterpret_cast<const uint64_t&>(c);
  uint64_t d_bits;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d_bits) : "l"(a_bits), "l"(b_bits), "l"(c_bits));
  return reinterpret_cast<float2&>(d_bits);
#else
  return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}

/**
 * \brief Fast uint32 division and modulo by a compile-time constant divisor.
 *
 * This struct is only exact when the divisor is less than `2^31`. Larger dividends
 * will lead to incorrect results. Be careful when using this struct.
 */
struct fast_mod_div_u32_t {
 public:
  explicit constexpr fast_mod_div_u32_t(uint32_t d) : m_divisor(d) {
    if (d > 1) {
      const uint32_t log2_ceil = 32 - std::countl_zero(d - 1);
      const uint32_t p = 31 + log2_ceil;
      m_magic = static_cast<uint32_t>(((uint64_t{1} << p) + d - 1) / d);
      m_shift = p - 32;
    } else {
      m_magic = 0;
      m_shift = 0;
    }
  }

  template <typename T>
  SGL_DEVICE friend uint32_t operator/(T n, const fast_mod_div_u32_t& d) {
    static_assert(std::is_unsigned_v<T> && sizeof(T) <= sizeof(uint32_t));
    return d.m_divisor == 1 ? n : __umulhi(n, d.m_magic) >> d.m_shift;
  }

  template <typename T>
  SGL_DEVICE friend uint32_t operator%(T n, const fast_mod_div_u32_t& d) {
    static_assert(std::is_unsigned_v<T> && sizeof(T) <= sizeof(uint32_t));
    return n - (n / d) * d.m_divisor;
  }

  constexpr operator uint32_t() const {
    return m_divisor;
  }

 private:
  uint32_t m_divisor;
  uint32_t m_magic;
  uint32_t m_shift;
};

template <typename T>
SGL_DEVICE uint2 div(T n, const fast_mod_div_u32_t& d) {
  return {n / d, n % d};
}

}  // namespace device::math

}  // namespace sglang
