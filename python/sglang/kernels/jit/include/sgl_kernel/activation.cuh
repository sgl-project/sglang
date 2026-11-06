/// \file activation.cuh
/// \brief FP32 activation functions shared by JIT kernels.
///
/// Every function takes `kFastMath` (default true):
///
/// - `kFastMath = true`: approximate PTX without slow-path branches, e.g. sigmoid
///   as `rcp.approx(1 + ex2.approx(-x * log2e))` (relative error around 1e-6).
///   `gelu_tanh` uses one `tanh.approx`: its relative error grows for negative x,
///   so use it only for outputs stored in bf16 / fp8.
/// - `kFastMath = false`: ATen's formula and operation order with libdevice
///   transcendentals and IEEE division, to reproduce eager PyTorch as closely as
///   the compiler's FMA contraction allows.
///
/// Neither mode depends on `--use_fast_math`: the fast mode is written with PTX,
/// and the precise mode calls libdevice symbols, which that flag does not rewrite.

#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>

#include <cmath>

#ifndef USE_ROCM
extern "C" __device__ float __nv_expf(float);
extern "C" __device__ float __nv_log1pf(float);
extern "C" __device__ float __nv_tanhf(float);
extern "C" __device__ float __nv_erff(float);
#endif

namespace sglang {

namespace device::activation {

namespace ptx {

/// \brief ex2.approx.ftz.f32: one MUFU.EX2.
SGL_DEVICE float ex2_approx_ftz(float x) {
#ifndef USE_ROCM
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return __builtin_amdgcn_exp2f(x);  // what hip-clang's own __expf is built on
#endif
}

/// \brief rcp.approx.ftz.f32: one MUFU.RCP, at most 1 ulp, results below 2^-126 flush to 0.
SGL_DEVICE float rcp_approx_ftz(float x) {
#ifndef USE_ROCM
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return __frcp_rn(x);
#endif
}

/// \brief tanh.approx.f32: one MUFU.TANH.
SGL_DEVICE float tanh_approx(float x) {
#ifndef USE_ROCM
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return tanhf(x);
#endif
}

/// \brief sqrt.approx.ftz.f32: one MUFU.SQRT.
SGL_DEVICE float sqrt_approx_ftz(float x) {
#ifndef USE_ROCM
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return sqrtf(x);
#endif
}

}  // namespace ptx

namespace libdevice {

SGL_DEVICE float exp(float x) {
#ifndef USE_ROCM
  return __nv_expf(x);
#else
  return expf(x);
#endif
}

SGL_DEVICE float log1p(float x) {
#ifndef USE_ROCM
  return __nv_log1pf(x);
#else
  return log1pf(x);
#endif
}

SGL_DEVICE float tanh(float x) {
#ifndef USE_ROCM
  return __nv_tanhf(x);
#else
  return tanhf(x);
#endif
}

SGL_DEVICE float erf(float x) {
#ifndef USE_ROCM
  return __nv_erff(x);
#else
  return erff(x);
#endif
}

}  // namespace libdevice

/// \brief sigmoid(x) = 1 / (1 + e^-x).
template <bool kFastMath = true>
SGL_DEVICE float sigmoid(float x) {
  if constexpr (kFastMath) {
    // The divisor is >= 1, so the ftz reciprocal only flushes results below 2^-126.
    return ptx::rcp_approx_ftz(1.0f + ptx::ex2_approx_ftz(x * -math::log2e));
  } else {
    return __fdiv_rn(1.0f, 1.0f + libdevice::exp(-x));
  }
}

/// \brief silu(x) = x * sigmoid(x).
template <bool kFastMath = true>
SGL_DEVICE float silu(float x) {
  if constexpr (kFastMath) {
    return x * sigmoid(x);
  } else {
    return __fdiv_rn(x, 1.0f + libdevice::exp(-x));
  }
}

/// \brief Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
template <bool kFastMath = true>
SGL_DEVICE float gelu(float x) {
  constexpr float kAlpha = 0.70710678118654752f;  // sqrt(1 / 2)
  const auto erf = libdevice::erf(x * kAlpha);
  if constexpr (kFastMath) {
    const auto h = 0.5f * x;
    return h + h * erf;
  } else {
    return x * 0.5f * (1.0f + erf);
  }
}

/// \brief Tanh-approximated GELU: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
template <bool kFastMath = true>
SGL_DEVICE float gelu_tanh(float x) {
  constexpr float kBeta = 0.79788456080286536f;  // sqrt(2 / pi)
  constexpr float kKappa = 0.044715f;
  if constexpr (kFastMath) {
    // One MUFU; 1 + tanh cancels for negative x, which only matters below bf16 rounding.
    const auto cdf = 0.5f * (1.0f + ptx::tanh_approx(kBeta * (x + kKappa * x * x * x)));
    return x * cdf;
  } else {
    const auto inner = kBeta * (x + kKappa * (x * x * x));
    return 0.5f * x * (1.0f + libdevice::tanh(inner));
  }
}

/// \brief softplus(x) = log(1 + e^x).
template <bool kFastMath = true>
SGL_DEVICE float softplus(float x) {
  if constexpr (kFastMath) {
    // max(x, 0) + log1p(e^-|x|): no overflow and no cutoff; a cutoff select would
    // branch around log1pf.
    return fmaxf(x, 0.0f) + libdevice::log1p(ptx::ex2_approx_ftz(fabsf(x) * -math::log2e));
  } else {
    // ATen: x > 20 ? x : log1p(exp(x)). Computed unconditionally so the select
    // stays a select; exp overflows to inf only where x is selected.
    const auto r = libdevice::log1p(libdevice::exp(x));
    return x > 20.0f ? x : r;
  }
}

/// \brief sqrt(softplus(x)).
template <bool kFastMath = true>
SGL_DEVICE float sqrt_softplus(float x) {
  if constexpr (kFastMath) {
    // sqrt(max(x, 0) + log1p(exp(-|x|))), branch-free: w = exp(-|x| / 2) = 2^n * 2^f with
    // a degree-7 polynomial for 2^f, z = w^2, log1p(z) = 2 atanh(z / (2 + z)) as an odd
    // series. For x <= 0 the result is w * sqrt(log1p(z) / z), which stays normal down to
    // x = -174 where z itself underflows; below about -176, where w underflows, it is 0.
    constexpr float kHalfL2e = 0.5f * math::log2e;
    constexpr float kHalfL2eLo = 0.5f * 1.925963033500011e-08f;  // (log2(e) - math::log2e) / 2
    // Bounds n for the exponent add below (also maps -inf and NaN to finite values).
    const auto nx = fmaxf(-fabsf(x), -500.0f);
    // n = rint(nx * kHalfL2e) via 1.5 * 2^23, whose low mantissa bits then hold n.
    const auto n_shift = nx * kHalfL2e + 12582912.0f;
    const auto n = n_shift - 12582912.0f;
    // f = nx * log2(e) / 2 - n: the fma rounds once, |f| <= 1/2.
    const auto f = fmaf(nx, kHalfL2eLo, fmaf(nx, kHalfL2e, -n));
    // 2^f = e^(f ln2), Taylor to degree 7 (error < 5e-9 for |f| <= 1/2).
    auto e = 1.0f / 5040.0f;
    e = fmaf(e, f * math::loge2, 1.0f / 720.0f);
    e = fmaf(e, f * math::loge2, 1.0f / 120.0f);
    e = fmaf(e, f * math::loge2, 1.0f / 24.0f);
    e = fmaf(e, f * math::loge2, 1.0f / 6.0f);
    e = fmaf(e, f * math::loge2, 0.5f);
    e = fmaf(e, f * math::loge2, 1.0f);
    e = fmaf(e, f * math::loge2, 1.0f);
    // w = e * 2^n by an exponent add (1.5 * 2^23 << 23 wraps to 0). When 2^n underflows
    // (n >= -181 here), the exponent borrows into the sign: a negative value or NaN, which
    // fmaxf turns into 0.
    const auto w = fmaxf(__int_as_float(__float_as_int(e) + (__float_as_int(n_shift) << 23)), 0.0f);
    const auto z = w * w;
    // 1 / (2 + z): rcp.approx and one Newton step, ~0.5 ulp.
    const auto d = 2.0f + z;
    const auto inv0 = ptx::rcp_approx_ftz(d);
    const auto inv = fmaf(inv0, fmaf(-d, inv0, 1.0f), inv0);
    const auto s = z * inv;
    const auto s2 = s * s;
    auto poly = 1.0f / 13.0f;
    poly = fmaf(poly, s2, 1.0f / 11.0f);
    poly = fmaf(poly, s2, 1.0f / 9.0f);
    poly = fmaf(poly, s2, 1.0f / 7.0f);
    poly = fmaf(poly, s2, 1.0f / 5.0f);
    poly = fmaf(poly, s2, 1.0f / 3.0f);
    // r = log1p(z) / z = 2 / (2 + z) * (1 + s^2 poly).
    const auto two_inv = 2.0f * inv;
    const auto r = fmaf(two_inv * s2, poly, two_inv);
    // NaN takes the x > 0 side and stays NaN. Selecting before the MUFU keeps it
    // unconditional; a select after it makes nvcc branch around the MUFU.
    const bool pos = !(x <= 0.0f);
    return ptx::sqrt_approx_ftz(pos ? fmaf(z, r, x) : r) * (pos ? 1.0f : w);
  } else {
    return __fsqrt_rn(softplus<false>(x));
  }
}

/// \brief relu2(x) = max(x, 0)^2. No transcendental, so both modes are equal.
template <bool kFastMath = true>
SGL_DEVICE float relu2(float x) {
  const auto r = fmaxf(x, 0.0f);
  return r * r;
}

}  // namespace device::activation

}  // namespace sglang
