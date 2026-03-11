/// \file math.cuh
/// \brief Device-side math helper functions and constants.
///
/// Provides type-generic wrappers around CUDA math intrinsics by
/// dispatching through `dtype_trait<T>`. All functions are forced-inline
/// device functions.

#pragma once
#include <sgl_kernel/type.cuh>

#include <cmath>

namespace device::math {

/// \brief Constant: log2(e)
inline constexpr float log2e = 1.44269504088896340736f;
/// \brief Constant: ln(2)
inline constexpr float loge2 = 0.693147180559945309417f;
/// \brief Maximum representable value for FP8 E4M3 format.
inline constexpr float FP8_E4M3_MAX = 448.0f;
static_assert(log2e * loge2 == 1.0f, "log2e * loge2 must be 1");

/// \brief Returns the larger of `a` and `b`.
template <typename T>
SGL_DEVICE T max(T a, T b) {
  return dtype_trait<T>::max(a, b);
}

/// \brief Returns the smaller of `a` and `b`.
template <typename T>
SGL_DEVICE T min(T a, T b) {
  return dtype_trait<T>::min(a, b);
}

/// \brief Returns the absolute value of `a`.
template <typename T>
SGL_DEVICE T abs(T a) {
  return dtype_trait<T>::abs(a);
}

/// \brief Returns the square root of `a`.
template <typename T>
SGL_DEVICE T sqrt(T a) {
  return dtype_trait<T>::sqrt(a);
}

/// \brief Returns the reciprocal square root of `a` (i.e. 1 / sqrt(a)).
template <typename T>
SGL_DEVICE T rsqrt(T a) {
  return dtype_trait<T>::rsqrt(a);
}

/// \brief Returns e^a.
template <typename T>
SGL_DEVICE T exp(T a) {
  return dtype_trait<T>::exp(a);
}

/// \brief Returns sin(a).
template <typename T>
SGL_DEVICE T sin(T a) {
  return dtype_trait<T>::sin(a);
}

/// \brief Returns cos(a).
template <typename T>
SGL_DEVICE T cos(T a) {
  return dtype_trait<T>::cos(a);
}

}  // namespace device::math
