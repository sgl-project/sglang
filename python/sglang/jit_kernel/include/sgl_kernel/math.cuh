#pragma once
#include <sgl_kernel/type.cuh>

namespace device::math {

inline constexpr float log2e = 1.44269504088896340736f;
inline constexpr float loge2 = 0.693147180559945309417f;
inline constexpr float FP8_E4M3_MAX = 448.0f;
static_assert(log2e * loge2 == 1.0f, "log2e * loge2 must be 1");

template <typename T>
SGL_DEVICE T max(T a, T b) {
  return dtype_trait<T>::max(a, b);
}

template <typename T>
SGL_DEVICE T min(T a, T b) {
  return dtype_trait<T>::min(a, b);
}

template <typename T>
SGL_DEVICE T abs(T a) {
  return dtype_trait<T>::abs(a);
}

template <typename T>
SGL_DEVICE T sqrt(T a) {
  return dtype_trait<T>::sqrt(a);
}

template <typename T>
SGL_DEVICE T rsqrt(T a) {
  return dtype_trait<T>::rsqrt(a);
}

}  // namespace device::math
