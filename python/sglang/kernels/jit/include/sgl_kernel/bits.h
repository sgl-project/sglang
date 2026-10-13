#pragma once
#include <sgl_kernel/cxx_compat.h>

#if SGL_USE_BITOPS
#include <bit>
#endif
#if SGL_USE_CONCEPTS
#include <concepts>
#endif
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace host {

#if SGL_USE_CONCEPTS
template <std::unsigned_integral T>
#else
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
#endif
inline constexpr bool is_pow2(T x) {
#if SGL_USE_BITOPS
  return std::has_single_bit(x);
#else
  return x != 0 && (x & (x - 1)) == 0;
#endif
}

/// \brief `floor(log2(x))`; -1 for `x == 0`.
#if SGL_USE_CONCEPTS
template <std::unsigned_integral T>
#else
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
#endif
inline constexpr int32_t log2_floor(T x) {
  if (x == 0) return -1;
#if SGL_USE_BITOPS
  return std::bit_width(x) - 1;
#else
  int32_t result = 0;
  while (x >>= 1) {
    ++result;
  }
  return result;
#endif
}

/// \brief `ceil(log2(x))`; -1 for `x == 0`.
#if SGL_USE_CONCEPTS
template <std::unsigned_integral T>
#else
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
#endif
inline constexpr int32_t log2_ceil(T x) {
  if (x == 0) return -1;
#if SGL_USE_BITOPS
  return std::bit_width(x - 1);
#else
  return log2_floor(static_cast<T>(x - 1)) + 1;
#endif
}

#if SGL_USE_CONCEPTS
template <std::unsigned_integral T>
#else
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
#endif
inline constexpr T round_up_pow2(T x) {
#if SGL_USE_BITOPS
  return std::bit_ceil(x);
#else
  if (x <= 1) return 1;
  return static_cast<T>(T{1} << log2_ceil(x));
#endif
}

#if SGL_USE_CONCEPTS
template <std::unsigned_integral T>
#else
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
#endif
inline constexpr T round_down_pow2(T x) {
#if SGL_USE_BITOPS
  return std::bit_floor(x);
#else
  return x == 0 ? 0 : static_cast<T>(T{1} << log2_floor(x));
#endif
}

}  // namespace host

}  // namespace sglang
