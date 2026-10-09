#pragma once
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace host {

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
inline constexpr bool is_pow2(T x) {
  return x != 0 && (x & (x - 1)) == 0;
}

/// \brief `floor(log2(x))`; -1 for `x == 0`.
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
inline constexpr int32_t log2_floor(T x) {
  if (x == 0) return -1;
  int32_t result = 0;
  while (x >>= 1) {
    ++result;
  }
  return result;
}

/// \brief `ceil(log2(x))`; -1 for `x == 0`.
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
inline constexpr int32_t log2_ceil(T x) {
  if (x == 0) return -1;
  return log2_floor(static_cast<T>(x - 1)) + 1;
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
inline constexpr T round_up_pow2(T x) {
  if (x <= 1) return 1;
  return static_cast<T>(T{1} << log2_ceil(x));
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
inline constexpr T round_down_pow2(T x) {
  return x == 0 ? 0 : static_cast<T>(T{1} << log2_floor(x));
}

}  // namespace host

}  // namespace sglang
