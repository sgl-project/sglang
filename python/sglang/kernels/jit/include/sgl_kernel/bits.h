#pragma once
#include <bit>
#include <concepts>
#include <cstdint>

namespace sglang {

namespace host {

template <std::unsigned_integral T>
inline constexpr bool is_pow2(T x) {
  return std::has_single_bit(x);
}

/// \brief `floor(log2(x))`; -1 for `x == 0`.
template <std::unsigned_integral T>
inline constexpr int32_t log2_floor(T x) {
  if (x == 0) return -1;
  return std::bit_width(x) - 1;
}

/// \brief `ceil(log2(x))`; -1 for `x == 0`.
template <std::unsigned_integral T>
inline constexpr int32_t log2_ceil(T x) {
  if (x == 0) return -1;
  return std::bit_width(x - 1);
}

template <std::unsigned_integral T>
inline constexpr T round_up_pow2(T x) {
  return std::bit_ceil(x);
}

template <std::unsigned_integral T>
inline constexpr T round_down_pow2(T x) {
  return std::bit_floor(x);
}

}  // namespace host

}  // namespace sglang
