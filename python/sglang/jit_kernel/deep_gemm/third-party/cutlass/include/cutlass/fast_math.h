/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#include <cmath>
#include <type_traits>
#endif
#if !defined(__QNX__)
#include CUDA_STD_HEADER(utility)
#endif
#include "cutlass/array.h"
#include "cutlass/uint128.h"
#include "cutlass/coord.h"
#include "cutlass/half.h"

/**
 * \file
 * \brief Math utilities
 */

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////
#if !defined(__QNX__)
using ::cuda::std::swap;
#else
template <typename T>
CUTLASS_HOST_DEVICE void swap(T &lhs, T &rhs) {
  T tmp = lhs;
  lhs = rhs;
  rhs = tmp;
}
#endif

/******************************************************************************
 * Static math utilities
 ******************************************************************************/

/// Mixed precision dot product
template <typename Index, typename LongIndex, int N>
CUTLASS_HOST_DEVICE LongIndex dot(
  Coord<N, Index> const &coord,
  Coord<N, LongIndex> const &stride,
  LongIndex acc = LongIndex()) {

  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < N; ++n) {
    acc += LongIndex(coord[n]) * stride[n];
  }
  return acc;
}

/**
 * Statically determine if N is a power-of-two
 */
template <int N>
struct is_pow2 {
  static bool const value = ((N & (N - 1)) == 0);
};

/**
 * Statically determine log2(N), rounded down
 */
template <int N, int CurrentVal = N, int Count = 0>
struct log2_down {
  /// Static logarithm value
  enum { value = log2_down<N, (CurrentVal >> 1), Count + 1>::value };
};

// Base case
template <int N, int Count>
struct log2_down<N, 1, Count> {
  enum { value = Count };
};

/**
 * Statically determine log2(N), rounded up
 */
template <int N, int CurrentVal = N, int Count = 0>
struct log2_up {
  /// Static logarithm value
  enum { value = log2_up<N, (CurrentVal >> 1), Count + 1>::value };
};

// Base case
template <int N, int Count>
struct log2_up<N, 1, Count> {
  enum { value = ((1 << Count) < N) ? Count + 1 : Count };
};

/**
 * Statically estimate sqrt(N) to the nearest power-of-two
 */
template <int N>
struct sqrt_est {
  enum { value = 1 << (log2_up<N>::value / 2) };
};

/**
 * For performing a constant-division with a compile-time assertion that the
 * Divisor evenly-divides the Dividend.
 */
template <int Dividend, int Divisor>
struct divide_assert {
  enum { value = Dividend / Divisor };

  static_assert((Dividend % Divisor == 0), "Not an even multiple");
};

/******************************************************************************
 * Rounding
 ******************************************************************************/

/**
 * Round dividend up to the nearest multiple of divisor
 */
template <typename dividend_t, typename divisor_t>
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
dividend_t round_nearest(dividend_t dividend, divisor_t divisor) {
  return ((dividend + divisor - 1) / divisor) * divisor;
}

template <typename value_t>
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
value_t abs_for_integer(value_t a) {
  return ((a > value_t{0}) ? a : -a);
}
/**
 * Greatest common divisor
 */
template <typename value_t>
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
value_t gcd(value_t a, value_t b) {
  for (;;) {
    if (a == value_t{0}) return cutlass::abs_for_integer(b);
    b %= a;
    if (b == value_t{0}) return cutlass::abs_for_integer(a);
    a %= b;
  }
}

/**
 * Least common multiple
 */
template <typename value_t>
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
value_t lcm(value_t a, value_t b) {
  value_t temp = cutlass::gcd(a, b);
  return (temp != value_t{0}) ? value_t(cutlass::abs_for_integer(a) / temp * cutlass::abs_for_integer(b)) : value_t{};
}

/**
 * Greatest common divisor
 */
template <typename value_t>
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
value_t gcd_cxx11(value_t a, value_t b) {
  return (a == value_t{0} || b == value_t{0}) ? cutlass::abs_for_integer(a | b) : cutlass::gcd_cxx11(b, a % b);
}

/**
 * Least common multiple
 */
template <typename value_t>
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
value_t lcm_cxx11(value_t a, value_t b) {
  return cutlass::gcd_cxx11(a, b) ? (cutlass::abs_for_integer(a) / cutlass::gcd_cxx11(a, b) *
                                    cutlass::abs_for_integer(b))
                                  : value_t{};
}

/// Returns the smallest value in the half-open range [a, a+b) that is a multiple of b
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
int round_up(int a, int b) {
  return ((a + b - 1) / b) * b;
}

/// Returns the ceiling of (a / b)
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * log2 computation, what's the
 * difference between the below codes and
 * log2_up/down codes?
 */
template <typename value_t>
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
value_t clz(value_t x) {
  for (int i = 31; i >= 0; --i) {
    if ((1 << i) & x)
      return value_t(31 - i);
  }
  return value_t(32);
}

template <typename value_t>
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
value_t find_log2(value_t x) {
  int a = int(31 - clz(x));
  a += (x & (x - 1)) != 0;  // Round up, add 1 if not a power of 2.
  return a;
}


/**
 * Find divisor, using find_log2
 */
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
void find_divisor(unsigned int& mul, unsigned int& shr, unsigned int denom) {
  if (denom == 1) {
    mul = 0;
    shr = 0;
  } else {
    unsigned int p = 31 + find_log2(denom);
    unsigned m = unsigned(((1ull << p) + unsigned(denom) - 1) / unsigned(denom));

    mul = m;
    shr = p - 32;
  }
}

/**
 * Find quotient and remainder using device-side intrinsics
 */
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
void fast_divmod(int& quo, int& rem, int src, int div, unsigned int mul, unsigned int shr) {

  #if defined(__CUDA_ARCH__)
  // Use IMUL.HI if div != 1, else simply copy the source.
  quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
  #else
  quo = int((div != 1) ? int(((int64_t)src * mul) >> 32) >> shr : src);
  #endif

  // The remainder.
  rem = src - (quo * div);
}

// For long int input
CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17
void fast_divmod(int& quo, int64_t& rem, int64_t src, int div, unsigned int mul, unsigned int shr) {

  #if defined(__CUDA_ARCH__)
  // Use IMUL.HI if div != 1, else simply copy the source.
  quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
  #else
  quo = int((div != 1) ? ((src * mul) >> 32) >> shr : src);
  #endif
  // The remainder.
  rem = src - (quo * div);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object to encapsulate the fast division+modulus operation.
///
/// This object precomputes two values used to accelerate the computation and is best used
/// when the divisor is a grid-invariant. In this case, it may be computed in host code and
/// marshalled along other kernel arguments using the 'Params' pattern.
///
/// Example:
///
///
///   int quotient, remainder, dividend, divisor;
///
///   FastDivmod divmod(divisor);
///
///   divmod(quotient, remainder, dividend);
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmod {
  using value_div_type = int;
  using value_mod_type = int64_t;
  int32_t divisor = 1;
  uint32_t multiplier = 0u;
  uint32_t shift_right = 0u;

  // Find quotient and remainder using device-side intrinsics
  CUTLASS_HOST_DEVICE
  void fast_divmod(int& quotient, int& remainder, int dividend) const {

#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if divisor != 1, else simply copy the source.
    quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
    quotient = int((divisor != 1) ? int(((int64_t)dividend * multiplier) >> 32) >> shift_right : dividend);
#endif

    // The remainder.
    remainder = dividend - (quotient * divisor);
  }

  /// For long int input
  CUTLASS_HOST_DEVICE
  void fast_divmod(int& quotient, int64_t& remainder, int64_t dividend) const {

#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if divisor != 1, else simply copy the source.
    quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
    quotient = int((divisor != 1) ? ((dividend * multiplier) >> 32) >> shift_right : dividend);
#endif
    // The remainder.
    remainder = dividend - (quotient * divisor);
  }


  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally expensive.

  constexpr FastDivmod() = default;

  CUTLASS_HOST_DEVICE
  FastDivmod(int divisor_): divisor(divisor_) {
    assert(divisor_ >= 0);
    if (divisor != 1) {
      unsigned int p = 31 + find_log2(divisor);
      unsigned m = unsigned(((1ull << p) + unsigned(divisor) - 1) / unsigned(divisor));

      multiplier = m;
      shift_right = p - 32;
    }
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  CUTLASS_HOST_DEVICE
  void operator()(int &quotient, int &remainder, int dividend) const {
    fast_divmod(quotient, remainder, dividend);
  }

  /// Computes integer division using precomputed values. This is computationally
  /// inexpensive.
  CUTLASS_HOST_DEVICE
  int div(int dividend) const {
    int quotient, remainder;
    fast_divmod(quotient, remainder, dividend);
    return quotient;
  }

  /// Alias for `div` to match the interface of FastDivmodU64
  CUTLASS_HOST_DEVICE
  int divide(int dividend) const {
    return div(dividend);
  }

  /// Computes integer division remainder using precomputed values.
  CUTLASS_HOST_DEVICE
  int rem(int dividend) const {
    int quotient, remainder;
    fast_divmod(quotient, remainder, dividend);
    return remainder;
  }

  /// Alias for `rem`
  CUTLASS_HOST_DEVICE
  int remainder(int dividend) const {
    return rem(dividend);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  ///
  /// Simply returns the quotient
  CUTLASS_HOST_DEVICE
  int divmod(int &remainder, int dividend) const {
    int quotient;
    fast_divmod(quotient, remainder, dividend);
    return quotient;
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  CUTLASS_HOST_DEVICE
  void operator()(int &quotient, int64_t &remainder, int64_t dividend) const {
    fast_divmod(quotient, remainder, dividend);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  CUTLASS_HOST_DEVICE
  int divmod(int64_t &remainder, int64_t dividend) const {
    int quotient;
    fast_divmod(quotient, remainder, dividend);
    return quotient;
  }

  /// Returns the divisor when cast to integer
  CUTLASS_HOST_DEVICE
  operator int() const { return divisor; }

};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object to encapsulate the fast division+modulus operation for 64b integer division.
///
/// This object precomputes two values used to accelerate the computation and is best used
/// when the divisor is a grid-invariant. In this case, it may be computed in host code and
/// marshalled along other kernel arguments using the 'Params' pattern.
///
/// Example:
///
///
///   uint64_t quotient, remainder, dividend, divisor;
///
///   FastDivmodU64 divmod(divisor);
///
///   divmod(quotient, remainder, dividend);
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmodU64 {

  uint64_t divisor;
  uint64_t multiplier;
  unsigned int shift_right;
  unsigned int round_up;

  //
  // Static methods
  //

  /// Computes b, where 2^b is the greatest power of two that is less than or equal to x
  CUTLASS_HOST_DEVICE
  static uint32_t integer_log2(uint64_t x) {
    uint32_t n = 0;
    while (x >>= 1) {
      ++n;
    }
    return n;
  }

  /// Default ctor
  CUTLASS_HOST_DEVICE
  FastDivmodU64(): divisor(0), multiplier(0), shift_right(0), round_up(0) { }

  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally expensive.
  CUTLASS_HOST_DEVICE
  FastDivmodU64(uint64_t divisor_): divisor(divisor_), multiplier(1), shift_right(0), round_up(0) {

    if (divisor) {
      shift_right = integer_log2(divisor);

      if ((divisor & (divisor - 1)) == 0) {
        multiplier = 0;
      }
      else {
        uint64_t power_of_two = (uint64_t(1) << shift_right);
        uint64_t multiplier_lo = uint128_t(0, power_of_two) / divisor;
        multiplier = uint128_t(power_of_two, power_of_two) / divisor;
        round_up = (multiplier_lo == multiplier ? 1 : 0);
      }
    }
  }

  /// Returns the quotient of floor(dividend / divisor)
  CUTLASS_HOST_DEVICE
  uint64_t divide(uint64_t dividend) const {
    uint64_t quotient = 0;

    #ifdef __CUDA_ARCH__
      uint64_t x = dividend;
      if (multiplier) {
        x = __umul64hi(dividend + round_up, multiplier);
      }
      quotient = (x >> shift_right);
    #else
      quotient = dividend / divisor;
    #endif

    return quotient;
  }

  /// Computes the remainder given a computed quotient and dividend
  CUTLASS_HOST_DEVICE
  uint64_t modulus(uint64_t quotient, uint64_t dividend) const {
    return dividend - quotient * divisor;
  }

  /// Returns the quotient of floor(dividend / divisor) and computes the remainder
  CUTLASS_HOST_DEVICE
  uint64_t divmod(uint64_t &remainder, uint64_t dividend) const {
    uint64_t quotient = divide(dividend);
    remainder = modulus(quotient, dividend);
    return quotient;
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  CUTLASS_HOST_DEVICE
  void operator()(uint64_t &quotient, uint64_t &remainder, uint64_t dividend) const {
    quotient = divmod(remainder, dividend);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object to encapsulate the fast division+modulus operation for 64b integer division
/// in which the divisor is a power of two.
struct FastDivmodU64Pow2 {

  uint64_t divisor;
  unsigned int shift_right;

  /// Default ctor
  CUTLASS_HOST_DEVICE
  FastDivmodU64Pow2(): divisor(0), shift_right(0) { }

  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally expensive.
  CUTLASS_HOST_DEVICE
  FastDivmodU64Pow2(uint64_t divisor_): divisor(divisor_), shift_right(FastDivmodU64::integer_log2(divisor_)) { }

  /// Returns the quotient of floor(dividend / divisor)
  CUTLASS_HOST_DEVICE
  uint64_t divide(uint64_t dividend) const {
    return dividend >> shift_right;
  }

  /// Computes the remainder given a computed quotient and dividend
  CUTLASS_HOST_DEVICE
  uint64_t modulus(uint64_t dividend) const {
    // See https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#division-modulo-operations
    return dividend & (divisor - 1);
  }

  /// Returns the quotient of floor(dividend / divisor) and computes the remainder
  CUTLASS_HOST_DEVICE
  uint64_t divmod(uint64_t &remainder, uint64_t dividend) const {
    uint64_t quotient = divide(dividend);
    remainder = modulus(dividend);
    return quotient;
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  CUTLASS_HOST_DEVICE
  void operator()(uint64_t &quotient, uint64_t &remainder, uint64_t dividend) const {
    quotient = divmod(remainder, dividend);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes the coordinate decomposition from a linear index (64-bit linear index => coord<int32_t>)
///
/// This decomposition is accelerated by the FastDivmodU64 object. It is assumed that
/// a coordinate of <Rank> indices can be decomposed by <Rank - 1> div/mod operations.
/// Note, is assumed that element divmod[0] divides by extent[1].
///
/// For example, assume 4-D coordinate (n, p, q, c) is mapped to a linear index `npqc`. This
/// can be decomposed via three divide and modulus operations:
///
///      c = npqc % C;         |  divmod[2] = FastDivmodU64(C)
///    npq = npqc / C;         |   coord[3] = c
///
///      q =  npq % Q;         |  divmod[1] = FastDivmodU64(Q)
///     np =  npq / Q;         |   coord[2] = q
///
///      p =   np % P;         |  divmod[0] = FastDivmodU64(P)
///      n =   np / P;         |   coord[1] = p
///
///                            |   coord[0] = n
///
template <int Rank>
CUTLASS_HOST_DEVICE Coord<Rank> CoordinateDecomposition(
  uint64_t linear_idx,                    ///< Linear index to decompose
  FastDivmodU64 const *divmod) {          ///< Pointer to array of Rank-1 FastDivmodU64 objects

  static_assert(Rank > 0, "CoordinateDecomposition requires Rank=1 or greater.");

  Coord<Rank> coord;

  CUTLASS_PRAGMA_UNROLL
  for (int i = Rank; i > 1; --i) {
    uint64_t remainder;
    linear_idx = divmod[i - 2].divmod(remainder, linear_idx);
    coord[i - 1] = int(remainder);
  }

  coord[0] = int(linear_idx);

  return coord;
}

/// Computes the coordinate decomposition from a linear index (32-bit linear index => coord<int32_t>)
template <int Rank>
CUTLASS_HOST_DEVICE Coord<Rank> CoordinateDecomposition(
  int linear_idx,                    ///< Linear index to decompose
  FastDivmod const *divmod) {          ///< Pointer to array of Rank-1 FastDivmodU64 objects

  static_assert(Rank > 0, "CoordinateDecomposition requires Rank=1 or greater.");

  Coord<Rank> coord;

  CUTLASS_PRAGMA_UNROLL
  for (int i = Rank; i > 1; --i) {
    int remainder;
    linear_idx = divmod[i - 2].divmod(remainder, linear_idx);
    coord[i - 1] = int(remainder);
  }

  coord[0] = int(linear_idx);

  return coord;
}

template <int Rank>
CUTLASS_HOST_DEVICE Coord<Rank> CoordinateDecompositionLittleEndian(
  uint64_t linear_idx,                    ///< Linear index to decompose
  FastDivmodU64 const *divmod) {          ///< Pointer to array of Rank-1 FastDivmodU64 objects

  static_assert(Rank > 0, "CoordinateDecomposition requires Rank=1 or greater.");

  Coord<Rank> coord;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Rank - 1; ++i) {
    uint64_t remainder;
    linear_idx = divmod[i].divmod(remainder, linear_idx);
    coord[i] = int(remainder);
  }

  coord[Rank - 1] = int(linear_idx);

  return coord;
}

/// Computes the coordinate decomposition from a linear index (32-bit linear index => coord<int32_t>)
template <int Rank>
CUTLASS_HOST_DEVICE Coord<Rank> CoordinateDecompositionLittleEndian(
  int linear_idx,                    ///< Linear index to decompose
  FastDivmod const *divmod) {          ///< Pointer to array of Rank-1 FastDivmodU64 objects

  static_assert(Rank > 0, "CoordinateDecomposition requires Rank=1 or greater.");

  Coord<Rank> coord;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Rank - 1; ++i) {
    int remainder;
    linear_idx = divmod[i].divmod(remainder, linear_idx);
    coord[i] = int(remainder);
  }

  coord[Rank - 1] = int(linear_idx);

  return coord;
}

/// Safely computes the offset of a linear index in bytes for all types
template <typename Element>
CUTLASS_HOST_DEVICE int64_t OffsetBytes(int64_t index) {

  static_assert(
    (sizeof_bits<Element>::value >= 8 && !(sizeof_bits<Element>::value % 8)) ||
    (sizeof_bits<Element>::value <  8 && !(8 % sizeof_bits<Element>::value)),
    "Size of numeric type in bits must either be divisible by 8 bits, or 8 bits must be divisible by the size.");

  if (sizeof_bits<Element>::value >= 8) {
    return index * (sizeof_bits<Element>::value / 8);
  }
  else {
    int const kElementsPerByte = ((8 / sizeof_bits<Element>::value) + ((sizeof_bits<Element>::value >= 8) ? 1 : 0));
    return index / kElementsPerByte;
  }
}

CUTLASS_HOST_DEVICE int64_t OffsetBytes(int64_t index, int64_t element_sizeof_bits) {
  if (element_sizeof_bits >= 8) {
    return index * (element_sizeof_bits / 8);
  }
  else {
    int64_t const kElementsPerByte = ((8 / element_sizeof_bits) + ((element_sizeof_bits >= 8) ? 1 : 0));
    return index / kElementsPerByte;
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// Min/Max
/////////////////////////////////////////////////////////////////////////////////////////////////

template <int A, int B>
struct Min {
  static int const kValue = (A < B) ? A : B;
};

template <int A, int B>
struct Max {
  static int const kValue = (A > B) ? A : B;
};

CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17 int const_min(int a, int b) {
    return (b < a ? b : a);
}

CUTLASS_HOST_DEVICE
CUTLASS_CONSTEXPR_IF_CXX17 int const_max(int a, int b) {
    return (b > a ? b : a);
}

template <typename T>
CUTLASS_HOST_DEVICE
T fast_min(T a, T b) {
  return (b < a ? b : a);
}

template <>
CUTLASS_HOST_DEVICE
float fast_min(float a, float b) {
  return fminf(a, b);
}

template <typename T>
CUTLASS_HOST_DEVICE
T fast_max(T a, T b) {
  return (a < b ? b : a);
}

template <>
CUTLASS_HOST_DEVICE
float fast_max(float a, float b) {
  return fmaxf(a, b);
}

CUTLASS_HOST_DEVICE
float fast_cos(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::cosf(theta);
  #else
  return std::cos(theta);
  #endif
}

CUTLASS_HOST_DEVICE
double fast_cos(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::cos(theta);
  #else
  return std::cos(theta);
  #endif
}

CUTLASS_HOST_DEVICE
float fast_sin(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::sinf(theta);
  #else
  return std::sin(theta);
  #endif
}

CUTLASS_HOST_DEVICE
double fast_sin(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::sin(theta);
  #else
  return std::sin(theta);
  #endif
}

CUTLASS_HOST_DEVICE
float fast_acos(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::acosf(theta);
  #else
  return std::acos(theta);
  #endif
}

CUTLASS_HOST_DEVICE
double fast_acos(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::acos(theta);
  #else
  return std::acos(theta);
  #endif
}

CUTLASS_HOST_DEVICE
float fast_asin(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::asinf(theta);
  #else
  return std::asin(theta);
  #endif
}

CUTLASS_HOST_DEVICE
double fast_asin(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::asin(theta);
  #else
  return std::asin(theta);
  #endif
}

CUTLASS_HOST_DEVICE
float fast_sqrt(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::sqrtf(theta);
  #else
  return std::sqrt(theta);
  #endif
}

CUTLASS_HOST_DEVICE
double fast_sqrt(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::sqrt(theta);
  #else
  return std::sqrt(theta);
  #endif
}

CUTLASS_HOST_DEVICE
float fast_exp(float x) {
  #if defined(__CUDA_ARCH__)
  return ::expf(x);
  #else
  return std::exp(x);
  #endif
}

CUTLASS_HOST_DEVICE
double fast_exp(double x) {
  #if defined(__CUDA_ARCH__)
  return ::exp(x);
  #else
  return std::exp(x);
  #endif
}

CUTLASS_HOST_DEVICE
half_t fast_exp(half_t x) {
  #if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 10) && (__CUDA_ARCH__ >= 750)
      return (half_t)(::hexp(x.to_half()));
  #else
      return (half_t)(fast_exp(float(x)));
  #endif
}

CUTLASS_HOST_DEVICE
float fast_log(float x) {
  #if defined(__CUDA_ARCH__)
  return ::logf(x);
  #else
  return std::log(x);
  #endif
}

CUTLASS_HOST_DEVICE
double fast_log(double x) {
  #if defined(__CUDA_ARCH__)
  return ::log(x);
  #else
  return std::log(x);
  #endif
}

CUTLASS_HOST_DEVICE
float fast_tanh(float x) {
  #if defined(__CUDA_ARCH__)
    #if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
      float y;
      asm volatile ( "tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
      return y;
    #else
      return ::tanhf(x);
    #endif
  #else
  return std::tanh(x);
  #endif
}

CUTLASS_HOST_DEVICE
double fast_tanh(double x) {
  #if defined(__CUDA_ARCH__)
  return ::tanh(x);
  #else
  return std::tanh(x);
  #endif
}

CUTLASS_HOST_DEVICE
half_t fast_tanh(half_t x) {
  #if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)

  asm volatile ( "tanh.approx.f16 %0, %1;" : "=h"(x.raw()) : "h"(x.raw()));
  return x;

  #else
  return half_t(fast_tanh(float(x)));
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct fast_exp_op {
  CUTLASS_HOST_DEVICE
  T operator()(T const &rhs) const {
    return fast_exp(rhs);
  }
};

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 10) && (__CUDA_ARCH__ >= 750)
template <int N>
struct fast_exp_op<Array<half_t, N>> {
  CUTLASS_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &rhs) const {

    Array<half_t, N> result;

    // use x2 specialization
    __half2 const *in  = reinterpret_cast<__half2 const *>(&rhs);
    __half2 *out = reinterpret_cast<__half2 *>(&result);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      out[i] = ::h2exp(in[i]);
    }

    // residual
    if (N % 2) {
      half_t last = rhs[N - 1];
      result[N - 1] = half_t(::hexp(last.to_half()));
    }

    return result;
  }
};
#endif // #if defined(__CUDA_ARCH__)

template <typename T, int N>
struct fast_exp_op<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {

    fast_exp_op<T> fast_op;
    Array<T, N> y;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = fast_op(rhs[i]);
    }

    return y;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct fast_tanh_op {
  CUTLASS_HOST_DEVICE
  T operator()(T const &rhs) const {
    return fast_tanh(rhs);
  }
};

#if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
template <int N>
struct fast_tanh_op<Array<half_t, N>> {
  CUTLASS_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &rhs) const {

    Array<half_t, N> result;

    // use x2 specialization
    uint32_t const *in  = reinterpret_cast<uint32_t const *>(&rhs);
    uint32_t *out = reinterpret_cast<uint32_t *>(&result);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm volatile ("tanh.approx.f16x2 %0, %1;" : "=r"(out[i]) : "r"(in[i]));
    }

    // residual
    if (N % 2) {
      uint16_t const *in = reinterpret_cast<uint16_t const *>(&rhs);
      uint16_t *out = reinterpret_cast<uint16_t *>(&result);
      asm volatile ("tanh.approx.f16 %0, %1;" : "=h"(out[N - 1]) : "h"(in[N - 1]));
    }

    return result;
  }
};
#endif // #if defined(__CUDA_ARCH__)

template <typename T, int N>
struct fast_tanh_op<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {

    fast_tanh_op<T> fast_op;
    Array<T, N> y;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = fast_op(rhs[i]);
    }

    return y;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Absolute value function
template <typename T>
CUTLASS_HOST_DEVICE
T absolute_value(T x) {
  if (x < T()) {
    return -x;
  }
  return x;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

