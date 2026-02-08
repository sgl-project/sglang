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
/*!
    \file
    \brief Defines a class for using integer types smaller than one byte in host or
      device code.
*/

#pragma once
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#endif

#include "cutlass/numeric_size.h"
#include "cutlass/platform/platform.h"

namespace cutlass {

template <int Bits, bool Signed = true>
struct integer_subbyte {
  using Storage = uint8_t;

  static_assert(Bits <= 8*sizeof(Storage), "Require a subbyte of bits in integer_subbyte");

  // "External type"; the integer type for which
  // integer_subbyte has a conversion-to operator
  using xint_t = typename cutlass::platform::conditional<Signed, int, unsigned>::type;

  // Bitmask for truncation from larger integers
  static constexpr Storage bits_mask_ = Storage(Storage(-1) >> (8 - Bits));
  // Bitmask for the sign bit
  static constexpr Storage sign_mask_ = Storage((Signed ? 1 : 0) << (Bits - 1));

  // Where the bits are stored
  Storage storage;

  // Default construction does NOT zero-initialize
  integer_subbyte() = default;

  // Implicit conversion is DEPRECATED.
  // Please use one of the two explicit constructors below.
  template<class T,
    class Enable = cutlass::platform::enable_if_t<cutlass::platform::is_convertible_v<T, int>>
  >
#if !defined(CUTLASS_EXTRA_WARNINGS)
  [[deprecated("Implicit conversion is deprecated; please use explicit construction instead")]]
#endif
  CUTLASS_HOST_DEVICE
  integer_subbyte(T value)
      : integer_subbyte(static_cast<xint_t>(value)) {}

  CUTLASS_HOST_DEVICE
  integer_subbyte(float value)
      : integer_subbyte(static_cast<xint_t>(value)) {}

  // CUTLASS code commonly converts both signed and unsigned integers
  // into integer_subbyte, so the class provides both explicit
  // conversions.

  // Precondition: If the external type is unsigned int, then value
  // fits in unsigned int (is nonnegative).
  CUTLASS_HOST_DEVICE explicit
  integer_subbyte(int value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_)
  {
    if constexpr (Signed) {
      [[maybe_unused]] constexpr int lower_bound = -(1 << (Bits - 1));
      [[maybe_unused]] constexpr int upper_bound = (1 << (Bits - 1)) - 1;
      assert(value >= lower_bound);
      assert(value <= upper_bound);
    }
    else {
      [[maybe_unused]] constexpr unsigned upper_bound = 1u << Bits;
      assert(value >= 0);
      assert(value < static_cast<int>(upper_bound));
    }
  }

  // Precondition: If the external type is (signed) int, then value
  // fits in int.
  CUTLASS_HOST_DEVICE explicit
  integer_subbyte(unsigned value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_)
  {
    if constexpr (Signed) {
      [[maybe_unused]] constexpr int lower_bound = -(1 << (Bits - 1));
      [[maybe_unused]] constexpr int upper_bound = (1 << (Bits - 1)) - 1;
      assert(value >= lower_bound);
      assert(value <= upper_bound);
    }
    else {
      [[maybe_unused]] constexpr unsigned upper_bound = 1u << Bits;
      assert(value < upper_bound);
    }
  }

  CUTLASS_HOST_DEVICE explicit
  integer_subbyte(uint8_t value)
    : integer_subbyte(static_cast<unsigned>(value)) {}

  // Convert to the "external" integer type (int or unsigned)
  CUTLASS_HOST_DEVICE
  operator xint_t() const {
    if (sign_mask_ & storage) {  // Sign extend
      return xint_t(storage) | ~xint_t(bits_mask_);
    } else {
      return xint_t(storage);
    }
  }

  CUTLASS_HOST_DEVICE
  bool operator==(integer_subbyte const& rhs) const {
    return storage == rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  bool operator!=(integer_subbyte const& rhs) const {
    return storage != rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  bool operator<(integer_subbyte const& rhs) const {
    if ((sign_mask_ & storage) == (sign_mask_ & rhs.storage)) {
      // If both *this and rhs have the same sign, compare storage directly.
      return storage < rhs.storage;
    }
    else {
      // If *this and rhs don't have the same sign,
      // then return whether *this is negative.
      return sign_mask_ & storage;
    }
  }

  CUTLASS_HOST_DEVICE
  bool operator<=(integer_subbyte const& rhs) const {
    if ((sign_mask_ & storage) == (sign_mask_ & rhs.storage)) {
      // If both *this and rhs have the same sign, compare storage directly.
      return storage <= rhs.storage;
    }
    else {
      // If *this and rhs don't have the same sign,
      // then return whether *this is negative.
      return sign_mask_ & storage;
    }
  }

  CUTLASS_HOST_DEVICE
  bool operator>=(integer_subbyte const& rhs) const {
    return !(*this < rhs);
  }

  CUTLASS_HOST_DEVICE
  bool operator>(integer_subbyte const& rhs) const {
    return !(*this <= rhs);
  }

  CUTLASS_HOST_DEVICE friend integer_subbyte
  conj(integer_subbyte const& x) {
    return x;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-bit binary type
using bin1_t = bool;

/// 1-bit Unsigned integer type
using uint1b_t = integer_subbyte<1, false>;

/// 2-bit Integer type
using int2b_t = integer_subbyte<2, true>;

/// 2-bit Unsigned integer type
using uint2b_t = integer_subbyte<2, false>;

/// 3-bit Integer type
using int3b_t = integer_subbyte<3, true>;

/// 3-bit Unsigned integer type
using uint3b_t = integer_subbyte<3, false>;

/// 4-bit Integer type
using int4b_t = integer_subbyte<4, true>;

/// 4-bit Unsigned integer type
using uint4b_t = integer_subbyte<4, false>;

/// 6-bit integer type
using int6b_t = integer_subbyte<6, true>;

/// 6-bit unsigned integer type
using uint6b_t = integer_subbyte<6, false>;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, bool Signed>
struct sizeof_bits<integer_subbyte<Bits,Signed>> {
  static constexpr int value = Bits;
};

/// Defines the size of an element in bits - specialized for bin1_t
template <>
struct sizeof_bits<bin1_t> {
  static constexpr int value = 1;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace platform {

/// Forward Declaration
template <class T>
struct numeric_limits;

// Specialization for signed integer_subbyte
template<int NumBits>
struct numeric_limits<cutlass::integer_subbyte<NumBits, true>> {
private:
  using value_type = cutlass::integer_subbyte<NumBits, true>;

public:
  CUTLASS_HOST_DEVICE static value_type lowest() noexcept {
    return value_type{
      -(1 << (NumBits - 1))
    };
  }

  CUTLASS_HOST_DEVICE static value_type max() noexcept {
    return value_type{
      (1 << (NumBits - 1)) - 1
    };
  }

  CUTLASS_HOST_DEVICE static value_type const min() noexcept {
    return lowest();
  }

  static constexpr bool is_integer = true;
  static constexpr bool is_signed = true;
  static constexpr bool has_infinity = false;
};

// Specialization for unsigned integer_subbyte
template<int NumBits>
struct numeric_limits<cutlass::integer_subbyte<NumBits, false>> {
private:
  using value_type = cutlass::integer_subbyte<NumBits, false>;

public:
  CUTLASS_HOST_DEVICE static value_type lowest() noexcept {
    return value_type{0u};
  }

  CUTLASS_HOST_DEVICE static value_type max() noexcept {
    return value_type{
      (1u << NumBits) - 1u
    };
  }

  CUTLASS_HOST_DEVICE static value_type const min() noexcept {
    return lowest();
  }

  static constexpr bool is_integer = true;
  static constexpr bool is_signed = false;
};

} // namespace platform
} // namespace cutlass
