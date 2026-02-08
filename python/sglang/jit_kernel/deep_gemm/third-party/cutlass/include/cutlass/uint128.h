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
  \brief Defines an unsigned 128b integer with several operators to support 64-bit integer division.
*/
#pragma once
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#endif


/// Optionally enable GCC's built-in type
#if (defined(__x86_64) || defined (__aarch64__)) && !(defined(__CUDA_ARCH__) && ((__CUDACC_VER_MAJOR__ <= 10) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ <= 4)))) && defined(__GNUC__)
#define CUTLASS_UINT128_NATIVE
#elif !defined(__CUDA_ARCH__)
// No custom support for 128b arithmetic on device
#if defined(_MSC_VER) && defined(_M_AMD64)
#define CUTLASS_INT128_ARITHMETIC
#include <intrin.h>
#if _MSC_VER >= 1920 && !defined(__CUDA_ARCH__)
#define CUTLASS_INT128_ARITHMETIC_DIV
#include <immintrin.h>
#endif
#endif
#endif

namespace cutlass {

///! Unsigned 128b integer type
struct alignas(16) uint128_t
{
  /// Size of one part of the uint's storage in bits
  static constexpr int storage_bits_ = 64;

  struct hilo
  {
    uint64_t lo;
    uint64_t hi;
  };

  // Use a union to store either low and high parts or, if present, a built-in 128b integer type.
  union {
    struct hilo hilo_;

#if defined(CUTLASS_UINT128_NATIVE)
    unsigned __int128 native;
#endif // defined(CUTLASS_UINT128_NATIVE)
  };

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  uint128_t() : hilo_{0, 0} {}

  /// Constructor from uint64
  CUTLASS_HOST_DEVICE
  uint128_t(uint64_t lo_) : hilo_{lo_, 0} {}

  /// Constructor from two 64b unsigned integers
  CUTLASS_HOST_DEVICE
  uint128_t(uint64_t lo_, uint64_t hi_) : hilo_{lo_, hi_} {}

  /// Optional constructor from native value
#if defined(CUTLASS_UINT128_NATIVE)
  uint128_t(unsigned __int128 value) : native(value) { }
#endif

  /// Lossily cast to uint64
  CUTLASS_HOST_DEVICE
  explicit operator uint64_t() const
  {
    return hilo_.lo;
  }

  CUTLASS_HOST_DEVICE
  static void exception()
  {
#if defined(__CUDA_ARCH__)
  asm volatile ("  brkpt;\n");
#else
  // throw std::runtime_error("Not yet implemented.");
  abort();
#endif
  }

  /// Add
  CUTLASS_HOST_DEVICE
  uint128_t operator+(uint128_t const& rhs) const
  {
    uint128_t y{};
#if defined(CUTLASS_UINT128_NATIVE)
    y.native = native + rhs.native;
#else
    y.hilo_.lo = hilo_.lo + rhs.hilo_.lo;
    y.hilo_.hi = hilo_.hi + rhs.hilo_.hi + (y.hilo_.lo < hilo_.lo);
#endif
    return y;
  }

  /// Subtract
  CUTLASS_HOST_DEVICE
  uint128_t operator-(uint128_t const& rhs) const
  {
    uint128_t y{};
#if defined(CUTLASS_UINT128_NATIVE)
    y.native = native - rhs.native;
#else
    y.hilo_.lo = hilo_.lo - rhs.hilo_.lo;
    y.hilo_.hi = hilo_.hi - rhs.hilo_.hi - (rhs.hilo_.lo && y.hilo_.lo > hilo_.lo);
#endif
    return y;
  }

  /// Multiply by unsigned 64b integer yielding 128b integer
  CUTLASS_HOST_DEVICE
  uint128_t operator*(uint64_t const& rhs) const
  {
    uint128_t y{};
#if defined(CUTLASS_UINT128_NATIVE)
    y.native = native * rhs;
#elif defined(CUTLASS_INT128_ARITHMETIC)
    // Multiply by the low part
    y.hilo_.lo = _umul128(hilo_.lo, rhs, &y.hilo_.hi);

    // Add the high part and ignore the overflow
    uint64_t overflow{0};
    y.hilo_.hi += _umul128(hilo_.hi, rhs, &overflow);
#else
    CUTLASS_UNUSED(rhs);
    exception();
#endif
    return y;
  }

  /// Divide 128b operation by 64b operation yielding a 64b quotient
  CUTLASS_HOST_DEVICE
  uint64_t operator/(uint64_t const& divisor) const
  {
    uint64_t quotient{0};
#if defined(CUTLASS_UINT128_NATIVE)
    quotient = uint64_t(native / divisor);
#elif defined(CUTLASS_INT128_ARITHMETIC_DIV)
    // implemented using MSVC's arithmetic intrinsics
    uint64_t remainder{0};
    quotient = _udiv128(hilo_.hi, hilo_.lo, divisor, &remainder);
#else
    CUTLASS_UNUSED(divisor);
    exception();
#endif
    return quotient;
  }

  /// Divide 128b operation by 64b operation yielding a 64b quotient
  CUTLASS_HOST_DEVICE
  uint64_t operator%(uint64_t const& divisor) const
  {
    uint64_t remainder{0};
#if defined(CUTLASS_UINT128_NATIVE)
    remainder = uint64_t(native % divisor);
#elif defined(CUTLASS_INT128_ARITHMETIC_DIV)
    // implemented using MSVC's arithmetic intrinsics
    (void)_udiv128(hilo_.hi, hilo_.lo, divisor, &remainder);
#else
    CUTLASS_UNUSED(divisor);
    exception();
#endif
    return remainder;
  }

  /// Computes the quotient and remainder in a single method.
  CUTLASS_HOST_DEVICE
  uint64_t divmod(uint64_t &remainder, uint64_t divisor) const
  {
    uint64_t quotient{0};
#if defined(CUTLASS_UINT128_NATIVE)
    quotient = uint64_t(native / divisor);
    remainder = uint64_t(native % divisor);
#elif defined(CUTLASS_INT128_ARITHMETIC_DIV)
    // implemented using MSVC's arithmetic intrinsics
    quotient = _udiv128(hilo_.hi, hilo_.lo, divisor, &remainder);
#else
    CUTLASS_UNUSED(remainder);
    CUTLASS_UNUSED(divisor);
    exception();
#endif
    return quotient;
  }

  /// Left-shifts a 128b unsigned integer
  CUTLASS_HOST_DEVICE
  uint128_t operator<<(int sh) const
  {
    if (sh == 0) {
      return *this;
    }
    else if (sh >= storage_bits_) {
      return uint128_t(0, hilo_.lo << (sh - storage_bits_));
    }
    else {
      return uint128_t(
        (hilo_.lo << sh),
        (hilo_.hi << sh) | uint64_t(hilo_.lo >> (storage_bits_ - sh))
      );
    }
  }

  /// Right-shifts a 128b unsigned integer
  CUTLASS_HOST_DEVICE
  uint128_t operator>>(int sh) const
  {
    if (sh == 0) {
      return *this;
    }
    else if (sh >= storage_bits_) {
      return uint128_t((hilo_.hi >> (sh - storage_bits_)), 0);
    }
    else {
      return uint128_t(
        (hilo_.lo >> sh) | (hilo_.hi << (storage_bits_ - sh)),
        (hilo_.hi >> sh)
      );
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
