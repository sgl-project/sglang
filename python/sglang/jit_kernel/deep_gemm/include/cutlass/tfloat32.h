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
    \brief Defines a proxy class for storing Tensor Float 32 data type.
*/
#pragma once

#if defined(__CUDACC_RTC__)
#include "cutlass/floating_point_nvrtc.h"
#else
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring> // std::memcpy
#endif

#include "cutlass/cutlass.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Tensor Float 32 data type
struct alignas(4) tfloat32_t {

  //
  // Data members
  //

  /// Storage type
  uint32_t storage;

  //
  // Methods
  //
  private:
    CUTLASS_HOST_DEVICE
    static uint32_t float_to_storage(float s) {
  #if defined(__CUDA_ARCH__)
      uint32_t result = reinterpret_cast<uint32_t const &>(s);
  #else
      uint32_t result;
      std::memcpy(&result, &s, sizeof(float));
  #endif
      return result;
    }

  public:
  /// Constructs from an unsigned int
  CUTLASS_HOST_DEVICE
  static tfloat32_t bitcast(uint32_t x) {
    tfloat32_t h;
    h.storage = x;
    return h;
  }

  /// Emulated rounding is fast in device code
  CUTLASS_HOST_DEVICE
  static tfloat32_t round_half_ulp_truncate(float const &s) {
    uint32_t x = float_to_storage(s);

    #if defined(__CUDA_ARCH__)
    if (::isfinite(s)) {
      x += 0x1000u;
    }
    #else
    if (std::isfinite(s)) {
      x += 0x1000u;
    }
    #endif

    return tfloat32_t::bitcast(x);
  }

  tfloat32_t() = default;

  /// Floating-point conversion - round toward nearest even
  CUTLASS_HOST_DEVICE
  explicit tfloat32_t(float x): storage(round_half_ulp_truncate(x).raw()) { }

  // Conversion from double (this rounds twice)
  CUTLASS_HOST_DEVICE
  explicit tfloat32_t(double x): tfloat32_t(float(x)) { }

  /// Integer conversion - round toward zero
  CUTLASS_HOST_DEVICE
  explicit tfloat32_t(int x) {
    float flt = static_cast<float>(x);
    #if defined(__CUDA_ARCH__)
    storage = reinterpret_cast<uint32_t const &>(flt);
    #else
    std::memcpy(&storage, &flt, sizeof(storage));
    #endif
  }

  // Conversion to float
  CUTLASS_HOST_DEVICE
  operator float() const {

    // Conversions to IEEE single-precision requires clearing dont-care bits
    // of the mantissa.
    unsigned bits = (storage & ~0x1fffu);

    #if defined(__CUDA_ARCH__)
    return reinterpret_cast<float const &>(bits);
    #else
    float flt;
    std::memcpy(&flt, &bits, sizeof(flt));
    return flt;
    #endif
  }

  /// Converts to double
  CUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(float(*this));
  }

  /// Converts to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(float(*this));
  }

  /// Casts to bool
  CUTLASS_HOST_DEVICE
  explicit operator bool() const {
    return (float(*this) != 0.0f);
  }

  /// Obtains raw bits
  CUTLASS_HOST_DEVICE
  uint32_t raw() const {
    return storage;
  }

  /// Returns the sign bit
  CUTLASS_HOST_DEVICE
  bool signbit() const {
    return ((raw() & 0x80000000) != 0);
  }

  /// Returns the biased exponent
  CUTLASS_HOST_DEVICE
  int exponent_biased() const {
    return int((raw() >> 23) & 0x0ff);
  }

  /// Returns the unbiased exponent
  CUTLASS_HOST_DEVICE
  int exponent() const {
    return exponent_biased() - 127;
  }

  /// Returns the mantissa
  CUTLASS_HOST_DEVICE
  int mantissa() const {
    return int(raw() & 0x7fffff);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool signbit(cutlass::tfloat32_t const& h) {
  return h.signbit();
}

CUTLASS_HOST_DEVICE
cutlass::tfloat32_t abs(cutlass::tfloat32_t const& h) {
  return cutlass::tfloat32_t::bitcast(h.raw() & 0x7fffffff);
}

CUTLASS_HOST_DEVICE
bool isnan(cutlass::tfloat32_t const& h) {
  return (h.exponent_biased() == 0x0ff) && h.mantissa();
}

CUTLASS_HOST_DEVICE
bool isfinite(cutlass::tfloat32_t const& h) {
  return (h.exponent_biased() != 0x0ff);
}

CUTLASS_HOST_DEVICE
cutlass::tfloat32_t nan_tf32(const char*) {
  // NVIDIA canonical NaN
  return cutlass::tfloat32_t::bitcast(0x7fffffff);
}

CUTLASS_HOST_DEVICE
bool isinf(cutlass::tfloat32_t const& h) {
  return (h.exponent_biased() == 0x0ff) && !h.mantissa();
}

CUTLASS_HOST_DEVICE
bool isnormal(cutlass::tfloat32_t const& h) {
  return h.exponent_biased() && h.exponent_biased() != 0x0ff;
}

CUTLASS_HOST_DEVICE
int fpclassify(cutlass::tfloat32_t const& h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x0ff) {
    if (mantissa) {
      return FP_NAN;
    }
    else {
      return FP_INFINITE;
    }
  }
  else if (!exp) {
    if (mantissa) {
      return FP_SUBNORMAL;
    }
    else {
      return FP_ZERO;
    }
  }
  return FP_NORMAL;
}

CUTLASS_HOST_DEVICE
cutlass::tfloat32_t sqrt(cutlass::tfloat32_t const& h) {
#if defined(__CUDACC_RTC__)
  return cutlass::tfloat32_t(sqrtf(float(h)));
#else
  return cutlass::tfloat32_t(std::sqrt(float(h)));
#endif
}

CUTLASS_HOST_DEVICE
tfloat32_t copysign(tfloat32_t const& a, tfloat32_t const& b) {

  uint32_t a_mag = (a.raw() & 0x7fffffff);
  uint32_t b_sign = (b.raw() & 0x80000000);
  uint32_t result = (a_mag | b_sign);

  return tfloat32_t::bitcast(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace std {

#if !defined(__CUDACC_RTC__)
/// Numeric limits
template <>
struct numeric_limits<cutlass::tfloat32_t> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 19;

  /// Least positive value
  static cutlass::tfloat32_t min() { return cutlass::tfloat32_t::bitcast(0x01); }

  /// Minimum finite value
  static cutlass::tfloat32_t lowest() { return cutlass::tfloat32_t::bitcast(0xff7fffff); }

  /// Maximum finite value
  static cutlass::tfloat32_t max() { return cutlass::tfloat32_t::bitcast(0x7f7fffff); }

  /// Returns smallest finite value
  static cutlass::tfloat32_t epsilon() { return cutlass::tfloat32_t::bitcast(0x1000); }

  /// Returns smallest finite value
  static cutlass::tfloat32_t round_error() { return cutlass::tfloat32_t(0.5f); }

  /// Returns smallest finite value
  static cutlass::tfloat32_t infinity() { return cutlass::tfloat32_t::bitcast(0x7f800000); }

  /// Returns smallest finite value
  static cutlass::tfloat32_t quiet_NaN() { return cutlass::tfloat32_t::bitcast(0x7fffffff); }

  /// Returns smallest finite value
  static cutlass::tfloat32_t signaling_NaN() { return cutlass::tfloat32_t::bitcast(0x7fffffff); }

  /// Returns smallest finite value
  static cutlass::tfloat32_t denorm_min() { return cutlass::tfloat32_t::bitcast(0x1); }
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool operator==(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return float(lhs) == float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator!=(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return float(lhs) != float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return float(lhs) < float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<=(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return float(lhs) <= float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return float(lhs) > float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>=(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return float(lhs) >= float(rhs);
}

CUTLASS_HOST_DEVICE
tfloat32_t operator+(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return tfloat32_t(float(lhs) + float(rhs));
}


CUTLASS_HOST_DEVICE
tfloat32_t operator-(tfloat32_t const& lhs) {
  return tfloat32_t::bitcast(0x80000000 ^ lhs.raw());
}

CUTLASS_HOST_DEVICE
tfloat32_t operator-(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return tfloat32_t(float(lhs) - float(rhs));
}

CUTLASS_HOST_DEVICE
tfloat32_t operator*(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return tfloat32_t(float(lhs) * float(rhs));
}

CUTLASS_HOST_DEVICE
tfloat32_t operator/(tfloat32_t const& lhs, tfloat32_t const& rhs) {
  return tfloat32_t(float(lhs) / float(rhs));
}

CUTLASS_HOST_DEVICE
tfloat32_t& operator+=(tfloat32_t & lhs, tfloat32_t const& rhs) {
  lhs = tfloat32_t(float(lhs) + float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
tfloat32_t& operator-=(tfloat32_t & lhs, tfloat32_t const& rhs) {
  lhs = tfloat32_t(float(lhs) - float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
tfloat32_t& operator*=(tfloat32_t & lhs, tfloat32_t const& rhs) {
  lhs = tfloat32_t(float(lhs) * float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
tfloat32_t& operator/=(tfloat32_t & lhs, tfloat32_t const& rhs) {
  lhs = tfloat32_t(float(lhs) / float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
tfloat32_t& operator++(tfloat32_t & lhs) {
  float tmp(lhs);
  ++tmp;
  lhs = tfloat32_t(tmp);
  return lhs;
}

CUTLASS_HOST_DEVICE
tfloat32_t& operator--(tfloat32_t & lhs) {
  float tmp(lhs);
  --tmp;
  lhs = tfloat32_t(tmp);
  return lhs;
}

CUTLASS_HOST_DEVICE
tfloat32_t operator++(tfloat32_t & lhs, int) {
  tfloat32_t ret(lhs);
  float tmp(lhs);
  tmp++;
  lhs = tfloat32_t(tmp);
  return ret;
}

CUTLASS_HOST_DEVICE
tfloat32_t operator--(tfloat32_t & lhs, int) {
  tfloat32_t ret(lhs);
  float tmp(lhs);
  tmp--;
  lhs = tfloat32_t(tmp);
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

CUTLASS_HOST_DEVICE
cutlass::tfloat32_t operator "" _tf32(long double x) {
  return cutlass::tfloat32_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::tfloat32_t operator "" _tf32(unsigned long long int x) {
  return cutlass::tfloat32_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
