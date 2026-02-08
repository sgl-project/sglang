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
    \brief Defines a class for using IEEE half-precision floating-point types in host or
      device code.
*/

#pragma once

#ifndef CUTLASS_ENABLE_F16C
#define CUTLASS_ENABLE_F16C 0
#endif

#if defined(__CUDACC_RTC__)

#include "cutlass/floating_point_nvrtc.h"

// F16C extensions are not meaningful when compiling for NVRTC which only accommodates device code.
#undef CUTLASS_ENABLE_F16C
#define CUTLASS_ENABLE_F16C 0

#else
//
// Standard Library headers belong here to avoid conflicts with NVRTC.
//
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/platform/platform.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// Optionally target F16C extensions to accelerate half-precision conversion.
#if !defined(__CUDA_ARCH__) && (CUTLASS_ENABLE_F16C)
#if defined(_MSC_VER)

#include <immintrin.h>

#if defined(__i386__) || defined(__x86_64__)
#include <intrin.h>
#endif

#define F16C_ROUND_NEAREST 0

#if !defined(__CUDA_ARCH__)
extern __inline float _cvtsh_ss (unsigned short __S) {
  __m128i packed;
  std::memcpy(&packed, &__S, sizeof(__S));

  __m128 result = _mm_cvtph_ps(packed);

  float flt;
  std::memcpy(&flt, &result, sizeof(flt));

  return flt;
}

__inline unsigned short _cvtss_sh (float __F, const int) {
  __m128 packed;
  std::memcpy(&packed, &__F, sizeof(__F));

  __m128i result = _mm_cvtps_ph(packed, F16C_ROUND_NEAREST);

  unsigned short u;
  std::memcpy(&u, &result, sizeof(u));

  return u;
}
#endif

#else

// Linux
#include <x86intrin.h>

#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif

#define F16C_ROUND_NEAREST (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)

#endif // _MSC_VER

class CpuId {

  bool f16c_enabled;

  CpuId() {
  #if defined(__i386__) || defined(__x86_64__)
    #if defined(_MSC_VER)
      int exx[4];

      __cpuid (exx, 1); 
      f16c_enabled = exx[2] & 0x20000000;

    #else 
    // GCC / Clang
       int eax, ebx, ecx, edx;

      __cpuid (1 , eax, ebx, ecx, edx); 
      f16c_enabled = ecx & 0x20000000;
    #endif
  #else 
  // Arm / PowerPC etc.
    f16c_enabled = false;
  #endif
  }

public:

  bool is_f16c_supported() const {
    return f16c_enabled;
  } 

  static const CpuId& instance() {
      static CpuId cpu;
      return cpu;
  }
};
#endif // !defined(__CUDA_ARCH__) && CUTLASS_ENABLE_F16C

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// IEEE half-precision floating-point type
struct alignas(2) half_t {

  //
  // Data members
  //

  /// Storage type
  uint16_t storage;

  //
  // Static conversion operators
  //

  /// Constructs from an unsigned short
  CUTLASS_HOST_DEVICE
  static half_t bitcast(uint16_t x) {
    half_t h;
    h.storage = x;
    return h;
  }

  /// FP32 -> FP16 conversion - rounds to nearest even
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
    // Avoid inlining in device code if no hardware support
    __device__ __noinline__
  #else
    CUTLASS_HOST_DEVICE
  #endif  
  static half_t convert(float const& flt) {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half_t(__float2half_rn(flt));
  #else

    #if !defined(__CUDA_ARCH__) && CUTLASS_ENABLE_F16C
      if( CpuId::instance().is_f16c_supported() ) {
        unsigned short u = _cvtss_sh(flt, F16C_ROUND_NEAREST);
        return bitcast(u);
      }
    #endif

    // software implementation rounds toward nearest even
    unsigned s;

    #if defined(__CUDA_ARCH__)
    s = reinterpret_cast<unsigned const &>(flt);
    #else
    std::memcpy(&s, &flt, sizeof(s));
    #endif

    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      return bitcast(sign);
    }

    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7c00;
      }
      return bitcast(u);
    }

    int sticky_bit = 0;

    if (exp >= -14) {
      // normal fp32 to normal fp16
      exp = uint16_t(exp + uint16_t(15));
      u = uint16_t(((exp & 0x1f) << 10));
      u = uint16_t(u | (mantissa >> 13));
    } else {
      // normal single-precision to subnormal half_t-precision representation
      int rshift = (-14 - exp);
      if (rshift < 32) {
        mantissa |= (1 << 23);

        sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 13) & 0x3ff);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    // round to nearest even
    int round_bit = ((mantissa >> 12) & 1);
    sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
      u = uint16_t(u + 1);
    }

    u |= sign;

    return bitcast(u);
  #endif
  }

  /// FP32 -> FP16 conversion - rounds to nearest even
  CUTLASS_HOST_DEVICE
  static half_t convert(int const& n) {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half_t(__int2half_rn(n));
  #else
    return convert(float(n));
  #endif
  }

  /// FP32 -> FP16 conversion - rounds to nearest even
  CUTLASS_HOST_DEVICE
  static half_t convert(unsigned const& n) {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half_t(__uint2half_rn(n));
  #else
    return convert(float(n));
  #endif
  }

  /// Converts a half-precision value stored as a uint16_t to a float
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
    // Avoid inlining in device code if no hardware support
    __device__ __noinline__
  #else
    CUTLASS_HOST_DEVICE
  #endif
  static float convert(half_t const& x) {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __half2float(x.to_half());
  #else

    #if !defined(__CUDA_ARCH__) && CUTLASS_ENABLE_F16C
      if( CpuId::instance().is_f16c_supported() ) {
        unsigned short u = x.storage;
        return _cvtsh_ss(u);
      }
    #endif

    uint16_t const &h = x.storage;
    uint32_t sign = ((h >> 15) & 1);
    uint32_t exp = ((h >> 10) & 0x1f);
    uint32_t mantissa = (h & 0x3ff);
    unsigned f = 0;

    if (exp > 0 && exp < 31) {
      // normal
      exp += 112;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    } else if (exp == 0) {
      if (mantissa) {
        // subnormal
        exp += 113;
        while ((mantissa & (1 << 10)) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x3ff;
        f = (sign << 31) | (exp << 23) | (mantissa << 13);
      } else {
        // sign-preserving zero
        f = (sign << 31);
      }
    } else if (exp == 31) {
      if (mantissa) {
        f = 0x7fffffff;  // not a number
      } else {
        f = (0xff << 23) | (sign << 31);  //  inf
      }
    }
    #if defined(__CUDA_ARCH__)
    return reinterpret_cast<float const&>(f);
    #else
    float flt;
    std::memcpy(&flt, &f, sizeof(flt));
    return flt;
    #endif
  #endif
  }

  //
  // Methods
  //

  /// Default constructor
  half_t() = default;

  /// Reinterpret cast from CUDA's half type
  CUTLASS_HOST_DEVICE
  explicit half_t(half const & x) {
    #if defined(__CUDA_ARCH__)
    storage = reinterpret_cast<uint16_t const &>(x);
    #else
    __half_raw raw(x);
    std::memcpy(&storage, &raw.x, sizeof(storage));
    #endif
  }

  /// Floating point conversion
  CUTLASS_HOST_DEVICE
  explicit half_t(float x) {
    storage = convert(x).storage;
  }

  /// Floating point conversion
  CUTLASS_HOST_DEVICE
  explicit half_t(double x): half_t(float(x)) {

  }

  /// float_e4m3_t conversion
  CUTLASS_HOST_DEVICE
  explicit half_t(float_e4m3_t x): half_t(float(x)) {

  }

  /// float_e5m2_t conversion
  CUTLASS_HOST_DEVICE
  explicit half_t(float_e5m2_t x): half_t(float(x)) {

  }

  /// Integer conversion - round to nearest even
  CUTLASS_HOST_DEVICE
  explicit half_t(int x) {
    storage = convert(x).storage;
  }

  /// Integer conversion - round toward zero
  CUTLASS_HOST_DEVICE
  explicit half_t(unsigned x) {
    storage = convert(x).storage;
  }

  /// Assignment
  CUTLASS_HOST_DEVICE
  half_t & operator=(half const &x) {
    #if defined(__CUDA_ARCH__)
    storage = reinterpret_cast<uint16_t const &>(x);
    #else
    __half_raw raw(x);
    std::memcpy(&storage, &raw.x, sizeof(storage));
    #endif
    return *this;
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  operator float() const {
    return convert(*this);
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  explicit operator double() const {
    return double(convert(*this));
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(convert(*this));
  }

  /// Casts to bool
  CUTLASS_HOST_DEVICE
  explicit operator bool() const {
    return (convert(*this) != 0.0f);
  }

  /// Bitcasts to CUDA's half type
  CUTLASS_HOST_DEVICE
  half to_half() const {
    #if defined(__CUDA_ARCH__)
    return reinterpret_cast<half const &>(storage);
    #else
    __half_raw raw;
    std::memcpy(&raw.x, &storage, sizeof(raw.x));
    return half(raw);
    #endif
  }

  /// Accesses raw internal state
  CUTLASS_HOST_DEVICE
  uint16_t& raw() {
    return storage;
  }

  /// Accesses raw internal state
  CUTLASS_HOST_DEVICE
  uint16_t raw() const {
    return storage;
  }

  /// Returns the sign bit
  CUTLASS_HOST_DEVICE
  bool signbit() const {
    return ((storage & 0x8000) != 0);
  }

  /// Returns the biased exponent
  CUTLASS_HOST_DEVICE
  int exponent_biased() const {
    return int((storage >> 10) & 0x1f);
  }

  /// Returns the unbiased exponent
  CUTLASS_HOST_DEVICE
  int exponent() const {
    return exponent_biased() - 15;
  }

  /// Returns the mantissa
  CUTLASS_HOST_DEVICE
  int mantissa() const {
    return int(storage & 0x3ff);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool signbit(cutlass::half_t const& h) {
  return ((h.raw() & 0x8000) != 0);
}

CUTLASS_HOST_DEVICE
cutlass::half_t abs(cutlass::half_t const& h) {
  return cutlass::half_t::bitcast(h.raw() & 0x7fff);
}

CUTLASS_HOST_DEVICE
bool isnan(cutlass::half_t const& h) {
  return (h.exponent_biased() == 0x1f) && h.mantissa();
}

CUTLASS_HOST_DEVICE
bool isfinite(cutlass::half_t const& h) {
  return (h.exponent_biased() != 0x1f);
}

CUTLASS_HOST_DEVICE
cutlass::half_t nanh(const char*) {
  // NVIDIA canonical NaN
  return cutlass::half_t::bitcast(0x7fff);
}

CUTLASS_HOST_DEVICE
bool isinf(cutlass::half_t const& h) {
  return (h.exponent_biased() == 0x1f) && !h.mantissa();
}

CUTLASS_HOST_DEVICE
bool isnormal(cutlass::half_t const& h) {
  return h.exponent_biased() && h.exponent_biased() != 0x1f;
}

CUTLASS_HOST_DEVICE
int fpclassify(cutlass::half_t const& h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x1f) {
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
cutlass::half_t sqrt(cutlass::half_t const& h) {
#if defined(__CUDACC_RTC__)
  return cutlass::half_t(sqrtf(float(h)));
#else
  return cutlass::half_t(std::sqrt(float(h)));
#endif
}

CUTLASS_HOST_DEVICE
half_t copysign(half_t const& a, half_t const& b) {

  uint16_t a_mag = (a.raw() & 0x7fff);  
  uint16_t b_sign = (b.raw() & 0x8000);
  uint16_t result = (a_mag | b_sign);

  return half_t::bitcast(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__CUDACC_RTC__)
namespace std {

/// Numeric limits
template <>
struct numeric_limits<cutlass::half_t> {
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
  static bool const is_iec559 = true;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 10;

  /// Least positive value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t min() { return cutlass::half_t::bitcast(0x0001); }

  /// Minimum finite value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t lowest() { return cutlass::half_t::bitcast(0xfbff); }

  /// Maximum finite value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t max() { return cutlass::half_t::bitcast(0x7bff); }

  /// Returns smallest finite value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t epsilon() { return cutlass::half_t::bitcast(0x1800); }

  /// Returns maximum rounding error
  CUTLASS_HOST_DEVICE
  static cutlass::half_t round_error() { return cutlass::half_t(0.5f); }

  /// Returns positive infinity value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t infinity() { return cutlass::half_t::bitcast(0x7c00); }

  /// Returns quiet NaN value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t quiet_NaN() { return cutlass::half_t::bitcast(0x7fff); }

  /// Returns signaling NaN value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t signaling_NaN() { return cutlass::half_t::bitcast(0x7fff); }

  /// Returns smallest positive subnormal value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t denorm_min() { return cutlass::half_t::bitcast(0x0001); }
};
}  // namespace std
#endif

namespace cutlass {
namespace platform {

/// Forward Declaration
template <class T>
struct numeric_limits;

/// Numeric limits
template <>
struct numeric_limits<cutlass::half_t> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
#if !defined(__CUDACC_RTC__)
  static std::float_denorm_style const has_denorm = std::denorm_present;
#endif
  static bool const has_denorm_loss = true;
#if !defined(__CUDACC_RTC__)
  static std::float_round_style const round_style = std::round_to_nearest;
#endif
  static bool const is_iec559 = true;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 10;

  /// Least positive value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t min() { return cutlass::half_t::bitcast(0x0001); }

  /// Minimum finite value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t lowest() { return cutlass::half_t::bitcast(0xfbff); }

  /// Maximum finite value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t max() { return cutlass::half_t::bitcast(0x7bff); }

  /// Returns smallest finite value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t epsilon() { return cutlass::half_t::bitcast(0x1800); }

  /// Returns maximum rounding error
  CUTLASS_HOST_DEVICE
  static cutlass::half_t round_error() { return cutlass::half_t(0.5f); }

  /// Returns positive infinity value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t infinity() { return cutlass::half_t::bitcast(0x7c00); }

  /// Returns quiet NaN value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t quiet_NaN() { return cutlass::half_t::bitcast(0x7fff); }

  /// Returns signaling NaN value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t signaling_NaN() { return cutlass::half_t::bitcast(0x7fff); }

  /// Returns smallest positive subnormal value
  CUTLASS_HOST_DEVICE
  static cutlass::half_t denorm_min() { return cutlass::half_t::bitcast(0x0001); }
};
}  // namespace platform 
}  // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool operator==(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __heq(lhs.to_half(), rhs.to_half());
#else
  return float(lhs) == float(rhs);
#endif
}

CUTLASS_HOST_DEVICE
bool operator!=(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hne(lhs.to_half(), rhs.to_half());
#else
  return float(lhs) != float(rhs);
#endif
}

CUTLASS_HOST_DEVICE
bool operator<(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hlt(lhs.to_half(), rhs.to_half());
#else
  return float(lhs) < float(rhs);
#endif
}

CUTLASS_HOST_DEVICE
bool operator<=(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hle(lhs.to_half(), rhs.to_half());
#else
  return float(lhs) <= float(rhs);
#endif
}

CUTLASS_HOST_DEVICE
bool operator>(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hgt(lhs.to_half(), rhs.to_half());
#else
  return float(lhs) > float(rhs);
#endif
}

CUTLASS_HOST_DEVICE
bool operator>=(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return __hge(lhs.to_half(), rhs.to_half());
#else
  return float(lhs) >= float(rhs);
#endif
}

CUTLASS_HOST_DEVICE
half_t operator+(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return half_t(__hadd(lhs.to_half(), rhs.to_half()));
#else
  return half_t(float(lhs) + float(rhs));
#endif
}

CUTLASS_HOST_DEVICE
half_t operator-(half_t const& lhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return half_t(__hneg(lhs.to_half()));
#else
  return half_t(-float(lhs));
#endif
}

CUTLASS_HOST_DEVICE
half_t operator-(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return half_t(__hsub(lhs.to_half(), rhs.to_half()));
#else
  return half_t(float(lhs) - float(rhs));
#endif
}

CUTLASS_HOST_DEVICE
half_t operator*(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return half_t(__hmul(lhs.to_half(), rhs.to_half()));
#else
  return half_t(float(lhs) * float(rhs));
#endif
}

CUTLASS_HOST_DEVICE
half_t operator/(half_t const& lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return half_t(__hdiv(lhs.to_half(), rhs.to_half()));
#else
  return half_t(float(lhs) / float(rhs));
#endif
}

CUTLASS_HOST_DEVICE
half_t& operator+=(half_t & lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  lhs = half_t(__hadd(lhs.to_half(), rhs.to_half()));
#else
  lhs = half_t(float(lhs) + float(rhs));
#endif
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator-=(half_t & lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  lhs = half_t(__hsub(lhs.to_half(), rhs.to_half()));
#else
  lhs = half_t(float(lhs) - float(rhs));
#endif
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator*=(half_t & lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  lhs = half_t(__hmul(lhs.to_half(), rhs.to_half()));
#else
  lhs = half_t(float(lhs) * float(rhs));
#endif
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator/=(half_t & lhs, half_t const& rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  lhs = half_t(__hdiv(lhs.to_half(), rhs.to_half()));
#else
  lhs = half_t(float(lhs) / float(rhs));
#endif
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator++(half_t & lhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  lhs = half_t(__hadd(lhs.to_half(), half_t(1.0f).to_half()));
#else
  float tmp(lhs);
  ++tmp;
  lhs = half_t(tmp);
#endif
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator--(half_t & lhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  lhs = half_t(__hsub(lhs.to_half(), half_t(1.0f).to_half()));
#else
  float tmp(lhs);
  --tmp;
  lhs = half_t(tmp);
#endif
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t operator++(half_t & lhs, int) {
  half_t ret(lhs);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  lhs = half_t(__hadd(lhs.to_half(), half_t(1.0f).to_half()));
#else
  float tmp(lhs);
  tmp++;
  lhs = half_t(tmp);
#endif
  return ret;
}

CUTLASS_HOST_DEVICE
half_t operator--(half_t & lhs, int) {
  half_t ret(lhs);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  lhs = half_t(__hsub(lhs.to_half(), half_t(1.0f).to_half()));
#else
  float tmp(lhs);
  tmp--;
  lhs = half_t(tmp);
#endif
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

CUTLASS_HOST_DEVICE
cutlass::half_t operator "" _hf(long double x) {
  return cutlass::half_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::half_t operator "" _hf(unsigned long long int x) {
  return cutlass::half_t(int(x));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
