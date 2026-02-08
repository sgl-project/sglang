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
    \brief Boost-like numeric conversion operator for CUTLASS numeric types
*/

#pragma once

#if !defined(__CUDACC_RTC__)
#include <cfenv>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/thread/unary_op.h"

#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Floating-point rounding style similar to Standard Library's formats but supporting
/// additional rounding options.
enum class FloatRoundStyle {
  round_indeterminate,          ///< rounding mode unknown
  round_toward_zero,            ///< round toward zero
  round_to_nearest,             ///< round to nearest even
  round_to_nearest_satfinite,   ///< round to nearest even, capping value to min and max of destination type
  round_toward_infinity,        ///< round toward infinity
  round_toward_neg_infinity,    ///< round toward negative infinity
  round_half_ulp_truncate,      ///< add 0.5ulp to integer representation then round toward zero
  round_half_ulp_trunc_dntz     ///< like round_half_ulp_truncate, except denorms are rounded *toward* zero
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
struct NumericConverter {

  using result_type = T;
  using source_type = S;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<result_type>(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => int32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    #if __CUDA_ARCH__
    return __float2int_rn(s);
    #elif !defined(__CUDACC_RTC__)
    std::fesetround(FE_TONEAREST);
    return static_cast<result_type>(std::nearbyint(s));
    #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    #if __CUDA_ARCH__
    return __float2int_rz(s);
    #elif !defined(__CUDACC_RTC__)
    std::fesetround(FE_TOWARDZERO);
    return (result_type)std::nearbyint(s);
    #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => int8_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    #if defined(__CUDA_ARCH__)
    int32_t intermediate;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(intermediate) : "f"(s));
    return static_cast<result_type>(intermediate);
    #elif !defined(__CUDACC_RTC__)
    std::fesetround(FE_TONEAREST);
    int32_t intermediate = (int32_t)std::nearbyint(s);
    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());
    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());
    return static_cast<result_type>(intermediate);
    #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    #if defined(__CUDA_ARCH__)
    int32_t intermediate;
    asm volatile("cvt.rzi.sat.s8.f32 %0, %1;" : "=r"(intermediate) : "f"(s));
    return static_cast<result_type>(intermediate);
    #elif !defined(__CUDACC_RTC__)
    std::fesetround(FE_TOWARDZERO);
    int32_t intermediate = (int32_t)std::nearbyint(s);
    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());
    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());
    return static_cast<result_type>(intermediate);
    #endif 
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<uint8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = uint8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    #if defined(__CUDA_ARCH__)
    int32_t intermediate;
    asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(intermediate) : "f"(s));
    return static_cast<result_type>(intermediate);
    #elif !defined(__CUDACC_RTC__)
    std::fesetround(FE_TONEAREST);
    int32_t intermediate = (int32_t)std::nearbyint(s);
    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<uint8_t>::lowest());
    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<uint8_t>::max());
    return static_cast<result_type>(intermediate);
    #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<uint8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = uint8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    #if __CUDA_ARCH__
    int32_t intermediate;
    asm volatile("cvt.rzi.sat.u8.f32 %0, %1;" : "=r"(intermediate) : "f"(s));
    return static_cast<result_type>(intermediate);
    #elif !defined(__CUDACC_RTC__)
    std::fesetround(FE_TOWARDZERO);
    int32_t intermediate = (int32_t)std::nearbyint(s);
    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<uint8_t>::lowest());
    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<uint8_t>::max());
    return static_cast<result_type>(intermediate);
    #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for cutlass::half_t => int8_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct NumericConverter<int8_t, cutlass::half_t, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = cutlass::half_t;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    #if defined(__CUDA_ARCH__)
    union { int8_t int8[2]; int16_t int16; };
    union { cutlass::half_t fp16; int16_t int16_in; };
    fp16 = s;
    asm volatile ("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
    return int8[0];
    #elif !defined(__CUDACC_RTC__)
    std::fesetround(FE_TONEAREST);
    int32_t intermediate = (int32_t)std::nearbyint(static_cast<float>(s));
    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());
    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());
    return static_cast<result_type>(intermediate);
    #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => integer_subbyte
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<int Bits, FloatRoundStyle Round>
struct NumericConverter<integer_subbyte<Bits, /* Signed = */ true>, float, Round> {
private:
  static constexpr bool result_is_signed = true;

public:
  using result_type = integer_subbyte<Bits, result_is_signed>;
  using source_type = float;
  static constexpr FloatRoundStyle round_style = Round;

  CUTLASS_HOST_DEVICE static result_type
  convert(source_type const& src) {
    using middle_type = int;
    static_assert(8 * sizeof(middle_type) > Bits, "This conversion "
      "requires that integer_subbyte have fewer representation bits "
      "than the number of bits in int.");

    auto middle = NumericConverter<middle_type, source_type, Round>::convert(src);
    return NumericConverter<result_type, middle_type, Round>::convert(middle);
  }

  CUTLASS_HOST_DEVICE result_type
  operator()(source_type const& s) const {
    return convert(s);
  }
};

template<int Bits, FloatRoundStyle Round>
struct NumericConverter<integer_subbyte<Bits, /* Signed = */ false>, float, Round> {
private:
  static constexpr bool result_is_signed = false;

public:
  using result_type = integer_subbyte<Bits, result_is_signed>;
  using source_type = float;
  static constexpr FloatRoundStyle round_style = Round;

  CUTLASS_HOST_DEVICE static result_type
  convert(source_type const& src) {
    using middle_type = unsigned;
    static_assert(8 * sizeof(middle_type) > Bits, "This conversion "
      "requires that integer_subbyte have fewer representation bits "
      "than the number of bits in unsigned int.");

    auto middle = NumericConverter<middle_type, source_type, Round>::convert(src);
    return NumericConverter<result_type, middle_type, Round>::convert(middle);
  }

  CUTLASS_HOST_DEVICE result_type  
  operator()(source_type const& s) const {
    return convert(s);
  }
};
  
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= cutlass::half_t
template <typename T, FloatRoundStyle Round>
struct NumericConverter<T, T, Round> {

  using result_type = T;
  using source_type = T;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return s;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> cutlass::half_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= cutlass::half_t
template <FloatRoundStyle Round>
struct NumericConverter<float, cutlass::half_t, Round> {

  using result_type = float;
  using source_type = cutlass::half_t;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<float>(s);

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<cutlass::half_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = cutlass::half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<cutlass::half_t>(s);

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<cutlass::half_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = cutlass::half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & flt) {

  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return cutlass::half_t(__float2half_rz(flt));
  #else
    // software implementation rounds toward nearest even
    unsigned const& s = reinterpret_cast<unsigned const &>(flt);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int32_t exp = int32_t((s >> 23) & 0xff) - 127;
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      return cutlass::half_t::bitcast(sign);
    }

    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7c00;
      }
      return cutlass::half_t::bitcast(u);
    }

    if (exp >= -14) {
      // normal fp32 to normal fp16
      u = uint16_t((uint32_t(exp + 15) & 0x1f) << 10);
      u = uint16_t(u | (mantissa >> 13));
    } else {
      // normal single-precision to subnormal cutlass::half_t-precision representation
      int rshift = (-14 - exp);
      if (rshift < 32) {
        mantissa |= (1 << 23);
        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 13) & 0x3ff);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    u |= sign;

    return cutlass::half_t::bitcast(u);

  #endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> cutlass::bfloat16_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= cutlass::bfloat16_t
template <FloatRoundStyle Round>
struct NumericConverter<float, cutlass::bfloat16_t, Round> {

  using result_type = float;
  using source_type = cutlass::bfloat16_t;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<cutlass::bfloat16_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = cutlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return static_cast<cutlass::bfloat16_t>(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<cutlass::bfloat16_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = cutlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);

    #if defined(__CUDA_ARCH__)
    if (::isfinite(s)) {
      x32 += 0x8000;
    }
    #else
    if (std::isfinite(s)) {
      x32 += 0x8000;
    }
    #endif

    uint16_t x16 = uint16_t((x32 >> 16) & 0xffff);
    return cutlass::bfloat16_t::bitcast(x16);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<cutlass::bfloat16_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = cutlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);
    uint16_t x16 = uint16_t(x32 >> 16);

    return cutlass::bfloat16_t::bitcast(x16);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> cutlass::tfloat32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= cutlass::tfloat32_t
template <FloatRoundStyle Round>
struct NumericConverter<float, cutlass::tfloat32_t, Round> {

  using result_type = float;
  using source_type = cutlass::tfloat32_t;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<cutlass::tfloat32_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = cutlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    unsigned storage = reinterpret_cast<unsigned const &>(s);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cvt.rn.tf32.f32 %0, %1;" : "=r"(storage) : "r"(storage));
#else
    if ((storage & 0x7f800000) != 0x7f800000) {

      bool mantissa_bit = ((storage & (1 << 13)) != 0);
      bool round_bit = ((storage & (1 << 12)) != 0);
      bool sticky_bit = ((storage & ((1 << 12) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        storage += uint32_t(1 << 13);
      }

      // Note, the following is intentionally commented out. TF32
      // does not define the low order bits, so they may be left in
      // an undefined state.
      //
      // By not truncating these bit explicitly, we avoid an extra logical
      // operation.
      //
      // TF32 may be implicitly converted to float by performing this
      // operation as needed.
      //
      // storage = (storage & ~0x1fff);
    }
    else if (storage & ~0xff800000) {
      storage = 0x7fffffff;
    }
#endif

    return cutlass::tfloat32_t::bitcast(storage);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<cutlass::tfloat32_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = cutlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return cutlass::tfloat32_t::round_half_ulp_truncate(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// This rounding operation is similar to half_ulp_truncate except it rounds denorms toward zero.
/// It avoids predicated code, though it requires a temporary register.
template <>
struct NumericConverter<cutlass::tfloat32_t, float, FloatRoundStyle::round_half_ulp_trunc_dntz> {
  using result_type = cutlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_trunc_dntz;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    unsigned y = reinterpret_cast<unsigned const &>(s);
    y = y & 0xff800000;
    float d = reinterpret_cast<float const &>(y);
    float z = d / float(1 << 11) + s;

    return reinterpret_cast<result_type const &>(z);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<cutlass::tfloat32_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = cutlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    uint32_t x = reinterpret_cast<uint32_t const &>(s);
    return cutlass::tfloat32_t::bitcast(x & 0xffffe000);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for float to cutlass::tfloat32_t big and small values
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  FloatRoundStyle RoundBig = FloatRoundStyle::round_toward_zero,
  FloatRoundStyle RoundSmall = FloatRoundStyle::round_half_ulp_truncate
>
struct NumericConverterFastF32 {

  // result_type holds big cutlass::tfloat32_t at idx(0) and small cutlass::tfloat32_t at idx(1)
  using result_type = Array<cutlass::tfloat32_t, 2>;

  // source data type
  using source_type = float;

  // rounding styles for big and small part
  static FloatRoundStyle const kRoundBig = RoundBig;
  static FloatRoundStyle const kRoundSmall = RoundSmall;

  CUTLASS_HOST_DEVICE
    static result_type convert(source_type const & source) {

    result_type result;
    NumericConverter<cutlass::tfloat32_t, float, kRoundBig> convert_big_;
    NumericConverter<cutlass::tfloat32_t, float, kRoundSmall> convert_small_;

    // convert and fill cutlass::tfloat32_t big at idx 0
    result[0] = convert_big_(source);

    // convert and fill cutlass::tfloat32_t small at idx 1
    result[1] = convert_small_(source - static_cast<float>(result[0]));

    return result;
  }

  CUTLASS_HOST_DEVICE
    result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion and Clamp operator for Integers
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S
>
struct NumericConverterClamp {

  using result_type = T;
  using source_type = S;

  CUTLASS_HOST_DEVICE
    static result_type convert(source_type const & s) {
    NumericConverter<result_type, source_type> convert_op;
    result_type const kClamp_max = cutlass::platform::numeric_limits<result_type>::max();
    result_type const kClamp_min = cutlass::platform::numeric_limits<result_type>::lowest();
    if (s < (source_type)kClamp_min)
      return kClamp_min;
    if (s > (source_type)kClamp_max)
      return kClamp_max;
    return convert_op(s);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

// This converter is needed to enable cutlass::half_t output types when using int32_t accumulators.
// Since floating-point types do not require a clamp, this converter simply casts from
// the source type to cutlass::half_t.
template <
  typename S
>
struct NumericConverterClamp<cutlass::half_t, S> {

  using result_type = cutlass::half_t;
  using source_type = S;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const &source) {
    return static_cast<cutlass::half_t>(source);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for Array
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conversion operator for Array
template <
  typename T,
  typename S,
  int N,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename Transform = cutlass::transform::thread::UnaryTransform::Identity
>
struct NumericArrayConverter {

  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result;
    NumericConverter<T, S, Round> convert_;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      if (platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value) {
        result[i] = convert_(s[i]);
      } else { // conjugate
        result[i] = conj(convert_(s[i]));
      }
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  typename T,
  int N,
  FloatRoundStyle Round,
  typename Transform
>
struct NumericArrayConverter<T, T, N, Round, Transform> {

  using result_type = Array<T, N>;
  using source_type = Array<T, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const &source) {
    if (platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value) {
      return source;
    } else {
      result_type result;
      for (int i = 0; i < N; ++i) {
        result[i] = conj(static_cast<typename source_type::Element>(source[i]));
      }
      return result;
    }
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half, 2> <= Array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<cutlass::half_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = Array<cutlass::half_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
      Array<cutlass::half_t, 2> result;
      reinterpret_cast<__half2 &>(result) = __float22half2_rn(reinterpret_cast<float2 const &>(source));
      return result;
    #else
      NumericConverter<cutlass::half_t, float, round_style> convert_;
      // NOTE: cutlass::Array<half, N> is NOT an aggregate type and
      //  below `{}` does NOT conduct zero initialization. Below `{}` will 
      //  conduct default initialization (calling default ctr). We use this syntax
      //  to resolve compiler warning on uninitialized member variable.
      Array<cutlass::half_t, 2> result{};
      result[0] = convert_(source[0]);
      result[1] = convert_(source[1]);
      return result;
    #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 2> <= Array<cutlass::half_t, 2>, round to nearest
template <FloatRoundStyle Round>
struct NumericArrayConverter<float, cutlass::half_t, 2, Round> {

  using result_type = Array<float, 2>;
  using source_type = Array<cutlass::half_t, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
      float2 result2 = __half22float2(reinterpret_cast<__half2 const &>(source));
      return {
        float{result2.x},
        float{result2.y}
      };
    #else
      NumericConverter<float, cutlass::half_t, round_style> convert_;
      return {
        convert_(source[0]),
        convert_(source[1])
      };
    #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::half_t, float, N, Round> {

  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<cutlass::half_t, float, 2, Round> convert_vector_;
    NumericConverter<cutlass::half_t, float, Round> convert_element_;

    result_type result;

    Array<cutlass::half_t, 2> *result_ptr = reinterpret_cast<Array<cutlass::half_t, 2> *>(&result);
    Array<float, 2> const *source_ptr = reinterpret_cast<Array<float, 2> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, cutlass::half_t, N, Round> {

  using result_type = Array<float, N>;
  using source_type = Array<cutlass::half_t, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<float, cutlass::half_t, 2, Round> convert_vector_;
    NumericConverter<float, cutlass::half_t, Round> convert_element_;

    result_type result;

    Array<float, 2> *result_ptr = reinterpret_cast<Array<float, 2> *>(&result);
    Array<cutlass::half_t, 2> const *source_ptr = reinterpret_cast<Array<cutlass::half_t, 2> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<cutlass::bfloat16_t, 2> <= Array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<cutlass::bfloat16_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = Array<cutlass::bfloat16_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned d;

    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(source[1]), "f"(source[0]) );

    return reinterpret_cast<result_type const &>(d);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/// Partial specialization for Array<cutlass::bfloat16_t, 2> <= Array<float, 2>, round to nearest with min/max saturation
template <>
struct NumericArrayConverter<cutlass::bfloat16_t, float, 2, FloatRoundStyle::round_to_nearest_satfinite> {

  using result_type = Array<cutlass::bfloat16_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest_satfinite;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned d;

    asm("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(source[1]), "f"(source[0]) );

    return reinterpret_cast<result_type const &>(d);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


/// Partial specialization for Array<cutlass::bfloat16_t> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::bfloat16_t, float, N, Round> {

  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<cutlass::bfloat16_t, float, 2, Round> convert_vector_;
    NumericConverter<cutlass::bfloat16_t, float, Round> convert_element_;

    result_type result;

    Array<cutlass::bfloat16_t, 2> *result_ptr = reinterpret_cast<Array<cutlass::bfloat16_t, 2> *>(&result);
    Array<float, 2> const *source_ptr = reinterpret_cast<Array<float, 2> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#endif // if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

/////////////////////////////////////////////////////////////////////////////////////////////////

// Conditional guards to enable partial specialization for packed integers
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && \
    ((__CUDACC_VER_MAJOR__ > 10) ||                     \
     ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))

/// Partial specialization for Array<int8_t, 1> <= Array<int, 1>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 1, Round> {

  using result_type = Array<int8_t, 1>;
  using source_type = Array<int, 1>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericConverter<int8_t, int, Round> convert_element_;

    result_type result;

    result[0] = convert_element_(source[0]);

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t, 2> <= Array<int, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 2, Round> {

  using result_type = Array<int8_t, 2>;
  using source_type = Array<int, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    uint32_t tmp;

    asm volatile(
      "cvt.pack.sat.s8.s32.b32   %0, %2, %1, 0;\n"
      : "=r"(tmp) : "r"(source[0]), "r"(source[1]));

    uint16_t out = (tmp & 0xffff);
    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t, 4> <= Array<int, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, 4, Round> {

  using result_type = Array<int8_t, 4>;
  using source_type = Array<int, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
      "{ .reg .u32 r4;"
      "cvt.pack.sat.s8.s32.b32   r4, %4, %3, 0;"
      "cvt.pack.sat.s8.s32.b32   %0, %2, %1, r4;"
      "}"
      : "=r"(out) : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t> <= Array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = Array<int8_t, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<int8_t, int, 4, Round> convert_vector_;

    result_type result;

    Array<int8_t, 4> *result_ptr = reinterpret_cast<Array<int8_t, 4> *>(&result);
    Array<int, 4> const *source_ptr = reinterpret_cast<Array<int, 4> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<uint8_t, 1> <= Array<int, 1>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 1, Round> {

  using result_type = Array<uint8_t, 1>;
  using source_type = Array<int, 1>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericConverter<uint8_t, int, Round> convert_element_;

    result_type result;

    result[0] = convert_element_(source[0]);

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<uint8_t, 2> <= Array<int, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 2, Round> {

  using result_type = Array<uint8_t, 2>;
  using source_type = Array<int, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    uint32_t tmp;

    asm volatile(
      "cvt.pack.sat.u8.s32.b32   %0, %2, %1, 0;\n"
      : "=r"(tmp) : "r"(source[0]), "r"(source[1]));

    uint16_t out = (tmp & 0xffff);
    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<uint8_t, 4> <= Array<int, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, 4, Round> {

  using result_type = Array<uint8_t, 4>;
  using source_type = Array<int, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
      "{ .reg .u32 r4;"
      "cvt.pack.sat.u8.s32.b32   r4, %4, %3, 0;"
      "cvt.pack.sat.u8.s32.b32   %0, %2, %1, r4;"
      "}"
      : "=r"(out) : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t> <= Array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, int, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = Array<uint8_t, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<uint8_t, int, 4, Round> convert_vector_;

    result_type result;

    Array<uint8_t, 4> *result_ptr = reinterpret_cast<Array<uint8_t, 4> *>(&result);
    Array<int, 4> const *source_ptr = reinterpret_cast<Array<int, 4> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, N> <=> Array<float_e4m3_t, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float, 2> <= Array<float_e4m3_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, cutlass::float_e4m3_t, 2, Round> {
  using result_element = float;
  using source_element = cutlass::float_e4m3_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out_fp16;
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);

    asm volatile( \
        "{\n" \
        "cvt.rn.f16x2.e4m3x2 %0, %1;\n" \
        "}\n" : "=r"(out_fp16): "h"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 2> <= Array<float, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, float, 2, Round> {
  using result_element = cutlass::float_e4m3_t;
  using source_element = float;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint16_t out;

    asm volatile( \
        "{\n" \
        "cvt.rn.satfinite.e4m3x2.f32   %0, %2, %1;\n" \
        "}" \
        : "=h"(out) : "f"(source[0]), "f"(source[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 2> <= Array<float_e5m2_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, cutlass::float_e5m2_t, 2, Round> {
  using result_element = float;
  using source_element = cutlass::float_e5m2_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out_fp16;
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);

    asm volatile( \
        "{\n" \
        "cvt.rn.f16x2.e5m2x2 %0, %1;\n" \
        "}\n" : "=r"(out_fp16): "h"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 2> <= Array<float, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, float, 2, Round> {
  using result_element = cutlass::float_e5m2_t;
  using source_element = float;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint16_t out;

    asm volatile( \
        "{\n" \
        "cvt.rn.satfinite.e5m2x2.f32   %0, %2, %1;\n" \
        "}" \
        : "=h"(out) : "f"(source[0]), "f"(source[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<half, N> <=> Array<float_e4m3_t, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half, 2> <= Array<float_e4m3_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::half_t, cutlass::float_e4m3_t, 2, Round> {
  using result_element = cutlass::half_t;
  using source_element = cutlass::float_e4m3_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    result_type out;
    uint32_t& reg = reinterpret_cast<uint32_t&>(out);
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);

    asm volatile( \
        "{\n" \
        "cvt.rn.f16x2.e4m3x2 %0, %1;\n" \
        "}\n" : "=r"(reg): "h"(src_packed));

    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 2> <= Array<half, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, cutlass::half_t, 2, Round> {
  using result_element = cutlass::float_e4m3_t;
  using source_element = cutlass::half_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint16_t out;

    asm volatile( \
        "{\n" \
        "cvt.rn.satfinite.e4m3x2.f16x2   %0, %1;\n" \
        "}" \
        : "=h"(out) : "r"(reinterpret_cast<uint32_t const&>(source)));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<half, 2> <= Array<float_e5m2_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::half_t, cutlass::float_e5m2_t, 2, Round> {
  using result_element = cutlass::half_t;
  using source_element = cutlass::float_e5m2_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    result_type out;
    uint32_t& reg = reinterpret_cast<uint32_t&>(out);
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);

    asm volatile( \
        "{\n" \
        "cvt.rn.f16x2.e5m2x2 %0, %1;\n" \
        "}\n" : "=r"(reg): "h"(src_packed));

    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 2> <= Array<half, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, cutlass::half_t, 2, Round> {
  using result_element = cutlass::float_e5m2_t;
  using source_element = cutlass::half_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint16_t out;

    asm volatile( \
        "{\n" \
        "cvt.rn.satfinite.e5m2x2.f16x2   %0, %1;\n" \
        "}" \
        : "=h"(out) : "r"(reinterpret_cast<uint32_t const&>(source)));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<bfloat16_t, N> <=> Array<float_e4m3_t, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<bfloat16_t, 2> <= Array<float_e4m3_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e4m3_t, 2, Round> {
  using result_element = cutlass::bfloat16_t;
  using source_element = cutlass::float_e4m3_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t res_half;
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);

    asm volatile( \
        "{\n" \
        "cvt.rn.f16x2.e4m3x2 %0, %1;\n" \
        "}\n" : "=r"(res_half): "h"(src_packed));
    float2 res_float = __half22float2(reinterpret_cast<__half2 &>(res_half));
    NumericArrayConverter<cutlass::bfloat16_t, float, 2, Round> converter;
    return converter(reinterpret_cast<Array<float, 2> const&>(res_float));
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 2> <= Array<bfloat16_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, cutlass::bfloat16_t, 2, Round> {
  using result_element = cutlass::float_e4m3_t;
  using source_element = cutlass::bfloat16_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    NumericArrayConverter<float, cutlass::bfloat16_t, 2, Round> converter;
    Array<float, 2> res_float = converter(source);
    uint16_t out;

    asm volatile( \
        "{\n" \
        "cvt.rn.satfinite.e4m3x2.f32   %0, %2, %1;\n" \
        "}" \
        : "=h"(out) : "f"(res_float[0]), "f"(res_float[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<bfloat16_t, 2> <= Array<float_e5m2_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e5m2_t, 2, Round> {
  using result_element = cutlass::bfloat16_t;
  using source_element = cutlass::float_e5m2_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t res_half;
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);

    asm volatile( \
        "{\n" \
        "cvt.rn.f16x2.e5m2x2 %0, %1;\n" \
        "}\n" : "=r"(res_half): "h"(src_packed));
    float2 res_float = __half22float2(reinterpret_cast<__half2 &>(res_half));
    NumericArrayConverter<cutlass::bfloat16_t, float, 2, Round> converter;
    return converter(reinterpret_cast<Array<float, 2> const&>(res_float));
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 2> <= Array<bfloat16_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, cutlass::bfloat16_t, 2, Round> {
  using result_element = cutlass::float_e5m2_t;
  using source_element = cutlass::bfloat16_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    NumericArrayConverter<float, cutlass::bfloat16_t, 2, Round> converter;
    Array<float, 2> res_float = converter(source);
    uint16_t out;

    asm volatile( \
        "{\n" \
        "cvt.rn.satfinite.e5m2x2.f32   %0, %2, %1;\n" \
        "}" \
        : "=h"(out) : "f"(res_float[0]), "f"(res_float[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

namespace detail {

/// Special converters that can be used with 4 8-bit elements packed in a register.
/// Common use is for fast FP8 converters.
template <
  typename T,
  typename S,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename Transform = cutlass::transform::thread::UnaryTransform::Identity
>
struct NumericArrayConverterPacked4Element {
  using result_type = Array<T, 4>;
  using source_type = Array<S, 4>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result;
    NumericConverter<T, S, Round> convert_;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      if (platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value) {
        result[i] = convert_(s[i]);
      }
      else { // conjugate
        result[i] = conj(convert_(s[i]));
      }
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 4> <= Array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float, cutlass::float_e4m3_t, Round> {
  using result_element = float;
  using source_element = cutlass::float_e4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out_fp16[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "r"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 4> <= Array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e4m3_t, float, Round> {
  using result_element = cutlass::float_e4m3_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
        "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/// Partial specialization for Array<float, 4> <= Array<float_ue4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float, float_ue4m3_t, Round> {
  using result_element = float;
  using source_element = float_ue4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out_fp16[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "r"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_ue4m3_t, 4> <= Array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_ue4m3_t, float, Round> {
  using result_element = float_ue4m3_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
        "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, N> <=> Array<float_ue8m0_t, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float, 4> <= Array<float_ue8m0_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float, float_ue8m0_t, Round> {
  using result_element = float;
  using source_element = float_ue8m0_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  using BfloatArr = Array<cutlass::bfloat16_t, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_UE8M0_CVT_ENABLED)
    uint32_t out_fp16[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);
    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.bf16x2.ue8m0x2 %0, lo;\n" \
        "cvt.rn.bf16x2.ue8m0x2 %1, hi;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "r"(src_packed));

    NumericArrayConverter<float, cutlass::bfloat16_t, 2> bf2fp32_converter;
    auto res0 = bf2fp32_converter(reinterpret_cast<Array<cutlass::bfloat16_t, 2> &>(out_fp16[0]));
    auto res1 = bf2fp32_converter(reinterpret_cast<Array<cutlass::bfloat16_t, 2> &>(out_fp16[1]));

    result_type out;
    out[0] = res0[0];
    out[1] = res0[1];
    out[2] = res1[0];
    out[3] = res1[1];
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/// Partial specialization for Array<float_ue8m0_t, 4> <= Array<float, 4>
template <>
struct NumericArrayConverterPacked4Element<float_ue8m0_t, float, FloatRoundStyle::round_toward_infinity> {
  using result_element = float_ue8m0_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_infinity;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_UE8M0_CVT_ENABLED)
    uint32_t out;
    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rp.satfinite.ue8m0x2.f32   lo, %2, %1;\n" \
        "cvt.rp.satfinite.ue8m0x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, FloatRoundStyle::round_toward_infinity> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }
    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_ue8m0_t, 4> <= Array<float, 4>
template <>
struct NumericArrayConverterPacked4Element<float_ue8m0_t, float, FloatRoundStyle::round_toward_zero> {
  using result_element = float_ue8m0_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_UE8M0_CVT_ENABLED)
    uint32_t out;
    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rz.satfinite.ue8m0x2.f32   lo, %2, %1;\n" \
        "cvt.rz.satfinite.ue8m0x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, FloatRoundStyle::round_toward_zero> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_ue8m0_t, float, Round> {
  using result_element = float_ue8m0_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_infinity;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    //default maps to RP mode.
    return NumericArrayConverterPacked4Element<float_ue8m0_t, float, FloatRoundStyle::round_toward_infinity>{}(source);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, N> <=> Array<float_e2m3_unpack8bits_t, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float_e2m3_unpack8bits_t, 4> <= Array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<cutlass::detail::float_e2m3_unpack8bits_t, float, Round> {
  using result_element = cutlass::detail::float_e2m3_unpack8bits_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    uint32_t out;

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e2m3x2.f32   lo, %2, %1;\n" \
        "cvt.rn.satfinite.e2m3x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 4> <= Array<float_e2m3_unpack8bits_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float, cutlass::detail::float_e2m3_unpack8bits_t, Round> {
  using result_element = float;
  using source_element = cutlass::detail::float_e2m3_unpack8bits_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    uint32_t out_fp16[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e2m3x2 %0, lo;\n" \
        "cvt.rn.f16x2.e2m3x2 %1, hi;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "r"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, 4> <=> Array<float_e3m2_unpack8bits_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float_e3m2_unpack8bits_t, 4> <= Array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<cutlass::detail::float_e3m2_unpack8bits_t, float, Round> {
  using result_element = cutlass::detail::float_e3m2_unpack8bits_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    uint32_t out;

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e3m2x2.f32   lo, %2, %1;\n" \
        "cvt.rn.satfinite.e3m2x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/// Partial specialization for Array<float, 4> <= Array<float_e3m2_unpack8bits_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float, cutlass::detail::float_e3m2_unpack8bits_t, Round> {
  using result_element = float;
  using source_element = cutlass::detail::float_e3m2_unpack8bits_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    uint32_t out_fp16[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e3m2x2 %0, lo;\n" \
        "cvt.rn.f16x2.e3m2x2 %1, hi;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "r"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, 4> <=> Array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float, 4> <= Array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float, cutlass::float_e5m2_t, Round> {
  using result_element = float;
  using source_element = cutlass::float_e5m2_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out_fp16[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
        "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "r"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 4> <= Array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e5m2_t, float, Round> {
  using result_element = cutlass::float_e5m2_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e5m2x2.f32   lo, %2, %1;\n" \
        "cvt.rn.satfinite.e5m2x2.f32   hi, %4, %3;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<cutlass::half_t, 4> <=> Array<float_e4m3_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<cutlass::half_t, 4> <= Array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<cutlass::half_t, cutlass::float_e4m3_t, Round> {
  using result_element = cutlass::half_t;
  using source_element = cutlass::float_e4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);
    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
        "}\n" : "=r"(out[0]), "=r"(out[1]) : "r"(src_packed));
    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 4> <= Array<cutlass::half_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e4m3_t, cutlass::half_t, Round> {
  using result_element = cutlass::float_e4m3_t;
  using source_element = cutlass::half_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;
    uint32_t const* src_packed = reinterpret_cast<uint32_t const*>(&source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e4m3x2.f16x2   lo, %1;\n" \
        "cvt.rn.satfinite.e4m3x2.f16x2   hi, %2;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "r"(src_packed[0]), "r"(src_packed[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<cutlass::half_t, 4> <=> Array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<cutlass::half_t, 4> <= Array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<cutlass::half_t, cutlass::float_e5m2_t, Round> {
  using result_element = cutlass::half_t;
  using source_element = cutlass::float_e5m2_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out[2];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);
    asm volatile( \
        "{\n" \
        ".reg .b16 lo, hi;\n" \
        "mov.b32 {lo, hi}, %2;\n" \
        "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
        "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
        "}\n" : "=r"(out[0]), "=r"(out[1]) : "r"(src_packed));
    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 4> <= Array<cutlass::half_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e5m2_t, cutlass::half_t, Round> {
  using result_element = cutlass::float_e5m2_t;
  using source_element = cutlass::half_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out;
    uint32_t const* src_packed = reinterpret_cast<uint32_t const*>(&source);

    asm volatile( \
        "{\n" \
        ".reg .b16 lo;\n" \
        ".reg .b16 hi;\n" \
        "cvt.rn.satfinite.e5m2x2.f16x2   lo, %1;\n" \
        "cvt.rn.satfinite.e5m2x2.f16x2   hi, %2;\n" \
        "mov.b32 %0, {lo, hi};\n" \
        "}" \
        : "=r"(out) : "r"(src_packed[0]), "r"(src_packed[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<cutlass::bfloat16_t, 4> <=> Array<float_e4m3_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<cutlass::bfloat16_t, 4> <= Array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<cutlass::bfloat16_t, cutlass::float_e4m3_t, Round> {
  using result_element = cutlass::bfloat16_t;
  using source_element = cutlass::float_e4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    // Convert f8 to float
    NumericArrayConverterPacked4Element<float, source_element, Round> src2float;
    Array<float, 4> tmp_floats = src2float(source);

    // Convert float to bf16
    result_type out;
    Array<float, 2>* packed_tmp = reinterpret_cast<Array<float, 2>*>(&tmp_floats);
    Array<result_element, 2>* packed_out = reinterpret_cast<Array<result_element, 2>*>(&out);
    NumericArrayConverter<result_element, float, 2, Round> float2result;
    packed_out[0] = float2result(packed_tmp[0]);
    packed_out[1] = float2result(packed_tmp[1]);

    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 4> <= Array<cutlass::bfloat16_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e4m3_t, cutlass::bfloat16_t, Round> {
  using result_element = cutlass::float_e4m3_t;
  using source_element = cutlass::bfloat16_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    // Convert bf16 to float
    Array<float, 4> tmp;
    Array<float, 2>* packed_tmp = reinterpret_cast<Array<float, 2>*>(&tmp);
    Array<source_element, 2> const* packed_source = reinterpret_cast<Array<source_element, 2> const*>(&source);
    NumericArrayConverter<float, source_element, 2, Round> src2float;
    packed_tmp[0] = src2float(packed_source[0]);
    packed_tmp[1] = src2float(packed_source[1]);

    // Convert float to f8
    NumericArrayConverterPacked4Element<result_element, float, Round> float2result;
    return float2result(tmp);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<cutlass::bfloat16_t, 4> <=> Array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<cutlass::bfloat16_t, 4> <= Array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<cutlass::bfloat16_t, cutlass::float_e5m2_t, Round> {
  using result_element = cutlass::bfloat16_t;
  using source_element = cutlass::float_e5m2_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    // Convert f8 to float
    NumericArrayConverterPacked4Element<float, source_element, Round> src2float;
    Array<float, 4> tmp_floats = src2float(source);

    // Convert float to bf16
    result_type out;
    Array<float, 2>* packed_tmp = reinterpret_cast<Array<float, 2>*>(&tmp_floats);
    Array<result_element, 2>* packed_out = reinterpret_cast<Array<result_element, 2>*>(&out);
    NumericArrayConverter<result_element, float, 2, Round> float2result;
    packed_out[0] = float2result(packed_tmp[0]);
    packed_out[1] = float2result(packed_tmp[1]);

    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 4> <= Array<cutlass::bfloat16_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e5m2_t, cutlass::bfloat16_t, Round> {
  using result_element = cutlass::float_e5m2_t;
  using source_element = cutlass::bfloat16_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    // Convert bf16 to float
    Array<float, 4> tmp;
    Array<float, 2>* packed_tmp = reinterpret_cast<Array<float, 2>*>(&tmp);
    Array<source_element, 2> const* packed_source = reinterpret_cast<Array<source_element, 2> const*>(&source);
    NumericArrayConverter<float, source_element, 2, Round> src2float;
    packed_tmp[0] = src2float(packed_source[0]);
    packed_tmp[1] = src2float(packed_source[1]);

    // Convert float to f8
    NumericArrayConverterPacked4Element<result_element, float, Round> float2result;
    return float2result(tmp);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float_e4m3_t, 4> <=> Array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float_e4m3_t, 4> <= Array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e4m3_t, cutlass::float_e5m2_t, Round> {
  using result_element = cutlass::float_e4m3_t;
  using source_element = cutlass::float_e5m2_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 4> <= Array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e5m2_t, cutlass::float_e4m3_t, Round> {
  using result_element = cutlass::float_e5m2_t;
  using source_element = cutlass::float_e4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for:
//       Array<T, N> <=> Array<float_e4m3_t, N>
//       Array<T, N> <=> Array<float_e5m2_t, N>
// using packed converter under the hood
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S,
  int N,
  FloatRoundStyle Round
>
struct PackedNumericArrayConverter {
  using result_element = T;
  using source_element = S;

  using result_type = Array<result_element, N>;
  using source_type = Array<source_element, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using packed_result_type = Array<result_element, 4>;
  using packed_source_type = Array<source_element, 4>;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    packed_result_type* packed_result = reinterpret_cast<packed_result_type*>(&result);
    const packed_source_type* packed_source = reinterpret_cast<const packed_source_type*>(&source);

    detail::NumericArrayConverterPacked4Element<result_element, source_element, Round> packed_converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      packed_result[i] = packed_converter(packed_source[i]);
    }

    // Handle leftovers
    NumericConverter<result_element, source_element, Round> converter;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N % 4; ++i) {
      int idx = ((N / 4) * 4) + i;
      result[idx] = converter(source[idx]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const{
    return convert(s);
  }
};

/// Partial specialization for Array<T, N> <= Array<float_e4m3_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, cutlass::float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<T, cutlass::float_e4m3_t, N, Round> {};

/// Partial specialization for Array<T, N> <= Array<float_e5m2_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, cutlass::float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<T, cutlass::float_e5m2_t, N, Round> {};

/// Partial specialization for Array<float_e4m3_t, N> <= Array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, S, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, S, N, Round> {};

/// Partial specialization for Array<float_e5m2_t, N> <= Array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, S, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, S, N, Round> {};

/// Partial specialization for Array<float_e4m3_t, N> <= Array<float_e5m2_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, cutlass::float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, cutlass::float_e5m2_t, N, Round> {};

/// Partial specialization for Array<float_e5m2_t, N> <= Array<float_e4m3_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, cutlass::float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, cutlass::float_e4m3_t, N, Round> {};

/// Partial specialization for Array<float_e4m3_t, N> <= Array<float_e4m3_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, cutlass::float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, cutlass::float_e4m3_t, N, Round> {};

/// Partial specialization for Array<float_e5m2_t, N> <= Array<float_e5m2_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, cutlass::float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, cutlass::float_e5m2_t, N, Round> {};


/// Partial specialization for Array<float, 2> <= Array<float_ue8m0_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, float_ue8m0_t, 2, Round> {
  using result_element = float;
  using source_element = float_ue8m0_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_UE8M0_CVT_ENABLED)
    uint32_t out_fp16;
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);

    asm volatile( \
        "{\n" \
        "cvt.rn.bf16x2.ue8m0x2 %0, %1;\n" \
        "}\n" : "=r"(out_fp16): "h"(src_packed));

    NumericArrayConverter<float, cutlass::bfloat16_t, 2> bf2fp32_converter;
    auto res0 = bf2fp32_converter(reinterpret_cast<Array<cutlass::bfloat16_t, 2> &>(out_fp16));

    result_type out;
    out[0] = res0[0];
    out[1] = res0[1];
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_ue8m0_t, 2> <= Array<float, 2>
template <>
struct NumericArrayConverter<float_ue8m0_t, float, 2, FloatRoundStyle::round_toward_infinity> {
  using result_element = float_ue8m0_t;
  using source_element = float;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_infinity;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_UE8M0_CVT_ENABLED)
    uint16_t out;
    asm volatile( \
        "{\n" \
        "cvt.rp.satfinite.ue8m0x2.f32   %0, %2, %1;\n" \
        "}" \
        : "=h"(out) : "f"(source[0]), "f"(source[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, FloatRoundStyle::round_toward_infinity> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_ue8m0_t, 2> <= Array<float, 2>
template <>
struct NumericArrayConverter<float_ue8m0_t, float, 2, FloatRoundStyle::round_toward_zero> {
  using result_element = float_ue8m0_t;
  using source_element = float;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_UE8M0_CVT_ENABLED)
    uint16_t out;
    asm volatile( \
        "{\n" \
        "cvt.rz.satfinite.ue8m0x2.f32   %0, %2, %1;\n" \
        "}" \
        : "=h"(out) : "f"(source[0]), "f"(source[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, FloatRoundStyle::round_toward_zero> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_ue8m0_t, float, 2, Round> {
  using result_element = float_ue8m0_t;
  using source_element = float;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    return NumericArrayConverter<float_ue8m0_t, float, 2, FloatRoundStyle::round_toward_infinity>{}(source);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<T, N> <= Array<float_ue8m0_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, float_ue8m0_t, N, Round> :
  public PackedNumericArrayConverter<T, float_ue8m0_t, N, Round> {};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, 2> <=> Array<float_ue4m3_t, 2>
//
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Array<float, 2> <= Array<float_ue4m3_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, float_ue4m3_t, 2, Round> {
  using result_element = float;
  using source_element = float_ue4m3_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint32_t out_fp16;
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);

    asm volatile( \
        "{\n" \
        "cvt.rn.f16x2.e4m3x2 %0, %1;\n" \
        "}\n" : "=r"(out_fp16): "h"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_ue4m3_t, 2> <= Array<float, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_ue4m3_t, float, 2, Round> {
  using result_element = float_ue4m3_t;
  using source_element = float;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
    uint16_t out;

    asm volatile( \
        "{\n" \
        "cvt.rn.satfinite.e4m3x2.f32   %0, %2, %1;\n" \
        "}" \
        : "=h"(out) : "f"(source[0]), "f"(source[1]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_ue8m0_t, N> <= Array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_ue8m0_t, S, N, Round> :
  public PackedNumericArrayConverter<float_ue8m0_t, S, N, Round> {};
/// Partial specialization for Array<T, N> <= Array<float_ue4m3_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, cutlass::float_ue4m3_t, N, Round> :
  public PackedNumericArrayConverter<T, cutlass::float_ue4m3_t, N, Round> {};

// Partial specialization for Array<float_ue4m3_t, N> <= Array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::float_ue4m3_t, S, N, Round> :
  public PackedNumericArrayConverter<cutlass::float_ue4m3_t, S, N, Round> {};


/// Partial specialization for Array<T, N> <= Array<float_e2m3_unpack8bits_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, cutlass::detail::float_e2m3_unpack8bits_t, N, Round> :
  public PackedNumericArrayConverter<T, cutlass::detail::float_e2m3_unpack8bits_t, N, Round> {};


/// Partial specialization for Array<float_e2m3_unpack8bits_t, N> <= Array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::detail::float_e2m3_unpack8bits_t, S, N, Round> :
  public PackedNumericArrayConverter<cutlass::detail::float_e2m3_unpack8bits_t, S, N, Round> {};

/// Partial specialization for Array<float_e2m3_unpack8bits_t, N> <= Array<float_e2m3_unpack8bits_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::detail::float_e2m3_unpack8bits_t, cutlass::detail::float_e2m3_unpack8bits_t, N, Round> :
  public PackedNumericArrayConverter<cutlass::detail::float_e2m3_unpack8bits_t, cutlass::detail::float_e2m3_unpack8bits_t, N, Round> {};

/// Partial specialization for Array<T, N> <= Array<float_e3m2_unpack8bits_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, cutlass::detail::float_e3m2_unpack8bits_t, N, Round> :
  public PackedNumericArrayConverter<T, cutlass::detail::float_e3m2_unpack8bits_t, N, Round> {};

/// Partial specialization for Array<float_e3m2_unpack8bits_t, N> <= Array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::detail::float_e3m2_unpack8bits_t, S, N, Round> :
  public PackedNumericArrayConverter<cutlass::detail::float_e3m2_unpack8bits_t, S, N, Round> {};

/// Partial specialization for Array<float_e3m2_unpack8bits_t, N> <= Array<float_e3m2_unpack8bits_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<cutlass::detail::float_e3m2_unpack8bits_t, cutlass::detail::float_e3m2_unpack8bits_t, N, Round> :
  public PackedNumericArrayConverter<cutlass::detail::float_e3m2_unpack8bits_t, cutlass::detail::float_e3m2_unpack8bits_t, N, Round> {};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, N> <=> Array<float_e2m1_t, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float, 8> <= Array<float_e2m1_t, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, cutlass::float_e2m1_t, 8, Round> {
  using result_element = float;
  using source_element = cutlass::float_e2m1_t;

  using result_type = Array<result_element, 8>;
  using source_type = Array<source_element, 8>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    uint32_t out_fp16[4];
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);

    asm volatile( \
        "{\n" \
        ".reg .b8 byte0, byte1, byte2, byte3;\n" \
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n" \
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n" \
        "cvt.rn.f16x2.e2m1x2 %1, byte1;\n" \
        "cvt.rn.f16x2.e2m1x2 %2, byte2;\n" \
        "cvt.rn.f16x2.e2m1x2 %3, byte3;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) , "=r"(out_fp16[2]), "=r"(out_fp16[3]): "r"(src_packed));

    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));
    float2 res2 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[2]));
    float2 res3 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[3]));

    result_type out;
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    out[4] = res2.x;
    out[5] = res2.y;
    out[6] = res3.x;
    out[7] = res3.y;
    return out;
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float> <= Array<float_e2m1_t>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, cutlass::float_e2m1_t, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");

  using result_type = Array<float, N>;
  using source_type = Array<float_e2m1_t, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<float, cutlass::float_e2m1_t, 8, Round> convert_vector_;

    result_type result;

    Array<float, 8> *result_ptr = reinterpret_cast<Array<float, 8> *>(&result);
    Array<float_e2m1_t, 8> const *source_ptr = reinterpret_cast<Array<float_e2m1_t, 8> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/// Partial specialization for Array<float_e2m1_t, 2> <= Array<float, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e2m1_t, float, 2, Round> {
  using result_element = float_e2m1_t;
  using source_element = float;

  using result_type = Array<float_e2m1_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    uint32_t tmp;
    asm volatile( \
      "{\n" \
      ".reg .b8 byte0;\n" \
      ".reg .b8 byte1;\n" \
      ".reg .b8 byte2;\n" \
      ".reg .b8 byte3;\n" \
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n" \
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n" \
      "}" \
      : "=r"(tmp) : "f"(source[0]), "f"(source[1]));
    
    uint8_t out = (tmp & 0xff);

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e2m1_t, 8> <= Array<float, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e2m1_t, float, 8, Round> {
  using result_element = cutlass::float_e2m1_t;
  using source_element = float;

  using result_type = Array<float_e2m1_t, 8>;
  using source_type = Array<float, 8>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    unsigned out;
    asm volatile( \
      "{\n" \
      ".reg .b8 byte0;\n" \
      ".reg .b8 byte1;\n" \
      ".reg .b8 byte2;\n" \
      ".reg .b8 byte3;\n" \
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n" \
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n" \
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n" \
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n" \
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n" \
      "}" \
      : "=r"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]),
                    "f"(source[4]), "f"(source[5]), "f"(source[6]), "f"(source[7]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e2m1_t, 4> <= Array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e2m1_t, float, 4, Round> {
  using result_element = float_e2m1_t;
  using source_element = float;

  using result_type = Array<float_e2m1_t, 4>;
  using source_type = Array<float, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    uint16_t out;
    asm volatile( \
      "{\n" \
      ".reg .b8 byte0;\n" \
      ".reg .b8 byte1;\n" \
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n" \
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n" \
      "mov.b16 %0, {byte0, byte1};\n" \
      "}" \
      : "=h"(out) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]));

    return reinterpret_cast<result_type const &>(out);
  #else
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e2m1_t> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e2m1_t, float, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");

  using result_type = Array<float_e2m1_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<float_e2m1_t, float, 8, Round> convert_vector_;

    result_type result;

    Array<float_e2m1_t, 8> *result_ptr = reinterpret_cast<Array<float_e2m1_t, 8> *>(&result);
    Array<float, 8> const *source_ptr = reinterpret_cast<Array<float, 8> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<int8_t> <= Array<float>
/// Conversion is performed with saturation regardless of setting of
/// the `Round` template parameter.
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, float, 1, Round> {

  using result_type = Array<int8_t, 1>;
  using source_type = Array<float, 1>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericConverter<int8_t, float, Round> destination_converter;
    result_type result;
    result[0] = destination_converter(source[0]);
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, float, 1, Round> {

  using result_type = Array<uint8_t, 1>;
  using source_type = Array<float, 1>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericConverter<uint8_t, float, Round> destination_converter;
    result_type result;
    result[0] = destination_converter(source[0]);
    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

// To convert a FP32 to Int that has less than 32 bits, we need to convert it to int32 first.
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayFP32ToIntConverter {

  using result_type = Array<T, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(cutlass::platform::numeric_limits<T>::is_integer, "the dest type has to be int.");

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    // Convert float to int
    Array<int32_t, N> temporary;

    NumericArrayConverter<int32_t, float, N, Round> compute_converter;
    temporary = compute_converter(source);

    // Convert to int to int8_t
    NumericArrayConverter<T, int32_t, N, Round> destination_converter;
    return destination_converter(temporary);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, float, N, Round> {

  using result_type = Array<int8_t, N>;
  using source_type = Array<float, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<int8_t, N, Round> converter;
    return converter(source);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, float, N, Round> {

  using result_type = Array<uint8_t, N>;
  using source_type = Array<float, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<uint8_t, N, Round> converter;
    return converter(source);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, float, N, Round> {

  using result_type = Array<int4b_t, N>;
  using source_type = Array<float, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<int4b_t, N, Round> converter;
    return converter(source);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, float, N, Round> {

  using result_type = Array<uint4b_t, N>;
  using source_type = Array<float, N>;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<uint4b_t, N, Round> converter;
    return converter(source);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && \
    ((__CUDACC_VER_MAJOR__ > 10) ||                     \
     ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))

/// Partial specialization for Array<int4b_t, 8> <= Array<int, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, int, 8, Round> {

  using result_type = Array<int4b_t, 8>;
  using source_type = Array<int, 8>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
        "{ .reg .u32 r4;"
        "cvt.pack.sat.s4.s32.b32   r4, %8, %7, 0;"
        "cvt.pack.sat.s4.s32.b32   r4, %6, %5, r4;"
        "cvt.pack.sat.s4.s32.b32   r4, %4, %3, r4;"
        "cvt.pack.sat.s4.s32.b32   %0, %2, %1, r4;"
        "}"
        : "=r"(out)
        : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]),
          "r"(source[4]), "r"(source[5]), "r"(source[6]), "r"(source[7]));

    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int4b_t> <= Array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, int, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");

  using result_type = Array<int4b_t, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<int4b_t, int, 8, Round> convert_vector_;

    result_type result;

    Array<int4b_t, 8> *result_ptr = reinterpret_cast<Array<int4b_t, 8> *>(&result);
    Array<int, 8> const *source_ptr = reinterpret_cast<Array<int, 8> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<uint4b_t, 8> <= Array<int, 8>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, int, 8, Round> {

  using result_type = Array<uint4b_t, 8>;
  using source_type = Array<int, 8>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    unsigned out;

    asm volatile(
        "{ .reg .u32 r4;"
        "cvt.pack.sat.u4.s32.b32   r4, %8, %7, 0;"
        "cvt.pack.sat.u4.s32.b32   r4, %6, %5, r4;"
        "cvt.pack.sat.u4.s32.b32   r4, %4, %3, r4;"
        "cvt.pack.sat.u4.s32.b32   %0, %2, %1, r4;"
        "}"
        : "=r"(out)
        : "r"(source[0]), "r"(source[1]), "r"(source[2]), "r"(source[3]),
          "r"(source[4]), "r"(source[5]), "r"(source[6]), "r"(source[7]));

    return reinterpret_cast<result_type const &>(out);
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int4b_t> <= Array<int>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, int, N, Round> {
  static_assert(!(N % 8), "N must be multiple of 8.");

  using result_type = Array<uint4b_t, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<uint4b_t, int, 8, Round> convert_vector_;

    result_type result;

    Array<uint4b_t, 8> *result_ptr = reinterpret_cast<Array<uint4b_t, 8> *>(&result);
    Array<int, 8> const *source_ptr = reinterpret_cast<Array<int, 8> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#endif  // Conditional guards to enable partial specialization for packed integers

namespace detail {

  /*
      A helper class that can vectorize a numeric converter with implementation for several vector widths.

      The vector widths must be giving in decreasing order or width, and must be a power of 2.

      The vector converters must produce identical results to the scalar converters for consistency.
    */
  class VectorizedConverter {
  private:
    // Base case to handle remainder elements as scalars.
    template <int Offset, size_t ParentWidth, typename ArrayConverter>
    CUTLASS_DEVICE
    static void convert_helper(
      typename ArrayConverter::result_type& result,
      typename ArrayConverter::source_type const& source) {

      using ElementRes = typename ArrayConverter::result_type::Element;
      using ElementSrc = typename ArrayConverter::source_type::Element;
      // If no more converters, handle the remaining elements as scalars.
      constexpr int total_elements = ArrayConverter::result_type::kElements;
      constexpr int remainder = total_elements - Offset;
      static_assert(remainder == (total_elements % ParentWidth), "Unexpected remainder.");

      typename ArrayConverter::ScalarConverter scalar_converter;
      CUTLASS_PRAGMA_UNROLL
      for (int i = Offset; i < ArrayConverter::result_type::kElements; ++i) {
        result[i] = scalar_converter(ElementSrc(source[i]));
      }
    }

    template <int Offset, size_t ParentWidth, typename ArrayConverter, typename ResultVectorArray, typename SourceVectorArray, typename... OtherVectorArrays>
    CUTLASS_DEVICE
    static void convert_helper(typename ArrayConverter::result_type& result, typename ArrayConverter::source_type const& source) {
      static_assert(sizeof...(OtherVectorArrays) % 2 == 0, "Vector converters must come in {dst, src} pairs");
      static_assert(ResultVectorArray::kElements == SourceVectorArray::kElements, "Vector converters must have the same vector width");
      static_assert(cutlass::platform::is_same<typename ArrayConverter::result_type::Element, typename ResultVectorArray::Element>::value,
        "ResultVectorArray must have the same type ArrayConverter::result_type");
      static_assert(cutlass::platform::is_same<typename ArrayConverter::source_type::Element, typename SourceVectorArray::Element>::value,
        "SourceVectorArray must have the same type ArrayConverter::result_type");
      static_assert(Offset >= 0 && Offset <= ArrayConverter::result_type::kElements, "Offset must be between 0 and N");

      static_assert(ParentWidth == 0 || ParentWidth > ResultVectorArray::kElements, "Vector arrays must be given in decreasing order of width");

      constexpr int vector_width = ResultVectorArray::kElements;
      static_assert(ispow2(vector_width), "Vector width must be a power of 2");

      using ElementRes = typename ArrayConverter::result_type::Element;
      using ElementSrc = typename ArrayConverter::source_type::Element;

      constexpr int vector_bits_res = vector_width * cutlass::sizeof_bits<ElementRes>::value;
      constexpr int vector_bits_src = vector_width * cutlass::sizeof_bits<ElementSrc>::value;

      static_assert(vector_bits_res % 8 == 0, "Result vector type must be byte addressed.");
      static_assert(vector_bits_src % 8 == 0, "Source vector type must be byte addressed.");

      constexpr int vector_offset = Offset / vector_width;
      ResultVectorArray* packed_result_vec = reinterpret_cast<ResultVectorArray*>(&result) + vector_offset;
      SourceVectorArray const* packed_source_vec = reinterpret_cast<SourceVectorArray const*>(&source) + vector_offset;

      // Convert the remaining elements as vectors.
      constexpr int total_elements = ArrayConverter::result_type::kElements;
      constexpr int groups_of_vec = (total_elements - Offset) / vector_width;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < groups_of_vec; ++i) {
        packed_result_vec[i] = ArrayConverter::template packed_convert<ResultVectorArray, SourceVectorArray>(packed_source_vec[i]);
      }

      constexpr int new_offset = Offset + vector_width * groups_of_vec;
      // Recurse to handle other vector converters, or the scalar base case.
      convert_helper<new_offset, ResultVectorArray::kElements, ArrayConverter, OtherVectorArrays...>(result, source);
    }

  public:
    /*
        A method to convert vectors of elements using the packed_convert method of the converter.

        Converters using this class must implement packed convert and support 1 or more vector conversions.
      */
    template <typename ArrayConverter, typename ResultVectorArray, typename SourceVectorArray, typename... OtherVectorArrays>
    CUTLASS_DEVICE
    static void convert(typename ArrayConverter::result_type& result, typename ArrayConverter::source_type const& source) {
      convert_helper<0, 0, ArrayConverter, ResultVectorArray, SourceVectorArray, OtherVectorArrays...>(result, source);
    }
  };
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Array<half_t, N> <= Array<float_e2m1_t, N>
template <
  FloatRoundStyle Round,
  int N
>
struct NumericArrayConverter<cutlass::half_t, cutlass::float_e2m1_t, N, Round> {
  using result_element = cutlass::half_t;
  using source_element = cutlass::float_e2m1_t;
  using result_type = Array<result_element, N>;
  using source_type = Array<source_element, N>;
  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<cutlass::half_t, 8>;
  using result_type_packed_4 = Array<cutlass::half_t, 4>;
  using result_type_packed_2 = Array<cutlass::half_t, 2>;
  using source_type_packed_8 = Array<cutlass::float_e2m1_t, 8>;
  using source_type_packed_4 = Array<cutlass::float_e2m1_t, 4>;
  using source_type_packed_2 = Array<cutlass::float_e2m1_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::half_t, cutlass::float_e2m1_t, Round>;

  #if defined(CUDA_PTX_FP8_CVT_ENABLED)
  CUTLASS_DEVICE
  static result_type_packed_8 ptx_convert(source_type_packed_8 const &source) {
    result_type_packed_8 out;
    uint32_t* out_fp16 = reinterpret_cast<uint32_t*>(&out);
    uint32_t const& src_packed = reinterpret_cast<uint32_t const&>(source);
    asm volatile( \
        "{\n" \
        ".reg .b8 byte0, byte1, byte2, byte3;\n" \
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n" \
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n" \
        "cvt.rn.f16x2.e2m1x2 %1, byte1;\n" \
        "cvt.rn.f16x2.e2m1x2 %2, byte2;\n" \
        "cvt.rn.f16x2.e2m1x2 %3, byte3;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) , "=r"(out_fp16[2]), "=r"(out_fp16[3]): "r"(src_packed));
    return out;
  }

  CUTLASS_DEVICE
  static result_type_packed_4 ptx_convert(source_type_packed_4 const &source) {
    result_type_packed_4 out;
    uint32_t* out_fp16 = reinterpret_cast<uint32_t*>(&out);
    uint16_t const& src_packed = reinterpret_cast<uint16_t const&>(source);
    asm volatile( \
        "{\n" \
        ".reg .b8 byte0, byte1;\n" \
        "mov.b16 {byte0, byte1}, %2;\n" \
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n" \
        "cvt.rn.f16x2.e2m1x2 %1, byte1;\n" \
        "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]) : "h"(src_packed));
    return out;
  }

  CUTLASS_DEVICE
  static result_type_packed_2 ptx_convert(source_type_packed_2 const &source) {
    result_type_packed_2 out;
    uint32_t* out_fp16 = reinterpret_cast<uint32_t*>(&out);
    uint16_t const& src_packed = static_cast<uint16_t const&>(reinterpret_cast<uint8_t const&>(source));
    asm volatile( \
        "{\n" \
        ".reg .b8 byte0, byte1;\n" \
        "mov.b16 {byte0, byte1}, %1;\n" \
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n" \
        "}\n" : "=r"(out_fp16[0]) : "h"(src_packed));
    return out;
  }
  #endif

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {
    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2, 4 or 8 to use private convert dispatch.");

  #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
    return ptx_convert(source);
  #else
    PackedResultType result;
    NumericConverter<result_element, source_element, Round> converter;

    const int k_packed = PackedResultType::kElements;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < k_packed; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  #endif
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::float_e4m3_t, N> <= Array<cutlass::int2b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e4m3_t, cutlass::int2b_t, N, Round> {
  using result_type = Array<cutlass::float_e4m3_t, N>;
  using source_type = Array<cutlass::int2b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_16 = Array<cutlass::float_e4m3_t, 16>;
  using result_type_packed_8 = Array<cutlass::float_e4m3_t, 8>;
  using source_type_packed_16 = Array<cutlass::int2b_t, 16>;
  using source_type_packed_8 = Array<cutlass::int2b_t, 8>;

  using ScalarConverter = NumericConverter<cutlass::float_e4m3_t, cutlass::int2b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_16 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_16>::value &&
                   platform::is_same<PackedResultType, result_type_packed_16>::value),
                  "Invalid PackedSrcType/PackedResultType must be 8 or 16 to use private convert dispatch.");

    // Hold output FP8s in reg. We need 1 reg for every 4 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 2;

    src_reg         &= 0x333333333333; // s14s12s10s8s6s4s2s0
    src_reg_shifted &= 0x333333333333; // s15s13s11s9s7s5s3s1

    // [0, 1, -2, -1] encoded as FP8
    static constexpr uint32_t E4M3_LUT = 0xB8C03800;

    const int iters = PackedSrcType::kElements / 4;
    #pragma unroll
    for (int ii = 0; ii < iters; ii += 2, src_reg >>= 16, src_reg_shifted >>= 16) {
      // This uses a look up table to convert packed int2s to packed fp8s, using the int4 value
      // as the index to prmt.
      // It first select both the positive and negative candidates, then uses the sign bit to
      // select the correct candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 f8_6420, f8_7531;\n"
          "  prmt.b32 f8_6420, %4, 0, %2;\n"
          "  prmt.b32 f8_7531, %4, 0, %3;\n"
          "  prmt.b32 %0, f8_6420, f8_7531, 0x5140;\n" // 3210
          "  prmt.b32 %1, f8_6420, f8_7531, 0x7362;\n" // 7654
          "}\n"
          : "=r"(r[ii]), "=r"(r[ii+1])
          : "r"(src_reg), "r"(src_reg_shifted), "n"(E4M3_LUT));
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_16, source_type_packed_16,
                                         result_type_packed_8, source_type_packed_8>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::float_e4m3_t, N> <= Array<cutlass::uint2b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e4m3_t, cutlass::uint2b_t, N, Round> {
  using result_type = Array<cutlass::float_e4m3_t, N>;
  using source_type = Array<cutlass::uint2b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_16 = Array<cutlass::float_e4m3_t, 16>;
  using result_type_packed_8 = Array<cutlass::float_e4m3_t, 8>;
  using source_type_packed_16 = Array<cutlass::uint2b_t, 16>;
  using source_type_packed_8 = Array<cutlass::uint2b_t, 8>;

  using ScalarConverter = NumericConverter<cutlass::float_e4m3_t, cutlass::uint2b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_16 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_16>::value &&
                   platform::is_same<PackedResultType, result_type_packed_16>::value),
                  "Invalid PackedSrcType/PackedResultType must be 8 or 16 to use private convert dispatch.");

    // Hold output FP8s in reg. We need 1 reg for every 4 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 2;

    src_reg         &= 0x333333333333; // u14u12u10u8u6u4u2u0
    src_reg_shifted &= 0x333333333333; // u15u13u11u9u7u5u3u1

    // [0, 1, 2, 3] encoded as FP8
    static constexpr uint32_t E4M3_LUT = 0x44403800;

    const int iters = PackedSrcType::kElements / 4;
    #pragma unroll
    for (int ii = 0; ii < iters; ii += 2, src_reg >>= 16, src_reg_shifted >>= 16) {
      // This uses a look up table to convert packed uint2s to packed fp8s, using the int4 value
      // as the index to prmt.
      // It first select both the positive and negative candidates, then uses the sign bit to
      // select the correct candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 f8_6420, f8_7531;\n"
          "  prmt.b32 f8_6420, %4, 0, %2;\n"
          "  prmt.b32 f8_7531, %4, 0, %3;\n"
          "  prmt.b32 %0, f8_6420, f8_7531, 0x5140;\n" // 3210
          "  prmt.b32 %1, f8_6420, f8_7531, 0x7362;\n" // 7654
          "}\n"
          : "=r"(r[ii]), "=r"(r[ii+1])
          : "r"(src_reg), "r"(src_reg_shifted), "n"(E4M3_LUT));
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_16, source_type_packed_16,
                                         result_type_packed_8, source_type_packed_8>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::float_e5m2_t, N> <= Array<cutlass::int2b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e5m2_t, cutlass::int2b_t, N, Round> {
  using result_type = Array<cutlass::float_e5m2_t, N>;
  using source_type = Array<cutlass::int2b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_16 = Array<cutlass::float_e5m2_t, 16>;
  using result_type_packed_8 = Array<cutlass::float_e5m2_t, 8>;
  using source_type_packed_16 = Array<cutlass::int2b_t, 16>;
  using source_type_packed_8 = Array<cutlass::int2b_t, 8>;

  using ScalarConverter = NumericConverter<cutlass::float_e5m2_t, cutlass::int2b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_16 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_16>::value &&
                   platform::is_same<PackedResultType, result_type_packed_16>::value),
                  "Invalid PackedSrcType/PackedResultType must be 8 or 16 to use private convert dispatch.");

    // Hold output FP8s in reg. We need 1 reg for every 4 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 2;

    src_reg         &= 0x333333333333; // s14s12s10s8s6s4s2s0
    src_reg_shifted &= 0x333333333333; // s15s13s11s9s7s5s3s1

    // [0, 1, -2, -1] encoded as FP8
    static constexpr uint32_t E4M3_LUT = 0xBCC03C00;

    const int iters = PackedSrcType::kElements / 4;
    #pragma unroll
    for (int ii = 0; ii < iters; ii += 2, src_reg >>= 16, src_reg_shifted >>= 16) {
      // This uses a look up table to convert packed int2s to packed fp8s, using the int4 value
      // as the index to prmt.
      // It first select both the positive and negative candidates, then uses the sign bit to
      // select the correct candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 f8_6420, f8_7531;\n"
          "  prmt.b32 f8_6420, %4, 0, %2;\n"
          "  prmt.b32 f8_7531, %4, 0, %3;\n"
          "  prmt.b32 %0, f8_6420, f8_7531, 0x5140;\n" // 3210
          "  prmt.b32 %1, f8_6420, f8_7531, 0x7362;\n" // 7654
          "}\n"
          : "=r"(r[ii]), "=r"(r[ii+1])
          : "r"(src_reg), "r"(src_reg_shifted), "n"(E4M3_LUT));
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_16, source_type_packed_16,
                                         result_type_packed_8, source_type_packed_8>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::float_e5m2_t, N> <= Array<cutlass::uint2b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e5m2_t, cutlass::uint2b_t, N, Round> {
  using result_type = Array<cutlass::float_e5m2_t, N>;
  using source_type = Array<cutlass::uint2b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_16 = Array<cutlass::float_e5m2_t, 16>;
  using result_type_packed_8 = Array<cutlass::float_e5m2_t, 8>;
  using source_type_packed_16 = Array<cutlass::uint2b_t, 16>;
  using source_type_packed_8 = Array<cutlass::uint2b_t, 8>;

  using ScalarConverter = NumericConverter<cutlass::float_e5m2_t, cutlass::uint2b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_16 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_16>::value &&
                   platform::is_same<PackedResultType, result_type_packed_16>::value),
                  "Invalid PackedSrcType/PackedResultType must be 8 or 16 to use private convert dispatch.");

    // Hold output FP8s in reg. We need 1 reg for every 4 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 2;

    src_reg         &= 0x333333333333; // u14u12u10u8u6u4u2u0
    src_reg_shifted &= 0x333333333333; // u15u13u11u9u7u5u3u1

    // [0, 1, 2, 3] encoded as FP8
    static constexpr uint32_t E4M3_LUT = 0x42403C00;

    const int iters = PackedSrcType::kElements / 4;
    #pragma unroll
    for (int ii = 0; ii < iters; ii += 2, src_reg >>= 16, src_reg_shifted >>= 16) {
      // This uses a look up table to convert packed uint2s to packed fp8s, using the int4 value
      // as the index to prmt.
      // It first select both the positive and negative candidates, then uses the sign bit to
      // select the correct candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 f8_6420, f8_7531;\n"
          "  prmt.b32 f8_6420, %4, 0, %2;\n"
          "  prmt.b32 f8_7531, %4, 0, %3;\n"
          "  prmt.b32 %0, f8_6420, f8_7531, 0x5140;\n" // 3210
          "  prmt.b32 %1, f8_6420, f8_7531, 0x7362;\n" // 7654
          "}\n"
          : "=r"(r[ii]), "=r"(r[ii+1])
          : "r"(src_reg), "r"(src_reg_shifted), "n"(E4M3_LUT));
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_16, source_type_packed_16,
                                         result_type_packed_8, source_type_packed_8>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<int8_t> <= Array<int4b_t>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, int4b_t, N, Round> {

  static_assert(N % 8 == 0, "N must be a multiple of 8");

  using result_type = Array<int8_t, N>;
  using source_type = Array<int4b_t, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
   
    #if defined(__CUDA_ARCH__)

    if constexpr ( N == 8 ) {
      
      unsigned const& storage = reinterpret_cast<unsigned const &>(source);
      unsigned out[2];

      asm volatile(
          "{\n"
          "  .reg .u32 tmp0, tmp1, tmp2;\n"
          "  shl.b32 tmp0, %2, 4;\n"                // tmp0 = x1x2x3x4x5x6x7__
          "  and.b32 tmp0, tmp0, 0xf0f0f0f0;\n"     // tmp0 = x1__x3__x5__x7__
          "  prmt.b32 tmp1, tmp0, tmp0, 0xba98;\n"  // tmp1 = s1s3s5s7
          "  and.b32 tmp1, tmp1, 0xf0f0f0f0;\n"     // tmp1 = s1__s3__s5__s7__
          "  shr.u32 tmp0, tmp0, 4;\n"              // tmp0 = __x1__x3__x5__x7
          "  or.b32 tmp2, tmp0, tmp1;\n"            // tmp2 = y1y3y5y7
          "  and.b32 tmp0, %2, 0xf0f0f0f0;\n"       // tmp0 = x0__x2__x4__x6__
          "  prmt.b32 tmp1, tmp0, tmp0, 0xba98;\n"  // tmp1 = s0s2s4s6
          "  and.b32 tmp1, tmp1, 0xf0f0f0f0;\n"     // tmp1 = s0__s2__s4__s6__
          "  shr.u32 tmp0, tmp0, 4;\n"              // tmp0 = __x0__x2__x4__x6
          "  or.b32 tmp0, tmp0, tmp1;\n"            // tmp0 = y0y2y4y6
          "  prmt.b32 %0, tmp2, tmp0, 0x5140;\n"    // %0 = y0y1y2y3
          "  prmt.b32 %1, tmp2, tmp0, 0x7362;\n"    // %1 = y4y5y6y7
          "}\n"
          : "=r"(out[0]), "=r"(out[1])
          : "r"(storage));

      return reinterpret_cast<result_type const &>(out);
      
    } else {
      
      NumericArrayConverter<int8_t, int4b_t, 8, Round> convert_vector_;
      
      result_type result;
      
      Array<int8_t, 8> *result_ptr = reinterpret_cast<Array<int8_t, 8> *>(&result);
      Array<int4b_t, 8> const *source_ptr = reinterpret_cast<Array<int4b_t, 8> const *>(&source);
      
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N / 8; ++i) {
        result_ptr[i] = convert_vector_(source_ptr[i]);
      }
      
      return result;
    }
    
    #else
    
    result_type result;
    NumericConverter<int8_t, int4b_t, Round> convert_;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = convert_(source[i]);
    }
    
    return result;
    
    #endif // __CUDA_ARCH__
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::float_e4m3_t, N> <= Array<cutlass::int4b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e4m3_t, cutlass::int4b_t, N, Round> {
  using result_type = Array<cutlass::float_e4m3_t, N>;
  using source_type = Array<cutlass::int4b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<cutlass::float_e4m3_t, 8>;
  using result_type_packed_4 = Array<cutlass::float_e4m3_t, 4>;
  using source_type_packed_8 = Array<cutlass::int4b_t, 8>;
  using source_type_packed_4 = Array<cutlass::int4b_t, 4>;

  using ScalarConverter = NumericConverter<cutlass::float_e4m3_t, cutlass::int4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses a lookup table to converts i4 -> e4m3.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 4 or 8 to use private convert dispatch.");

    // Hold FP8 outputs in reg. We need 1 reg for every 4 outputs.
    cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)> r;

    // View the input as reg
    uint32_t reg = to_reg(source);

    // Determines if to get from the signed or unsigned candidates
    uint32_t sign = (reg & 0x88888888) >> 1;

    // Ignore sign bit when indexing into LUT
    uint32_t lut_idx = (reg & 0x77777777);

    // Signed is OR'd with 0x32103210 to find the correct value in the LUT
    const uint32_t final_prmt_base = 0x32103210;

    // [0, 1, 2, 3] encoded as FP8
    static constexpr uint32_t POS_E4M3s_REG1 = 0x44403800;
    // [4, 5, 6, 7] encoded as FP8
    static constexpr uint32_t POS_E4M3s_REG2 = 0x4E4C4A48;
    // [-8, -7, -6, -5] encoded as FP8
    static constexpr uint32_t NEG_E4M3s_REG1 = 0xCACCCED0;
    // [-4, -3, -2, -1] encoded as FP8
    static constexpr uint32_t NEG_E4M3s_REG2 = 0xB8C0C4C8;


    const int iters = PackedSrcType::kElements / 4;
    #pragma unroll
    for (int ii = 0; ii < iters; ++ii, lut_idx >>=16, sign >>=16) {
      uint32_t final_prmt_idx = final_prmt_base | sign;

      // This uses a look up table to convert packed int4s to packed fp8s, using the int4 value
      // as the index to prmt.
      // It first select both the positive and negative candidates, then uses the sign bit to
      // select the correct candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 pos_f8s, neg_f8s;\n"
          "  prmt.b32 pos_f8s, %1, %2, %5;\n"
          "  prmt.b32 neg_f8s, %3, %4, %5;\n"
          "  prmt.b32 %0, pos_f8s, neg_f8s, %6;\n"
          "}\n"
          : "=r"(r[ii])
          : "n"(POS_E4M3s_REG1), "n"(POS_E4M3s_REG2), "n"(NEG_E4M3s_REG1), "n"(NEG_E4M3s_REG2),
            "r"(lut_idx), "r"(final_prmt_idx));
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4>(result, source);

    return result;
  }


  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::float_e5m2_t, N> <= Array<cutlass::int4b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e5m2_t, cutlass::int4b_t, N, Round> {
  using result_type = Array<cutlass::float_e5m2_t, N>;
  using source_type = Array<cutlass::int4b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<cutlass::float_e5m2_t, 8>;
  using result_type_packed_4 = Array<cutlass::float_e5m2_t, 4>;
  using source_type_packed_8 = Array<cutlass::int4b_t, 8>;
  using source_type_packed_4 = Array<cutlass::int4b_t, 4>;

  using ScalarConverter = NumericConverter<cutlass::float_e5m2_t, cutlass::int4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses a lookup table to converts i4 -> e5m2.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 4 or 8 to use private convert dispatch.");

    // Hold FP8 outputs in reg. We need 1 reg for every 4 outputs.
    cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)> r;

    // View the input as reg
    uint32_t reg = to_reg(source);

    // Determines if to get from the signed or unsigned candidates
    uint32_t sign = (reg & 0x88888888) >> 1;

    // Ignore sign bit when indexing into LUT
    uint32_t lut_idx = (reg & 0x77777777);

    // Signed is OR'd with 0x32103210 to find the correct value in the LUT
    const uint32_t final_prmt_base = 0x32103210;

    // [0, 1, 2, 3] encoded as FP8
    static constexpr uint32_t POS_E5M2s_REG1 = 0x42403C00;
    // [4, 5, 6, 7] encoded as FP8
    static constexpr uint32_t POS_E5M2s_REG2 = 0x47464544;
    // [-8, -7, -6, -5] encoded as FP8
    static constexpr uint32_t NEG_E5M2s_REG1 = 0xC5C6C7C8;
    // [-4, -3, -2, -1] encoded as FP8
    static constexpr uint32_t NEG_E5M2s_REG2 = 0xBCC0C2C4;


    const int iters = PackedSrcType::kElements / 4;
    #pragma unroll
    for (int ii = 0; ii < iters; ++ii, lut_idx >>=16, sign >>=16) {
      uint32_t final_prmt_idx = final_prmt_base | sign;

      // This uses a look up table to convert packed int4s to packed fp8s, using the int4 value
      // as the index to prmt.
      // It first select both the positive and negative candidates, then uses the sign bit to
      // select the correct candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 pos_f8s, neg_f8s;\n"
          "  prmt.b32 pos_f8s, %1, %2, %5;\n"
          "  prmt.b32 neg_f8s, %3, %4, %5;\n"
          "  prmt.b32 %0, pos_f8s, neg_f8s, %6;\n"
          "}\n"
          : "=r"(r[ii])
          : "n"(POS_E5M2s_REG1), "n"(POS_E5M2s_REG2), "n"(NEG_E5M2s_REG1), "n"(NEG_E5M2s_REG2),
            "r"(lut_idx), "r"(final_prmt_idx));
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4>(result, source);

    return result;
  }


  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::float_e4m3_t, N> <= Array<cutlass::uint4b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::float_e4m3_t, cutlass::uint4b_t, N, Round> {
  using result_type = Array<cutlass::float_e4m3_t, N>;
  using source_type = Array<cutlass::uint4b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<cutlass::float_e4m3_t, 8>;
  using result_type_packed_4 = Array<cutlass::float_e4m3_t, 4>;
  using source_type_packed_8 = Array<cutlass::uint4b_t, 8>;
  using source_type_packed_4 = Array<cutlass::uint4b_t, 4>;

  using ScalarConverter = NumericConverter<cutlass::float_e4m3_t, cutlass::uint4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses a lookup table to converts u4 -> e4m3.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 4 or 8 to use private convert dispatch.");

    // Hold FP8 outputs in reg. We need 1 reg for every 4 outputs.
    cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 4, sizeof(PackedResultType)> r;

    // View the input as reg
    uint32_t reg = to_reg(source);

    // Determines if to get from the [0-7] or [8-15] candidates
    uint32_t sign = (reg & 0x88888888) >> 1;

    // Ignore sign bit when indexing into LUT
    uint32_t lut_idx = (reg & 0x77777777);

    // Signed is OR'd with 0x32103210 to find the correct value in the LUT
    const uint32_t final_prmt_base = 0x32103210;

    // [0, 1, 2, 3] encoded as FP8
    static constexpr uint32_t E4M3s_REG1 = 0x44403800;
    // [4, 5, 6, 7] encoded as FP8
    static constexpr uint32_t E4M3s_REG2 = 0x4E4C4A48;
    // [8, 9, 10, 11] encoded as FP8
    static constexpr uint32_t E4M3s_REG3 = 0x53525150;
    // [12, 13, 14, 15] encoded as FP8
    static constexpr uint32_t E4M3s_REG4 = 0x57565554;


    const int iters = PackedSrcType::kElements / 4;
    #pragma unroll
    for (int ii = 0; ii < iters; ++ii, lut_idx >>=16, sign >>=16) {
      uint32_t final_prmt_idx = final_prmt_base | sign;

      // This uses a look up table to convert packed int4s to packed fp8s, using the int4 value
      // as the index to prmt.
      // It first select both the positive and negative candidates, then uses the sign bit to
      // select the correct candidate.
      asm volatile(
          "{\n"
          "  .reg .b32 f8s_1, f8s_2;\n"
          "  prmt.b32 f8s_1, %1, %2, %5;\n"
          "  prmt.b32 f8s_2, %3, %4, %5;\n"
          "  prmt.b32 %0, f8s_1, f8s_2, %6;\n"
          "}\n"
          : "=r"(r[ii])
          : "n"(E4M3s_REG1), "n"(E4M3s_REG2), "n"(E4M3s_REG3), "n"(E4M3s_REG4),
            "r"(lut_idx), "r"(final_prmt_idx));
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4>(result, source);

    return result;
  }


  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, N> <= Array<cutlass::int4b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<float, cutlass::int4b_t, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<cutlass::int4b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<float, 8>;
  using result_type_packed_4 = Array<float, 4>;
  using result_type_packed_2 = Array<float, 2>;
  using source_type_packed_8 = Array<cutlass::int4b_t, 8>;
  using source_type_packed_4 = Array<cutlass::int4b_t, 4>;
  using source_type_packed_2 = Array<cutlass::int4b_t, 2>;

  using ScalarConverter = NumericConverter<float, cutlass::int4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <int offset, int elements_to_convert, typename PackedResultType>
  CUTLASS_DEVICE
  static void packed_convert_vec(PackedResultType& result, uint32_t src_reg) {
    static_assert(offset == 0 || offset == 4, "Invalid offset");
    // Selects one of the bottom int4s and constructs:
    // 8388608 + (x + 8)
    // 8388608 + 16 * (x + 8)
    // 8388608 + 256 * (x + 8)
    // 8388608 + 4096 * (x + 8)
    uint32_t const and_masks[4] = {0x0000000F, 0x000000F0, 0x00000F00, 0x0000F000};
    uint32_t const xor_masks[4] = {0x4B000008, 0x4B000080, 0x4B000800, 0x4B008000};

    float const scales[4] = {1.f, 1.f / 16.f, 1.f / 256.f, 1.f / 4096.f};
    float const offsets[4] = {-8388616.f, -524296.f, -32776.f, -2056.f};

    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    uint32_t* result_as_int = reinterpret_cast<uint32_t*>(&result);

    // For each operand, computes:
    // r[i] = (r[i] & and_mask) ^ xor_mask
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < elements_to_convert; ++ii) {
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %1, %2, %3, %4;\n"
          "}\n"
          : "=r"(result_as_int[offset + ii])
          : "r"(src_reg), "r"(and_masks[ii]), "r"(xor_masks[ii]), "n"(immLut));

      result[offset + ii] = __fmaf_rn(result[offset + ii], scales[ii], offsets[ii]);
    }
  }

  // The core converter uses bit tricks to construct a known FP16 number, then does a
  // subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 1, 2, 4 or 8 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    PackedResultType r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    constexpr int total_elements = PackedResultType::kElements == 8 ? 4 : PackedResultType::kElements;
    packed_convert_vec<0, total_elements>(r, src_reg);


    if (PackedResultType::kElements == 8) {
      uint32_t src_reg_shifted = src_reg >> 16;
      packed_convert_vec<4, 4>(r, src_reg_shifted);
    }
    return r;
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, N> <= Array<int8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<float, int8_t, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<int8_t, N>;
  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_4 = Array<float, 4>;
  using result_type_packed_2 = Array<float, 2>;
  using source_type_packed_4 = Array<int8_t, 4>;
  using source_type_packed_2 = Array<int8_t, 2>;

  using ScalarConverter = NumericConverter<float, int8_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  CUTLASS_DEVICE
  static int32_t to_int32(source_type_packed_2 const& source) {
    return static_cast<int32_t>(reinterpret_cast<const int16_t&>(source));
  }

  CUTLASS_DEVICE
  static int32_t to_int32(source_type_packed_4 const& source) {
    return reinterpret_cast<const int32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2 or 4 to use private convert dispatch.");

    PackedResultType r;
  #if defined __CUDA_ARCH__ && __CUDA_ARCH__ <= 800
    // View the input as reg
    uint32_t src_reg = to_reg(source);
    static constexpr int fp32_base = 0x4B400000;
    uint32_t const prmt_indices[4] = {0x8880, 0x9991, 0xAAA2, 0xBBB3};

    int* result_as_int = reinterpret_cast<int*>(&r);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < PackedResultType::kElements; ++ii) {
      asm volatile("prmt.b32 %0,%1,%1,%2;\n" : "=r"(result_as_int[ii]) : "r"(src_reg), "r"(prmt_indices[ii]));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < PackedResultType::kElements; ++ii)
    {
      result_as_int[ii] += fp32_base;
      r[ii] -= reinterpret_cast<const float&>(fp32_base);
    }
  #else
    int32_t x = to_int32(source);
    int32_t t[4];
    constexpr int32_t mask[4] = {0x00000001, 0x00000100, 0x00010000, 0x01000000};

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < PackedResultType::kElements; ++ii) {
      t[ii] = __dp4a(x, mask[ii], 0);
      r[ii] = static_cast<float>(t[ii]);
    }
  #endif

    return r;
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, N> <= Array<uint8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<float, uint8_t, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<uint8_t, N>;
  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_4 = Array<float, 4>;
  using result_type_packed_2 = Array<float, 2>;
  using source_type_packed_4 = Array<uint8_t, 4>;
  using source_type_packed_2 = Array<uint8_t, 2>;

  using ScalarConverter = NumericConverter<float, uint8_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2 or 4 to use private convert dispatch.");

    PackedResultType r;
    // View the input as reg
    uint32_t src_reg = to_reg(source);

    // __byte_perm simulates the add.u32 0x4B000000 to every u8 element of u8x4 source and stores
    // the result in r (without introducing extra cvt.u32.u8 instruction)
    uint32_t const prmt_indices[4] = {0x7650, 0x7651, 0x7652, 0x7653};
    uint32_t* result_as_int = reinterpret_cast<uint32_t*>(&r);
    for (int ii = 0; ii < PackedResultType::kElements; ++ii) {
      result_as_int[ii] = __byte_perm(src_reg, 0x4B000000, prmt_indices[ii]);
      // Subtract the magic number 0x4B000000 from tmp in floating-point arithmetic to obtain final result
      r[ii] -= 8388608.f;
    }

    return r;
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Array<cutlass::half_t, N> <= Array<cutlass::int2b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, cutlass::int2b_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<cutlass::int2b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_16 = Array<cutlass::half_t, 16>;
  using result_type_packed_8 = Array<cutlass::half_t, 8>;
  using result_type_packed_4 = Array<cutlass::half_t, 4>;
  using source_type_packed_16 = Array<cutlass::int2b_t, 16>;
  using source_type_packed_8 = Array<cutlass::int2b_t, 8>;
  using source_type_packed_4 = Array<cutlass::int2b_t, 4>;

  using ScalarConverter = NumericConverter<cutlass::half_t, cutlass::int2b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_16 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_16>::value &&
                   platform::is_same<PackedResultType, result_type_packed_16>::value),
                  "Invalid PackedSrcType/PackedResultType must be 4, 8 or 16 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 4;

    // Below constructs the following temporary:
    // f1f0   = {0x00, i3i2i1i0,     0x00, i3i2i1i0}
    // f3f2   = {0x00, i5i4i3i2,     0x00, i5i4i3i2}
    // f5f4   = {0x00, i7i6i5i4,     0x00, i7i6i5i4}
    // f7f6   = {0x00, i9i8i7i6,     0x00, i9i8i7i6}
    // f9f8   = {0x00, i11i10i9i8,   0x00, i11i10i9i8}
    // f11f10 = {0x00, i13i12i11i10, 0x00, i13i12i11i10}
    // f13f12 = {0x00, i15i14i13i12, 0x00, i15i14i13i12}
    // f15f14 = {0x00, 0000i15i14,   0x00, 0000i15i14}
    // We use inline asm instead of __byte_perm intrinsic since we don't want the documented (& 0x7) on the index. NVCC
    // might be able to optimize it out since the index is a constexpr, but we choose to be safe about it here.
    uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
    static_assert(RegArray::kElements <= 8, "Too many inputs for I2 -> FP16 vector converter");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      asm volatile(
          "{ prmt.b32 %0, %1, %2, %3; }\n"
          : "=r"(r[ii])
          : "r"(src_reg), "n"(0), "r"(prmt_indices[ii / 2]));

      asm volatile(
           "{ prmt.b32 %0, %1, %2, %3; }\n"
           : "=r"(r[ii + 1])
           : "r"(src_reg_shifted), "n"(0), "r"(prmt_indices[ii / 2]));
    }

    // The below XOR does the following:
    // Sets the exponent bits of the FP16 to the correct value for the FP16 magic_num. We will be constructing
    // 1024 + x + 2, 1024 + 4 * (x + 2)
    // We use lop3 so that we can use 1 instruction for AND and XOR.
    // static constexpr uint32_t xor_mask[2] = { 0x64086402, 0x64806420};
    // static constexpr uint32_t and_mask[2] = { 0x000C0003, 0x00C00030};
    static constexpr uint32_t xor_mask = 0x64086402;
    static constexpr uint32_t and_mask = 0x000C0003;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    // For each operand, computes:
    // r[i] = (r[i] & and_mask[i / 2]) ^ xor_mask[i / 2]
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{ lop3.b32 %0, %0, %1, %2, %3; }\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    // {-258, -1026}
    static constexpr uint32_t hfma_bias_rep = 0xDC08E402;
    // {1/4, 1}
    static constexpr uint32_t hfma_scale_rep = 0x34003C00;

    // Scale and subtract the FP16s to get the original int4 number as FP16.
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
      fp16x2_val = __hfma2(fp16x2_val,
                           reinterpret_cast<const half2&>(hfma_scale_rep),
                           reinterpret_cast<const half2&>(hfma_bias_rep));
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_16, source_type_packed_16,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::half_t, N> <= Array<cutlass::uint2b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, cutlass::uint2b_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<cutlass::uint2b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_16 = Array<cutlass::half_t, 16>;
  using result_type_packed_8 = Array<cutlass::half_t, 8>;
  using result_type_packed_4 = Array<cutlass::half_t, 4>;
  using source_type_packed_16 = Array<cutlass::uint2b_t, 16>;
  using source_type_packed_8 = Array<cutlass::uint2b_t, 8>;
  using source_type_packed_4 = Array<cutlass::uint2b_t, 4>;

  using ScalarConverter = NumericConverter<cutlass::half_t, cutlass::uint2b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_16 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_16>::value &&
                   platform::is_same<PackedResultType, result_type_packed_16>::value),
                  "Invalid PackedSrcType/PackedResultType must be 4, 8 or 16 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 4;

    // Below constructs the following temporary:
    // f1f0   = {0x00, u3u2u1u0,     0x00, u3u2u1u0}
    // f3f2   = {0x00, u5u4u3u2,     0x00, u5u4u3u2}
    // f5f4   = {0x00, u7u6u5u4,     0x00, u7u6u5u4}
    // f7f6   = {0x00, u9u8u7u6,     0x00, u9u8u7u6}
    // f9f8   = {0x00, u11u10u9u8,   0x00, u11u10u9u8}
    // f11f10 = {0x00, u13u12u11u10, 0x00, u13u12u11u10}
    // f13f12 = {0x00, u15u14u13u12, 0x00, u15u14u13u12}
    // f15f14 = {0x00, 0000u15u14,   0x00, 0000u15u14}
    // We use inline asm instead of __byte_perm intrinsic since we don't want the documented (& 0x7) on the index. NVCC
    // might be able to optimize it out since the index is a constexpr, but we choose to be safe about it here.
    uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
    static_assert(RegArray::kElements <= 8, "Too many inputs for I2 -> FP16 vector converter");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      asm volatile(
          "{ prmt.b32 %0, %1, %2, %3; }\n"
          : "=r"(r[ii])
          : "r"(src_reg), "n"(0), "r"(prmt_indices[ii / 2]));

      asm volatile(
           "{ prmt.b32 %0, %1, %2, %3; }\n"
           : "=r"(r[ii + 1])
           : "r"(src_reg_shifted), "n"(0), "r"(prmt_indices[ii / 2]));
    }

    // The below XOR does the following:
    // Sets the exponent bits of the FP16 to the correct value for the FP16 magic_num. We will be constructing
    // 1024 + x, 1024 + 4 * x
    // We use lop3 so that we can use 1 instruction for AND and OR.
    static constexpr uint32_t xor_mask = 0x64006400;
    static constexpr uint32_t and_mask = 0x000C0003;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    // For each operand, computes:
    // r[i] = (r[i] & and_mask[i / 2]) ^ xor_mask[i / 2]
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{ lop3.b32 %0, %0, %1, %2, %3; }\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    // {-256, -1024}
    static constexpr uint32_t hfma_bias_rep = 0xDC00E400;
    // {1/4, 1}
    static constexpr uint32_t hfma_scale_rep = 0x34003C00;

    // Scale and subtract the FP16s to get the original int4 number as FP16.
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
      fp16x2_val = __hfma2(fp16x2_val,
                           reinterpret_cast<const half2&>(hfma_scale_rep),
                           reinterpret_cast<const half2&>(hfma_bias_rep));
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_16, source_type_packed_16,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::half_t, N> <= Array<cutlass::int4b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, cutlass::int4b_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<cutlass::int4b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<cutlass::half_t, 8>;
  using result_type_packed_4 = Array<cutlass::half_t, 4>;
  using result_type_packed_2 = Array<cutlass::half_t, 2>;
  using source_type_packed_8 = Array<cutlass::int4b_t, 8>;
  using source_type_packed_4 = Array<cutlass::int4b_t, 4>;
  using source_type_packed_2 = Array<cutlass::int4b_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::half_t, cutlass::int4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses bit tricks to construct a known FP16 number, then does a
  // subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2, 4 or 8 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);

    // Below constructs the following temporary:
    // fp16s_01 = {0x00, i4_01, 0x00, i4_01}
    // fp16s_23 = {0x00, i4_23, 0x00, i4_23}
    // fp16s_45 = {0x00, i4_45, 0x00, i4_45}
    // fp16s_67 = {0x00, i4_67, 0x00, i4_67}
    // We use inline asm instead of __byte_perm intrinsic since we don't want the documented (& 0x7) on the index. NVCC
    // might be able to optimize it out since the index is a constexpr, but we choose to be safe about it here.
    uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
    static_assert(RegArray::kElements <= 4, "Too many inputs for I4 ->F16 vector converter");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{ prmt.b32 %0, %1, %2, %3; }\n"
          : "=r"(r[ii])
          : "r"(src_reg), "n"(0), "r"(prmt_indices[ii]));
    }

    // The below XOR does the following:
    // 1) Sets the exponent bits of the FP16 to the correct value for the FP16 magic_num. We will be constructing
    //    1024 + x + 8 OR 1024 + 16 * (x + 8), then using hfma to subtract 1032 from that
    // 2) Adds 8 to the int4 value that we will process in the FP16 (for uint4, we can simply avoid this step)
    // The AND does the following:
    // 1) Clear the set bits for the int4 we will ignore.
    // We use lop3 so that we can use 1 instruction for AND and XOR.
    static constexpr uint32_t xor_mask = 0x64806408;
    static constexpr uint32_t and_mask = 0xFFF0FF0F;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    // For each operand, computes:
    // r[i] = (r[i] & and_mask) ^ xor_mask
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    // We will issue 2 hfmas that do the following:
    // For the high FP16:
    //  Divide by 16 {packed as a operand} to get:
    //    64 + (x + 8)
    //    x + 72
    //  Subtract 72 {packed as c operand} to get x
    // For the low FP16:
    //    1024 + (x + 8)
    //    x + 1032
    // So, we subtract 1032 {packed as c operand} to get x

    // {-72, -1032}
    static constexpr uint32_t hfma_bias_rep = 0xD480E408;
    // {1 / 16, 1}
    static constexpr uint32_t hfma_scale_rep = 0x2C003C00;

    const half2& hfma_bias = reinterpret_cast<const half2&>(hfma_bias_rep);
    const half2& hfma_scale = reinterpret_cast<const half2&>(hfma_scale_rep);
    // Scale and subtract the FP16s to get the original int4 number as FP16.
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
      fp16x2_val = __hfma2(hfma_scale, fp16x2_val, hfma_bias);
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;
public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::half_t, N> <= Array<cutlass::uint4b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, cutlass::uint4b_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<cutlass::uint4b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<cutlass::half_t, 8>;
  using result_type_packed_4 = Array<cutlass::half_t, 4>;
  using result_type_packed_2 = Array<cutlass::half_t, 2>;
  using source_type_packed_8 = Array<cutlass::uint4b_t, 8>;
  using source_type_packed_4 = Array<cutlass::uint4b_t, 4>;
  using source_type_packed_2 = Array<cutlass::uint4b_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::half_t, cutlass::uint4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses bit tricks to construct a known FP16 number, then does a
  // subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2, 4 or 8 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    // Below constructs the following temporary:
    // fp16s_01 = {0x00, u4_01, 0x00, u4_01}
    // fp16s_23 = {0x00, u4_23, 0x00, u4_23}
    // fp16s_45 = {0x00, u4_45, 0x00, u4_45}
    // fp16s_67 = {0x00, u4_67, 0x00, u4_67}
    uint32_t prmt_indices[4] = {0x4040, 0x4141, 0x4242, 0x4343};
    static_assert(RegArray::kElements <= 4, "Too many inputs for u4 -> f16 vector converter");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{ prmt.b32 %0, %1, %2, %3; }\n"
          : "=r"(r[ii])
          : "r"(src_reg), "n"(0), "r"(prmt_indices[ii]));
    }

    // The below XOR does the following:
    // Sets the exponent bits of the FP16 to the correct value for the FP16 magic_num. We will be constructing
    // 1024 + x, then using hsub2 to subtract 1024 from that
    static constexpr uint32_t or_mask  = 0x64006400;
    static constexpr uint32_t and_mask = 0x00F0000F;
    static constexpr uint32_t immLut   = (0xf0 & 0xcc) | 0xaa;

    // For each operand, computes:
    // r[i] = (r[i] & and_mask) | or_mask
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(or_mask), "n"(immLut));

      // We will issue 2 hfmas that do the following:
      // For the high FP16:
      //  Divide by 16 {packed as a operand} to get:
      //    64 + x
      //  Subtract 64 {packed as c operand} to get x
      // For the low FP16:
      // we subtract 1024 {packed as c operand} to get x

      static constexpr uint32_t hfma_bias  = 0xD400E400; // {-64, -1024}
      static constexpr uint32_t hfma_scale = 0x2C003C00; // {1 / 16, 1}
      
      {
        __half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
        fp16x2_val = __hfma2(fp16x2_val, reinterpret_cast<const __half2&>(hfma_scale), reinterpret_cast<const __half2&>(hfma_bias));
      }
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::half_t, N> <= Array<int8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, int8_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<int8_t, N>;
  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_4 = Array<cutlass::half_t, 4>;
  using result_type_packed_2 = Array<cutlass::half_t, 2>;
  using source_type_packed_4 = Array<int8_t, 4>;
  using source_type_packed_2 = Array<int8_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::half_t, int8_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses bit tricks to construct a known FP16 number, then does a
  // subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2 or 4 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    #if 0 // Scalar conversion (Please keep this code for reference for vectorized version below)
    auto result = reinterpret_cast<PackedResultType&>(r);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < PackedResultType::kElements; ++i) {
      int16_t tmp = source[i] + 26112 /* 0x6600 */;
      result[i] = reinterpret_cast<cutlass::half_t const &>(tmp) - 1536.0_hf;
    }
    #endif

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t const prmt_indices[2] = {0x9180, 0xB3A2};

    // Pack s8x2 (s8[1], s8[0]) -> s16x2 (sext.s8[1], sext.s8[0])
    // (See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt)
    // The inline ptx below uses `msb=0` and `msb=1` from the above link to sign-extend the sign bit in 0, 1, 2, 3 bytes of s8x4
    // into result_ptr[0] and result_ptr[1]'s 08-15 and 24-31 bits, respectively.
    // Note that `__byte_perm(source_ptr[0], source_ptr[0], 0x9180);` won't achieve the same result and doesn't sign-extend the sign bit.
    // Thus, we use inline ptx `prmt.b32` instruction for the desired sign extend from s8x2 to s16x2.
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile("prmt.b32 %0,%1,%1,%2;\n" : "=r"(r[ii]) : "r"(src_reg), "r"(prmt_indices[ii]));
    }

    // In the absence of add.s16x2 instruction, use bit-wise operation to execute signed addition with magic numbers to achieve
    // the same result as add.s16x2 instruction.
    // (See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3)
    // For a logical operation F(a, b, c) the value of kImmLut can be computed by applying the same operation to
    // three predefined constant values as follows:
    //                                        ta = 0xF0;
    //                                        tb = 0xCC;
    //                                        tc = 0xAA;
    //                                   kImmLut = F(ta, tb, tc);
    // If we want F = ((a & b) ^ c) then set kImmLut = (0xF0 & 0xCC) ^ 0xAA
    static constexpr uint32_t kImmLut = (0xF0 & 0xCC) ^ 0xAA;

    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      // The bit-wise operation executed below is `r[ii] = (r[ii] & 0x03FF03FF) ^ 0x66006600;`
      asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" :
                                "=r"(r[ii]) : "r"(r[ii]), "n"(0x03FF03FF), "n"(0x66006600), "n"(kImmLut));
    }

    static constexpr uint32_t bias_rep = 0x66006600;
    const half2& bias = reinterpret_cast<const half2&>(bias_rep);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
      fp16x2_val = __hsub2(fp16x2_val, bias);
    }
    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);
    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::half_t, N> <= Array<uint8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::half_t, uint8_t, N, Round> {
  using result_type = Array<cutlass::half_t, N>;
  using source_type = Array<uint8_t, N>;
  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_4 = Array<cutlass::half_t, 4>;
  using result_type_packed_2 = Array<cutlass::half_t, 2>;
  using source_type_packed_4 = Array<uint8_t, 4>;
  using source_type_packed_2 = Array<uint8_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::half_t, uint8_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2 or 4 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t const prmt_indices[2] = {0x5150, 0x5352};
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(r[ii]) : "r"(src_reg), "n"(start_byte_for_fp16), "r"(prmt_indices[ii]));
    }

    static constexpr uint32_t bias_rep = 0x64006400;
    const half2& bias = reinterpret_cast<const half2&>(bias_rep);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii]);
      fp16x2_val = __hsub2(fp16x2_val, bias);
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Array<cutlass::bfloat16_t, N> <= Array<cutlass::int2b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, cutlass::int2b_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<cutlass::int2b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_16 = Array<cutlass::bfloat16_t, 16>;
  using result_type_packed_8 = Array<cutlass::bfloat16_t, 8>;
  using result_type_packed_4 = Array<cutlass::bfloat16_t, 4>;
  using source_type_packed_16 = Array<cutlass::int2b_t, 16>;
  using source_type_packed_8 = Array<cutlass::int2b_t, 8>;
  using source_type_packed_4 = Array<cutlass::int2b_t, 4>;

  using ScalarConverter = NumericConverter<cutlass::bfloat16_t, cutlass::int2b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_16 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_16>::value &&
                   platform::is_same<PackedResultType, result_type_packed_16>::value),
                  "Invalid PackedSrcType/PackedResultType must be 4, 8 or 16 to use private convert dispatch.");

    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted_two = src_reg >> 2;
    uint32_t src_reg_shifted_four = src_reg >> 4;
    uint32_t src_reg_shifted_six = src_reg >> 6;

    // Modified prmt indices for signed 2-bit values 
    uint32_t const prmt_indices[4] = {0xF4F0, 0xF5F1, 0xF6F2, 0xF7F3};

    static_assert(RegArray::kElements <= 8, "Too many inputs for I2 -> BF16 vector converter");

    // First pass: extract and sign extend the 2-bit values
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      asm volatile(
          "{ prmt.b32 %0, %1, %2, %3; }\n"
          : "=r"(r[ii])
          : "r"(src_reg), "r"(src_reg_shifted_two), "r"(prmt_indices[ii / 2]));

      asm volatile(
           "{ prmt.b32 %0, %1, %2, %3; }\n"
           : "=r"(r[ii + 1])
           : "r"(src_reg_shifted_four), "r"(src_reg_shifted_six), "r"(prmt_indices[ii / 2]));
    }

    // For signed 2-bit integers:
    // 00 ->  0     (0)
    // 01 ->  1     (1)
    // 10 -> -2     (2 with sign extension)
    // 11 -> -1     (3 with sign extension)
    //static constexpr uint32_t sign_mask = 0x00020002;  // Mask to check sign bit
    static constexpr uint32_t and_mask = 0x00030003;   // Mask for 2 bits

    // Modified for signed range (-2 to 1)
    // We'll construct numbers in the form 128 + (x + 2) and then subtract 130
    // to get back to our original range
    static constexpr uint32_t xor_mask = 0x43024302;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    // Bias represents 130 in bfloat16 format
    // Subtracting 130 brings us back to our signed range (-2 to 1)
    static constexpr uint32_t bias_rep = 0x43024302;  // {130, 130} in bfloat16
    const __nv_bfloat162& bias = reinterpret_cast<const __nv_bfloat162&>(bias_rep);

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
      bf16x2_val = __hsub2(bf16x2_val, bias);
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_16, source_type_packed_16,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::bfloat16_t, N> <= Array<cutlass::uint2b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, cutlass::uint2b_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<cutlass::uint2b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_16 = Array<cutlass::bfloat16_t, 16>;
  using result_type_packed_8 = Array<cutlass::bfloat16_t, 8>;
  using result_type_packed_4 = Array<cutlass::bfloat16_t, 4>;
  using source_type_packed_16 = Array<cutlass::uint2b_t, 16>;
  using source_type_packed_8 = Array<cutlass::uint2b_t, 8>;
  using source_type_packed_4 = Array<cutlass::uint2b_t, 4>;

  using ScalarConverter = NumericConverter<cutlass::bfloat16_t, cutlass::uint2b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_16 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_16>::value &&
                   platform::is_same<PackedResultType, result_type_packed_16>::value),
                  "Invalid PackedSrcType/PackedResultType must be 4, 8 or 16 to use private convert dispatch.");

    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted_two = src_reg >> 2;
    uint32_t src_reg_shifted_four = src_reg >> 4;
    uint32_t src_reg_shifted_six = src_reg >> 6;

    // Modified prmt indices for signed 2-bit values 
    uint32_t const prmt_indices[4] = {0xF4F0, 0xF5F1, 0xF6F2, 0xF7F3};

    static_assert(RegArray::kElements <= 8, "Too many inputs for U2 -> BF16 vector converter");

    // First pass: extract and sign extend the 2-bit values
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      asm volatile(
          "{ prmt.b32 %0, %1, %2, %3; }\n"
          : "=r"(r[ii])
          : "r"(src_reg), "r"(src_reg_shifted_two), "r"(prmt_indices[ii / 2]));

      asm volatile(
           "{ prmt.b32 %0, %1, %2, %3; }\n"
           : "=r"(r[ii + 1])
           : "r"(src_reg_shifted_four), "r"(src_reg_shifted_six), "r"(prmt_indices[ii / 2]));
    }

    static constexpr uint32_t and_mask = 0x00030003;   // Mask for 2 bits
    static constexpr uint32_t xor_mask = 0x43004300;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{ lop3.b32 %0, %0, %1, %2, %3; }"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    static constexpr uint32_t bias_rep = xor_mask;  // {128, 128} in bfloat16
    const __nv_bfloat162& bias = reinterpret_cast<const __nv_bfloat162&>(bias_rep);

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
      bf16x2_val = __hsub2(bf16x2_val, bias);
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_16, source_type_packed_16,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::bfloat16_t, N> <= Array<cutlass::int4b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, cutlass::int4b_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<cutlass::int4b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<cutlass::bfloat16_t, 8>;
  using result_type_packed_4 = Array<cutlass::bfloat16_t, 4>;
  using result_type_packed_2 = Array<cutlass::bfloat16_t, 2>;
  using source_type_packed_8 = Array<cutlass::int4b_t, 8>;
  using source_type_packed_4 = Array<cutlass::int4b_t, 4>;
  using source_type_packed_2 = Array<cutlass::int4b_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::bfloat16_t, cutlass::int4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses bit tricks to construct a known FP16 number, then does a
  // subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2, 4 or 8 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 4;

    // Below constructs the following temporary:
    uint32_t const prmt_indices[4] = {0xF4F0, 0xF5F1, 0xF6F2, 0xF7F3};
    static_assert(RegArray::kElements <= 4, "Too many inputs for BF16 -> I4 vector converter");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{ prmt.b32 %0, %1, %2, %3; }\n"
          : "=r"(r[ii])
          : "r"(src_reg), "r"(src_reg_shifted), "r"(prmt_indices[ii]));
    }

    // The below XOR does the following:
    // 1) Sets the exponent bits of the FP16 to the correct value for the FP16 magic_num. We will be constructing
    //    128 + (x + 8) and subtracting 136 to get x
    static constexpr uint32_t xor_mask = 0x43084308;
    static constexpr uint32_t and_mask = 0x000F000F;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    // For each operand, computes:
    // r[i] = (r[i] & and_mask) ^ xor_mask
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{ lop3.b32 %0, %0, %1, %2, %3; }\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    // We will issue 2 bfmas that do the following:
    // high BF16:
    // hi_bf16 - 136, lo_bf16 - 136

    // This is the BF16 {136, 136} represented as an integer.
    static constexpr uint32_t bias_rep = 0x43084308;
    const __nv_bfloat162& bias = reinterpret_cast<const __nv_bfloat162&>(bias_rep);

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
      bf16x2_val = __hsub2(bf16x2_val, bias);
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Array<cutlass::bfloat16_t, N> <= Array<cutlass::uint4b_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, cutlass::uint4b_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<cutlass::uint4b_t, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_8 = Array<cutlass::bfloat16_t, 8>;
  using result_type_packed_4 = Array<cutlass::bfloat16_t, 4>;
  using result_type_packed_2 = Array<cutlass::bfloat16_t, 2>;
  using source_type_packed_8 = Array<cutlass::uint4b_t, 8>;
  using source_type_packed_4 = Array<cutlass::uint4b_t, 4>;
  using source_type_packed_2 = Array<cutlass::uint4b_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::bfloat16_t, cutlass::uint4b_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint8_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_8 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  // The core converter uses bit tricks to construct a known FP16 number, then does a
  // subtraction in FP16 for the final result.
  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_8>::value &&
                   platform::is_same<PackedResultType, result_type_packed_8>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2, 4 or 8 to use private convert dispatch.");

    // Hold output FP16s in reg. We need 1 reg for every 2 elements
    using RegArray = cutlass::AlignedArray<uint32_t, PackedResultType::kElements / 2, sizeof(PackedResultType)>;
    RegArray r;

    // View the input as reg
    uint32_t src_reg = to_reg(source);
    uint32_t src_reg_shifted = src_reg >> 4;

    // Below constructs the following temporary:
    // fp16s_01 = {0x00,  u4_21, 0x00, u4_10}
    // fp16s_23 = {0x00,  u4_43, 0x00, u4_32}
    // fp16s_45 = {0x00,  u4_65, 0x00, u4_54}
    // fp16s_67 = {0x000, u4_7,  0x00, u4_76}
    static constexpr uint32_t prmt_indices[4] = {0xF4F0, 0xF5F1, 0xF6F2, 0xF7F3};
    static_assert(RegArray::kElements <= 4, "Too many inputs for BF16 -> I4 vector converter");
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  prmt.b32 %0, %1, %2, %3;\n"
          "}\n"
          : "=r"(r[ii])
          : "r"(src_reg), "r"(src_reg_shifted), "r"(prmt_indices[ii]));
    }

    static constexpr uint32_t xor_mask = 0x43004300;
    static constexpr uint32_t and_mask = 0x000F000F;
    static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;

    // For each operand, computes:
    // r[i] = (r[i] & and_mask) ^ xor_mask
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(and_mask), "n"(xor_mask), "n"(immLut));
    }

    // We will issue 2 bfmas that do the following:
    // high BF16:
    // hi_bf16 - 128, lo_bf16 - 128

    // This is the BF16 {128, 128} represented as an integer.
    static constexpr uint32_t bias = xor_mask;

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ++ii) {
      __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
      bf16x2_val = __hsub2(bf16x2_val, reinterpret_cast<const __nv_bfloat162&>(bias));
    }

    return reinterpret_cast<PackedResultType&>(r);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_8, source_type_packed_8,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::bfloat16_t, N> <= Array<int8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, int8_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<int8_t, N>;
  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_4 = Array<cutlass::bfloat16_t, 4>;
  using result_type_packed_2 = Array<cutlass::bfloat16_t, 2>;
  using source_type_packed_4 = Array<int8_t, 4>;
  using source_type_packed_2 = Array<int8_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::bfloat16_t, int8_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2 or 4 to use private convert dispatch.");

    NumericArrayConverter<float, int8_t, PackedResultType::kElements, Round> convert_int8_to_f32;
    Array<float, PackedResultType::kElements> tmp = convert_int8_to_f32(source);
    NumericArrayConverter<cutlass::bfloat16_t, float, PackedResultType::kElements, Round> convert_f32_to_bf16;
    return convert_f32_to_bf16(tmp);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<cutlass::bfloat16_t, N> <= Array<uint8_t, N>
template <FloatRoundStyle Round, int N>
struct NumericArrayConverter<cutlass::bfloat16_t, uint8_t, N, Round> {
  using result_type = Array<cutlass::bfloat16_t, N>;
  using source_type = Array<uint8_t, N>;
  static FloatRoundStyle const round_style = Round;

private:
  using result_type_packed_4 = Array<cutlass::bfloat16_t, 4>;
  using result_type_packed_2 = Array<cutlass::bfloat16_t, 2>;
  using source_type_packed_4 = Array<uint8_t, 4>;
  using source_type_packed_2 = Array<uint8_t, 2>;

  using ScalarConverter = NumericConverter<cutlass::bfloat16_t, uint8_t, Round>;

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_2 const& source) {
    return static_cast<uint32_t>(
      reinterpret_cast<const uint16_t&>(source));
  }

  CUTLASS_DEVICE
  static uint32_t to_reg(source_type_packed_4 const& source) {
    return reinterpret_cast<const uint32_t&>(source);
  }

  template <typename PackedResultType, typename PackedSrcType>
  CUTLASS_DEVICE
  static PackedResultType packed_convert(PackedSrcType const &source) {

    static_assert((platform::is_same<PackedSrcType, source_type_packed_2>::value &&
                   platform::is_same<PackedResultType, result_type_packed_2>::value) ||
                  (platform::is_same<PackedSrcType, source_type_packed_4>::value &&
                   platform::is_same<PackedResultType, result_type_packed_4>::value),
                  "Invalid PackedSrcType/PackedResultType must be 2 or 4 to use private convert dispatch.");

    NumericArrayConverter<float, uint8_t, PackedResultType::kElements, Round> convert_uint8_to_f32;
    Array<float, PackedResultType::kElements> tmp = convert_uint8_to_f32(source);
    NumericArrayConverter<cutlass::bfloat16_t, float, PackedResultType::kElements, Round> convert_f32_to_bf16_;
    return convert_f32_to_bf16_(tmp);
  }

  friend class detail::VectorizedConverter;

public:
  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;
    using ConverterType = NumericArrayConverter<typename result_type::Element, typename source_type::Element, N, Round>;
    detail::VectorizedConverter::convert<ConverterType,
                                         result_type_packed_4, source_type_packed_4,
                                         result_type_packed_2, source_type_packed_2>(result, source);

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

/////////////////////////////////////////////////////////////////////////////////////////////////

/// FastNumericArrayConverter only works when the source is within center range.
/// Conversion operator for Array.  See the comments before
/// FastLinearCombinationClamp.
template <typename T, typename S, int N,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          typename Enable = void>
struct FastNumericArrayConverter {
  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &s) {
    NumericArrayConverter<T, S, N, Round> convert_;

    return convert_(s);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/// Partial specialization for Array<float> <= Array<int>
template <int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<float, int, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int tmp = source[i] + 1262485504 /*0x4B400000*/;
      result[i] = reinterpret_cast<float const &>(tmp) - 12582912.0f;
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/// Partial specialization for Array<int8_t, 4> <= Array<float, 4>
template <FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, 4, Round> {
  using result_type = Array<int8_t, 4>;
  using source_type = Array<float, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    Array<int32_t, 4> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      float tmp = source[i] + 12582912.0f;
      result[i] = reinterpret_cast<int32_t const &>(tmp);
    }

    result[0] = __byte_perm(result[0], result[1], 0x40);
    result[2] = __byte_perm(result[2], result[3], 0x40);
    result[0] = __byte_perm(result[0], result[2], 0x5410);

    return reinterpret_cast<result_type const &>(result[0]);
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/// Partial specialization for Array<int8_t> <= Array<float>
template <int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = Array<int8_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    FastNumericArrayConverter<int8_t, float, 4, Round> convert_vector_;

    result_type result;

    Array<int8_t, 4> *result_ptr =
        reinterpret_cast<Array<int8_t, 4> *>(&result);
    Array<float, 4> const *source_ptr =
        reinterpret_cast<Array<float, 4> const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines preferred rounding mode for a pair of types
template <typename T, typename S>
struct PreferredRoundingMode {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_to_nearest;
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
/// Defines preferred rounding mode for a pair of types
template <>
struct PreferredRoundingMode<cutlass::tfloat32_t, float> {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_half_ulp_truncate;
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Packs predicates into an array.
template <int N>
struct PackPredicates {
  using result_type = Array<uint1b_t, N>;

  static_assert(!(N % 4), "Must pack predicates in a count that is a multiple of 4");

  CUTLASS_HOST_DEVICE
  result_type operator()(bool const predicates[]) {

    result_type packed;
    packed.clear();

    int const kWordSize = 8;
    uint8_t *bytes = reinterpret_cast<uint8_t *>(packed.data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      uint8_t mask = static_cast<uint8_t>((predicates[i] ? 1u : 0u) << bit_idx);
      bytes[word_idx] = (bytes[word_idx] | mask);
    }
    return packed;
  }
};

/// Packs predicates into an array
template <int N>
struct UnpackPredicates {
  using result_type = Array<uint1b_t, N>;

  static_assert(!(N % 4), "Must unpack predicates in a count that is a multiple of 4");

  CUTLASS_HOST_DEVICE
  void operator()(bool predicates[], result_type const &packed) {

    int const kWordSize = 8;
    uint8_t const *bytes = reinterpret_cast<uint8_t const *>(packed.data());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      predicates[i] = bool((bytes[word_idx] >> bit_idx) & 0x1);
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
