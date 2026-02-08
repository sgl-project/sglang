/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Defines classes for FP4/FP6 datatypes
*/
#pragma once

#include "cutlass/arch/config.h"
#include "cutlass/float8.h"

// FP4 types are available starting CUDA 12+
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUDA_FP4_ENABLED 1
#endif

#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED))
#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1
#endif

#if (defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) || defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM103F_ENABLED) || defined(CUTLASS_ARCH_MMA_SM110F_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121F_ENABLED))
#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1
#endif

#include "cutlass/cutlass.h"
#include "cutlass/exmy_base.h"

#include "cute/util/type_traits.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

// FP4 and FP6 types
struct float_e2m1_t;
struct float_e3m2_t;
// E2M1:
//   2 Exponent bits with 1 Mantissa bit
//   Range: +-[0,0.5,1,1.5,2,3,4,5,6]
//   has_Inf: false
//   has_NaN: false
//   has_denorm: true
//   Exponent bias (exp_bias): 1

struct float_e2m1_t : public float_exmy_base<cutlass::detail::FpEncoding::E2M1, float_e2m1_t> {
  
  using Base = float_exmy_base<cutlass::detail::FpEncoding::E2M1, float_e2m1_t>;

  float_e2m1_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e2m1_t(double x) : Base(float(x)) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m1_t(float x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m1_t(int x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  float_e2m1_t(Base x) : Base(x) {
  }
};

namespace detail {

// This new type is used to select correct MMA type and TMA type.
struct float_e2m1_unpacksmem_t : public float_exmy_base<cutlass::detail::FpEncoding::E2M1, float_e2m1_t> {

  using Base = float_exmy_base<cutlass::detail::FpEncoding::E2M1, float_e2m1_t>;

  float_e2m1_unpacksmem_t() = default;

  CUTLASS_HOST_DEVICE
  float_e2m1_unpacksmem_t(float_e2m1_unpacksmem_t const& x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m1_unpacksmem_t(double x) : Base(float(x)) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m1_unpacksmem_t(float x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m1_unpacksmem_t(int x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  float_e2m1_unpacksmem_t(Base x) : Base(x) {
  }
};

} // namespace detail

/// Defines the size of an element in bits - specialized for float_e2m1_t
template <>
struct sizeof_bits<float_e2m1_t> {
  static constexpr int value = 4;
};

template <>
struct sizeof_bits<detail::float_e2m1_unpacksmem_t> {
  static constexpr int value = 4;
};

CUTLASS_HOST_DEVICE
float_e2m1_t abs(float_e2m1_t const& val) {
  using BaseType = typename float_e2m1_t::Base;
  return float_e2m1_t(abs(BaseType{val.raw()}));
}


// E2M3:
//   2 Exponent bits with 3 Mantissa bit
//   Range: [-7.5,+7.5]
//   has_Inf: false
//   has_NaN: false
//   has_denorm: true
//   Exponent bias (exp_bias): 1

struct float_e2m3_t : public float_exmy_base<cutlass::detail::FpEncoding::E2M3, float_e2m3_t> {

  using Base = float_exmy_base<cutlass::detail::FpEncoding::E2M3, float_e2m3_t>;

  float_e2m3_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_t(double x) : Base(float(x)) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_t(float x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_t(int x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  float_e2m3_t(Base x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_t(float_e3m2_t x);
};

namespace detail {

struct float_e2m3_unpack8bits_t: public float_exmy_base<cutlass::detail::FpEncoding::E2M3, float_e2m3_unpack8bits_t> {
  // Used in register.
  using Base = float_exmy_base<cutlass::detail::FpEncoding::E2M3, float_e2m3_unpack8bits_t>;

  float_e2m3_unpack8bits_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_unpack8bits_t(double x) : Base(float(x)) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_unpack8bits_t(float x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_unpack8bits_t(int x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  float_e2m3_unpack8bits_t(Base x) : Base(x) {
  }
};

// This new type is used to select correct MMA type and TMA type.
struct float_e2m3_unpacksmem_t : public float_exmy_base<cutlass::detail::FpEncoding::E2M3, float_e2m3_t> {

  using Base = float_exmy_base<cutlass::detail::FpEncoding::E2M3, float_e2m3_t>;

  float_e2m3_unpacksmem_t() = default;

  CUTLASS_HOST_DEVICE
  float_e2m3_unpacksmem_t(float_e2m3_unpacksmem_t const& x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_unpacksmem_t(double x) : Base(float(x)) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_unpacksmem_t(float x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e2m3_unpacksmem_t(int x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  float_e2m3_unpacksmem_t(Base x) : Base(x) {
  }
};

} // namespace detail

/// Defines the size of an element in bits - specialized for float_e2m3_t
template <>
struct sizeof_bits<float_e2m3_t> {
  static constexpr int value = 6;
};

/// Defines the size of an element in bits - specialized for float_e2m3_unpacksmem_t
template <>
struct sizeof_bits<detail::float_e2m3_unpacksmem_t> {
  static constexpr int value = 6;
};

CUTLASS_HOST_DEVICE
float_e2m3_t abs(float_e2m3_t const& val) {
  using BaseType = typename float_e2m3_t::Base;
  return float_e2m3_t(abs(BaseType{val.raw()}));
}

// E3M2:
//   3 Exponent bits, 2 Mantissa bits
//   Range: [-28:+28]
//   has_inf: false
//   has_NaN: false
//   has_denorm: true
//   Exponent bias (exp_bias): 3

struct float_e3m2_t : public float_exmy_base<cutlass::detail::FpEncoding::E3M2, float_e3m2_t> {

  using Base = float_exmy_base<cutlass::detail::FpEncoding::E3M2, float_e3m2_t>;

  float_e3m2_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_t(double x) : Base(float(x)) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_t(float x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_t(int x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  float_e3m2_t(Base x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_t(float_e2m3_t x);
};

namespace detail {

struct float_e3m2_unpack8bits_t : public float_exmy_base<cutlass::detail::FpEncoding::E3M2, float_e3m2_unpack8bits_t> {

  using Base = float_exmy_base<cutlass::detail::FpEncoding::E3M2, float_e3m2_unpack8bits_t>;

  float_e3m2_unpack8bits_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_unpack8bits_t(double x) : Base(float(x)) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_unpack8bits_t(float x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_unpack8bits_t(int x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  float_e3m2_unpack8bits_t(Base x) : Base(x) {
  }
};

// This new type is used to select correct MMA type and TMA type.
struct float_e3m2_unpacksmem_t : public float_exmy_base<cutlass::detail::FpEncoding::E3M2, float_e3m2_t> {

  using Base = float_exmy_base<cutlass::detail::FpEncoding::E3M2, float_e3m2_t>;

  float_e3m2_unpacksmem_t() = default;

  CUTLASS_HOST_DEVICE
  float_e3m2_unpacksmem_t(float_e3m2_unpacksmem_t const& x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_unpacksmem_t(double x) : Base(float(x)) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_unpacksmem_t(float x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  explicit float_e3m2_unpacksmem_t(int x) : Base(x) {
  }

  CUTLASS_HOST_DEVICE
  float_e3m2_unpacksmem_t(Base x) : Base(x) {
  }
};

} // namespace detail

/// Defines the size of an element in bits - specialized for float_e3m2_t
template <>
struct sizeof_bits<float_e3m2_t> {
  static constexpr int value = 6;
};

/// Defines the size of an element in bits - specialized for float_e3m2_unpacksmem_t
template <>
struct sizeof_bits<detail::float_e3m2_unpacksmem_t> {
  static constexpr int value = 6;
};

CUTLASS_HOST_DEVICE
float_e3m2_t abs(float_e3m2_t const& val) {
  using BaseType = typename float_e3m2_t::Base;
  return float_e3m2_t(abs(BaseType{val.raw()}));
}

/// Defines the size of an element in bits - specialized for float_e3m2_unpack8bits_t
template <>
struct sizeof_bits<detail::float_e3m2_unpack8bits_t> {
  static constexpr int value = 8;
};

/// Defines the size of an element in bits - specialized for float_e2m3_unpack8bits_t
template <>
struct sizeof_bits<detail::float_e2m3_unpack8bits_t> {
  static constexpr int value = 8;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Get the register type used in kernel
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
struct get_unpacked_element_type;

template <>
struct get_unpacked_element_type<float_e2m3_t> {
  using type = detail::float_e2m3_unpack8bits_t;
};

template <>
struct get_unpacked_element_type<float_e3m2_t> {
  using type = detail::float_e3m2_unpack8bits_t;
};
} // namespace detail
// ///////////////////////////////////////////////////////////////////////////////////////////////////
// //
// // float_e2m3_t <=> float_e3m2_t conversions
// //
// ///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
float_e2m3_t::float_e2m3_t(float_e3m2_t x)
{
  storage = convert_from_float(float(x)).storage;
}

CUTLASS_HOST_DEVICE
float_e3m2_t::float_e3m2_t(float_e2m3_t x)
{
  storage = convert_from_float(float(x)).storage;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
///
/// Umbrella floating-point 6-bit data type : type_erased_dynamic_float6_t
/// This umbrella datatype can be enabled when a user provides a specific
/// datatype in runtime argument list.
/// 
/// Currently supported runtime datatypes compatible with type_erased_dynamic_float6_t:
///   MXF8F6F4Format::E2M3
///   MXF8F6F4Format::E3M2
///
///////////////////////////////////////////////////////////////

union type_erased_dynamic_float6_t {
  cutlass::float_e2m3_t e2m3;
  cutlass::float_e3m2_t e3m2;

  CUTLASS_HOST_DEVICE
  explicit operator cutlass::float_e2m3_t() const { 
    return e2m3;
  }

  CUTLASS_HOST_DEVICE
  explicit operator cutlass::float_e3m2_t() const { 
    return e3m2;
  }
};

template <>
struct sizeof_bits<type_erased_dynamic_float6_t> {
  static constexpr int value = 6;
};

///////////////////////////////////////////////////////////////
///
/// Umbrella floating-point 4-bit data type : type_erased_dynamic_float4_t
/// This umbrella datatype can be enabled when a user provides a specific
/// datatype in runtime argument list.
/// 
/// Currently supported runtime datatypes compatible with type_erased_dynamic_float4_t:
///   MXF8F6F4Format::E2M1
///
///////////////////////////////////////////////////////////////

union type_erased_dynamic_float4_t {
  cutlass::float_e2m1_t e2m1;
  CUTLASS_HOST_DEVICE
  explicit operator cutlass::float_e2m1_t() const { 
    return e2m1;
  }
};

template <>
struct sizeof_bits<type_erased_dynamic_float4_t> {
  static constexpr int value = 4;
};


///////////////////////////////////////////////////////////////
/// MX/NV types for float6 and float4
/// Intended to be used in builders
///////////////////////////////////////////////////////////////

template <class F6Type>
struct mx_float6_t
{
  static_assert(cute::is_same_v<F6Type,cutlass::float_e2m3_t>
                || cute::is_same_v<F6Type,cutlass::float_e3m2_t>
                || cute::is_same_v<F6Type,type_erased_dynamic_float6_t>
                , "Only float_e2m3_t, float_e3m2_t can have scale factors for MXFP6");
  using ScaleFactorType = cutlass::float_ue8m0_t;
  using DataType = F6Type;
};

using type_erased_dynamic_mx_float6_t = mx_float6_t<type_erased_dynamic_float6_t>;

template <class F4Type>
struct mx_float4_t
{
  static_assert(cute::is_same_v<F4Type,cutlass::float_e2m1_t>
                || cute::is_same_v<F4Type,type_erased_dynamic_float4_t>
                , "Only float_e2m1_t type_erased_dynamic_float4_t can have scale factors for MXFP4");
  using ScaleFactorType = cutlass::float_ue8m0_t;
  using DataType = F4Type;
};

using type_erased_dynamic_mx_float4_t = mx_float4_t<type_erased_dynamic_float4_t>;

template <class F4Type>
struct nv_float4_t
{
  static_assert(cute::is_same_v<F4Type,cutlass::float_e2m1_t>
                || cute::is_same_v<F4Type,type_erased_dynamic_float4_t>
                , "Only float_e2m1_t type_erased_dynamic_float4_t can have scale factors for NVFP4");
  using ScaleFactorType = cutlass::float_ue4m3_t;
  using DataType = F4Type;
};

using type_erased_dynamic_nv_float4_t = nv_float4_t<type_erased_dynamic_float4_t>;


namespace detail {

union type_erased_dynamic_float6_unpacksmem_t {
  cutlass::detail::float_e2m3_unpacksmem_t e2m3_unpacksmem;
  cutlass::detail::float_e3m2_unpacksmem_t e3m2_unpacksmem;

  CUTLASS_HOST_DEVICE
  explicit operator cutlass::detail::float_e2m3_unpacksmem_t() const { 
    return e2m3_unpacksmem;
  }
  
  CUTLASS_HOST_DEVICE
  explicit operator cutlass::detail::float_e3m2_unpacksmem_t() const { 
    return e3m2_unpacksmem;
  }
};

union type_erased_dynamic_float4_unpacksmem_t {
  cutlass::detail::float_e2m1_unpacksmem_t e2m1_unpacksmem;

  CUTLASS_HOST_DEVICE
  explicit operator cutlass::detail::float_e2m1_unpacksmem_t() const { 
    return e2m1_unpacksmem;
  }
};

};

template <>
struct sizeof_bits<detail::type_erased_dynamic_float6_unpacksmem_t> {
  static constexpr int value = 6;
};


template <>
struct sizeof_bits<detail::type_erased_dynamic_float4_unpacksmem_t> {
  static constexpr int value = 4;
};

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#if !defined(__CUDACC_RTC__)
namespace std {
/// Numeric limits common to all float4 types
template <typename T>
struct float_subbyte_base_numeric_limits
{
private:
  using type = T;

public:
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_quiet_NaN = false;
  static bool const has_signaling_NaN = false;
  static bool const has_denorm_loss = true;
  static cutlass::platform::float_denorm_style const has_denorm = cutlass::platform::denorm_present;
  static cutlass::platform::float_round_style const round_style = cutlass::platform::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = type::Base::BitRepresentation::NUM_MANTISSA_BITS;
  static bool const has_infinity = false;

  /// Least positive value
  static type min() { return type::bitcast(0x01); }

  /// Maximum finite value
  static type max() { return type::bitcast(type::Base::BitRepresentation::MAX_VALUE); }

  /// Returns maximum rounding error
  static type round_error() { return type(0.5f); }

  /// Returns positive infinity value
  static type infinity() { return type::bitcast(type::Base::BitRepresentation::INF_MASK); }

  /// Returns quiet NaN value
  static type quiet_NaN() { return type::bitcast(type::Base::BitRepresentation::INF_MASK); }

  /// Returns signaling NaN value
  static type signaling_NaN() { return type::bitcast(type::Base::BitRepresentation::INF_MASK); }

  /// Returns smallest positive subnormal value
  static type denorm_min() { return type::bitcast(0x01); }
};
/// Numeric limits for float_e2m1_t
template <>
struct numeric_limits<cutlass::float_e2m1_t> : public float_subbyte_base_numeric_limits<cutlass::float_e2m1_t>
{
  /// Minimum finite value
  static cutlass::float_e2m1_t lowest() { return cutlass::float_e2m1_t::bitcast(0xf); }

  /// Returns machine epsilon, that is, the difference between 1.0 and the next value representable by the floating-point
  static cutlass::float_e2m1_t epsilon() { return cutlass::float_e2m1_t::bitcast(0x1); }
};

/// Numeric limits for float_e2m3_t
template <>
struct numeric_limits<cutlass::float_e2m3_t> : public float_subbyte_base_numeric_limits<cutlass::float_e2m3_t>
{
  /// Minimum finite value
  static cutlass::float_e2m3_t lowest() { return cutlass::float_e2m3_t::bitcast(0x2f); }

  /// Returns machine epsilon, that is, the difference between 1.0 and the next value representable by the floating-point
  static cutlass::float_e2m3_t epsilon() { return cutlass::float_e2m3_t::bitcast(0x1); }   
};

/// Numeric limits for float_e3m2_t

template <>
struct numeric_limits<cutlass::float_e3m2_t> : public float_subbyte_base_numeric_limits<cutlass::float_e3m2_t>
{
  /// Minimum finite value
  static cutlass::float_e3m2_t lowest() { return cutlass::float_e3m2_t::bitcast(0x2f); }

  /// Returns machine epsilon, that is, the difference between 1.0 and the next value representable by the floating-point
  static cutlass::float_e3m2_t epsilon() { return cutlass::float_e3m2_t::bitcast(0x4); }
};
} // namespace std
#endif

namespace cutlass {
namespace platform {

/// Numeric limits common to all float4 types
template <typename T>
struct float_subbyte_base_numeric_limits
{
private:
  using type = T;

public:
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_quiet_NaN = false;
  static bool const has_signaling_NaN = false;
  static bool const has_denorm_loss = true;
  static cutlass::platform::float_denorm_style const has_denorm = cutlass::platform::denorm_present;
  static cutlass::platform::float_round_style const round_style = cutlass::platform::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = type::Base::BitRepresentation::NUM_MANTISSA_BITS;
  static bool const has_infinity = false;

  /// Least positive value
  static type min() { return type::bitcast(0x01); }

  /// Maximum finite value
  CUTLASS_HOST_DEVICE static type max() { return type::bitcast(type::Base::BitRepresentation::MAX_VALUE); }

  /// Returns maximum rounding error
  static type round_error() { return type(0.5f); }

  /// Returns positive infinity value
  static type infinity() { return type::bitcast(type::Base::BitRepresentation::INF_MASK); }

  /// Returns quiet NaN value
  static type quiet_NaN() { return type::bitcast(type::Base::BitRepresentation::INF_MASK); }

  /// Returns signaling NaN value
  static type signaling_NaN() { return type::bitcast(type::Base::BitRepresentation::INF_MASK); }

  /// Returns smallest positive subnormal value
  static type denorm_min() { return type::bitcast(0x01); }
};

/// Forward Declaration
template <class T>
struct numeric_limits;
/// Numeric limits for float_e2m1_t
template <>
struct numeric_limits<cutlass::float_e2m1_t> : public float_subbyte_base_numeric_limits<cutlass::float_e2m1_t>
{
  /// Minimum finite value
  static cutlass::float_e2m1_t lowest() { return cutlass::float_e2m1_t::bitcast(0xf); }

  /// Returns machine epsilon, that is, the difference between 1.0 and the next value representable by the floating-point
  static cutlass::float_e2m1_t epsilon() { return cutlass::float_e2m1_t::bitcast(0x1); }
};

/// Numeric limits for float_e2m3_t
template <>
struct numeric_limits<cutlass::float_e2m3_t> : public float_subbyte_base_numeric_limits<cutlass::float_e2m3_t>
{
  /// Minimum finite value
  static cutlass::float_e2m3_t lowest() { return cutlass::float_e2m3_t::bitcast(0x2f); }

  /// Returns machine epsilon, that is, the difference between 1.0 and the next value representable by the floating-point
  static cutlass::float_e2m3_t epsilon() { return cutlass::float_e2m3_t::bitcast(0x1); }   
};

/// Numeric limits for float_e3m2_t

template <>
struct numeric_limits<cutlass::float_e3m2_t> : public float_subbyte_base_numeric_limits<cutlass::float_e3m2_t>
{
  /// Minimum finite value
  static cutlass::float_e3m2_t lowest() { return cutlass::float_e3m2_t::bitcast(0x2f); }

  /// Returns machine epsilon, that is, the difference between 1.0 and the next value representable by the floating-point
  static cutlass::float_e3m2_t epsilon() { return cutlass::float_e3m2_t::bitcast(0x4); }
};

/// Numeric limits for float_e2m3_unpack8bits_t
template <>
struct numeric_limits<cutlass::detail::float_e2m3_unpack8bits_t> : public float_subbyte_base_numeric_limits<cutlass::detail::float_e2m3_unpack8bits_t>
{
  /// Minimum finite value
  static cutlass::detail::float_e2m3_unpack8bits_t lowest() { return cutlass::detail::float_e2m3_unpack8bits_t::bitcast(0x2f); }

  /// Returns machine epsilon, that is, the difference between 1.0 and the next value representable by the floating-point
  static cutlass::detail::float_e2m3_unpack8bits_t epsilon() { return cutlass::detail::float_e2m3_unpack8bits_t::bitcast(0x1); }   
};

/// Numeric limits for float_e3m2_unpack8bits_t

template <>
struct numeric_limits<cutlass::detail::float_e3m2_unpack8bits_t> : public float_subbyte_base_numeric_limits<cutlass::detail::float_e3m2_unpack8bits_t>
{
  /// Minimum finite value
  static cutlass::detail::float_e3m2_unpack8bits_t lowest() { return cutlass::detail::float_e3m2_unpack8bits_t::bitcast(0x2f); }

  /// Returns machine epsilon, that is, the difference between 1.0 and the next value representable by the floating-point
  static cutlass::detail::float_e3m2_unpack8bits_t epsilon() { return cutlass::detail::float_e3m2_unpack8bits_t::bitcast(0x4); }
};
} // namespace platform

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//
CUTLASS_HOST_DEVICE
cutlass::float_e2m1_t operator"" _fe2m1(long double x)
{
  return cutlass::float_e2m1_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::float_e2m1_t operator"" _fe2m1(unsigned long long int x)
{
  return cutlass::float_e2m1_t(int(x));
}
CUTLASS_HOST_DEVICE
cutlass::float_e2m3_t operator"" _fe2m3(long double x)
{
  return cutlass::float_e2m3_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::float_e2m3_t operator"" _fe2m3(unsigned long long int x)
{
  return cutlass::float_e2m3_t(int(x));
}

CUTLASS_HOST_DEVICE
cutlass::float_e3m2_t operator"" _fe3m2(long double x)
{
  return cutlass::float_e3m2_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::float_e3m2_t operator"" _fe3m2(unsigned long long int x)
{
  return cutlass::float_e3m2_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
