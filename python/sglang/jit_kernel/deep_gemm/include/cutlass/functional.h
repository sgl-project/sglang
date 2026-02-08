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
/*! \file
    \brief Define basic numeric operators

    This is inspired by the Standard Library's <functional> header.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#if defined(__CUDACC_RTC__)
#include "cutlass/floating_point_nvrtc.h"
#endif

#include <cuda_runtime.h>

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include <mma.h>
#endif // defined(CUTLASS_ARCH_WMMA_ENABLED)

#ifdef _MSC_VER
// Provides support for alternate operators such as 'and', 'or', ...
#include <ciso646>
#include <intrin.h>
#endif // _MSC_VER

#if defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) ||\
    defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM103F_ENABLED)
#  define CUTLASS_ARCH_CREDUX_ENABLED
#endif

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

  CUTLASS_HOST_DEVICE int32_t popcount(int32_t x) {
    #if defined(__CUDA_ARCH__)
    return __popc(x);
    #elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
    #elif (defined(_MSC_VER) && !defined(_M_ARM64))
    return __popcnt(x);
    #else
    int32_t count = 0;
    while (x) {
      count += x & 1;
      x >>= 1;
    }
    return count;
    #endif
  }

  CUTLASS_HOST_DEVICE int64_t popcount(int64_t x) {
    #if defined(__CUDA_ARCH__)
    return __popcll(x);
    #elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
    #elif (defined(_MSC_VER) && !defined(_M_ARM64))
    return __popcnt64(x);
    #else
    int64_t count = 0;
    while (x) {
      count += x & 1;
      x >>= 1;
    }
    return count;
    #endif
  }

} // namespace detail
  
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct absolute_value_op {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return abs(lhs);
  }
};

template <>
struct absolute_value_op<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float lhs) const { return fabs(lhs); }
};

template <typename T>
struct plus {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs += rhs;
    return lhs;
  }
};

template <typename T>
struct minus {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs -= rhs;
    return lhs;
  }
};

template <typename T>
struct multiplies {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs *= rhs;
    return lhs;
  }
};

template <typename T>
struct scale {
  T const scaling_factor_;

  CUTLASS_HOST_DEVICE
  scale(float scaling_factor) : scaling_factor_(scaling_factor) {
  }

  T operator()(T const &rhs) const {
    T result = rhs * scaling_factor_;
    return result;
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
/// Partial specializations needed when __CUDA_NO_HALF2_OPERATORS__ is set
template<>
struct plus<__half2> {
  CUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hadd2(lhs, rhs);
  }
};

template<>
struct minus<__half2> {
  CUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hsub2(lhs, rhs);
  }
};

template<>
struct multiplies<__half2> {
  CUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hmul2(lhs, rhs);
  }
};

/// Partial specializations needed when __CUDA_NO_HALF_OPERATORS__ is set
template<>
struct plus<__half> {
  CUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hadd(lhs, rhs);
  }
};

template<>
struct minus<__half> {
  CUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hsub(lhs, rhs);
  }
};

template<>
struct multiplies<__half> {
  CUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hmul(lhs, rhs);
  }
};
#endif // defined(__CUDA_ARCH__)


/// Squares with optional conversion
template <typename T, typename Output = T>
struct square {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Returns the magnitude squared of an element.
template <typename T, typename Output = T>
struct magnitude_squared {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct square_difference {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct magnitude_squared_difference {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

// Computes the reciprocal square root
template <typename T>
struct inverse_square_root;

template <>
struct inverse_square_root<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs) const {
#if defined(__CUDA_ARCH__)
    return rsqrtf(lhs);
#else
    return 1.f / std::sqrt(lhs);
#endif
  }
};

template <>
struct inverse_square_root<half_t> {
  CUTLASS_HOST_DEVICE
  half_t operator()(half_t const &lhs) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 520)
    auto result = hrsqrt(reinterpret_cast<__half const &>(lhs));
    return reinterpret_cast<half_t const &>(result);
#else
    return half_t(1.f / std::sqrt(half_t::convert(lhs)));
#endif
  }
};

/// Divides
template <typename T>
struct divides {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs /= rhs;
    return lhs;
  }
};

/// reciprocal_approximate
template <typename T>
struct reciprocal_approximate {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return divides<T>{}(T(1), lhs);
  }
};

template <>
struct reciprocal_approximate <float> {
  CUTLASS_HOST_DEVICE
  float operator()(float lhs) const {
    float ret;
    #if defined(__CUDA_ARCH__)
      asm volatile ("rcp.approx.f32 %0, %1;\n" : "=f"(ret) : "f"(lhs));
    #else
      ret = 1.0f / lhs;
    #endif
    return ret;
  }
};


template <>
struct reciprocal_approximate<cutlass::float_ue8m0_t> {
  CUTLASS_HOST_DEVICE
  cutlass::float_ue8m0_t operator()(cutlass::float_ue8m0_t lhs) const {
    return cutlass::float_ue8m0_t::bitcast(static_cast<uint8_t>(static_cast<uint8_t>(254u) - lhs.storage));
  }
};


/// reciprocal_approximate with ftz
template<typename T>
struct reciprocal_approximate_ftz :  reciprocal_approximate<T>
{};

template <>
struct reciprocal_approximate_ftz <float> {
  CUTLASS_HOST_DEVICE
  float operator()(float lhs) const {
    float ret;
    #if defined(__CUDA_ARCH__)
      asm volatile ("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(ret) : "f"(lhs));
    #else
      if (std::fpclassify(lhs) == FP_SUBNORMAL) {
        lhs = 0.0f;
      }
      ret = 1.0f / lhs;
      if (std::fpclassify(ret) == FP_SUBNORMAL) {
        ret = 0.0f;
      }
    #endif
    return ret;
  }
};

/// Negate
template <typename T>
struct negate {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return -lhs;
  }
};

/// Greater equal
template <typename T>
struct greater_equal {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs >= rhs);
  }
};

/// Greater
template <typename T>
struct greater {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs > rhs);
  }
};

/// Less equal
template <typename T>
struct less_equal {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs <= rhs);
  }
};

/// Less
template <typename T>
struct less {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs);
  }
};

template <typename T, bool PropagateNaN = false>
struct maximum {
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    if constexpr (PropagateNaN && cutlass::platform::is_floating_point<T>::value) {
      using CUTLASS_CMATH_NAMESPACE :: isnan;

      // Call isnan unqualified, so argument-dependent lookup (ADL)
      // will find overloads such as cutlass::isnan(half_t).
      // Calling ::isnan or std::isnan directly would force
      // implicit conversions to float of custom number types
      // in the cutlass namespace (e.g., cutlass::half_t).
      return lhs > rhs || isnan(lhs) ? lhs : rhs;
    }
    else {
      return (lhs < rhs ? rhs : lhs);
    }

    CUTE_GCC_UNREACHABLE;
  }
};

// This is a subclass and not an alias
// in order to work around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template<typename T>
struct maximum_with_default_nan_propagation : public maximum<T>
{};

template <>
struct maximum<float, false> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fmaxf(lhs, rhs);
  }
};

template <>
struct maximum<float, true> {
  CUTLASS_HOST_DEVICE
  float operator()(float lhs, float rhs) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    float res;
    asm volatile("max.NaN.f32 %0, %1, %2;\n" : "=f"(res) : "f"(lhs), "f"(rhs));
    return res;
#else
    using CUTLASS_CMATH_NAMESPACE :: isnan;

    return lhs > rhs || isnan(lhs) ? lhs : rhs;
#endif
  }
};

// This is a subclass and not an alias
// in order to work around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template <typename T>
struct maximum_with_nan_propagation : maximum<T, true>
{};

// This alias exists for backwards compatibility only.
// Please use the correctly spelled class template above.
template <typename T>
using maximum_with_nan_propogation = maximum_with_nan_propagation<T>;

template <typename T, bool PropagateNaN = false>
struct minimum {
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    if constexpr (PropagateNaN && cutlass::platform::is_floating_point<T>::value) {
      using CUTLASS_CMATH_NAMESPACE :: isnan;

      return lhs < rhs || isnan(lhs) ? lhs : rhs;
    }
    else {
      return (rhs < lhs ? rhs : lhs);
    }
  }
};

template <>
struct minimum<float, false> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fminf(lhs, rhs);
  }
};

template <>
struct minimum<float, true> {
  CUTLASS_HOST_DEVICE
  float operator()(float lhs, float rhs) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    float res;
    asm volatile("min.NaN.f32 %0, %1, %2;\n" : "=f"(res) : "f"(lhs), "f"(rhs));
    return res;
#else
    // No need for ADL; call std::isnan(float) on host and ::isnan(float) on device.
    return lhs < rhs || (CUTLASS_CMATH_NAMESPACE :: isnan(lhs)) ? lhs : rhs;
#endif
  }
};

template <typename T>
struct minimum_with_nan_propagation : minimum<T, true> 
{};

template <typename T, bool PropagateNaN = false>
struct maximum_absolute_value {
  CUTLASS_HOST_DEVICE
  float operator()(T const &lhs, T const &rhs) const {
    absolute_value_op<T> abs_op;
    maximum<T, PropagateNaN> max_op;

    return max_op(abs_op(lhs), abs_op(rhs));
  }
};

// assumes the left operand is already an absolute value
template <typename T, bool PropagateNaN = false>
struct maximum_absolute_value_reduction {
  CUTLASS_HOST_DEVICE
  float operator()(T const &lhs, T const &rhs) const {
    absolute_value_op<T> abs_op;
    maximum<T, PropagateNaN> max_op;

    return max_op(lhs, abs_op(rhs));
  }
};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    return C(a) * C(b) + c;
  }
};

template <typename T>
struct square_and_plus {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    multiply_add<T> multiply_add_op;
    return multiply_add_op(rhs, rhs, lhs);
  }
};

// Fused multiply-add that takes exactly one template parameter.
// This is useful for working around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template <typename A>
struct homogeneous_multiply_add : public multiply_add<A, A, A>
{};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add_relu0 {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    maximum<C> mx;
    return mx(C(a) * C(b) + c, C(0));
  }
};

/// Guarded-multiply-add
template <typename A, typename B = A, typename C = A>
struct guarded_multiply_add {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    using CUTLASS_CMATH_NAMESPACE :: isnan;

    if (isnan(a) || isnan(b)) {
      return C(0);
    }
    return C(a) * C(b) + c;
  }
};

/// Guarded-multiply-add
template <>
struct guarded_multiply_add<half_t, half_t, half_t> {
  CUTLASS_HOST_DEVICE
  half_t operator()(half_t const &a, half_t const &b, half_t const &c) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    half_t result;
    asm ("fma.rn.oob.f16 %0, %1, %2, %3;\n"
      : "=h"(*reinterpret_cast<uint16_t*>(&result))
      : "h"(*reinterpret_cast<uint16_t const*>(&a)), "h"(*reinterpret_cast<uint16_t const*>(&b)), "h"(*reinterpret_cast<uint16_t const*>(&c)));
    return result;
#else
    // Namespace-qualifying isnan as cutlass::isnan saves the compiler
    // the trouble of argument-dependent lookup.  Calling std::isnan or
    // ::isnan here would result in unwanted implicit conversion to float.
    if (cutlass::isnan(a) || cutlass::isnan(b)) {
      return half_t(0);
    }
    return a * b + c;
#endif
  }
};

/// Guarded-multiply-add-relu0
template <typename A, typename B = A, typename C = A>
struct guarded_multiply_add_relu0 {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    using CUTLASS_CMATH_NAMESPACE :: isnan;

    if (isnan(a) || isnan(b)) {
      return C(0);
    }
    maximum<C> mx;
    return mx(C(a) * C(b) + c, C(0));
  }
};

template <>
struct guarded_multiply_add_relu0<half_t, half_t, half_t> {
  CUTLASS_HOST_DEVICE
  half_t operator()(half_t const &a, half_t const &b, half_t const &c) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    half_t result;
    asm ("fma.rn.oob.relu.f16 %0, %1, %2, %3;\n"
      : "=h"(*reinterpret_cast<uint16_t*>(&result))
      : "h"(*reinterpret_cast<uint16_t const*>(&a)), "h"(*reinterpret_cast<uint16_t const*>(&b)), "h"(*reinterpret_cast<uint16_t const*>(&c)));
    return result;
#else
    if (cutlass::isnan(a) || cutlass::isnan(b)) {
      return half_t(0);
    }
    maximum<half_t> mx;
    return mx(a * b + c, half_t(0));
#endif
  }
};


/// Fused and-popc-add
template <typename A, typename B = A, typename C = A>
struct and_popc_add {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    A and_result = a & b;
    int32_t popc_result = detail::popcount(and_result);
    return C(popc_result) + c;
  }
};

/// Fused and-add
template <typename T>
struct and_add {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a & b) + c);
  }
};



/// Fused xor-popc-add
template <typename A, typename B = A, typename C = A>
struct xor_popc_add {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    A xor_result = a ^ b;
    int32_t popc_result = detail::popcount(xor_result);
    return C(popc_result) + c;
  }
};

/// Fused xor-add
template <typename T>
struct xor_add {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a ^ b) + c);
  }
};


/// Fused or-popc-add
template <typename A, typename B = A, typename C = A>
struct or_popc_add {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    A or_result = a | b;
    int32_t popc_result = detail::popcount(or_result);
    return C(popc_result) + c;
  }
};


/// Fused or-add
template <typename T>
struct or_add {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a | b) + c);
  }
};

namespace detail {

// Whether namespace-unqualified conj(t) for t of type T is
// well-formed.  This says whether the compiler can find
// namespace-unqualified conj(T) via argument-dependent lookup.
// If so, then CUTLASS assumes that conj(t) returns
// the complex conjugate of t.
template <typename T, typename Enable = void>
struct has_unqualified_conj : cutlass::platform::false_type
{};

template<typename T>
struct has_unqualified_conj<
    T,
    decltype(static_cast<void>(conj(cutlass::platform::declval<T>())), void())
  > : cutlass::platform::true_type
{};

template <typename T>
constexpr bool has_unqualified_conj_v = has_unqualified_conj<T>::value;
  
} // namespace detail

// forward declaration (needed for conjugate below)
template<class T>
CUTLASS_HOST_DEVICE T conj(T const& z);

namespace detail {

// Whether cutlass::conj(t) for t of type T is well-formed.
// If so, then CUTLASS assumes that cutlass::conj(t)
// returns the complex conjugate of t.
template <typename T, typename Enable = void>
struct has_cutlass_conj : cutlass::platform::false_type
{};

template<typename T>
struct has_cutlass_conj<
    T,
    decltype(cutlass::conj(cutlass::platform::declval<T>()), void())
  > : cutlass::platform::true_type
{};

template <typename T>
constexpr bool has_cutlass_conj_v = has_cutlass_conj<T>::value;

} // namespace detail
  
// Return the complex conjugate of the input.
//
// If the struct hasn't already been specialized for type T, then
//
// 1. for arithmetic types, return z;
//
// 2. for types where either (namespace-unqualified) conj(z) or
//    cutlass::conj(z) is well formed, declare "using cutlass::conj;"
//    and return conj(z); and
//
// 3. for everything else, return z.
//
// Regarding (1), the C++ Standard Library makes std::conj always
// return std::complex, even for (noncomplex) arithmetic types.
// cutlass::conj(T t) needs to return type T.  This follows the
// convention of linear algebra software like the BLAS, where
// "conjugate transpose" means the same thing as "transpose" for a
// matrix of noncomplex numbers.
//
// Case (2) covers std::complex, cuda::std::complex, and non-Standard
// (including user-defined) complex number types (for which "conj(z)"
// is findable via argument-dependent lookup).  cutlass::conj has a
// totally generic overload, but a more type-specific overload in any
// namespace will take precedence.
//
// Case (3) covers non-Standard non-complex number types.
//
// Users should not generally need to specialize this struct for their
// own custom complex or noncomplex types.  The idiomatic way to
// identify a type T as "complex" is to make namespace-unqualified
// calls to conj(T) findable via argument-dependent lookup.
template <typename T>
struct conjugate {
  CUTLASS_HOST_DEVICE
  T operator()(T const& z) const {
    if constexpr (cutlass::platform::is_arithmetic_v<T>) {
      return z;
    }
    else if constexpr (detail::has_unqualified_conj_v<T> || detail::has_cutlass_conj_v<T>) {
      using cutlass::conj;
      return conj(z);
    }
    else {
      return z;
    }
  }
};

template <typename T>
struct first {
  CUTLASS_HOST_DEVICE
  T operator()(T const & first, T const &...) const {
    return first;
  }
  CUTLASS_HOST_DEVICE
  T operator()(T const & first) const {
    return first;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct logical_and {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((static_cast<bool>(a) && static_cast<bool>(b)) ? T(1) : T());
  }
};

template <typename T>
struct logical_or {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((static_cast<bool>(a) || static_cast<bool>(b)) ? T(1) : T());
  }
};

template <typename T>
struct logical_not {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return T(!(a));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct bit_and {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a & b;
  }
};

template <typename T>
struct bit_or {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a | b;
  }
};

template <typename T>
struct bit_not {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return ~a;
  }
};

template <typename T>
struct bit_xor {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a ^ b;
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Atomic reductions

template <typename T>
struct atomic_add
{
  CUTLASS_DEVICE
  void operator()(T *ptr, const T &data)
  {
#if defined(__CUDA_ARCH__)
    atomicAdd(ptr, data);
#else
    CUTLASS_UNUSED(ptr);
    CUTLASS_UNUSED(data);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

template<>
struct atomic_add<double>
{
  CUTLASS_DEVICE
  void operator()(double *ptr, const double &data)
  {
#if !defined(__CUDA_ARCH__)
    CUTLASS_UNUSED(ptr);
    CUTLASS_UNUSED(data);
    CUTLASS_NOT_IMPLEMENTED();
#elif (__CUDA_ARCH__ >= 600)
    atomicAdd(ptr, data);
#else
    // Use CAS loop
    unsigned long long int* ptr_int = reinterpret_cast<unsigned long long int*>(ptr);
    unsigned long long int old_int = *ptr_int;
    unsigned long long int assumed_int;

    do {
      double update = data + __longlong_as_double(old_int);
      assumed_int = old_int;
      old_int = atomicCAS(ptr_int, assumed_int, __double_as_longlong(update));
    } while (assumed_int != old_int);
#endif // (__CUDA_ARCH__ >= 600)
  }
};

template<>
struct atomic_add<half2>
{
  CUTLASS_DEVICE
  void operator()(half2 *ptr, const half2 &data)
  {
#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__)  && (__CUDA_ARCH__ < 600))
      CUTLASS_UNUSED(ptr);
      CUTLASS_UNUSED(data);
      CUTLASS_NOT_IMPLEMENTED();
#else
    // Vector-2 atomic reduction requires .target sm_60 or higher
    uint32_t word = reinterpret_cast<const uint32_t&>(data);
    asm volatile ("red.gpu.global.add.noftz.f16x2 [%0], %1;\n" : : "l"(ptr), "r"(word));
#endif // (__CUDA_ARCH__ >= 600)
  }
};

template <typename T>
using red [[deprecated("use atomic_add instead")]] = atomic_add<T>;

template <typename T>
struct atomic_maximum {
  CUTLASS_DEVICE
  T operator()(T *ptr, T value) const {
#if defined(__CUDA_ARCH__)
    return atomicMax(ptr, value);
#else
    CUTLASS_UNUSED(ptr);
    CUTLASS_UNUSED(value);
    CUTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
  }
};

template <>
struct atomic_maximum<float> {
  CUTLASS_DEVICE
  float operator()(float *ptr, float value) const {
#if defined(__CUDA_ARCH__)
    // In device code, make sure that we do NOT try to use
    // std::signbit, as that won't work if building with NVRTC.
    // Instead, prefix "::" to call signbit from the global namespace,
    // which CUDA guarantees to work in device code without including
    // any headers.
    //
    return ! ::signbit(value) ?
      __int_as_float(atomicMax((int*)ptr, __float_as_int(value))) :
      __uint_as_float(atomicMin((unsigned int*)ptr, __float_as_uint(value)));
#else
    CUTLASS_UNUSED(ptr);
    CUTLASS_UNUSED(value);
    CUTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
  }
};

// is_atomic
template <class Fn>
struct is_atomic : platform::false_type {};
template <class T>
struct is_atomic<atomic_add<T>> : platform::true_type {};
template <class T>
struct is_atomic<atomic_maximum<T>> : platform::true_type {};


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Parallel Synchronization and Communication Instructions
template <typename T>
struct redux_abs_max_nan_propagation_sync_warp;

template <>
struct redux_abs_max_nan_propagation_sync_warp <float>{
  CUTLASS_DEVICE
  float operator()(float const &lhs) const {
#if defined(CUTLASS_ARCH_CREDUX_ENABLED)
    float result;
    asm volatile("redux.sync.max.abs.NaN.f32 %0, %1, 0xffffffff;\n" : "=f"(result) : "f"(lhs));
    return result;
#elif defined(__CUDA_ARCH__)
    cutlass::maximum<float, /*PropagateNaN*/true> max_op;
    int shuffle_width = 32;
    float abs_max = cutlass::absolute_value_op<float>{}(lhs);
    CUTLASS_PRAGMA_UNROLL
    for(int offset = shuffle_width / 2; offset > 0; offset /= 2) {
      float value = __shfl_down_sync(0xffffffff, abs_max, offset, shuffle_width);
      abs_max = max_op(abs_max,value);
    }
    // Broadcast the maximum to all threads participating in the reduction.
    abs_max = __shfl_sync(0xffffffff, abs_max, 0, shuffle_width);
    return abs_max;
#else
    CUTLASS_UNUSED(lhs);
    CUTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
  }
};

template <typename T>
struct redux_abs_max_nan_propagation_sync_warp_t0t15_t16t31;

template <>
struct redux_abs_max_nan_propagation_sync_warp_t0t15_t16t31<float>{
  CUTLASS_DEVICE
  float operator()(float const &max) const {
#if defined(CUTLASS_ARCH_CREDUX_ENABLED)
    int half_warp_idx = threadIdx.x / (NumThreadsPerWarp / 2);
    bool first_half_threads = (half_warp_idx % 2) == 0;
    float value0 =  first_half_threads ? max : 0;
    float v0 = cutlass::redux_abs_max_nan_propagation_sync_warp<float>{}(value0);

    float value1 = !first_half_threads ? max : 0;
    float v1 = cutlass::redux_abs_max_nan_propagation_sync_warp<float>{}(value1);
    return first_half_threads ? v0: v1;
    
#elif defined(__CUDA_ARCH__)
    float abs_max = cutlass::absolute_value_op<float>{}(max);
    cutlass::maximum<float, /*PropagateNaN*/true> max_op;
    constexpr int shuffle_width = 16;
    CUTLASS_PRAGMA_UNROLL
    for(int offset = shuffle_width/2; offset > 0; offset /= 2) {
      float value = __shfl_down_sync(0xffffffff, abs_max, offset, shuffle_width);
        abs_max  = max_op(abs_max,value);
    }
    // Broadcast the maximum to all threads participating in the reduction.
    abs_max = __shfl_sync(0xffffffff, abs_max, 0, shuffle_width);
    return abs_max;
#else 
    CUTLASS_UNUSED(max);
    CUTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for nvcuda::wmma::fragment<Use, m, n, k, T, Layout>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

template<typename Use, int m, int n, int k, typename T, typename Layout>
struct plus<nvcuda::wmma::fragment<Use, m, n, k, T, Layout>>
{
  using Fragment = nvcuda::wmma::fragment<Use, m, n, k, T, Layout>;
  using ElementType = typename Fragment::element_type;

  CUTLASS_HOST_DEVICE
  Fragment operator()(Fragment const &lhs, Fragment const &rhs) const
  {
    Fragment result;
    plus<ElementType> scalar_op;

    ElementType *result_elts = reinterpret_cast<ElementType*>(&result);
    const ElementType *lhs_elts = reinterpret_cast<const ElementType*>(&lhs);
    const ElementType *rhs_elts = reinterpret_cast<const ElementType*>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Fragment::num_elements; i++) {
      result_elts[i] = scalar_op(lhs_elts[i], rhs_elts[i]);
    }

    return result;
  }
};

#endif // defined(CUTLASS_ARCH_WMMA_ENABLED)



/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
