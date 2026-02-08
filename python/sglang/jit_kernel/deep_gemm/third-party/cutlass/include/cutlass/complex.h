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

#include <cuComplex.h>

#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#endif
#include "cutlass/functional.h"
#include "cutlass/platform/platform.h"
#include "cutlass/real.h"

#include "cutlass/numeric_types.h"

#include "cutlass/fast_math.h"

#if !defined(__CUDACC_RTC__)
#include <iosfwd>
#endif

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Enumeraed type describing a transformation on a complex value.
enum class ComplexTransform {
  kNone,
  kConjugate
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines ComplexTransform inversions
template <ComplexTransform kTransform>
struct InvertComplexTransform;

/// Invert ComplexTransform from kNone to kConjugate
template <>
struct InvertComplexTransform<ComplexTransform::kNone> {
  static ComplexTransform const transform = ComplexTransform::kConjugate;
};

/// Invert ComplexTransform from kConjugate to kNone
template <>
struct InvertComplexTransform<ComplexTransform::kConjugate> {
  static ComplexTransform const transform = ComplexTransform::kNone;
};
/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for CUDA complex types
//

#if !defined(__CUDACC_RTC__)
/// Returns the real part of the complex number
CUTLASS_HOST_DEVICE
float const &real(cuFloatComplex const &z) { return z.x; }

/// Returns the real part of the complex number
CUTLASS_HOST_DEVICE
float &real(cuFloatComplex &z) { return z.x; }

/// Returns the real part of the complex number
CUTLASS_HOST_DEVICE
double const &real(cuDoubleComplex const &z) { return z.x; }

/// Returns the real part of the complex number
CUTLASS_HOST_DEVICE
double &real(cuDoubleComplex &z) { return z.x; }

/// Returns the imaginary part of the complex number
CUTLASS_HOST_DEVICE
float const &imag(cuFloatComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
CUTLASS_HOST_DEVICE
float &imag(cuFloatComplex &z) { return z.y; }

/// Returns the imaginary part of the complex number
CUTLASS_HOST_DEVICE
double const &imag(cuDoubleComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
CUTLASS_HOST_DEVICE
double &imag(cuDoubleComplex &z) { return z.y; }

// Returns the conjugate of the complex number
CUTLASS_HOST_DEVICE cuFloatComplex
conj(cuFloatComplex const& z) {
  return make_cuFloatComplex(z.x, -z.y);
}

// Returns the conjugate of the complex number
CUTLASS_HOST_DEVICE cuDoubleComplex
conj(cuDoubleComplex const& z) {
  return make_cuDoubleComplex(z.x, -z.y);
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for representing and manipulating complex numbers with conversions from built-in CUDA
/// complex types.

template <typename T>
class complex
{
 public:
  /// Type alias for scalar type
  using value_type = T;

 private:
  //
  // Data members
  //

  /// Real part
  T _real;

  /// Imaginary part
  T _imag;

 public:

//
// Methods
//

  /// Default constructor
  complex() = default;

  /// Constructor
  CUTLASS_HOST_DEVICE
  complex(T r) : _real(r), _imag(T(0)) {}

  /// Constructor
  CUTLASS_HOST_DEVICE
  complex(T r, T i) : _real(r), _imag(i) {}

  /// Constructor
  template<typename A>
  CUTLASS_HOST_DEVICE
  complex(complex<A> const &z) : _real(static_cast<T>(z.real())), _imag(static_cast<T>(z.imag())) {}


  #if !defined(__CUDACC_RTC__)
  /// Conversion from cuFloatComplex
  CUTLASS_HOST_DEVICE
  complex(cuFloatComplex const &z) : _real(static_cast<T>(cuCrealf(z))), _imag(static_cast<T>(cuCimagf(z))) {}

  /// Conversion from cuDoubleComplex
  CUTLASS_HOST_DEVICE
  complex(cuDoubleComplex const &z) : _real(static_cast<T>(cuCreal(z))), _imag(static_cast<T>(cuCimag(z))) {}
  #endif

  /// Equality operator
  CUTLASS_HOST_DEVICE bool operator==(complex<T> const &rhs) const {
    return this->real() == rhs.real() && this->imag() == rhs.imag();
  }

  /// Inequality operator
  CUTLASS_HOST_DEVICE bool operator!=(complex<T> const &rhs) const {
    return !(*this == rhs);
  }

  /// Addition
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator+(complex<A> const &rhs) const {
    return complex<T>(this->real() + rhs.real(), this->imag() + rhs.imag());
  }

  /// Reduction into memory address.  Components may update out of order.
  template <typename OtherT>
  CUTLASS_DEVICE void red(complex<OtherT> *ptr) const {
    static_assert(platform::is_same<T, OtherT>::value, "Component type must match");
    cutlass::atomic_add<T> reduce;
    reduce(&ptr->_real, _real);
    reduce(&ptr->_imag, _imag);
  }

  /// Reduction into memory address.  Components may update out of order.  (Half specialization)
  CUTLASS_DEVICE void red(complex<half_t> *ptr) const {
    static_assert(platform::is_same<T, half_t>::value, "Component type must match");
    half2 *h2_ptr = reinterpret_cast<half2*>(ptr);
    half2 h2_data = reinterpret_cast<half2&>(*this);
    cutlass::atomic_add<half2> reduce;
    reduce(h2_ptr, h2_data);
  }

  /// Subtraction
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator-(complex<A> const &rhs) const {
    return complex<T>(this->real() - rhs.real(), this->imag() - rhs.imag());
  }

  /// Multiplication
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator*(complex<A> const &rhs) const {
    return complex<T>(this->real() * rhs.real() - this->imag() * rhs.imag(),
                      this->real() * rhs.imag() + this->imag() * rhs.real());
  }

  /// Scalar Multiplication
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator*(A const &s) const {
    return complex<T>(this->real() * s, this->imag() * s);
  }

  /// Division
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator/(complex<A> const &rhs) const {
    T d = T(rhs.real() * rhs.real() + rhs.imag() * rhs.imag());

    return complex<T>(
      (real() * rhs.real() + imag() * rhs.imag()) / d,
      (imag() * rhs.real() - real() * rhs.imag()) / d
    );
  }

  /// Scalar Division
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> operator/(A const &s) const {
    return complex<T>(this->real() / s, this->imag() / s);
  }

  /// Addition
    template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator+=(complex<A> const &rhs) {
      *this = *this + rhs;
      return *this;
  }

  /// Subtraction
  template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator-=(complex<A> const &rhs) {
      *this = *this - rhs;
      return *this;
  }

  /// Multiplication
  template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator*=(complex<A> const &rhs) {
      *this = *this * rhs;
      return *this;
  }

  /// Scalar multiplication
  template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator*=(A s) {
      *this = *this * s;
      return *this;
  }

  /// Division
  template <typename A>
  CUTLASS_HOST_DEVICE complex<T> &operator/=(complex<A> const &rhs) {
      *this = *this / rhs;
      return *this;
  }

  /// Accesses the real part of the complex number
  CUTLASS_HOST_DEVICE
  T const &real() const { return _real; }

  /// Accesses the real part of the complex number
  CUTLASS_HOST_DEVICE
  T &real() { return _real; }

  /// Accesses the imaginary part of the complex number
  CUTLASS_HOST_DEVICE
  T const &imag() const { return _imag; }

  /// Accesses the imaginary part of the complex number
  CUTLASS_HOST_DEVICE
  T &imag() { return _imag; }

  /// Set the real part of the complex number
  CUTLASS_HOST_DEVICE
  void real(T real) { _real = real; }

  /// Set the imaginary part of the complex number
  CUTLASS_HOST_DEVICE
  void imag(T imag) { _imag = imag; }

  #if !defined(__CUDACC_RTC__)
  /// Converts to cuFloatComplex
  CUTLASS_HOST_DEVICE
  explicit operator cuFloatComplex() const { return make_cuFloatComplex(float(real()), float(imag())); }

  /// Converts to cuDoubleComplex
  CUTLASS_HOST_DEVICE
  explicit operator cuDoubleComplex() const { return make_cuDoubleComplex(real(), imag()); }
  #endif
};

// Complex conjugate
template<class T>
CUTLASS_HOST_DEVICE complex<T> conj(complex<T> const& z) {
  return {z.real(), -z.imag()};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for complex template
//

// Nonmember real and imag need to work for non-complex numbers too.
// That means cutlass::complex, std::complex, cuda::std::complex, and
// any user-defined complex number type that looks like std::complex.
// It's reasonable to assume that a "complex number type" has
// zero-argument real() and imag() member functions returning
// non-void.  While cuFloatComplex and cuDoubleComplex lack those
// member functions, one-argument nonmember real and imag overloads
// for those types are defined above.

namespace detail {

template <typename T, typename Enable = void>
struct has_zero_argument_real_member_function :
  cutlass::platform::false_type
{};

template <typename T>
struct has_zero_argument_real_member_function<T,
  cutlass::platform::enable_if_t<
    ! cutlass::platform::is_void_v<
      decltype(cutlass::platform::declval<T>().real())
    >
  >
> : cutlass::platform::true_type
{};

template <typename T>
constexpr bool has_zero_argument_real_member_function_v =
  has_zero_argument_real_member_function<T>::value;

template <typename T, typename Enable = void>
struct has_zero_argument_imag_member_function :
  cutlass::platform::false_type
{};

template <typename T>
struct has_zero_argument_imag_member_function<T,
  cutlass::platform::enable_if_t<
    ! cutlass::platform::is_void_v<
      decltype(cutlass::platform::declval<T>().imag())
    >
  >
> : cutlass::platform::true_type
{};

template <typename T>
constexpr bool has_zero_argument_imag_member_function_v =
  has_zero_argument_imag_member_function<T>::value;

} // namespace detail

template<typename T>
CUTLASS_HOST_DEVICE auto real(T z) {
  if constexpr (detail::has_zero_argument_real_member_function_v<T>) {
    return z.real();
  } else {
    return z;
  }
}
  
template<typename T>
CUTLASS_HOST_DEVICE auto imag(T z) {
  if constexpr (detail::has_zero_argument_imag_member_function_v<T>) {
    return z.imag();
  } else {
    // Imaginary part of a non-complex input has the same type as the
    // input, and its value is zero.  CUTLASS assumes in this case
    // that value-initializing T is well-formed and results in zero.
    return T{};
  }
}
  
//
// Output operators
//

#if !defined(__CUDACC_RTC__)
template <typename T>
std::ostream &operator<<(std::ostream &out, complex<T> const &z) {
  T _r = real(z);
  T _i = imag(z);

  if (bool(_i)) {
    return out << _r << "+i" << _i;
  }
  return out << _r;
}
#endif

//
// Non-member operators defined for complex types
//


//
// Non-member functions defined for complex numbers
//

// abs returns the magnitude of the complex number.

CUTLASS_HOST_DEVICE float abs(complex<float> const &z) {
  return ::hypot(z.real(), z.imag());
}

CUTLASS_HOST_DEVICE double abs(complex<double> const &z) {
  return ::hypot(z.real(), z.imag());
}

// In theory, it would make sense to add a complex<long double>
// specialization of abs here, since hypot works for long double too.
// In practice, long double doesn't have a portable number of bits or
// behavior, so users who care about higher-precision floating-point
// computation should probably insist on an actual FP128 type.

template <typename T>
CUTLASS_HOST_DEVICE T abs(complex<T> const &z) {
  // cutlass::complex permits all kinds of T, including types that
  // don't have NaN.  For a generic floating-point type with Inf
  // and/or NaN, LAPACK's DLAPY2 algorithm would make sense, as it
  // would handle issues like avoiding unwarranted overflow if
  // z.real() or z.imag() is slightly bigger than the square root of
  // the max finite number.  That could be a future improvement; for
  // now, the code just uses the naive algorithm.
  //
  // Use the "swap two-step" idiom so that argument-dependent lookup
  // can find any CUTLASS-specific overloads.
  using cutlass::sqrt;
  return sqrt(z.real() * z.real() + z.imag() * z.imag());
}

/// Returns the magnitude of the complex number
template <typename T>
CUTLASS_HOST_DEVICE T arg(complex<T> const &z) {
  return atan2(imag(z), real(z));
}

/// Returns the squared magnitude of a real number
template <typename T>
CUTLASS_HOST_DEVICE T norm(T const &z) {
    return z * z;
}

/// Returns the squared magnitude of a real number
template <>
CUTLASS_HOST_DEVICE int8_t norm(int8_t const &z) {
    return static_cast<int8_t>(z * z);
}

/// Returns the squared magnitude of a complex number
template <typename T>
CUTLASS_HOST_DEVICE double norm(complex<T> const &z) {
  return real(z) * real(z) + imag(z) * imag(z);
}

/// Norm-accumulate calculation
template <typename T, typename R>
CUTLASS_HOST_DEVICE R norm_accumulate(T const &x, R const & accumulator) {
  return accumulator + static_cast<R>(x) * static_cast<R>(x);
}

/// Norm accumulate specialized for complex types
template <typename T, typename R>
CUTLASS_HOST_DEVICE R norm_accumulate(complex<T> const &z, R const &accumulator) {
  return accumulator + static_cast<R>(real(z)) * static_cast<R>(real(z)) +
    static_cast<R>(imag(z)) * static_cast<R>(imag(z));
}

namespace detail {
  
template<class T>
CUTLASS_HOST_DEVICE T conj_impl(T const& z, cutlass::platform::true_type) {
  return conj(z);
}

template<class T>
CUTLASS_HOST_DEVICE T conj_impl(T const& z, cutlass::platform::false_type) {
  return z;
}

template<class T>
CUTLASS_HOST_DEVICE T conj_impl(T const& z) {
  constexpr bool use_unqualified_conj =
    ! cutlass::platform::is_arithmetic_v<T> &&
    ! detail::has_cutlass_conj_v<T> &&
    detail::has_unqualified_conj_v<T>;
  return conj_impl(z, cutlass::platform::bool_constant<use_unqualified_conj>{});
}
  
} // namespace detail

// Return the complex conjugate of the input.
//
// This MUST be a function and not a function object, because it may
// be common practice for downstream types to define specifically
// cutlass::conj overloads, instead of overloads in their namespace.
//
// As a result of this being a function and not a function object,
// CUTLASS code needs to declare "using cutlass::conj;" in scope and
// then call this function unqualified, just like std::swap.
//
// If an overload already exists for cutlass::conj(T), that overload
// will be called instead of this one.  Otherwise:
//
// 1. for arithmetic types, return z;
//
// 2. for types where (namespace-unqualified) conj(z) is well formed
//    and cutlass::conj(z) is NOT well formed, return conj(z); and,
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
// is findable via argument-dependent lookup, but does not live in the
// cutlass namespace).  It excludes cutlass::conj(z) in order to
// prevent infinite recursion.
//
// Case (3) covers non-Standard non-complex number types.
template<class T>
CUTLASS_HOST_DEVICE T conj(T const& z) {
  return detail::conj_impl(z);
}

/// Projects the complex number z onto the Riemann sphere
template <typename T>
CUTLASS_HOST_DEVICE complex<T> proj(complex<T> const &z) {
  T d = real(z) * real(z) + imag(z) * imag(z) + T(1);
  return complex<T>((T(2) * real(z)) / d, (T(2) * imag(z)) / d);
}

/// Returns a complex number with magnitude r and phase theta
template <typename T>
CUTLASS_HOST_DEVICE complex<T> polar(T const &r, T const &theta = T()) {
  return complex<T>(r * cos(theta), r * sin(theta));
}

/// Computes the complex exponential of z.
template <typename T>
CUTLASS_HOST_DEVICE complex<T> exp(complex<T> const &z) {
  return complex<T>(fast_exp(real(z)) * fast_cos(imag(z)), fast_exp(real(z)) * fast_sin(imag(z)));
}

/// Computes the log of z
template <typename T>
CUTLASS_HOST_DEVICE complex<T> log(complex<T> const &z) {
  return complex<T>(log(abs(z)), arg(z));
}

/// Computes the log base 10 of z
template <typename T>
CUTLASS_HOST_DEVICE complex<T> log10(complex<T> const &z) {
  return log(z) / T(log(T(10)));
}

/// Computes the square root of complex number z
template <typename T>
CUTLASS_HOST_DEVICE complex<T> sqrt(complex<T> const &z) {
  return sqrt(T(2)) / T(2) *
         complex<T>(sqrt(sqrt(norm(z)) + real(z)),
                    (imag(z) < 0 ? T(-1) : T(1)) * sqrt(sqrt(norm(z)) - real(z)));
}

/// Computes the cosine of complex z.
template <typename T>
CUTLASS_HOST_DEVICE complex<T> cos(complex<T> const &z) {
  return (exp(z) + exp(-z)) / T(2);
}

/// Computes the sin of complex z.
template <typename T>
CUTLASS_HOST_DEVICE complex<T> sin(complex<T> const &z) {
  return (exp(-z) - exp(z)) * complex<T>(T(0), T(1) / T(2));
}

/// Comparison
template <typename T>
CUTLASS_HOST_DEVICE bool operator<(complex<T> const &lhs, complex<T> const &rhs) {
  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex-valued type.
template <typename T>
struct RealType< complex<T> >
{
  using Type = T;

  /// Number of elements
  static int const kExtent = 2;

  CUTLASS_HOST_DEVICE
  static complex<T> from_real(double x) {
    return complex<T>(static_cast<T>(x));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_HOST_DEVICE
cutlass::complex<half_t> from_real<cutlass::complex<half_t> >(double r) {
  return cutlass::complex<half_t>(half_t(r));
}

template <>
CUTLASS_HOST_DEVICE
cutlass::complex<float> from_real<cutlass::complex<float> >(double r) {
  return cutlass::complex<float>(float(r));
}

template <>
CUTLASS_HOST_DEVICE
cutlass::complex<double> from_real<cutlass::complex<double> >(double r) {
  return cutlass::complex<double>(r);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct is_complex {
  static bool const value = false;
};

template <typename T>
struct is_complex<complex<T>> {
  static bool const value = true;
};


/////////////////////////////////////////////////////////////////////////////////////////////////
// functional.h numeric specializations
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Squares with optional conversion
template <typename T, typename Output>
struct magnitude_squared<complex<T>, Output> {
  CUTLASS_HOST_DEVICE
  Output operator()(complex<T> lhs) const {
    multiplies<Output> mul_op;

    Output y_r = Output(lhs.real());
    Output y_i = Output(lhs.imag());

    return mul_op(y_r, y_r) + mul_op(y_i, y_i);
  }
};

/// Fused multiply-add
template <typename T>
struct multiply_add<complex<T>, complex<T>, complex<T>> {
  CUTLASS_HOST_DEVICE
  complex<T> operator()(
    complex<T> const &a,
    complex<T> const &b,
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a.real() * b.real();
    real += -a.imag() * b.imag();
    imag += a.real() * b.imag();
    imag += a.imag () * b.real();

    return complex<T>{
      real,
      imag
    };
  }
};

/// Fused multiply-add
template <typename T>
struct multiply_add<complex<T>, T, complex<T>> {
  CUTLASS_HOST_DEVICE
  complex<T> operator()(
    complex<T> const &a,
    T const &b,
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a.real() * b;
    imag += a.imag () * b;

    return complex<T>{
      real,
      imag
    };
  }
};

/// Fused multiply-add
template <typename T>
struct multiply_add<T, complex<T>, complex<T>> {
  CUTLASS_HOST_DEVICE
  complex<T> operator()(
    T const &a,
    complex<T> const &b,
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a * b.real();
    imag += a * b.imag();

    return complex<T>{
      real,
      imag
    };
  }
};

/// Conjugate
template <typename T>
struct conjugate<complex<T>>  {
  CUTLASS_HOST_DEVICE
  complex<T> operator()(complex<T> const &a) const {
    // Invoke the complex<T> overload specifically, rather than
    // wasting the compiler's effort on overload resolution.
    return cutlass::conj(a);
  }
};

#if ! defined(__CUDACC_RTC__)
template <>
struct conjugate<cuFloatComplex>  {
  CUTLASS_HOST_DEVICE
  cuFloatComplex operator()(cuFloatComplex const& z) const {
    return make_cuFloatComplex(z.x, -z.y);
  }
};

template <>
struct conjugate<cuDoubleComplex>  {
  CUTLASS_HOST_DEVICE
  cuDoubleComplex operator()(cuDoubleComplex const& z) const {
    return make_cuDoubleComplex(z.x, -z.y);
  }
};
#endif
  
/// Computes the square of a difference with optional conversion
template <typename T, typename Output>
struct magnitude_squared_difference<complex<T>, Output> {
  CUTLASS_HOST_DEVICE
  Output operator()(complex<T> lhs, complex<T> rhs) const {
    multiplies<Output> mul_op;

    Output y_r = Output(lhs.real()) - Output(rhs.real());
    Output y_i = Output(lhs.imag()) - Output(rhs.imag());

    return mul_op(y_r, y_r) + mul_op(y_i, y_i);
  }
};

/// Reduces value into the data pointed to by ptr (complex<T> specialization)
template <typename T>
struct atomic_add<complex<T>> {
  CUTLASS_DEVICE
  void operator()(complex<T> *ptr, const complex<T> &data)
  {
    data.red(ptr);
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

//////////////////////////////////////////////////////////////////////////////////////////////////
