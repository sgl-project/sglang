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
#pragma once

#include <cute/numeric/math.hpp>      // cute::max, etc
#include <cute/util/print.hpp>        // cute::print
#include <cute/util/type_traits.hpp>  // __CUTE_REQUIRES, cute::is_std_integral

namespace cute
{

// A constant value: short name and type-deduction for fast compilation
template <auto v>
struct C {
  using type = C<v>;
  static constexpr auto value = v;
  using value_type = decltype(v);
  CUTE_HOST_DEVICE constexpr operator   value_type() const noexcept { return value; }
  CUTE_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

// Deprecate
template <class T, T v>
using constant = C<v>;

template <bool b>
using bool_constant = C<b>;

using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;

// A more std:: conforming integral_constant that enforces type but interops with C<v>
template <class T, T v>
struct integral_constant : C<v> {
  using type = integral_constant<T,v>;
  static constexpr T value = v;
  using value_type = T;
  // Disambiguate C<v>::operator value_type()
  //CUTE_HOST_DEVICE constexpr operator   value_type() const noexcept { return value; }
  CUTE_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

//
// Traits
//

// Use cute::is_std_integral<T> to match built-in integral types (int, int64_t, unsigned, etc)
// Use cute::is_integral<T> to match both built-in integral types AND static integral types.
template <class T>
struct is_integral : bool_constant<is_std_integral<T>::value> {};
template <auto v>
struct is_integral<C<v>                  > : true_type {};
template <class T, T v>
struct is_integral<integral_constant<T,v>> : true_type {};
template <class T>
constexpr bool is_integral_v = is_integral<T>::value;

// Register FastDivmod as integral type
template<>
struct is_integral<cutlass::FastDivmod> : true_type {};

// is_static detects if an (abstract) value is defined completely by its type (no members)
template <class T>
struct is_static : bool_constant<is_empty<T>::value> {};
template <class T>
struct is_static<T const > : is_static<T> {};
template <class T>
struct is_static<T const&> : is_static<T> {};
template <class T>
struct is_static<T      &> : is_static<T> {};
template <class T>
struct is_static<T     &&> : is_static<T> {};
template <class T>
constexpr bool is_static_v = is_static<T>::value;

// is_constant detects if a type is a static integral type and if v is equal to a value
template <auto n, class T>
struct is_constant : false_type {};
template <auto n, auto v>
struct is_constant<n, C<v>                  > : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant<n, integral_constant<T,v>> : bool_constant<v == n> {};
template <auto n, class T>
struct is_constant<n, T const > : is_constant<n,T> {};
template <auto n, class T>
struct is_constant<n, T const&> : is_constant<n,T> {};
template <auto n, class T>
struct is_constant<n, T      &> : is_constant<n,T> {};
template <auto n, class T>
struct is_constant<n, T     &&> : is_constant<n,T> {};
template <auto n, class T>
constexpr bool is_constant_v = is_constant<n,T>::value;

//
// Specializations
//

template <int v>
using Int = C<v>;

using _m32    = Int<-32>;
using _m24    = Int<-24>;
using _m16    = Int<-16>;
using _m12    = Int<-12>;
using _m10    = Int<-10>;
using _m9     = Int<-9>;
using _m8     = Int<-8>;
using _m7     = Int<-7>;
using _m6     = Int<-6>;
using _m5     = Int<-5>;
using _m4     = Int<-4>;
using _m3     = Int<-3>;
using _m2     = Int<-2>;
using _m1     = Int<-1>;
using _0      = Int<0>;
using _1      = Int<1>;
using _2      = Int<2>;
using _3      = Int<3>;
using _4      = Int<4>;
using _5      = Int<5>;
using _6      = Int<6>;
using _7      = Int<7>;
using _8      = Int<8>;
using _9      = Int<9>;
using _10     = Int<10>;
using _12     = Int<12>;
using _16     = Int<16>;
using _24     = Int<24>;
using _32     = Int<32>;
using _40     = Int<40>;
using _48     = Int<48>;
using _56     = Int<56>;
using _64     = Int<64>;
using _72     = Int<72>;
using _80     = Int<80>;
using _88     = Int<88>;
using _96     = Int<96>;
using _104    = Int<104>;
using _112    = Int<112>;
using _120    = Int<120>;
using _128    = Int<128>;
using _136    = Int<136>;
using _144    = Int<144>;
using _152    = Int<152>;
using _160    = Int<160>;
using _168    = Int<168>;
using _176    = Int<176>;
using _184    = Int<184>;
using _192    = Int<192>;
using _200    = Int<200>;
using _208    = Int<208>;
using _216    = Int<216>;
using _224    = Int<224>;
using _232    = Int<232>;
using _240    = Int<240>;
using _248    = Int<248>;
using _256    = Int<256>;
using _384    = Int<384>;
using _512    = Int<512>;
using _768    = Int<768>;
using _1024   = Int<1024>;
using _2048   = Int<2048>;
using _4096   = Int<4096>;
using _8192   = Int<8192>;
using _16384  = Int<16384>;
using _32768  = Int<32768>;
using _65536  = Int<65536>;
using _131072 = Int<131072>;
using _262144 = Int<262144>;
using _524288 = Int<524288>;

/***************/
/** Operators **/
/***************/

#define CUTE_LEFT_UNARY_OP(OP)                                       \
  template <auto t>                                                  \
  CUTE_HOST_DEVICE constexpr                                         \
  C<(OP t)> operator OP (C<t>) {                                     \
    return {};                                                       \
  }
#define CUTE_RIGHT_UNARY_OP(OP)                                      \
  template <auto t>                                                  \
  CUTE_HOST_DEVICE constexpr                                         \
  C<(t OP)> operator OP (C<t>) {                                     \
    return {};                                                       \
  }
#define CUTE_BINARY_OP(OP)                                           \
  template <auto t, auto u>                                          \
  CUTE_HOST_DEVICE constexpr                                         \
  C<(t OP u)> operator OP (C<t>, C<u>) {                             \
    return {};                                                       \
  }

CUTE_LEFT_UNARY_OP(+);
CUTE_LEFT_UNARY_OP(-);
CUTE_LEFT_UNARY_OP(~);
CUTE_LEFT_UNARY_OP(!);
CUTE_LEFT_UNARY_OP(*);

CUTE_BINARY_OP( +);
CUTE_BINARY_OP( -);
CUTE_BINARY_OP( *);
CUTE_BINARY_OP( /);
CUTE_BINARY_OP( %);
CUTE_BINARY_OP( &);
CUTE_BINARY_OP( |);
CUTE_BINARY_OP( ^);
CUTE_BINARY_OP(<<);
CUTE_BINARY_OP(>>);

CUTE_BINARY_OP(&&);
CUTE_BINARY_OP(||);

CUTE_BINARY_OP(==);
CUTE_BINARY_OP(!=);
CUTE_BINARY_OP( >);
CUTE_BINARY_OP( <);
CUTE_BINARY_OP(>=);
CUTE_BINARY_OP(<=);

#undef CUTE_BINARY_OP
#undef CUTE_LEFT_UNARY_OP
#undef CUTE_RIGHT_UNARY_OP

//
// Mixed static-dynamic special cases
//

template <auto t, class U,
          __CUTE_REQUIRES(is_std_integral<U>::value && t == 0)>
CUTE_HOST_DEVICE constexpr
C<0>
operator*(C<t>, U) {
  return {};
}

template <class U, auto t,
          __CUTE_REQUIRES(is_std_integral<U>::value && t == 0)>
CUTE_HOST_DEVICE constexpr
C<0>
operator*(U, C<t>) {
  return {};
}

template <auto t, class U,
          __CUTE_REQUIRES(is_std_integral<U>::value && t == 0)>
CUTE_HOST_DEVICE constexpr
C<0>
operator/(C<t>, U) {
  return {};
}

template <class U, auto t,
          __CUTE_REQUIRES(is_std_integral<U>::value && (t == 1 || t == -1))>
CUTE_HOST_DEVICE constexpr
C<0>
operator%(U, C<t>) {
  return {};
}

template <auto t, class U,
          __CUTE_REQUIRES(is_std_integral<U>::value && t == 0)>
CUTE_HOST_DEVICE constexpr
C<0>
operator%(C<t>, U) {
  return {};
}

template <auto t, class U,
          __CUTE_REQUIRES(is_std_integral<U>::value && t == 0)>
CUTE_HOST_DEVICE constexpr
C<0>
operator&(C<t>, U) {
  return {};
}

template <class U, auto t,
          __CUTE_REQUIRES(is_std_integral<U>::value && t == 0)>
CUTE_HOST_DEVICE constexpr
C<0>
operator&(U, C<t>) {
  return {};
}

template <auto t, class U,
          __CUTE_REQUIRES(is_std_integral<U>::value && !bool(t))>
CUTE_HOST_DEVICE constexpr
C<false>
operator&&(C<t>, U) {
  return {};
}

template <auto t, class U,
          __CUTE_REQUIRES(is_std_integral<U>::value && !bool(t))>
CUTE_HOST_DEVICE constexpr
C<false>
operator&&(U, C<t>) {
  return {};
}

template <class U, auto t,
          __CUTE_REQUIRES(is_std_integral<U>::value && bool(t))>
CUTE_HOST_DEVICE constexpr
C<true>
operator||(C<t>, U) {
  return {};
}

template <class U, auto t,
          __CUTE_REQUIRES(is_std_integral<U>::value && bool(t))>
CUTE_HOST_DEVICE constexpr
C<true>
operator||(U, C<t>) {
  return {};
}

//
// Named functions from math.hpp
//

#define CUTE_NAMED_UNARY_FN(OP)                                      \
  template <auto t>                                                  \
  CUTE_HOST_DEVICE constexpr                                         \
  auto OP (C<t>) {                                                   \
    return C<OP(t)>{};                                               \
  }
#define CUTE_NAMED_BINARY_FN(OP)                                     \
  template <auto t, auto u>                                          \
  CUTE_HOST_DEVICE constexpr                                         \
  auto OP (C<t>, C<u>) {                                             \
    return C<OP(t,u)>{};                                             \
  }                                                                  \
  template <auto t, class U,                                         \
            __CUTE_REQUIRES(is_std_integral<U>::value)>              \
  CUTE_HOST_DEVICE constexpr                                         \
  auto OP (C<t>, U u) {                                              \
    return OP(t,u);                                                  \
  }                                                                  \
  template <class T, auto u,                                         \
            __CUTE_REQUIRES(is_std_integral<T>::value)>              \
  CUTE_HOST_DEVICE constexpr                                         \
  auto OP (T t, C<u>) {                                              \
    return OP(t,u);                                                  \
  }

CUTE_NAMED_UNARY_FN(abs);
CUTE_NAMED_UNARY_FN(signum);
CUTE_NAMED_UNARY_FN(has_single_bit);

CUTE_NAMED_BINARY_FN(max);
CUTE_NAMED_BINARY_FN(min);
CUTE_NAMED_BINARY_FN(shiftl);
CUTE_NAMED_BINARY_FN(shiftr);
CUTE_NAMED_BINARY_FN(gcd);
CUTE_NAMED_BINARY_FN(lcm);

#undef CUTE_NAMED_UNARY_FN
#undef CUTE_NAMED_BINARY_FN

//
// Other functions
//

template <auto t, auto u>
CUTE_HOST_DEVICE constexpr
C<t / u>
safe_div(C<t>, C<u>) {
  static_assert(t % u == 0, "Static safe_div requires t % u == 0");
  return {};
}

template <auto t, class U,
          __CUTE_REQUIRES(is_std_integral<U>::value)>
CUTE_HOST_DEVICE constexpr
auto
safe_div(C<t>, U u) {
  return t / u;
}

template <class T, auto u,
          __CUTE_REQUIRES(is_std_integral<T>::value)>
CUTE_HOST_DEVICE constexpr
auto
safe_div(T t, C<u>) {
  return t / u;
}

template <class TrueType, class FalseType>
CUTE_HOST_DEVICE constexpr
decltype(auto)
conditional_return(true_type, TrueType&& t, FalseType&&) {
  return static_cast<TrueType&&>(t);
}

template <class TrueType, class FalseType>
CUTE_HOST_DEVICE constexpr
decltype(auto)
conditional_return(false_type, TrueType&&, FalseType&& f) {
  return static_cast<FalseType&&>(f);
}

template <auto v>
CUTE_HOST_DEVICE constexpr
auto
conditional_return(bool b, C<v> const&, C<v> const&) {
  return C<v>{};
}

template <auto v, auto u>
CUTE_HOST_DEVICE constexpr
auto
conditional_return(bool b, C<v> const&, C<u> const&) {
  return b ? v : u;
}

// TrueType and FalseType must have a common type
template <class TrueType, class FalseType>
CUTE_HOST_DEVICE constexpr
auto
conditional_return(bool b, TrueType const& t, FalseType const& f) {
  return b ? t : f;
}

// TrueType and FalseType don't require a common type
template <bool b, class TrueType, class FalseType>
CUTE_HOST_DEVICE constexpr
auto
conditional_return(TrueType const& t, FalseType const& f) {
  if constexpr (b) {
    return t;
  } else {
    return f;
  }
}

template <class Trait>
CUTE_HOST_DEVICE constexpr
auto
static_value()
{
  if constexpr (is_std_integral<decltype(Trait::value)>::value) {
    return Int<Trait::value>{};
  } else {
    return Trait::value;
  }
  CUTE_GCC_UNREACHABLE;
}

//
// Display utilities
//

template <auto Value>
CUTE_HOST_DEVICE void print(C<Value>) {
  printf("_");
  ::cute::print(Value);
}

#if !defined(__CUDACC_RTC__)
template <auto t>
CUTE_HOST std::ostream& operator<<(std::ostream& os, C<t> const&) {
  return os << "_" << t;
}
#endif


namespace detail {

// parse_int_digits takes a variadic number of digits and converts them into an int
template <class... Ts>
constexpr uint64_t parse_int_digits(uint64_t result, int digit, Ts... digits)
{
  if constexpr (sizeof...(Ts) == 0) {
    return 10 * result + digit;
  } else {
    return parse_int_digits(10 * result + digit, digits...);
  }
}

} // end namespace detail


// This user-defined literal operator allows cute::constant written as literals. For example,
//
//    auto var = 32_c;
//
//  var has type cute::constant<int,32>.
//
template <char... digits>
constexpr cute::constant<int,detail::parse_int_digits(0, (digits - '0')...)> operator "" _c()
{
  static_assert((('0' <= digits && digits <= '9') && ...),
                "Expected 0 <= digit <= 9 for each digit of the integer.");
  return {};
}

} // end namespace cute
