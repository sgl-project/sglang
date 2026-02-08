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

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE
#include <cute/numeric/integral_constant.hpp>  // cute::false_type, cute::true_type
#include <cute/numeric/math.hpp>               // cute::signum
#include <cute/util/type_traits.hpp>           // __CUTE_REQUIRES

namespace cute
{

/** Compile-time rational arithmetic type.
 * Like cute::C for std::integral_constant, cute::R for std::ratio has a short name
 *   for error messages and compile times.
 * The static data members @a num and @a den represent the reduced numerator and denominator
 *   of the rational value. Thus, two cute::R types with different @a n or @a d are distinct types
 *   even if they represent the same rational value.
 * A cute::R exposes the reduced canonical type via its ::type member.
 *   That is, cute::R<3,6>::type is cute::R<1,2> and cute::R<6,3>::type is cute::C<2>.
 * A cute::R<n,d>::value can be used much like any other trait::value. It can be involved in
 *   arithmetic expressions (according to the operator-overloads for cute::C and cute::R,
 *   though these may be incomplete) but with a potential rational value rather than an integral value.
 */
template <auto n, auto d>
class R {
  static_assert(d != 0);
  static constexpr auto an  = abs(n);
  static constexpr auto ad  = abs(d);
  static constexpr auto g   = gcd(an, ad);

 public:
  static constexpr auto num = signum(n) * signum(d) * an / g;
  static constexpr auto den =                         ad / g;
  // RI: den >= 1 && gcd(abs(num),den) == 1
  using type = typename conditional<num == 0 || den == 1, C<num>, R<num,den>>::type;
};

template <class T>
struct is_ratio : false_type {};
template <auto n, auto d>
struct is_ratio<R<n,d>> : true_type {};

template <auto a, auto b>
CUTE_HOST_DEVICE constexpr
typename R<a,b>::type
ratio(C<a>, C<b>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
typename R<a*c,b>::type
ratio(C<a>, R<b,c>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
typename R<b,a*c>::type
ratio(R<b,c>, C<a>) {
  return {};
}

template <auto a, auto b, auto c, auto d>
CUTE_HOST_DEVICE constexpr
typename R<a*d,b*c>::type
ratio(R<a,b>, R<c,d>) {
  return {};
}

//
// Non-reduced ratio implementations
//

template <auto a, auto b>
CUTE_HOST_DEVICE constexpr
R<a,b>
nratio(C<a>, C<b>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
R<a*c,b>
nratio(C<a>, R<b,c>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
R<b,a*c>
nratio(R<b,c>, C<a>) {
  return {};
}

template <auto a, auto b, auto c, auto d>
CUTE_HOST_DEVICE constexpr
R<a*d,b*c>
nratio(R<a,b>, R<c,d>) {
  return {};
}

//
// Operators
//

template <auto a, auto b, auto x, auto y>
CUTE_HOST_DEVICE constexpr
typename R<a*x,b*y>::type
operator*(R<a,b>, R<x,y>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
typename R<a*c,b>::type
operator*(R<a,b>, C<c>) {
  return {};
}

template <auto c, auto a, auto b>
CUTE_HOST_DEVICE constexpr
typename R<a*c,b>::type
operator*(C<c>, R<a,b>) {
  return {};
}

// Product with dynamic type needs to produce an integer...
template <class C, auto a, auto b,
          __CUTE_REQUIRES(cute::is_std_integral<C>::value)>
CUTE_HOST_DEVICE constexpr
auto
operator*(C const& c, R<a,b>) {
  return c * R<a,b>::num / R<a,b>::den;
}

// Product with dynamic type needs to produce an integer...
template <auto a, auto b, class C,
          __CUTE_REQUIRES(cute::is_std_integral<C>::value)>
CUTE_HOST_DEVICE constexpr
auto
operator*(R<a,b>, C const& c) {
  return c * R<a,b>::num / R<a,b>::den;
}

template <class C, auto a, auto b>
CUTE_HOST_DEVICE constexpr
auto
operator/(C const& c, R<a,b>) {
  return c * R<b,a>{};
}

template <auto a, auto b, auto x, auto y>
CUTE_HOST_DEVICE constexpr
typename R<a*y+b*x, b*y>::type
operator+(R<a,b>, R<x,y>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
typename R<a+c*b,b>::type
operator+(R<a,b>, C<c>) {
  return {};
}

template <auto c, auto a, auto b>
CUTE_HOST_DEVICE constexpr
typename R<a+c*b,b>::type
operator+(C<c>, R<a,b>) {
  return {};
}

/////////////////
// Comparisons //
/////////////////

template <auto a, auto b, auto x, auto y>
CUTE_HOST_DEVICE constexpr
bool_constant<R<a,b>::num == R<x,y>::num && R<a,b>::den == R<x,y>::den>
operator==(R<a,b>, R<x,y>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
bool_constant<R<a,b>::num == c && R<a,b>::den == 1>
operator==(R<a,b>, C<c>) {
  return {};
}

template <auto c, auto a, auto b>
CUTE_HOST_DEVICE constexpr
bool_constant<R<a,b>::num == c && R<a,b>::den == 1>
operator==(C<c>, R<a,b>) {
  return {};
}

template <auto a, auto b, auto x, auto y>
CUTE_HOST_DEVICE constexpr
bool_constant<R<a,b>::num * R<x,y>::den < R<x,y>::num * R<a,b>::den>
operator<(R<a,b>, R<x,y>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
bool_constant<R<a,b>::num < c * R<a,b>::den>
operator<(R<a,b>, C<c>) {
  return {};
}

template <auto c, auto x, auto y>
CUTE_HOST_DEVICE constexpr
bool_constant<c * R<x,y>::den < R<x,y>::num>
operator<(C<c>, R<x,y>) {
  return {};
}

///////////////////////
// Special functions //
///////////////////////

template <auto a, auto b, auto x, auto y>
CUTE_HOST_DEVICE constexpr
typename R<gcd(a*y,b*x),b*x>::type
gcd(R<a,b>, R<x,y>) {
  return {};
}

template <auto a, auto b, auto c>
CUTE_HOST_DEVICE constexpr
typename R<gcd(a,b*c),b*c>::type
gcd(R<a,b>, C<c>) {
  return {};
}

template <auto c, auto a, auto b>
CUTE_HOST_DEVICE constexpr
typename R<gcd(a,b*c),b*c>::type
gcd(C<c>, R<a,b>) {
  return {};
}

template <auto a, auto b>
CUTE_HOST_DEVICE constexpr
typename R<abs(a),abs(b)>::type
abs(R<a,b>) {
  return {};
}

template <auto a, auto b>
CUTE_HOST_DEVICE constexpr
int32_t
log_2(R<a,b>) {
  static_assert(R<a,b>::num > 0);
  static_assert(R<a,b>::den > 0);
  return log_2(static_cast<uint32_t>(R<a,b>::num)) - log_2(static_cast<uint32_t>(R<a,b>::den));
}

// @return A non-reduced ratio cute::R of the Trait0::value / Trait1::value
template <class Trait0, class Trait1>
CUTE_HOST_DEVICE constexpr
auto
trait_ratio(Trait0, Trait1) {
  return nratio(static_value<Trait0>(), static_value<Trait1>());
}

//
// Display utilities
//

template <auto a, auto b>
CUTE_HOST_DEVICE void print(R<a,b>) {
  print(C<a>{}); print("/"); print(C<b>{});
}

#if !defined(__CUDACC_RTC__)
template <auto a, auto b>
CUTE_HOST std::ostream& operator<<(std::ostream& os, R<a,b>) {
  return os << "_" << C<a>{} << "/" << C<b>{};
}
#endif

} // end namespace cute
