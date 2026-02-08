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

#include <cute/config.hpp>                     // CUTE_INLINE_CONSTANT, CUTE_HOST_DEVICE
#include <cute/container/tuple.hpp>            // cute::is_tuple
#include <cute/numeric/integral_constant.hpp>  // cute::false_type, cute::true_type

namespace cute
{

// For slicing
struct Underscore : Int<0> {};

CUTE_INLINE_CONSTANT Underscore _;

// Convenient alias
using X = Underscore;

// Treat Underscore as an integral like integral_constant
template <>
struct is_integral<Underscore> : true_type {};

template <class T>
struct is_underscore : false_type {};
template <>
struct is_underscore<Underscore> : true_type {};

// Tuple trait for detecting static member element
template <class Tuple, class Elem, class Enable = void>
struct has_elem : false_type {};
template <class Elem>
struct has_elem<Elem, Elem> : true_type {};
template <class Tuple, class Elem>
struct has_elem<Tuple, Elem, enable_if_t<is_tuple<Tuple>::value> >
    : has_elem<Tuple, Elem, tuple_seq<Tuple> > {};
template <class Tuple, class Elem, int... Is>
struct has_elem<Tuple, Elem, seq<Is...>>
    : disjunction<has_elem<tuple_element_t<Is, Tuple>, Elem>...> {};

// Tuple trait for detecting static member element
template <class Tuple, class Elem, class Enable = void>
struct all_elem : false_type {};
template <class Elem>
struct all_elem<Elem, Elem> : true_type {};
template <class Tuple, class Elem>
struct all_elem<Tuple, Elem, enable_if_t<is_tuple<Tuple>::value> >
    : all_elem<Tuple, Elem, tuple_seq<Tuple> > {};
template <class Tuple, class Elem, int... Is>
struct all_elem<Tuple, Elem, seq<Is...>>
    : conjunction<all_elem<tuple_element_t<Is, Tuple>, Elem>...> {};

// Tuple trait for detecting Underscore member
template <class Tuple>
using has_underscore = has_elem<Tuple, Underscore>;

template <class Tuple>
using all_underscore = all_elem<Tuple, Underscore>;

template <class Tuple>
using has_int1 = has_elem<Tuple, Int<1>>;

template <class Tuple>
using has_int0 = has_elem<Tuple, Int<0>>;

//
// Slice keeps only the elements of Tuple B that are paired with an Underscore
//

namespace detail {

template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
lift_slice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return lift_slice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return cute::tuple<B>{b};
  } else {
    return cute::tuple<>{};
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

// Entry point overrides the lifting so that slice(_,b) == b
template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
slice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return detail::lift_slice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return b;
  } else {
    return cute::tuple<>{};
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Dice keeps only the elements of Tuple B that are paired with an Int
//

namespace detail {

template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
lift_dice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return lift_dice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return cute::tuple<>{};
  } else {
    return cute::tuple<B>{b};
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

// Entry point overrides the lifting so that dice(1,b) == b
template <class A, class B>
CUTE_HOST_DEVICE constexpr
auto
dice(A const& a, B const& b)
{
  if constexpr (is_tuple<A>::value) {
    static_assert(tuple_size<A>::value == tuple_size<B>::value, "Mismatched Ranks");
    return filter_tuple(a, b, [](auto const& x, auto const& y) { return detail::lift_dice(x,y); });
  } else if constexpr (is_underscore<A>::value) {
    return cute::tuple<>{};
  } else {
    return b;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Display utilities
//

CUTE_HOST_DEVICE void print(Underscore const&) {
  printf("_");
}

#if !defined(__CUDACC_RTC__)
CUTE_HOST std::ostream& operator<<(std::ostream& os, Underscore const&) {
  return os << "_";
}
#endif

} // end namespace cute
