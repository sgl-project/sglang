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

#include <cute/config.hpp>

#include <cute/container/tuple.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/util/type_traits.hpp>

namespace cute
{

template <class... T>
struct ArithmeticTuple : public tuple<T...> {
  CUTE_HOST_DEVICE constexpr
  ArithmeticTuple() : tuple<T...>() {}

  CUTE_HOST_DEVICE constexpr
  ArithmeticTuple(tuple<T...> const& t) : tuple<T...>(t) {}

  CUTE_HOST_DEVICE constexpr
  ArithmeticTuple(T const&... t) : tuple<T...>(t...) {}
};

template <class... T>
struct is_tuple<ArithmeticTuple<T...>> : true_type {};

template <class... Ts>
struct is_flat<ArithmeticTuple<Ts...>> : is_flat<tuple<Ts...>> {};

template <class... T>
CUTE_HOST_DEVICE constexpr
auto
make_arithmetic_tuple(T const&... t) {
  return ArithmeticTuple<T...>(t...);
}

template <class T>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(T const& t) {
  if constexpr (is_tuple<T>::value) {
    return detail::tapply(t, [](auto const& x){ return as_arithmetic_tuple(x); },
                          [](auto const&... a){ return make_arithmetic_tuple(a...); },
                          tuple_seq<T>{});
  } else {
    return t;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Numeric operators
//

// Addition
template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(append<R>(t,Int<0>{}), append<R>(u,Int<0>{}), plus{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, tuple<U...> const& u) {
  return t + ArithmeticTuple<U...>(u);
}

template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(tuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  return ArithmeticTuple<T...>(t) + u;
}

// Subtraction
template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator-(ArithmeticTuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(append<R>(t,Int<0>{}), append<R>(u,Int<0>{}), minus{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator-(ArithmeticTuple<T...> const& t, tuple<U...> const& u) {
  return t - ArithmeticTuple<U...>(u);
}

template <class... T, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator-(tuple<T...> const& t, ArithmeticTuple<U...> const& u) {
  return ArithmeticTuple<T...>(t) - u;
}

// Negation
template <class... T>
CUTE_HOST_DEVICE constexpr
auto
operator-(ArithmeticTuple<T...> const& t) {
  return transform_apply(t, negate{}, [](auto const&... a){ return make_arithmetic_tuple(a...); });
}

//
// Special cases for C<0>
//

template <auto t, class... U>
CUTE_HOST_DEVICE constexpr
ArithmeticTuple<U...>
operator+(C<t>, ArithmeticTuple<U...> const& u) {
  static_assert(t == 0, "Arithmetic tuple op+ error!");
  return u;
}

template <class... T, auto u>
CUTE_HOST_DEVICE constexpr
ArithmeticTuple<T...>
operator+(ArithmeticTuple<T...> const& t, C<u>) {
  static_assert(u == 0, "Arithmetic tuple op+ error!");
  return t;
}

template <auto t, class... U>
CUTE_HOST_DEVICE constexpr
ArithmeticTuple<U...>
operator-(C<t>, ArithmeticTuple<U...> const& u) {
  static_assert(t == 0, "Arithmetic tuple op- error!");
  return -u;
}

template <class... T, auto u>
CUTE_HOST_DEVICE constexpr
ArithmeticTuple<T...>
operator-(ArithmeticTuple<T...> const& t, C<u>) {
  static_assert(u == 0, "Arithmetic tuple op- error!");
  return t;
}

//
// ArithmeticTupleIterator
//

template <class ArithTuple>
struct ArithmeticTupleIterator
{
  using value_type   = ArithTuple;
  using element_type = ArithTuple;
  using reference    = ArithTuple;

  ArithTuple coord_;

  CUTE_HOST_DEVICE constexpr
  ArithmeticTupleIterator(ArithTuple const& coord = {}) : coord_(coord) {}

  CUTE_HOST_DEVICE constexpr
  ArithTuple operator*() const { return coord_; }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto operator[](Coord const& c) const { return *(*this + c); }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto operator+(Coord const& c) const {
    return ArithmeticTupleIterator<remove_cvref_t<decltype(coord_ + c)>>(coord_ + c);
  }
};

template <class... Ts>
CUTE_HOST_DEVICE constexpr
auto
make_inttuple_iter(Ts const&... ts) {
  return ArithmeticTupleIterator(as_arithmetic_tuple(ts...));
}

//
// ArithmeticTuple "basis" elements
//   A ScaledBasis<T,Ns...> is a (at least) rank-N+1 ArithmeticTuple:
//      (_0,_0,...,T,_0,...)
//   with value T in the Nth mode

template <class T, int... Ns>
struct ScaledBasis : private tuple<T>
{
  CUTE_HOST_DEVICE constexpr
  ScaledBasis(T const& t = {}) : tuple<T>(t) {}

  CUTE_HOST_DEVICE constexpr
  decltype(auto) value()       { return get<0>(static_cast<tuple<T>      &>(*this)); }
  CUTE_HOST_DEVICE constexpr
  decltype(auto) value() const { return get<0>(static_cast<tuple<T> const&>(*this)); }

  // Deprecated: Get the first hierarchical mode in this basis.
  CUTE_HOST_DEVICE static constexpr
  auto mode() { return get<0>(int_sequence<Ns...>{}); }
};

// Ensure flat representation
template <class T, int... Ms, int... Ns>
struct ScaledBasis<ScaledBasis<T, Ms...>, Ns...> : ScaledBasis<T, Ns..., Ms...> {};

template <class T>
struct is_scaled_basis : false_type {};
template <class T, int... Ns>
struct is_scaled_basis<ScaledBasis<T,Ns...>> : true_type {};

template <class T, int... Ns>
struct is_integral<ScaledBasis<T,Ns...>> : true_type {};

// Shortcuts
// E<>    := _1
// E<0>   := (_1,_0,_0,...)
// E<1>   := (_0,_1,_0,...)
// E<0,0> := ((_1,_0,_0,...),_0,_0,...)
// E<0,1> := ((_0,_1,_0,...),_0,_0,...)
// E<1,0> := (_0,(_1,_0,_0,...),_0,...)
// E<1,1> := (_0,(_0,_1,_0,...),_0,...)
template <int... Ns>
using E = ScaledBasis<Int<1>,Ns...>;

// Apply the Ns... pack to another Tuple
template <class T, class Tuple>
CUTE_HOST_DEVICE decltype(auto)
basis_get(T const&, Tuple&& t)
{
  return static_cast<Tuple&&>(t);
}

template <class T, int... Ns, class Tuple>
CUTE_HOST_DEVICE decltype(auto)
basis_get(ScaledBasis<T,Ns...> const&, Tuple&& t)
{
  if constexpr (sizeof...(Ns) == 0) {
    return static_cast<Tuple&&>(t);
  } else {
    return get<Ns...>(static_cast<Tuple&&>(t));
  }
  CUTE_GCC_UNREACHABLE;
}

template <class T>
CUTE_HOST_DEVICE decltype(auto)
basis_value(T const& e) {
  if constexpr (is_scaled_basis<T>::value) {
    return e.value();
  } else {
    return e;
  }
  CUTE_GCC_UNREACHABLE;
}

namespace detail {

template <class T, int... I>
CUTE_HOST_DEVICE constexpr
auto
to_atuple_i(T const& t, seq<I...>) {
  return make_arithmetic_tuple((void(I),Int<0>{})..., t);
}

} // end namespace detail

// Turn a ScaledBases<T,N> into a rank-N+1 ArithmeticTuple
//    with N prefix 0s:  (_0,_0,...N...,_0,T)
template <class T>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(ScaledBasis<T> const& t) {
  return t.value();
}

template <class T, int N, int... Ns>
CUTE_HOST_DEVICE constexpr
auto
as_arithmetic_tuple(ScaledBasis<T,N,Ns...> const& t) {
  return detail::to_atuple_i(as_arithmetic_tuple(ScaledBasis<T,Ns...>{t.value()}), make_seq<N>{});
}

template <int... Ns, class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_basis_like(Shape const& shape)
{
  if constexpr (is_tuple<Shape>::value) {
    // Generate bases for each mode of shape
    return transform(tuple_seq<Shape>{}, shape, [](auto I, auto si) {
      // Generate bases for each si and add an i on end
      return make_basis_like<Ns...,decltype(I)::value>(si);
    });
  } else {
    return E<Ns...>{};
  }
  CUTE_GCC_UNREACHABLE;
}

//
// Arithmetic
//

template <class T, int... Ns, class U>
CUTE_HOST_DEVICE constexpr
auto
safe_div(ScaledBasis<T,Ns...> const& b, U const& u)
{
  auto t = safe_div(b.value(), u);
  return ScaledBasis<decltype(t),Ns...>{t};
}

template <class T, int... Ns, class U>
CUTE_HOST_DEVICE constexpr
auto
ceil_div(ScaledBasis<T,Ns...> const& b, U const& u)
{
  auto t = ceil_div(b.value(), u);
  return ScaledBasis<decltype(t),Ns...>{t};
}

template <class T, int... Ns>
CUTE_HOST_DEVICE constexpr
auto
abs(ScaledBasis<T,Ns...> const& e)
{
  auto t = abs(e.value());
  return ScaledBasis<decltype(t),Ns...>{t};
}

// Equality
template <class T, int... Ns, class U, int... Ms>
CUTE_HOST_DEVICE constexpr
auto
operator==(ScaledBasis<T,Ns...> const& t, ScaledBasis<U,Ms...> const& u) {
  if constexpr (sizeof...(Ns) == sizeof...(Ms)) {
    return bool_constant<((Ns == Ms) && ...)>{} && t.value() == u.value();
  } else {
    return false_type{};
  }
  CUTE_GCC_UNREACHABLE;
}

// Not equal to anything else
template <class T, int... Ns, class U>
CUTE_HOST_DEVICE constexpr
false_type
operator==(ScaledBasis<T,Ns...> const&, U const&) {
  return {};
}

template <class T, class U, int... Ms>
CUTE_HOST_DEVICE constexpr
false_type
operator==(T const&, ScaledBasis<U,Ms...> const&) {
  return {};
}

// Multiplication
template <class A, class T, int... Ns>
CUTE_HOST_DEVICE constexpr
auto
operator*(A const& a, ScaledBasis<T,Ns...> const& e) {
  auto r = a * e.value();
  return ScaledBasis<decltype(r),Ns...>{r};
}

template <class T, int... Ns, class B>
CUTE_HOST_DEVICE constexpr
auto
operator*(ScaledBasis<T,Ns...> const& e, B const& b) {
  auto r = e.value() * b;
  return ScaledBasis<decltype(r),Ns...>{r};
}

// Addition
template <class T, int... Ns, class U, int... Ms>
CUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,Ns...> const& t, ScaledBasis<U,Ms...> const& u) {
  return as_arithmetic_tuple(t) + as_arithmetic_tuple(u);
}

template <class T, int... Ns, class... U>
CUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,Ns...> const& t, ArithmeticTuple<U...> const& u) {
  return as_arithmetic_tuple(t) + u;
}

template <class... T, class U, int... Ms>
CUTE_HOST_DEVICE constexpr
auto
operator+(ArithmeticTuple<T...> const& t, ScaledBasis<U,Ms...> const& u) {
  return t + as_arithmetic_tuple(u);
}

template <auto t, class U, int... Ms>
CUTE_HOST_DEVICE constexpr
auto
operator+(C<t>, ScaledBasis<U,Ms...> const& u) {
  if constexpr (sizeof...(Ms) == 0) {
    return C<t>{} + u.value();
  } else {
    static_assert(t == 0, "ScaledBasis op+ error!");
    return u;
  }
  CUTE_GCC_UNREACHABLE;
}

template <class T, int... Ns, auto u>
CUTE_HOST_DEVICE constexpr
auto
operator+(ScaledBasis<T,Ns...> const& t, C<u>) {
  if constexpr (sizeof...(Ns) == 0) {
    return t.value() + C<u>{};
  } else {
    static_assert(u == 0, "ScaledBasis op+ error!");
    return t;
  }
  CUTE_GCC_UNREACHABLE;
}

//
// Display utilities
//

template <class ArithTuple>
CUTE_HOST_DEVICE void print(ArithmeticTupleIterator<ArithTuple> const& iter)
{
  printf("ArithTuple"); print(iter.coord_);
}

template <class T, int... Ns>
CUTE_HOST_DEVICE void print(ScaledBasis<T,Ns...> const& e)
{
  print(e.value());
  // Param pack trick to print in reverse
  [[maybe_unused]] int dummy; (dummy = ... = (void(printf("@%d", Ns)), 0));
}

#if !defined(__CUDACC_RTC__)
template <class ArithTuple>
CUTE_HOST std::ostream& operator<<(std::ostream& os, ArithmeticTupleIterator<ArithTuple> const& iter)
{
  return os << "ArithTuple" << iter.coord_;
}

template <class T, int... Ns>
CUTE_HOST std::ostream& operator<<(std::ostream& os, ScaledBasis<T,Ns...> const& e)
{
  os << e.value();
  // Param pack trick to print in reverse
  [[maybe_unused]] int dummy; (dummy = ... = (void(os << "@" << Ns),0));
  return os;
}
#endif

} // end namespace cute


namespace CUTE_STL_NAMESPACE
{

template <class... T>
struct tuple_size<cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::tuple_element<I, CUTE_STL_NAMESPACE::tuple<T...>>
{};

} // end namespace CUTE_STL_NAMESPACE

#ifdef CUTE_STL_NAMESPACE_IS_CUDA_STD
namespace std
{

#if defined(__CUDACC_RTC__)
template <class... _Tp>
struct tuple_size;

template <size_t _Ip, class... _Tp>
struct tuple_element;
#endif

template <class... T>
struct tuple_size<cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, cute::ArithmeticTuple<T...>>
  : CUTE_STL_NAMESPACE::tuple_element<I, CUTE_STL_NAMESPACE::tuple<T...>>
{};

} // end namespace std
#endif // CUTE_STL_NAMESPACE_IS_CUDA_STD
