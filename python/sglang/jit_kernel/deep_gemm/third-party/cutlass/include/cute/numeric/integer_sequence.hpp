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
#include <cute/util/type_traits.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace cute
{

using CUTE_STL_NAMESPACE::integer_sequence;
using CUTE_STL_NAMESPACE::make_integer_sequence;

namespace detail {

template <class T, class S, T Begin>
struct range_impl;

template <class T, T... N, T Begin>
struct range_impl<T, integer_sequence<T, N...>, Begin> {
  using type = integer_sequence<T, N+Begin...>;
};

template <class S>
struct reverse_impl;

template <class T, T... N>
struct reverse_impl<integer_sequence<T, N...>> {
  using type = integer_sequence<T, sizeof...(N)-1-N...>;
};

} // end namespace detail

template <class T, T Begin, T End>
using make_integer_range = typename detail::range_impl<
    T,
    make_integer_sequence<T, (End-Begin > 0) ? (End-Begin) : 0>,
    Begin>::type;

template <class T, T N>
using make_integer_sequence_reverse = typename detail::reverse_impl<
    make_integer_sequence<T, N>>::type;

//
// Common aliases
//

// int_sequence

template <int... Ints>
using int_sequence = integer_sequence<int, Ints...>;

template <int N>
using make_int_sequence = make_integer_sequence<int, N>;

template <int N>
using make_int_rsequence = make_integer_sequence_reverse<int, N>;

template <int Begin, int End>
using make_int_range = make_integer_range<int, Begin, End>;

// index_sequence

template <size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

template <size_t N>
using make_index_rsequence = make_integer_sequence_reverse<size_t, N>;

template <size_t Begin, size_t End>
using make_index_range = make_integer_range<size_t, Begin, End>;

//
// Shortcuts
//

template <int... Ints>
using seq = int_sequence<Ints...>;

template <int N>
using make_seq = make_int_sequence<N>;

template <int N>
using make_rseq = make_int_rsequence<N>;

template <int Min, int Max>
using make_range = make_int_range<Min, Max>;

template <class Tuple>
using tuple_seq = make_seq<tuple_size<remove_cvref_t<Tuple>>::value>;

template <class Tuple>
using tuple_rseq = make_rseq<tuple_size<remove_cvref_t<Tuple>>::value>;

//
// Convert a parameter pack to an int sequence
//

template <class T>
struct to_seq;

template <>
struct to_seq<integer_sequence<int>> {
  using type = seq<>;
};

template <int I, int... Is>
struct to_seq<integer_sequence<int, I, Is...>> {
  using type = seq<I, Is...>;
};

template <template <class...> class TupleLike, class... Ts>
struct to_seq<TupleLike<Ts...>> {
  using type = seq<Ts::value...>;
};

template <class T>
using to_seq_t = typename to_seq<T>::type;

//
// Specialize cute::tuple-traits for std::integer_sequence
//

template <class T, T... Ints>
struct tuple_size<integer_sequence<T, Ints...>>
    : cute::integral_constant<size_t, sizeof...(Ints)>
{};

template <size_t I, class T, T... Is>
struct tuple_element<I, integer_sequence<T, Is...>>
{
  constexpr static T idx[sizeof...(Is)] = {Is...};
  using type = cute::integral_constant<T, idx[I]>;
};

template <size_t I, class T, T... Ints>
CUTE_HOST_DEVICE constexpr
tuple_element_t<I, integer_sequence<T, Ints...>>
get(integer_sequence<T, Ints...>) {
  static_assert(I < sizeof...(Ints), "Index out of range");
  return {};
}

} // end namespace cute
