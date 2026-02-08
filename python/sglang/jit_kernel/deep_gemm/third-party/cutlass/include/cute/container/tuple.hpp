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
#include <cute/numeric/integral_constant.hpp>  // cute::true_type, cute::false_type
#include <cute/numeric/integer_sequence.hpp>

#include <cute/container/cuda_types.hpp>
#include <cute/container/type_list.hpp>
//#include <cute/container/array.hpp>            // Advanced optimizations

// cute::tuple is like std::tuple, with differences:
//
// 1. It works on both host and device.
// 2. Its template arguments must be semiregular types.
// 3. It is always a standard-layout type if all of its template arguments are standard-layout types.
// 4. It is always an empty type if all of its template arguments are empty types.
//
// Semiregular types are default constructible and copyable.
// They include "value types" like int or float,
// but do _not_ include references like int& or float&.
// (See std::tie for an example of a tuple of references.)
//
// Standard-layout types preserve ABI across host-device boundaries. They are safe to use as device kernel parameters.
// The standard-layout requirement prevents a more common EBO-based implemented of cute::tuple.
//
// The cute::tuple is also simplified over the implementations in std::, cuda::std::, and thrust:: by ignoring much of
// the conversion SFINAE, special overloading, and avoiding cvref template types.
//
// Over standard-conforming tuple implementations, this appears to accelerate compilation times by over 3x.

namespace cute
{

template <class... T>
struct tuple;

namespace eso
{

// ESO stands for "empty structure optimization."
// We use this technique to ensure that cute::tuple doesn't waste space
// storing template arguments that have no data (like integral_constant).
// Empty types in the template argument list are not even constructed,
// and do not have unique element addresses. Calling `get`
// constructs and returns an instance of an empty type on demand.

template <bool IsFirstEmpty, bool IsRestEmpty, class... T>
struct ESO;

template <class First, class... Rest>
static constexpr bool is_first_empty_v = cute::is_empty<First>::value;
template <class First, class... Rest>
static constexpr bool is_rest_empty_v  = (cute::is_empty<Rest>::value && ...);

template <class... T>
using ESO_t = ESO<is_first_empty_v<T...>, is_rest_empty_v<T...>, T...>;

// Empty First and Empty Rest...
template <class First, class... Rest>
struct ESO<true, true, First, Rest...> {
  CUTE_HOST_DEVICE constexpr
  ESO() {}

  CUTE_HOST_DEVICE constexpr
  ESO(First const&, Rest const&...) {}
};

// NonEmpty First and Empty Rest...
template <class First, class... Rest>
struct ESO<false, true, First, Rest...> {
  CUTE_HOST_DEVICE constexpr
  ESO() : first_{} {}

  CUTE_HOST_DEVICE constexpr
  ESO(First const& first, Rest const&...) : first_{first} {}

  First first_;
};

// Empty First and NonEmpty Rest...
template <class First, class... Rest>
struct ESO<true, false, First, Rest...> {
  CUTE_HOST_DEVICE constexpr
  ESO() : rest_{} {}

  CUTE_HOST_DEVICE constexpr
  ESO(First const&, Rest const&... rest) : rest_{rest...} {}

  ESO_t<Rest...> rest_;
};

// NonEmpty T and NonEmpty Rest...
template <class First, class... Rest>
struct ESO<false, false, First, Rest...> {
  CUTE_HOST_DEVICE constexpr
  ESO() : first_{}, rest_{} {}

  CUTE_HOST_DEVICE constexpr
  ESO(First const& first, Rest const&... rest) : first_{first}, rest_{rest...} {}

  First first_;
  ESO_t<Rest...> rest_;
};

// Get Nth value from ESO
template <class R, size_t N, class S>
CUTE_HOST_DEVICE constexpr
R
getr(S&& s) noexcept
{
  if constexpr (N == 0) {
    return static_cast<S&&>(s).first_;
  } else {
    return getr<R,N-1>(static_cast<S&&>(s).rest_);
  }
  CUTE_GCC_UNREACHABLE;
}

// Compilers disagree on decltype(auto), so these implementations avoid it at cost
template <size_t N, bool F, bool R, class... T>
CUTE_HOST_DEVICE constexpr
cute::conditional_t<cute::is_empty<cute::tuple_element_t<N, cute::tuple<T...>>>::value,
                    cute::tuple_element_t<N, cute::tuple<T...>>,
                    cute::tuple_element_t<N, cute::tuple<T...>> const&>
getv_cr(ESO<F, R, T...> const& s) noexcept
{
  if constexpr (cute::is_empty<cute::tuple_element_t<N, cute::tuple<T...>>>::value) {
    return {};
  } else {
    return getr<cute::tuple_element_t<N, cute::tuple<T...>> const&, N>(s);
  }
  CUTE_GCC_UNREACHABLE;
}

template <size_t N, bool F, bool R, class... T>
CUTE_HOST_DEVICE constexpr
cute::conditional_t<cute::is_empty<cute::tuple_element_t<N, cute::tuple<T...>>>::value,
                    cute::tuple_element_t<N, cute::tuple<T...>>,
                    cute::tuple_element_t<N, cute::tuple<T...>> &>
getv_r(ESO<F, R, T...>& s) noexcept
{
  if constexpr (cute::is_empty<cute::tuple_element_t<N, cute::tuple<T...>>>::value) {
    return {};
  } else {
    return getr<cute::tuple_element_t<N, cute::tuple<T...>> &, N>(s);
  }
  CUTE_GCC_UNREACHABLE;
}

template <size_t N, bool F, bool R, class... T>
CUTE_HOST_DEVICE constexpr
cute::conditional_t<cute::is_empty<cute::tuple_element_t<N, cute::tuple<T...>>>::value,
                    cute::tuple_element_t<N, cute::tuple<T...>>,
                    cute::tuple_element_t<N, cute::tuple<T...>> &&>
getv_rr(ESO<F, R, T...>&& s) noexcept
{
  if constexpr (cute::is_empty<cute::tuple_element_t<N, cute::tuple<T...>>>::value) {
    return {};
  } else {
    return getr<cute::tuple_element_t<N, cute::tuple<T...>> &&, N>(static_cast<ESO<F, R, T...>&&>(s));
  }
  CUTE_GCC_UNREACHABLE;
}

} // end namespace eso

template <class... T>
struct tuple : eso::ESO_t<T...>
{
  CUTE_HOST_DEVICE constexpr
  tuple() {}

  CUTE_HOST_DEVICE constexpr
  tuple(T const&... t) : eso::ESO_t<T...>(t...) {}
};

template <>
struct tuple<> {};

//
// make_tuple (value-based implementation)
//

template <class... T>
CUTE_HOST_DEVICE constexpr
tuple<T...>
make_tuple(T const&... t)
{
  return {t...};
}

// Returns the element in the ith position of the tuple
template <size_t I, class... T>
CUTE_HOST_DEVICE constexpr
decltype(auto)
get(tuple<T...> const& t) noexcept
{
  static_assert(I < sizeof...(T), "Index out of range");
  return eso::getv_cr<I>(t);
}

template <size_t I, class... T>
CUTE_HOST_DEVICE constexpr
decltype(auto)
get(tuple<T...>& t) noexcept
{
  static_assert(I < sizeof...(T), "Index out of range");
  return eso::getv_r<I>(t);
}

template <size_t I, class... T>
CUTE_HOST_DEVICE constexpr
decltype(auto)
get(tuple<T...>&& t) noexcept
{
  static_assert(I < sizeof...(T), "Index out of range");
  return eso::getv_rr<I>(static_cast<eso::ESO_t<T...>&&>(t));
}

// Returns the first position of type X (as a static integer) in the tuple
// type's argument list.
template <class X, class... T>
CUTE_HOST_DEVICE constexpr
auto
find(tuple<T...> const&) noexcept
{
  return cute::C<find_true_v<cute::is_same_v<X,T>...>>{};
}

//
// Custom is_tuple trait simply checks the existence of tuple_size
//      and assumes get<I>(.), tuple_element<I,.>
//
namespace detail {

template <class T>
auto has_tuple_size( T*) -> bool_constant<(0 <= tuple_size<T>::value)>;
auto has_tuple_size(...) -> false_type;

} // end namespace detail

template <class T>
struct is_tuple : decltype(detail::has_tuple_size((T*)0)) {};

template <class T>
static constexpr bool is_tuple_v = cute::is_tuple<T>::value;

//
// tuple_cat concatenates multiple cute::tuple into a single cute::tuple,
// just like std::tuple_cat for std::tuple.
//

#if 0
// Original implementation

namespace detail {

template <class T0, class T1,
          size_t... I0, size_t... I1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1,
          index_sequence<I0...>, index_sequence<I1...>)
{
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)...);
}

} // end namespace detail

CUTE_HOST_DEVICE constexpr
tuple<>
tuple_cat()
{
  return {};
}

template <class Tuple,
          __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST_DEVICE constexpr
Tuple const&
tuple_cat(Tuple const& t)
{
  return t;
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1)
{
  return detail::tuple_cat(t0, t1,
                           make_index_sequence<tuple_size<T0>::value>{},
                           make_index_sequence<tuple_size<T1>::value>{});
}

template <class T0, class T1, class T2, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2, Ts const&... ts)
{
  return cute::tuple_cat(cute::tuple_cat(t0,t1),t2,ts...);
}
#endif

#if 1
// Extended implementation

namespace detail {

template <class T0, class T1,
          size_t... I0, size_t... I1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1,
          index_sequence<I0...>, index_sequence<I1...>)
{
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)...);
}

template <class T0, class T1, class T2,
          size_t... I0, size_t... I1, size_t... I2>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2,
          index_sequence<I0...>, index_sequence<I1...>, index_sequence<I2...>)
{
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)...);
}

template <class T0, class T1, class T2, class T3,
          size_t... I0, size_t... I1, size_t... I2, size_t... I3>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3,
          index_sequence<I0...>, index_sequence<I1...>, index_sequence<I2...>, index_sequence<I3...>)
{
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)...);
}

template <class T0, class T1, class T2, class T3, class T4,
          size_t... I0, size_t... I1, size_t... I2, size_t... I3, size_t... I4>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4,
          index_sequence<I0...>, index_sequence<I1...>, index_sequence<I2...>, index_sequence<I3...>, index_sequence<I4...>)
{
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)..., get<I4>(t4)...);
}

template <class T0, class T1>
struct tuple_cat_static;

template <class... T0s, class... T1s>
struct tuple_cat_static<tuple<T0s...>, tuple<T1s...>> {
  using type = tuple<T0s..., T1s...>;
};

} // end namespace detail

CUTE_HOST_DEVICE constexpr
tuple<>
tuple_cat()
{
  return {};
}

template <class Tuple,
          __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST_DEVICE constexpr
Tuple const&
tuple_cat(Tuple const& t)
{
  return t;
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1)
{
  if constexpr (is_static<T0>::value && is_static<T1>::value &&
		is_tuple<T0>::value && is_tuple<T1>::value) {
    return typename detail::tuple_cat_static<T0, T1>::type{};
  } else {
    return detail::tuple_cat(t0, t1,
                           make_index_sequence<tuple_size<T0>::value>{},
                           make_index_sequence<tuple_size<T1>::value>{});
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T0, class T1, class T2>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2)
{
  return detail::tuple_cat(t0, t1, t2,
                           make_index_sequence<tuple_size<T0>::value>{},
                           make_index_sequence<tuple_size<T1>::value>{},
                           make_index_sequence<tuple_size<T2>::value>{});
}

template <class T0, class T1, class T2, class T3>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3)
{
  return detail::tuple_cat(t0, t1, t2, t3,
                           make_index_sequence<tuple_size<T0>::value>{},
                           make_index_sequence<tuple_size<T1>::value>{},
                           make_index_sequence<tuple_size<T2>::value>{},
                           make_index_sequence<tuple_size<T3>::value>{});
}

template <class T0, class T1, class T2, class T3, class T4>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4)
{
  return detail::tuple_cat(t0, t1, t2, t3, t4,
                           make_index_sequence<tuple_size<T0>::value>{},
                           make_index_sequence<tuple_size<T1>::value>{},
                           make_index_sequence<tuple_size<T2>::value>{},
                           make_index_sequence<tuple_size<T3>::value>{},
                           make_index_sequence<tuple_size<T4>::value>{});
}

template <class T0, class T1, class T2, class T3, class T4, class T5, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4, T5 const& t5, Ts const&... ts)
{
  return cute::tuple_cat(cute::tuple_cat(t0,t1,t2,t3,t4), cute::tuple_cat(t5, ts...));
}
#endif

#if 0
// Outer-Inner indexing trick to concat all tuples at once

namespace detail {

template <size_t... Ns>
struct tuple_cat_helper
{
  static constexpr cute::array<size_t,sizeof...(Ns)> ns = {Ns...};

  static constexpr size_t total_size() {
    size_t sum = 0;
    for (size_t n : ns) sum += n;
    return sum;
  }
  static constexpr size_t total_size_ = total_size();

  static constexpr auto values() {
    cute::array<size_t[2],total_size_> outer_inner = {};

    size_t idx = 0;
    for (size_t i = 0; i < ns.size(); ++i) {
      for (size_t j = 0; j < ns[i]; ++j, ++idx) {
        outer_inner[idx][0] = i;
        outer_inner[idx][1] = j;
      }
    }
    return outer_inner;
  }
  static constexpr auto outer_inner_ = values();

  using total_sequence = make_index_sequence<total_size_>;
};

template <class Helper, class Tuple, size_t... I>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(Tuple const& t, index_sequence<I...>)
{
  return cute::make_tuple(get<Helper::outer_inner_[I][1]>(get<Helper::outer_inner_[I][0]>(t))...);
}

template <class T0, class T1,
          size_t... I0, size_t... I1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1,
          index_sequence<I0...>, index_sequence<I1...>)
{
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)...);
}

} // end namespace detail

CUTE_HOST_DEVICE constexpr
tuple<>
tuple_cat()
{
  return {};
}

template <class Tuple,
          __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST_DEVICE constexpr
Tuple const&
tuple_cat(Tuple const& t)
{
  return t;
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1)
{
  return detail::tuple_cat(t0, t1,
                           make_index_sequence<tuple_size<T0>::value>{},
                           make_index_sequence<tuple_size<T1>::value>{});
}

template <class... Tuples>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(Tuples const&... ts)
{
  using Helper = detail::tuple_cat_helper<tuple_size<Tuples>::value...>;
  return detail::tuple_cat<Helper>(cute::make_tuple(ts...), typename Helper::total_sequence{});
}
#endif

//
// Equality operators
//

namespace detail {

template <class TupleA, class TupleB, size_t... I>
CUTE_HOST_DEVICE constexpr
auto
equal_impl(TupleA const& a, TupleB const& b, index_sequence<I...>)
{
  return (cute::true_type{} && ... && (get<I>(a) == get<I>(b)));
}

} // end namespace detail

template <class TupleT, class TupleU,
          __CUTE_REQUIRES(is_tuple<TupleT>::value && is_tuple<TupleU>::value)>
CUTE_HOST_DEVICE constexpr
auto
operator==(TupleT const& t, TupleU const& u)
{
  if constexpr (tuple_size<TupleT>::value == tuple_size<TupleU>::value) {
    return detail::equal_impl(t, u, make_index_sequence<tuple_size<TupleT>::value>{});
  } else {
    return cute::false_type{};
  }

  CUTE_GCC_UNREACHABLE;
}

template <class TupleT, class TupleU,
          __CUTE_REQUIRES(is_tuple<TupleT>::value ^ is_tuple<TupleU>::value)>
CUTE_HOST_DEVICE constexpr
auto
operator==(TupleT const& t, TupleU const& u)
{
  return cute::false_type{};
}

template <class TupleT, class TupleU,
          __CUTE_REQUIRES(is_tuple<TupleT>::value && is_tuple<TupleU>::value)>
CUTE_HOST_DEVICE constexpr
auto
operator!=(TupleT const& t, TupleU const& u)
{
  return !(t == u);
}

template <class TupleT, class TupleU,
          __CUTE_REQUIRES(is_tuple<TupleT>::value ^ is_tuple<TupleU>::value)>
CUTE_HOST_DEVICE constexpr
auto
operator!=(TupleT const& t, TupleU const& u)
{
  return cute::true_type{};
}

//
// Comparison operators
//

//
// There are many ways to compare tuple of elements and because CuTe is built
//   on parameterizing layouts of coordinates, some comparisons are appropriate
//   only in certain cases.
//  -- lexicographical comparison [reverse, reflected, revref]
//  -- colexicographical comparison [reverse, reflected, revref]
//  -- element-wise comparison [any,all]
// This can be very confusing. To avoid errors in selecting the appropriate
//   comparison, op<|op<=|op>|op>= are *not* implemented for cute::tuple.
//
// That said, see int_tuple for more explicitly named common comparison ops.
//

//
// Display utilities
//

namespace detail {

template <class Tuple, size_t... Is>
CUTE_HOST_DEVICE void print_tuple(Tuple const& t, index_sequence<Is...>, char s = '(', char e = ')')
{
  using cute::print;
  if (sizeof...(Is) == 0) {
    print(s);
  } else {
    ((void(print(Is == 0 ? s : ',')), void(print(get<Is>(t)))), ...);
  }
  print(e);
}

#if !defined(__CUDACC_RTC__)
template <class Tuple, std::size_t... Is>
CUTE_HOST std::ostream& print_tuple_os(std::ostream& os, Tuple const& t, index_sequence<Is...>, char s = '(', char e = ')')
{
  if (sizeof...(Is) == 0) {
    os << s;
  } else {
    (void(os << (Is == 0 ? s : ',') << get<Is>(t)), ...);
  }
  return os << e;
}
#endif // !defined(__CUDACC_RTC__)

} // end namespace detail

template <class Tuple,
          __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST_DEVICE void print(Tuple const& t)
{
  return detail::print_tuple(t, make_index_sequence<tuple_size<Tuple>::value>{});
}

#if !defined(__CUDACC_RTC__)
template <class Tuple,
          __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST std::ostream& operator<<(std::ostream& os, Tuple const& t)
{
  return detail::print_tuple_os(os, t, make_index_sequence<tuple_size<Tuple>::value>{});
}
#endif // !defined(__CUDACC_RTC__)

} // end namespace cute

namespace CUTE_STL_NAMESPACE
{

template <class... T>
struct tuple_size<cute::tuple<T...>>
    : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, cute::tuple<T...>>
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
struct tuple_size<cute::tuple<T...>>
    : CUTE_STL_NAMESPACE::integral_constant<size_t, sizeof...(T)>
{};

template <size_t I, class... T>
struct tuple_element<I, cute::tuple<T...>>
    : CUTE_STL_NAMESPACE::tuple_element<I, CUTE_STL_NAMESPACE::tuple<T...>>
{};

} // end namespace std
#endif // CUTE_STL_NAMESPACE_IS_CUDA_STD
