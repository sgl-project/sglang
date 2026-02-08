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
#include <cute/container/type_list.hpp>
#include <cute/container/tuple.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/numeric/integer_sequence.hpp>
#include <cute/numeric/integral_constant.hpp>

/// @file tuple_algorithms.hpp
/// @brief Common algorithms on (hierarchical) tuples
///
/// Code guidelines and style preferences:
///
/// For perfect forwarding, don't use std::forward, because it may not
/// be defined in device code when compiling with NVRTC. Instead, use
/// `static_cast<ParameterType&&>(parameter_name)`.
///
/// CuTe generally does not bother forwarding functions, as
/// reference-qualified member functions are rare in this code base.
///
/// Throughout CUTLASS, cute::make_tuple always needs to be called
/// namespace-qualified, EVEN If inside the cute namespace and/or in
/// scope of a "using namespace cute" declaration. Otherwise, the
/// compiler may select std::make_tuple instead of cute::make_tuple,
/// due to argument-dependent lookup.

namespace cute
{

//
// Apply (Unpack)
// (t, f) => f(t_0,t_1,...,t_n)
//

namespace detail {

template <class T, class F, int... I>
CUTE_HOST_DEVICE constexpr
auto
apply(T&& t, F&& f, seq<I...>)
{
  return f(get<I>(static_cast<T&&>(t))...);
}

} // end namespace detail

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
apply(T&& t, F&& f)
{
  return detail::apply(static_cast<T&&>(t), f, tuple_seq<T>{});
}

//
// Transform Apply
// (t, f, g) => g(f(t_0),f(t_1),...)
//

namespace detail {

template <class T, class F, class G, int... I>
CUTE_HOST_DEVICE constexpr
auto
tapply(T&& t, F&& f, G&& g, seq<I...>)
{
  return g(f(get<I>(static_cast<T&&>(t)))...);
}

template <class T0, class T1, class F, class G, int... I>
CUTE_HOST_DEVICE constexpr
auto
tapply(T0&& t0, T1&& t1, F&& f, G&& g, seq<I...>)
{
  return g(f(get<I>(static_cast<T0&&>(t0)),
             get<I>(static_cast<T1&&>(t1)))...);
}

template <class T0, class T1, class T2, class F, class G, int... I>
CUTE_HOST_DEVICE constexpr
auto
tapply(T0&& t0, T1&& t1, T2&& t2, F&& f, G&& g, seq<I...>)
{
  return g(f(get<I>(static_cast<T0&&>(t0)),
             get<I>(static_cast<T1&&>(t1)),
             get<I>(static_cast<T2&&>(t2)))...);
}

} // end namespace detail

template <class T, class F, class G>
CUTE_HOST_DEVICE constexpr
auto
transform_apply(T&& t, F&& f, G&& g)
{
  if constexpr (is_tuple<remove_cvref_t<T>>::value) {
    return detail::tapply(static_cast<T&&>(t), f, g, tuple_seq<T>{});
  } else {
    return g(f(static_cast<T&&>(t)));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T0, class T1, class F, class G>
CUTE_HOST_DEVICE constexpr
auto
transform_apply(T0&& t0, T1&& t1, F&& f, G&& g)
{
  if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
    return detail::tapply(static_cast<T0&&>(t0), static_cast<T1&&>(t1), f, g, tuple_seq<T0>{});
  } else {
    return g(f(static_cast<T0&&>(t0), static_cast<T1&&>(t1)));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T0, class T1, class T2, class F, class G>
CUTE_HOST_DEVICE constexpr
auto
transform_apply(T0&& t0, T1&& t1, T2&& t2, F&& f, G&& g)
{
  if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
    return detail::tapply(static_cast<T0&&>(t0), static_cast<T1&&>(t1), static_cast<T2&&>(t2), f, g, tuple_seq<T0>{});
  } else {
    return g(f(static_cast<T0&&>(t0), static_cast<T1&&>(t1), static_cast<T2&&>(t2)));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// For Each
// (t, f) => f(t_0),f(t_1),...,f(t_n)
//

template <class T, class F>
CUTE_HOST_DEVICE constexpr
void
for_each(T&& t, F&& f)
{
  if constexpr (is_tuple<remove_cvref_t<T>>::value) {
    return detail::apply(t, [&](auto&&... a) { (f(static_cast<decltype(a)&&>(a)), ...); }, tuple_seq<T>{});
  } else {
    return f(static_cast<T&&>(t));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
for_each_leaf(T&& t, F&& f)
{
  if constexpr (is_tuple<remove_cvref_t<T>>::value) {
    return detail::apply(static_cast<T&&>(t), [&](auto&&... a){ return (for_each_leaf(static_cast<decltype(a)&&>(a), f), ...); }, tuple_seq<T>{});
  } else {
    return f(static_cast<T&&>(t));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Transform
// (t, f) => (f(t_0),f(t_1),...,f(t_n))
//

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
transform(T const& t, F&& f)
{
  if constexpr (is_tuple<T>::value) {
    return detail::tapply(t, f, [](auto const&... a){ return cute::make_tuple(a...); }, tuple_seq<T>{});
  } else {
    return f(t);
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T0, class T1, class F>
CUTE_HOST_DEVICE constexpr
auto
transform(T0 const& t0, T1 const& t1, F&& f)
{
  if constexpr (is_tuple<T0>::value) {
    static_assert(tuple_size<T0>::value == tuple_size<T1>::value, "Mismatched tuple_size");
    return detail::tapply(t0, t1, f, [](auto const&... a){ return cute::make_tuple(a...); }, tuple_seq<T0>{});
  } else {
    return f(t0, t1);
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T0, class T1, class T2, class F>
CUTE_HOST_DEVICE constexpr
auto
transform(T0 const& t0, T1 const& t1, T2 const& t2, F&& f)
{
  if constexpr (is_tuple<T0>::value) {
    static_assert(tuple_size<T0>::value == tuple_size<T1>::value, "Mismatched tuple_size");
    static_assert(tuple_size<T0>::value == tuple_size<T2>::value, "Mismatched tuple_size");
    return detail::tapply(t0, t1, t2, f, [](auto const&... a){ return cute::make_tuple(a...); }, tuple_seq<T0>{});
  } else {
    return f(t0, t1, t2);
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
transform_leaf(T const& t, F&& f)
{
  if constexpr (is_tuple<T>::value) {
    return transform(t, [&](auto const& a) { return transform_leaf(a, f); });
  } else {
    return f(t);
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T0, class T1, class F>
CUTE_HOST_DEVICE constexpr
auto
transform_leaf(T0 const& t0, T1 const& t1, F&& f)
{
  if constexpr (is_tuple<T0>::value) {
    return transform(t0, t1, [&](auto const& a, auto const& b) { return transform_leaf(a, b, f); });
  } else {
    return f(t0, t1);
  }

  CUTE_GCC_UNREACHABLE;
}

//
// find and find_if
//

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
find_if(T const& t, F&& f)
{
  if constexpr (is_tuple<T>::value) {
    return detail::tapply(t, f, [] (auto... a) { return cute::C<find_true_v<decltype(a)::value...>>{}; }, tuple_seq<T>{});
  } else {
    return cute::C<decltype(f(t))::value ? 0 : 1>{};
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class X>
CUTE_HOST_DEVICE constexpr
auto
find(T const& t, X const& x)
{
  return find_if(t, [&](auto const& v) { return v == x; });  // This should always return a static true/false
}

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
any_of(T const& t, F&& f)
{
  if constexpr (is_tuple<T>::value) {
    return detail::tapply(t, f, [] (auto... a) { return (false_type{} || ... || a); }, tuple_seq<T>{});
  } else {
    return f(t);
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
all_of(T const& t, F&& f)
{
  if constexpr (is_tuple<T>::value) {
    return detail::tapply(t, f, [] (auto... a) { return (true_type{} && ... && a); }, tuple_seq<T>{});
  } else {
    return f(t);
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
none_of(T const& t, F&& f)
{
  return not any_of(t, f);
}

//
// Filter
// (t, f) => <f(t_0),f(t_1),...,f(t_n)>
//

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
filter_tuple(T const& t, F&& f)
{
  return transform_apply(t, f, [](auto const&... a) { return cute::tuple_cat(a...); });
}

template <class T0, class T1, class F>
CUTE_HOST_DEVICE constexpr
auto
filter_tuple(T0 const& t0, T1 const& t1, F&& f)
{
  return transform_apply(t0, t1, f, [](auto const&... a) { return cute::tuple_cat(a...); });
}

template <class T0, class T1, class T2, class F>
CUTE_HOST_DEVICE constexpr
auto
filter_tuple(T0 const& t0, T1 const& t1, T2 const& t2, F&& f)
{
  return transform_apply(t0, t1, t2, f, [](auto const&... a) { return cute::tuple_cat(a...); });
}

//
// Fold (Reduce, Accumulate)
// (t, v, f) => f(...f(f(v,t_0),t_1),...,t_n)
//

namespace detail {

template <class Fn, class Val>
struct FoldAdaptor {
  template <class X>
  CUTE_HOST_DEVICE constexpr auto operator|(X&& x) {
    auto r = fn_(val_, static_cast<X&&>(x));
    return FoldAdaptor<Fn, decltype(r)>{fn_, r};
  }
  Fn fn_;
  Val val_;
};

template <class T, class V, class F, int... Is>
CUTE_HOST_DEVICE constexpr
auto
fold(T&& t, V const& v, F&& f, seq<Is...>)
{
  return (FoldAdaptor<F,V>{f,v} | ... | get<Is>(static_cast<T&&>(t))).val_;
}

} // end namespace detail

template <class T, class V, class F>
CUTE_HOST_DEVICE constexpr
auto
fold(T&& t, V const& v, F&& f)
{
  if constexpr (is_tuple<remove_cvref_t<T>>::value) {
    return detail::fold(static_cast<T&&>(t), v, f, tuple_seq<T>{});
  } else {
    return f(v, static_cast<T&&>(t));
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class F>
CUTE_HOST_DEVICE constexpr
auto
fold_first(T&& t, F&& f)
{
  if constexpr (is_tuple<remove_cvref_t<T>>::value) {
    return detail::fold(static_cast<T&&>(t), get<0>(t), f, make_range<1,tuple_size<remove_cvref_t<T>>::value>{});
  } else {
    return t;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// front, back, take, select, unwrap
//

// Get the first non-tuple element in a hierarchical tuple
template <class T>
CUTE_HOST_DEVICE constexpr
decltype(auto)
front(T&& t)
{
  if constexpr (is_tuple<remove_cvref_t<T>>::value) {
    return front(get<0>(static_cast<T&&>(t)));
  } else {
    return static_cast<T&&>(t);
  }

  CUTE_GCC_UNREACHABLE;
}

// Get the last non-tuple element in a hierarchical tuple
template <class T>
CUTE_HOST_DEVICE constexpr
decltype(auto)
back(T&& t)
{
  if constexpr (is_tuple<remove_cvref_t<T>>::value) {
    constexpr int N = tuple_size<remove_cvref_t<T>>::value;

    // MSVC needs a bit of extra help here deducing return types.
    // We help it by peeling off the nonrecursive case a level "early."
    if constexpr (! is_tuple<remove_cvref_t<decltype(get<N - 1>(static_cast<T&&>(t)))>>::value) {
      return get<N - 1>(static_cast<T&&>(t));
    } else {
      return back(get<N - 1>(static_cast<T&&>(t)));
    }
  } else {
    return static_cast<T&&>(t);
  }

  CUTE_GCC_UNREACHABLE;
}

// Takes the elements in the range [B,E)
template <int B, int E, class T>
CUTE_HOST_DEVICE constexpr
auto
take(T const& t)
{
  if constexpr (E == -1) {
    if constexpr (is_tuple<T>::value) {
      return take<B,tuple_size<T>::value>(t);
    } else {
      return take<B,1>(t);
    }
  } else
  if constexpr (B <= E) {
    return detail::apply(t, [](auto const&... a) { return cute::make_tuple(a...); }, make_range<B,E>{});
  } else {
    static_assert(B <= E);
  }

  CUTE_GCC_UNREACHABLE;
}

// Select tuple elements with given indices.
template <int... I, class T>
CUTE_HOST_DEVICE constexpr
auto
select(T const& t)
{
  return cute::make_tuple(get<I>(t)...);
}

// Wrap non-tuples into rank-1 tuples or forward
template <class T>
CUTE_HOST_DEVICE constexpr
auto
wrap(T const& t)
{
  if constexpr (is_tuple<T>::value) {
    return t;
  } else {
    return cute::make_tuple(t);
  }

  CUTE_GCC_UNREACHABLE;
}

// Unwrap rank-1 tuples until we're left with a rank>1 tuple or a non-tuple
template <class T>
CUTE_HOST_DEVICE constexpr
auto
unwrap(T const& t)
{
  if constexpr (is_tuple<T>::value) {
    if constexpr (tuple_size<T>::value == 1) {
      return unwrap(get<0>(t));
    } else {
      return t;
    }
  } else {
    return t;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Flatten and Unflatten
//

template <class T>
struct is_flat : true_type {};

template <class... Ts>
struct is_flat<tuple<Ts...>> : bool_constant<(true && ... && (not is_tuple<Ts>::value))> {};

// Flatten a hierarchical tuple to a tuple of depth one
//   and wrap non-tuples into a rank-1 tuple.
template <class T>
CUTE_HOST_DEVICE constexpr
auto
flatten_to_tuple(T const& t)
{
  if constexpr (is_tuple<T>::value) {
    if constexpr (is_flat<T>::value) {      // Shortcut for perf
      return t;
    } else {
      return filter_tuple(t, [](auto const& a) { return flatten_to_tuple(a); });
    }
  } else {
    return cute::make_tuple(t);
  }

  CUTE_GCC_UNREACHABLE;
}

// Flatten a hierarchical tuple to a tuple of depth one
//   and leave non-tuple untouched.
template <class T>
CUTE_HOST_DEVICE constexpr
auto
flatten(T const& t)
{
  if constexpr (is_tuple<T>::value) {
    if constexpr (is_flat<T>::value) {      // Shortcut for perf
      return t;
    } else {
      return filter_tuple(t, [](auto const& a) { return flatten_to_tuple(a); });
    }
  } else {
    return t;
  }

  CUTE_GCC_UNREACHABLE;
}

namespace detail {

template <class FlatTuple, class TargetProfile>
CUTE_HOST_DEVICE constexpr
auto
unflatten_impl(FlatTuple const& flat_tuple, TargetProfile const& target_profile)
{
  if constexpr (is_tuple<TargetProfile>::value) {
    return fold(target_profile, cute::make_tuple(cute::make_tuple(), flat_tuple), [](auto const& v, auto const& t) {
      auto [result, remaining_tuple] = v;
      auto [sub_result, sub_tuple] = unflatten_impl(remaining_tuple, t);
      return cute::make_tuple(append(result, sub_result), sub_tuple);
    });
  } else {
    return cute::make_tuple(get<0>(flat_tuple), take<1, decltype(rank(flat_tuple))::value>(flat_tuple));
  }

  CUTE_GCC_UNREACHABLE;
}

}  // end namespace detail

// Unflatten a flat tuple into a hierarchical tuple
// @pre flatten(@a flat_tuple) == @a flat_tuple
// @pre rank(flatten(@a target_profile)) == rank(@a flat_tuple)
// @post congruent(@a result, @a target_profile)
// @post flatten(@a result) == @a flat_tuple
template <class FlatTuple, class TargetProfile>
CUTE_HOST_DEVICE constexpr
auto
unflatten(FlatTuple const& flat_tuple, TargetProfile const& target_profile)
{
  auto [unflatten_tuple, flat_remainder] = detail::unflatten_impl(flat_tuple, target_profile);
  CUTE_STATIC_ASSERT_V(rank(flat_remainder) == Int<0>{});
  return unflatten_tuple;
}

//
// insert and remove and replace
//

namespace detail {

// Shortcut around cute::tuple_cat for common insert/remove/repeat cases
template <class T, class X, int... I, int... J, int... K>
CUTE_HOST_DEVICE constexpr
auto
construct(T const& t, X const& x, seq<I...>, seq<J...>, seq<K...>)
{
  return cute::make_tuple(get<I>(t)..., (void(J),x)..., get<K>(t)...);
}

} // end namespace detail

// Insert x into the Nth position of the tuple
template <int N, class T, class X>
CUTE_HOST_DEVICE constexpr
auto
insert(T const& t, X const& x)
{
  return detail::construct(t, x, make_seq<N>{}, seq<0>{}, make_range<N,tuple_size<T>::value>{});
}

// Remove the Nth element of the tuple
template <int N, class T>
CUTE_HOST_DEVICE constexpr
auto
remove(T const& t)
{
  return detail::construct(t, 0, make_seq<N>{}, seq<>{}, make_range<N+1,tuple_size<T>::value>{});
}

// Replace the Nth element of the tuple with x
template <int N, class T, class X>
CUTE_HOST_DEVICE constexpr
auto
replace(T const& t, X const& x)
{
  if constexpr (is_tuple<T>::value) {
    return detail::construct(t, x, make_seq<N>{}, seq<0>{}, make_range<N+1,tuple_size<T>::value>{});
  } else {
    static_assert(N == 0);
    return x;
  }

  CUTE_GCC_UNREACHABLE;
}

// Replace the first element of the tuple with x
template <class T, class X>
CUTE_HOST_DEVICE constexpr
auto
replace_front(T const& t, X const& x)
{
  if constexpr (is_tuple<T>::value) {
    return detail::construct(t, x, seq<>{}, seq<0>{}, make_range<1,tuple_size<T>::value>{});
  } else {
    return x;
  }

  CUTE_GCC_UNREACHABLE;
}

// Replace the last element of the tuple with x
template <class T, class X>
CUTE_HOST_DEVICE constexpr
auto
replace_back(T const& t, X const& x)
{
  if constexpr (is_tuple<T>::value) {
    return detail::construct(t, x, make_seq<tuple_size<T>::value-1>{}, seq<0>{}, seq<>{});
  } else {
    return x;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Make a tuple of Xs of tuple_size N
//

template <int N, class X>
CUTE_HOST_DEVICE constexpr
auto
tuple_repeat(X const& x)
{
  return detail::construct(0, x, seq<>{}, make_seq<N>{}, seq<>{});
}

//
// Make repeated Xs of rank N
//

template <int N, class X>
CUTE_HOST_DEVICE constexpr
auto
repeat(X const& x)
{
  if constexpr (N == 1) {
    return x;
  } else {
    return detail::construct(0, x, seq<>{}, make_seq<N>{}, seq<>{});
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Make a tuple of Xs the same profile as tuple T
//

template <class T, class X>
CUTE_HOST_DEVICE constexpr
auto
repeat_like(T const& t, X const& x)
{
  if constexpr (is_tuple<T>::value) {
    return transform(t, [&](auto const& a) { return repeat_like(a,x); });
  } else {
    return x;
  }

  CUTE_GCC_UNREACHABLE;
}

// Group the elements [B,E) of a T into a single element
// e.g. group<2,4>(T<_1,_2,_3,_4,_5,_6>{})
//              => T<_1,_2,T<_3,_4>,_5,_6>{}
template <int B, int E, class T>
CUTE_HOST_DEVICE constexpr
auto
group(T const& t)
{
  if constexpr (not is_tuple<T>::value) {
    if constexpr (E == -1) {
      return group<B,1>(t);
    } else {
      return detail::construct(t, take<B,E>(t), make_seq<B>{}, make_seq<(B < E)>{}, make_range<E,1>{});
    }
  } else
  if constexpr (E == -1) {
    return group<B,tuple_size<T>::value>(t);
  } else
  if constexpr (B <= E) {
    return detail::construct(t, take<B,E>(t), make_seq<B>{}, make_seq<(B < E)>{}, make_range<E,tuple_size<T>::value>{});
  } else {
    static_assert(B <= E);
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Extend a T to rank N by appending/prepending an element
//

template <int N, class T, class X>
CUTE_HOST_DEVICE constexpr
auto
append(T const& a, X const& x)
{
  if constexpr (is_tuple<T>::value) {
    if constexpr (N == tuple_size<T>::value) {
      return a;
    } else {
      static_assert(N > tuple_size<T>::value);
      return detail::construct(a, x, make_seq<tuple_size<T>::value>{}, make_seq<N-tuple_size<T>::value>{}, seq<>{});
    }
  } else {
    if constexpr (N == 1) {
      return a;
    } else {
      return detail::construct(cute::make_tuple(a), x, seq<0>{}, make_seq<N-1>{}, seq<>{});
    }
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class X>
CUTE_HOST_DEVICE constexpr
auto
append(T const& a, X const& x)
{
  if constexpr (is_tuple<T>::value) {
    return detail::construct(a, x, make_seq<tuple_size<T>::value>{}, seq<0>{}, seq<>{});
  } else {
    return cute::make_tuple(a, x);
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class T, class X>
CUTE_HOST_DEVICE constexpr
auto
prepend(T const& a, X const& x)
{
  if constexpr (is_tuple<T>::value) {
    if constexpr (N == tuple_size<T>::value) {
      return a;
    } else {
      static_assert(N > tuple_size<T>::value);
      return detail::construct(a, x, seq<>{}, make_seq<N-tuple_size<T>::value>{}, make_seq<tuple_size<T>::value>{});
    }
  } else {
    if constexpr (N == 1) {
      return a;
    } else {
      static_assert(N > 1);
      return detail::construct(cute::make_tuple(a), x, seq<>{}, make_seq<N-1>{}, seq<0>{});
    }
  }

  CUTE_GCC_UNREACHABLE;
}

template <class T, class X>
CUTE_HOST_DEVICE constexpr
auto
prepend(T const& a, X const& x)
{
  if constexpr (is_tuple<T>::value) {
    return detail::construct(a, x, seq<>{}, seq<0>{}, make_seq<tuple_size<T>::value>{});
  } else {
    return cute::make_tuple(x, a);
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Inclusive scan (prefix sum)
//

namespace detail {

template <class T, class V, class F, int I, int... Is>
CUTE_HOST_DEVICE constexpr
auto
iscan(T const& t, V const& v, F&& f, seq<I,Is...>)
{
  // Apply the function to v and the element at I
  auto v_next = f(v, get<I>(t));
  // Replace I with v_next
  auto t_next = replace<I>(t, v_next);

#if 0
  std::cout << "ISCAN i" << I << std::endl;
  std::cout << "  t      " << t << std::endl;
  std::cout << "  i      " << v << std::endl;
  std::cout << "  f(i,t) " << v_next << std::endl;
  std::cout << "  t_n    " << t_next << std::endl;
#endif

  if constexpr (sizeof...(Is) == 0) {
    return t_next;
  } else {
    return iscan(t_next, v_next, f, seq<Is...>{});
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

template <class T, class V, class F>
CUTE_HOST_DEVICE constexpr
auto
iscan(T const& t, V const& v, F&& f)
{
  return detail::iscan(t, v, f, tuple_seq<T>{});
}

//
// Exclusive scan (prefix sum)
//

namespace detail {

template <class T, class V, class F, int I, int... Is>
CUTE_HOST_DEVICE constexpr
auto
escan(T const& t, V const& v, F&& f, seq<I,Is...>)
{
  if constexpr (sizeof...(Is) == 0) {
    // Replace I with v
    return replace<I>(t, v);
  } else {
    // Apply the function to v and the element at I
    auto v_next = f(v, get<I>(t));
    // Replace I with v
    auto t_next = replace<I>(t, v);

#if 0
    std::cout << "ESCAN i" << I << std::endl;
    std::cout << "  t      " << t << std::endl;
    std::cout << "  i      " << v << std::endl;
    std::cout << "  f(i,t) " << v_next << std::endl;
    std::cout << "  t_n    " << t_next << std::endl;
#endif

    // Recurse
    return escan(t_next, v_next, f, seq<Is...>{});
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

template <class T, class V, class F>
CUTE_HOST_DEVICE constexpr
auto
escan(T const& t, V const& v, F&& f)
{
  return detail::escan(t, v, f, tuple_seq<T>{});
}

//
// Zip (Transpose)
//

// Take       ((a,b,c,...),(x,y,z,...),...)        rank-R0 x rank-R1 input
// to produce ((a,x,...),(b,y,...),(c,z,...),...)  rank-R1 x rank-R0 output

namespace detail {

template <int J, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
zip_(Ts const&... ts)
{
  return cute::make_tuple(get<J>(ts)...);
}

template <class T, int... Is, int... Js>
CUTE_HOST_DEVICE constexpr
auto
zip(T const& t, seq<Is...>, seq<Js...>)
{
  static_assert(conjunction<bool_constant<tuple_size<tuple_element_t<0,T>>::value == tuple_size<tuple_element_t<Is,T>>::value>...>::value, "Mismatched Ranks");
  return cute::make_tuple(zip_<Js>(get<Is>(t)...)...);
}

} // end namespace detail

template <class T>
CUTE_HOST_DEVICE constexpr
auto
zip(T const& t)
{
  if constexpr (is_tuple<T>::value) {
    if constexpr (is_tuple<tuple_element_t<0,T>>::value) {
      return detail::zip(t, tuple_seq<T>{}, tuple_seq<tuple_element_t<0,T>>{});
    } else {
      return cute::make_tuple(t);
    }
  } else {
    return t;
  }

  CUTE_GCC_UNREACHABLE;
}

// Convenient to pass them in separately
template <class T0, class T1, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
zip(T0 const& t0, T1 const& t1, Ts const&... ts)
{
  return zip(cute::make_tuple(t0, t1, ts...));
}

//
// zip2_by -- A guided zip for rank-2 tuples
//   Take a tuple like ((A,a),((B,b),(C,c)),d)
//   and produce a tuple ((A,(B,C)),(a,(b,c),d))
//   where the rank-2 modes are selected by the terminals of the guide (X,(X,X))
//

namespace detail {

template <class T, class TG, int... Is, int... Js>
CUTE_HOST_DEVICE constexpr
auto
zip2_by(T const& t, TG const& guide, seq<Is...>, seq<Js...>)
{
  // zip2_by produces the modes like ((A,a),(B,b),...)
  auto split = cute::make_tuple(zip2_by(get<Is>(t), get<Is>(guide))...);

  // Rearrange and append missing modes from t to make ((A,B,...),(a,b,...,x,y))
  return cute::make_tuple(cute::make_tuple(get<0>(get<Is>(split))...),
                          cute::make_tuple(get<1>(get<Is>(split))..., get<Js>(t)...));
}

} // end namespace detail

template <class T, class TG>
CUTE_HOST_DEVICE constexpr
auto
zip2_by(T const& t, TG const& guide)
{
  if constexpr (is_tuple<TG>::value) {
    constexpr int TR = tuple_size<T>::value;
    constexpr int GR = tuple_size<TG>::value;
    static_assert(TR >= GR, "Mismatched ranks");
    return detail::zip2_by(t, guide,
                           make_range< 0, GR>{},
                           make_range<GR, TR>{});
  } else {
    static_assert(tuple_size<T>::value == 2, "Mismatched ranks");
    return t;
  }

  CUTE_GCC_UNREACHABLE;
}

/// @return A tuple of the elements of @c t in reverse order.
template <class T>
CUTE_HOST_DEVICE constexpr
auto
reverse(T const& t)
{
  if constexpr (is_tuple<T>::value) {
    return detail::apply(t, [](auto const&... a){ return cute::make_tuple(a...); }, tuple_rseq<T>{});
  } else {
    return t;
  }
}

} // end namespace cute
