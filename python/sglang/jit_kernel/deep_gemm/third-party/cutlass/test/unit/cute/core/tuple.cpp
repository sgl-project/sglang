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

#include "cutlass_unit_test.h"

#include <cutlass/trace.h>

#include <cassert>
#include <cstdint>

#include <tuple>
#include <cute/container/tuple.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/tensor.hpp>

TEST(CuTe_core, Tuple)
{
  using namespace cute;

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("SIMPLE STATIC AND DYNAMIC TUPLES");
  CUTLASS_TRACE_HOST("-------------------------------");

  using tuple_2d_s_type = tuple<_8, _4>;                            // (8,4)
  using tuple_3d_s_type = tuple<_8, _4, _2>;                        // (8,4,2)
  using tuple_3h_s_type = tuple<tuple<_1, _2>, _8, _2>;             // ((1,2),8,2)

  using tuple_2d_d_type = tuple<int, int>;                          // (8,4)
  using tuple_3d_d_type = tuple<int, int, int>;                     // (8,4,2)
  using tuple_3h_d_type = tuple<tuple<int, int>, int, int>;         // ((1,2),8,2)

  using tuple_2d_m_type = tuple<_8, int>;                           // (8,4)
  using tuple_3d_m_type = tuple<int, int, _2>;                      // (8,4,2)
  using tuple_3h_m_type = tuple<tuple<int, _2>, int, int>;          // ((1,2),8,2)

  tuple_2d_s_type tuple_2d_s;
  tuple_3d_s_type tuple_3d_s;
  tuple_3h_s_type tuple_3h_s;

  tuple_2d_d_type tuple_2d_d(8,4);
  tuple_3d_d_type tuple_3d_d(8,4,2);
  tuple_3h_d_type tuple_3h_d(tuple<int,int>(1,2),8,2);

  tuple_2d_m_type tuple_2d_m(_8{}, 4);
  tuple_3d_m_type tuple_3d_m(8,4,_2{});
  tuple_3h_m_type tuple_3h_m(tuple<int,_2>(1,_2{}),8,2);

  CUTLASS_TRACE_HOST(tuple_2d_s << (is_static<tuple_2d_s_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_2d_s_type));
  ASSERT_TRUE(is_static<tuple_2d_s_type>::value == true);
  ASSERT_TRUE(sizeof(tuple_2d_s_type) == 1);
  ASSERT_TRUE(std::is_empty<tuple_2d_s_type>::value);

  CUTLASS_TRACE_HOST(tuple_3d_s << (is_static<tuple_3d_s_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3d_s_type));
  ASSERT_TRUE(is_static<tuple_3d_s_type>::value == true);
  ASSERT_TRUE(sizeof(tuple_3d_s_type) == 1);
  ASSERT_TRUE(std::is_empty<tuple_3d_s_type>::value);

  CUTLASS_TRACE_HOST(tuple_3h_s << (is_static<tuple_3h_s_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3h_s_type));
  ASSERT_TRUE(is_static<tuple_3h_s_type>::value == true);
  ASSERT_TRUE(sizeof(tuple_3h_s_type) == 1);
  ASSERT_TRUE(std::is_empty<tuple_3h_s_type>::value);

  CUTLASS_TRACE_HOST(tuple_2d_d << (is_static<tuple_2d_d_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_2d_d_type));
  ASSERT_TRUE(is_static<tuple_2d_d_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_2d_d_type) == 8);
  ASSERT_TRUE(!std::is_empty<tuple_2d_d_type>::value);

  CUTLASS_TRACE_HOST(tuple_3d_d << (is_static<tuple_3d_d_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3d_d_type));
  ASSERT_TRUE(is_static<tuple_3d_d_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_3d_d_type) == 12);
  ASSERT_TRUE(!std::is_empty<tuple_3d_d_type>::value);

  CUTLASS_TRACE_HOST(tuple_3h_d << (is_static<tuple_3h_d_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3h_d_type));
  ASSERT_TRUE(is_static<tuple_3h_d_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_3h_d_type) == 16);
  ASSERT_TRUE(!std::is_empty<tuple_3h_d_type>::value);

  CUTLASS_TRACE_HOST(tuple_2d_m << (is_static<tuple_2d_m_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_2d_m_type));
  ASSERT_TRUE(is_static<tuple_2d_m_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_2d_m_type) == 4);
  ASSERT_TRUE(!std::is_empty<tuple_2d_m_type>::value);

  CUTLASS_TRACE_HOST(tuple_3d_m << (is_static<tuple_3d_m_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3d_m_type));
  ASSERT_TRUE(is_static<tuple_3d_m_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_3d_m_type) == 8);
  ASSERT_TRUE(!std::is_empty<tuple_3d_m_type>::value);

  CUTLASS_TRACE_HOST(tuple_3h_m << (is_static<tuple_3h_m_type>::value ? "  Static  " : "  Dynamic  ")
            << "sizeof = " << sizeof(tuple_3h_m_type));
  ASSERT_TRUE(is_static<tuple_3h_m_type>::value == false);
  ASSERT_TRUE(sizeof(tuple_3h_m_type) == 12);
  ASSERT_TRUE(!std::is_empty<tuple_3h_m_type>::value);

  ASSERT_TRUE(sizeof(cute::tuple<_1, _1, cute::tuple<int32_t>>) == 4);
  ASSERT_TRUE(sizeof(cute::tuple<_1, _0, cute::tuple<int32_t>>) == 4);
  ASSERT_TRUE(sizeof(cute::tuple<_1, cute::tuple<_1, int32_t>>) == 4);
  ASSERT_TRUE(sizeof(cute::tuple<_1, cute::tuple<_0, int32_t>>) == 4);

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("SIMPLE TUPLE OPS");
  CUTLASS_TRACE_HOST("-------------------------------");

  CUTLASS_TRACE_HOST("product(" << tuple_2d_s << ") => " << product(tuple_2d_s));
  CUTE_STATIC_ASSERT_V(product(tuple_2d_s) == _32{});
  CUTLASS_TRACE_HOST("product(" << tuple_3d_s << ") => " << product(tuple_3d_s));
  CUTE_STATIC_ASSERT_V(product(tuple_3d_s) == _64{});
  CUTLASS_TRACE_HOST("product(" << tuple_3h_s << ") => " << product(tuple_3h_s));
  CUTE_STATIC_ASSERT_V(product(tuple_3h_s) == _32{});

  CUTLASS_TRACE_HOST("product(" << tuple_2d_d << ") => " << product(tuple_2d_d));
  ASSERT_TRUE(product(tuple_2d_d) == 32);
  CUTLASS_TRACE_HOST("product(" << tuple_3d_d << ") => " << product(tuple_3d_d));
  ASSERT_TRUE(product(tuple_3d_d) == 64);
  CUTLASS_TRACE_HOST("product(" << tuple_3h_d << ") => " << product(tuple_3h_d));
  ASSERT_TRUE(product(tuple_3h_d) == 32);

  CUTLASS_TRACE_HOST("product(" << tuple_2d_m << ") => " << product(tuple_2d_m));
  ASSERT_TRUE(product(tuple_2d_m) == 32);
  CUTLASS_TRACE_HOST("product(" << tuple_3d_m << ") => " << product(tuple_3d_m));
  ASSERT_TRUE(product(tuple_3d_m) == 64);
  CUTLASS_TRACE_HOST("product(" << tuple_3h_m << ") => " << product(tuple_3h_m));
  ASSERT_TRUE(product(tuple_3h_m) == 32);

  CUTLASS_TRACE_HOST("max(" << tuple_2d_s << ") => " << max(tuple_2d_s));
  CUTE_STATIC_ASSERT_V(max(tuple_2d_s) == _8{});
  CUTLASS_TRACE_HOST("max(" << tuple_3d_s << ") => " << max(tuple_3d_s));
  CUTE_STATIC_ASSERT_V(max(tuple_3d_s) == _8{});
  CUTLASS_TRACE_HOST("max(" << tuple_3h_s << ") => " << max(tuple_3h_s));
  CUTE_STATIC_ASSERT_V(max(tuple_3h_s) == _8{});

  CUTLASS_TRACE_HOST("max(" << tuple_2d_d << ") => " << max(tuple_2d_d));
  ASSERT_TRUE(max(tuple_2d_d) == 8);
  CUTLASS_TRACE_HOST("max(" << tuple_3d_d << ") => " << max(tuple_3d_d));
  ASSERT_TRUE(max(tuple_3d_d) == 8);
  CUTLASS_TRACE_HOST("max(" << tuple_3h_d << ") => " << max(tuple_3h_d));
  ASSERT_TRUE(max(tuple_3h_d) == 8);

  CUTLASS_TRACE_HOST("max(" << tuple_2d_m << ") => " << max(tuple_2d_m));
  ASSERT_TRUE(max(tuple_2d_m) == 8);
  CUTLASS_TRACE_HOST("max(" << tuple_3d_m << ") => " << max(tuple_3d_m));
  ASSERT_TRUE(max(tuple_3d_m) == 8);
  CUTLASS_TRACE_HOST("max(" << tuple_3h_m << ") => " << max(tuple_3h_m));
  ASSERT_TRUE(max(tuple_3h_m) == 8);

  // 2d s|d|m
  CUTLASS_TRACE_HOST("inner_product(" << tuple_2d_s << ", " << tuple_2d_s << ") => "
            << inner_product(tuple_2d_s, tuple_2d_s));
  CUTE_STATIC_ASSERT_V(inner_product(tuple_2d_s, tuple_2d_s) == Int<80>{});
  CUTLASS_TRACE_HOST("inner_product(" << tuple_2d_d << ", " << tuple_2d_d << ") => "
            << inner_product(tuple_2d_d, tuple_2d_d));
  ASSERT_TRUE(inner_product(tuple_2d_d, tuple_2d_d) == 80);
  CUTLASS_TRACE_HOST("inner_product(" << tuple_2d_m << ", " << tuple_2d_m << ") => "
            << inner_product(tuple_2d_m, tuple_2d_m));
  ASSERT_TRUE(inner_product(tuple_2d_m, tuple_2d_m) == 80);

  // 3d s|d|m
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3d_s << ", " << tuple_3d_s << ") => "
            << inner_product(tuple_3d_s, tuple_3d_s));
  CUTE_STATIC_ASSERT_V(inner_product(tuple_3d_s, tuple_3d_s) == Int<84>{});
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3d_d << ", " << tuple_3d_d << ") => "
            << inner_product(tuple_3d_d, tuple_3d_d));
  ASSERT_TRUE(inner_product(tuple_3d_d, tuple_3d_d) == 84);
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3d_m << ", " << tuple_3d_m << ") => "
            << inner_product(tuple_3d_m, tuple_3d_m));
  ASSERT_TRUE(inner_product(tuple_3d_m, tuple_3d_m) == 84);

  // 3h s|d|m
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3h_s << ", " << tuple_3h_s << ") => "
            << inner_product(tuple_3h_s, tuple_3h_s));
  CUTE_STATIC_ASSERT_V(inner_product(tuple_3h_s, tuple_3h_s) == Int<73>{});
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3h_d << ", " << tuple_3h_d << ") => "
            << inner_product(tuple_3h_d, tuple_3h_d));
  ASSERT_TRUE(inner_product(tuple_3h_d, tuple_3h_d) == 73);
  CUTLASS_TRACE_HOST("inner_product(" << tuple_3h_m << ", " << tuple_3h_m << ") => "
            << inner_product(tuple_3h_m, tuple_3h_m));
  ASSERT_TRUE(inner_product(tuple_3h_m, tuple_3h_m) == 73);

  CUTLASS_TRACE_HOST("col_major(" << tuple_2d_s << ") => " << compact_col_major(tuple_2d_s));
  CUTE_STATIC_ASSERT_V((compact_col_major(tuple_2d_s) == make_tuple(_1{},_8{})));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3d_s << ") => " << compact_col_major(tuple_3d_s));
  CUTE_STATIC_ASSERT_V((compact_col_major(tuple_3d_s) == make_tuple(_1{},_8{},_32{})));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3h_s << ") => " << compact_col_major(tuple_3h_s));
  CUTE_STATIC_ASSERT_V((compact_col_major(tuple_3h_s) == make_tuple(make_tuple(_0{},_1{}),_2{},_16{})));

  CUTLASS_TRACE_HOST("col_major(" << tuple_2d_d << ") => " << compact_col_major(tuple_2d_d));
  ASSERT_TRUE((compact_col_major(tuple_2d_d) == make_tuple(_1{},8)));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3d_d << ") => " << compact_col_major(tuple_3d_d));
  ASSERT_TRUE((compact_col_major(tuple_3d_d) == make_tuple(_1{},8,32)));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3h_d << ") => " << compact_col_major(tuple_3h_d));
  ASSERT_TRUE((compact_col_major(tuple_3h_d) == make_tuple(make_tuple(_1{},1),2,16)));

  CUTLASS_TRACE_HOST("col_major(" << tuple_2d_m << ") => " << compact_col_major(tuple_2d_m));
  ASSERT_TRUE((compact_col_major(tuple_2d_m) == make_tuple(_1{},_8{})));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3d_m << ") => " << compact_col_major(tuple_3d_m));
  ASSERT_TRUE((compact_col_major(tuple_3d_m) == make_tuple(_1{},8,32)));
  CUTLASS_TRACE_HOST("col_major(" << tuple_3h_m << ") => " << compact_col_major(tuple_3h_m));
  ASSERT_TRUE((compact_col_major(tuple_3h_m) == make_tuple(make_tuple(_1{},1),2,16)));

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("SLICING TUPLES");
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto a = Coord<_2,_3,_4,Coord<_5,_6>>{};

    CUTLASS_TRACE_HOST("a = " << a);

    CUTLASS_TRACE_HOST("a(1) = " << slice(1, a));

    CUTLASS_TRACE_HOST("a(_) = " << slice(_, a));

    CUTLASS_TRACE_HOST("a(_,1,_,_) = " << slice(make_coord(_,1,_,_), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(_,_)) = " << slice(make_coord(_,1,_,make_coord(_,_)), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(_,2)) = " << slice(make_coord(_,1,_,make_coord(_,2)), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(1,2)) = " << slice(make_coord(_,1,_,make_coord(1,2)), a));
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("DICING TUPLES");
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto a = Coord<_2,_3,_4,Coord<_5,_6>>{};

    CUTLASS_TRACE_HOST("a = " << a);

    CUTLASS_TRACE_HOST("a(1) = " << dice(1, a));

    CUTLASS_TRACE_HOST("a(_) = " << dice(_, a));

    CUTLASS_TRACE_HOST("a(_,1,_,_) = " << dice(make_coord(_,1,_,_), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(_,_)) = " << dice(make_coord(_,1,_,make_coord(_,_)), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(_,2)) = " << dice(make_coord(_,1,_,make_coord(_,2)), a));

    CUTLASS_TRACE_HOST("a(_,1,_,(1,2)) = " << dice(make_coord(_,1,_,make_coord(1,2)), a));
  }
}

namespace pt_test {

template <class T>
struct Nonempty {
  T datum;

  Nonempty(T const& t) : datum{t} {}

  friend bool operator==(Nonempty<T> const& lhs, Nonempty<T> const& rhs) {
    return lhs.datum == rhs.datum;
  }

  friend bool operator!=(Nonempty<T> const& lhs, Nonempty<T> const& rhs) {
    return !(lhs == rhs);
  }
};

template <int V>
struct Empty {
  template <int W>
  friend bool operator==(Empty<V> const&, Empty<W> const&) {
    return V == W;
  }

  template <int W>
  friend bool operator!=(Empty<V> const& lhs, Empty<W> const& rhs) {
    return !(lhs == rhs);
  }
};

// std::tuple
static_assert(cute::is_standard_layout_v<std::tuple<>>); // it happens to be
static_assert(cute::is_standard_layout_v<std::tuple<int>>); // it happens to be
static_assert(cute::is_standard_layout_v<std::tuple<double>>); // it happens to be
static_assert(not cute::is_standard_layout_v<std::tuple<int, double>>); // it's not

// cute::tuple
static_assert(cute::is_standard_layout_v<cute::tuple<>>);
static_assert(cute::is_standard_layout_v<cute::tuple<int>>);
static_assert(cute::is_standard_layout_v<cute::tuple<double>>);
static_assert(cute::is_standard_layout_v<cute::tuple<int, double>>);  // it is
static_assert(cute::is_standard_layout_v<cute::tuple<int, int, int, int>>);  // it is
static_assert(cute::is_standard_layout_v<cute::tuple<int, cute::tuple<int, int>, int>>);  // it is
static_assert(cute::is_standard_layout_v<cute::tuple<int, cute::tuple<Empty<0>, Empty<0>>, int>>);  // it is

//////////////////////////////////////////////////////////////////////
// tuple test starts here
//////////////////////////////////////////////////////////////////////

template <
  class ExpectedPackedType,
  size_t ExpectedPackedSize,
  class ... Args>
constexpr void
test_packed_type_alias([[maybe_unused]] ExpectedPackedType packed, std::tuple<Args...> unpacked)
{
  using cute::tuple;

  if constexpr ((cute::is_standard_layout_v<Args> && ...)) {
    static_assert(cute::is_standard_layout_v<tuple<Args...>>);
  }

  if constexpr ((cute::is_empty_v<Args> && ...)) {
    static_assert(cute::is_empty_v<tuple<Args...>>);
  }

  static_assert(cute::tuple_size_v<tuple<Args...>> == sizeof...(Args));

  auto test_element = [unpacked] (auto index) {
    static_assert(cute::is_same_v<
      std::tuple_element_t<index, tuple<Args...>>,
      std::tuple_element_t<index, std::tuple<Args...>>
    >);

    tuple<Args...> sl = cute::apply(unpacked, [](auto... a){ return cute::make_tuple(a...); });
    EXPECT_EQ(std::get<index>(unpacked), cute::get<index>(sl));
  };
  cute::for_each(std::make_index_sequence<sizeof...(Args)>(), test_element);
}

void test_packed_type_aliases() {
  using cute::tuple;
  test_packed_type_alias<tuple<>, 0>({}, {});

  test_packed_type_alias<tuple<int>, 1, int>({7}, {7});
  test_packed_type_alias<tuple<double>, 1, double>({1.5}, {1.5});

  // Make sure that class types are handled the same as scalar types
  test_packed_type_alias<tuple<Nonempty<int>>, 1, Nonempty<int>>(
    {Nonempty{7}}, {Nonempty{7}});
  test_packed_type_alias<tuple<Nonempty<double>>, 1, Nonempty<double>>(
    {Nonempty{1.5}}, {Nonempty{1.5}});

  test_packed_type_alias<tuple<>, 0, Empty<0>>({}, {});
  test_packed_type_alias<tuple<>, 0, Empty<0>, Empty<1>>(
    {}, {Empty<0>{}, Empty<1>{}});
  test_packed_type_alias<tuple<>, 0, Empty<0>, Empty<1>, Empty<2>>(
    {}, {Empty<0>{}, Empty<1>{}, Empty<2>{}});

  test_packed_type_alias<tuple<int>, 1, Empty<0>, int>(
    {7}, {Empty<0>{}, 7});
  test_packed_type_alias<tuple<int>, 1, int, Empty<0>>(
    {7}, {7, Empty<0>{}});

  test_packed_type_alias<tuple<int>, 1, int, Empty<0>, Empty<1>>(
    {7}, {7, Empty<0>{}, Empty<1>{}});
  test_packed_type_alias<tuple<int>, 1, Empty<0>, int, Empty<1>>(
    {7}, {Empty<0>{}, 7, Empty<1>{}});
  test_packed_type_alias<tuple<int>, 1, Empty<0>, Empty<1>, int>(
    {7}, {Empty<0>{}, Empty<1>{}, 7});

  test_packed_type_alias<tuple<int, double>, 2, int, double, Empty<0>>(
    {7, 1.5}, {7, 1.5, Empty<0>{}});
  test_packed_type_alias<tuple<int, double>, 2, int, Empty<0>, double>(
    {7, 1.5}, {7, Empty<0>{}, 1.5});
  test_packed_type_alias<tuple<int, double>, 2, int, double, Empty<0>>(
    {7, 1.5}, {7, 1.5, Empty<0>{}});

  test_packed_type_alias<tuple<int, double>, 2, int, double, Empty<0>, Empty<1>>(
    {7, 1.5}, {7, 1.5, Empty<0>{}, Empty<1>{}});
  test_packed_type_alias<tuple<int, double>, 2, int, Empty<0>, double, Empty<1>>(
    {7, 1.5}, {7, Empty<0>{}, 1.5, Empty<1>{}});
  test_packed_type_alias<tuple<int, double>, 2, int, Empty<0>, Empty<1>, double>(
    {7, 1.5}, {7, Empty<0>{}, Empty<1>{}, 1.5});
  test_packed_type_alias<tuple<int, double>, 2, Empty<0>, int, Empty<1>, double>(
    {7, 1.5}, {Empty<0>{}, 7, Empty<1>{}, 1.5});
  test_packed_type_alias<tuple<int, double>, 2, Empty<0>, Empty<1>, int, double>(
    {7, 1.5}, {Empty<0>{}, Empty<1>{}, 7, 1.5});

  test_packed_type_alias<tuple<int, double, float>, 3, Empty<0>, int, double, float>(
    {7, 1.5, 2.5f}, {Empty<0>{}, 7, 1.5, 2.5f});
  test_packed_type_alias<tuple<int, double, float>, 3, int, Empty<0>, double, float>(
    {7, 1.5, 2.5f}, {7, Empty<0>{}, 1.5, 2.5f});
  test_packed_type_alias<tuple<int, double, float>, 3, int, double, Empty<0>, float>(
    {7, 1.5, 2.5f}, {7, 1.5, Empty<0>{}, 2.5f});
  test_packed_type_alias<tuple<int, double, float>, 3, int, double, float, Empty<0>>(
    {7, 1.5, 2.5f}, {7, 1.5, 2.5f, Empty<0>{}});
}

template <class Tuple, size_t Which, class ExpectedElementType>
constexpr bool test_tuple_element() {
  return cute::is_same_v<std::tuple_element_t<Which, Tuple>, ExpectedElementType>;
}

void test_tuple_elements() {
  using cute::tuple;

  static_assert(test_tuple_element<std::tuple<Empty<0>>, 0, Empty<0>>());
  static_assert(test_tuple_element<tuple<Empty<0>>, 0, Empty<0>>());
}

// A default-constructible type.
template <size_t Value>
struct DefaultConstructible {};

void test_default_constructibility() {
  using cute::tuple;
  {
    [[maybe_unused]] tuple<> t_p_0;
    [[maybe_unused]] tuple<DefaultConstructible<0>> t_p_1;
    [[maybe_unused]] tuple<DefaultConstructible<0>, DefaultConstructible<1>> t_p_2;
    [[maybe_unused]] tuple<DefaultConstructible<0>, int, DefaultConstructible<1>> t_p_3;
  }
}

void test_sizes_and_not_storing_empty_types() {
  using cute::tuple;

  [[maybe_unused]] tuple<
    int,
    pt_test::Empty<0>,
    double
  > pt{42, pt_test::Empty<0>{}, 1.5};
  static_assert(cute::is_standard_layout_v<decltype(pt)>);
  // packed_result_type must only store the packed tuple,
  // and not the integer_sequence(s) used to access it.
  // The latter can be represented entirely at compile time as types.
  struct { int i; double j; } IntDouble;
  static_assert(sizeof(pt) == sizeof(IntDouble));

  EXPECT_EQ(cute::get<0>(pt), 42);
  EXPECT_EQ(cute::get<1>(pt), pt_test::Empty<0>{});
  EXPECT_EQ(cute::get<2>(pt), 1.5);
  tuple<
    pt_test::Empty<0>,
    pt_test::Empty<1>,
    tuple<
      pt_test::Empty<0>,
      pt_test::Empty<1>,
      tuple<pt_test::Empty<0>, tuple<>>
    >
  > pt_empty{};
  static_assert(cute::is_empty_v<decltype(pt_empty)>);
  static_assert(cute::is_standard_layout_v<decltype(pt_empty)>);
  static_assert(sizeof(pt_empty) == 1);

  // Template arguments must be default constructible,
  // and tuple itself needs a default constructor.
  [[maybe_unused]] tuple<
    tuple<int, pt_test::Empty<2>>,
    double,
    pt_test::Empty<3>> pt2;
  static_assert(cute::is_standard_layout_v<decltype(pt2)>);

  // cute::tuple, like the original cute::tuple, does not
  // promise to have working CTAD (constructor template argument
  // deduction).
  [[maybe_unused]] tuple<
    tuple<int, pt_test::Empty<0>>,
    pt_test::Empty<1>
  > pt3{
    tuple<int, pt_test::Empty<0>>{42, pt_test::Empty<0>{}},
    pt_test::Empty<1>{}
  };
  static_assert(cute::is_standard_layout_v<decltype(pt3)>);
  static_assert(cute::is_same_v<
    cute::tuple_element_t<0, decltype(pt3)>,
    tuple<int, pt_test::Empty<0>>>);
  static_assert(cute::is_same_v<
    cute::tuple_element_t<1, decltype(pt3)>,
    pt_test::Empty<1>>);
  static_assert(cute::tuple_size_v<cute::tuple_element_t<0, decltype(pt3)>> == 2u);

  tuple<int, pt_test::Empty<0>> pt3_0 = cute::get<0>(pt3);
  auto pt3_0_1 = cute::get<1>(pt3_0);
  static_assert(cute::is_same_v<decltype(pt3_0_1), pt_test::Empty<0>>);

  EXPECT_EQ(cute::get<0>(cute::get<0>(pt3)), 42);
  EXPECT_EQ(cute::get<1>(cute::get<0>(pt3)), pt_test::Empty<0>{});
}

} // namespace test

TEST(CuTe_core, PackedTuple)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("tuple");
  CUTLASS_TRACE_HOST("-------------------------------");

  pt_test::test_packed_type_aliases();
  pt_test::test_tuple_elements();
  pt_test::test_default_constructibility();
  pt_test::test_sizes_and_not_storing_empty_types();
}

TEST(CuTe_core, PackedTupleGet) {
  using cute::tuple;
  using pt_test::Empty;
  using pt_test::Nonempty;

  {
    using tuple_type = tuple<int>;
    tuple_type pt{42};
    static_assert(cute::tuple_size_v<tuple_type> == 1u);
    static_assert(cute::is_same_v<cute::tuple_element_t<0, tuple_type>, int>);
    EXPECT_EQ(cute::get<0>(pt), 42);
    cute::get<0>(pt) = 43;
    EXPECT_EQ(cute::get<0>(pt), 43);
  }
  {
    using tuple_type = tuple<int>;
    tuple_type const pt{42};
    EXPECT_EQ(cute::get<0>(pt), 42);
    static_assert(cute::is_same_v<decltype(cute::get<0>(pt)), int const&>);
  }
  {
    EXPECT_EQ(cute::get<0>(tuple<int>{42}), 42);
  }

  {
    using tuple_type = tuple<pt_test::Empty<0>>;
    tuple_type pt;
    static_assert(cute::tuple_size_v<tuple_type> == 1u);
    static_assert(cute::is_same_v<cute::tuple_element_t<0, tuple_type>, pt_test::Empty<0>>);
    EXPECT_EQ(cute::get<0>(pt), pt_test::Empty<0>{});
  }
  {
    using tuple_type = tuple<pt_test::Empty<0>>;
    tuple_type const pt;
    EXPECT_EQ(cute::get<0>(pt), pt_test::Empty<0>{});
  }
  {
    using tuple_type = tuple<pt_test::Empty<0>>;
    EXPECT_EQ(cute::get<0>(tuple_type{}), pt_test::Empty<0>{});
  }

  {
    using tuple_type = tuple<int, double>;
    tuple_type pt{1, 2.5};
    static_assert(cute::tuple_size_v<tuple_type> == 2u);
    static_assert(cute::is_same_v<cute::tuple_element_t<0, tuple_type>, int>);
    static_assert(cute::is_same_v<cute::tuple_element_t<1, tuple_type>, double>);
    EXPECT_EQ(cute::get<0>(pt), 1);
    cute::get<0>(pt) = 2;
    EXPECT_EQ(cute::get<0>(pt), 2);
    EXPECT_EQ(cute::get<1>(pt), 2.5);
    cute::get<1>(pt) = 3.5;
    EXPECT_EQ(cute::get<1>(pt), 3.5);
  }
  {
    using tuple_type = tuple<int, double>;
    tuple_type const pt{1, 2.5};
    EXPECT_EQ(cute::get<0>(pt), 1);
    static_assert(cute::is_same_v<decltype(cute::get<0>(pt)), int const&>);
    EXPECT_EQ(cute::get<1>(pt), 2.5);
    static_assert(cute::is_same_v<decltype(cute::get<1>(pt)), double const&>);
  }
  {
    using tuple_type = tuple<int, double>;
    EXPECT_EQ(cute::get<0>(tuple_type{1, 2.5}), 1);
    EXPECT_EQ(cute::get<1>(tuple_type{1, 2.5}), 2.5);
  }

  {
    using tuple_type = tuple<Empty<0>, double>;
    tuple_type pt{Empty<0>{}, 2.5};
    static_assert(cute::tuple_size_v<tuple_type> == 2u);
    static_assert(cute::is_same_v<cute::tuple_element_t<0, tuple_type>, Empty<0>>);
    static_assert(cute::is_same_v<cute::tuple_element_t<1, tuple_type>, double>);
    EXPECT_EQ(cute::get<0>(pt), Empty<0>{});
    EXPECT_EQ(cute::get<1>(pt), 2.5);
    cute::get<1>(pt) = 3.5;
    EXPECT_EQ(cute::get<1>(pt), 3.5);
  }
  {
    using tuple_type = tuple<Empty<0>, double>;
    tuple_type const pt{Empty<0>{}, 2.5};
    EXPECT_EQ(cute::get<0>(pt), Empty<0>{});
    static_assert(cute::is_same_v<decltype(cute::get<0>(pt)), Empty<0>>);
    EXPECT_EQ(cute::get<1>(pt), 2.5);
    static_assert(cute::is_same_v<decltype(cute::get<1>(pt)), double const&>);
  }
  {
    using tuple_type = tuple<Empty<0>, double>;
    EXPECT_EQ(cute::get<0>(tuple_type{Empty<0>{}, 2.5}), Empty<0>{});
    EXPECT_EQ(cute::get<1>(tuple_type{Empty<0>{}, 2.5}), 2.5);
  }

  {
    using tuple_type = tuple<int, double, Nonempty<float>>;
    tuple_type pt{1, 2.5, Nonempty{3.25f}};
    static_assert(cute::tuple_size_v<tuple_type> == 3u);
    static_assert(cute::is_same_v<cute::tuple_element_t<0, tuple_type>, int>);
    static_assert(cute::is_same_v<cute::tuple_element_t<1, tuple_type>, double>);
    static_assert(cute::is_same_v<cute::tuple_element_t<2, tuple_type>, Nonempty<float>>);
    EXPECT_EQ(cute::get<0>(pt), 1);
    EXPECT_EQ(cute::get<1>(pt), 2.5);
    EXPECT_EQ(cute::get<2>(pt), Nonempty{3.25f});

    cute::get<0>(pt) = 42;
    EXPECT_EQ(cute::get<0>(pt), 42);
    cute::get<1>(pt) = 4.5;
    EXPECT_EQ(cute::get<1>(pt), 4.5);
    cute::get<2>(pt) = Nonempty<float>{3.75f};
    EXPECT_EQ(cute::get<2>(pt), Nonempty<float>{3.75f});
  }
  {
    using tuple_type = tuple<int, double, Nonempty<float>>;
    tuple_type const pt{1, 2.5, Nonempty{3.25f}};
    EXPECT_EQ(cute::get<0>(pt), 1);
    EXPECT_EQ(cute::get<1>(pt), 2.5);
    EXPECT_EQ(cute::get<2>(pt), Nonempty{3.25f});
  }
  {
    using tuple_type = tuple<int, double, Nonempty<float>>;
    EXPECT_EQ((cute::get<0>(tuple_type{1, 2.5, Nonempty{3.25f}})), 1);
    EXPECT_EQ((cute::get<1>(tuple_type{1, 2.5, Nonempty{3.25f}})), 2.5);
    EXPECT_EQ((cute::get<2>(tuple_type{1, 2.5, Nonempty{3.25f}})), Nonempty{3.25f});
  }

  {
    using tuple_type = tuple<int, Empty<0>, Nonempty<float>>;
    tuple<int, Empty<0>, Nonempty<float>> pt{1, Empty<0>{}, Nonempty{3.25f}};
    static_assert(cute::tuple_size_v<tuple_type> == 3u);
    static_assert(cute::is_same_v<cute::tuple_element_t<0, tuple_type>, int>);
    static_assert(cute::is_same_v<cute::tuple_element_t<1, tuple_type>, Empty<0>>);
    static_assert(cute::is_same_v<cute::tuple_element_t<2, tuple_type>, Nonempty<float>>);
    EXPECT_EQ(cute::get<0>(pt), 1);
    EXPECT_EQ(cute::get<1>(pt), Empty<0>{});
    EXPECT_EQ(cute::get<2>(pt), Nonempty{3.25f});

    cute::get<0>(pt) = 42;
    EXPECT_EQ(cute::get<0>(pt), 42);
    cute::get<2>(pt) = Nonempty<float>{3.75f};
    EXPECT_EQ(cute::get<2>(pt), Nonempty<float>{3.75f});
  }
  {
    using tuple_type = tuple<int, Empty<0>, Nonempty<float>>;
    tuple_type const pt{1, Empty<0>{}, Nonempty{3.25f}};
    EXPECT_EQ(cute::get<0>(pt), 1);
    EXPECT_EQ(cute::get<1>(pt), Empty<0>{});
    EXPECT_EQ(cute::get<2>(pt), Nonempty{3.25f});
  }
  {
    using tuple_type = tuple<int, Empty<0>, Nonempty<float>>;
    EXPECT_EQ((cute::get<0>(tuple_type{1, Empty<0>{}, Nonempty{3.25f}})), 1);
    EXPECT_EQ((cute::get<1>(tuple_type{1, Empty<0>{}, Nonempty{3.25f}})), Empty<0>{});
    EXPECT_EQ((cute::get<2>(tuple_type{1, Empty<0>{}, Nonempty{3.25f}})), Nonempty{3.25f});
  }
}

TEST(CuTe_core, PackedTupleGetValueCategory) {
  using cute::tuple;
  using pt_test::Empty;
  using pt_test::Nonempty;

  tuple<Nonempty<int>, int, Empty<42>> tup(Nonempty<int>{42}, 7, Empty<42>{});

  // Lvalue ref
  decltype(auto) t0 = cute::get<0>(tup);
  decltype(auto) t1 = cute::get<1>(tup);
  decltype(auto) t2 = cute::get<2>(tup);

  EXPECT_TRUE((cute::is_same_v<decltype(t0), Nonempty<int>&>));
  EXPECT_TRUE((cute::is_same_v<decltype(t1), int&>));
  EXPECT_TRUE((cute::is_same_v<decltype(t2), Empty<42>>));

  // Const lvalue ref
  auto const& ctup = tup;
  decltype(auto) ct0 = cute::get<0>(ctup);
  decltype(auto) ct1 = cute::get<1>(ctup);
  decltype(auto) ct2 = cute::get<2>(ctup);

  EXPECT_TRUE((cute::is_same_v<decltype(ct0), Nonempty<int> const&>));
  EXPECT_TRUE((cute::is_same_v<decltype(ct1), int const&>));
  EXPECT_TRUE((cute::is_same_v<decltype(ct2), Empty<42>>));

  // Rvalue ref
  decltype(auto) r0 = cute::get<0>(cute::move(tup));
  decltype(auto) r1 = cute::get<1>(cute::move(tup));
  decltype(auto) r2 = cute::get<2>(cute::move(tup));

  EXPECT_TRUE((cute::is_same_v<decltype(r0), Nonempty<int>&&>));
  EXPECT_TRUE((cute::is_same_v<decltype(r1), int&&>));
  EXPECT_TRUE((cute::is_same_v<decltype(r2), Empty<42>>));
}

namespace pt_test {

// An empty class type to which Empty is convertible.
template <int Value>
struct ConvertibleFromEmpty {
  constexpr ConvertibleFromEmpty() = default;
  constexpr ConvertibleFromEmpty(Empty<Value>) {}

  template <int OtherValue>
  friend constexpr bool operator==(ConvertibleFromEmpty<Value> const&, ConvertibleFromEmpty<OtherValue> const&) {
    return Value == OtherValue;
  }

  template <int OtherValue>
  friend constexpr bool operator!=(ConvertibleFromEmpty<Value> const& lhs, ConvertibleFromEmpty<OtherValue> const& rhs) {
    return !(lhs == rhs);
  }
};

} // end namespace pt_test

TEST(CuTe_core, PackedTupleConstexprDefaultConstruction) {
  // Make sure that tuple's default constructor is constexpr.
  // MSVC makes this a bit more challenging than usual.

  using pt_test::Empty;
  {
    [[maybe_unused]] constexpr cute::eso::ESO_t<Empty<0>> eso1{};
    [[maybe_unused]] constexpr cute::eso::ESO_t<int64_t> eso2{};
  }
  {
    [[maybe_unused]] constexpr cute::eso::ESO_t<Empty<0>, Empty<1>> eso0{};
    [[maybe_unused]] constexpr cute::eso::ESO_t<int64_t, Empty<1>> eso1{};
    [[maybe_unused]] constexpr cute::eso::ESO_t<Empty<0>, int64_t> eso2{};
    [[maybe_unused]] constexpr cute::eso::ESO_t<int64_t, int64_t> eso3{};
  }
}

TEST(CuTe_core, PackedTupleConvertingConstruction) {
  using cute::tuple;
  using pt_test::ConvertibleFromEmpty;
  using pt_test::Empty;
  using pt_test::Nonempty;

  {
    using tuple_type = cute::tuple<Nonempty<int>>;
    [[maybe_unused]] tuple_type t(7);
    EXPECT_EQ(cute::get<0>(t), Nonempty<int>(7));
  }
  {
    using tuple_type = tuple<Nonempty<int>>;
    [[maybe_unused]] tuple_type t(7);
    EXPECT_EQ(cute::get<0>(t), Nonempty<int>(7));
  }
  {
    using tuple_type = cute::tuple<ConvertibleFromEmpty<0>>;
    [[maybe_unused]] tuple_type t(Empty<0>{});
    EXPECT_EQ(cute::get<0>(t), ConvertibleFromEmpty<0>{});
  }
  {
    using tuple_type = tuple<ConvertibleFromEmpty<0>>;
    [[maybe_unused]] tuple_type t(Empty<0>{});
    EXPECT_EQ(cute::get<0>(t), ConvertibleFromEmpty<0>{});
  }

  {
    using tuple_type = cute::tuple<float, Nonempty<int>>;
    [[maybe_unused]] tuple_type t(1.5f, 7);
    EXPECT_EQ(cute::get<0>(t), 1.5f);
    EXPECT_EQ(cute::get<1>(t), Nonempty<int>(7));
  }
  {
    using tuple_type = tuple<float, Nonempty<int>>;
    [[maybe_unused]] tuple_type t(1.5f, 7);
    EXPECT_EQ(cute::get<0>(t), 1.5f);
    EXPECT_EQ(cute::get<1>(t), Nonempty<int>(7));
  }

  {
    using tuple_type = cute::tuple<Empty<0>, Nonempty<int>>;
    [[maybe_unused]] tuple_type t(Empty<0>{}, 7);
    EXPECT_EQ(cute::get<0>(t), Empty<0>{});
    EXPECT_EQ(cute::get<1>(t), Nonempty<int>(7));
  }
  {
    using tuple_type = tuple<Empty<0>, Nonempty<int>>;
    [[maybe_unused]] tuple_type t(Empty<0>{}, 7);
    EXPECT_EQ(cute::get<0>(t), Empty<0>{});
    EXPECT_EQ(cute::get<1>(t), Nonempty<int>(7));
  }

  {
    using tuple_type = cute::tuple<ConvertibleFromEmpty<0>, Nonempty<int>>;
    [[maybe_unused]] tuple_type t(Empty<0>{}, 7);
    EXPECT_EQ(cute::get<0>(t), ConvertibleFromEmpty<0>{});
    EXPECT_EQ(cute::get<1>(t), Nonempty<int>(7));
  }
  {
    using tuple_type = tuple<ConvertibleFromEmpty<0>, Nonempty<int>>;
    [[maybe_unused]] tuple_type t(Empty<0>{}, 7);
    EXPECT_EQ(cute::get<0>(t), ConvertibleFromEmpty<0>{});
    EXPECT_EQ(cute::get<1>(t), Nonempty<int>(7));
  }

  {
    using inner_tuple_type = cute::tuple<Empty<0>>;
    using outer_tuple_type = cute::tuple<inner_tuple_type>;
    [[maybe_unused]] outer_tuple_type t(inner_tuple_type{Empty<0>{}});
  }
  {
    using inner_tuple_type = tuple<Empty<0>>;
    using outer_tuple_type = tuple<inner_tuple_type>;
    [[maybe_unused]] outer_tuple_type t(inner_tuple_type{Empty<0>{}});
  }
  {
    using inner_tuple_type = cute::tuple<ConvertibleFromEmpty<0>>;
    using outer_tuple_type = cute::tuple<inner_tuple_type>;
    [[maybe_unused]] outer_tuple_type t(inner_tuple_type{Empty<0>{}});
  }
  {
    using inner_tuple_type = tuple<ConvertibleFromEmpty<0>>;
    using outer_tuple_type = tuple<inner_tuple_type>;
    [[maybe_unused]] outer_tuple_type t(inner_tuple_type{Empty<0>{}});
  }
}

namespace test {

template <size_t ExpectedIndex, class X, class Tuple>
void test_tuple_find(Tuple const& t) {
  auto index = cute::find<X>(t);
  static_assert(decltype(index)::value == ExpectedIndex);
}

template <template <class...> class Tuple>
void test_tuple_find_all() {
  using test::test_tuple_find;
  using cute::_1;
  using cute::_2;
  using cute::_4;

  test_tuple_find<0, _1>(Tuple<_1>{});
  test_tuple_find<1, _2>(Tuple<_1>{});
  test_tuple_find<0, int>(Tuple<int>{7});

  test_tuple_find<0, _1>(Tuple<_1, _2>{});
  test_tuple_find<0, _1>(Tuple<_1, int>{_1{}, 7});
  test_tuple_find<0, float>(Tuple<float, int>{15.5f, 7});
  test_tuple_find<1, _2>(Tuple<_1, _2>{});
  test_tuple_find<1, int>(Tuple<_1, int>{_1{}, 7});
  test_tuple_find<1, int>(Tuple<float, int>{15.5f, 7});

  test_tuple_find<0, _1>(Tuple<_1, _2, _4>{_1{}, _2{}, _4{}});
  test_tuple_find<0, _1>(Tuple<_1, _2, int>{_1{}, _2{}, 7});
  test_tuple_find<0, _1>(Tuple<_1, float, _4>{_1{}, 15.5f, _4{}});
  test_tuple_find<0, _1>(Tuple<_1, float, int>{_1{}, 15.5f, 7});
  test_tuple_find<0, double>(Tuple<double, _2, _4>{105.5, _2{}, _4{}});
  test_tuple_find<0, double>(Tuple<double, float, _4>{105.5, 15.5f, _4{}});
  test_tuple_find<0, double>(Tuple<double, float, int>{105.5, 15.5f, 7});

  test_tuple_find<1, _2>(Tuple<_1, _2, _4>{_1{}, _2{}, _4{}});
  test_tuple_find<1, _2>(Tuple<_1, _2, int>{_1{}, _2{}, 7});
  test_tuple_find<1, float>(Tuple<_1, float, _4>{_1{}, 15.5f, _4{}});
  test_tuple_find<1, float>(Tuple<_1, float, int>{_1{}, 15.5f, 7});
  test_tuple_find<1, _2>(Tuple<double, _2, _4>{105.5, _2{}, _4{}});
  test_tuple_find<1, float>(Tuple<double, float, _4>{105.5, 15.5f, _4{}});
  test_tuple_find<1, float>(Tuple<double, float, int>{105.5, 15.5f, 7});

  test_tuple_find<2, _4>(Tuple<_1, _2, _4>{_1{}, _2{}, _4{}});
  test_tuple_find<2, int>(Tuple<_1, _2, int>{_1{}, _2{}, 7});
  test_tuple_find<2, _4>(Tuple<_1, float, _4>{_1{}, 15.5f, _4{}});
  test_tuple_find<2, int>(Tuple<_1, float, int>{_1{}, 15.5f, 7});
  test_tuple_find<2, _4>(Tuple<double, _2, _4>{105.5, _2{}, _4{}});
  test_tuple_find<2, _4>(Tuple<double, float, _4>{105.5, 15.5f, _4{}});
  test_tuple_find<2, int>(Tuple<double, float, int>{105.5, 15.5f, 7});
}

} // end namespace test

TEST(CuTe_core, TupleFind)
{
  test::test_tuple_find_all<cute::tuple>();
}
