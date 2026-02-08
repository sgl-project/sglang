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

#include "cutlass_unit_test.h"

#include <cutlass/trace.h>

#include <cassert>
#include <type_traits>

#include <cute/container/tuple.hpp>
#include <cute/int_tuple.hpp>

template<class T>
class ConvertibleTo {
public:
  ConvertibleTo(T val) : val_(val) {}

  operator T () const { return val_; }

private:
  T val_ = 0;
};

template<class Integral, Integral Value>
using IC = std::integral_constant<Integral, Value>;

TEST(CuTe_core_msvc_compilation, TupleAssignment)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("cute::tuple creation and assignment");
  CUTLASS_TRACE_HOST("-------------------------------");

  using forty_two_type = IC<int, 42>;
  using forty_three_type = IC<size_t, 43>;

  int val41 = ConvertibleTo{41};
  assert(val41 == 41);
  size_t val43 = ConvertibleTo{size_t(43u)};
  assert(val43 == size_t{43u});

  using tuple_0d_type = cute::tuple<>;
  using tuple_1d_d_type = cute::tuple<int>;
  using tuple_2d_dd_type = cute::tuple<int, size_t>;

  [[maybe_unused]] tuple_0d_type t0;

  // Symptom: "illegal member initialization: 'TupleBase<int>' is not a base or member"
  [[maybe_unused]] tuple_1d_d_type t1{ 42 };
  [[maybe_unused]] tuple_1d_d_type t1a{ 43 };
  t1 = t1a;

  [[maybe_unused]] tuple_2d_dd_type t3{ 42, size_t(43u) };
  [[maybe_unused]] tuple_2d_dd_type t3a{ 44, size_t(45u) };
  // Symptom: "illegal member initialization:
  // 'TupleBase<int, unsigned __int64>' is not a base or member"
  t3 = t3a;
}

TEST(CuTe_core_msvc_compilation, TupleGetSingleInteger)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("cute::get<I> on cute::tuple for single integer I");
  CUTLASS_TRACE_HOST("-------------------------------");

  cute::tuple<int, ConvertibleTo<size_t>, IC<int, 43>> t0{ 41, size_t(42u), IC<int, 43>{} };

  [[maybe_unused]] auto t0_0 = cute::get<0>(t0);
  static_assert(std::is_same_v<decltype(t0_0), int>);
  assert(t0_0 == 41);

  [[maybe_unused]] auto t0_1 = cute::get<1>(t0);
  static_assert(std::is_same_v<decltype(t0_1), ConvertibleTo<size_t>>);

  [[maybe_unused]] auto t0_2 = cute::get<2>(t0);
  static_assert(std::is_same_v<decltype(t0_2), IC<int, 43>>);
}

TEST(CuTe_core_msvc_compilation, TupleGetRecursive)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("cute::get<I...> on cute::tuple");
  CUTLASS_TRACE_HOST("-------------------------------");

  using inner_tuple_type = cute::tuple<int, ConvertibleTo<size_t>, IC<int, 43>>;
  using outer_tuple_type = cute::tuple<IC<int, 40>, inner_tuple_type, size_t>;

  inner_tuple_type t0_inner{ 41, size_t(42u), IC<int, 43>{} };
  outer_tuple_type t0_outer{ IC<int, 40>{}, t0_inner, size_t(44u) };

        [[maybe_unused]] auto t0_outer_0 = cute::get<0>(t0_outer);
        static_assert(std::is_same_v<decltype(t0_outer_0), IC<int, 40>>);

        [[maybe_unused]] auto t0_outer_1 = cute::get<1>(t0_outer);
        static_assert(std::is_same_v<decltype(t0_outer_1), inner_tuple_type>);

        [[maybe_unused]] auto t0_outer_2 = cute::get<2>(t0_outer);
        static_assert(std::is_same_v<decltype(t0_outer_2), size_t>);
        assert(t0_outer_2 == size_t(44u));

  // Leftmost index is innermost in the nexted get sequence.
  [[maybe_unused]] auto t0_outer_10 = cute::get<1, 0>(t0_outer);
  static_assert(std::is_same_v<decltype(t0_outer_10), int>);
  assert(t0_outer_10 == 41);
}
