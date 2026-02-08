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

#include <cute/layout.hpp>

using namespace cute;

TEST(CuTe_core, WeaklyCongruent)
{
  auto a = _1{};
  auto b = _2{};
  EXPECT_TRUE (weakly_congruent(a, a));
  EXPECT_TRUE (weakly_congruent(b, b));
  EXPECT_TRUE (weakly_congruent(a, b));

  auto a0 = Shape<_1>{};
  auto b0 = Shape<_2>{};
  EXPECT_TRUE (weakly_congruent(a , a0));
  EXPECT_TRUE (weakly_congruent(b , b0));
  EXPECT_TRUE (weakly_congruent(a , b0));
  EXPECT_TRUE (weakly_congruent(b , a0));
  EXPECT_FALSE(weakly_congruent(a0, a ));
  EXPECT_FALSE(weakly_congruent(b0, b ));
  EXPECT_FALSE(weakly_congruent(a0, b ));
  EXPECT_FALSE(weakly_congruent(b0, a ));
  EXPECT_TRUE (weakly_congruent(a0, a0));
  EXPECT_TRUE (weakly_congruent(b0, b0));
  EXPECT_TRUE (weakly_congruent(a0, b0));

  auto a1 = Shape<_1, _1>{};
  EXPECT_TRUE (weakly_congruent(a , a1));
  EXPECT_FALSE(weakly_congruent(a0, a1));
  EXPECT_TRUE (weakly_congruent(a1, a1));

  auto a2 = Shape<_1, Shape<_1,_1>>{};
  EXPECT_TRUE (weakly_congruent(a , a2));
  EXPECT_FALSE(weakly_congruent(a0, a2));
  EXPECT_TRUE (weakly_congruent(a1, a2));

  auto b1 = Shape<_2, _2>{};
  EXPECT_TRUE (weakly_congruent(b , b1));
  EXPECT_FALSE(weakly_congruent(b0, b1));
  EXPECT_TRUE (weakly_congruent(a1, b1));

  auto b2 = Shape<_2, Shape<_2,_2>>{};
  EXPECT_FALSE(weakly_congruent(a2, b0));
  EXPECT_FALSE(weakly_congruent(a2, a1));
  EXPECT_TRUE (weakly_congruent(a2, b2));

  auto b3 = Shape<Shape<_2,_2>, Shape<_2,_2>>{};
  EXPECT_FALSE(weakly_congruent(a0, b3));
  EXPECT_TRUE (weakly_congruent(a1, b3));
  EXPECT_TRUE (weakly_congruent(a2, b3));
}

template <class A, class B>
auto test_evenly_divides(A const& a, B const& b)
{
  auto result = evenly_divides(a, b);
  // If A and B are static, then result should be as well
  if constexpr (is_static<A>::value && is_static<B>::value) {
    static_assert(is_static<decltype(result)>::value);
  }
  // If result is true_type, then confirm divisibillity
  if constexpr (is_constant<true, decltype(result)>::value) {
    CUTE_STATIC_ASSERT_V(size(a) == size(logical_divide(make_layout(shape(a)), b)));
  }

  return result;
}

TEST(CuTe_core, Divides)
{
  {
  auto a = _16{};
  auto b = _12{};
  auto c = _8{};
  EXPECT_TRUE (test_evenly_divides(a, a));
  EXPECT_TRUE (test_evenly_divides(b, b));
  EXPECT_TRUE (test_evenly_divides(c, c));
  EXPECT_FALSE(test_evenly_divides(a, b));
  EXPECT_TRUE (test_evenly_divides(a, c));
  EXPECT_FALSE(test_evenly_divides(c, a));

  auto a0 = Shape<_16>{};
  EXPECT_TRUE (test_evenly_divides(a0, a0));
  EXPECT_TRUE (test_evenly_divides(a , a0));
  EXPECT_TRUE (test_evenly_divides(a0, a ));
  EXPECT_FALSE(test_evenly_divides(c , a0));
  EXPECT_TRUE (test_evenly_divides(a0, c ));
  EXPECT_FALSE(test_evenly_divides(b , a0));
  EXPECT_FALSE(test_evenly_divides(a0, b ));

  auto a1 = Shape<_2,_8>{};
  EXPECT_TRUE (test_evenly_divides(a1, a1));
  EXPECT_FALSE(test_evenly_divides(a , a1));
  EXPECT_FALSE(test_evenly_divides(a0, a1));
  EXPECT_FALSE(test_evenly_divides(a1, a0));
  EXPECT_FALSE(test_evenly_divides(a1, Shape<_2,Shape<_2,_4>>{}));

  auto a2 = Shape<Shape<_2,_8>>{};
  EXPECT_TRUE (test_evenly_divides(a2, a2));
  EXPECT_FALSE(test_evenly_divides(a , a2));
  EXPECT_FALSE(test_evenly_divides(c , a2));
  EXPECT_FALSE(test_evenly_divides(a0, a2));
  EXPECT_TRUE (test_evenly_divides(a2, a0));

  auto a3 = Shape<Shape<_2,Shape<_4,_2>>>{};
  EXPECT_TRUE (test_evenly_divides(a3, a3));
  EXPECT_FALSE(test_evenly_divides(a , a3));
  EXPECT_FALSE(test_evenly_divides(c , a3));
  EXPECT_FALSE(test_evenly_divides(a0, a3));
  EXPECT_TRUE (test_evenly_divides(a3, a0));
  EXPECT_FALSE(test_evenly_divides(a2, a3));
  EXPECT_TRUE (test_evenly_divides(a3, a2));
  }

  {
  auto a = 16;
  auto b = 12;
  auto c =  8;
  EXPECT_TRUE (test_evenly_divides(a, a));
  EXPECT_TRUE (test_evenly_divides(b, b));
  EXPECT_TRUE (test_evenly_divides(c, c));
  EXPECT_FALSE(test_evenly_divides(a, b));
  EXPECT_TRUE (test_evenly_divides(a, c));
  EXPECT_FALSE(test_evenly_divides(c, a));

  auto a0 = make_shape(16);
  EXPECT_TRUE (test_evenly_divides(a0, a0));
  EXPECT_TRUE (test_evenly_divides(a , a0));
  EXPECT_TRUE (test_evenly_divides(a0, a ));
  EXPECT_FALSE(test_evenly_divides(c , a0));
  EXPECT_TRUE (test_evenly_divides(a0, c ));
  EXPECT_FALSE(test_evenly_divides(b , a0));
  EXPECT_FALSE(test_evenly_divides(a0, b ));

  auto a1 = make_shape(2, 8);
  EXPECT_TRUE (test_evenly_divides(a1, a1));
  EXPECT_FALSE(test_evenly_divides(a , a1));
  EXPECT_FALSE(test_evenly_divides(a0, a1));
  EXPECT_FALSE(test_evenly_divides(a1, a0));
  EXPECT_FALSE(test_evenly_divides(a1, make_shape(2,make_shape(2,4))));

  auto a2 = make_shape(make_shape(2,8));
  EXPECT_TRUE (test_evenly_divides(a2, a2));
  EXPECT_FALSE(test_evenly_divides(a , a2));
  EXPECT_FALSE(test_evenly_divides(c , a2));
  EXPECT_FALSE(test_evenly_divides(a0, a2));
  EXPECT_TRUE (test_evenly_divides(a2, a0));

  auto a3 = make_shape(make_shape(2,make_shape(4,2)));
  EXPECT_TRUE (test_evenly_divides(a3, a3));
  EXPECT_FALSE(test_evenly_divides(a , a3));
  EXPECT_FALSE(test_evenly_divides(c , a3));
  EXPECT_FALSE(test_evenly_divides(a0, a3));
  EXPECT_TRUE (test_evenly_divides(a3, a0));
  EXPECT_FALSE(test_evenly_divides(a2, a3));
  EXPECT_TRUE (test_evenly_divides(a3, a2));
  }

  {
  auto a = Shape<_32,_64>{};
  EXPECT_TRUE (test_evenly_divides(a, Int<128>{}));
  EXPECT_TRUE (test_evenly_divides(a, Tile<Layout<_8,_2>, _32>{}));
  EXPECT_FALSE(test_evenly_divides(a, Tile<Layout<_8,_3>, _32>{}));
  }
}
