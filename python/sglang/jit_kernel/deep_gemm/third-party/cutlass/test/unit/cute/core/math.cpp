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
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/math.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/util/type_traits.hpp>

// If cute::gcd returns auto instead of common_type_t<T, U>,
// then GCC 7.5 reports the following error;
//
// ... /include/cute/numeric/math.hpp:103:26: error:
// inconsistent deduction for auto return type: ‘int’ and then ‘bool’
//      if (u == 0) { return t; }
//                           ^
// Note that common_type_t<C<42>, C<1>>::value_type might still be bool.
TEST(CuTe_core, gcd_returns_common_type)
{
  using cute::C;

  constexpr auto fifteen = C<3 * 5>{};
  static_assert(cute::is_same_v<decltype(fifteen)::value_type, int>);
  static_assert(int(fifteen) == 15);

  constexpr auto forty_two = C<2 * 3 * 7>{};
  static_assert(cute::is_same_v<decltype(forty_two)::value_type, int>);
  static_assert(int(forty_two) == 42);

  // C<1>::value_type (as well as C<0>::value_type) may be bool.
  constexpr auto one = C<1>{};

  // Both inputs have value_type int.
  {
    constexpr auto result = cute::gcd(fifteen, forty_two);
    static_assert(cute::is_same_v<decltype(result)::value_type, int>);
    static_assert(int(result) == 3);
  }

  // One input has value_type int, and the other may have value_type bool.
  {
    constexpr auto result = cute::gcd(one, forty_two);
    static_assert(int(result) == 1);
  }
  {
    constexpr auto result = cute::gcd(forty_two, one);
    static_assert(int(result) == 1);
  }

  // Both inputs may have value_type bool.
  {
    constexpr auto result = cute::gcd(one, one);
    static_assert(int(result) == 1);
  }
}

TEST(CuTe_core, lcm_returns_common_type)
{
  using cute::C;

  constexpr auto six = C<2 * 3>{};
  static_assert(cute::is_same_v<decltype(six)::value_type, int>);
  static_assert(int(six) == 6);

  constexpr auto fifteen = C<3 * 5>{};
  static_assert(cute::is_same_v<decltype(fifteen)::value_type, int>);
  static_assert(int(fifteen) == 15);

  // C<1>::value_type (as well as C<0>::value_type) may be bool.
  constexpr auto one = C<1>{};

  // Both inputs have value_type int.
  {
    constexpr auto result = cute::lcm(six, fifteen);
    static_assert(cute::is_same_v<decltype(result)::value_type, int>);
    static_assert(int(result) == 30);
  }

  // One input has value_type int, and the other may have value_type bool.
  {
    constexpr auto result = cute::lcm(one, six);
    static_assert(cute::is_same_v<decltype(result)::value_type, int>);
    static_assert(int(result) == 6);
  }
  {
    constexpr auto result = cute::lcm(six, one);
    static_assert(cute::is_same_v<decltype(result)::value_type, int>);
    static_assert(int(result) == 6);
  }

  // Both inputs may have value_type bool.
  {
    constexpr auto result = cute::lcm(one, one);
    static_assert(int(result) == 1);
  }
}

TEST(CuTe_core, max_alignment)
{
  {
    constexpr auto swizzle = cute::Swizzle<3,4,3>{};
    static_assert(cute::max_alignment(swizzle) == 1 << 4);
  }
}
