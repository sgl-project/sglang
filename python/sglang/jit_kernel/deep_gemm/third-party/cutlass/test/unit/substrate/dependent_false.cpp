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
#include "cutlass/detail/dependent_false.hpp"

namespace { // (anonymous)

template<class ... Args>
void test_dependent_bool_value()
{
  static_assert(cutlass::detail::dependent_bool_value<true, Args...> == true);
  static_assert(cutlass::detail::dependent_bool_value<false, Args...> == false);
}

template<class ... Args>
void test_dependent_false()
{
  static_assert(cutlass::detail::dependent_false<Args...> == false);
}

template<class ... Args>
void test_all()
{
  test_dependent_bool_value<Args...>();
  test_dependent_false<Args...>();
}

// Types to use in Args
struct Type0 {};
struct Type1 {};
struct Type2 {};

} // end namespace (anonymous)

TEST(LibcudacxxNext, DependentBoolValue)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("dependent_bool_value");
  CUTLASS_TRACE_HOST("-------------------------------");

  test_dependent_bool_value<int>();
  test_dependent_bool_value<float>();
  test_dependent_bool_value<int, float>();
  test_dependent_bool_value<Type0, int, float, Type1, float, int, Type2>();
}

TEST(LibcudacxxNext, DependentFalse)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("dependent_false");
  CUTLASS_TRACE_HOST("-------------------------------");

  test_dependent_false<int>();
  test_dependent_false<float>();
  test_dependent_false<int, float>();
  test_dependent_false<Type0, int, float, Type1, float, int, Type2>();
}
