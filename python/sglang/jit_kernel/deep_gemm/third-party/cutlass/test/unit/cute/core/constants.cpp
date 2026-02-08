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
#include <cute/algorithm/tuple_algorithms.hpp>

TEST(CuTe_core, MakeIntegerSequence) {
  cute::for_each(cute::make_integer_sequence<uint32_t, 13>{}, [](auto c) {
    using c_type = decltype(c);
    constexpr auto c_value = c_type::value;
    using expected_type = cute::integral_constant<uint32_t, c_value>;
    static_assert(cute::is_same_v<c_type, expected_type>);
    static_assert(cute::is_same_v<typename c_type::value_type, uint32_t>);
    static_assert(cute::is_constant<c_value, c_type>::value);
    static_assert(cute::is_constant<0, decltype(c * cute::Int<0>{})>::value);
    static_assert(cute::is_constant<2*c_value, decltype(c * cute::Int<2>{})>::value);
  });

  cute::for_each(cute::make_integer_sequence<int64_t, 17>{}, [](auto c) {
    using c_type = decltype(c);
    constexpr auto c_value = c_type::value;
    using expected_type = cute::integral_constant<int64_t, c_value>;
    static_assert(cute::is_same_v<c_type, expected_type>);
    static_assert(cute::is_same_v<typename c_type::value_type, int64_t>);
    static_assert(cute::is_constant<c_value, c_type>::value);
    static_assert(cute::is_constant<0, decltype(c * cute::Int<0>{})>::value);
    static_assert(cute::is_constant<2*c_value, decltype(c * cute::Int<2>{})>::value);
  });
}
