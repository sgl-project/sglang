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

#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <numeric>

#include <cute/container/bit_field.hpp>

#include <cute/algorithm/tuple_algorithms.hpp>

using namespace cute;

TEST(CuTe_core, Bitfield)
{
  for_each(make_int_range<1,65>{}, [&](auto NumBits) {
    constexpr auto num_bits = cute::remove_cvref_t<decltype(NumBits)>::value;
    for_each(make_int_range<0, 129>{}, [&](auto BitStart) {
      constexpr auto bit_start = cute::remove_cvref_t<decltype(BitStart)>::value;
      using BF = bit_field<bit_start, cute::remove_cvref_t<decltype(NumBits)>::value>;

#if 0
      printf("bit_field<%d,%d>:\n", bit_start, num_bits);
      printf("  value_type_bits  : %d\n", BF::value_type_bits);
      printf("  storage_type_bits: %d\n", BF::storage_type_bits);
      printf("  N                : %d\n", BF::N);
      printf("  idx              : %d\n", BF::idx);
      printf("  bit_lo           : %d\n", BF::bit_lo);
      printf("  bit_hi           : %d\n", BF::bit_hi);
      printf("  mask             : 0x%lx\n", uint64_t(BF::mask));
      printf("  mask_lo          : 0x%lx\n", uint64_t(BF::mask_lo));
      printf("  mask_hi          : 0x%lx\n", uint64_t(BF::mask_hi));
#endif

      // Test
      uint64_t v = num_bits == 64 ? uint64_t(-1) : ((uint64_t(1) << NumBits) - 1);

      BF bf{};
      bf = v;
      EXPECT_EQ(v, uint64_t(bf));
    });
  });

  for_each(make_int_range<0,129>{}, [&](auto BitStart) {

    using BF = bit_field<cute::remove_cvref_t<decltype(BitStart)>::value, 32, float>;

    BF bf{};
    bf = 3.14f;
    EXPECT_EQ(3.14f, float(bf));
  });

}
