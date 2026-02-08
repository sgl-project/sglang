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
#include <cute/stride.hpp>

TEST(CuTe_core, CompactColMajor_Static)
{
  using namespace cute;

  CUTE_STATIC_ASSERT_V((compact_col_major(Int<1>{}) == Int<0>{}));
  CUTE_STATIC_ASSERT_V((compact_col_major(Int<1>{}, Int<3>{}) == Int<0>{}));
  CUTE_STATIC_ASSERT_V((compact_col_major(Int<8>{}) == Int<1>{}));
  CUTE_STATIC_ASSERT_V((compact_col_major(Int<8>{}, Int<3>{}) == Int<3>{}));

  CUTE_STATIC_ASSERT_V((compact_col_major(1) == Int<1>{}));
  CUTE_STATIC_ASSERT_V((compact_col_major(8) == Int<1>{}));

  {
    auto test   = make_tuple(Int<4>{}, Int<8>{});
    auto result = make_tuple(Int<1>{}, Int<4>{});
    CUTE_STATIC_ASSERT_V((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(Int<4>{}, Int<8>{}, Int< 2>{});
    auto result = make_tuple(Int<1>{}, Int<4>{}, Int<32>{});
    CUTE_STATIC_ASSERT_V((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(Int<4>{}, Int<8>{}, Int<1>{}, Int< 2>{});
    auto result = make_tuple(Int<1>{}, Int<4>{}, Int<0>{}, Int<32>{});
    CUTE_STATIC_ASSERT_V((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(make_tuple(Int<4>{}, Int<8>{}), Int<1>{}, Int< 2>{});
    auto result = make_tuple(make_tuple(Int<1>{}, Int<4>{}), Int<0>{}, Int<32>{});
    CUTE_STATIC_ASSERT_V((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(Int<4>{}, make_tuple(Int<8>{}, Int<1>{}, Int< 2>{}));
    auto result = make_tuple(Int<1>{}, make_tuple(Int<4>{}, Int<0>{}, Int<32>{}));
    CUTE_STATIC_ASSERT_V((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(Int<4>{}, make_tuple(Int<8>{}, Int<1>{}, make_tuple(Int< 2>{}, Int< 3>{})));
    auto result = make_tuple(Int<1>{}, make_tuple(Int<4>{}, Int<0>{}, make_tuple(Int<32>{}, Int<64>{})));
    CUTE_STATIC_ASSERT_V((compact_col_major(test) == result));
  }
}

TEST(CuTe_core, CompactColMajor_Dynamic)
{
  using namespace cute;

  ASSERT_TRUE((compact_col_major(1) == 1));
  ASSERT_TRUE((compact_col_major(1, 3) == 3));
  ASSERT_TRUE((compact_col_major(8) == 1));
  ASSERT_TRUE((compact_col_major(8, 3) == 3));

  ASSERT_TRUE((compact_col_major(1) == 1));
  ASSERT_TRUE((compact_col_major(8) == 1));

  {
    auto test   = make_tuple(4, 8);
    auto result = make_tuple(1, 4);
    ASSERT_TRUE((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(4, 8,  2);
    auto result = make_tuple(1, 4, 32);
    ASSERT_TRUE((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(4, 8,  1,  2);
    auto result = make_tuple(1, 4, 32, 32);
    ASSERT_TRUE((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(make_tuple(4, 8),  1,  2);
    auto result = make_tuple(make_tuple(1, 4), 32, 32);
    ASSERT_TRUE((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(4, make_tuple(8,  1, 2));
    auto result = make_tuple(1, make_tuple(4, 32, 32));
    ASSERT_TRUE((compact_col_major(test) == result));
  }

  {
    auto test   = make_tuple(4, make_tuple(8,  1, make_tuple( 2, 3)));
    auto result = make_tuple(1, make_tuple(4, 32, make_tuple(32, 64)));
    ASSERT_TRUE((compact_col_major(test) == result));
  }
}

TEST(CuTe_core, CompactRowMajor_Static)
{
  using namespace cute;

  CUTE_STATIC_ASSERT_V((compact_row_major(Int<1>{}) == Int<0>{}));
  CUTE_STATIC_ASSERT_V((compact_row_major(Int<1>{}, Int<3>{}) == Int<0>{}));
  CUTE_STATIC_ASSERT_V((compact_row_major(Int<8>{}) == Int<1>{}));
  CUTE_STATIC_ASSERT_V((compact_row_major(Int<8>{}, Int<3>{}) == Int<3>{}));

  CUTE_STATIC_ASSERT_V((compact_row_major(1) == Int<1>{}));
  CUTE_STATIC_ASSERT_V((compact_row_major(8) == Int<1>{}));

  {
    auto test   = make_tuple(Int<4>{}, Int<8>{});
    auto result = make_tuple(Int<8>{}, Int<1>{});
    CUTE_STATIC_ASSERT_V((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple(Int< 4>{}, Int<8>{}, Int<2>{});
    auto result = make_tuple(Int<16>{}, Int<2>{}, Int<1>{});
    CUTE_STATIC_ASSERT_V((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple(Int< 4>{}, Int<8>{}, Int<1>{}, Int<2>{});
    auto result = make_tuple(Int<16>{}, Int<2>{}, Int<0>{}, Int<1>{});
    CUTE_STATIC_ASSERT_V((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple(make_tuple(Int< 4>{}, Int<8>{}), Int<1>{}, Int<2>{});
    auto result = make_tuple(make_tuple(Int<16>{}, Int<2>{}), Int<0>{}, Int<1>{});
    CUTE_STATIC_ASSERT_V((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple(Int< 4>{}, make_tuple(Int<8>{}, Int<1>{}, Int<2>{}));
    auto result = make_tuple(Int<16>{}, make_tuple(Int<2>{}, Int<0>{}, Int<1>{}));
    CUTE_STATIC_ASSERT_V((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple(Int< 4>{}, make_tuple(Int<8>{}, Int<1>{}, make_tuple(Int<2>{}, Int<3>{})));
    auto result = make_tuple(Int<48>{}, make_tuple(Int<6>{}, Int<0>{}, make_tuple(Int<3>{}, Int<1>{})));
    CUTE_STATIC_ASSERT_V((compact_row_major(test) == result));
  }
}

TEST(CuTe_core, CompactRowMajor_Dynamic)
{
  using namespace cute;

  ASSERT_TRUE((compact_row_major(1) == 1));
  ASSERT_TRUE((compact_row_major(1, 3) == 3));
  ASSERT_TRUE((compact_row_major(8) == 1));
  ASSERT_TRUE((compact_row_major(8, 3) == 3));

  ASSERT_TRUE((compact_row_major(1) == 1));
  ASSERT_TRUE((compact_row_major(8) == 1));

  {
    auto test   = make_tuple(4, 8);
    auto result = make_tuple(8, 1);
    ASSERT_TRUE((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple( 4, 8, 2);
    auto result = make_tuple(16, 2, 1);
    ASSERT_TRUE((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple( 4, 8, 1, 2);
    auto result = make_tuple(16, 2, 2, 1);
    ASSERT_TRUE((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple(make_tuple( 4, 8), 1, 2);
    auto result = make_tuple(make_tuple(16, 2), 2, 1);
    ASSERT_TRUE((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple( 4, make_tuple(8, 1, 2));
    auto result = make_tuple(16, make_tuple(2, 2, 1));
    ASSERT_TRUE((compact_row_major(test) == result));
  }

  {
    auto test   = make_tuple( 4, make_tuple(8, 1, make_tuple(2, 3)));
    auto result = make_tuple(48, make_tuple(6, 6, make_tuple(3, 1)));
    ASSERT_TRUE((compact_row_major(test) == result));
  }
}
