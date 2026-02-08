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

//#define CUTLASS_DEBUG_TRACE_LEVEL 1

#include "cutlass_unit_test.h"

#include <cutlass/trace.h>

#include <cute/tensor.hpp>

using namespace cute;

template <class LayoutA, class LayoutB>
void
test_logical_divide(LayoutA const& layoutA,
                    LayoutB const& layoutB)
{
  auto layoutR = logical_divide(layoutA, layoutB);

  CUTLASS_TRACE_HOST("test_logical_divide()");
  CUTLASS_TRACE_HOST( shape(layoutA) << " / " <<  shape(layoutB) << "  =>  " <<  shape(layoutR));
  CUTLASS_TRACE_HOST(stride(layoutA) << "   " << stride(layoutB) << "  =>  " << stride(layoutR));

  // Test that layout B is compatible with layout R_0
  ASSERT_EQ(rank(layoutR), 2);
  ASSERT_TRUE(compatible(layoutB, layout<0>(layoutR)));
}

TEST(CuTe_core, Logical_divide)
{
  {
  auto layout = Layout<_1,_0>{};
  auto tile   = Layout<_1,_0>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_1,_0>{};
  auto tile   = Layout<_1,_1>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_1,_1>{};
  auto tile   = Layout<_1,_0>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_1,_1>{};
  auto tile   = Layout<_1,_1>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_6,_1>{};
  auto tile   = Layout<_2,_1>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_6,_1>{};
  auto tile   = Layout<_2,_3>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_6,_1>{};
  auto tile   = Layout<Shape<_2,_3>,Stride<_3,_1>>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_6,_2>{};
  auto tile   = Layout<_2,_1>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_6,_2>{};
  auto tile   = Layout<_2,_3>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_6,_2>{};
  auto tile   = Layout<Shape<_2,_3>,Stride<_3,_1>>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<Shape<_6,_6>,Stride<_1,_12>>{};
  auto tile   = Layout<Shape<_6,_3>,Stride<_3,_1>>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<Shape<_6,_6>,Stride<_12,_1>>{};
  auto tile   = Layout<Shape<_6,_3>,Stride<_3,_1>>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<_32>{};
  auto tile   = Layout<_2,_8>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<Shape<_4,_1>,Stride<_1,_1>>{};
  auto tile   = Layout<_2,_1>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<Shape<_4,_1>,Stride<_1,_1>>{};
  auto tile   = Layout<_2,_2>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<Shape<_8,_8>,Stride<_1,_8>>{};
  auto tile   = Layout<_32,_2>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = Layout<Shape<_8,_8>,Stride<_8,_1>>{};
  auto tile   = Layout<_32,_2>{};

  test_logical_divide(layout, tile);
  }

  //
  // Dynamic
  //

  {
  auto layout = make_layout(2);
  auto tile   = Layout<_32>{};

  test_logical_divide(layout, tile);

  // Enforcement for dynamic cases
  auto result = logical_divide(layout, tile);
  ASSERT_TRUE(decltype(shape<0>(result) == Int<32>{})::value);
  ASSERT_TRUE(decltype(stride<0>(result) == Int<1>{})::value);
  ASSERT_TRUE(shape<1>(result) == 1);
  ASSERT_TRUE(decltype(stride<1>(result) == Int<32>{})::value);
  }

  {
  auto layout = make_layout(48);
  auto tile   = Layout<_32>{};

  test_logical_divide(layout, tile);

  // Enforcement for dynamic cases
  auto result = logical_divide(layout, tile);
  ASSERT_TRUE(decltype(shape<0>(result) == Int<32>{})::value);
  ASSERT_TRUE(decltype(stride<0>(result) == Int<1>{})::value);
  ASSERT_TRUE(shape<1>(result) == 2);
  ASSERT_TRUE(decltype(stride<1>(result) == Int<32>{})::value);
  }

  {
  auto layout = make_layout(96);
  auto tile   = Layout<_32,_2>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = make_layout(32);
  auto tile   = Layout<Int<48>>{};

  test_logical_divide(layout, tile);

  // Enforcement for dynamic cases
  auto result = logical_divide(layout, tile);
  ASSERT_TRUE(decltype(shape<0>(result) == Int<48>{})::value);
  ASSERT_TRUE(decltype(stride<0>(result) == Int<1>{})::value);
  ASSERT_TRUE(shape<1>(result) == 1);
  ASSERT_TRUE(decltype(stride<1>(result) == Int<48>{})::value);
  }

  {
  auto layout = make_layout(make_shape(Int<32>{}, Int<4>{}, 4));
  auto tile   = Layout<_64>{};

  test_logical_divide(layout, tile);

  // Enforcement of result
  auto result = logical_divide(layout, tile);
  ASSERT_TRUE(bool( shape(result) == make_shape (_64{}, make_shape ( _2{},     4))));
  ASSERT_TRUE(bool(stride(result) == make_stride( _1{}, make_stride(_64{},_128{}))));
  }


  //
  // ALLOWED, but dangerous due to the dynamic lhs shapes
  //   Consider disallowing...
  //

  {
  auto layout = make_layout(make_shape(128,4,3), make_stride(1,512,0));
  auto tile   = Layout<_32>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = make_layout(make_shape(128,4,3), make_stride(1,512,0));
  auto tile   = Layout<_32,_2>{};

  test_logical_divide(layout, tile);
  }

  {
  auto layout = make_layout(make_shape(16,4,3), make_stride(1,512,0));
  auto tile   = Layout<_32>{};

  test_logical_divide(layout, tile);
  }
}
