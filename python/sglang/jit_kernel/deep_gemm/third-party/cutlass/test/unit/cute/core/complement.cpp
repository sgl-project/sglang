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

template <class Layout, class CoTarget>
void
test_complement(Layout const& layout, CoTarget const& cotarget)
{
  using namespace cute;

  auto result = complement(layout, cotarget);

  CUTLASS_TRACE_HOST("complement(" << layout << ", " << cotarget << ")  =>  " << result);

  auto completed = make_layout(layout, result);

  // Lower-bound on the codomain size of the layout ++ complement (1)
  EXPECT_GE(cosize(completed), size(cotarget));
  // Upper-bound on the codomain size of the complement (2)
  EXPECT_LE(cosize(result), cute::round_up(size(cotarget), cosize(layout)));

  // Post-condition on the codomain of the complement
  for (int i = 1; i < size(result); ++i) {
    EXPECT_LT(result(i-1), result(i));         // Ordered (3)
    for (int j = 0; j < size(layout); ++j) {
      EXPECT_NE(result(i), layout(j));         // Disjoint (4)
    }
  }

  // Other observations
  EXPECT_LE(size(result), cosize(result));                        // As a result of the ordered condition (3)
  EXPECT_GE(size(result), size(cotarget) / size(filter(layout)));
  EXPECT_LE(cosize(completed), cosize(result) + cosize(layout));
  EXPECT_GE(cosize(result), size(cotarget) / size(filter(layout)));
  if constexpr (is_static<decltype(stride(completed))>::value) {  // If we can apply complement again
    EXPECT_EQ(size(complement(completed)), 1);                    // There's no more codomain left over
  }
  if constexpr (is_static<decltype(result)>::value && is_static<decltype(layout)>::value) {
    EXPECT_TRUE(bool(complement(complement(result,cosize(layout)),cotarget) == result));
  }
}

template <class Layout>
void
test_complement(Layout const& layout)
{
  return test_complement(layout, cosize(layout));
}

TEST(CuTe_core, Complement)
{
  using namespace cute;

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("COMPLEMENT");
  CUTLASS_TRACE_HOST("-------------------------------");

  {
  auto layout = Layout<_1,_0>{};

  test_complement(layout);
  test_complement(layout, Int<2>{});
  test_complement(layout, Int<5>{});
  test_complement(layout, make_shape(Int<2>{}, 2));
  }

  {
  auto layout = Layout<_1,_1>{};

  test_complement(layout);
  test_complement(layout, Int<2>{});
  test_complement(layout, Int<5>{});
  test_complement(layout, make_shape(Int<2>{}, 2));
  }

  {
  auto layout = Layout<_1,_2>{};

  test_complement(layout, Int<1>{});
  test_complement(layout, Int<2>{});
  test_complement(layout, Int<8>{});
  test_complement(layout, Int<5>{});
  test_complement(layout, make_shape(Int<2>{}, 2));
  }

  {
  auto layout = Layout<_4,_0>{};

  test_complement(layout, Int<1>{});
  test_complement(layout, Int<2>{});
  test_complement(layout, Int<8>{});
  }

  {
  auto layout = Layout<_4,_1>{};

  test_complement(layout, Int<1>{});
  test_complement(layout, Int<2>{});
  test_complement(layout, Int<8>{});
  }

  {
  auto layout = Layout<_4,_2>{};

  test_complement(layout, Int<1>{});
  test_complement(layout);
  test_complement(layout, Int<16>{});
  test_complement(layout, Int<19>{});
  test_complement(layout, make_shape(Int<2>{}, 2));
  }

  {
  auto layout = Layout<_4,_4>{};

  test_complement(layout, Int<1>{});
  test_complement(layout);
  test_complement(layout, Int<17>{});
  test_complement(layout, make_shape(Int<2>{}, 2));
  }

  {
  auto layout = Layout<Shape<_2,_4>>{};

  test_complement(layout);
  }

  {
  auto layout = Layout<Shape<_2,_3>>{};

  test_complement(layout);
  }

  {
  auto layout = Layout<Shape<_2,_4>, Stride<_1,_4>>{};

  test_complement(layout);
  }

  {
  auto layout = Layout<Shape<_2,_4>, Stride<_1,_6>>{};

  test_complement(layout);
  }

  {
  auto layout = Layout<Shape<_2,_4,_8>, Stride<_8,_1,_64>>{};

  test_complement(layout);
  }

  {
  auto layout = Layout<Shape<_2,_4,_8>, Stride<_8,_1,_0>>{};

  test_complement(layout);
  test_complement(layout, Int<460>{});
  }

  {
  auto layout = make_layout(Shape <Shape <_2,_2>,Shape <_2, _2>>{},
                            Stride<Stride<_1,_4>,Stride<_8,_32>>{});

  test_complement(layout);
  }

  {
  auto layout = make_layout(Shape <Shape <_2, _2>,Shape <_2,_2>>{},
                            Stride<Stride<_1,_32>,Stride<_8,_4>>{});

  test_complement(layout);
  }

  // Fails due to non-injective layout
  // {
  // auto layout = make_layout(Shape <Shape <_2,_2>,Shape <_2,_2>>{},
  //                           Stride<Stride<_1,_8>,Stride<_8,_4>>{});

  // test_complement(layout);
  // }

  // Fails due to non-injective layout
  // {
  // auto layout = Layout<Shape<_2,_2>, Stride<_2,_3>>{};

  // test_complement(layout);
  // test_complement(layout, Int<19>{});
  // }

  {
  auto layout = Layout<Shape<_4,_6>, Stride<_1,_6>>{};

  test_complement(layout);
  }

  {
  auto layout = Layout<Shape<_4,_2>, Stride<_1,_10>>{};

  test_complement(layout);
  }

  {
  auto layout = Layout<Shape<_4,_2>, Stride<_1,_16>>{};

  test_complement(layout);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("Dynamic shapes/strides");
  CUTLASS_TRACE_HOST("-------------------------------");

  {
  auto layout = make_layout(12);

  test_complement(layout, 1);
  test_complement(layout);
  test_complement(layout, 53);
  test_complement(layout, 128);
  }

  {
  auto layout = make_layout(12, 1);

  test_complement(layout, 1);
  test_complement(layout);
  test_complement(layout, 53);
  test_complement(layout, 128);
  }

  {
  auto layout = make_layout(12, Int<2>{});

  test_complement(layout, 1);
  test_complement(layout);
  test_complement(layout, 53);
  test_complement(layout, 128);
  }

  {
  auto layout = make_layout(12, 2);

  test_complement(layout, 1);
  test_complement(layout);
  test_complement(layout, 53);
  test_complement(layout, 128);
  }

  {
  auto layout = make_layout(make_shape(3,6),make_stride(_1{}, _3{}));

  test_complement(layout);
  }

  {
  auto layout = make_layout(make_shape(3,6),make_stride(_1{}, _9{}));

  test_complement(layout);
  }

  {
  auto layout = make_layout(make_shape(3,6),make_stride(_1{}, _10{}));

  test_complement(layout);
  }

  {
  auto layout = make_layout(make_shape(make_shape(2,2), make_shape(2,2)),
                            Stride<Stride<_1,_4>,Stride<_8,_32>>{});

  test_complement(layout);
  }

  {
  auto layout = make_layout(Int<64>{});

  test_complement(layout, make_shape(Int<32>{}, Int<4>{}, Int<4>{}));
  test_complement(layout, make_shape(Int<32>{}, Int<4>{}, 4));
  }
}
