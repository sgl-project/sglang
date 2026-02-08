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


#include <cute/tensor.hpp>

using namespace cute;

template <class Layout>
void
test_coalesce(Layout const& layout)
{
  auto coalesce_layout = coalesce(layout);

  CUTLASS_TRACE_HOST(shape (layout) << "  =>  " << shape (coalesce_layout));
  CUTLASS_TRACE_HOST(stride(layout) << "      " << stride(coalesce_layout));

  CUTE_STATIC_ASSERT_V(depth(coalesce_layout) <= Int<1>{});

  ASSERT_EQ(size(coalesce_layout),  size(layout));

  for (int i = 0; i < size(layout); ++i) {
    EXPECT_EQ(coalesce_layout(i), layout(i));
  }
}

TEST(CuTe_core, Coalesce)
{
  {
  auto layout = make_layout(Int<1>{}, Int<0>{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(Int<1>{}, Int<1>{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, Int<4>{}));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, Int<4>{}, Int<6>{}));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape (Int<2>{}, Int<1>{}, Int<6>{}),
                            make_stride(Int<1>{}, Int<6>{}, Int<2>{}));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape (Int<2>{}, Int<1>{}, Int<6>{}),
                            make_stride(Int<1>{},        7, Int<2>{}));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape (Int<2>{}, Int<1>{}, Int<6>{}),
                            make_stride(Int<4>{},        7, Int<8>{}));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(2, Int<4>{}, Int<6>{}));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, 4, Int<6>{}));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, Int<4>{}, 6));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, Int<4>{}), GenRowMajor{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, Int<4>{}, Int<6>{}), GenRowMajor{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(2, Int<4>{}, Int<6>{}), GenRowMajor{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, 4, Int<6>{}), GenRowMajor{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, Int<4>{}, 6), GenRowMajor{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, Int<1>{}, Int<3>{}), GenRowMajor{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, 1, Int<3>{}), GenRowMajor{});

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, 1, Int<3>{}), make_stride(Int<2>{}, 4, Int<4>{}));

  test_coalesce(layout);
  }

  {
  auto layout = make_layout(make_shape(Int<2>{}, 1, Int<3>{}), make_stride(Int<2>{}, Int<0>{}, Int<4>{}));

  test_coalesce(layout);
  }

  {
  auto layout = Layout<Shape<Shape<_2,_2>,Shape<_2, _2>>,
                       Stride<Stride<_1,_4>,Stride<_8,_32>>>{};

  test_coalesce(layout);
  }
}
