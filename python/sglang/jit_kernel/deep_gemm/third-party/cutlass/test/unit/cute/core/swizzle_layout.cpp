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

#include <cute/tensor_impl.hpp>
#include <cute/swizzle_layout.hpp>

template <class SwLayout>
void
test_swizzle_2d(SwLayout const& sw_layout)
{
  using namespace cute;

  auto sw_tensor = make_tensor(counting_iterator<int>{0}, sw_layout);

  //print_tensor(sw_tensor);

  // Dynamic slicing
  for (int i = 0; i < size<0>(sw_tensor); ++i) {
    auto sliced_tensor = sw_tensor(i,_);
    //printf("sw_tensor(%d,_) => ", int(i)); print(sliced_tensor); printf("\n");
    for (int j = 0; j < size<1>(sw_tensor); ++j) {
      EXPECT_EQ(sw_tensor(i,j), sliced_tensor(j));
    }
  }

  // Static slicing
  cute::for_each(make_int_sequence<size<0>(sw_tensor)>{}, [&] (auto i) {
    auto sliced_tensor = sw_tensor(i,_);
    //printf("sw_tensor(%d,_) => ", int(i)); print(sliced_tensor); printf("\n");
    // If sw_tensor is static, then sliced_tensor should be too
    auto sw_tensor_2 = sw_tensor;
    static_assert(is_static<decltype(layout(sliced_tensor))>::value || not is_static<decltype(layout(sw_tensor_2))>::value);
    cute::for_each(make_int_sequence<size(sliced_tensor)>{}, [&] (auto j) {
      EXPECT_EQ(sw_tensor(i,j), sliced_tensor(j));
    });
  });

  // Dynamic slicing
  for (int j = 0; j < size<1>(sw_tensor); ++j) {
    auto sliced_tensor = sw_tensor(_,j);
    //printf("sw_tensor(_,%d) => ", int(j)); print(sliced_tensor); printf("\n");
    for (int i = 0; i < size<0>(sw_tensor); ++i) {
      EXPECT_EQ(sw_tensor(i,j), sliced_tensor(i));
    }
  }

  // Static slicing
  cute::for_each(make_int_sequence<size<1>(sw_tensor)>{}, [&] (auto j) {
    auto sliced_tensor = sw_tensor(_,j);
    //printf("sw_tensor(_,%d) => ", int(j)); print(sliced_tensor); printf("\n");
    // If sw_tensor is static, then sliced_tensor should be too
    auto sw_tensor_2 = sw_tensor;
    static_assert(is_static<decltype(layout(sliced_tensor))>::value || not is_static<decltype(layout(sw_tensor_2))>::value);
    cute::for_each(make_int_sequence<size(sliced_tensor)>{}, [&] (auto i) {
      EXPECT_EQ(sw_tensor(i,j), sliced_tensor(i));
    });
  });
}

TEST(CuTe_core, SwizzleLayout)
{
  using namespace cute;

  {
  auto sw_layout = composition(Swizzle<3,0,3>{},
                               Layout<Shape <_8,_8>,
                                      Stride<_8,_1>>{});
  test_swizzle_2d(sw_layout);
  }

  {
  auto sw_layout = composition(Swizzle<3,0,-3>{},
                               Layout<Shape <_8,_8>,
                                      Stride<_8,_1>>{});
  test_swizzle_2d(sw_layout);
  }

  {
  auto sw_layout = composition(Swizzle<2,1,3>{},
                               Layout<Shape <Shape < _2,_2,_2>,Shape <_2,_2, _2>>,
                                      Stride<Stride<_32,_2,_8>,Stride<_4,_1,_16>>>{});
  test_swizzle_2d(sw_layout);
  }
}
