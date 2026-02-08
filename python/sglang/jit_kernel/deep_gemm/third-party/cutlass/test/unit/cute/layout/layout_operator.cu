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

/*! \file
    \brief Unit tests Generic CuTe Layouts
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/matrix_coord.h"

// Cute includes
#include <cute/layout.hpp>
#include <cute/int_tuple.hpp>
#include <cute/swizzle.hpp>
#include <cute/layout_composed.hpp>
#include <cute/swizzle_layout.hpp>

using namespace cutlass;
using namespace cute;

namespace test {
namespace layout {

template <typename GenericLayout, typename Layout> 
  struct Testbed {


    Testbed() {}

    bool run() {
      GenericLayout generic_layout;
      Layout layout = Layout::packed({size<0>(generic_layout), size<1>(generic_layout)});

      for (int m = 0; m < size<0>(generic_layout); m++) {
        for (int n = 0; n < size<1>(generic_layout); n++) {
          if (generic_layout(m, n) != layout({m, n})) return false;
        }
      }

      return true;
    }
  };

}
}

//////////////////////////////////////////////////////////////////////////
//                      Test Generic CuTe Layouts
//////////////////////////////////////////////////////////////////////////

/// Canonical Layouts

TEST(GenericLayout, ColumnMajor) {
  using GenericLayout = cute::Layout<Shape<_8, _4>, Stride<_1, _8>>;
  using Layout = cutlass::layout::ColumnMajor;

  test::layout::Testbed<GenericLayout, Layout> testbed;

  EXPECT_TRUE(testbed.run());
}
//////////////////////////////////////////////////////////////////////////

TEST(GenericLayout, RowMajor) {
  using GenericLayout = cute::Layout<Shape<_8, _4>, Stride<_4, _1>>;
  using Layout = cutlass::layout::RowMajor;

  test::layout::Testbed<GenericLayout, Layout> testbed;

  EXPECT_TRUE(testbed.run());
}
//////////////////////////////////////////////////////////////////////////


/// Swizzle Shared Memory layouts

TEST(GenericLayout, RowMajorTensorOpMultiplicandCrosswise) {

  using GenericLayout = decltype(
        composition(
          Swizzle<3,3,3>{},
          Layout<Shape<_128, _64>, Stride<_64, _1>>{})
  );

  using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<cutlass::half_t>::value, 64>;

  test::layout::Testbed<GenericLayout, Layout> testbed;

  EXPECT_TRUE(testbed.run());
}
//////////////////////////////////////////////////////////////////////////

TEST(GenericLayout, ColumnMajorTensorOpMultiplicandCongruous) {

  using GenericLayout = decltype(
        composition(
          Swizzle<3,3,4>{},
          Layout<Shape<_128, _64>>{})
  );

  using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
    cutlass::sizeof_bits<cutlass::half_t>::value, 64>;


  test::layout::Testbed<GenericLayout, Layout> testbed;

  EXPECT_TRUE(testbed.run());
}
//////////////////////////////////////////////////////////////////////////
