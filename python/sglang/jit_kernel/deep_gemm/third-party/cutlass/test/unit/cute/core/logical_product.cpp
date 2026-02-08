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
#include <cute/tensor.hpp>

using namespace cute;

template <class LayoutA, class LayoutB>
void
test_logical_product(LayoutA const& layoutA,
                     LayoutB const& layoutB)
{
  auto layoutR = logical_product(layoutA, layoutB);

  CUTLASS_TRACE_HOST(shape(layoutA)  << " x " << shape(layoutB)  << "  =>  " << shape(layoutR) );
  CUTLASS_TRACE_HOST(stride(layoutA) << "   " << stride(layoutB) << "  =>  " << stride(layoutR));

  ASSERT_EQ(rank(layoutR), 2);
  ASSERT_TRUE(layoutA == layout<0>(layoutR));
  ASSERT_TRUE(compatible(layoutB, layout<1>(layoutR)));
}

TEST(CuTe_core, Logical_product)
{
  {
    auto vec  = Layout<_1,_0>{};
    auto tile = Layout<_1,_0>{};

    test_logical_product(vec, tile);
  }

  {
    auto vec  = Layout<_1,_1>{};
    auto tile = Layout<_1,_0>{};

    test_logical_product(vec, tile);
  }

  {
    auto vec  = Layout<_1,_0>{};
    auto tile = Layout<_1,_1>{};

    test_logical_product(vec, tile);
  }

  {
    auto vec  = Layout<_1,_1>{};
    auto tile = Layout<_1,_1>{};

    test_logical_product(vec, tile);
  }

  {
    auto vec  = Layout<_3,_1>{};
    auto tile = Layout<_4,_0>{};

    test_logical_product(vec, tile);
  }

  {
    auto vec  = Layout<_3,_0>{};
    auto tile = Layout<_4,_1>{};

    test_logical_product(vec, tile);
  }

  {
    auto vec  = Layout<_3,_0>{};
    auto tile = Layout<_4,_0>{};

    test_logical_product(vec, tile);
  }

  {
    auto vec  = Layout<_3,_2>{};
    auto tile = Layout<_4,_1>{};

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_3>{});
    auto tile = make_layout(Shape<_2,_4>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_2,_4>{});
    auto tile = make_layout(Shape<_3>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_8,Shape<_2,_2>>{});
    auto tile = make_layout(Shape<_4>{}, Stride<_2>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_2,_2>{});
    auto tile = make_layout(Shape<_3,_3>{}, Stride<_3,_1>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_3>{}, Stride<_32>{});
    auto tile = make_layout(Shape<_32>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_3>{}, Stride<_2>{});
    auto tile = make_layout(Shape<_4>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_3>{}, Stride<_32>{});
    auto tile = make_layout(Shape<_128>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_3>{}, Stride<_32>{});
    auto tile = make_layout(Shape<_8,_8>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<_3>{}, Stride<_32>{});
    auto tile = make_layout(Shape<_8,_8>{}, Stride<_8,_1>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<Shape<_4,_2>>{}, Stride<Stride<_1,_16>>{});
    auto tile = make_layout(Shape<_4,_4>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<Shape<_4,_2>>{}, Stride<Stride<_1,_16>>{});
    auto tile = make_layout(Shape<_4,_2>{}, Stride<_2,_1>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<Shape<_2,_2>,Shape<_2, _2>>{},
                           Stride<Stride<_1,_4>,Stride<_8,_32>>{});
    auto tile = make_layout(Shape<_2,_2>{}, Stride<_1,_2>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape<Shape<_2,_2>,Shape<_2, _2>>{},
                           Stride<Stride<_1,_4>,Stride<_8,_32>>{});
    auto tile = make_layout(Shape<_2,_2>{},
                            Stride<_2,_1>{});

    test_logical_product(vec, tile);
  }

  {
    auto vec = make_layout(Shape <Shape <_4,_6>>{},
                           Stride<Stride<_1,_6>>{});
    auto tile = make_layout(Shape <_3>{},
                            Stride<_1>{});

    test_logical_product(vec, tile);
  }
}
