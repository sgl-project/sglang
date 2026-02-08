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

#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>  // cute::composition
#include <cute/swizzle.hpp>          // cute::Swizzle
#include <cute/swizzle_layout.hpp>   // cute::composition
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;


template <class LayoutA, class LayoutB>
void
test_composition(LayoutA const& layoutA,
                 LayoutB const& layoutB)
{
  auto layoutR = composition(layoutA, layoutB);

  CUTLASS_TRACE_HOST("test_composition()");
  CUTLASS_TRACE_HOST(layoutA << " o " << layoutB);
  CUTLASS_TRACE_HOST("  =>  ");
  CUTLASS_TRACE_HOST(layoutR);

  // Test that layout B is compatible with layout R
  EXPECT_TRUE(compatible(layoutB, layoutR));

  // Test that R(c) = A(B(c)) for all coordinates c in layoutB
  for (int c = 0; c < size(layoutB); ++c) {
    EXPECT_EQ(layoutR(c), layoutA(layoutB(c)));
  }
}


TEST(CuTe_core, Composition)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("COMPOSITION"                    );
  CUTLASS_TRACE_HOST("-------------------------------");

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("Simple tests"                   );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto a = Layout<_1,_0>{};
    auto b = Layout<_1,_0>{};

    test_composition(a, b);
  }

  {
    auto a = Layout<_1,_0>{};
    auto b = Layout<_1,_1>{};

    test_composition(a, b);
  }

  {
    auto a = Layout<_1,_1>{};
    auto b = Layout<_1,_0>{};

    test_composition(a, b);
  }

  {
    auto a = Layout<_1,_1>{};
    auto b = Layout<_1,_1>{};

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{});
    auto b = make_layout(Shape<_4>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{}, Stride<_2>{});
    auto b = make_layout(Shape<_4>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{}, Stride<_0>{});
    auto b = make_layout(Shape<_4>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{});
    auto b = make_layout(Shape<_4>{}, Stride<_0>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{});
    auto b = make_layout(Shape<_1>{}, Stride<_0>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{});
    auto b = make_layout(Shape<_2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{}, Stride<_2>{});
    auto b = make_layout(Shape<_2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{});
    auto b = make_layout(Shape<_2>{}, Stride<_2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{}, Stride<_2>{});
    auto b = make_layout(Shape<_2>{}, Stride<_2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3>{});
    auto b = make_layout(Shape<_12>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_12>{});
    auto b = make_layout(Shape<_4,_3>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_12>{}, Stride<_2>{});
    auto b = make_layout(Shape<_4,_3>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_12>{});
    auto b = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_12>{}, Stride<_2>{});
    auto b = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_12>{});
    auto b = make_layout(Shape<_2,_3>{}, Stride<_2,_4>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3>{});
    auto b = make_layout(Shape<_4,_3>{});

    test_composition(a, b);
  }

  {
   auto a = make_layout(Shape<_4,_3>{});
   auto b = make_layout(Shape<_6>{});

   test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3>{});
    auto b = make_layout(Shape<_6>{}, Stride<_2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3>{});
    auto b = make_layout(Shape<_6,_2>{}, Stride<_2,_1>{});

    test_composition(a, b);
  }

  {
   auto a = make_layout(Shape<_4,_3>{});
   auto b = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});

   test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});
    auto b = make_layout(Shape<_4,_3>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});
    auto b = make_layout(Shape<_12>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});
    auto b = make_layout(Shape<_6>{}, Stride<_2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});
    auto b = make_layout(Shape<_6,_2>{}, Stride<_2,_1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_8,_8>{});
    auto b = make_layout(Shape <Shape <_2, _2,_2>, Shape <_2,_2, _2>>{},
                         Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>{});
    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_8,_8>{}, Stride<_8,_1>{});
    auto b = make_layout(Shape <Shape <_2, _2,_2>, Shape <_2,_2, _2>>{},
                         Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<Shape<_4,_2>>{}, Stride<Stride<_1,_16>>{});
    auto b = make_layout(Shape<_4,_2>{}, Stride<_2,_1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_2,_2>{}, Stride<_2,_1>{});
    auto b = make_layout(Shape<_2,_2>{}, Stride<_2,_1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_8,_2>{});
    auto b = make_layout(Shape<_2,_2,_2>{}, Stride<_2,_8,_1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_8,_2>{}, Stride<_2,_8,_1>{});
    auto b = make_layout(Shape<_2,_2,_2>{}, Stride<_1,_8,_2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_8,_2>{}, Stride<_2,_8,_1>{});
    auto b = make_layout(Shape<_4,_2,_2>{}, Stride<_2,_8,_1>{});

    test_composition(a, b);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("Dynamic shapes/strides"         );
  CUTLASS_TRACE_HOST("-------------------------------");


  {
    auto a = make_layout(12, 1);
    auto b = make_layout(_4{}, _1{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(12, 1);
    auto b = make_layout(_4{}, 1);

    test_composition(a, b);
  }

  {
    auto a = make_layout(12, _1{});
    auto b = make_layout(_4{}, 1);

    test_composition(a, b);
  }

  {
    auto a = make_layout(12, _1{});
    auto b = make_layout(_4{}, _1{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(make_shape(12,3), make_stride(1,24));
    auto b = make_layout(Shape<_4>{}, Stride<_1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(16, 2);
    auto b = make_layout(4, 2);

    test_composition(a, b);
  }

  {
    auto a = make_layout(make_shape(128,24,5), make_stride(1,128,3072));
    auto b = make_layout(64, 2);

    test_composition(a, b);
  }

  {
    auto a = make_layout(make_shape(128,24,5), make_stride(1,128,3072));
    auto b = make_layout(480, Int<32>{});

    test_composition(a, b);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("cosize(b) > size(a) and divisibility");
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto a = make_layout(Shape<_1>{}, Stride<_0>{});
    auto b = make_layout(Shape<_4>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_1>{}, Stride<_1>{});
    auto b = make_layout(Shape<_4>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{});
    auto b = make_layout(Shape<_4>{}, Stride<_2>{});

    test_composition(a, b);
  }

  // Last mode gets extended
  {
    auto a = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});
    auto b = make_layout(Shape<_24>{});

    test_composition(a, b);
  }

  // Last mode extension even without last mode divisibility
  {
    auto a = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});
    auto b = make_layout(Shape<_8>{});

    test_composition(a, b);
  }

  // Capping a Layout with 1:0 extends in stride-0
  {
    auto a = make_layout(Shape<_4,_3,_1>{}, Stride<_3,_1,_0>{});
    auto b = make_layout(Shape<_24>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_3,_1>{}, Stride<_3,_1,_0>{});
    auto b = make_layout(Shape<_4>{});

    test_composition(a, b);
  }

  // Pre-coalesced LHS
  {
    auto a = make_layout(Shape<_4,_6,_8>{}, Stride<_1,_4,_7>{});
    auto b = make_layout(_6{}, _1{});

    test_composition(a, b);
  }

  // Mid-layout truncation
  {
    auto a = make_layout(Shape<_4,_6,_8,_10>{}, Stride<_2,_3,_5,_7>{});
    auto b = make_layout(_6{}, _12{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_8,_8>{}, Stride<_8,_1>{});
    auto b = make_layout(_2{}, _3{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_8,_8>{}, Stride<_8,_1>{});
    auto b = make_layout(_3{}, _3{});

    test_composition(a, b);
  }

  // Should fail to a static divisibility condition
  // {
  //   auto a = make_layout(Shape<_8,_8>{}, Stride<_8,_1>{});
  //   auto b = make_layout(_4{}, _3{});
  //   test_composition(a, b);
  // }

  {
    auto a = make_layout(3, _1{});
    auto b = make_layout(_4{}, _1{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(make_shape(48,24,5), make_stride(_1{},128,3072));
    auto b = make_layout(32, Int<1>{});

    test_composition(a, b);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("Swizzle composition"            );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto a = Layout<Shape<_8,_8>, Stride<_8,_1>>{};
    auto b = composition(Swizzle<2,0,-3>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{});

    test_composition(a, b);
  }

  {
    auto a = composition(Swizzle<2,0, 3>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{});
    auto b = composition(Swizzle<2,0,-3>{}, Layout<Shape<_8,_8>, Stride<_8,_1>>{});

    test_composition(a, b);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("BETA: Negative strides"         );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto a = make_layout(Shape<_4>{}, Stride<_m1>{});
    auto b = make_layout(Shape<_4>{}, Stride<_1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{}, Stride<_1>{});
    auto b = make_layout(Shape<_4>{}, Stride<_m1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{}, Stride<_m1>{});
    auto b = make_layout(Shape<_4>{}, Stride<_m1>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4>{}, Stride<_1>{});
    auto b = make_layout(Shape<_4>{}, Stride<_m2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_4>{}, Stride<_m1,_1>{});
    auto b = make_layout(Shape<_2,_4,_2>{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_4,_4>{}, Stride<_m1,_1>{});
    auto b = make_layout(Shape<_2,_4,_2>{}, Stride<_1,_4,_2>{});

    test_composition(a, b);
  }

  // The SM80 fp64 MMA NT problem
  {
    auto a = make_layout(Shape<_1,Shape<_2,_4>>{}, Stride<_0,Stride<_m1,_512>>{});
    auto b = make_layout(_2{}, _m1{});

    test_composition(a, b);
  }

  {
    auto a = make_layout(Shape<_1,Shape<_2,_4>>{}, Stride<_0,Stride<_m1,_512>>{});
    auto b = make_layout(_4{}, _m1{});

    test_composition(a, b);
  }

  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("BETA: Tuple strides"            );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
   auto a = make_layout(Shape<_4,_4>{}, Stride<_4,_1>{});
   auto b = make_layout(Shape<_4,_4>{}, Stride<E<1>,E<0>>{});

   test_composition(a, b);
  }

  {
   auto a = make_layout(Shape<_4,Shape<_2,_3>>{}, Stride<_6,Stride<_3,_1>>{});
   auto b = make_layout(Shape<_2,_4>{}, Stride<E<1,1>,E<0>>{});

   test_composition(a, b);
  }
}
