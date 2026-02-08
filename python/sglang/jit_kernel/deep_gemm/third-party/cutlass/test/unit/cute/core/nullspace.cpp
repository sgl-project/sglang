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
#include <iostream>

#include <cute/tensor.hpp>

using namespace cute;

template <class Layout, class KerLayout>
void
test_postconditions(Layout const& layout, KerLayout const& ker_layout)
{
  EXPECT_EQ(size(ker_layout), size(layout) / size(filter(layout)));

  for (int i = 0; i < size(ker_layout); ++i) {
    //printf("%3d: %3d  %3d\n", i, int(ker_layout(i)), int(layout(ker_layout(i))));
    EXPECT_EQ(layout(ker_layout(i)), 0);
  }
}

template <class Layout>
void
test_nullspace(Layout const& layout)
{
  auto ker_layout = nullspace(layout);

  CUTLASS_TRACE_HOST("ker(" << layout << ")\n" << "  =>  \n" << ker_layout);
  CUTLASS_TRACE_HOST("Composition: " << coalesce(composition(layout, ker_layout)) << std::endl);

  test_postconditions(layout, ker_layout);
}

TEST(CuTe_core, Layout_nullspace)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("NULLSPACE"                      );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto layout = Layout<Shape<_2,_2,_2>,Stride<_0,_0,_0>>{};

    test_nullspace(layout);
  }

  {
    auto layout = Layout<Shape<_7,_5,_16>,Stride<_0,_0,_0>>{};

    test_nullspace(layout);
  }

  {
    auto layout = Layout<Shape<_2,_2,_2>,Stride<_1,_0,_2>>{};

    test_nullspace(layout);
  }

  {
    auto layout = Layout<Shape<_7,_5,_16>,Stride<_3,_1,_0>>{};

    test_nullspace(layout);
  }
}
