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
/*! \file
    \brief Statically sized array of elements that accommodates all CUTLASS-supported numeric types
           and is safe to use in a union.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(half_t, host_conversion) {
  for (int i = -1024; i < 1024; ++i) {
    float f = static_cast<float>(i);

    cutlass::half_t x = static_cast<cutlass::half_t>(i);
    cutlass::half_t y = static_cast<cutlass::half_t>(f);

    EXPECT_TRUE(static_cast<int>(x) == i);
    EXPECT_TRUE(static_cast<float>(y) == f);
  }

  // Try out default-ctor (zero initialization of primitive proxy type)
  EXPECT_TRUE(cutlass::half_t() == 0.0_hf);

  // Try out user-defined literals
  EXPECT_TRUE(cutlass::half_t(7) == 7_hf);
  EXPECT_TRUE(7 == static_cast<int>(7_hf));
}

TEST(half_t, host_arithmetic) {

  for (int i = -100; i < 100; ++i) {
    for (int j = -100; j < 100; ++j) {

      cutlass::half_t x = static_cast<cutlass::half_t>(i);
      cutlass::half_t y = static_cast<cutlass::half_t>(j);

      EXPECT_TRUE(static_cast<int>(x + y) == (i + j));
    }
  }

  for (int i = -6; i < 6; ++i) {
    for (int j = -6; j < 6; ++j) {

      cutlass::half_t x = static_cast<cutlass::half_t>(i);
      cutlass::half_t y = static_cast<cutlass::half_t>(j);

      EXPECT_TRUE(static_cast<int>(x * y) == (i * j));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
