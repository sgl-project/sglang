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
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(tfloat32_t, host_conversion) {
  for (int i = -1024; i < 1024; ++i) {
    float f = static_cast<float>(i);

    cutlass::tfloat32_t x = static_cast<cutlass::tfloat32_t>(i);
    cutlass::tfloat32_t y = static_cast<cutlass::tfloat32_t>(f);

    EXPECT_TRUE(static_cast<int>(x) == i);
    EXPECT_TRUE(static_cast<float>(y) == f);
  }

  // Try out default-ctor (zero initialization of primitive proxy type)
  EXPECT_TRUE(cutlass::tfloat32_t() == 0.0_tf32);

  // Try out user-defined literals
  EXPECT_TRUE(cutlass::tfloat32_t(7) == 7_tf32);
  EXPECT_TRUE(7 == static_cast<int>(7_tf32));
}

TEST(tfloat32_t, host_arithmetic) {

  for (int i = -100; i < 100; ++i) {
    for (int j = -100; j < 100; ++j) {

      cutlass::tfloat32_t x = static_cast<cutlass::tfloat32_t>(i);
      cutlass::tfloat32_t y = static_cast<cutlass::tfloat32_t>(j);

      EXPECT_TRUE(static_cast<int>(x + y) == (i + j));
    }
  }
}

TEST(tfloat32_t, host_round_nearest) {
  
  struct {
    uint32_t f32_bits;
    uint32_t expected;
  } tests[] = {
    {0x40000000, 0x40000000},  // M=0, R=0, S=0 => rtz
    {0x40001000, 0x40000000},  // M=0, R=1, S=0 => rtz
    {0x40000001, 0x40000000},  // M=0, R=0, S=1 => rtz
    {0x40001001, 0x40002000},  // M=0, R=1, S=1 => +inf
    {0x40002000, 0x40002000},  // M=1, R=0, S=0 => rtz
    {0x40002001, 0x40002000},  // M=1, R=0, S=1 => rtz
    {0x40003000, 0x40004000},  // M=1, R=1, S=0 => +inf
    {0x40003001, 0x40004000},  // M=1, R=1, S=1 => +inf
    {0x7f800000, 0x7f800000},  // +inf
    {0xff800000, 0xff800000},  // -inf
    {0x7fffffff, 0x7fffffff},  // canonical NaN to canonical NaN
    {0x7f800001, 0x7fffffff},  // NaN to canonical NaN
    {0xff800001, 0x7fffffff},  // NaN to canonical NaN
    {0, 0}
  };
  
  bool running = true;
  for (int i = 0; running; ++i) {

    float f32 = reinterpret_cast<float const &>(tests[i].f32_bits);

    cutlass::NumericConverter<
      cutlass::tfloat32_t, 
      float, 
      cutlass::FloatRoundStyle::round_to_nearest> converter;

    cutlass::tfloat32_t tf32 = converter(f32);
    
    // note, we must explicitly truncate the low-order bits since they are not defined in TF32.
    if (cutlass::isfinite(tf32)) {
      tf32.storage &= 0xffffe000;
    }

    bool passed = (tests[i].expected == tf32.raw());
    
    EXPECT_TRUE(passed)
      << "Error - convert(f32: 0x" << std::hex << tests[i].f32_bits 
      << ") -> 0x" << std::hex << tests[i].expected << "\ngot: 0x" << std::hex << tf32.raw();

    if (!tests[i].f32_bits) {
      running = false;
    }
  }
}

namespace test {
namespace core {

__global__ void convert_tf32_half_ulp(cutlass::tfloat32_t *out, float const *in) {

  cutlass::NumericConverter<
    cutlass::tfloat32_t, 
    float,
    cutlass::FloatRoundStyle::round_half_ulp_truncate> convert;

  *out = convert(*in);
}

}
}


TEST(tfloat32_t, host_round_half_ulp) {
  
  struct {
    uint32_t f32_bits;
    uint32_t expected;
  } tests[] = {
    {0x40001fff, 0x40002000},
    {0x40000000, 0x40000000},  // M=0, R=0, S=0 => rtz
    {0x40001000, 0x40002000},  // M=0, R=1, S=0 => rtz  - this difers from RNE
    {0x40000001, 0x40000000},  // M=0, R=0, S=1 => rtz
    {0x40001001, 0x40002000},  // M=0, R=1, S=1 => +inf
    {0x40002000, 0x40002000},  // M=1, R=0, S=0 => rtz
    {0x40002001, 0x40002000},  // M=1, R=0, S=1 => rtz
    {0x40003000, 0x40004000},  // M=1, R=1, S=0 => +inf
    {0x40003001, 0x40004000},  // M=1, R=1, S=1 => +inf
    {0x7f800000, 0x7f800000},  // +inf
    {0xff800000, 0xff800000},  // -inf
    {0x7fffffff, 0x7fffffff},  // canonical NaN to canonical NaN
    {0x7f800001, 0x7f800001},  // NaN to NaN
    {0xff800001, 0xff800001},  // NaN to NaN
    {0, 0}
  };

  cutlass::NumericConverter<
    cutlass::tfloat32_t, 
    float,
    cutlass::FloatRoundStyle::round_half_ulp_truncate> convert;
  
  bool running = true;
  for (int i = 0; running; ++i) {

    float f32 = reinterpret_cast<float const &>(tests[i].f32_bits);

    cutlass::tfloat32_t tf32 = convert(f32);

    // note, for this test, we must explicitly truncate the low-order bits since they are not 
    // defined in TF32.
    if (cutlass::isfinite(tf32)) {
      tf32.storage &= 0xffffe000;
    }

    bool passed = (tests[i].expected == tf32.raw());
    
    EXPECT_TRUE(passed)
      << "Error - convert(f32: 0x" << std::hex << tests[i].f32_bits 
      << ") -> 0x" << std::hex << tests[i].expected << "\ngot: 0x" << std::hex << tf32.raw();

    if (!tests[i].f32_bits) {
      running = false;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Device
//
/////////////////////////////////////////////////////////////////////////////////////////////////
