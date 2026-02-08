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
#include "cutlass/core_io.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convert_bf16_f32(cutlass::bfloat16_t *output, float const *input, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    output[tid] = static_cast<cutlass::bfloat16_t>(input[tid]);
  }
}

__global__ void convert_and_pack_bf16(cutlass::bfloat16_t *output, float const *input, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid * 2 < N) {

    cutlass::NumericArrayConverter<cutlass::bfloat16_t, float, 2> convert;

    cutlass::Array<cutlass::bfloat16_t, 2> *dst_ptr = 
      reinterpret_cast<cutlass::Array<cutlass::bfloat16_t, 2> *>(output + tid * 2);

    cutlass::Array<float, 2> const *src_ptr = 
      reinterpret_cast<cutlass::Array<float, 2> const *>(input + tid * 2);

    *dst_ptr = convert(*src_ptr);
  } 
}

TEST(bfloat16_t, device_conversion) {
  using T = cutlass::bfloat16_t;
  using S = float;

  int const N = 256;

  cutlass::HostTensor<T, cutlass::layout::RowMajor> destination({N, 1});
  cutlass::HostTensor<S, cutlass::layout::RowMajor> source({N, 1});

  for (int i = 0; i < N; ++i) {
    source.at({i, 0}) = float(i - 128);
    destination.at({i, 0}) = T(0);
  }

  source.sync_device();
  destination.sync_device();

  convert_bf16_f32<<< dim3(1,1), dim3(N, 1) >>>(destination.device_data(), source.device_data(), N);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch error.";

  destination.sync_host();

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    T got = destination.at({i, 0});
    S expected = source.at({i, 0});

    if (S(got) != expected) {
      ++errors;
      if (errors < 10) {
        std::cerr << "Basic conversion error - [" << i << "] - got " << got << ", expected " << expected << "\n";
      }
    }

    destination.at({i, 0}) = T(0);
  }

  destination.sync_device();
  
  convert_and_pack_bf16<<< dim3(1,1), dim3(N, 1) >>>(destination.device_data(), source.device_data(), N);
  
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch error.";
  
  destination.sync_host();

  for (int i = 0; i < N; ++i) {
    T got = destination.at({i, 0});
    S expected = source.at({i, 0});

    if (S(got) != expected) {
      ++errors;
      if (errors < 10) {
        std::cerr << "Convert and pack error - [" << i << "] - got " << got << ", expected " << expected << "\n";
      }
    }
  }

  EXPECT_EQ(errors, 0);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(bfloat16_t, host_conversion) {
  for (int i = -128; i < 128; ++i) {
    float f = static_cast<float>(i);

    cutlass::bfloat16_t x = static_cast<cutlass::bfloat16_t>(i);
    cutlass::bfloat16_t y = static_cast<cutlass::bfloat16_t>(f);

    EXPECT_TRUE(static_cast<int>(x) == i);
    EXPECT_TRUE(static_cast<float>(y) == f);
  }

  // Try out default-ctor (zero initialization of primitive proxy type)
  EXPECT_TRUE(cutlass::bfloat16_t() == 0.0_bf16);

  // Try out user-defined literals
  EXPECT_TRUE(cutlass::bfloat16_t(7) == 7_bf16);
  EXPECT_TRUE(7 == static_cast<int>(7_bf16));
}

TEST(bfloat16_t, host_arithmetic) {

  for (int i = -100; i < 100; ++i) {
    for (int j = -100; j < 100; ++j) {

      cutlass::bfloat16_t x = static_cast<cutlass::bfloat16_t>(i);
      cutlass::bfloat16_t y = static_cast<cutlass::bfloat16_t>(j);

      EXPECT_TRUE(static_cast<int>(x + y) == (i + j));
    }
  }
}

TEST(bfloat16_t, host_round) {
  
  struct {
    uint32_t f32_bits;
    uint16_t expected;
  } tests[] = {
    {0x40040000, 0x4004},  // M=0, R=0, S=0 => rtz
    {0x40048000, 0x4004},  // M=0, R=1, S=0 => rtz
    {0x40040001, 0x4004},  // M=0, R=1, S=1 => +inf
    {0x4004c000, 0x4005},  // M=0, R=1, S=1 => +inf
    {0x4004a000, 0x4005},  // M=0, R=1, S=1 => +inf
    {0x40050000, 0x4005},  // M=1, R=0, S=0 => rtz
    {0x40054000, 0x4005},  // M=1, R=0, S=1 => rtz
    {0x40058000, 0x4006},  // M=1, R=1, S=0 => +inf
    {0x40058001, 0x4006},  // M=1, R=1, S=1 => +inf
    {0x7f800000, 0x7f80},  // +inf
    {0xff800000, 0xff80},  // -inf
    {0x7fffffff, 0x7fff},  // canonical NaN
    {0x7ff00001, 0x7fff},  // NaN -> canonical NaN
    {0xfff00010, 0x7fff},  // Nan -> canonical NaN
    {0, 0}
  };
  
  bool running = true;
  for (int i = 0; running; ++i) {

    float f32 = reinterpret_cast<float const &>(tests[i].f32_bits);

    cutlass::bfloat16_t bf16 = cutlass::bfloat16_t(f32);

    bool passed = (tests[i].expected == bf16.raw());
    
    EXPECT_TRUE(passed)
      << "Error - convert(f32: 0x" << std::hex << tests[i].f32_bits 
      << ") -> 0x" << std::hex << tests[i].expected << "\ngot: 0x" << std::hex << bf16.raw();

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
