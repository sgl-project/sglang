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
    \brief Tests for basic uint128 functionality
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(uint128_t, host_arithmetic) {
  using T = cutlass::uint128_t;

  // only low 64bit
  for (uint64_t i = 0; i < 1024; ++i) {
    for (uint64_t j = 0; j < 1024; ++j) {
      T x = i;
      T y = j;

      EXPECT_TRUE(static_cast<uint64_t>(x + y) == (i + j));
    }
  }

  // carry overflow for low uint64_t 
  {
    for (uint64_t i = 0; i < 1024; ++i) {
      T x = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFF);
      T y = i + 1;

      T z = x + y;

      EXPECT_EQ(z.hilo_.hi, static_cast<uint64_t>(0x1));
      EXPECT_EQ(z.hilo_.lo, i);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Device
//
/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void uint128_add_operator(cutlass::uint128_t *output, cutlass::uint128_t const *input, cutlass::uint128_t base, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    output[tid] = input[tid] + base;
  }
}

TEST(uint128_t, device_arithmetic) {
  using T = cutlass::uint128_t;

  int const N = 1024;

  cutlass::HostTensor<T, cutlass::layout::RowMajor> input({N, 1});
  cutlass::HostTensor<T, cutlass::layout::RowMajor> sum({N, 1});

  for (int i = 0; i < N; ++i) {
    input.at({i, 0}) = static_cast<uint64_t>(i + 1);
  }

  T b = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFF);

  input.sync_device();

  uint128_add_operator<<< dim3(1,1), dim3(N, 1) >>>(sum.device_data(), input.device_data(), b, N);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch error.";

  sum.sync_host();

  for (int i = 0; i < N; ++i) {
    T got = sum.at({i, 0});
    uint64_t expected_hi = static_cast<uint64_t>(0x1);
    uint64_t expected_lo = static_cast<uint64_t>(i);

    EXPECT_EQ(got.hilo_.hi, expected_hi);
    EXPECT_EQ(got.hilo_.lo, expected_lo);
  }
}
