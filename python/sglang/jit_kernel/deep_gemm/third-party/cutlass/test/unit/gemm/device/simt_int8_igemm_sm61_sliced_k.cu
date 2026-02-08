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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

////////////////////////////////////////////////////////////////////////////////

TEST(SM61_Device_Gemm_s8n_s8t_simt_op_dp4a_sliced_k, 32x32x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::ColumnMajor,
    int8_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM61_Device_Gemm_s8n_s8t_simt_op_dp4a_sliced_k, 32x64x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::ColumnMajor,
    int8_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM61_Device_Gemm_s8t_s8n_simt_op_dp4a_sliced_k, 32x32x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM61_Device_Gemm_s8t_s8n_simt_op_dp4a_sliced_k, 32x64x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM61_Device_Gemm_s8t_s8t_simt_op_dp4a_sliced_k, 32x32x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM61_Device_Gemm_s8t_s8t_simt_op_dp4a_sliced_k, 32x64x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::RowMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM61_Device_Gemm_s8n_s8n_simt_op_dp4a_sliced_k, 32x32x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::ColumnMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM61_Device_Gemm_s8n_s8n_simt_op_dp4a_sliced_k, 32x64x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::ColumnMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm61,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}
