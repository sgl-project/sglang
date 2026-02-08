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

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed.h"

////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 64x64x32_32x32x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 10>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 32x128x32_32x32x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32, 128, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 32x256x32_32x64x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32, 256, 32>,
      cutlass::gemm::GemmShape<32, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 128x32x32_32x32x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 32, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 256x32x32_64x32x32) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 32, 32>,
      cutlass::gemm::GemmShape<64, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 32x128x64_32x32x64) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32, 128, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 32x256x64_32x64x64) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32, 256, 64>,
      cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 128x32x64_32x32x64) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 32, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 256x32x64_64x32x64) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 32, 64>,
      cutlass::gemm::GemmShape<64, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}
TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 16x128x64_16x32x64) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<16, 128, 64>,
      cutlass::gemm::GemmShape<16, 32, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 16x256x64_16x64x64) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<16, 256, 64>,
      cutlass::gemm::GemmShape<16, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 128x16x64_32x16x64) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 16, 64>,
      cutlass::gemm::GemmShape<32, 16, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 64 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 5>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM80_Device_Gemm_f16n_f16t_f16t_tensor_op_f32, 256x16x64_64x16x64) {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 16, 64>,
      cutlass::gemm::GemmShape<64, 16, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 64 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}
////////////////////////////////////////////////////////////////////////////////

#endif  // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////
