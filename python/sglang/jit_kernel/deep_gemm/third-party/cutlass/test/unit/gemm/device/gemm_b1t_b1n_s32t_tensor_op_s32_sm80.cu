/**************************************************************************************************
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

#if defined(CUTLASS_ARCH_MMA_B1_XOR_SM80_ENABLED)

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 128x256x1024_64x64x1024, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 256, 1024>,
      cutlass::gemm::GemmShape<64, 64, 1024>,
      cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 256x128x1024_64x64x1024, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 1024>,
      cutlass::gemm::GemmShape<64, 64, 1024>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 128x128x1024_64x64x1024, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 1024>,
      cutlass::gemm::GemmShape<64, 64, 1024>,
      cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 256x64x1024_64x64x1024, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 64, 1024>,
      cutlass::gemm::GemmShape<64, 64, 1024>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 64x256x1024_64x64x1024, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 256, 1024>,
      cutlass::gemm::GemmShape<64, 64, 1024>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 64x128x1024_32x64x1024, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 1024>,
      cutlass::gemm::GemmShape<32, 64, 1024>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 128x64x1024_64x32x1024, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 64, 1024>,
      cutlass::gemm::GemmShape<64, 32, 1024>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 64x64x1024_32x32x1024, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 1024>,
      cutlass::gemm::GemmShape<32, 32, 1024>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 128x256x512_64x64x512, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 256, 512>,
      cutlass::gemm::GemmShape<64, 64, 512>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 256x128x512_64x64x512, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 512>,
      cutlass::gemm::GemmShape<64, 64, 512>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 128x128x512_64x64x512, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 512>,
      cutlass::gemm::GemmShape<64, 64, 512>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 256x64x512_64x64x512, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 64, 512>,
      cutlass::gemm::GemmShape<64, 64, 512>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 64x256x512_64x64x512, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 256, 512>,
      cutlass::gemm::GemmShape<64, 64, 512>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 64x128x512_32x64x512, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 512>,
      cutlass::gemm::GemmShape<32, 64, 512>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 128x64x512_64x32x512, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 64, 512>,
      cutlass::gemm::GemmShape<64, 32, 512>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

CUTLASS_TEST_L1(SM80_Device_Gemm_XOR_b1t_b1n_s32t_tensor_op_s32, 64x64x512_32x32x512, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 512>,
      cutlass::gemm::GemmShape<32, 32, 512>, cutlass::gemm::GemmShape<16, 8, 256>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6, 128, 128,
      false, cutlass::arch::OpXorPopc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////

#endif  // #if defined(CUTLASS_ARCH_MMA_B1_XOR_SM80_ENABLED)
