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
    \brief Tests for device-wide TRMM interface

  
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/blas3.h"
#include "cutlass/gemm/device/trmm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/trmm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_trmm_universal.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_l_nu_tensor_op_fast_f32_align1_align1, 64x128x32_32x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float, cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft, cutlass::FillMode::kLower, cutlass::DiagType::kNonUnit,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#if 0 
TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_l_nu_tensor_op_fast_f32_align1_align4, 128x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kLower,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}
#endif 

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_l_nu_tensor_op_fast_f32_align1_align4, 256x128x16_64x64x16) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kLower,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_l_nu_tensor_op_fast_f32_align1_align4, 128x256x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kLower,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;
  
  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_l_nu_tensor_op_fast_f32_align1_align4, 256x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kLower,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_u_nu_tensor_op_fast_f32_align1_align4, 64x64x16_32x32x16) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kUpper,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    10,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_u_nu_tensor_op_fast_f32_align1_align4, 128x128x16_64x64x16) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kUpper,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_u_nu_tensor_op_fast_f32_align1_align4, 256x128x16_128x64x16) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kUpper,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_u_nu_tensor_op_fast_f32_align1_align4, 128x256x16_64x128x16) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kUpper,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 16>,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;
  
  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Trmm_f32t_f32n_f32n_ls_u_nu_tensor_op_fast_f32_align1_align4, 256x256x16_64x128x16) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Trmm = cutlass::gemm::device::Trmm<
    float,
    cutlass::layout::RowMajor,
    cutlass::SideMode::kLeft,
    cutlass::FillMode::kUpper,
    cutlass::DiagType::kNonUnit,
    float,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 256, 16>,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;

  EXPECT_TRUE(test::gemm::device::TestAllTrmmUniversal<Trmm>());
}
/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
