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
    \brief Unit tests for thread-level GEMM
*/
#include "cutlass/arch/wmma.h"

#ifdef CUTLASS_ARCH_WMMA_SM75_ENABLED
#include "mma_pipelined_testbed.h"
#include "cutlass/gemm/threadblock/default_mma_core_wmma.h"

/// All tests use single staged (kStages=1) mma pipeline for the gemm mainloop
/// Test name format: SM[arch]_gemm_threadblock_singlestage_wmma_tensor_op_[alayout]_[blayout]_[clayout]_[atype].[threadblock_shape]_[warp_shape]_[instruction_shape]

/////////////////////////////////////////////////////////////////////////
///       Integer (s8 and u8) WMMA threadblock level tests          ////
/////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_INTEGER_MATRIX_MULTIPLY_ENABLED)
TEST(SM75_gemm_threadblock_singlestage_wmma_tensor_op_row_col_row_s8, 64x64x32_64x64x32_16x16x16) {
 
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 1; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM75_gemm_threadblock_singlestage_wmma_tensor_op_row_col_row_s8, 64x64x64_64x64x64_16x16x16) {
 
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 1; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


TEST(SM75_gemm_threadblock_singlestage_wmma_tensor_op_col_row_row_s8, 64x64x32_64x64x32_16x16x16) {
 
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 1; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM75_gemm_threadblock_singlestage_wmma_tensor_op_col_row_row_s8, 64x64x64_64x64x64_16x16x16) {
 
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 1; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}
#endif //CUTLASS_ARCH_INTEGER_MATRIX_MULTIPLY_ENABLED


////////////////////////////////////////////////////////////////////////
///      SUBBYTE (s4 and b1) WMMA threadblock level tests          ////
///////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_SUBBYTE_INTEGER_MATRIX_MULTIPLY_ENABLED)
TEST(SM75_gemm_threadblock_singlestage_wmma_tensor_op_row_col_row_s4, 64x64x128_64x64x128_8x8x32) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 1; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;

  float alpha = 1.f;
  float beta = 0.f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC, 
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


TEST(SM75_gemm_threadblock_singlestage_wmma_tensor_op_row_col_col_s4, 64x64x64_64x64x64_8x8x32) {
  using ElementA = cutlass::int4b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  static const int kStages = 1; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 64);

  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;

  float alpha = 1.f;
  float beta = 0.f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC, 
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM75_gemm_threadblock_singlestage_wmma_tensor_op_row_col_row_b1, 64x64x512_64x64x512_8x8x128) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 1; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 2048);

  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 128>;

  float alpha = 1.f;
  float beta = 0.f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC, 
      cutlass::arch::OpClassWmmaTensorOp, kStages,
      cutlass::arch::OpXorPopc>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM75_gemm_threadblock_singlestage_wmma_tensor_op_row_col_col_b1, 64x64x512_64x64x512_8x8x128) {
  using ElementA = cutlass::uint1b_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::uint1b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  static const int kStages = 1; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 2048);

  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 128>;

  float alpha = 1.f;
  float beta = 0.f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC, 
      cutlass::arch::OpClassWmmaTensorOp, kStages,
      cutlass::arch::OpXorPopc>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}
#endif //CUTLASS_SUBBYTE_INTEGER_MATRIX_MULTIPLY_ENABLED

#endif //CUTLASS_ARCH_WMMA_SM75_ENABLED
