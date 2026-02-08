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

#ifdef CUTLASS_ARCH_WMMA_SM70_ENABLED
#include "mma_pipelined_testbed.h"
#include "cutlass/gemm/threadblock/default_mma_core_wmma.h"

/// All tests use double-buffered (kStages=2) mma pipeline for the gemm mainloop
/// Test name format: SM[arch]_gemm_threadblock_wmma_tensor_op_[alayout]_[blayout]_[clayout]_[dtype].[threadblock_shape]_[warp_shape]

//////////////// [START] Verifying all layouts {N,T}x{N,T}=>{N,T} for WMMA 16x16x16 [START] //////////////////////

///////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 (wmma native size 16x16x16)
////////////////////////////////////////////////////////////

// tests for {N,T}x{N,T}=>{T}
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

///////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16 (wmma native size 16x16x16)
///////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_col_row_row_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM70_gemm_threadblock_wmma_tensor_op_col_row_row_f16, 128x128x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(128, 128, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


///////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16 (wmma native size 16x16x16)
///////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_row_row_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM70_gemm_threadblock_wmma_tensor_op_row_row_row_f16, 128x128x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(128, 128, 96);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

///////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.col.m16n16k16.f16.f16 (wmma native size 16x16x16)
///////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_col_col_row_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM70_gemm_threadblock_wmma_tensor_op_col_col_row_f16, 128x128x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(128, 128, 96);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

// tests for {N,T}x{N,T}=>{N}
///////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 (wmma native size 16x16x16)
////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_col_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

///////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16 (wmma native size 16x16x16)
///////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_col_row_col_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


///////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16 (wmma native size 16x16x16)
///////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_row_col_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


///////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.col.col.m16n16k16.f16.f16 (wmma native size 16x16x16)
///////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_col_col_col_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

//////////////// [END] Verifying all layouts {N,T}x{N,T}=>{N,T} for WMMA 16x16x16 [END] //////////////////////

TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f16, 128x128x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(128, 128, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f16, multicta_256x256x96_128x128x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(256, 256, 96);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

///////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m32n8k16.f16.f16 (wmma native size 32x8x16)
///////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f16, 64x64x32_64x64x32_32x8x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<32, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

//////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m8n32k16.f16.f16  (wmma native size 8x32x16)
//////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f16, 64x64x32_64x64x32_8x32x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 32, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

//////////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32   (wmma native size 16x16x16)
//////////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f32, 64x64x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f32, 128x128x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f32, multicta_256x256x96_128x128x32_64x64x32_16x16x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(256, 256, 96);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

///////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32   (wmma native size 32x8x16)
////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f32, 64x64x32_64x64x32_32x8x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<32, 8, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

/////////////////////////////////////////////////////////////////////////////////
/// wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype
/// wmma.mma.sync.aligned.row.col.m8n32k16.f32.f32   (wmma native size 8x32x16)
/////////////////////////////////////////////////////////////////////////////////
TEST(SM70_gemm_threadblock_wmma_tensor_op_row_col_row_f32, 64x64x32_64x64x32_8x32x16) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  static const int kStages = 2; 

  cutlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 32, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassWmmaTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

#endif //CUTLASS_ARCH_WMMA_SM70_ENABLED
