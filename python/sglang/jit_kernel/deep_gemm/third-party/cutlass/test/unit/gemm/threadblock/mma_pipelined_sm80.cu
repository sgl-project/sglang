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
    \brief Unit tests for threadblock-level GEMM
*/

#include "mma_pipelined_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous, tensor_op_64x64x16_64x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous, tensor_op_128x64x16_64x32x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous, tensor_op_64x128x16_32x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous, tensor_op_128x128x16_64x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_256x256x96_128x128x16_64x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 96);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_congruous,
     multicta_512x256x192_256x128x16_64x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise, tensor_op_64x64x16_64x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise, tensor_op_32x32x16_16x16x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(32, 32, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise, tensor_op_32x64x16_16x32x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(32, 64, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise, tensor_op_64x32x16_32x16x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 32, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise, tensor_op_64x64x16_32x32x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise, tensor_op_128x64x16_64x32x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 64, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise, tensor_op_64x128x16_32x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 128, 64);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise, tensor_op_128x128x16_64x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(128, 128, 48);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_256x256x48_128x128x16_64x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(256, 256, 48);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp>;

  dim3 grid(2, 2);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_crosswise,
     multicta_512x256x192_256x128x16_64x64x16_16x8x4) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(512, 256, 192);

  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp>;

  dim3 grid(2, 2);
  dim3 block(32, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

#endif // if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)


