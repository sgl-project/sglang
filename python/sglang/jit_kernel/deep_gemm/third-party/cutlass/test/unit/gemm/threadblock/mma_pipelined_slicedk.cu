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
    \brief Unit tests for CTA-level GEMM specifically for sliced-k kernels (SM_61 and SM_75)
*/

#include "mma_pipelined_testbed_slicedk.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// igemm_NT DP4A
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM61_igemm_sliced_k, igemm_int8_nt_32x32x128_32x32x4) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 32, 128>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 32, 32>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2>;                                     // Stages,

    cutlass::gemm::GemmCoord problem_size(32, 32, 128);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm_sliced_k_big, igemm_int8_nt_32x32x128_32x32x4_bigk) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 32, 128>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 32, 32>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2>;                                     // Stages,

    cutlass::gemm::GemmCoord problem_size(32, 32, 1024);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}


TEST(SM61_igemm_sliced_k, igemm_int8_nt_32x64x128_32x32x4) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 64, 128>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 32, 64>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2>;                                     // Stages,

    cutlass::gemm::GemmCoord problem_size(32, 64, 256);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
// Tensor Op GEMM for SM_75
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_gemm_threadblock_congruous_sliced, tensor_op_64x64x256_tb64x64x64_warp64x32x32_16x8x8) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp, 2,
      cutlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(SM75_gemm_threadblock_crosswise_sliced, tensor_op_64x64x256_tb64x64x64_warp64x32x32_16x8x8) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, cutlass::arch::OpClassTensorOp, 2,
      cutlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(32, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

#endif
