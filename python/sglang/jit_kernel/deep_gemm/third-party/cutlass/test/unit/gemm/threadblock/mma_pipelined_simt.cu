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

#include "mma_pipelined_testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm_NT
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_sgemm, sgemm_nt_32x64x8_32x64x1) {
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      cutlass::gemm::GemmShape<32, 64, 8>,    // ThreadblockShape,
      cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
      cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
      float,                                  // ElementA,
      cutlass::layout::ColumnMajor,           // LayoutA,
      float,                                  // ElementB,
      cutlass::layout::RowMajor,              // LayoutB,
      float,                                  // ElementC,
      cutlass::layout::RowMajor,              // LayoutC,
      cutlass::arch::OpClassSimt,             // OpClass,
      2,                                      // Stages,
      cutlass::arch::OpMultiplyAdd            // Operator,
      >;                                     

  cutlass::gemm::GemmCoord problem_size(32, 64, 48);
  float alpha = 1.f;
  float beta = 0.0f;
  dim3 grid(1, 1);
  dim3 block(32, 1, 1);
  test::gemm::threadblock::Testbed<MmaCore>(
      problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
      .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_sgemm, sgemm_nt_64x64x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        float,                                  // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        float,                                  // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_sgemm, sgemm_nt_32x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 128, 8>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        float,                                  // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        float,                                  // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(32, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_sgemm, sgemm_nt_64x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        float,                                  // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        float,                                  // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 128, 16);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_sgemm, sgemm_nt_128x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 128, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,      // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,        // InstructionShape,
        float,                                    // ElementA,
        cutlass::layout::ColumnMajor,             // LayoutA,
        float,                                    // ElementB,
        cutlass::layout::RowMajor,                // LayoutB,
        float,                                    // ElementC,
        cutlass::layout::RowMajor,                // LayoutC,
        cutlass::arch::OpClassSimt,               // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd              // Operator,
        >;                                       

    cutlass::gemm::GemmCoord problem_size(128, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(),
        problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// dgemm_NN
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_dgemm, dgemm_nt_32x64x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 64, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        double,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        double,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        double,                                 // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(32, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_dgemm, dgemm_nt_64x64x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        double,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        double,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        double,                                 // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_dgemm, dgemm_nt_32x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 128, 8>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        double,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        double,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        double,                                 // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(32, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_dgemm, dgemm_nt_64x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        double,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        double,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        double,                                 // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 128, 16);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_dgemm, dgemm_nt_128x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 128, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,      // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,        // InstructionShape,
        double,                                   // ElementA,
        cutlass::layout::ColumnMajor,             // LayoutA,
        double,                                   // ElementB,
        cutlass::layout::RowMajor,                // LayoutB,
        double,                                   // ElementC,
        cutlass::layout::RowMajor,                // LayoutC,
        cutlass::arch::OpClassSimt,               // OpClass
        2,                                        // Stages,
        cutlass::arch::OpMultiplyAdd              // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(),
        problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// igemm_NN
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_igemm, igemm_nt_32x64x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 64, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        int,                                    // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int,                                    // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(32, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_igemm, igemm_nt_64x64x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        int,                                    // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int,                                    // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_igemm, igemm_nt_32x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 128, 8>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        int,                                    // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int,                                    // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(32, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_igemm, igemm_nt_64x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        int,                                    // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int,                                    // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 128, 16);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_igemm, igemm_nt_128x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 128, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,      // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,        // InstructionShape,
        int,                                      // ElementA,
        cutlass::layout::ColumnMajor,             // LayoutA,
        int,                                      // ElementB,
        cutlass::layout::RowMajor,                // LayoutB,
        int,                                      // ElementC,
        cutlass::layout::RowMajor,                // LayoutC,
        cutlass::arch::OpClassSimt,               // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd              // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(),
        problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// hgemm_NN
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM50_hgemm, hgemm_nt_32x64x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 64, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        cutlass::half_t,                        // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        cutlass::half_t,                        // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        cutlass::half_t,                        // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(32, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_hgemm, hgemm_nt_64x64x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        cutlass::half_t,                        // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        cutlass::half_t,                        // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        cutlass::half_t,                        // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_hgemm, hgemm_nt_32x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<32, 128, 8>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        cutlass::half_t,                        // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        cutlass::half_t,                        // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        cutlass::half_t,                        // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(32, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_hgemm, hgemm_nt_64x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        cutlass::half_t,                        // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        cutlass::half_t,                        // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        cutlass::half_t,                        // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 128, 16);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM50_hgemm, hgemm_nt_128x128x8_32x64x1) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 128, 8>,    // ThreadblockShape,
        cutlass::gemm::GemmShape<32, 64, 8>,      // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>,        // InstructionShape,
        cutlass::half_t,                          // ElementA,
        cutlass::layout::ColumnMajor,             // LayoutA,
        cutlass::half_t,                          // ElementB,
        cutlass::layout::RowMajor,                // LayoutB,
        cutlass::half_t,                          // ElementC,
        cutlass::layout::RowMajor,                // LayoutC,
        cutlass::arch::OpClassSimt,               // OpClass
        2,                                        // Stages,
        cutlass::arch::OpMultiplyAdd              // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(),
        problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// igemm_NT DP4A
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM61_igemm, igemm_int8_nt_64x64x16_64x64x4) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_nt_64x64x32_64x64x4) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 32>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 4096);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_nt_64x64x16_64x64x8) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_nt_128x64x16_64x64x8) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 64, 16>,  // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_nt_128x128x16_64x64x8) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 128, 16>, // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 128, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_nt_256x128x16_64x64x8) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<256, 256, 16>, // ThreadblockShape,
        cutlass::gemm::GemmShape<128, 64, 16>,   // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(256, 256, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_nt_128x256x64_64x64x16) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 256, 64>, // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 64>,   // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 256, 64);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_nt_256x128x64_64x64x16) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<256, 128, 64>, // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 64>,   // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(256, 128, 64);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_tn_64x64x16_64x64x4) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_tn_64x64x32_64x64x4) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 32>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 4096);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_tn_64x64x16_64x64x8) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}
TEST(SM61_igemm, igemm_int8_tn_128x64x16_64x64x8) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 64, 16>,  // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_tn_128x128x16_64x64x8) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 128, 16>, // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 128, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_tn_256x128x16_64x64x8) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<256, 256, 16>, // ThreadblockShape,
        cutlass::gemm::GemmShape<128, 64, 16>,   // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(256, 256, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_tn_128x256x64_64x64x16) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<128, 256, 64>, // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 64>,   // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(128, 256, 64);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_tn_256x128x64_64x64x16) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<256, 128, 64>, // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 64>,   // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(256, 128, 64);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

TEST(SM61_igemm, igemm_int8_nn_64x64x16_64x64x4) {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        cutlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        cutlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        cutlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        cutlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        cutlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        cutlass::layout::RowMajor,              // LayoutC,
        cutlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        cutlass::arch::OpMultiplyAdd            // Operator,
        >;

    cutlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(32, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, cutlass::Distribution::Uniform, cutlass::Distribution::Uniform);
}

