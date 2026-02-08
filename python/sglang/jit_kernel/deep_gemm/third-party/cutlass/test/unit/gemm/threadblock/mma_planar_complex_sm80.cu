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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma_planar_complex_multistage.h"

#include "mma_planar_complex_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_gemm_threadblock_planar_complex_congruous, tensor_op_64x64x32_64x64x32_16x8x16_3stage) {

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;

  cutlass::gemm::GemmCoord problem_size(64, 64, 8);

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  int const Stages = 3;

  // Define the MmaCore components
  using Mma = typename cutlass::gemm::threadblock::DefaultMmaPlanarComplexMultistage<
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementC, LayoutC, 
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape,
      Stages>::ThreadblockMma;

  dim3 grid(1, 1);
  dim3 block(32, Mma::WarpCount::kCount, 1);

  test::gemm::threadblock::TestbedPlanarComplex<Mma>(problem_size.m(), problem_size.n(),
                                            problem_size.k())
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////
#endif  // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
