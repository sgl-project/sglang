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

    \brief Unit tests for thread-level GEMM with Hopper FP64
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"

#include "cutlass/gemm/warp/default_mma_tensor_op.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED)
TEST(SM90_warp_gemm_tensor_op_congruous_f64, 16x16x4_16x16x4_16x8x4) {
  using Shape = cutlass::gemm::GemmShape<16, 16, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Element = double;
  using ElementC = double;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<16, 16, 4> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM90_warp_gemm_tensor_op_congruous_f64, 32x16x4_32x16x4_16x8x4) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Element = double;
  using ElementC = double;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<32, 16, 4> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM90_warp_gemm_tensor_op_congruous_f64, 32x32x4_32x32x4_16x8x4) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Element = double;
  using ElementC = double;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<32, 32, 4> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM90_warp_gemm_tensor_op_congruous_f64, 32x64x4_32x64x4_16x8x4) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Element = double;
  using ElementC = double;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<32, 64, 4> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM90_warp_gemm_tensor_op_crosswise_f64, 16x16x16_16x16x16_16x8x4) {
  using Shape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Element = double;
  using ElementC = double;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<16, 16, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM90_warp_gemm_tensor_op_crosswise_f64, 32x32x16_32x32x16_16x8x4) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Element = double;
  using ElementC = double;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<32, 32, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM90_warp_gemm_tensor_op_crosswise_f64, 64x32x16_64x32x16_16x8x4) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Element = double;
  using ElementC = double;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 32, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM90_warp_gemm_tensor_op_crosswise_f64, 32x64x16_32x64x16_16x8x4) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  using Element = double;
  using ElementC = double;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<32, 64, 16> >()
      .run();
}
////////////////////////////////////////////////////////////////////////////////
#endif // if defined(CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED)
