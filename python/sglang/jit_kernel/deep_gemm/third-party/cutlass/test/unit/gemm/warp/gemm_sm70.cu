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

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"

#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM70_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_warp_gemm_tensor_op_congruous, 128x128x16_64x64x16_16x16x4) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<MmaTensorOp, cutlass::gemm::GemmShape<128, 128, 16> >().run();
}

TEST(SM70_warp_gemm_tensor_op_congruous, 128x64x4_64x64x4_16x16x4) {

  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<MmaTensorOp, cutlass::gemm::GemmShape<128, 64, 4> >().run();
}

TEST(SM70_warp_gemm_tensor_op_congruous, 128x128x4_32x32x4_16x16x4) {

  using Shape = cutlass::gemm::GemmShape<32, 32, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementA>::value>;
  using LayoutB = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementB>::value>;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<MmaTensorOp, cutlass::gemm::GemmShape<128, 128, 4> >().run();
}

TEST(SM70_warp_gemm_tensor_op_crosswise, 64x64x32_64x64x32_16x16x4) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 32>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 32>;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::RowMajor,
      ElementB,
      cutlass::layout::ColumnMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<MmaTensorOp, cutlass::gemm::GemmShape<64, 64, 32> >().run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM70_warp_gemm_volta_tensor_op_canonical_f32_row_col, 64x64x16_64x64x4_8x8x4) {
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      cutlass::layout::RowMajor,
      ElementB,
      cutlass::layout::ColumnMajor,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 16> >()
      .run();
}

TEST(SM70_warp_gemm_volta_tensor_op_canonical_f32_col_row, 64x64x16_64x64x4_8x8x4) {
  
  using Shape = cutlass::gemm::GemmShape<64, 64, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      cutlass::gemm::GemmShape<16, 16, 4>,
      32,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
    Shape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    cutlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 16> >()
      .run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // CUTLASS_ARCH_MMA_SM70_SUPPORTED
