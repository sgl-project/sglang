/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/gemm/warp/default_mma_tensor_op.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)


////////////////////////////////////////////////////////////////////////////////
/// F32 <= F16 * I8 + F32 (Upcast on Operand B)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_f16_i8, 128x128x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = int8_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_f16_i8, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = int8_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// F32 <= I8 * F16 + F32 (Upcast on Operand A)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_i8_f16, 128x128x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = int8_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_i8_f16, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = int8_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// F32 <= F16 * U8 + F32 (Upcast on Operand B)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_f16_u8, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = uint8_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_f16_u8, 128x128x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = cutlass::half_t;
  using ElementB = uint8_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// F32 <= U8 * F16 + F32 (Upcast on Operand A)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_u8_f16, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = uint8_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_u8_f16, 128x128x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = uint8_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// F32 <= B16 * U8 + F32 (Upcast on Operand B)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_bf16_u8, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = uint8_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// F32 <= U8 * BF16 + F32 (Upcast on Operand A)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_u8_bf16, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = uint8_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// F32 <= I8 * BF16 + F32 (Upcast on Operand A)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_bf16_i8, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = int8_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// F32 <= B16 * I8 + F32 (Upcast on Operand B)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_i8_bf16, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementA = int8_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// S32 <= I4 * I8 + S32 (Upcast on Operand A)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_i4_i8, 64x64x64_64x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using ElementA = cutlass::int4b_t;
  using ElementB = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
/// S32 <= I8 * I4 + S32 (Upcast on Operand B)
////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_mixed_input_tensor_op_crosswise_i8_i4, 64x64x64_64x64x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using ElementA = int8_t;
  using ElementB = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementB>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddMixedInputUpcast>::Type;

  test::gemm::warp::TransformTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

#endif // if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
