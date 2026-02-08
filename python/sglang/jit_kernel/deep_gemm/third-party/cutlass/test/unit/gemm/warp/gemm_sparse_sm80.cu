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

#include "cutlass/gemm/warp/default_mma_sparse_tensor_op.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 128x128x64_64x64x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                                  cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 128x128x64_64x32x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 128x128x64_32x64x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 128x128x64_32x32x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 128x128x64_32x16x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 128x64x128_64x32x128_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 64, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 64x128x128_32x64x128_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 128, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 64x64x128_32x32x128_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_f16, 64x32x128_32x16x128_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 32, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 128x128x64_64x64x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 128x128x64_64x32x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 128x128x64_32x64x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 128x128x64_32x32x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 64x32x64_32x32x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 32, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 64x16x64_32x16x64_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 16>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 16, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 128x64x128_64x32x128_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 64, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 64x128x128_32x64x128_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 128, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_f16, 64x64x128_32x32x128_16x8x32) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Element = cutlass::half_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 128x128x128_64x64x128_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                                  cutlass::gemm::GemmShape<128, 128, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 128x128x128_64x32x128_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 128x128x128_32x64x128_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 128x128x128_32x32x128_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 128x128x128_32x16x128_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 128> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 128x64x256_64x32x256_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 64, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 64x128x256_32x64x256_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 128, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 64x64x256_32x32x256_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s8, 64x32x256_32x16x256_16x8x64) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 32, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 128x128x32_64x64x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                                  cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 128x128x32_64x32x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 128x128x32_32x64x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 128x128x32_32x32x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 128x128x32_32x16x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 128x64x256_64x32x256_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 64x128x64_32x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 64x64x64_32x32x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_tf32, 64x32x64_32x16x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 32, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_tf32, 128x128x32_64x64x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_tf32, 128x128x32_64x32x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_tf32, 128x128x32_32x64x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_tf32, 128x128x32_32x32x32_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_tf32, 128x64x64_64x32x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_tf32, 64x128x64_32x64x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_congruous_tf32, 64x64x64_32x32x64_16x8x16) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using Element = cutlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 128x128x256_64x64x256_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<64, 64, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                                  cutlass::gemm::GemmShape<128, 128, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 128x128x256_64x32x256_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 128x128x256_32x64x256_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 128x128x256_32x32x256_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 128x128x256_32x16x256_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 256>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 128>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 128, 256> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 128x64x512_64x32x512_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<64, 32, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<128, 64, 512> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 64x128x512_32x64x512_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<32, 64, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 128, 512> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 64x64x512_32x32x512_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<32, 32, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 64, 512> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_sparse_tensor_op_crosswise_s4, 64x32x512_32x16x512_16x8x128) {
  using Shape = cutlass::gemm::GemmShape<32, 16, 512>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 128>;
  using Element = cutlass::int4b_t;
  using ElementC = int32_t;
  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<Element>::value, 256>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>::Type;

  test::gemm::warp::SparseTestbed<MmaTensorOp,
                            cutlass::gemm::GemmShape<64, 32, 512> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_SUPPORTED)
