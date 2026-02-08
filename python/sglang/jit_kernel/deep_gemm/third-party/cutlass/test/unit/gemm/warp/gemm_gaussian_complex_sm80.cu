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

#include "cutlass/cutlass.h"
#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"

#include "cutlass/gemm/warp/default_mma_complex_tensor_op.h"

#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_gaussian_complex_tensor_op, 8x8x4_8x8x4_nt) {

  using Shape = cutlass::gemm::GemmShape<8, 8, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  
  using Element = cutlass::complex<double>;
  using ElementC = cutlass::complex<double>;

  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
    Shape, 
    InstructionShape, 
    Element, 
    LayoutA, 
    Element, 
    LayoutB, 
    ElementC,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddGaussianComplex
  >::Type;

  test::gemm::warp::TestbedComplex<MmaTensorOp, cutlass::gemm::GemmShape<8, 8, 4> >().run();
}

TEST(SM80_warp_gemm_gaussian_complex_tensor_op, 16x16x4_8x8x4_nt) {

  using Shape = cutlass::gemm::GemmShape<16, 16, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  
  using Element = cutlass::complex<double>;
  using ElementC = cutlass::complex<double>;

  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
    Shape, 
    InstructionShape, 
    Element, 
    LayoutA, 
    Element, 
    LayoutB, 
    ElementC,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddGaussianComplex
  >::Type;

  test::gemm::warp::TestbedComplex<MmaTensorOp, cutlass::gemm::GemmShape<16, 16, 4> >().run();
}


TEST(SM80_warp_gemm_gaussian_complex_tensor_op, 16x32x4_8x8x4_nt) {

  using Shape = cutlass::gemm::GemmShape<16, 32, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  
  using Element = cutlass::complex<double>;
  using ElementC = cutlass::complex<double>;

  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
    Shape, 
    InstructionShape, 
    Element, 
    LayoutA, 
    Element, 
    LayoutB, 
    ElementC,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddGaussianComplex
  >::Type;

  test::gemm::warp::TestbedComplex<MmaTensorOp, cutlass::gemm::GemmShape<16, 32, 4> >().run();
}

TEST(SM80_warp_gemm_gaussian_complex_tensor_op, 32x16x4_8x8x4_nt) {

  using Shape = cutlass::gemm::GemmShape<32, 16, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  
  using Element = cutlass::complex<double>;
  using ElementC = cutlass::complex<double>;

  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
    Shape, 
    InstructionShape, 
    Element, 
    LayoutA, 
    Element, 
    LayoutB, 
    ElementC,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddGaussianComplex
  >::Type;

  test::gemm::warp::TestbedComplex<MmaTensorOp, cutlass::gemm::GemmShape<32, 16, 4> >().run();
}

TEST(SM80_warp_gemm_gaussian_complex_tensor_op, 32x32x4_8x8x4_nt) {

  using Shape = cutlass::gemm::GemmShape<32, 32, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  
  using Element = cutlass::complex<double>;
  using ElementC = cutlass::complex<double>;

  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
    Shape, 
    InstructionShape, 
    Element, 
    LayoutA, 
    Element, 
    LayoutB, 
    ElementC,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddGaussianComplex
  >::Type;

  test::gemm::warp::TestbedComplex<MmaTensorOp, cutlass::gemm::GemmShape<32, 32, 4> >().run();
}

TEST(SM80_warp_gemm_gaussian_complex_tensor_op, 32x32x4_8x8x4_nh) {

  using Shape = cutlass::gemm::GemmShape<32, 32, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  
  using Element = cutlass::complex<double>;
  using ElementC = cutlass::complex<double>;

  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
    Shape, 
    InstructionShape, 
    Element, 
    LayoutA, 
    Element, 
    LayoutB, 
    ElementC,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kConjugate,
    cutlass::arch::OpMultiplyAddGaussianComplex
  >::Type;

  test::gemm::warp::TestbedComplex<MmaTensorOp, cutlass::gemm::GemmShape<32, 32, 4> >().run();
}

TEST(SM80_warp_gemm_gaussian_complex_tensor_op, 32x32x4_8x8x4_ct) {

  using Shape = cutlass::gemm::GemmShape<32, 32, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  
  using Element = cutlass::complex<double>;
  using ElementC = cutlass::complex<double>;

  using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;
  using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
    Shape, 
    InstructionShape, 
    Element, 
    LayoutA, 
    Element, 
    LayoutB, 
    ElementC,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kConjugate,
    cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddGaussianComplex
  >::Type;

  test::gemm::warp::TestbedComplex<MmaTensorOp, cutlass::gemm::GemmShape<32, 32, 4> >().run();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_warp_gemm_gaussian_complex_tensor_op, 16x16x4_8x8x4_tn) {

  using Shape = cutlass::gemm::GemmShape<16, 16, 4>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  
  using Element = cutlass::complex<double>;
  using ElementC = cutlass::complex<double>;

  using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4;
  using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
    Shape, 
    InstructionShape, 
    Element, 
    LayoutA, 
    Element, 
    LayoutB, 
    ElementC,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::arch::OpMultiplyAddGaussianComplex
  >::Type;

  test::gemm::warp::TestbedComplex<MmaTensorOp, cutlass::gemm::GemmShape<16, 16, 4> >().run();
}
///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

