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
    \brief Default warp-level GEMM operators selected by data type, size, and layouts of operands.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_complex_tensor_op.h"
#include "cutlass/gemm/warp/mma_complex_tensor_op_fast_f32.h"
#include "cutlass/gemm/warp/mma_gaussian_complex_tensor_op.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements
    typename ElementA_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA_,
    /// Data type of B elements
    typename ElementB_,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB_,
    /// Element type of C matrix
    typename ElementC_,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC_,
    /// Complex transform on A operand
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// Complex transform on B operand
    ComplexTransform TransformB = ComplexTransform::kNone,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_ = arch::OpMultiplyAddComplex>
struct DefaultMmaComplexTensorOp;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex<T>*complex<T> case
//  4 real-valued mma operations
//  A = (ar + j ai), B (br +j bi), D = AB
//  D = dr + j di = (ar*br - ai*bi) + j (ar*bi + ai*br) 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Real-valued underlying type of complex-valued A operand
    typename RealElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Real-valued underlying type of complex-valued B operand
    typename RealElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Real-valued underlying type of complex-valued C operand
    typename RealElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Complex transform on A operand
    ComplexTransform TransformA,
    /// Complex transform on B operand
    ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
    WarpShape_,
    InstructionShape_,
    complex<RealElementA>,
    LayoutA,
    complex<RealElementB>,
    LayoutB,
    complex<RealElementC>,
    LayoutC,
    TransformA,
    TransformB,
    arch::OpMultiplyAddComplex> {

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        RealElementA,
        cutlass::layout::RowMajor,
        RealElementB,
        cutlass::layout::ColumnMajor,
        RealElementC,
        cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>
    >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaComplexTensorOp<
    WarpShape_,
    complex<RealElementA>,
    LayoutA,
    complex<RealElementB>,
    LayoutB,
    complex<RealElementC>,
    LayoutC, 
    Policy,
    TransformA,
    TransformB>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex<T>*complex<T> case using GaussianComplex operation
//  3 real-valued mma operations
//  A  = (ar + j ai), B = (br +j bi), D = AB
//  P1 = (ar + ai) * br, P2 = - ar * (br - bi), P3 = ai * (br + bi) 
//  D  = dr + j di = (P1 - P3) + j (P1 + P2)
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Real-valued underlying type of complex-valued A operand
    typename RealElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Real-valued underlying type of complex-valued B operand
    typename RealElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Real-valued underlying type of complex-valued C operand
    typename RealElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Complex transform on A operand
    ComplexTransform TransformA,
    /// Complex transform on B operand
    ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
    WarpShape_,
    InstructionShape_,
    complex<RealElementA>,
    LayoutA,
    complex<RealElementB>,
    LayoutB,
    complex<RealElementC>,
    LayoutC,
    TransformA,
    TransformB,
    arch::OpMultiplyAddGaussianComplex> {

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        RealElementA,
        cutlass::layout::RowMajor,
        RealElementB,
        cutlass::layout::ColumnMajor,
        RealElementC,
        cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>
    >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaGaussianComplexTensorOp<
    WarpShape_,
    complex<RealElementA>,
    LayoutA,
    complex<RealElementB>,
    LayoutB,
    complex<RealElementC>,
    LayoutC, 
    Policy,
    TransformA,
    TransformB>;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization - input and output types are complex<float>*complex<float> 
//  Use TF32 tensor operation internally
//  4 real-valued mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32 operations on TF32 
//  A = (ar + j ai), B (br +j bi), D = AB
//  D = dr + j di = (ar*br - ai*bi) + j (ar*bi + ai*br) 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Complex transform on A operand
    ComplexTransform TransformA,
    /// Complex transform on B operand
    ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
    WarpShape_,
    InstructionShape_,
    complex<float>,
    LayoutA,
    complex<float>,
    LayoutB,
    complex<float>,
    LayoutC,
    TransformA,
    TransformB,
    arch::OpMultiplyAddComplex> {

  // Complex floating point tensor operation use mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32 mma instruction
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        tfloat32_t,
        cutlass::layout::RowMajor,
        tfloat32_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>
    >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaComplexTensorOp<
    WarpShape_,
    complex<float>,
    LayoutA,
    complex<float>,
    LayoutB,
    complex<float>,
    LayoutC, 
    Policy,
    TransformA,
    TransformB>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization - input and output types are complex<float>*complex<float> 
//  Use BF16 tensor operation internally
//  4 real-valued mma.sync.aligned.m16n8k8.f32.bf16.bf16.f32 operations on BF16
//  A = (ar + j ai), B (br +j bi), D = AB
//  D = dr + j di = (ar*br - ai*bi) + j (ar*bi + ai*br) 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Complex transform on A operand
    ComplexTransform TransformA,
    /// Complex transform on B operand
    ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
    WarpShape_,
    InstructionShape_,
    complex<float>,
    LayoutA,
    complex<float>,
    LayoutB,
    complex<float>,
    LayoutC,
    TransformA,
    TransformB,
    arch::OpMultiplyAddFastBF16> {

  // Complex floating point tensor operation use mma.sync.aligned.m16n8k8.f32.bf16.bf16.f32 mma instruction
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        bfloat16_t,
        cutlass::layout::RowMajor,
        bfloat16_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>
    >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaComplexTensorOp<
    WarpShape_,
    complex<float>,
    LayoutA,
    complex<float>,
    LayoutB,
    complex<float>,
    LayoutC, 
    Policy,
    TransformA,
    TransformB>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization - input and output types are complex<float>*complex<float> 
//  Use F16 tensor operation internally
//  4 real-valued mma.sync.aligned.m16n8k8.f32.f16.f16.f32 operations on F16
//  A = (ar + j ai), B (br +j bi), D = AB
//  D = dr + j di = (ar*br - ai*bi) + j (ar*bi + ai*br) 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Complex transform on A operand
    ComplexTransform TransformA,
    /// Complex transform on B operand
    ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
    WarpShape_,
    InstructionShape_,
    complex<float>,
    LayoutA,
    complex<float>,
    LayoutB,
    complex<float>,
    LayoutC,
    TransformA,
    TransformB,
    arch::OpMultiplyAddFastF16> {

  // Complex floating point tensor operation use mma.sync.aligned.m16n8k8.f32.f16.f16.f32 mma instruction
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        half_t,
        cutlass::layout::RowMajor,
        half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>
    >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaComplexTensorOp<
    WarpShape_,
    complex<float>,
    LayoutA,
    complex<float>,
    LayoutB,
    complex<float>,
    LayoutC, 
    Policy,
    TransformA,
    TransformB>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// 3xTF32 or 4xTF32 (fast and accurate complex<float> operation)
/// Partial specialization - input and output types are complex<float> * complex<float> 
//  Use 3xTF32 or 4xTF32 tensor operation internally
//  4 real-valued mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32 operations on TF32 
//  A = (ar + j ai), B (br +j bi), D = AB
//  D = dr + j di = 3x[(ar*br - ai*bi) + j (ar*bi + ai*br)]
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Complex transform on A operand
    ComplexTransform TransformA,
    /// Complex transform on B operand
    ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
    WarpShape_,
    InstructionShape_,
    complex<float>,
    LayoutA,
    complex<float>,
    LayoutB,
    complex<float>,
    LayoutC,
    TransformA,
    TransformB,
    arch::OpMultiplyAddComplexFastF32> {

  // Complex floating point tensor operation use mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32 mma instruction
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        tfloat32_t,
        cutlass::layout::RowMajor,
        tfloat32_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>
    >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaComplexTensorOpFastF32<
    WarpShape_,
    complex<float>,
    LayoutA,
    complex<float>,
    LayoutB,
    complex<float>,
    LayoutC, 
    Policy,
    TransformA,
    TransformB>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex<double>*complex<double> case
//  4 real-valued mma.sync.aligned.m16n8k4.f64.f64.f64.f64 operations
//  A = (ar + j ai), B (br +j bi), D = AB
//  D = dr + j di = (ar*br - ai*bi) + j (ar*bi + ai*br) 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Real-valued underlying type of complex-valued A operand
    typename RealElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Real-valued underlying type of complex-valued B operand
    typename RealElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Real-valued underlying type of complex-valued C operand
    typename RealElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Complex transform on A operand
    ComplexTransform TransformA,
    /// Complex transform on B operand
    ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
    WarpShape_,
    GemmShape<16, 8, 4>,
    complex<RealElementA>,
    LayoutA,
    complex<RealElementB>,
    LayoutB,
    complex<RealElementC>,
    LayoutC,
    TransformA,
    TransformB,
    arch::OpMultiplyAddComplex> {

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 8, 4>,
        32, 
        RealElementA,
        cutlass::layout::RowMajor,
        RealElementB,
        cutlass::layout::ColumnMajor,
        RealElementC,
        cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>
    >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaComplexTensorOp<
    WarpShape_,
    complex<RealElementA>,
    LayoutA,
    complex<RealElementB>,
    LayoutB,
    complex<RealElementC>,
    LayoutC, 
    Policy,
    TransformA,
    TransformB,
    true>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for complex<T>*complex<T> case using GaussianComplex operation
//  3 real-valued mma.sync.aligned.m16n8k4.f64.f64.f64.f64 operations 
//  A  = (ar + j ai), B = (br +j bi), D = AB
//  P1 = (ar + ai) * br, P2 = - ar * (br - bi), P3 = ai * (br + bi) 
//  D  = dr + j di = (P1 - P3) + j (P1 + P2)
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Real-valued underlying type of complex-valued A operand
    typename RealElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Real-valued underlying type of complex-valued B operand
    typename RealElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Real-valued underlying type of complex-valued C operand
    typename RealElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Complex transform on A operand
    ComplexTransform TransformA,
    /// Complex transform on B operand
    ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
    WarpShape_,
    GemmShape<16, 8, 4>,
    complex<RealElementA>,
    LayoutA,
    complex<RealElementB>,
    LayoutB,
    complex<RealElementC>,
    LayoutC,
    TransformA,
    TransformB,
    arch::OpMultiplyAddGaussianComplex> {

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 8, 4>,
        32, 
        RealElementA,
        cutlass::layout::RowMajor,
        RealElementB,
        cutlass::layout::ColumnMajor,
        RealElementC,
        cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>
    >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaGaussianComplexTensorOp<
    WarpShape_,
    complex<RealElementA>,
    LayoutA,
    complex<RealElementB>,
    LayoutB,
    complex<RealElementC>,
    LayoutC, 
    Policy,
    TransformA,
    TransformB,
    true>;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass
