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
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_mixed_input_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_fast_f32.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses BF16 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  GemmShape<16, 8, 8>, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAddFastBF16, 
  PartitionsK, AccumulatorsInRowMajor> {

  // Uses BF16 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 8, 8>, 
        32, 
        bfloat16_t, cutlass::layout::RowMajor, 
        bfloat16_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses F16 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  GemmShape<16, 8, 8>, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAddFastF16, 
  PartitionsK, AccumulatorsInRowMajor> {

  // Uses F16 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 8, 8>, 
        32, 
        half_t, cutlass::layout::RowMajor, 
        half_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses TF32 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of target matrix multiply instruction (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  InstructionShape_, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAdd, PartitionsK, AccumulatorsInRowMajor> {

  // Uses TF32 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        tfloat32_t, cutlass::layout::RowMajor, 
        tfloat32_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses TF32 for Fast Accurate FP32
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of target matrix multiply instruction (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_, 
  InstructionShape_, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAddFastF32, PartitionsK, AccumulatorsInRowMajor> {

  // Uses TF32 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        cutlass::tfloat32_t, cutlass::layout::RowMajor, 
        cutlass::tfloat32_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOpFastF32<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs are mixed types  - uses wider datatype internally.
/// (e.g. F16 <= F16 x S8 + F16, F16 <= BF16 x S8 + F32)
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Element type of A matrix
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Element type of B matrix
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Element type of C matrix
    typename ElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_,
  GemmShape<16, 8, 16>,                 // InstructionShape
  ElementA,                             // Element type of A matrix in Global Memory
  LayoutA,                              // Layout of A matrix in Global Memory
  ElementB,                             // Element type of B matrix in Global Memory
  LayoutB,                              // Layout of B matrix in Global Memory
  ElementC,                             // Element type of C matrix in Global Memory
  LayoutC,                              // Layout of C matrix in Global Memory
  arch::OpMultiplyAddMixedInputUpcast,  // Tag to indicate mixed-input datatype, where narrower datatype is upcasted to wider datatype
  PartitionsK, AccumulatorsInRowMajor> {


  // Check if the ElementA and ElementB are of different data types
  static_assert(!platform::is_same<ElementA, ElementB>::value,
    "DefaultMmaTensorOp with arch::OpMultiplyAddMixedInputUpcast ElementA and ElementB cannot be of the same data type");

  // Data type used for internal computation - use the wider of the two data types for mma.sync operands
  using ElementOperand = typename platform::conditional<(sizeof_bits<ElementA>::value > sizeof_bits<ElementB>::value),
                                                    ElementA, ElementB>::type;

  // Operand datatypes in the internal MMA instruction - use the wider of the two data types
  using ElementAMma = ElementOperand;
  using ElementBMma = ElementOperand;
  using MmaElementC = ElementC;

  // Uses
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 8, 16>,
        32,
        ElementAMma, cutlass::layout::RowMajor,
        ElementBMma, cutlass::layout::ColumnMajor,
        MmaElementC, cutlass::layout::RowMajor,
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaMixedInputTensorOp<
      WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs are mixed types  - uses wider datatype internally.
/// (e.g. S32 <= S4 x S8 + S32, S32 <= S8 x S4 + S32)
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Element type of A matrix
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Element type of B matrix
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Element type of C matrix
    typename ElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<
  WarpShape_,
  GemmShape<16, 8, 32>,                 // InstructionShape
  ElementA,                             // Element type of A matrix in Global Memory
  LayoutA,                              // Layout of A matrix in Global Memory
  ElementB,                             // Element type of B matrix in Global Memory
  LayoutB,                              // Layout of B matrix in Global Memory
  ElementC,                             // Element type of C matrix in Global Memory
  LayoutC,                              // Layout of C matrix in Global Memory
  arch::OpMultiplyAddMixedInputUpcast,  // Tag to indicate mixed-input datatype, where narrower datatype is upcasted to wider datatype
  PartitionsK, AccumulatorsInRowMajor> {


  // Check if the ElementA and ElementB are of different data types
  static_assert(!platform::is_same<ElementA, ElementB>::value,
    "DefaultMmaTensorOp with arch::OpMultiplyAddMixedInputUpcast ElementA and ElementB cannot be of the same data type");

  // Data type used for internal computation - use the wider of the two data types for mma.sync operands
  using ElementOperand = typename platform::conditional<(sizeof_bits<ElementA>::value > sizeof_bits<ElementB>::value),
                                                    ElementA, ElementB>::type;

  // Operand datatypes in the internal MMA instruction - use the wider of the two data types
  using MmaElementA = ElementOperand;
  using MmaElementB = ElementOperand;
  using MmaElementC = ElementC;

  // Uses
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 8, 32>,
        32,
        MmaElementA, cutlass::layout::RowMajor,
        MmaElementB, cutlass::layout::ColumnMajor,
        MmaElementC, cutlass::layout::RowMajor,
        arch::OpMultiplyAddSaturate
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaMixedInputTensorOp<
      WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
