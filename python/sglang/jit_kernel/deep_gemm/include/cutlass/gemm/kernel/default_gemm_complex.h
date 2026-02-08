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
    \brief 
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.
  
      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/default_epilogue_complex_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Operator class tag
  typename OperatorClass,
  /// Tag indicating architecture to tune for
  typename ArchTag,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Complex elementwise transformation on A operand
  ComplexTransform TransformA,
  /// Complex elementwise transformation on B operand
  ComplexTransform TransformB,
  /// Multiply-add operator 
  // (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
  typename Operator,
  /// If true, kernel is configured to support serial reduction in the epilogue
  bool SplitKSerial
>
struct DefaultGemmComplex;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Hopper Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Multiply-add operator 
    // (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial
  >
struct DefaultGemmComplex<
  ElementA, LayoutA, ElementB, LayoutB, ElementC,
  layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm90, ThreadblockShape, WarpShape, InstructionShape,
  EpilogueOutputOp, ThreadblockSwizzle, Stages, TransformA, TransformB, Operator, SplitKSerial> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
      ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
      layout::RowMajor, arch::OpClassTensorOp, arch::Sm90, ThreadblockShape,
      WarpShape, InstructionShape, Stages, TransformA, TransformB, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOp<
          ThreadblockShape, typename Mma::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Multiply-add operator 
    // (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial
  >
struct DefaultGemmComplex<
  ElementA, LayoutA, ElementB, LayoutB, ElementC,
  layout::RowMajor, ElementAccumulator, arch::OpClassSimt,
  arch::Sm50, ThreadblockShape, WarpShape, InstructionShape,
  EpilogueOutputOp, ThreadblockSwizzle, Stages, TransformA, TransformB, Operator, SplitKSerial> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape,
    WarpShape, 
    InstructionShape, 
    ElementA, LayoutA, 
    ElementB, LayoutB, 
    ElementAccumulator, layout::RowMajor, 
    arch::OpClassSimt,
    Stages,
    Operator,
    false,
    cutlass::arch::CacheOperation::Global,
    cutlass::arch::CacheOperation::Global,
    TransformA, 
    TransformB
  >;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, 
          typename MmaCore::IteratorThreadMapA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, 
          typename MmaCore::IteratorThreadMapB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using Mma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>;

  /// Define the epilogue
  using Epilogue =
    typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
        ThreadblockShape, 
        typename Mma::Operator, 
        EpilogueOutputOp,
        EpilogueOutputOp::kCount
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Multiply-add operator 
    // (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial
  >
struct DefaultGemmComplex<
  ElementA, LayoutA, ElementB, LayoutB, ElementC,
  layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
  EpilogueOutputOp, ThreadblockSwizzle, Stages, TransformA, TransformB, Operator, SplitKSerial> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
      ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
      layout::RowMajor, arch::OpClassTensorOp, arch::Sm80, ThreadblockShape,
      WarpShape, InstructionShape, Stages, TransformA, TransformB, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOp<
          ThreadblockShape, typename Mma::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Multiply-add operator 
    // (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial
  >
struct DefaultGemmComplex<
  ElementA, LayoutA, ElementB, LayoutB, ElementC,
  layout::RowMajor, ElementAccumulator, arch::OpClassSimt,
  arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
  EpilogueOutputOp, ThreadblockSwizzle, Stages, TransformA, TransformB, Operator, SplitKSerial> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
      ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
      layout::RowMajor, arch::OpClassSimt, arch::Sm80, ThreadblockShape,
      WarpShape, InstructionShape, Stages, TransformA, TransformB, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
    typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
        ThreadblockShape, 
        typename Mma::Operator, 
        EpilogueOutputOp,
        EpilogueOutputOp::kCount
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
