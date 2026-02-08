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
      Default kernel-level RankK definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

  
*/

#pragma once

#include "cutlass/blas3.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/rank_k_universal.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/default_epilogue_complex_tensor_op_blas3.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif //CUTLASS_ARCH_WMMA_ENABLED


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
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC_,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Blas3 computation mode
    BlasMode BlasMode_ = BlasMode::kSymmetric>
struct DefaultRankKComplex;


////////////////////////////////////////////////////////////////////////////////
namespace detail {

template <
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Complex elementwise transformation 
  ComplexTransform TransformA,
  /// Blas3 computation mode (symmetric/hermitian)
  BlasMode BlasMode_
  > struct RankKTransposedComplexTransform {
  
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformA;

};
  
  // partial specializations for HERK CUBLAS_OP_N layout (ColumMajor)
template <>
  struct RankKTransposedComplexTransform <
  layout::ColumnMajor, 
  ComplexTransform::kNone,
  BlasMode::kHermitian> {

  static ComplexTransform const kTransformA = ComplexTransform::kConjugate;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

};

  // partial specializations for HERK CUBLAS_OP_C layout (RowMajor + Complex conjugate) 
template <>
  struct RankKTransposedComplexTransform <
  layout::RowMajor, 
  ComplexTransform::kConjugate,
  BlasMode::kHermitian> {

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kConjugate;

};

}
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Hopper Architecture complex datatype (symmetric)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultRankKComplex<
  ElementA, LayoutA, ElementC, 
  layout::RowMajor, FillModeC, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm90, ThreadblockShape, WarpShape, InstructionShape, 
  EpilogueOutputOp, ThreadblockSwizzle, Stages, 
  TransformA, Operator, SplitKSerial, BlasMode::kSymmetric> {

  static BlasMode const kBlasMode = BlasMode::kSymmetric;
  
  /// Define the threadblock-scoped matrix multiply-accumulate (A x B^T)
  using Mma = typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
      ElementA, LayoutA, 
      ElementA, typename layout::LayoutTranspose<LayoutA>::type, 
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm90, 
      ThreadblockShape, WarpShape, InstructionShape, Stages, 
      TransformA, TransformA, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOpBlas3<
          ThreadblockShape, typename Mma::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator, kBlasMode>::Epilogue;

  /// Define the kernel-level RankK operator.
  using RankKkernel = kernel::RankKUniversal<Mma, Epilogue, ThreadblockSwizzle, FillModeC>;

};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Hopper Architecture complex datatype (hermitian)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultRankKComplex<
  ElementA, LayoutA, ElementC, 
  layout::RowMajor, FillModeC, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm90, ThreadblockShape, WarpShape, InstructionShape, 
  EpilogueOutputOp, ThreadblockSwizzle, Stages, 
  TransformA, Operator, SplitKSerial, BlasMode::kHermitian> {

  static BlasMode const kBlasMode = BlasMode::kHermitian;

  // Complex transform for input A and B matrices (function on input layout)
  static ComplexTransform const kTransformA = TransformA;

  using TransposedComplexTransform = detail::RankKTransposedComplexTransform<
                                        LayoutA, 
                                        TransformA,
                                        kBlasMode>;

  // Complex transform on operandA and operandB (function of blas3 computation)
  static ComplexTransform const kTransformOperandA = TransposedComplexTransform::kTransformA;
  static ComplexTransform const kTransformOperandB = TransposedComplexTransform::kTransformB;

  /// Define the threadblock-scoped matrix multiply-accumulate (A x A^H)
  using Mma = typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
      ElementA, LayoutA, 
      ElementA, typename layout::LayoutTranspose<LayoutA>::type, 
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm90, 
      ThreadblockShape, WarpShape, InstructionShape, Stages, 
      kTransformOperandA, kTransformOperandB, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOpBlas3<
          ThreadblockShape, typename Mma::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator, kBlasMode>::Epilogue;

  /// Define the kernel-level RankK operator.
  using RankKkernel = kernel::RankKUniversal<Mma, Epilogue, ThreadblockSwizzle, FillModeC>;

};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture complex datatype (symmetric)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultRankKComplex<
  ElementA, LayoutA, ElementC, 
  layout::RowMajor, FillModeC, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, 
  EpilogueOutputOp, ThreadblockSwizzle, Stages, 
  TransformA, Operator, SplitKSerial, BlasMode::kSymmetric> {

  static BlasMode const kBlasMode = BlasMode::kSymmetric;
  
  /// Define the threadblock-scoped matrix multiply-accumulate (A x B^T)
  using Mma = typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
      ElementA, LayoutA, 
      ElementA, typename layout::LayoutTranspose<LayoutA>::type, 
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm80, 
      ThreadblockShape, WarpShape, InstructionShape, Stages, 
      TransformA, TransformA, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOpBlas3<
          ThreadblockShape, typename Mma::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator, kBlasMode>::Epilogue;

  /// Define the kernel-level RankK operator.
  using RankKkernel = kernel::RankKUniversal<Mma, Epilogue, ThreadblockSwizzle, FillModeC>;

};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture complex datatype (hermitian)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultRankKComplex<
  ElementA, LayoutA, ElementC, 
  layout::RowMajor, FillModeC, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, 
  EpilogueOutputOp, ThreadblockSwizzle, Stages, 
  TransformA, Operator, SplitKSerial, BlasMode::kHermitian> {

  static BlasMode const kBlasMode = BlasMode::kHermitian;

  // Complex transform for input A and B matrices (function on input layout)
  static ComplexTransform const kTransformA = TransformA;

  using TransposedComplexTransform = detail::RankKTransposedComplexTransform<
                                        LayoutA, 
                                        TransformA,
                                        kBlasMode>;

  // Complex transform on operandA and operandB (function of blas3 computation)
  static ComplexTransform const kTransformOperandA = TransposedComplexTransform::kTransformA;
  static ComplexTransform const kTransformOperandB = TransposedComplexTransform::kTransformB;

  /// Define the threadblock-scoped matrix multiply-accumulate (A x A^H)
  using Mma = typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
      ElementA, LayoutA, 
      ElementA, typename layout::LayoutTranspose<LayoutA>::type, 
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm80, 
      ThreadblockShape, WarpShape, InstructionShape, Stages, 
      kTransformOperandA, kTransformOperandB, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOpBlas3<
          ThreadblockShape, typename Mma::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator, kBlasMode>::Epilogue;

  /// Define the kernel-level RankK operator.
  using RankKkernel = kernel::RankKUniversal<Mma, Epilogue, ThreadblockSwizzle, FillModeC>;

};

////////////////////////////////////////////////////////////////////////////////


}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
