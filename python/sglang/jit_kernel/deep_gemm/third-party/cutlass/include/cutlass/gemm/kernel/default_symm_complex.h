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
      Default kernel-level SYMM/HEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

  
*/

#pragma once

#include "cutlass/blas3.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/symm_universal.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_multistage_trmm_complex.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/default_epilogue_complex_tensor_op.h"
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
    /// Side Mode for A (kLeft or kRight)
    SideMode kSideModeA,
    /// Fill Mode for A (kLower or kUpper)
    FillMode kFillModeA,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Blas3 computation mode
    BlasMode BlasMode_ = BlasMode::kSymmetric>
struct DefaultSymmComplex;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Hopper Architecture complex datatype (symmetric)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Side Mode for A (kLeft or kRight)
    SideMode kSideModeA,
    /// Fill Mode for A (kLower or kUpper)
    FillMode kFillModeA,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultSymmComplex<
  ElementA, LayoutA, kSideModeA, kFillModeA, ElementB, LayoutB, ElementC, 
  layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm90, ThreadblockShape, WarpShape, InstructionShape, 
  EpilogueOutputOp, ThreadblockSwizzle, Stages, 
  Operator, SplitKSerial, BlasMode::kSymmetric> {

  static BlasMode const kBlasMode = BlasMode::kSymmetric;
  // Complex Transform don't apply to A or B for SYMM
  static ComplexTransform const TransformA = ComplexTransform::kNone; 
  static ComplexTransform const TransformB = ComplexTransform::kNone; 

  /// Define the threadblock-scoped triagular matrix multiply-accumulate
  /// TRMM - with diagonal: alpha * A * B or alpha * B * A
	static const DiagType kDiagTypeMma1 = DiagType::kNonUnit;
  using Mma1 = typename cutlass::gemm::threadblock::DefaultMultistageTrmmComplex<
      ElementA, LayoutA, 
      ElementB, LayoutB, 
      kSideModeA, kFillModeA, kDiagTypeMma1, 
      ElementAccumulator, layout::RowMajor, 
      arch::OpClassTensorOp, arch::Sm90,
      ThreadblockShape, WarpShape, InstructionShape,
      Stages, TransformA, TransformB, Operator>::ThreadblockMma;

  /// Define the threadblock-scoped triagular matrix multiply-accumulate
  /// TRMM - withOUT diagonal: alpha * AT * B or alpha * B * AT
	static const DiagType kDiagTypeMma2 = DiagType::kZero;
  using LayoutAMma2 = typename platform::conditional<
                                (kSideModeA == SideMode::kLeft), 
                                typename layout::LayoutTranspose<LayoutA>::type, 
                                LayoutA
                              >::type;
  using LayoutBMma2 = typename platform::conditional<
                                (kSideModeA == SideMode::kLeft), 
                                LayoutB, 
                                typename layout::LayoutTranspose<LayoutB>::type
                              >::type; 
	using Mma2 = typename cutlass::gemm::threadblock::DefaultMultistageTrmmComplex<
			ElementA, LayoutAMma2, 
			ElementB, LayoutBMma2, 
			kSideModeA, InvertFillMode<kFillModeA>::mode, kDiagTypeMma2, 
			ElementAccumulator, layout::RowMajor, 
			arch::OpClassTensorOp, arch::Sm90,
			ThreadblockShape, WarpShape, InstructionShape,
			Stages, TransformA, TransformB, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOp<
          ThreadblockShape, typename Mma1::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator>::Epilogue;

  /// Define the kernel-level Symm operator.
  using SymmKernel = kernel::SymmUniversal<Mma1, Mma2, Epilogue, ThreadblockSwizzle, kSideModeA, kFillModeA>;

};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Hopper Architecture complex datatype (hermitian)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Side Mode for A (kLeft or kRight)
    SideMode kSideModeA,
    /// Fill Mode for A (kLower or kUpper)
    FillMode kFillModeA,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultSymmComplex<
  ElementA, LayoutA, kSideModeA, kFillModeA, ElementB, LayoutB, ElementC, 
  layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm90, ThreadblockShape, WarpShape, InstructionShape, 
  EpilogueOutputOp, ThreadblockSwizzle, Stages, 
  Operator, SplitKSerial, BlasMode::kHermitian> {

  static BlasMode const kBlasMode = BlasMode::kHermitian;


  /// Define the threadblock-scoped triagular matrix multiply-accumulate
  /// TRMM - with diagonal: alpha * A * B or alpha * B * A
	static const DiagType kDiagTypeMma1 = DiagType::kNonUnit;
  static ComplexTransform const TransformAMma1 = ComplexTransform::kNone; 
  static ComplexTransform const TransformBMma1 = ComplexTransform::kNone; 
  using Mma1 = typename cutlass::gemm::threadblock::DefaultMultistageTrmmComplex<
      ElementA, LayoutA, 
      ElementB, LayoutB, 
      kSideModeA, kFillModeA, kDiagTypeMma1, 
      ElementAccumulator, layout::RowMajor, 
      arch::OpClassTensorOp, arch::Sm90,
      ThreadblockShape, WarpShape, InstructionShape,
      Stages, TransformAMma1, TransformBMma1, Operator, BlasMode::kHermitian>::ThreadblockMma;

  /// Define the threadblock-scoped triagular matrix multiply-accumulate
  /// TRMM - withOUT diagonal - with conjugate transpose: alpha * AT * B or alpha * B * AT
	static const DiagType kDiagTypeMma2 = DiagType::kZero;
  using LayoutAMma2 = typename platform::conditional<
                                (kSideModeA == SideMode::kLeft), 
                                typename layout::LayoutTranspose<LayoutA>::type, 
                                LayoutA
                              >::type;
  using LayoutBMma2 = typename platform::conditional<
                                (kSideModeA == SideMode::kLeft), 
                                LayoutB, 
                                typename layout::LayoutTranspose<LayoutB>::type
                              >::type;
  static ComplexTransform const TransformAMma2 = (kSideModeA == SideMode::kLeft) ? 
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;
  static ComplexTransform const TransformBMma2 = (kSideModeA == SideMode::kLeft) ? 
                                              ComplexTransform::kNone : ComplexTransform::kConjugate;

	using Mma2 = typename cutlass::gemm::threadblock::DefaultMultistageTrmmComplex<
			ElementA, LayoutAMma2, 
			ElementB, LayoutBMma2, 
			kSideModeA, InvertFillMode<kFillModeA>::mode, kDiagTypeMma2, 
			ElementAccumulator, layout::RowMajor, 
			arch::OpClassTensorOp, arch::Sm90,
			ThreadblockShape, WarpShape, InstructionShape,
			Stages, TransformAMma2, TransformBMma2, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOp<
          ThreadblockShape, typename Mma1::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator>::Epilogue;

  /// Define the kernel-level Symm operator.
  using SymmKernel = kernel::SymmUniversal<Mma1, Mma2, Epilogue, ThreadblockSwizzle, kSideModeA, kFillModeA>;

};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture complex datatype (symmetric)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Side Mode for A (kLeft or kRight)
    SideMode kSideModeA,
    /// Fill Mode for A (kLower or kUpper)
    FillMode kFillModeA,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultSymmComplex<
  ElementA, LayoutA, kSideModeA, kFillModeA, ElementB, LayoutB, ElementC, 
  layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, 
  EpilogueOutputOp, ThreadblockSwizzle, Stages, 
  Operator, SplitKSerial, BlasMode::kSymmetric> {

  static BlasMode const kBlasMode = BlasMode::kSymmetric;
  // Complex Transform don't apply to A or B for SYMM
  static ComplexTransform const TransformA = ComplexTransform::kNone; 
  static ComplexTransform const TransformB = ComplexTransform::kNone; 

  /// Define the threadblock-scoped triagular matrix multiply-accumulate
  /// TRMM - with diagonal: alpha * A * B or alpha * B * A
	static const DiagType kDiagTypeMma1 = DiagType::kNonUnit;
  using Mma1 = typename cutlass::gemm::threadblock::DefaultMultistageTrmmComplex<
      ElementA, LayoutA, 
      ElementB, LayoutB, 
      kSideModeA, kFillModeA, kDiagTypeMma1, 
      ElementAccumulator, layout::RowMajor, 
      arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape,
      Stages, TransformA, TransformB, Operator>::ThreadblockMma;

  /// Define the threadblock-scoped triagular matrix multiply-accumulate
  /// TRMM - withOUT diagonal: alpha * AT * B or alpha * B * AT
	static const DiagType kDiagTypeMma2 = DiagType::kZero;
  using LayoutAMma2 = typename platform::conditional<
                                (kSideModeA == SideMode::kLeft), 
                                typename layout::LayoutTranspose<LayoutA>::type, 
                                LayoutA
                              >::type;
  using LayoutBMma2 = typename platform::conditional<
                                (kSideModeA == SideMode::kLeft), 
                                LayoutB, 
                                typename layout::LayoutTranspose<LayoutB>::type
                              >::type; 
	using Mma2 = typename cutlass::gemm::threadblock::DefaultMultistageTrmmComplex<
			ElementA, LayoutAMma2, 
			ElementB, LayoutBMma2, 
			kSideModeA, InvertFillMode<kFillModeA>::mode, kDiagTypeMma2, 
			ElementAccumulator, layout::RowMajor, 
			arch::OpClassTensorOp, arch::Sm80,
			ThreadblockShape, WarpShape, InstructionShape,
			Stages, TransformA, TransformB, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOp<
          ThreadblockShape, typename Mma1::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator>::Epilogue;

  /// Define the kernel-level Symm operator.
  using SymmKernel = kernel::SymmUniversal<Mma1, Mma2, Epilogue, ThreadblockSwizzle, kSideModeA, kFillModeA>;

};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture complex datatype (hermitian)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Side Mode for A (kLeft or kRight)
    SideMode kSideModeA,
    /// Fill Mode for A (kLower or kUpper)
    FillMode kFillModeA,
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
    /// Operation performed by GEMM
    typename Operator,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultSymmComplex<
  ElementA, LayoutA, kSideModeA, kFillModeA, ElementB, LayoutB, ElementC, 
  layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
  arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, 
  EpilogueOutputOp, ThreadblockSwizzle, Stages, 
  Operator, SplitKSerial, BlasMode::kHermitian> {

  static BlasMode const kBlasMode = BlasMode::kHermitian;


  /// Define the threadblock-scoped triagular matrix multiply-accumulate
  /// TRMM - with diagonal: alpha * A * B or alpha * B * A
	static const DiagType kDiagTypeMma1 = DiagType::kNonUnit;
  static ComplexTransform const TransformAMma1 = ComplexTransform::kNone; 
  static ComplexTransform const TransformBMma1 = ComplexTransform::kNone; 
  using Mma1 = typename cutlass::gemm::threadblock::DefaultMultistageTrmmComplex<
      ElementA, LayoutA, 
      ElementB, LayoutB, 
      kSideModeA, kFillModeA, kDiagTypeMma1, 
      ElementAccumulator, layout::RowMajor, 
      arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape,
      Stages, TransformAMma1, TransformBMma1, Operator, BlasMode::kHermitian>::ThreadblockMma;

  /// Define the threadblock-scoped triagular matrix multiply-accumulate
  /// TRMM - withOUT diagonal - with conjugate transpose: alpha * AT * B or alpha * B * AT
	static const DiagType kDiagTypeMma2 = DiagType::kZero;
  using LayoutAMma2 = typename platform::conditional<
                                (kSideModeA == SideMode::kLeft), 
                                typename layout::LayoutTranspose<LayoutA>::type, 
                                LayoutA
                              >::type;
  using LayoutBMma2 = typename platform::conditional<
                                (kSideModeA == SideMode::kLeft), 
                                LayoutB, 
                                typename layout::LayoutTranspose<LayoutB>::type
                              >::type;
  static ComplexTransform const TransformAMma2 = (kSideModeA == SideMode::kLeft) ? 
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;
  static ComplexTransform const TransformBMma2 = (kSideModeA == SideMode::kLeft) ? 
                                              ComplexTransform::kNone : ComplexTransform::kConjugate;

	using Mma2 = typename cutlass::gemm::threadblock::DefaultMultistageTrmmComplex<
			ElementA, LayoutAMma2, 
			ElementB, LayoutBMma2, 
			kSideModeA, InvertFillMode<kFillModeA>::mode, kDiagTypeMma2, 
			ElementAccumulator, layout::RowMajor, 
			arch::OpClassTensorOp, arch::Sm80,
			ThreadblockShape, WarpShape, InstructionShape,
			Stages, TransformAMma2, TransformBMma2, Operator>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueComplexTensorOp<
          ThreadblockShape, typename Mma1::Operator, 1, EpilogueOutputOp,
          EpilogueOutputOp::kCount, Operator>::Epilogue;

  /// Define the kernel-level Symm operator.
  using SymmKernel = kernel::SymmUniversal<Mma1, Mma2, Epilogue, ThreadblockSwizzle, kSideModeA, kFillModeA>;

};

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
