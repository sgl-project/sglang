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
      Default kernel-level grouped Rank2K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/kernel/rank_2k_transpose_operands.h"
#include "cutlass/gemm/kernel/default_rank_2k.h"
#include "cutlass/gemm/kernel/default_rank_2k_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
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
    /// Blas3 computation mode
    BlasMode BlasMode_ = BlasMode::kSymmetric,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_ = GroupScheduleMode::kDeviceOnly,
    ///
    typename Enable = void
    >
struct DefaultRank2KGrouped;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued grouped Rank2K
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
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
    /// Blas3 computation mode
    BlasMode BlasMode_,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_
    >
struct DefaultRank2KGrouped<ElementA, LayoutA, TransformA, kAlignmentA,
          ElementB, LayoutB, TransformB, kAlignmentB,
          ElementC, LayoutC,
          FillModeC, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
          WarpShape, InstructionShape, EpilogueOutputOp,
          ThreadblockSwizzle, Stages, Operator, BlasMode_, GroupScheduleMode_,
          typename platform::enable_if< ! cutlass::is_complex<ElementAccumulator>::value>::type
> {
  // If true, we must construct a 'transposed-and-exchanged' Rank2K operator.
  static bool const kInternalTranspose = platform::is_same<LayoutC, layout::ColumnMajor>::value;

  using MapArguments = kernel::detail::Rank2KMapArguments<
    ElementA,
    LayoutA,
    TransformA,
    kAlignmentA,
    ElementB,
    LayoutB,
    TransformB,
    kAlignmentB,
    LayoutC,
    FillModeC,
    kInternalTranspose
  >;

  // Define the default grouped Rank2K kernel
  using DefaultRank2Kkernel = typename kernel::DefaultRank2K<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    MapArguments::kFillModeC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    false,                  // SplitKSerial
    Operator,
    BlasMode_
  >::Rank2Kkernel;

  /// Define the kernel in terms of the default kernel
  using Rank2Kkernel = kernel::Rank2KGrouped<
    typename DefaultRank2Kkernel::Mma1,
    typename DefaultRank2Kkernel::Mma2,
    typename DefaultRank2Kkernel::Epilogue,
    ThreadblockSwizzle,
    TransformA,
    TransformB,
    DefaultRank2Kkernel::kFillModeC,
    DefaultRank2Kkernel::kBlasMode,
    GroupScheduleMode_,
    kInternalTranspose
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Complex-valued grouped Rank2K
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
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
    /// Blas3 computation mode
    BlasMode BlasMode_,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_
    >
struct DefaultRank2KGrouped<ElementA, LayoutA, TransformA, kAlignmentA,
          ElementB, LayoutB, TransformB, kAlignmentB,
          ElementC, LayoutC,
          FillModeC, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
          WarpShape, InstructionShape, EpilogueOutputOp,
          ThreadblockSwizzle, Stages, Operator, BlasMode_, GroupScheduleMode_,
          typename platform::enable_if<cutlass::is_complex<ElementAccumulator>::value>::type
> {
  // If true, we must construct a 'transposed-and-exchanged' Rank2K operator.
  static bool const kInternalTranspose = platform::is_same<LayoutC, layout::ColumnMajor>::value;

  using MapArguments = kernel::detail::Rank2KMapArguments<
    ElementA,
    LayoutA,
    TransformA,
    kAlignmentA,
    ElementB,
    LayoutB,
    TransformB,
    kAlignmentB,
    LayoutC,
    FillModeC,
    kInternalTranspose
  >;

  // Define the default grouped Rank2K kernel
  using DefaultRank2Kkernel = typename kernel::DefaultRank2KComplex<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    ElementC,
    typename MapArguments::LayoutC,
    MapArguments::kFillModeC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    MapArguments::kTransformA,
    MapArguments::kTransformB,
    Operator,
    false,                  // SplitKSerial
    BlasMode_
  >::Rank2Kkernel;

  /// Define the kernel in terms of the default kernel
  /// Pass through the user-provided TransformA and TransformB so as to
  /// correctly set public-facing TransformA and TransformB in kernel::Rank2KGrouped.
  /// This is needed because kernel::DefaultRank2KComplex may change TransformA and
  /// TransformB that become template arguments to Mma1 and Mma2.
  using Rank2Kkernel = kernel::Rank2KGrouped<
    typename DefaultRank2Kkernel::Mma1,
    typename DefaultRank2Kkernel::Mma2,
    typename DefaultRank2Kkernel::Epilogue,
    ThreadblockSwizzle,
    TransformA,
    TransformB,
    DefaultRank2Kkernel::kFillModeC,
    DefaultRank2Kkernel::kBlasMode,
    GroupScheduleMode_,
    kInternalTranspose
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
