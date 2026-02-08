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
    \brief Default kernel-level Blocked-Ell sparse gemm operators.
      This operator combines threadblock-scoped ELL MMA
      with the appropriate threadblock-scoped epilogue.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif //CUTLASS_ARCH_WMMA_ENABLED

#include "cutlass/gemm/kernel/ell_gemm.h"
#include "cutlass/gemm/threadblock/default_ell_mma.h"

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
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
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
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse>
struct DefaultEllGemm;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
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
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse
>
struct DefaultEllGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
                   layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
                   arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
                   EpilogueOutputOp, ThreadblockSwizzle, Stages, SplitKSerial,
                   Operator, IsASparse> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, Stages,
      Operator>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Turing Architecture
template <
  /// Element type for A matrix operand
  typename ElementA,
  /// Layout type for A matrix operand
  typename LayoutA,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB,
  /// Layout type for B matrix operand
  typename LayoutB,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB,
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
  /// If true, kernel is configured to support serial reduction in the epilogue
  bool SplitKSerial,
  /// Operation performed by GEMM
  typename Operator,
  /// Sparse matrix is A or not
  bool IsASparse
>
struct DefaultEllGemm<
  ElementA, LayoutA, kAlignmentA,
  ElementB, LayoutB, kAlignmentB,
  ElementC, layout::RowMajor,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm75,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  SplitKSerial,
  Operator,
  IsASparse
> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    layout::RowMajor,
    arch::OpClassTensorOp,
    arch::Sm75,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    2,
    Operator
  >::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    typename Mma::Operator,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Integer Matrix Multiply Interleaved layout
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
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
    /// Number of Interleaved k
    int InterleavedK,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse>
struct DefaultEllGemm<
    ElementA, layout::ColumnMajorInterleaved<InterleavedK>, kAlignmentA,
    ElementB, layout::RowMajorInterleaved<InterleavedK>, kAlignmentB, ElementC,
    layout::ColumnMajorInterleaved<InterleavedK>, int32_t,
    arch::OpClassTensorOp, arch::Sm80, ThreadblockShape, WarpShape,
    InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages,
    SplitKSerial, Operator, IsASparse> {
  using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
  using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
  using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

  using ElementAccumulator = int32_t;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, Stages, Operator,
      true>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultInterleavedEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          64 / sizeof_bits<ElementC>::value, InterleavedK>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Turing Integer Matrix Multiply Interleaved layout
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
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
    /// Number of Interleaved k
    int InterleavedK,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse>
struct DefaultEllGemm<ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
                   kAlignmentA, ElementB,
                   layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
                   ElementC, layout::ColumnMajorInterleaved<InterleavedK>,
                   int32_t, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
                   WarpShape, InstructionShape, EpilogueOutputOp,
                   ThreadblockSwizzle, 2, SplitKSerial, Operator, IsASparse> {
  using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
  using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
  using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

  using ElementAccumulator = int32_t;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
      arch::OpClassTensorOp, arch::Sm75, ThreadblockShape, WarpShape,
      InstructionShape, 2, Operator, true>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultInterleavedEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          64 / sizeof_bits<ElementC>::value, InterleavedK>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};

////////////////////////////////////////////////////////////////////////////////


/// Partial specialization for Volta architecture
template <
  /// Element type for A matrix operand
  typename ElementA,
  /// Layout type for A matrix operand
  typename LayoutA,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB,
  /// Layout type for B matrix operand
  typename LayoutB,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// If true, kernel is configured to support serial reduction in the epilogue
  bool SplitKSerial,
  /// Operation performed by GEMM
  typename Operator,
  /// Sparse matrix is A or not
  bool IsASparse
>
struct DefaultEllGemm<
  ElementA, LayoutA, kAlignmentA,
  ElementB, LayoutB, kAlignmentB,
  ElementC, layout::RowMajor,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm70,
  ThreadblockShape,
  WarpShape,
  GemmShape<8, 8, 4>,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  SplitKSerial,
  Operator,
  IsASparse
> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    layout::RowMajor,
    arch::OpClassTensorOp,
    arch::Sm70,
    ThreadblockShape,
    WarpShape,
    GemmShape<8, 8, 4>,
    2,
    Operator
  >::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
    ThreadblockShape,
    typename Mma::Operator,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for SIMT
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse
  >
struct DefaultEllGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    layout::RowMajor,
    ElementAccumulator,
    arch::OpClassSimt,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    GemmShape<1, 1, 1>,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    2,
    SplitKSerial,
    Operator,
    IsASparse> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      layout::RowMajor,
      arch::OpClassSimt,
      arch::Sm50,
      ThreadblockShape,
      WarpShape,
      GemmShape<1, 1, 1>,
      2,
      Operator>::ThreadblockMma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages
    int Stages,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator, 
    /// Sparse matrix is A or not
    bool IsASparse
    >
struct DefaultEllGemm<ElementA,
                   LayoutA,
                   kAlignmentA,
                   ElementB,
                   LayoutB,
                   kAlignmentB,
                   ElementC,
                   layout::RowMajor,
                   ElementAccumulator,
                   arch::OpClassSimt,
                   arch::Sm80,
                   ThreadblockShape,
                   WarpShape,
                   GemmShape<1, 1, 1>,
                   EpilogueOutputOp,
                   ThreadblockSwizzle,
                   Stages,
                   SplitKSerial,
                   Operator,
                   IsASparse> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, layout::RowMajor, arch::OpClassSimt, arch::Sm80,
      ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, Stages,
      Operator>::ThreadblockMma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial,IsASparse>;
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for SIMT DP4A

template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Layout type for C matrix operand
    typename LayoutC,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse
    >
struct DefaultEllGemm<int8_t, LayoutA, kAlignmentA, int8_t, LayoutB, kAlignmentB,
                   ElementC, LayoutC, ElementAccumulator, arch::OpClassSimt,
                   ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 4>,
                   EpilogueOutputOp, ThreadblockSwizzle, 2, SplitKSerial,
                   Operator, IsASparse> {
  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using ElementB = int8_t;

  using OperatorClass =  arch::OpClassSimt;
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      arch::OpClassSimt,
      arch::Sm50,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      2,
      Operator
      >::ThreadblockMma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Wmma Gemm Kernel
template <
    ///< Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
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
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse
    > 
struct DefaultEllGemm<
  ElementA, LayoutA, kAlignmentA, 
  ElementB, LayoutB, kAlignmentB, 
  ElementC, LayoutC, 
  ElementAccumulator, 
  arch::OpClassWmmaTensorOp,
  ArchTag, 
  ThreadblockShape, WarpShape, InstructionShape,
  EpilogueOutputOp, 
  ThreadblockSwizzle, 
  Stages, 
  SplitKSerial,
  Operator,
  IsASparse> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultEllMma<
      ElementA, LayoutA, kAlignmentA,
      ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, 
      arch::OpClassWmmaTensorOp, 
      ArchTag,
      ThreadblockShape, 
      WarpShape, 
      InstructionShape, 
      Stages,
      Operator>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue 
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWmmaTensorOp<
      ThreadblockShape,
      typename Mma::Operator, 
      kPartitionsK, 
      EpilogueOutputOp,
      EpilogueOutputOp::kCount
  >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};
////////////////////////////////////////////////////////////////////////////////
#endif //CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
