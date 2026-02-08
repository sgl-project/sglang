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
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/vector_iterator.h"
#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"

#include "kernel/b2b_gemm.h"
#include "threadblock/default_b2b_mma.h"
#include "threadblock/default_b2b_mma_smem_accumulator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

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
    typename ThreadblockShape0,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp0,
    /// Epilogue output operator
    typename EpilogueOutputOp1,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultB2bGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
                   layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
                   arch::Sm80, ThreadblockShape0, ThreadblockShape1,
                   WarpShape0, WarpShape1, InstructionShape,
                   EpilogueOutputOp0, EpilogueOutputOp1, ThreadblockSwizzle, Stages,
                   Operator, true> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
      InstructionShape, Stages, Operator, EpilogueOutputOp0, false, true>::ThreadblockB2bMma;

  static const int kPartitionsK1 = ThreadblockShape1::kK / WarpShape1::kK;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape1, typename B2bMma::Operator1, kPartitionsK1, EpilogueOutputOp1,
          EpilogueOutputOp1::kCount>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle>;
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
  typename ThreadblockShape0,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape1,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape0,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape1,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  /// Epilogue output operator
  typename EpilogueOutputOp0,
  /// Epilogue output operator
  typename EpilogueOutputOp1,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// Operation performed by GEMM
  typename Operator
>
struct DefaultB2bGemm<
  ElementA, LayoutA, kAlignmentA,
  ElementB, LayoutB, kAlignmentB,
  ElementC, layout::RowMajor,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm75,
  ThreadblockShape0,
  ThreadblockShape1,
  WarpShape0,
  WarpShape1,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  ThreadblockSwizzle,
  2,
  Operator,
  true
> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<
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
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    2,
    Operator,
    EpilogueOutputOp0,
    false,
    true
  >::ThreadblockB2bMma;

  static const int kPartitionsK1 = ThreadblockShape1::kK / WarpShape1::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape1,
    typename B2bMma::Operator1,
    kPartitionsK1,
    EpilogueOutputOp1,
    EpilogueOutputOp1::kCount
  >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle>;
};


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
    typename ThreadblockShape0,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp0,
    /// Epilogue output operator
    typename EpilogueOutputOp1,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Number of Interleaved k
    int InterleavedK,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultB2bGemm<
    ElementA, layout::ColumnMajorInterleaved<InterleavedK>, kAlignmentA,
    ElementB, layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
    ElementC, layout::ColumnMajorInterleaved<InterleavedK>, int32_t,
    arch::OpClassTensorOp, arch::Sm80,
    ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
    InstructionShape, EpilogueOutputOp0, EpilogueOutputOp1,
    ThreadblockSwizzle, Stages,
    Operator, true> {
  using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
  using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
  using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

  using ElementAccumulator = int32_t;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
      InstructionShape, Stages, Operator, EpilogueOutputOp0,
      true, true>::ThreadblockB2bMma;

  static const int kPartitionsK1 = ThreadblockShape1::kK / WarpShape1::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultInterleavedEpilogueTensorOp<
          ThreadblockShape1, typename B2bMma::Operator1, kPartitionsK1, EpilogueOutputOp1,
          64 / sizeof_bits<ElementC>::value, InterleavedK>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle>;
};

////////////////////////////////////////////////////////////////////////////////


/// Partial specialization for Turing Integer Tensor Core Interleaved layout
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
    typename ThreadblockShape0,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape0,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape1,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp0,
    /// Epilogue output operator
    typename EpilogueOutputOp1,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of Interleaved k
    int InterleavedK,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultB2bGemm<ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
                   kAlignmentA, ElementB,
                   layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
                   ElementC, layout::ColumnMajorInterleaved<InterleavedK>,
                   int32_t, arch::OpClassTensorOp, arch::Sm75,
                   ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
                   InstructionShape, EpilogueOutputOp0, EpilogueOutputOp1,
                   ThreadblockSwizzle, 2, Operator, true> {
  using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
  using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
  using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

  using ElementAccumulator = int32_t;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm75,
      ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
      InstructionShape, 2, Operator, EpilogueOutputOp0, true, true>::ThreadblockB2bMma;

  static const int kPartitionsK1 = ThreadblockShape1::kK / WarpShape1::kK;

  /// Define the epilogue for the 2nd Gemm
  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultInterleavedEpilogueTensorOp<
          ThreadblockShape1, typename B2bMma::Operator1, kPartitionsK1, EpilogueOutputOp1,
          64 / sizeof_bits<ElementC>::value, InterleavedK>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle>;
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
