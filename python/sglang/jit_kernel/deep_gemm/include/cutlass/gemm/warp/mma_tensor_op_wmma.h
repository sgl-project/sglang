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
    \brief Templates implementing warp-level matrix multiply-accumulate operations targeting
      Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include "cutlass/wmma_array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_wmma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

///< Structure to compute the matrix product targeting CUDA cores via WMMA.
template < 
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  ///< Data type of A elements
  typename ElementA_,
  ///< Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  ///< Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  ///< Element type of C matrix
  typename ElementC_,
  ///< Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  ///< Policy describing warp-level Wmma operation (concept: MmaTensorOpPolicy)
  typename Policy_,
  ///< Number of partitions along K dimension
  int PartitionsK_ = 1,
  ///< Used for partial specialization
  typename Enable = bool
>
class MmaTensorOpWmma {
public:
  ///< Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  ///< Data type of multiplicand A
  using ElementA = ElementA_;

  ///< Layout of multiplicand A
  using LayoutA = LayoutA_;

  ///< Data type of multiplicand B
  using ElementB = ElementB_;

  ///< Layout of multiplicand B
  using LayoutB = LayoutB_;

  ///< Data type of accumulator matrix C
  using ElementC = ElementC_;

  ///< Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaTensorOpPolicy)
  using Policy = Policy_;

  /// Underlying instruction shape
  using InstructionShape = typename Policy::Operator::Shape;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;
  
  /// Underlying architecture tag
  using ArchTag = typename Policy::Operator::ArchTag;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassWmmaTensorOp;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpWmmaMultiplicandTileIterator<
     MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
     Policy::OpDelta::kRow, kThreadCount, Policy>;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpWmmaMultiplicandTileIterator<
     MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
     Policy::OpDelta::kRow, kThreadCount, Policy>;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpWmmaAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
    typename Policy::OpDelta, Policy>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

private:

  static_assert(
    !(Shape::kM % Policy::Operator::Shape::kM) && 
    !(Shape::kN % Policy::Operator::Shape::kN),
    "Shape of warp-level Wmma must be divisible by operator shape (wmma native size)");

  /// Number of wmma operations performed
  using WmmaIterations = MatrixShape<
    Shape::kM / Policy::Operator::Shape::kM,
    Shape::kN / Policy::Operator::Shape::kN 
  >;

public:

  /// Underlying matrix multiply operator (concept: cutlass::arch::Wmma)
  typename Policy::Operator wmma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaTensorOpWmma() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A, 
    FragmentB const &B, 
    FragmentC const &C) const {

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < WmmaIterations::kColumn; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < WmmaIterations::kRow; ++m) {

        // accumulate wmma mma
        wmma(D[m * WmmaIterations::kColumn + n], A[m], B[n], C[m * WmmaIterations::kColumn + n]);
      }
    }  
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

#endif // if defined(CUTLASS_ARCH_WMMA_ENABLED)

