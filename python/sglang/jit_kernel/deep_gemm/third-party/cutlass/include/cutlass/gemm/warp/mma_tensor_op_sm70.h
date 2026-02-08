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

    This is a work in progress.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/mma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
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
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaVoltaTensorOp {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Architecture tag
  using ArchTag = arch::Sm70;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;
  
  /// Underlying instruction shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// interleaved 32x32 tiles
  using InterleavedTileShape = GemmShape<32, 32, 4>;

  static_assert(!(Shape::kM % InterleavedTileShape::kM) &&
                !(Shape::kN % InterleavedTileShape::kN),
                "Shape must be a multiple of InterleavedTileShape.");
public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaVoltaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,
    Operand::kA,
    ElementA,
    LayoutA,
    MatrixShape<
      ArchMmaOperator::Shape::kM,
      ArchMmaOperator::Shape::kK
    >,
    Policy::OpDelta::kRow,
    kThreadCount
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Iterates over the B operand in memory
  using IteratorB = MmaVoltaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kK, Shape::kN>,
    Operand::kB,
    ElementB,
    LayoutB,
    MatrixShape<
      ArchMmaOperator::Shape::kK,
      ArchMmaOperator::Shape::kN
    >,
    Policy::OpDelta::kRow,
    kThreadCount
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Iterates over the C operand in memory
  using IteratorC = MmaVoltaTensorOpAccumulatorTileIterator<
    MatrixShape<Shape::kM, Shape::kN>,
    ElementC,
    LayoutC,
    typename ArchMmaOperator::Shape,
    typename Policy::OpDelta
  >;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

private:

  static_assert(
    !(Shape::kM % ArchMmaOperator::Shape::kM) && 
    !(Shape::kN % ArchMmaOperator::Shape::kN),
    "Shape of warp-level Mma must be divisible by operator shape.");

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    InterleavedTileShape::kM / ArchMmaOperator::Shape::kM,
    InterleavedTileShape::kN / ArchMmaOperator::Shape::kN
  >;
  using TileIterations = MatrixShape<
    Shape::kM / InterleavedTileShape::kM,
    Shape::kN / InterleavedTileShape::kN
  >;

  // Whether matrix B is reordered
  bool reorder_B_;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //
  
  /// Ctor
  CUTLASS_DEVICE
  MmaVoltaTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A, 
    FragmentB const &B, 
    FragmentC const &C)  {

    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    CUTLASS_PRAGMA_UNROLL
    for (int outer_col = 0; outer_col < TileIterations::kColumn; ++outer_col) {
      CUTLASS_PRAGMA_UNROLL
      for (int inner_col = 0; inner_col < MmaIterations::kColumn; ++inner_col) {
        CUTLASS_PRAGMA_UNROLL
        for (int outer_row = 0; outer_row < TileIterations::kRow; ++outer_row) {
          CUTLASS_PRAGMA_UNROLL

          for (int inner_row = 0; inner_row < MmaIterations::kRow; ++inner_row) {
      
            int op_col = inner_col + MmaIterations::kColumn * outer_col;

            // Column-major serpentine sequence to maximize reuse of A operand.
            int inner_row_serp = inner_row;
            int outer_row_serp = outer_row;
            if (op_col & 1) {
              inner_row_serp = MmaIterations::kRow - inner_row - 1;
              outer_row_serp = TileIterations::kRow - outer_row - 1;
            }
            int op_row = inner_row_serp + MmaIterations::kRow * outer_row_serp;
            int op_idx = inner_row_serp + MmaIterations::kRow * 
                         (inner_col + MmaIterations::kColumn * 
                          (outer_row_serp + TileIterations::kRow * outer_col));
            mma(
              ptr_D[op_idx],
              ptr_A[op_row],
              ptr_B[op_col],
              ptr_D[op_idx]);

          }
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass
