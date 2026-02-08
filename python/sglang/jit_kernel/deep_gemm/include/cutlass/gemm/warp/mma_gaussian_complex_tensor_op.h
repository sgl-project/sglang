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

#include "cutlass/array.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_gaussian_complex_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename RealElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename RealElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename RealElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Complex transform on A operand
  ComplexTransform TransformA = ComplexTransform::kNone,
  /// Complex transform on B operand
  ComplexTransform TransformB = ComplexTransform::kNone,
  /// Do source operands need more than one elements
  bool GeneralizedOperatorElements = false,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaGaussianComplexTensorOp;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex*complex+complex => complex using real-valued TensorOps
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename RealElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename RealElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename RealElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Complex transform on A operand
  ComplexTransform TransformA,
  /// Complex transform on B operand
  ComplexTransform TransformB
>
class MmaGaussianComplexTensorOp<
  Shape_, 
  complex<RealElementA>, 
  LayoutA_, 
  complex<RealElementB>,
  LayoutB_,
  complex<RealElementC>,
  LayoutC_,
  Policy_,
  TransformA,
  TransformB>  {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = complex<RealElementA>;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = complex<RealElementB>;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = complex<RealElementC>;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Underlying arch tag
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Indicates math operator 
  using MathOperator = arch::OpMultiplyAddGaussianComplex;
  
  /// Complex transform on A operand
  static ComplexTransform const kTransformA = TransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = TransformB;


  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,
    Operand::kA,
    ElementA,
    LayoutA,
    MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
    Policy::OpDelta::kRow,
    32,
    1
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA = FragmentA;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kK, Shape::kN>,
    Operand::kB,
    ElementB,
    LayoutB,
    MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
    Policy::OpDelta::kColumn,
    32,
    1
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB = FragmentB;

  static_assert(
    !(Shape::kM % ArchMmaOperator::Shape::kM) && 
    !(Shape::kN % ArchMmaOperator::Shape::kN),
    "Shape of warp-level Mma must be divisible by operator shape.");

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    Shape::kM / ArchMmaOperator::Shape::kM,
    Shape::kN / ArchMmaOperator::Shape::kN
  >;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpGaussianComplexAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, 
     ElementC, 
     LayoutC,
     typename ArchMmaOperator::Shape, 
     typename Policy::OpDelta>;

  /// Storage for C tile, the accumulator. Note, regardless of multiplicand type, this
  /// storage arrangement is to be considered 'gaussian complex' in the sense that the accumulation is
  /// done in three parts namely part1, part2, and part3. The parts 1, 2, and 3 are stored consecutively 
  /// in InteratorC::Frament. This matches the structure of Tensor Cores which are always real-valued matrix multiplies.
  using FragmentC = typename IteratorC::Fragment;

  static_assert(
    FragmentC::kElements == 3 * MmaIterations::kCount * ArchMmaOperator::FragmentC::kElements,
    "Unexpected gaussian complex fragment length.");

private:

  //
  // Data members
  //

  /// Underlying real-valued matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaGaussianComplexTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A, 
    FragmentB const &B, 
    FragmentC const &C
  ) const {

    // Alias types for underlying real-valued matrix multiply operator
    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    static_assert(MmaOperandA::kElements == 1, 
      "This implementation only supports math instructions in which exactly one element is needed for the A operand."
      "We can geneneralize later.");

    static_assert(MmaOperandB::kElements == 1, 
      "This implementation only supports math instructions in which exactly one element is needed for the B operand."
      "We can geneneralize later.");

    D = C;

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaIterations::kRow; ++m) {

      // mma(accum.part1(), (a.real() + a.imag()), b.real(), accum.part1());
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_Asum;
        MmaOperandB operand_Br;

        operand_Asum[0] = A[m].real() + ((kTransformA == ComplexTransform::kConjugate) ? -A[m].imag() : +A[m].imag());
        operand_Br[0] = B[n].real();

        // accumulator part1
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow);

        mma(*accum, operand_Asum, operand_Br, *accum);
      }

      // mma(accum.part2(), -a.real(), (b.real() - b.imag()), accum.part2()); 
      CUTLASS_PRAGMA_UNROLL
      for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_Ar;
        MmaOperandB operand_Bdiff;

        operand_Ar[0] = -A[m].real();
        operand_Bdiff[0] = B[n].real() - ((kTransformB == ComplexTransform::kConjugate) ? -B[n].imag() : +B[n].imag());

        // accumulator part2
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + MmaIterations::kCount;

        mma(*accum, operand_Ar, operand_Bdiff, *accum);
      }

      // mma(accum.part3(), a.imag(), (b.real() + b.imag()), accum.part3())
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_Ai;
        MmaOperandB operand_Bsum;

        operand_Ai[0] = (kTransformA == ComplexTransform::kConjugate) ? -A[m].imag() : +A[m].imag();
        operand_Bsum[0] = B[n].real() + ((kTransformB == ComplexTransform::kConjugate) ? -B[n].imag() : +B[n].imag());

        // accumulator part3
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + 2 * MmaIterations::kCount;

        mma(*accum, operand_Ai, operand_Bsum, *accum);
      }
    }
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {
    dst_A = A;
    dst_B = B;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex*complex+complex => complex using real-valued TensorOps
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename RealElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename RealElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename RealElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Complex transform on A operand
  ComplexTransform TransformA,
  /// Complex transform on B operand
  ComplexTransform TransformB
>
class MmaGaussianComplexTensorOp<
  Shape_, 
  complex<RealElementA>, 
  LayoutA_, 
  complex<RealElementB>,
  LayoutB_,
  complex<RealElementC>,
  LayoutC_,
  Policy_,
  TransformA,
  TransformB,
  true>  {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = complex<RealElementA>;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = complex<RealElementB>;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = complex<RealElementC>;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Underlying arch tag
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Indicates math operator 
  using MathOperator = arch::OpMultiplyAddGaussianComplex;
  
  /// Complex transform on A operand
  static ComplexTransform const kTransformA = TransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = TransformB;


  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,
    Operand::kA,
    ElementA,
    LayoutA,
    MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
    Policy::OpDelta::kRow,
    32,
    1
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA = FragmentA;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kK, Shape::kN>,
    Operand::kB,
    ElementB,
    LayoutB,
    MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
    Policy::OpDelta::kColumn,
    32,
    1
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB = FragmentB;

  static_assert(
    !(Shape::kM % ArchMmaOperator::Shape::kM) && 
    !(Shape::kN % ArchMmaOperator::Shape::kN),
    "Shape of warp-level Mma must be divisible by operator shape.");

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    Shape::kM / ArchMmaOperator::Shape::kM,
    Shape::kN / ArchMmaOperator::Shape::kN
  >;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpGaussianComplexAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, 
     ElementC, 
     LayoutC,
     typename ArchMmaOperator::Shape, 
     typename Policy::OpDelta>;

  /// Storage for C tile, the accumulator. Note, regardless of multiplicand type, this
  /// storage arrangement is to be considered 'gaussian complex' in the sense that the accumulation is
  /// done in three parts namely part1, part2, and part3. The parts 1, 2, and 3 are stored consecutively 
  /// in InteratorC::Frament. This matches the structure of Tensor Cores which are always real-valued matrix multiplies.
  using FragmentC = typename IteratorC::Fragment;

  static_assert(
    FragmentC::kElements == 3 * MmaIterations::kCount * ArchMmaOperator::FragmentC::kElements,
    "Unexpected gaussian complex fragment length.");

private:

  //
  // Data members
  //

  /// Underlying real-valued matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaGaussianComplexTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A, 
    FragmentB const &B, 
    FragmentC const &C
  ) const {

    // Alias types for underlying real-valued matrix multiply operator
    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaIterations::kRow; ++m) {

      // mma(accum.part1(), (a.real() + a.imag()), b.real(), accum.part1());
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_Asum;
        MmaOperandB operand_Br;

        CUTLASS_PRAGMA_UNROLL
        for (int mk = 0; mk < MmaOperandA::kElements; ++mk)
          operand_Asum[mk] = A[m*MmaOperandA::kElements + mk].real() + ((kTransformA == ComplexTransform::kConjugate) ?
                            -A[m*MmaOperandA::kElements + mk].imag() : +A[m*MmaOperandA::kElements + mk].imag());

        CUTLASS_PRAGMA_UNROLL
        for (int nk = 0; nk < MmaOperandB::kElements; ++nk)
          operand_Br[nk] = B[n*MmaOperandB::kElements + nk].real();

        // accumulator part1
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow);

        mma(*accum, operand_Asum, operand_Br, *accum);
      }

      // mma(accum.part2(), -a.real(), (b.real() - b.imag()), accum.part2()); 
      CUTLASS_PRAGMA_UNROLL
      for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_Ar;
        MmaOperandB operand_Bdiff;

        CUTLASS_PRAGMA_UNROLL
        for (int mk = 0; mk < MmaOperandA::kElements; ++mk)
          operand_Ar[mk] = -A[m*MmaOperandA::kElements + mk].real();

        CUTLASS_PRAGMA_UNROLL
        for (int nk = 0; nk < MmaOperandB::kElements; ++nk)
          operand_Bdiff[nk] = B[n*MmaOperandB::kElements + nk].real() - ((kTransformB == ComplexTransform::kConjugate) ?
                              -B[n*MmaOperandB::kElements + nk].imag() : +B[n*MmaOperandB::kElements + nk].imag());

        // accumulator part2
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + MmaIterations::kCount;

        mma(*accum, operand_Ar, operand_Bdiff, *accum);
      }

      // mma(accum.part3(), a.imag(), (b.real() + b.imag()), accum.part3())
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_Ai;
        MmaOperandB operand_Bsum;

        CUTLASS_PRAGMA_UNROLL
        for (int mk = 0; mk < MmaOperandA::kElements; ++mk)
          operand_Ai[mk] = (kTransformA == ComplexTransform::kConjugate) ?
                           -A[m*MmaOperandA::kElements + mk].imag() : +A[m*MmaOperandA::kElements + mk].imag();

        CUTLASS_PRAGMA_UNROLL
        for (int nk = 0; nk < MmaOperandB::kElements; ++nk)
          operand_Bsum[nk] = B[n*MmaOperandB::kElements + nk].real() + ((kTransformB == ComplexTransform::kConjugate) ?
                             -B[n*MmaOperandB::kElements + nk].imag() : +B[n*MmaOperandB::kElements + nk].imag());

        // accumulator part3
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + 2 * MmaIterations::kCount;

        mma(*accum, operand_Ai, operand_Bsum, *accum);
      }
    }
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {
    dst_A = A;
    dst_B = B;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
