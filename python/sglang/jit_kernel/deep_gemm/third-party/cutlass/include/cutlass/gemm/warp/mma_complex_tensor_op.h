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
#include "cutlass/functional.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/arch/mma_sm90.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"
#include "cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  /// Data type of real & imag members of complex numbers in the SourceFragment
  typename RealElement,
  /// Destination fragment required by the mma operation 
  typename DestinationFragment,
  /// Source fragment holding complex<RealElement> elements
  typename SourceFragment,
  /// Number of mma operations performed
  typename MmaIterations,
  /// Shape of operand elements
  typename MmaOperandShape,
  /// Complex transform on A operand
  ComplexTransform Transform_,
  /// Operand A or Operand B
  Operand Operand_,
  /// Floating-point rounding style
  FloatRoundStyle Round_>
struct UnpackComplexConvertAndPackForMma;

// Partial specialization for OperandA and Congruous smem layout
template <
  typename RealElement,
  typename DestinationFragment, 
  typename SourceFragment,
  typename MmaIterations,
  typename MmaOperandShape,
  ComplexTransform Transform_,
  FloatRoundStyle Round_>
struct UnpackComplexConvertAndPackForMma <
  RealElement,
  DestinationFragment,
  SourceFragment,
  MmaIterations,
  MmaOperandShape,
  Transform_,
  Operand::kA,
  Round_> {
  
  //
  // Type definitions
  //
  static Operand const kOperand = Operand::kA;
  static ComplexTransform const kTransform = Transform_;
  static FloatRoundStyle const kRound = Round_;

  // Data type of elements in the destination fragment
  using MmaElement = typename DestinationFragment::Element;

  // Numeric convertor MmaElement <= RealElement
  using Converter = NumericConverter<MmaElement, RealElement, kRound>;

  // Operand layout parameters
  using SourceFragmentLayout = layout::ColumnMajor;
  static int const kLdm = MmaIterations::kRow * MmaOperandShape::kRow;

  /// Ctor
  CUTLASS_DEVICE
  UnpackComplexConvertAndPackForMma() {}

  CUTLASS_DEVICE
  void operator()(DestinationFragment *dest, SourceFragment const &source) {
    
    Converter convert_op;
    SourceFragmentLayout layout(kLdm);

    CUTLASS_PRAGMA_UNROLL
    for(int i=0; i<MmaIterations::kRow; i++) {
      int pos = 0;
      CUTLASS_PRAGMA_UNROLL
      for(int c=0; c<MmaOperandShape::kColumn; c++) {
        CUTLASS_PRAGMA_UNROLL
        for(int r=0; r<MmaOperandShape::kRow; r++) {
          // Logical position of element in source fragment
          int row = r + i * MmaOperandShape::kRow;
          int col = c;

          // Access complex<RealElement> and apply rounding on real and imag parts
          MmaElement a = convert_op(source[layout(MatrixCoord{row,col})].real());
          MmaElement b = convert_op(source[layout(MatrixCoord{row,col})].imag());

          // Unpack rounded complex<MmaElement> and pack into DestinationFragment for mma operation
          dest[i][pos] = a;
          dest[i+MmaIterations::kRow][pos++] = (kTransform == ComplexTransform::kConjugate ? -b : b);

        }
      }
    }
  }
};

// Partial specialization for OperandB and Congruous smem layout
template <
  typename RealElement,
  typename DestinationFragment, 
  typename SourceFragment,
  typename MmaIterations,
  typename MmaOperandShape,
  ComplexTransform Transform_,
  FloatRoundStyle Round_>
struct UnpackComplexConvertAndPackForMma <
  RealElement,
  DestinationFragment,
  SourceFragment,
  MmaIterations,
  MmaOperandShape,
  Transform_,
  Operand::kB,
  Round_> {
  
  //
  // Type definitions
  //
  static Operand const kOperand = Operand::kB;
  static ComplexTransform const kTransform = Transform_;
  static FloatRoundStyle const kRound = Round_;

  // Data type of elements in the destination fragment
  using MmaElement = typename DestinationFragment::Element;

  // Numeric convertor MmaElement <= RealElement
  using Converter = NumericConverter<MmaElement, RealElement, kRound>;

  // Operand layout parameters
  using SourceFragmentLayout = layout::RowMajor;
  static int const kLdm = MmaIterations::kColumn * MmaOperandShape::kColumn;

  /// Ctor
  CUTLASS_DEVICE
  UnpackComplexConvertAndPackForMma() {}

  CUTLASS_HOST_DEVICE
  void operator()(DestinationFragment *dest, SourceFragment const &source) {
    
    Converter convert_op;
    SourceFragmentLayout layout(kLdm);

    CUTLASS_PRAGMA_UNROLL
    for(int i=0; i<MmaIterations::kColumn; i++) {
      int pos = 0;
      CUTLASS_PRAGMA_UNROLL
      for(int c=0; c<MmaOperandShape::kColumn; c++) {
        CUTLASS_PRAGMA_UNROLL
        for(int r=0; r<MmaOperandShape::kRow; r++) {
          // Logical position of element in source fragment
          int row = r;
          int col = c + i * MmaOperandShape::kColumn;

          // Access complex<RealElement> apply rounding on real and imag parts
          MmaElement a = convert_op(source[layout(MatrixCoord{row,col})].real());
          MmaElement b = convert_op(source[layout(MatrixCoord{row,col})].imag());

          // Unpack rounded complex<MmaElement> and pack into DestinationFragment for mma operation
          dest[i][pos] = a;
          dest[i+MmaIterations::kColumn][pos++] = (kTransform == ComplexTransform::kConjugate ? -b : b);
        }
      }
    }
  }
};
} // namespace detail 

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
class MmaComplexTensorOp;

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
class MmaComplexTensorOp<
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

  /// Shape of the warp in units of thread (concept: MmaLanePolicyTensorOp)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Architecture tag from underlying instruction
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Indicates math operator 
  using MathOperator = arch::OpMultiplyAddComplex;

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
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, 
     ElementC, 
     LayoutC,
     typename ArchMmaOperator::Shape, 
     typename Policy::OpDelta>;

  /// Storage for C tile, the accumulator. Note, regardless of multiplicand type, this
  /// storage arrangement is to be considered 'planar complex' in the sense that all real-valued
  /// parts are stored consecutively followed by all imaginary parts. This matches the structure
  /// of Tensor Cores which are always real-valued matrix multiplies.
  using FragmentC = typename IteratorC::Fragment;

  static_assert(
    FragmentC::kElements == 2 * MmaIterations::kCount * ArchMmaOperator::FragmentC::kElements,
    "Unexpected planar complex fragment length.");

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
  MmaComplexTensorOp() {}

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

      // mma(accum.real(), a.real(), b.real(), accum.real());
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_A;
        MmaOperandB operand_B;

        operand_A[0] = A[m].real();
        operand_B[0] = B[n].real();

        // Real-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow);

          mma(*accum, operand_A, operand_B, *accum);
      }

      // mma(accum.imag(), a.real(), b.imag(), accum.imag()); 
      CUTLASS_PRAGMA_UNROLL
      for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_A;
        MmaOperandB operand_B;

        operand_A[0] = A[m].real();
        operand_B[0] = (kTransformB == ComplexTransform::kConjugate ? -B[n].imag() : B[n].imag());

        // Complex-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + MmaIterations::kCount;

        mma(*accum, operand_A, operand_B, *accum);
      }

      // mma(accum.real(), -a.imag(), b.imag(), accum.real())
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_A;
        MmaOperandB operand_B;

        // A imaginary part is intentionally negated
        operand_A[0] = (kTransformA == ComplexTransform::kConjugate ? A[m].imag() : -A[m].imag());
        operand_B[0] = (kTransformB == ComplexTransform::kConjugate ? -B[n].imag() : B[n].imag());

        // Real-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow);

        mma(*accum, operand_A, operand_B, *accum);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_A;
        MmaOperandB operand_B;

        operand_A[0] = (kTransformA == ComplexTransform::kConjugate ? -A[m].imag() : A[m].imag());
        operand_B[0] = B[n].real();

        // Complex-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + MmaIterations::kCount;

        mma(*accum, operand_A, operand_B, *accum);
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

/// Partial specialization for complex*complex+complex => complex:
//  Operands data type: complex<float>
//  Rounding: float -> tfloat32_t (round half_ulp_truncate nearest)
//  Math instruction: mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32
//  Output data type: complex<float>
// 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Complex transform on A operand
  ComplexTransform TransformA,
  /// Complex transform on B operand
  ComplexTransform TransformB
>
class MmaComplexTensorOp<
  Shape_, 
  complex<float>, 
  LayoutA_, 
  complex<float>,
  LayoutB_,
  complex<float>,
  LayoutC_,
  Policy_,
  TransformA,
  TransformB>  {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of members of complex multiplicand A
  using RealElementA = float;

  /// Data type of multiplicand A
  using ElementA = complex<RealElementA>;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of members of complex multiplicand B
  using RealElementB = float;

  /// Data type of multiplicand B
  using ElementB = complex<RealElementB>;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of members of complex accumulator matrix C
  using RealElementC = float;

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
  using MathOperator = typename arch::OpMultiplyAddComplex;
  
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
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements * 2>;

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
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements * 2>;

  static_assert(
    !(Shape::kM % ArchMmaOperator::Shape::kM) && 
    !(Shape::kN % ArchMmaOperator::Shape::kN),
    "Shape of warp-level Mma must be divisible by operator shape.");

  /// Number of complex products operations performed (one complex product needs four mma instructions)
  using MmaIterations = MatrixShape<
    Shape::kM / ArchMmaOperator::Shape::kM,
    Shape::kN / ArchMmaOperator::Shape::kN
  >;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, 
     ElementC, 
     LayoutC,
     typename ArchMmaOperator::Shape, 
     typename Policy::OpDelta>;

  /// Storage for C tile, the accumulator. Note, regardless of multiplicand type, this
  /// storage arrangement is to be considered 'planar complex' in the sense that all real-valued
  /// parts are stored consecutively followed by all imaginary parts. This matches the structure
  /// of Tensor Cores which are always real-valued matrix multiplies.
  using FragmentC = typename IteratorC::Fragment;

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
  MmaComplexTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    TransformedFragmentA const &A, 
    TransformedFragmentB const &B, 
    FragmentC const &C
  ) const {

    // Alias types for underlying real-valued matrix multiply operator
    using InstMmaOperandA = typename ArchMmaOperator::FragmentA;
    using InstMmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    static_assert(platform::is_same<cutlass::gemm::GemmShape<16, 8, 8>, typename ArchMmaOperator::Shape>::value, 
      "This implementation only supports mma.m16n8k8 math instructions.");

    static_assert(InstMmaOperandA::kElements == 4, 
      "This implementation only supports math instructions in which exactly four element is needed for the A operand."
      "We can geneneralize later.");

    static_assert(InstMmaOperandB::kElements == 2, 
      "This implementation only supports math instructions in which exactly two element is needed for the B operand."
      "We can geneneralize later.");

    // Instruction Operands A & B holding real part followed by imaginary part for mma operations
    InstMmaOperandA const *operand_A = reinterpret_cast<InstMmaOperandA const *>(&A);
    InstMmaOperandB const *operand_B = reinterpret_cast<InstMmaOperandB const *>(&B);

    //
    // Accumulate in place
    //
    D = C;

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaIterations::kRow; ++m) {

      // mma(accum.real(), a.real(), b.real(), accum.real());
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Real-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow);

          mma(*accum, operand_A[m], operand_B[n], *accum);
      }

      // mma(accum.imag(), a.real(), b.imag(), accum.imag()); 
      CUTLASS_PRAGMA_UNROLL
      for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {

        // Complex-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + MmaIterations::kCount;

        mma(*accum, operand_A[m], operand_B[n+MmaIterations::kColumn], *accum);
      }

      // mma(accum.real(), a.imag(), -b.imag(), accum.real())
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // negate OperandB to accumulate  -(a.imag()*b.imag())
        // negating OperandB emits less instructions than negating OperandA as OperandB has less elements
        negate<InstMmaOperandB> negate_op;

        // Real-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow);

        mma(*accum, operand_A[m+MmaIterations::kRow], negate_op(operand_B[n+MmaIterations::kColumn]), *accum);
      }

      // mma(accum.imag(), a.imag(), b.real(), accum.imag())
      CUTLASS_PRAGMA_UNROLL
      for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {

        // Complex-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + MmaIterations::kCount;

        mma(*accum, operand_A[m+MmaIterations::kRow], operand_B[n], *accum);
      }
    }
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {
    // Alias types for underlying real-valued matrix multiply operator
    using InstMmaOperandA = typename ArchMmaOperator::FragmentA;
    using InstMmaOperandB = typename ArchMmaOperator::FragmentB;

    //
    // Define conversions from source type to instruction operands' type
    //

    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    FloatRoundStyle const kRoundA = FloatRoundStyle::round_to_nearest;
    FloatRoundStyle const kRoundB = FloatRoundStyle::round_to_nearest;
    #else
    FloatRoundStyle const kRoundA = FloatRoundStyle::round_half_ulp_trunc_dntz; 
    FloatRoundStyle const kRoundB = FloatRoundStyle::round_half_ulp_trunc_dntz;
    #endif

    detail::UnpackComplexConvertAndPackForMma <
      RealElementA,
      InstMmaOperandA,
      FragmentA,
      MmaIterations,
      MatrixShape<2, 2>,
      kTransformA,
      Operand::kA,
      kRoundA> convert_A;

    detail::UnpackComplexConvertAndPackForMma <
      RealElementB,
      InstMmaOperandB,
      FragmentB,
      MmaIterations,
      MatrixShape<2, 1>,
      kTransformB,
      Operand::kB,
      kRoundB> convert_B;

    // Convert Fragment[A|B] holding complex<RealElement[A|B]> to InstMmaOperand[A|B] holding InstMmaOperand[A|B]::Element
    convert_A(reinterpret_cast<InstMmaOperandA *>(&dst_A), A); 
    convert_B(reinterpret_cast<InstMmaOperandB *>(&dst_B), B); 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for complex*complex+complex => complex:
//  Operands data type: complex<double>
//  Math instruction: mma.sync.aligned.m16n8k4.f64.f64.f64.f64
//  Output data type: complex<double>
// 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Complex transform on A operand
  ComplexTransform TransformA,
  /// Complex transform on B operand
  ComplexTransform TransformB
>
class MmaComplexTensorOp<
  Shape_, 
  complex<double>, 
  LayoutA_, 
  complex<double>,
  LayoutB_,
  complex<double>,
  LayoutC_,
  Policy_,
  TransformA,
  TransformB,
  true>  {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of members of complex multiplicand A
  using RealElementA = double;

  /// Data type of multiplicand A
  using ElementA = complex<RealElementA>;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of members of complex multiplicand B
  using RealElementB = double;

  /// Data type of multiplicand B
  using ElementB = complex<RealElementB>;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of members of complex accumulator matrix C
  using RealElementC = double;

  /// Data type of accumulator matrix C
  using ElementC = complex<RealElementC>;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicyTensorOp)
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
  using MathOperator = typename arch::OpMultiplyAddComplex;

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
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, 
     ElementC, 
     LayoutC,
     typename ArchMmaOperator::Shape, 
     typename Policy::OpDelta>;

  /// Storage for C tile, the accumulator. Note, regardless of multiplicand type, this
  /// storage arrangement is to be considered 'planar complex' in the sense that all real-valued
  /// parts are stored consecutively followed by all imaginary parts. This matches the structure
  /// of Tensor Cores which are always real-valued matrix multiplies.
  using FragmentC = typename IteratorC::Fragment;

  static_assert(
    FragmentC::kElements == 2 * MmaIterations::kCount * ArchMmaOperator::FragmentC::kElements,
    "Unexpected planar complex fragment length.");

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
  MmaComplexTensorOp() {}

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

      // mma(accum.real(), a.real(), b.real(), accum.real());
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_A;
        MmaOperandB operand_B;

        CUTLASS_PRAGMA_UNROLL
        for (int mk = 0; mk < MmaOperandA::kElements; ++mk)
          operand_A[mk] = A[m*MmaOperandA::kElements + mk].real();

        CUTLASS_PRAGMA_UNROLL
        for (int nk = 0; nk < MmaOperandB::kElements; ++nk)
          operand_B[nk] = B[n*MmaOperandB::kElements + nk].real();

        // Real-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow);

          mma(*accum, operand_A, operand_B, *accum);
      }

      // mma(accum.imag(), a.real(), b.imag(), accum.imag()); 
      CUTLASS_PRAGMA_UNROLL
      for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_A;
        MmaOperandB operand_B;

        CUTLASS_PRAGMA_UNROLL
        for (int mk = 0; mk < MmaOperandA::kElements; ++mk)
          operand_A[mk] = A[m*MmaOperandA::kElements + mk].real();

        CUTLASS_PRAGMA_UNROLL
        for (int nk = 0; nk < MmaOperandB::kElements; ++nk)
          operand_B[nk] = (kTransformB == ComplexTransform::kConjugate ? 
                          -B[n*MmaOperandB::kElements + nk].imag() : B[n*MmaOperandB::kElements + nk].imag());

        // Complex-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + MmaIterations::kCount;

        mma(*accum, operand_A, operand_B, *accum);
      }

      // mma(accum.real(), -a.imag(), b.imag(), accum.real())
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_A;
        MmaOperandB operand_B;

        // A imaginary part is intentionally negated
        CUTLASS_PRAGMA_UNROLL
        for (int mk = 0; mk < MmaOperandA::kElements; ++mk)
          operand_A[mk] = (kTransformA == ComplexTransform::kConjugate ?
                          A[m*MmaOperandA::kElements + mk].imag() : -A[m*MmaOperandA::kElements + mk].imag());

        CUTLASS_PRAGMA_UNROLL
        for (int nk = 0; nk < MmaOperandB::kElements; ++nk)
            operand_B[nk] = (kTransformB == ComplexTransform::kConjugate ?
                            -B[n*MmaOperandB::kElements + nk].imag() : B[n*MmaOperandB::kElements + nk].imag());

        // Real-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow);

        mma(*accum, operand_A, operand_B, *accum);
      }

      // mma(accum.imag(), a.imag(), b.real(), accum.imag())
      CUTLASS_PRAGMA_UNROLL
      for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {

        // Pack operands together. This may result in actual MOVs 
        MmaOperandA operand_A;
        MmaOperandB operand_B;

        CUTLASS_PRAGMA_UNROLL
        for (int mk = 0; mk < MmaOperandA::kElements; ++mk)
          operand_A[mk] = (kTransformA == ComplexTransform::kConjugate ?
                          -A[m*MmaOperandA::kElements + mk].imag() : A[m*MmaOperandA::kElements + mk].imag());

        CUTLASS_PRAGMA_UNROLL
        for (int nk = 0; nk < MmaOperandB::kElements; ++nk)
          operand_B[nk] = B[n*MmaOperandB::kElements + nk].real();

        // Complex-valued accumulator part
        MmaOperandC *accum = reinterpret_cast<MmaOperandC *>(&D) + 
          (m + n * MmaIterations::kRow) + MmaIterations::kCount;

        mma(*accum, operand_A, operand_B, *accum);
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
