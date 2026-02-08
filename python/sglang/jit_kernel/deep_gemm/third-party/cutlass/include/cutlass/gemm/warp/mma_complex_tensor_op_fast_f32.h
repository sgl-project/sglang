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
  /// Floating-point rounding style for big part
  FloatRoundStyle RoundBig_,
  /// Floating-point rounding style for small part
  FloatRoundStyle RoundSmall_>
struct UnpackComplexConvertAndPackForMmaFastF32;

// Partial specialization for OperandA and Congruous smem layout
template <
  typename RealElement,
  typename DestinationFragment, 
  typename SourceFragment,
  typename MmaIterations,
  typename MmaOperandShape,
  ComplexTransform Transform_,
  FloatRoundStyle RoundBig_,
  FloatRoundStyle RoundSmall_>
struct UnpackComplexConvertAndPackForMmaFastF32 <
  RealElement,
  DestinationFragment,
  SourceFragment,
  MmaIterations,
  MmaOperandShape,
  Transform_,
  Operand::kA,
  RoundBig_,
  RoundSmall_> {
  
  //
  // Type definitions
  //
  static Operand const kOperand = Operand::kA;
  static ComplexTransform const kTransform = Transform_;
  static FloatRoundStyle const kRoundBig = RoundBig_;
  static FloatRoundStyle const kRoundSmall = RoundSmall_;

  // Data type of elements in the destination fragment
  using MmaElement = typename DestinationFragment::Element;

  // Numeric convertor MmaElementBig, MmaElementSmall <= RealElement
  using Converter = NumericConverterFastF32<kRoundBig, kRoundSmall>;

  // Operand layout parameters
  using SourceFragmentLayout = layout::ColumnMajor;
  static int const kLdm = MmaIterations::kRow * MmaOperandShape::kRow;

  // BigSmall Fragment holding two TF32 elements (big, small) for every float
  using BigSmallFragment = Array<MmaElement, 2>;

  /// Index in fargments for the big and small part
  static int const kBigIndex = 0;
  static int const kSmallIndex = 1;

  /// Ctor
  CUTLASS_DEVICE
  UnpackComplexConvertAndPackForMmaFastF32() {}

  CUTLASS_DEVICE
  void operator()(DestinationFragment *dest, SourceFragment const &source) {
    
    Converter convert_op;
    SourceFragmentLayout layout(kLdm);

    DestinationFragment *dest_big_ = reinterpret_cast<DestinationFragment*>(dest);
    DestinationFragment *dest_small_ = reinterpret_cast<DestinationFragment*>(&dest[MmaIterations::kRow * 2]);

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
          BigSmallFragment a = convert_op(source[layout(MatrixCoord{row,col})].real());
          BigSmallFragment b = convert_op(source[layout(MatrixCoord{row,col})].imag());

          // Unpack rounded complex<MmaElement> and pack into DestinationFragment for mma operation
          dest_big_[i][pos] = a[kBigIndex];
          dest_big_[i+MmaIterations::kRow][pos] = (kTransform == ComplexTransform::kConjugate ? -b[kBigIndex] : b[kBigIndex]);

          // Unpack rounded complex<MmaElement> and pack into DestinationFragment for mma operation
          dest_small_[i][pos] = a[kSmallIndex];
          dest_small_[i+MmaIterations::kRow][pos] = (kTransform == ComplexTransform::kConjugate ? -b[kSmallIndex] : b[kSmallIndex]);

          // Next position
          pos++;
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
  FloatRoundStyle RoundBig_,
  FloatRoundStyle RoundSmall_>
struct UnpackComplexConvertAndPackForMmaFastF32 <
  RealElement,
  DestinationFragment,
  SourceFragment,
  MmaIterations,
  MmaOperandShape,
  Transform_,
  Operand::kB,
  RoundBig_,
  RoundSmall_> {
  
  //
  // Type definitions
  //
  static Operand const kOperand = Operand::kB;
  static ComplexTransform const kTransform = Transform_;
  static FloatRoundStyle const kRoundBig = RoundBig_;
  static FloatRoundStyle const kRoundSmall = RoundSmall_;

  // Data type of elements in the destination fragment
  using MmaElement = typename DestinationFragment::Element;

  // Numeric convertor MmaElementBig, MmaElementSmall <= RealElement
  using Converter = NumericConverterFastF32<kRoundBig, kRoundSmall>;

  // Operand layout parameters
  using SourceFragmentLayout = layout::RowMajor;
  static int const kLdm = MmaIterations::kColumn * MmaOperandShape::kColumn;

  // BigSmall Fragment holding two TF32 elements (big, small) for every float
  using BigSmallFragment = Array<MmaElement, 2>;

  /// Index in fargments for the big and small part
  static int const kBigIndex = 0;
  static int const kSmallIndex = 1;

  /// Ctor
  CUTLASS_DEVICE
  UnpackComplexConvertAndPackForMmaFastF32() {}

  CUTLASS_HOST_DEVICE
  void operator()(DestinationFragment *dest, SourceFragment const &source) {
    
    Converter convert_op;
    SourceFragmentLayout layout(kLdm);

    DestinationFragment *dest_big_ = reinterpret_cast<DestinationFragment*>(dest);
    DestinationFragment *dest_small_ = reinterpret_cast<DestinationFragment*>(&dest[MmaIterations::kColumn * 2]);

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
          BigSmallFragment a = convert_op(source[layout(MatrixCoord{row,col})].real());
          BigSmallFragment b = convert_op(source[layout(MatrixCoord{row,col})].imag());

          // Unpack rounded complex<MmaElement> and pack into DestinationFragment for mma operation
          dest_big_[i][pos] = a[kBigIndex];
          dest_big_[i+MmaIterations::kColumn][pos] = (kTransform == ComplexTransform::kConjugate ? -b[kBigIndex] : b[kBigIndex]);

          // Unpack rounded complex<MmaElement> and pack into DestinationFragment for mma operation
          dest_small_[i][pos] = a[kSmallIndex];
          dest_small_[i+MmaIterations::kColumn][pos] = (kTransform == ComplexTransform::kConjugate ? -b[kSmallIndex] : b[kSmallIndex]);

          // next position
          pos++;       
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
  /// Used for partial specialization
  typename Enable = bool
>
class MmaComplexTensorOpFastF32;

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
  ComplexTransform TransformB,
  /// Used for partial specialization
  typename Enable
>
class MmaComplexTensorOpFastF32<
  Shape_, 
  complex<float>, 
  LayoutA_, 
  complex<float>,
  LayoutB_,
  complex<float>,
  LayoutC_,
  Policy_,
  TransformA,
  TransformB,
  Enable>  {
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
  using MathOperator = arch::OpMultiplyAddComplexFastF32;
  
  /// Complex transform on A operand
  static ComplexTransform const kTransformA = TransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = TransformB;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;


  /// Tune F32 to TF32 big small conversion for complex<float> operation
  /// Different combination of big small conversin can cause different tradeoff
  /// between speed and accuracy.  Generally, use round_half_ulp_truncate can
  /// improve the performance but hur the accuracy.
  using ComplexFastF32 = FastF32 <
    FloatRoundStyle::round_toward_zero,        // kRoundBigA
    FloatRoundStyle::round_half_ulp_truncate,  // kRoundSmallA
    FloatRoundStyle::round_toward_zero,        // kRoundBigB
    FloatRoundStyle::round_half_ulp_truncate,  // kRoundSmallB
    TensorFloat32Op::k3xTF32                   // Number of TF32 operations 
  >;

  /// Index in fargments for the big and small part
  static int const kBigIndex = 0;
  static int const kSmallIndex = 1;

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
  // (4 times the original FragmentA::kElements)
  // (real_big), (imag_big), (real_small), (imag_small)
  using TransformedFragmentA = Array<typename ArchMmaOperator::ElementA, 
                                              FragmentA::kElements * 2 * 2>;

  // Fragment bisecting big and small sections
  // (real_big, imag_big), (real_small, imag_small)
  using AccessTypeFragmentA = Array<typename ArchMmaOperator::ElementA, 
                                                    FragmentA::kElements * 2>;

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
  // (4 times the original FragmentB::kElements)
  // (real_big), (imag_big), (real_small), (imag_small)
  using TransformedFragmentB = Array<typename ArchMmaOperator::ElementB, 
                                              FragmentB::kElements * 2 * 2>;

  // Fragment bisecting big and small sections
  // (real_big, imag_big), (real_small, imag_small)
  using AccessTypeFragmentB = Array<typename ArchMmaOperator::ElementB, 
                                                    FragmentB::kElements * 2>;

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

  //
  // Alias types for underlying real-valued matrix multiply operator
  //
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
  MmaComplexTensorOpFastF32() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    TransformedFragmentA const &A, 
    TransformedFragmentB const &B, 
    FragmentC const &C
  ) const {

    AccessTypeFragmentA const *complex_A = reinterpret_cast<AccessTypeFragmentA const*>(&A);
    AccessTypeFragmentB const *complex_B = reinterpret_cast<AccessTypeFragmentB const*>(&B);

    //
    // Accumulate in place
    //
    D = C;


    complex_mma_operator(D, complex_A[kSmallIndex], complex_B[kBigIndex], D);

    complex_mma_operator(D, complex_A[kBigIndex], complex_B[kSmallIndex], D);

    complex_mma_operator(D, complex_A[kBigIndex], complex_B[kBigIndex], D);

    if (ComplexFastF32::kPrecision == TensorFloat32Op::k4xTF32)
      complex_mma_operator(D, complex_A[kSmallIndex], complex_B[kSmallIndex], D);
  }

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void complex_mma_operator(
    FragmentC &D, 
    AccessTypeFragmentA const &complex_A, 
    AccessTypeFragmentB const &complex_B, 
    FragmentC const &C
  ) const {

    // Instruction Operands A & B holding real part followed by imaginary part for mma operations
    InstMmaOperandA const *operand_A = reinterpret_cast<InstMmaOperandA const *>(&complex_A);
    InstMmaOperandB const *operand_B = reinterpret_cast<InstMmaOperandB const *>(&complex_B);


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

    detail::UnpackComplexConvertAndPackForMmaFastF32 <
      RealElementA,
      InstMmaOperandA,
      FragmentA,
      MmaIterations,
      MatrixShape<2, 2>,
      kTransformA,
      Operand::kA,
      ComplexFastF32::kRoundBigA,
      ComplexFastF32::kRoundSmallA> convert_A;

    detail::UnpackComplexConvertAndPackForMmaFastF32 <
      RealElementB,
      InstMmaOperandB,
      FragmentB,
      MmaIterations,
      MatrixShape<2, 1>,
      kTransformB,
      Operand::kB,
      ComplexFastF32::kRoundBigB,
      ComplexFastF32::kRoundSmallB> convert_B;

    // Convert Fragment[A|B] holding complex<RealElement[A|B]> to InstMmaOperand[A|B] holding InstMmaOperand[A|B]::Element
    convert_A(reinterpret_cast<InstMmaOperandA *>(&dst_A), A); 
    convert_B(reinterpret_cast<InstMmaOperandB *>(&dst_B), B); 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
