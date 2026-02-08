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
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class TensorFloat32Op {
  k3xTF32, 
  k4xTF32 
}; 

template <
  /// Floating-point rounding style
  FloatRoundStyle RoundBigA_,
  /// Floating-point rounding style
  FloatRoundStyle RoundSmallA_,
  /// Floating-point rounding style
  FloatRoundStyle RoundBigB_ = RoundBigA_,
  /// Floating-point rounding style
  FloatRoundStyle RoundSmallB_ = RoundSmallA_,
  /// Precision for TensorFloat32Op 
  // (k3xTF32: BigxBig, BigxSmall, SmallxBig)
  // (k4xTF32: BigxBig, BigxSmall, SmallxBig, SmallxSmall)
  TensorFloat32Op Precision_ = TensorFloat32Op::k3xTF32
  >
struct FastF32 {

  static FloatRoundStyle const kRoundBigA = RoundBigA_;
  static FloatRoundStyle const kRoundSmallA = RoundSmallA_;
  static FloatRoundStyle const kRoundBigB = RoundBigB_;
  static FloatRoundStyle const kRoundSmallB = RoundSmallB_;
  static TensorFloat32Op const kPrecision = Precision_;
};


namespace detail {

  template<
    int N,
    FloatRoundStyle RoundBig = FloatRoundStyle::round_toward_zero,
    FloatRoundStyle RoundSmall = FloatRoundStyle::round_half_ulp_truncate
  >
  struct ConvertAndPackAccurateF32 {
  
    /// Rounding styles for big and small part
    static FloatRoundStyle const kRoundBig = RoundBig;
    static FloatRoundStyle const kRoundSmall = RoundSmall;

    /// Converter type
    using Converter = NumericConverterFastF32<kRoundBig, kRoundSmall>;

    /// Source fragement
    using SourceFragment = Array<float, N>;

    /// Destination fragment
    using DestinationFragment = Array<tfloat32_t, N>;

    /// Converter Fragment holding two tfloat32_t elements for every float
    using ConverterFragment = Array<tfloat32_t, 2>;

    /// Index in fargments for the big and small part
    static int const kBigIndex = 0;
    static int const kSmallIndex = 1;

    CUTLASS_HOST_DEVICE
    void operator()(SourceFragment const &source,
                    DestinationFragment &dst_big,
                    DestinationFragment &dst_small) {
      
      Converter convert_;
      ConverterFragment result_;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        // convert source to result fragment
        result_ = convert_(source[i]);

        // store converted result fragments to destination fragment
        dst_big[i] = result_[kBigIndex];
        dst_small[i] = result_[kSmallIndex];
      }
    }
  };
} // namespace detail

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
  /// Number of partitions along K dimension
  int PartitionsK_ = 1,
  /// Store the accumulators in row major or column major.  Row major is used
  /// when output layout is interleaved.
  bool AccumulatorsInRowMajor = false,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaTensorOpFastF32;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float*float+float => float using TF32 TensorOps
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
  /// Number of partitions along K dimension
  int PartitionsK_,
  /// Store the accumulators in row major or column major.  Row major is used
  /// when output layout is interleaved.
  bool AccumulatorsInRowMajor,
  /// Used for partial specialization
  typename Enable
>
class MmaTensorOpFastF32<
  Shape_,
  float, LayoutA_,
  float, LayoutB_,
  float, LayoutC_,
  Policy_, PartitionsK_,
  AccumulatorsInRowMajor, Enable> {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = float;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = float;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = float;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator 
  using MathOperator = arch::OpMultiplyAddFastF32;

  /// Architecture tag from underlying instruction
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// Tune F32 to TF32 big small conversion for float operation
  /// Different combination of big small conversin can cause different tradeoff
  /// between speed and accuracy.  Generally, use round_half_ulp_truncate can
  /// improve the performance but hur the accuracy.
  using MmaFastF32 = FastF32 <
    FloatRoundStyle::round_toward_zero,        // kRoundBigA
    FloatRoundStyle::round_half_ulp_truncate,  // kRoundSmallA
    FloatRoundStyle::round_toward_zero,        // kRoundBigB
    FloatRoundStyle::round_half_ulp_truncate,  // kRoundSmallB
    TensorFloat32Op::k3xTF32                   // Number of TF32 operations 
  >;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kM, Shape::kK>, 
      Operand::kA, 
      ElementA, 
      LayoutA,
      MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
      Policy::OpDelta::kRow, 
      kThreadCount, 
      kPartitionsK
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements * 2>;

  /// Fragment bisecting big and small sections
  using AccessTypeFragmentA = 
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, 
      Operand::kB, 
      ElementB, 
      LayoutB,
      MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, 
      kThreadCount, 
      kPartitionsK
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements * 2>;

  /// Fragment bisecting big and small sections
  using AccessTypeFragmentB = 
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

  /// Index in fargments for the big and small part
  static int const kBigIndex = 0;
  static int const kSmallIndex = 1;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
     typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    (Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
    (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN
  >;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaTensorOpFastF32() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    TransformedFragmentA const &A, 
    TransformedFragmentB const &B, 
    FragmentC const &C
  ) const {

    AccessTypeFragmentA const *ptr_A = reinterpret_cast<AccessTypeFragmentA const*>(&A);
    AccessTypeFragmentB const *ptr_B = reinterpret_cast<AccessTypeFragmentB const*>(&B);

    //
    // Accumulate in place
    //
    D = C;
    
    mma_operator(D, ptr_A[kSmallIndex], ptr_B[kBigIndex], D);

    mma_operator(D, ptr_A[kBigIndex], ptr_B[kSmallIndex], D);

    mma_operator(D, ptr_A[kBigIndex], ptr_B[kBigIndex], D);

    if (MmaFastF32::kPrecision == TensorFloat32Op::k4xTF32)
      mma_operator(D, ptr_A[kSmallIndex], ptr_B[kSmallIndex], D);
  }

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void mma_operator(
    FragmentC &D, 
    AccessTypeFragmentA const &A, 
    AccessTypeFragmentB const &B, 
    FragmentC const &C
  ) const {

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

      using MmaOperandA = typename ArchMmaOperator::FragmentA;
      using MmaOperandB = typename ArchMmaOperator::FragmentB;
      using MmaOperandC = typename ArchMmaOperator::FragmentC;

      MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
      MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
      MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

      // Serpentine visitation order maximizing reuse of Ra
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {

          // This allows to reuse of Rb when at serpentine turns
          int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D[n_serpentine + m * MmaIterations::kColumn],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[n_serpentine + m * MmaIterations::kColumn]);
          } else {
            mma(
              ptr_D[m + n_serpentine * MmaIterations::kRow],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[m + n_serpentine * MmaIterations::kRow]);
          }
        } // end n loop
      } // end m loop
    #else
      assert(0);
    #endif
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {

    //
    // Define conversions from source type to instruction type
    //
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      
      detail::ConvertAndPackAccurateF32<
        FragmentA::kElements / 2,
        MmaFastF32::kRoundBigA,
        MmaFastF32::kRoundSmallA> convert_A;
      
      detail::ConvertAndPackAccurateF32<
        FragmentB::kElements,
        MmaFastF32::kRoundBigB,
        MmaFastF32::kRoundSmallB> convert_B;
      
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements> *ptr_dst_B = 
        reinterpret_cast<Array<typename ArchMmaOperator::ElementB, FragmentB::kElements> *>(&dst_B);
      
      convert_B(B, ptr_dst_B[0], ptr_dst_B[1]);

      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *ptr_dst_A =
        reinterpret_cast<Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *>(&dst_A);
      
      Array<ElementA, FragmentA::kElements / 2> const *ptr_A = 
        reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&A);
      
      convert_A(ptr_A[0], ptr_dst_A[0], ptr_dst_A[2]);
      
      convert_A(ptr_A[1], ptr_dst_A[1], ptr_dst_A[3]);
    #else
      assert(0);
    #endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
