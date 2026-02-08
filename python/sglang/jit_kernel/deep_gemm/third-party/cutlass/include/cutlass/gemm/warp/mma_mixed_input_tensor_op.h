/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h" 
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

////////////////////////////////////////////////////////////////////////////////
// Shuffle registers for layout conversion
////////////////////////////////////////////////////////////////////////////////
template <
  /// Element type for the operand in registers for the mma.sync
  typename ElementMma_, 
  /// Element type for the operand in shared memory for ldmatrix
  typename ElementLoad_,
  /// Number of mma.sync operations performed along rows or columns         
  int NumMmaInstructions,
  /// Number of elements in warp fragment
  int NumElementsInWarpFragment,
  /// Number of elements in mma fragment
  int NumElementsInMmaFragment,
  /// Identifies A or B multiplicand
  Operand Operand_,
  ///
  typename Enable = void >
struct FragmentShuffler {
  public:
  using ElementMma = ElementMma_;
  using ElementLoad = ElementLoad_;

  static int const kNumMmaInstructions = NumMmaInstructions;
  static int const kNumElementsInWarpFragment = NumElementsInWarpFragment;
  static int const kNumElementsInMmaFragment = NumElementsInMmaFragment;
  static Operand const kOperand = Operand_;

  using WarpFragment = Array<ElementLoad, kNumElementsInWarpFragment>;
  using MmaFragment = Array<ElementLoad, kNumElementsInMmaFragment>;

  CUTLASS_DEVICE
  WarpFragment operator()(WarpFragment const &src) {
    return src;
  }
};
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for `mma.sync` on 16b (F16/BF16) and `ldmatrix` on 8b (S8/U8)
/// or for `mma.sync` on 8b (S8/U8) and `ldmatrix` on 4b (S4/U4)
/// for operand A multiplicand going through upcasting. 
template <
  /// Element type for the operand in registers for the mma.sync
  typename ElementMma_, 
  /// Element type for the operand in shared memory for ldmatrix
  typename ElementLoad_,
  /// Number of mma.sync operations performed along rows or columns         
  int NumMmaInstructions,
  /// Number of elements in warp fragment
  int NumElementsInWarpFragment,
  /// Number of elements in mma fragment
  int NumElementsInMmaFragment
> 
struct FragmentShuffler <ElementMma_, ElementLoad_,
                         NumMmaInstructions, 
                         NumElementsInWarpFragment, 
                         NumElementsInMmaFragment,
                         Operand::kA,
                         typename platform::enable_if<(sizeof_bits<ElementMma_>::value /
                                                 sizeof_bits<ElementLoad_>::value == 2)>::type> {
public:
  using ElementMma = ElementMma_;
  using ElementLoad = ElementLoad_;

  static int const kNumMmaInstructions = NumMmaInstructions;
  static int const kNumElementsInWarpFragment = NumElementsInWarpFragment;
  static int const kNumElementsInMmaFragment = NumElementsInMmaFragment;
  static Operand const kOperand = Operand::kA;

  using WarpFragment = Array<ElementLoad, kNumElementsInWarpFragment>;
  using MmaFragment = Array<ElementLoad, kNumElementsInMmaFragment>;

  static uint32_t const kSelectBytesEvenThread = 0x5410;
  static uint32_t const kSelectBytesOddThread = 0x7632;

private:
  int delta_up_;
  int delta_down_;
  int odd_even_lane_id_;
  uint32_t byte_selector_;

public:
  CUTLASS_DEVICE
  FragmentShuffler() {
    int lane_id = cutlass::arch::LaneId();
    delta_up_ = (lane_id & 1) + ((lane_id & 2) >> 1);
    delta_down_ = 2 - delta_up_;
    odd_even_lane_id_ = static_cast<int>(lane_id & 1);
    byte_selector_ = odd_even_lane_id_ * kSelectBytesOddThread +
                    (1 - odd_even_lane_id_) * kSelectBytesEvenThread;
  }

  CUTLASS_DEVICE
  WarpFragment operator()(WarpFragment const &src) {

    WarpFragment result;
    MmaFragment const* mma_frag_src_ptr = reinterpret_cast<MmaFragment const*>(&src);
    MmaFragment* mma_frag_dst_ptr = reinterpret_cast<MmaFragment*>(&result);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kNumMmaInstructions; n++) {

        uint32_t const* src_ptr = reinterpret_cast<uint32_t const *>(&mma_frag_src_ptr[n]);
        uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(&mma_frag_dst_ptr[n]);

        // Shuffle data within the warp, pull from other threads within the warp
        uint32_t tmp0 = __shfl_up_sync(0xFFFFFFFF, src_ptr[0], delta_up_);
        uint32_t tmp1 = __shfl_down_sync(0xFFFFFFFF, src_ptr[0], delta_down_);
        uint32_t tmp2 = __shfl_up_sync(0xFFFFFFFF, src_ptr[1], delta_up_);
        uint32_t tmp3 = __shfl_down_sync(0xFFFFFFFF, src_ptr[1], delta_down_);

        // Reorder the data within the 32-bit word (4x8b) required for mma.sync
        dst_ptr[0] = __byte_perm(tmp0, tmp2, byte_selector_);
        dst_ptr[1] = __byte_perm(tmp1, tmp3, byte_selector_);
    }

    return result;
  }

};
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for `mma.sync` on 16b (F16/BF16) and `ldmatrix` on 8b (S8/U8)
/// or for `mma.sync` on 8b (S8/U8) and `ldmatrix` on 4b (S4/U4)
/// for operand B multiplicand going through upcasting. 
template <
  /// Element type for the operand in registers for the mma.sync
  typename ElementMma_, 
  /// Element type for the operand in shared memory for ldmatrix
  typename ElementLoad_,
  /// Number of mma.sync operations performed along rows or columns         
  int NumMmaInstructions,
  /// Number of elements in warp fragment
  int NumElementsInWarpFragment,
  /// Number of elements in mma fragment
  int NumElementsInMmaFragment
> 
struct FragmentShuffler <ElementMma_, ElementLoad_,
                         NumMmaInstructions, 
                         NumElementsInWarpFragment, 
                         NumElementsInMmaFragment,
                         Operand::kB,
                         typename platform::enable_if<(sizeof_bits<ElementMma_>::value /
                                                 sizeof_bits<ElementLoad_>::value == 2)>::type> {
public:
  using ElementMma = ElementMma_;
  using ElementLoad = ElementLoad_;

  static int const kNumMmaInstructions = NumMmaInstructions;
  static int const kNumElementsInWarpFragment = NumElementsInWarpFragment;
  static int const kNumElementsInMmaFragment = NumElementsInMmaFragment;
  static Operand const kOperand = Operand::kB;

  using WarpFragment = Array<ElementLoad, kNumElementsInWarpFragment>;
  using MmaFragment = Array<ElementLoad, kNumElementsInMmaFragment>;

  static uint32_t const kSelectBytesEvenThread = 0x5410;
  static uint32_t const kSelectBytesOddThread = 0x7632;

private:
  int delta_up_;
  int delta_down_;
  int odd_even_lane_id_;
  uint32_t byte_selector_;

public:
  CUTLASS_DEVICE
  FragmentShuffler() {
    int lane_id = cutlass::arch::LaneId();
    delta_up_ = (lane_id & 1) + ((lane_id & 2) >> 1);
    delta_down_ = 2 - delta_up_;
    odd_even_lane_id_ = static_cast<int>(lane_id & 1);
    byte_selector_ = odd_even_lane_id_ * kSelectBytesOddThread +
                    (1 - odd_even_lane_id_) * kSelectBytesEvenThread;
  }

  CUTLASS_DEVICE
  WarpFragment operator()(WarpFragment const &src) {

    WarpFragment result;

    MmaFragment const* mma_frag_src_ptr = reinterpret_cast<MmaFragment const *>(&src);
    MmaFragment* mma_frag_dst_ptr = reinterpret_cast<MmaFragment *>(&result);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kNumMmaInstructions; n++) {

        uint32_t const* src_ptr = reinterpret_cast<uint32_t const*>(&mma_frag_src_ptr[n]);
        uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(&mma_frag_dst_ptr[n]);

        // Shuffle data within the warp, pull from other threads within the warp
        uint32_t tmp0 = __shfl_up_sync(0xFFFFFFFF, src_ptr[0], delta_up_);
        uint32_t tmp1 = __shfl_down_sync(0xFFFFFFFF, src_ptr[0], delta_down_);

        // Reorder the data within the 32-bit word (4x8b) required for mma.sync
        dst_ptr[0] = __byte_perm(tmp0, tmp1, byte_selector_);
    }

    return result;
  }

};

////////////////////////////////////////////////////////////////////////////////
// Data type conversion
////////////////////////////////////////////////////////////////////////////////
template <
  /// Destination type
  typename ElementDst_, 
  /// Source type
  typename ElementSrc_,
  /// Number of elements
  int N,
  ///
  typename Enable = void> 
struct FragmentConverter {

  using ElementDst = ElementDst_;
  using ElementSrc = ElementSrc_;

  // Operand fragment registers in destination and source types
  using DestinationFragment = Array<ElementDst, N>;
  using SourceFragment = Array<ElementSrc, N>;

  FastNumericArrayConverter<ElementDst, ElementSrc, N> convert;

  CUTLASS_DEVICE
  DestinationFragment operator()(SourceFragment const &src) const {
    return convert(src);
  }
};
////////////////////////////////////////////////////////////////////////////////

// Partial specialization for when Destination type is the *same* as 
// Source type
template <
  /// Data type
  typename Element,
  /// Number of elements
  int N,
  /// 
  typename Enable>
struct FragmentConverter<Element, Element, N, Enable> {

  using DestinationFragment = Array<Element, N>;
  using SourceFragment = Array<Element, N>;

  CUTLASS_DEVICE
  DestinationFragment operator()(SourceFragment const &src) const {
    return src;
  }
};

} // namespace detail

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
class MmaMixedInputTensorOp {
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

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Underlying arch::Mma instruction datatype for A operand
  using ElementAMma = typename ArchMmaOperator::ElementA;

  /// Underlying arch::Mma instruction datatype for B operand
  using ElementBMma = typename ArchMmaOperator::ElementB;

  /// Underlying arch::Mma instruction datatype for C operand
  using MmaElementC = typename ArchMmaOperator::ElementC;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;

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

  /// 
  // static int const kLoadShapeK = InstructionShape::kK * 
  //  (sizeof_bits<ElementAMma>::value / sizeof_bits<ElementB>::value);

public:

  /// Iterates over the A operand in Shared Memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
     MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
     MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
     Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for A tile in registers (loaded from Shared Memory)
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile in registers (for use in Mma instruction)
  using TransformedFragmentA =
      Array<ElementAMma, FragmentA::kElements>;

  /// Underlying arch::Mma instruction operand fragment for matrix A
  using MmaOperandA = typename ArchMmaOperator::FragmentA;

  /// Iterates over the B operand in Shared Memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
      MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for B tile in registers (loaded from Shared Memory)
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile in registers (for use in Mma instruction)
  using TransformedFragmentB =
      Array<ElementBMma, FragmentB::kElements>;

  /// Underlying arch::Mma instruction operand fragment for matrix B
  using MmaOperandB = typename ArchMmaOperator::FragmentB;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
     typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

  /// Underlying arch::Mma instruction operand fragment for matrix C
  using MmaOperandC = typename ArchMmaOperator::FragmentC;

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
  MmaMixedInputTensorOp() {}

    /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    TransformedFragmentA const &A, 
    TransformedFragmentB const &B, 
    FragmentC const &C
  ) const {

    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaIterations::kRow; ++m) {

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

        if (AccumulatorsInRowMajor) {  // matrix B is reordered
          mma(
            ptr_D[n_serpentine + m * MmaIterations::kColumn],
            ptr_A[m],
            ptr_B[n_serpentine],
            ptr_D[n_serpentine + m * MmaIterations::kColumn]);
        } else {
          mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[m + n_serpentine * MmaIterations::kRow]);
        }
      }
    }
  }

  /// Transform the operand warp fragment register to the required data types and layout 
  /// for the `cultass::arch::Mma`
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {

    // Shuffle data within warp to obtain the mma.sync operand layout
    detail::FragmentShuffler<ElementBMma, ElementB, MmaIterations::kColumn, 
             FragmentB::kElements, MmaOperandB::kElements, Operand::kB> shuffler_B;
    FragmentB tmp_B; 
    tmp_B = shuffler_B(B);

    // Convert the B operand to the Mma Instruction operand type
    detail::FragmentConverter<ElementBMma, ElementB, FragmentB::kElements> convert_B;
    dst_B = convert_B(tmp_B);

    FragmentA tmp_A;

    Array<ElementA, FragmentA::kElements / 2> *
        ptr_tmp_A = reinterpret_cast<Array<ElementA,
                                             FragmentA::kElements / 2> *>(&tmp_A);
    Array<ElementAMma, FragmentA::kElements / 2> *
        ptr_dst_A = reinterpret_cast<Array<ElementAMma,
                                             FragmentA::kElements / 2> *>(&dst_A);

    // Shuffle data within warp to obtain the mma.sync operand layout
    detail::FragmentShuffler<ElementAMma, ElementA, MmaIterations::kRow,
             FragmentA::kElements, MmaOperandA::kElements, Operand::kA> shuffler_A;

    // Convert the A operand to the Mma Instruction operand type
    detail::FragmentConverter<ElementAMma, ElementA, FragmentA::kElements / 2> convert_A;

    tmp_A = shuffler_A(A);
    ptr_dst_A[0] = convert_A(ptr_tmp_A[0]);

    ptr_dst_A[1] = convert_A(ptr_tmp_A[1]);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
