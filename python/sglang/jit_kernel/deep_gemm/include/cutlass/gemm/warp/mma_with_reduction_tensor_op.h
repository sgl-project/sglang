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

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
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
  /// Reduce operand A or B along K dimension
  bool ReduceKForA_,
  /// Number of partitions along K dimension
  int PartitionsK_ = 1,
  /// Store the accumulators in row major or column major.  Row major is used
  /// when output layout is interleaved.
  bool AccumulatorsInRowMajor = false,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaWithReductionTensorOp {
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

  static bool const kReduceKForA = ReduceKForA_;

  static_assert(platform::is_same<ElementA, cutlass::half_t>::value ||
                platform::is_same<ElementA, cutlass::bfloat16_t>::value,
                "ElementA needs to be fp16 or bf16.");

  static_assert(platform::is_same<ElementB, cutlass::half_t>::value ||
                platform::is_same<ElementB, cutlass::bfloat16_t>::value,
                "ElementB needs to be fp16 or bf16.");

  static_assert(platform::is_same<InstructionShape,
                                  cutlass::gemm::GemmShape<16, 8, 16>>::value,
                "Only supports 16x8x16 tensor core instruction.");

  static_assert(!AccumulatorsInRowMajor,
                "Only calls tensor core instructions in column major.");

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
     MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
     MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
     Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
      MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

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

  using FragmentReduction = Array<ElementC, kReduceKForA ? (Shape::kM / 8) : (Shape::kN / 8)>;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaWithReductionTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D,
    TransformedFragmentA const &A,
    TransformedFragmentB const &B,
    FragmentC const &C,
    FragmentReduction &gemm_k_reduction
  ) const {

    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;

    [[maybe_unused]] MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    [[maybe_unused]] MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    [[maybe_unused]] MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
      assert(0);
    #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      // Serpentine visitation order maximizing reuse of Ra
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {

          int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

          mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[m + n_serpentine * MmaIterations::kRow]);

          if (!kReduceKForA && m == 0) {
            #if 0
            gemm_k_reduction[n_serpentine] += float(B[n_serpentine * 4]);
            gemm_k_reduction[n_serpentine] += float(B[n_serpentine * 4 + 1]);
            gemm_k_reduction[n_serpentine] += float(B[n_serpentine * 4 + 2]);
            gemm_k_reduction[n_serpentine] += float(B[n_serpentine * 4 + 3]);
            #else
            uint32_t const *tmp = reinterpret_cast<uint32_t const *>(&B);

            if (platform::is_same<ElementB, cutlass::half_t>::value) {
              asm volatile(
                "{\n\t"
                " .reg .f16 low, high;\n\t"
                " .reg .f32 tmp;\n\t"
                " mov.b32 {low, high}, %1;\n\t"
                " cvt.f32.f16 tmp, low;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " cvt.f32.f16 tmp, high;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " mov.b32 {low, high}, %2;\n\t"
                " cvt.f32.f16 tmp, low;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " cvt.f32.f16 tmp, high;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                "}\n\t"
                : "+f"(gemm_k_reduction[n_serpentine])
                : "r"(tmp[n_serpentine * 2]), "r"(tmp[n_serpentine * 2 + 1]));
            } else if (platform::is_same<ElementB, cutlass::bfloat16_t>::value) {
              asm volatile(
                "{\n\t"
                " .reg .f32 tmp;\n\t"
                " shl.b32 tmp, %1, 16;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " and.b32 tmp, %1, 0xffff0000;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " shl.b32 tmp, %2, 16;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " and.b32 tmp, %2, 0xffff0000;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                "}\n\t"
                : "+f"(gemm_k_reduction[n_serpentine])
              : "r"(tmp[n_serpentine * 2]), "r"(tmp[n_serpentine * 2 + 1]));
            } else {
                assert(0);
            }
            #endif
          }

          if (kReduceKForA && (n == 0)) {
            #if 0
            gemm_k_reduction[m * 2] += float(A[m * 8]);
            gemm_k_reduction[m * 2] += float(A[m * 8 + 1]);
            gemm_k_reduction[m * 2] += float(A[m * 8 + 4]);
            gemm_k_reduction[m * 2] += float(A[m * 8 + 5]);

            gemm_k_reduction[m * 2 + 1] += float(A[m * 8 + 2]);
            gemm_k_reduction[m * 2 + 1] += float(A[m * 8 + 3]);
            gemm_k_reduction[m * 2 + 1] += float(A[m * 8 + 6]);
            gemm_k_reduction[m * 2 + 1] += float(A[m * 8 + 7]);
            #else
            uint32_t const *tmp = reinterpret_cast<uint32_t const *>(&A);

            if (platform::is_same<ElementA, cutlass::half_t>::value) {
              asm volatile(
                "{\n\t"
                " .reg .f16 low, high;\n\t"
                " .reg .f32 tmp;\n\t"
                " mov.b32 {low, high}, %2;\n\t"
                " cvt.f32.f16 tmp, low;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " cvt.f32.f16 tmp, high;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " mov.b32 {low, high}, %3;\n\t"
                " cvt.f32.f16 tmp, low;\n\t"
                " add.f32 %1, tmp, %1;\n\t"
                " cvt.f32.f16 tmp, high;\n\t"
                " add.f32 %1, tmp, %1;\n\t"
                " mov.b32 {low, high}, %4;\n\t"
                " cvt.f32.f16 tmp, low;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " cvt.f32.f16 tmp, high;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " mov.b32 {low, high}, %5;\n\t"
                " cvt.f32.f16 tmp, low;\n\t"
                " add.f32 %1, tmp, %1;\n\t"
                " cvt.f32.f16 tmp, high;\n\t"
                " add.f32 %1, tmp, %1;\n\t"
                "}\n\t"
                : "+f"(gemm_k_reduction[m * 2]), "+f"(gemm_k_reduction[m * 2 + 1])
                : "r"(tmp[m * 4]), "r"(tmp[m * 4 + 1]),"r"(tmp[m * 4 + 2]), "r"(tmp[m * 4 + 3]));

            } else if (platform::is_same<ElementA, cutlass::bfloat16_t>::value) {

              asm volatile(
                "{\n\t"
                " .reg .f32 tmp;\n\t"
                " shl.b32 tmp, %2, 16;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " and.b32 tmp, %2, 0xffff0000;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " shl.b32 tmp, %3, 16;\n\t"
                " add.f32 %1, tmp, %1;\n\t"
                " and.b32 tmp, %3, 0xffff0000;\n\t"
                " add.f32 %1, tmp, %1;\n\t"
                " shl.b32 tmp, %4, 16;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " and.b32 tmp, %4, 0xffff0000;\n\t"
                " add.f32 %0, tmp, %0;\n\t"
                " shl.b32 tmp, %5, 16;\n\t"
                " add.f32 %1, tmp, %1;\n\t"
                " and.b32 tmp, %5, 0xffff0000;\n\t"
                " add.f32 %1, tmp, %1;\n\t"
                "}\n\t"
                : "+f"(gemm_k_reduction[m * 2]), "+f"(gemm_k_reduction[m * 2 + 1])
                : "r"(tmp[m * 4]), "r"(tmp[m * 4 + 1]),"r"(tmp[m * 4 + 2]), "r"(tmp[m * 4 + 3]));

            } else {
              assert(0);
            }
            #endif
          }
        }
      }
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
    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;
    FloatRoundStyle const kRoundB =
        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                              ElementB>::kRound;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements, kRoundA>
          convert_A;
      NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements / 2, kRoundB>
          convert_B;
      Array<ElementB, FragmentB::kElements / 2> const *ptr_B =
          reinterpret_cast<Array<ElementB, FragmentB::kElements / 2> const *>(&B);
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements / 2> *
          ptr_dst_B = reinterpret_cast<Array<typename ArchMmaOperator::ElementB,
                                             FragmentB::kElements / 2> *>(&dst_B);

      dst_A = convert_A(A);

      ptr_dst_B[0] = convert_B(ptr_B[0]);
      ptr_dst_B[1] = convert_B(ptr_B[1]);

    #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements / 2, kRoundA>
          convert_A;
      NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements, kRoundB>
          convert_B;
      Array<ElementA, FragmentA::kElements / 2> const *ptr_A =
          reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&A);
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *
          ptr_dst_A = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                             FragmentA::kElements / 2> *>(&dst_A);

      dst_B = convert_B(B);

      ptr_dst_A[0] = convert_A(ptr_A[0]);
      ptr_dst_A[1] = convert_A(ptr_A[1]);
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
