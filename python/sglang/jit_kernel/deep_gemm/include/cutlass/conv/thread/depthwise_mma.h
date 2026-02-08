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
    \brief Templates exposing architecture support for depthwise convolution
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/thread/mma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// MMA operation
template <
  /// Size of the matrix product (concept: GemmShape)
  typename Shape_,
  /// Number of threads participating
  int kThreads_,
  /// Data type of A elements
  typename ElementA,
  /// Data type of B elements
  typename ElementB,
  /// Element type of C matrix
  typename ElementC,
  /// Inner product operator
  typename Operator
>
struct ElementwiseInnerProduct;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// General implementation
template <
    /// Size of the matrix product (concept: GemmShape)
    typename Shape_,
    /// Data type of A elements
    typename ElementA_,
    /// Data type of B elements
    typename ElementB_,
    /// Element type of C matrix
    typename ElementC_>
struct ElementwiseInnerProduct<Shape_, 1, ElementA_, ElementB_, ElementC_, arch::OpMultiplyAdd> {
  using Shape = Shape_;
  using Operator = arch::OpMultiplyAdd;
  using ElementC = ElementC_;

  CUTLASS_HOST_DEVICE
  void operator()(Array<ElementC_, Shape::kN> &d,
                  Array<ElementA_, Shape::kN> const &a,
                  Array<ElementB_, Shape::kN> const &b,
                  Array<ElementC_, Shape::kN> const &c) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Shape::kN; ++i) {
      d[i] = a[i] * b[i] + c[i];
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Specialization of half_t
template <>
struct ElementwiseInnerProduct<
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  half_t,
  half_t,
  arch::OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator =  arch::OpMultiplyAdd;
  using ElementC = half_t;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 2> const &c
  ) {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))

    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 const & B = reinterpret_cast<__half2 const &>(b);
    __half2 const & C = reinterpret_cast<__half2 const &>(c);

    __half2 tmp_D = __hfma2(A, B, C);

    d = reinterpret_cast<Array<half_t, 2> const &>(tmp_D);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[i] * b[i] + c[i];
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape,
  /// Data type of A elements
  typename ElementA,
  /// Data type of B elements
  typename ElementB,
  /// Element type of C matrix
  typename ElementC,
  /// Concept: arch::OpMultiplyAdd or arch::Mma<>
  typename Operator = arch::OpMultiplyAdd,
  /// Used for partial specialization
  typename Enable = bool
>
struct DepthwiseDirectConvElementwiseInnerProduct;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles all packed matrix layouts
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Data type of B elements
  typename ElementB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Operator used to compute GEMM
  typename Operator_
>
struct DepthwiseDirectConvElementwiseInnerProductGeneric {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = ElementA_;

  /// Data type of operand B
  using ElementB = ElementB_;

  /// Element type of operand C
  using ElementC = ElementC_;

  /// Underlying mathematical operator
  using Operator = Operator_;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMN>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Instruction
  using MmaOp = cutlass::conv::thread::ElementwiseInnerProduct<
    gemm::GemmShape<Shape::kN, Shape::kN, 1>,
    1,
    ElementA,
    ElementB,
    ElementC,
    Operator>;


  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {
    Array<ElementC, Shape::kN> *ptr_D = reinterpret_cast<Array<ElementC, Shape::kN> *>(&D);
    Array<ElementA, Shape::kN> const *ptr_A =
        reinterpret_cast<Array<ElementA, Shape::kN> const *>(&A);
    Array<ElementB, Shape::kN> const *ptr_B =
        reinterpret_cast<Array<ElementB, Shape::kN> const *>(&B);

    MmaOp mma_op;

    // Copy accumulators
    D = C;

    // Compute matrix product
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Shape::kN / MmaOp::Shape::kN; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Shape::kM; ++m) {

          Array<ElementC, MmaOp::Shape::kN> tmpD = ptr_D[m * Shape::kN / MmaOp::Shape::kN + n];
          Array<ElementA, MmaOp::Shape::kN> tmpA = ptr_A[m * Shape::kN / MmaOp::Shape::kN + n];
          Array<ElementB, MmaOp::Shape::kN> tmpB = ptr_B[n];

          mma_op(tmpD, tmpA, tmpB, tmpD);

          ptr_D[m * Shape::kN / MmaOp::Shape::kN + n] = tmpD;

        }
      }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
    /// Data type of A elements
  typename ElementA_,
  /// Data type of B elements
  typename ElementB_,
  /// Element type of C matrix
  typename ElementC_
>
struct DepthwiseDirectConvElementwiseInnerProduct<
  Shape_,
  ElementA_,
  ElementB_,
  ElementC_,
  arch::OpMultiplyAdd
  > {
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = ElementA_;

  /// Data type of operand B
  using ElementB = ElementB_;

  /// Element type of operand C
  using ElementC = ElementC_;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  /// A operand storage
  using FragmentA =
      Array<ElementA, Shape::kMN>;  // output_tile_size per thread * groups_per_thread

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kN>;  // 1 * groups_per_thread

  /// C operand storage
  using FragmentC =
      Array<ElementC, Shape::kMN>;  // output_tile_size per thread * groups_per_thread

  static bool const use_optimized = 0;

  using ArchMmaOperator =  DepthwiseDirectConvElementwiseInnerProductGeneric<Shape,
                                                        ElementA,
                                                        ElementB,
                                                        ElementC,
                                                        Operator>;

  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    ArchMmaOperator mma;

    mma(D, A, B, C);

  }
};

} // namespace thread
} // namespace conv
} // namespace cutlass
