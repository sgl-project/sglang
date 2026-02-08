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
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the operation implied by MMA.
struct OpMultiplyAdd {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the result is saturated to MAX_FLOAT|MIN_FLOAT or MAX_INT|MIN_INT
struct OpMultiplyAddSaturate {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the input is converted to a narrower type (BF16)
struct OpMultiplyAddFastBF16 {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the input is converted to a narrower type (F16)
struct OpMultiplyAddFastF16 {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the input data types are mixed and the narrower type is 
/// upcasted to the wider type
struct OpMultiplyAddMixedInputUpcast {};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Tag indicating the input is converted to 2 (big and small) TF32 or FP16 components
//  Perform 3xTF32 or 4xTF32 for every F32 output element on Ampere
//  Perform 3xFP16 or 4xFP16 for every F32 output element on Hopper with axiswise quantization factor support
struct OpMultiplyAddFastF32 {};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Tag indicating the input is converted to 2 (big and small) TF32 or FP16 components
//  Perform 3xTF32 or 4xTF32 for every complex<F32> output element on Ampere
//  Perform 3xFP16 or 4xFP16 for every complex<F32> output element on Hopper with axiswise quantization factor support
struct OpMultiplyAddComplexFastF32 {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating that staged accumulation is not to be used. This is valid only for SM89
/// FP8 kernels.
struct OpMultiplyAddFastAccum;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the complex multiply-add operation
struct OpMultiplyAddComplex {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the gaussian complex multiply-add operation
struct OpMultiplyAddGaussianComplex {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the inner product is defined by (XOR, POPC)
struct OpXorPopc {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag indicating the inner product is defined by (AND, POPC)
struct OpAndPopc {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag classifying math operators as thread-level operations.
struct OpClassSimt {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag classifying operators as Tensor Core operations.
struct OpClassTensorOp {};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Tag classifying operators as WMMA Tensor Core operations
struct OpClassWmmaTensorOp {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tag classifying operators as Tensor Core with structure sparse operations.
struct OpClassSparseTensorOp {};


/// Tag classifying operators as Tensor Core with blockScaled
struct OpClassBlockScaledTensorOp {};

/// Tag classifying operators as Tensor Core with blockScaled structured sparse operations.
struct OpClassBlockScaledSparseTensorOp {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Size of the matrix product (concept: GemmShape)
  typename Shape_,
  /// Number of threads participating
  int kThreads_,
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC,
  /// Inner product operator
  typename Operator
>
struct Mma;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - specialized for 1x1x1x1 matrix multiply operation
template <
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC,
  /// Inner product operator
  typename Operator_
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, ElementA, LayoutA, ElementB, LayoutB, ElementC_, LayoutC, Operator_> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = Operator_;
  using ElementC = ElementC_;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<ElementC, 1> &d,
    Array<ElementA, 1> const &a,
    Array<ElementB, 1> const &b,
    Array<ElementC, 1> const &c
  ) {

    multiply_add<ElementA, ElementB, ElementC> op;

    d[0] = op(a[0], b[0], c[0]);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specifies internal data type for computation
struct SPFormatType {
  enum Kind {
    Thread
  };
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Size of the matrix product (concept: GemmShape)
  typename Shape_,
  /// Number of threads participating
  int kThreads_,
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC,
  /// Inner product operator
  typename Operator,
  /// Specifies meta data format
  SPFormatType::Kind SPFormat = SPFormatType::Thread
>
struct SparseMma;

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Specializations for each compute capability
//

#include "cutlass/arch/mma_sm50.h"
#include "cutlass/arch/mma_sm60.h"
#include "cutlass/arch/mma_sm61.h"
#include "cutlass/arch/mma_sm70.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/arch/mma_sparse_sm80.h"
#include "cutlass/arch/mma_sm89.h"
#include "cutlass/arch/mma_sparse_sm89.h"
#include "cutlass/arch/mma_sm90.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {
namespace detail {
/// Helper for determining whether staged accumulation should be used for a given operator
template <typename Operator>
struct UseStagedAccumulation {
  static bool const value = platform::is_same<typename Operator::MathOperator, OpMultiplyAddFastF32>::value ||
                            platform::is_same<typename Operator::MathOperator, OpMultiplyAddComplexFastF32>::value ||
                            is_sm89_staged_policy_v<Operator>;
};
} // namespace detail
} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
