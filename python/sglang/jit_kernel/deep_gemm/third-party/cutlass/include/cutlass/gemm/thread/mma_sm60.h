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

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/thread/mma.h"
#include "cutlass/functional.h"
#include "cutlass/reduction/thread/reduce.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Structure to compute the matrix product for HFMA
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape,

  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,

  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,

  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC,

  /// Type of GEMM inner vs outer product
  bool
>
struct Mma_HFMA2;


/////////////////////////////
// Specialization for NNN  //
/////////////////////////////

template <typename Shape_>
struct Mma_HFMA2 <
  Shape_,
  layout::ColumnMajor,
  layout::ColumnMajor,
  layout::ColumnMajor,
  true
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

   /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kM % 2),
    "Mma_HFMA2 requires the M dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x1x1 HFMA2 sequence for bulk of computation
    using Mma = arch::Mma<
      gemm::GemmShape<2,1,1>,
      1,
      half_t,
      layout::ColumnMajor,
      half_t,
      layout::ColumnMajor,
      half_t,
      layout::ColumnMajor,
      arch::OpMultiplyAdd>;

    Array<half_t, 2> *ptr_D = reinterpret_cast<Array<half_t, 2> *>(&D);
    Array<half_t, 2> const *ptr_A = reinterpret_cast<Array<half_t, 2> const *>(&A);
    Array<half_t, 1> const *ptr_B = reinterpret_cast<Array<half_t, 1> const *>(&B);

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for(auto k=0; k <  Shape::kK / Mma::Shape::kK; k++){

      CUTLASS_PRAGMA_UNROLL
      for(auto m=0; m < Shape::kM / Mma::Shape::kM; m++){

        CUTLASS_PRAGMA_UNROLL
        for(auto n=0; n < Shape::kN / Mma::Shape::kN; n++){

            Array<half_t, 2> tmp { ptr_D[n*Shape::kM/2 + m] };

            mma(
                tmp,
                ptr_A[k*Shape::kM/2 + m],
                ptr_B[n*Shape::kK + k],
                tmp);

            ptr_D[n*Shape::kM/2 + m] = tmp;
        }
      }
    }
  }
};

/////////////////////////////
// Specialization for NNT  //
/////////////////////////////

template <typename Shape_>
struct Mma_HFMA2<
  Shape_,
  layout::ColumnMajor,
  layout::ColumnMajor,
  layout::RowMajor,
  true
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

   /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kN % 2),
    "Mma_HFMA2 requires the N dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x2x1 HFMA2 sequence for bulk of computation
    using Mma = arch::Mma<
      gemm::GemmShape<1,2,1>,
      1,
      half_t,
      layout::ColumnMajor,
      half_t,
      layout::ColumnMajor,
      half_t,
      layout::RowMajor,
      arch::OpMultiplyAdd>;

    Array<half_t, 2> *ptr_D = reinterpret_cast<Array<half_t, 2> *>(&D);
    Array<half_t, 1> const *ptr_A = reinterpret_cast<Array<half_t, 1> const *>(&A);
    Array<half_t, 2> const *ptr_B = reinterpret_cast<Array<half_t, 2> const *>(&B);

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for(auto k=0; k <  Shape::kK / Mma::Shape::kK; k++){

        CUTLASS_PRAGMA_UNROLL
        for(auto n=0; n < Shape::kN / Mma::Shape::kN; n++){

          CUTLASS_PRAGMA_UNROLL
          for(auto m=0; m < Shape::kM / Mma::Shape::kM; m++){

            Array<half_t, 2> tmp { ptr_D[m*Shape::kN/2 + n] };

            Array<half_t, 2> tmp_B;
            tmp_B[0] = ptr_B->at(2*n*Shape::kK + k);
            tmp_B[1] = ptr_B->at((2*n+1)*Shape::kK + k);

            mma(
                tmp,
                ptr_A[k*Shape::kM + m],
                tmp_B,
                tmp);

            ptr_D[m*Shape::kN/2 + n] = tmp;
        }
      }
    }
  }
};


/////////////////////////////
// Specialization for NTN  //
/////////////////////////////

template <typename Shape_>
struct Mma_HFMA2 <
  Shape_,
  layout::ColumnMajor,
  layout::RowMajor,
  layout::ColumnMajor,
  true
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kM % 2),
    "Mma_HFMA2 requires the GEMM M dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    using Mma = arch::Mma<
      gemm::GemmShape<2,1,1>,
      1,
      half_t,
      layout::ColumnMajor,
      half_t,
      layout::RowMajor,
      half_t,
      layout::ColumnMajor,
      arch::OpMultiplyAdd>;

    Array<half_t, 2> *ptr_D = reinterpret_cast<Array<half_t, 2> *>(&D);
    Array<half_t, 2> const *ptr_A = reinterpret_cast<Array<half_t, 2> const *>(&A);
    Array<half_t, 1> const *ptr_B = reinterpret_cast<Array<half_t, 1> const *>(&B);

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK / Mma::Shape::kK; ++k) {

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Shape::kM / Mma::Shape::kM; ++m) {

          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < Shape::kN / Mma::Shape::kN; ++n) {

          Array<half_t, 2> tmp { ptr_D[m + n * Shape::kM/2] };

          mma(
            tmp,
            ptr_A[m + k * Shape::kM/2],
            ptr_B[k * Shape::kN + n],
            tmp);

          ptr_D[m + n * Shape::kM/2] = tmp;
        }
      }
    }
  }
};

/////////////////////////////
// Specialization for NTT  //
/////////////////////////////

template <typename Shape_>
struct Mma_HFMA2<
  Shape_,
  layout::ColumnMajor,
  layout::RowMajor,
  layout::RowMajor,
  true
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kN % 2),
    "Mma_HFMA2 requires the N dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x2x1 HFMA2 sequence for bulk of computation
    using Mma = arch::Mma<
      gemm::GemmShape<1,2,1>,
      1,
      half_t,
      layout::ColumnMajor,
      half_t,
      layout::RowMajor,
      half_t,
      layout::RowMajor,
      arch::OpMultiplyAdd>;

    Array<half_t, 2> *ptr_D = reinterpret_cast<Array<half_t, 2> *>(&D);
    Array<half_t, 1> const *ptr_A = reinterpret_cast<Array<half_t, 1> const *>(&A);
    Array<half_t, 2> const *ptr_B = reinterpret_cast<Array<half_t, 2> const *>(&B);

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for(auto k=0; k <  Shape::kK / Mma::Shape::kK; k++){

        CUTLASS_PRAGMA_UNROLL
        for(auto n=0; n < Shape::kN / Mma::Shape::kN; n++){

          CUTLASS_PRAGMA_UNROLL
          for(auto m=0; m < Shape::kM / Mma::Shape::kM; m++){

            Array<half_t, 2> tmp { ptr_D[m*Shape::kN/2 + n] };

            mma(
                tmp,
                ptr_A[k*Shape::kM + m],
                ptr_B[k*Shape::kN/2 + n],
                tmp);

            ptr_D[m*Shape::kN/2 + n] = tmp;
        }
      }
    }
  }
};


/////////////////////////////
// Specialization for TNN  //
/////////////////////////////

template <typename Shape_>
struct Mma_HFMA2 <
  Shape_,
  layout::RowMajor,
  layout::ColumnMajor,
  layout::ColumnMajor,
  true
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kM % 2),
    "Mma_HFMA2 requires the M dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x1x1 HFMA2 sequence for bulk of computation
    using Mma = arch::Mma<
      gemm::GemmShape<2,1,1>,
      1,
      half_t,
      layout::RowMajor,
      half_t,
      layout::ColumnMajor,
      half_t,
      layout::ColumnMajor,
      arch::OpMultiplyAdd>;

    Array<half_t, 2> *ptr_D = reinterpret_cast<Array<half_t, 2> *>(&D);
    Array<half_t, 2> const *ptr_A = reinterpret_cast<Array<half_t, 2> const *>(&A);
    Array<half_t, 1> const *ptr_B = reinterpret_cast<Array<half_t, 1> const *>(&B);

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for(auto k=0; k <  Shape::kK / Mma::Shape::kK; k++){

      CUTLASS_PRAGMA_UNROLL
      for(auto m=0; m < Shape::kM / Mma::Shape::kM; m++){

        CUTLASS_PRAGMA_UNROLL
        for(auto n=0; n < Shape::kN / Mma::Shape::kN; n++){

            Array<half_t, 2> tmp { ptr_D[n*Shape::kM/2 + m] };

            Array<half_t, 2> tmp_A;
            tmp_A[0] = ptr_A->at(2*m*Shape::kK + k);
            tmp_A[1] = ptr_A->at((2*m+1)*Shape::kK + k);

            mma(
                tmp,
                tmp_A,
                ptr_B[n*Shape::kK + k],
                tmp);

            ptr_D[n*Shape::kM/2 + m] = tmp;
        }
      }
    }
  }
};

/////////////////////////////
// Specialization for TNT  //
/////////////////////////////

template <typename Shape_>
struct Mma_HFMA2 <
  Shape_,
  layout::RowMajor,
  layout::ColumnMajor,
  layout::RowMajor,
  true
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

   /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kN % 2),
    "Mma_HFMA2 requires the N dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x2x1 HFMA2 sequence for bulk of computation
    using Mma = arch::Mma<
      gemm::GemmShape<1,2,1>,
      1,
      half_t,
      layout::RowMajor,
      half_t,
      layout::ColumnMajor,
      half_t,
      layout::RowMajor,
      arch::OpMultiplyAdd>;

    Array<half_t, 2> *ptr_D = reinterpret_cast<Array<half_t, 2> *>(&D);
    Array<half_t, 1> const *ptr_A = reinterpret_cast<Array<half_t, 1> const *>(&A);
    Array<half_t, 2> const *ptr_B = reinterpret_cast<Array<half_t, 2> const *>(&B);

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for(auto k=0; k <  Shape::kK / Mma::Shape::kK; k++){

        CUTLASS_PRAGMA_UNROLL
        for(auto n=0; n < Shape::kN / Mma::Shape::kN; n++){

          CUTLASS_PRAGMA_UNROLL
          for(auto m=0; m < Shape::kM / Mma::Shape::kM; m++){

            Array<half_t, 2> tmp { ptr_D[m*Shape::kN/2 + n] };

            Array<half_t, 2> tmp_B;
            tmp_B[0] = ptr_B->at(2*n*Shape::kK + k);
            tmp_B[1] = ptr_B->at((2*n+1)*Shape::kK + k);

            mma(
                tmp,
                ptr_A[m*Shape::kK + k],
                tmp_B,
                tmp);

            ptr_D[m*Shape::kN/2 + n] = tmp;
        }
      }
    }
  }
};

/////////////////////////////
// Specialization for TTN  //
/////////////////////////////

template <typename Shape_>
struct Mma_HFMA2 <
  Shape_,
  layout::RowMajor,
  layout::RowMajor,
  layout::ColumnMajor,
  true
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

   /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kM % 2),
    "Mma_HFMA2 requires the M dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x2x1 HFMA2 sequence for bulk of computation
    using Mma = arch::Mma<
      gemm::GemmShape<2,1,1>,
      1,
      half_t,
      layout::RowMajor,
      half_t,
      layout::RowMajor,
      half_t,
      layout::ColumnMajor,
      arch::OpMultiplyAdd>;

    Array<half_t, 2> *ptr_D = reinterpret_cast<Array<half_t, 2> *>(&D);
    Array<half_t, 2> const *ptr_A = reinterpret_cast<Array<half_t, 2> const *>(&A);
    Array<half_t, 1> const *ptr_B = reinterpret_cast<Array<half_t, 1> const *>(&B);

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for(auto k=0; k <  Shape::kK / Mma::Shape::kK; k++){

      CUTLASS_PRAGMA_UNROLL
      for(auto m=0; m < Shape::kM / Mma::Shape::kM; m++){

        CUTLASS_PRAGMA_UNROLL
        for(auto n=0; n < Shape::kN / Mma::Shape::kN; n++){

            Array<half_t, 2> tmp { ptr_D[n*Shape::kM/2 + m] };

            Array<half_t, 2> tmp_A;
            tmp_A[0] = ptr_A->at(2*m*Shape::kK + k);
            tmp_A[1] = ptr_A->at((2*m+1)*Shape::kK + k);

            mma(
                tmp,
                tmp_A,
                ptr_B[k*Shape::kN + n],
                tmp);

            ptr_D[n*Shape::kM/2 + m] = tmp;
        }
      }
    }
  }
};


/////////////////////////////
// Specialization for TTT  //
/////////////////////////////

template <typename Shape_>
struct Mma_HFMA2<
  Shape_,
  layout::RowMajor,
  layout::RowMajor,
  layout::RowMajor,
  true
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kN % 2),
    "Mma_HFMA2 requires the N dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x2x1 HFMA2 sequence for bulk of computation
    using Mma = arch::Mma<
      gemm::GemmShape<1,2,1>,
      1,
      half_t,
      layout::RowMajor,
      half_t,
      layout::RowMajor,
      half_t,
      layout::RowMajor,
      arch::OpMultiplyAdd>;

    Array<half_t, 2> *ptr_D = reinterpret_cast<Array<half_t, 2> *>(&D);
    Array<half_t, 1> const *ptr_A = reinterpret_cast<Array<half_t, 1> const *>(&A);
    Array<half_t, 2> const *ptr_B = reinterpret_cast<Array<half_t, 2> const *>(&B);

    Mma mma;

    CUTLASS_PRAGMA_UNROLL
    for(auto k=0; k <  Shape::kK / Mma::Shape::kK; k++){

        CUTLASS_PRAGMA_UNROLL
        for(auto n=0; n < Shape::kN / Mma::Shape::kN; n++){

          CUTLASS_PRAGMA_UNROLL
          for(auto m=0; m < Shape::kM / Mma::Shape::kM; m++){

            Array<half_t, 2> tmp { ptr_D[m*Shape::kN/2 + n] };

            mma(
                tmp,
                ptr_A[m*Shape::kK + k],
                ptr_B[k*Shape::kN/2 + n],
                tmp);

            ptr_D[m*Shape::kN/2 + n] = tmp;
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////
// Specialization for TNT + Inner Product  or 1x1x2K + LayoutC = T //
/////////////////////////////////////////////////////////////////////

template <typename Shape_, typename LayoutA, typename LayoutB>
struct Mma_HFMA2<
  Shape_,
  LayoutA,
  LayoutB,
  layout::RowMajor,
  false
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kK % 2),
    "Mma_HFMA2 requires the K dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x1x2 HFMA2 sequence for bulk of computation
    using GemmShape = gemm::GemmShape<1,1,2>;

    Array<half_t, 1> *ptr_D = reinterpret_cast<Array<half_t, 1> *>(&D);
    Array<half_t, 2> const *ptr_A = reinterpret_cast<Array<half_t, 2> const *>(&A);
    Array<half_t, 2> const *ptr_B = reinterpret_cast<Array<half_t, 2> const *>(&B);

    // Inner product is calculated using MACs, followed by final reduction
    multiply_add<Array<half_t, 2>> mac;
    cutlass::reduction::thread::Reduce< plus<half_t>, Array<half_t, 2> > reduce;

    CUTLASS_PRAGMA_UNROLL
    for(auto n=0; n < Shape::kN / GemmShape::kN; n++){ 

      CUTLASS_PRAGMA_UNROLL
      for(auto m=0; m < Shape::kM / GemmShape::kM; m++){

        Array<half_t, 2> tmp_C;
        tmp_C.clear();
        Array<half_t, 1> *ptr_tmp_C = reinterpret_cast<Array<half_t, 1> *>(&tmp_C);
        ptr_tmp_C[0] = ptr_D[n*Shape::kM + m];

        CUTLASS_PRAGMA_UNROLL
        for(auto k=0; k <  Shape::kK / GemmShape::kK; k++){ 
          tmp_C = mac(ptr_A[m*Shape::kK/2 + k], ptr_B[n*Shape::kK/2 + k], tmp_C);
        }

        Array<half_t, 1> res;
        Array<half_t, 1> *ptr_res = &res;
        res = reduce(tmp_C);

        ptr_D[m*Shape::kN + n] = ptr_res[0];
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////
// Specialization for TNN + Inner Product  or 1x1x2K + LayoutC = N //
/////////////////////////////////////////////////////////////////////

template <typename Shape_, typename LayoutA, typename LayoutB>
struct Mma_HFMA2<
  Shape_,
  LayoutA,
  LayoutB,
  layout::ColumnMajor,
  false
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// A operand storage
  using FragmentA = Array<half_t, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<half_t, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<half_t, Shape::kMN>;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  static_assert(
    !(Shape::kK % 2),
    "Mma_HFMA2 requires the K dimension to be divisible by 2."
  );

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

    /// Initialize output with input
    D = C;

    /// Use 1x1x2 HFMA2 sequence for bulk of computation
    using GemmShape= gemm::GemmShape<1,1,2>;

    Array<half_t, 1> *ptr_D = reinterpret_cast<Array<half_t, 1> *>(&D);
    Array<half_t, 2> const *ptr_A = reinterpret_cast<Array<half_t, 2> const *>(&A);
    Array<half_t, 2> const *ptr_B = reinterpret_cast<Array<half_t, 2> const *>(&B);

    // Inner product is calculated using MACs, followed by final reduction
    multiply_add<Array<half_t, 2>> mac;
    cutlass::reduction::thread::Reduce< plus<half_t>, Array<half_t, 2> > reduce;

    CUTLASS_PRAGMA_UNROLL
    for(auto n=0; n < Shape::kN / GemmShape::kN; n++){ 

      CUTLASS_PRAGMA_UNROLL
      for(auto m=0; m < Shape::kM / GemmShape::kM; m++){

        Array<half_t, 2> tmp_C;
        tmp_C.clear();
        Array<half_t, 1> *ptr_tmp_C = reinterpret_cast<Array<half_t, 1> *>(&tmp_C);
        ptr_tmp_C[0] = ptr_D[n*Shape::kM + m];

        CUTLASS_PRAGMA_UNROLL
        for(auto k=0; k <  Shape::kK / GemmShape::kK; k++){ 

          tmp_C = mac(ptr_A[m*Shape::kK/2 + k], ptr_B[n*Shape::kK/2 + k], tmp_C);

        }

        Array<half_t, 1> res;
        Array<half_t, 1> *ptr_res = &res;
        res = reduce(tmp_C);

        ptr_D[n*Shape::kM + m] = ptr_res[0];
      }
    }
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_, typename LayoutA, typename LayoutB, typename LayoutC
>
struct Mma<
  Shape_,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  LayoutC,
  arch::OpMultiplyAdd
  > {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = half_t;

  /// Data type of operand B
  using ElementB = half_t;

  /// Element type of operand C
  using ElementC = half_t;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  static bool const a_row_major = platform::is_same< LayoutA, layout::RowMajor>::value;
  static bool const b_column_major = platform::is_same< LayoutB, layout::ColumnMajor>::value;
  static bool const c_row_major = platform::is_same< LayoutC, layout::RowMajor>::value;
  static bool const c_column_major = platform::is_same< LayoutC, layout::ColumnMajor>::value;

  static bool const m_mod2 = !(Shape::kM % 2);
  static bool const n_mod2 = !(Shape::kN % 2);
  static bool const k_mod2 = !(Shape::kK % 2);

  // HFMA based MMA optimizations are of 2 types :
  // 1. Inner product 
  // 2. Outer product
  // It is chosen based on LayoutC (for outer product gemm) or
  // Using LayoutA and LayoutB or shape=1x1x2K (for inner product gemms)
  // If all fails, we choose the generic MMA
  static bool const use_outer_prod = (c_column_major && m_mod2) || (c_row_major && n_mod2);
  static bool const use_inner_prod = (a_row_major && b_column_major && k_mod2) || (Shape::kM==1 && Shape::kN==1 && k_mod2);
  static bool const use_optimized =  (use_outer_prod || use_inner_prod);

  using ArchMmaOperator = typename platform::conditional< use_optimized, 
    detail::Mma_HFMA2<Shape, LayoutA, LayoutB, LayoutC, use_outer_prod>, 
    MmaGeneric <Shape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Operator> 
  >::type;

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

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

  /// Determines whether to enable thread::Gemm<> specializations compatible with SM50
  template <
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB>
  struct EnableMma_Crow_SM60 {

    static bool const kIsConventionalLayout =
      (platform::is_same<LayoutA, layout::RowMajor>::value ||
        platform::is_same<LayoutA, layout::ColumnMajor>::value) &&
      (platform::is_same<LayoutB, layout::RowMajor>::value ||
        platform::is_same<LayoutB, layout::ColumnMajor>::value);

    static bool const value = kIsConventionalLayout;
  };
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes matrix product when C is row-major
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  typename LayoutA_,
  typename LayoutB_
>
struct Mma<
  Shape_,
  half_t,
  LayoutA_,
  half_t,
  LayoutB_,
  half_t,
  layout::RowMajor,
  arch::OpMultiplyAdd,
  typename platform::enable_if<detail::EnableMma_Crow_SM60<
    LayoutA_,
    LayoutB_
    >::value>::type>{

  using Shape = Shape_;
  using ElementA = half_t;
  using LayoutA = LayoutA_;
  using ElementB = half_t;
  using LayoutB = LayoutB_;
  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using Operator = arch::OpMultiplyAdd;

  using TransposeMma = Mma<
    GemmShapeTranspose<Shape>,
    half_t,
    typename layout::LayoutTranspose<LayoutB>::type,
    half_t,
    typename layout::LayoutTranspose<LayoutA>::type,
    half_t,
    layout::ColumnMajor,
    arch::OpMultiplyAdd,
    bool>;

  using FragmentA = Array<ElementA, Shape::kMK>;
  using FragmentB = Array<ElementB, Shape::kKN>;
  using FragmentC = Array<ElementC, Shape::kMN>;

  using ArchMmaOperator = typename TransposeMma::ArchMmaOperator;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TransposeMma mma;

    mma(D, B, A, C);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
