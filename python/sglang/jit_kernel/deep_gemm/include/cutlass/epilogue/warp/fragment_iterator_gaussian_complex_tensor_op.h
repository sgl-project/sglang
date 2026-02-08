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
    \brief This defines a "fragment" iterator for visiting the fragments of an accumulator tile
      that participate in one warp-level store operation.

      Typically, the accumulator tile is the largest single block of register-backed storage 
      within the kernel. Storing it to memory is best accomplished by partitioning it into
      smaller tiles and storing these sequentially.

      Round trips through shared memory during the Epilogue phase require partitioning, as
      shared memory capacity is typically insufficient for a threadblock's total accumulator
      size.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/epilogue/warp/tensor_op_policy.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

/// 
template <
  typename WarpShape,         ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape,     ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorElementC,  ///< matrix multiply operation data type (concept: data type)
  typename OperatorFragmentC, ///< matrix multiply operation fragment (concept: Array)
  typename Layout             ///< target shared memory layout
>
class FragmentIteratorGaussianComplexTensorOp;

////////////////////////////////////////////////////////////////////////////////


/// Partial specialization for row-major shared memory
template <
  typename WarpShape_,         ///< shape of the warp-level GEMM tile
  typename OperatorShape_,     ///< underlying real-valued matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorElementC_,  ///< underlying real-valued matrix multiply operation data type
  typename OperatorFragmentC_  ///< underlying real-valued matrix multiply operation fragment (concept: Array)
>
class FragmentIteratorGaussianComplexTensorOp<WarpShape_, OperatorShape_, OperatorElementC_, OperatorFragmentC_, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using OperatorElementC = OperatorElementC_;
  using OperatorFragmentC = OperatorFragmentC_;
  using Layout = layout::RowMajor;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    complex<OperatorElementC>, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// Size of one part of accumulator of 3-part accumulator in units of number of OperatorElementC
  static int const kElementsAccumulatorPerPart = 
    OperatorFragmentC::kElements * Policy::OperatorCount::kRow * Policy::OperatorCount::kColumn;

  /// Offset into the accumulator fragment part 1
  static int const kPart1Index = kElementsAccumulatorPerPart * 0;

  /// Offset into the accumulator fragment part 2
  static int const kPart2Index = kElementsAccumulatorPerPart * 1;

  /// Offset into the accumulator fragment part 3
  static int const kPart3Index = kElementsAccumulatorPerPart * 2;

  /// This is the complete warp-level accumulator tile holding part1, part2, and part3
  using AccumulatorTile = Array<OperatorElementC, kElementsAccumulatorPerPart * 3>;

  /// This is the complete warp-level accumulator tile holding final output of complex<T> type 
  using OutputAccumulatorTile = Array<complex<OperatorElementC>, kElementsAccumulatorPerPart>;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

private:

  /// Internal access type
  using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;

  using FragmentAccessType = Array<complex<OperatorElementC>, Policy::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

public:

  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  FragmentIteratorGaussianComplexTensorOp(AccumulatorTile const &accum): 
    accumulators_(reinterpret_cast<AccessType const *>(&accum)), 
    index_(0) {
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  FragmentIteratorGaussianComplexTensorOp &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  FragmentIteratorGaussianComplexTensorOp &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {

    int index = index_ + index_offset;

    FragmentAccessType *frag_ptr = reinterpret_cast<FragmentAccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int accumulator_access_offset = 
        index + n * Policy::kAccumulatorColumnStride / Policy::kElementsPerAccess;

      auto const & part1_accum_array = accumulators_[accumulator_access_offset + kPart1Index];
      auto const & part2_accum_array = accumulators_[accumulator_access_offset + kPart2Index / Policy::kElementsPerAccess];
      auto const & part3_accum_array = accumulators_[accumulator_access_offset + kPart3Index / Policy::kElementsPerAccess];

      // Pack parts 1, 2, and 3 into a structure. This is likely to result in MOVs
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Policy::kElementsPerAccess; ++i) {

        frag_ptr[n][i].real() = part1_accum_array[i] - part3_accum_array[i];
        frag_ptr[n][i].imag() = part1_accum_array[i] + part2_accum_array[i]; 
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
