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
    \brief Templates implementing loading of convolution tiles mapped to GEMM B (filter tile) 
    matrix from memory.

    This iterator assumes TensorNHWC layout of tensors in Global Memory.

    The iterator is specialized for each of the three convolution operators: forward propagation (Fprop),
    backward data gradient (Dgrad), and backward weight gradient (Wgrad). 
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv3d_problem_size.h"

#include "cutlass/conv/threadblock/conv3d_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_,
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kUnity
>
class Conv3dDgradFilterTileAccessIteratorOptimized {
public:
  
  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNDHWC;
  using ThreadMap = ThreadMap_;
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = StrideSupport_;
  static int const kConvDim = 3;
  using ConvProblemSize = typename conv::Conv3dProblemSize;
  static int const kAccessesPerVector = 1;
  
  //
  // Parameters structure
  //

  struct Params : Conv3dDgradFilterIteratorOptimizedParams {

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Conv3dDgradFilterIteratorOptimizedParams const &base): 
      Conv3dDgradFilterIteratorOptimizedParams(base) { }
      
    CUTLASS_HOST_DEVICE
    Params(
      Conv3dProblemSize const &problem_size, 
      Layout const &layout
    ):
      Conv3dDgradFilterIteratorOptimizedParams(
        problem_size,
        layout,
        sizeof_bits<Element>::value,
        {Shape::kRow, Shape::kColumn},
        ThreadMap::kThreads,
        ThreadMap::kElementsPerAccess,
        {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
        {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided}
      ) { }

  };

private:

  Conv3dDgradFilterIteratorOptimizedParams const &params_;
  Conv3dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  char const *pointer_;

  uint32_t predicates_;
  int filter_trs_;
  int filter_k_;

  //
  // Assertions
  //

  // We map predicates into bits packed in this uint32_t container
  static_assert(ThreadMap::Iterations::kStrided *
    ThreadMap::Iterations::kContiguous < sizeof(predicates_) * 8,
    "Currently, the number of loads per iteration is limited by the size of the predicates container.");

public:

  CUTLASS_HOST_DEVICE
  Conv3dDgradFilterTileAccessIteratorOptimized(
    Conv3dDgradFilterIteratorOptimizedParams const &params,
    Conv3dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params), 
    problem_size_(problem_size),
    pointer_(reinterpret_cast<char const *>(ptr)),
    predicates_(0),
    filter_trs_(0),
    filter_k_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.row() + thread_coord.strided();
    Index column = threadblock_offset.column() + thread_coord.contiguous();

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int filter_k = filter_k_ + s * ThreadMap::Delta::kStrided;
        int filter_c = column + c * ThreadMap::Delta::kContiguous;

        uint32_t pred = ((filter_k < problem_size_.K && filter_c < problem_size_.C) ? 1u : 0);

        int pred_idx = c + s * ThreadMap::Iterations::kContiguous;
        
        predicates_ |= (pred << pred_idx);
      }
    }

    pointer_ += (
      filter_k_ * params.layout.stride()[3] + column
    ) * sizeof_bits<Element>::value / 8;

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {

    pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_HOST_DEVICE
  void advance() {

    LongIndex next = params_.inc_next_trs;

    // moves to the next tile
    ++filter_trs_;
    if (filter_trs_ == params_.TRS) {

      filter_trs_ = 0;
      next = params_.inc_next_k;
      filter_k_ += params_.filter_k_delta;
    }

    // Clear predicates if needed
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      if (filter_k_ + s * ThreadMap::Delta::kStrided >= problem_size_.K) {
        uint32_t kClearMask = ((1u << ThreadMap::Iterations::kContiguous) - 1) << (s * ThreadMap::Iterations::kContiguous);

        predicates_ = (predicates_ & (~kClearMask));
      }
    }
      
    pointer_ += next;
  }

  /// Returns true if the current coordinate is within the filter tensor W
  CUTLASS_HOST_DEVICE
  bool valid() {
    LongIndex pred_idx = iteration_contiguous_ + iteration_strided_ * ThreadMap::Iterations::kContiguous;
    return (predicates_ & (1u << pred_idx));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    return reinterpret_cast<AccessType const *>(pointer_ + 
      iteration_contiguous_ * ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value / 8);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv3dDgradFilterTileAccessIteratorOptimized &operator++() {
    ++iteration_contiguous_;
    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }
    iteration_contiguous_ = 0;
    
    ++iteration_strided_;
    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {

      // Move to the next K coordinate within the tile
      pointer_ += params_.inc_next_strided;

      return *this;
    }
    iteration_strided_ = 0;
 
    return *this;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv3dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.C % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
