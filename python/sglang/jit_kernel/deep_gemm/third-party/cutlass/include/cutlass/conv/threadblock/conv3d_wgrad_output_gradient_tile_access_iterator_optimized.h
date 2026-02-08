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
    \brief Templates implementing loading of convolution tiles mapped to GEMM A (output gradient tile) 
    matrix from memory.

    This iterator assumes TensorNDHWC layout of tensors in Global Memory.

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
  typename ThreadMap_
>
class Conv3dWgradOutputGradientTileAccessIteratorOptimized {
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
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 3;
  using ConvProblemSize = typename conv::Conv3dProblemSize;
  static int const kAccessesPerVector = 1;  
  static_assert(sizeof_bits<Element>::value >= 8,
    "WGRAD requires elements of size 8b or greater.");

  //
  // Parameters structure
  //

  struct Params : Conv3dWgradOutputGradientIteratorOptimizedParams {
    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(Conv3dWgradOutputGradientIteratorOptimizedParams const &base)
          : Conv3dWgradOutputGradientIteratorOptimizedParams(base) {}

    CUTLASS_HOST_DEVICE
    Params(Conv3dProblemSize const &problem_size, Layout const &layout)
          : Conv3dWgradOutputGradientIteratorOptimizedParams(
            problem_size,
            layout,
            sizeof_bits<Element>::value,
            {Shape::kRow, Shape::kColumn},
            ThreadMap::kThreads,
            ThreadMap::kElementsPerAccess,
            {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
            {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided}) {}
    };

private:

  Params const &params_;
  Conv3dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  char const *pointer_;
    
  uint32_t predicates_;
  int filter_k_;
  int offset_nzpq_;

public:

  CUTLASS_HOST_DEVICE
  Conv3dWgradOutputGradientTileAccessIteratorOptimized(
    Params const &params, 
    Conv3dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params), 
    problem_size_(problem_size),
    pointer_(reinterpret_cast<char const *>(ptr)),
    predicates_(0),
    filter_k_(0),
    offset_nzpq_(0) {


    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.row() + thread_coord.contiguous();
    offset_nzpq_ = threadblock_offset.column() + thread_coord.strided();

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int filter_k = filter_k_ + c * ThreadMap::Delta::kContiguous;
        int offset_nzpq = offset_nzpq_ + s * ThreadMap::Delta::kStrided;

        bool predicate = valid_(at_(offset_nzpq, filter_k));

        uint32_t pred = (predicate ? 1u : 0);

        int pred_idx = c + s * ThreadMap::Iterations::kContiguous;
        
        predicates_ |= (pred << pred_idx);
      }
    }

    // Offset pointer to (iteration_strided_, iteration_contiguous_) = (0, 0) 
    pointer_ += (
      offset_nzpq_ * params.layout.stride()[0] + filter_k_
    ) * sizeof_bits<Element>::value / 8;

    set_iteration_index(0);
  }

  CUTLASS_HOST_DEVICE
  static Params getParams(Conv3dProblemSize const &problem_size, Layout const &layout) {
    return Params(problem_size, layout);
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
    // moves to the next GEMM-K offset (offset_npq_) in GEMM-A by a CTA-K tile
    offset_nzpq_ += Shape::kColumn * problem_size_.split_k_slices;

    // Clear predicates if needed
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      if (offset_nzpq_ + s * ThreadMap::Delta::kStrided >= params_.NZPQ) {
        uint32_t kClearMask = ((1u << ThreadMap::Iterations::kContiguous) - 1) << (s * ThreadMap::Iterations::kContiguous); 
        predicates_ = (predicates_ & (~kClearMask));
      }
    }
    pointer_ += params_.inc_next_nzpq; 
  }

private:
  /// Returns the coordinate in the output gradient tensor Dy that is (offset_nzpq, k) pointed to
  /// by the iterator.
  CUTLASS_HOST_DEVICE
  TensorCoord at_(int offset_nzpq, int k) const {

    // The subseqnet fast_divmod() operations are equivalent to the following logical computation:
    //
    //
    // int nzpq = offset_nzpq_;
    // int n = nzpq / (problem_size_.Z * problem_size_.P * problem_size_.Q);
    // int residual = nzpq % (problem_size_.Z * problem_size_.P * problem_size_.Q);
    //
    // int z = residual / (problem_size_.P * problem_size_.Q);
    // residual = residual % (problem_size_.P * problem_size_.Q);
    //
    // int p = residual / problem_size_.Q;
    // int q = residual % problem_size_.Q;

    int residual, n, z, p, q;
    fast_divmod(n, residual, offset_nzpq, params_.ZPQ, params_.zpq_mul, params_.zpq_shr);
    fast_divmod(z, residual, residual, params_.PQ, params_.pq_mul, params_.pq_shr);
    fast_divmod(p, q, residual, problem_size_.Q, params_.q_mul, params_.q_shr);

    return TensorCoord(n, z, p, q, k);
  }

  /// Returns true if the coord is within the output gradient tensor Dy
  CUTLASS_HOST_DEVICE
  bool valid_(TensorCoord coord) const {

    return coord.n() < problem_size_.N &&
      coord.c() < problem_size_.K;
  }

public:

  /// Returns true if the current coordinate is within the output gradient tensor Dy
  CUTLASS_HOST_DEVICE
  bool valid() const {

    LongIndex pred_idx = iteration_contiguous_ + iteration_strided_ * ThreadMap::Iterations::kContiguous;
    return (predicates_ & (1u << pred_idx));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    
    return reinterpret_cast<AccessType const *>(
      pointer_ +
      iteration_strided_ * params_.offset_next_strided + 
      iteration_contiguous_ * params_.offset_next_contiguous
    );

  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv3dWgradOutputGradientTileAccessIteratorOptimized &operator++() {
    ++iteration_contiguous_;
    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }
    iteration_contiguous_ = 0;
    ++iteration_strided_;
    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }
    iteration_strided_ = 0;
 
    return *this;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv3dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.K % AccessType::kElements) {
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


