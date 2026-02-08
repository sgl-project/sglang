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
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/threadblock/conv2d_params.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

template <typename Shape_,
          typename Element_,
          typename Layout_,
          typename ThreadMap_,
          typename AccessType_ = cutlass::AlignedArray<Element_, ThreadMap_::kElementsPerAccess> >
class DepthwiseFpropFilterDirectConvTileAccessIteratorOptimized {
public:   
  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;
 
  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
  
  static int const kFilterSize = ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess * ThreadMap::kThreads *
           sizeof_bits<Element>::value / 8;

  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements), 
    "Vectors implied by the thread map must be divisible by the access type.");
 
  //
  // Simplifying assertions
  //
  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "Require Iterations::kContiguous == 1");

  //
  // Parameters structure
  //
  using Params = Depthwise2dFpropDirectConvFilterIteratorParams<Layout>;

 protected:

  Conv2dProblemSize const &problem_size_;
  Params const &params_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  char const *pointer_;

  int filter_k_;
  int offset_trs_[ThreadMap::Iterations::kStrided];

public:



  CUTLASS_HOST_DEVICE
  DepthwiseFpropFilterDirectConvTileAccessIteratorOptimized(
    Params const &params, 
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params), 
    problem_size_(problem_size), 
    pointer_(reinterpret_cast<char const *>(ptr)), 
    filter_k_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.column() + thread_coord.contiguous();

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      offset_trs_[s] = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;
    }

    set_iteration_index(0);
  }

  CUTLASS_HOST_DEVICE
  static Params getParams(Conv2dProblemSize const &problem_size, Layout const &layout) {
      return Params(problem_size, layout, {Shape::kRow, Shape::kColumn}, kFilterSize);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;
    iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset * 8 / sizeof_bits<Element>::value;
  }

  CUTLASS_HOST_DEVICE
  void advance() {
    // Do nothing because the filter is persistent in the SMEM
  }

  /// Returns the coordinate in the filter tensor W that is currently pointed to
  /// by the iterator.
  CUTLASS_HOST_DEVICE
  TensorCoord at() const {

    int k = filter_k_ + iteration_vector_ * AccessType::kElements;
    int trs =  offset_trs_[iteration_strided_];

    return TensorCoord(k, trs, 0 , 0);  // As a 2D-matrix
  }

  /// Returns true if the current coordinate is within the activations tensor W
  CUTLASS_HOST_DEVICE
  bool valid() const {

    TensorCoord coord = at();

    return coord.n() < problem_size_.K &&
            coord.h() < Shape::kColumn;
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    TensorCoord coord = at();
    int64_t offset = coord.n();
    if (params_.is_convolution) {
      offset += (Shape::kColumn - coord.h() - 1)* problem_size_.K;
    } else {
      offset += coord.h() * problem_size_.K;
    }

    return reinterpret_cast<AccessType const *>(pointer_ +
                                                offset * sizeof_bits<Element>::value / 8);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  DepthwiseFpropFilterDirectConvTileAccessIteratorOptimized &operator++() {
    ++iteration_vector_;
    if (iteration_vector_ < kAccessesPerVector) {
      return *this;
    }
    iteration_vector_ = 0;

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

  /// Determines the filter size loaded by iterator
  CUTLASS_HOST_DEVICE
  int get_load_size() {
    return kFilterSize;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.K % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    // check whether runtime filter size is same as templated filter size.
    if ((problem_size.R * problem_size.S) != Shape::kColumn) {
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
