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

    This iterator assumes TensorNHWC or TensorCxRSKx<Interleave> layout of tensors in Global Memory.

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
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/threadblock/conv2d_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,
  typename Element_,
  typename Layout_,
  typename ThreadMap_,
  typename AccessType_ = cutlass::AlignedArray<Element_, ThreadMap_::kElementsPerAccess>,
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone,
  bool IsDeconv_ = false
>
class Conv2dFpropFilterTileAccessIteratorAnalytic {
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
  static bool const IsDeconv = IsDeconv_;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kAnalytic;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;
  static conv::GroupMode const kGroupMode = GroupMode_;
 
  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
  
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

  using Params = Conv2dAnalyticParams<Layout>;

private:

  Params const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  char const *pointer_;

  int filter_r_;
  int filter_s_;
  int filter_c_;
  int filter_c_init_;
  int crs_cnt_;
  int crs_per_group_;  
  int group_idx_offset_c_;
  int channels_per_group_;

  int offset_k_[ThreadMap::Iterations::kStrided];
  int group_idx_offset_k_[ThreadMap::Iterations::kStrided];

public:

  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterTileAccessIteratorAnalytic(
    Params const &params, 
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params), 
    problem_size_(problem_size), 
    pointer_(reinterpret_cast<char const *>(ptr)), 
    crs_cnt_(0),
    group_idx_offset_c_(0),
    filter_r_(0),
    filter_s_(0),
    filter_c_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_c_ = threadblock_offset.row() + thread_coord.contiguous();

    auto input_channels = (IsDeconv ? problem_size_.K : problem_size_.C);
    auto output_channels = (IsDeconv ? problem_size_.C : problem_size_.K);

    if (kGroupMode != conv::GroupMode::kNone) {
      filter_c_init_ = filter_c_;
      if (kGroupMode == conv::GroupMode::kDepthwise){
        channels_per_group_ = 1;
        crs_per_group_ = problem_size_.S * problem_size_.R;
      } else {
        channels_per_group_ = input_channels / problem_size_.groups;
        crs_per_group_ = problem_size_.S * problem_size_.R * ((channels_per_group_ + Shape::kRow - 1) / Shape::kRow);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      offset_k_[s] = threadblock_offset.column() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;
      if (kGroupMode != conv::GroupMode::kNone && kGroupMode != conv::GroupMode::kDepthwise) {
        group_idx_offset_k_[s] = (thread_coord.strided() + s * ThreadMap::Delta::kStrided) / (output_channels / problem_size_.groups);
      }
    }

    set_iteration_index(0);
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
    // moves to the next tile
    if (kGroupMode != conv::GroupMode::kNone) {
      ++crs_cnt_;
    }

    ++filter_s_;
    if (filter_s_ < problem_size_.S) {
      return;
    }
    filter_s_ = 0;
    
    ++filter_r_;
    if (filter_r_ < problem_size_.R) {
      return;
    }
    filter_r_ = 0;

    if (kGroupMode == conv::GroupMode::kNone) {
      filter_c_ += Shape::kRow * problem_size_.split_k_slices;
    } else {
      if (crs_cnt_ == crs_per_group_) {
        crs_cnt_ = 0;
        filter_c_ = filter_c_init_;
        if (kGroupMode != conv::GroupMode::kDepthwise) {
          // moves to next group
          ++group_idx_offset_c_;
        }
      } else {
        filter_c_ += Shape::kRow * problem_size_.split_k_slices;
      }
    }
  }

  /// Returns the coordinate in the filter tensor W that is currently pointed to
  /// by the iterator.
  CUTLASS_HOST_DEVICE
  TensorCoord at() const {

    int k = offset_k_[iteration_strided_];
    int c = filter_c_ + iteration_vector_ * AccessType::kElements;

    return TensorCoord(k, filter_r_, filter_s_, c);
  }

  /// Returns true if the current coordinate is within the activations tensor W
  CUTLASS_HOST_DEVICE
  bool valid() const {

    TensorCoord coord = at();

    auto input_channels = (IsDeconv ? problem_size_.K : problem_size_.C);
    auto output_channels = (IsDeconv ? problem_size_.C : problem_size_.K);

    if (kGroupMode == conv::GroupMode::kNone) {
      return coord.n() < output_channels && coord.c() < input_channels;
    } else if (kGroupMode == conv::GroupMode::kDepthwise) {
      return coord.n() < output_channels && coord.c() < 1; // channels_per_group_ is always equal to ONE.
    } else {
      return coord.n() < output_channels && coord.c() < channels_per_group_ &&
             group_idx_offset_c_ == group_idx_offset_k_[iteration_strided_];
    }
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {

    TensorCoord coord = at();
    LongIndex offset = params_.layout(coord);
    
    return reinterpret_cast<AccessType const *>(pointer_ + offset * sizeof_bits<Element>::value / 8);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterTileAccessIteratorAnalytic &operator++() {
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

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    auto input_channels = (IsDeconv ? problem_size.K : problem_size.C);
    auto output_channels = (IsDeconv ? problem_size.C : problem_size.K);

    // check alignment constraint on iterator's contiguous dimension
    if ((input_channels / problem_size.groups) % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    if (platform::is_same<Layout, layout::TensorCxRSKx<32>>::value) {
      if (output_channels % 32) {
        return Status::kErrorInvalidProblem;
      }
    }

    if (platform::is_same<Layout, layout::TensorCxRSKx<64>>::value) {
      if (output_channels % 64) {
        return Status::kErrorInvalidProblem;
      }
    }

    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
