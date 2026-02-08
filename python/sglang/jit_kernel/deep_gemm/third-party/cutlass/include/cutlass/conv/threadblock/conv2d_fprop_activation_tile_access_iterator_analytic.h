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
    \brief Templates implementing loading of convolution tiles mapped to GEMM A (activation tile) 
    matrix from memory.

    This iterator assumes TensorNHWC or TensorNCxHWx<Interleave> layout of tensors in Global Memory.

    The iterator is specialized for each of the three convolution operators: forward propagation (Fprop),
    backward data gradient (Dgrad), and backward weight gradient (Wgrad). 
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/matrix_shape.h"
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
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone
>
class Conv2dFpropActivationTileAccessIteratorAnalytic {
public:
  
  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
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

  int filter_c_;
  int filter_r_;
  int filter_s_;
  int filter_c_init_;
  int group_idx_offset_;
  int channels_per_group_;
  int crs_cnt_;
  int crs_per_group_;

  int offset_n_[ThreadMap::Iterations::kStrided];
  int offset_p_[ThreadMap::Iterations::kStrided];
  int offset_q_[ThreadMap::Iterations::kStrided];

public:

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationTileAccessIteratorAnalytic(
    Params const &params, 
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params), 
    problem_size_(problem_size), 
    pointer_(reinterpret_cast<char const *>(ptr)), 
    crs_cnt_(0),
    group_idx_offset_(0),
    filter_c_(0), 
    filter_r_(0), 
    filter_s_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_c_ = threadblock_offset.column() + thread_coord.contiguous();

    if (kGroupMode != conv::GroupMode::kNone) {
      filter_c_init_ = filter_c_;
      channels_per_group_ = problem_size_.C / problem_size_.groups;
      crs_per_group_ = problem_size_.S * problem_size_.R * ((channels_per_group_ + Shape::kColumn - 1) / Shape::kColumn);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      int offset_npq = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;
    
      offset_n_[s] = offset_npq / (problem_size_.P * problem_size_.Q);
      int residual = offset_npq % (problem_size_.P * problem_size_.Q);

      offset_p_[s] = residual / problem_size_.Q;
      offset_q_[s] = residual % problem_size_.Q;
    }

    set_iteration_index(0);
  }

  CUTLASS_HOST_DEVICE
  static Params getParams(Conv2dProblemSize const &problem_size, Layout const &layout) {
    return Params(problem_size, layout);
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
    pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
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
      filter_c_ += Shape::kColumn * problem_size_.split_k_slices;
    } else {
      if (crs_cnt_ == crs_per_group_) {
        // moves to next group
        crs_cnt_ = 0;
        ++group_idx_offset_;
        filter_c_ = group_idx_offset_ * channels_per_group_ + filter_c_init_;
      } else {
        filter_c_ += Shape::kColumn * problem_size_.split_k_slices;
      }
    }
  }

  /// Returns the coordinate in the activations tensor X that is currently pointed to
  /// by the iterator.
  CUTLASS_HOST_DEVICE
  TensorCoord at() const {
    int n = offset_n_[iteration_strided_];
    int p = offset_p_[iteration_strided_];
    int q = offset_q_[iteration_strided_];

    int r = filter_r_;
    int s = filter_s_;

    if (problem_size_.mode == Mode::kConvolution) {
      r = (problem_size_.R - 1 - filter_r_);
      s = (problem_size_.S - 1 - filter_s_);
    }

    int h = p * problem_size_.stride_h - problem_size_.pad_h + r * problem_size_.dilation_h;
    int w = q * problem_size_.stride_w - problem_size_.pad_w + s * problem_size_.dilation_w;

    int c = filter_c_ + iteration_vector_ * AccessType::kElements; 

    return TensorCoord(n, h, w, c);
  }

  /// Returns true if the current coordinate is within the activations tensor X
  CUTLASS_HOST_DEVICE
  bool valid() const {

    TensorCoord coord = at();

    return coord.n() < problem_size_.N &&
      coord.h() >= 0 && coord.h() < problem_size_.H &&
      coord.w() >= 0 && coord.w() < problem_size_.W &&
      coord.c() < problem_size_.C;
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {

    TensorCoord coord = at();
    LongIndex offset = params_.layout(coord);
    
    AccessType const *ptr = reinterpret_cast<AccessType const *>(pointer_ + offset * sizeof_bits<Element>::value / 8);

    return ptr;
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationTileAccessIteratorAnalytic &operator++() {
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

    // check alignment constraint on iterator's contiguous dimension
    if ((problem_size.C / problem_size.groups) % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    if (platform::is_same<Layout, layout::TensorNCxHWx<32>>::value) {
      if (problem_size.C % 32) {
        return Status::kErrorInvalidProblem;
      }
    }

    if (platform::is_same<Layout, layout::TensorNCxHWx<64>>::value) {
      if (problem_size.C % 64) {
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
