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

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_,
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kStrided
>
class Conv3dDgradOutputGradientTileAccessIteratorAnalytic;
/////////////////////////////////////////////////////////////////////////////////////////////////

// Conv3dDgradOutputGradientTileAccessIteratorAnalytic strided dgrad needs special handling using
// unscaled coordinations
template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_
>
class Conv3dDgradOutputGradientTileAccessIteratorAnalytic <
  Shape_,
  Element_,
  ThreadMap_,
  conv::StrideSupport::kStrided
> {
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
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kAnalytic;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 3;
  using ConvProblemSize = typename conv::Conv3dProblemSize;
  static int const kAccessesPerVector = 1;
  
  static_assert(sizeof_bits<Element>::value >= 8,
    "DGRAD requires elements of size 8b or greater.");
 
  //
  // Simpligying assertions
  //

  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "Require Iterations::kContiguous == 1");

  //
  // Parameters structure
  //

  struct Params {

    Layout layout;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      ConvProblemSize const &problem_size, 
      Layout const &layout
    ): layout(layout) {

    }
  };

private:

  Params const &params_;
  ConvProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  char const *pointer_;

  int filter_k_;
  int filter_t_;
  int filter_r_;
  int filter_s_;

  int offset_n_[ThreadMap::Iterations::kStrided];
  int offset_d_[ThreadMap::Iterations::kStrided];
  int offset_w_[ThreadMap::Iterations::kStrided];
  int offset_h_[ThreadMap::Iterations::kStrided];
  
private:

  /// Returns the coordinate in the output tensor Dy that is currently pointed to
  /// by the iterator but DOES NOT scale by the convolution stride. This is needed
  /// to compute predicates in the valid() method. The return value of the public at()
  /// method is correctly scaled.
  CUTLASS_HOST_DEVICE
  TensorCoord unscaled_at_() const {
    int n = offset_n_[iteration_strided_];
    int d = offset_d_[iteration_strided_];
    int h = offset_h_[iteration_strided_];
    int w = offset_w_[iteration_strided_];

    int t = filter_t_;
    int r = filter_r_;
    int s = filter_s_;

    if (problem_size_.mode == Mode::kConvolution) {
      t = (problem_size_.T - 1 - t);
      r = (problem_size_.R - 1 - r);
      s = (problem_size_.S - 1 - s);
    }

    int z = (d + problem_size_.pad_d - t * problem_size_.dilation_d);
    int p = (h + problem_size_.pad_h - r * problem_size_.dilation_h);
    int q = (w + problem_size_.pad_w - s * problem_size_.dilation_w);

    return TensorCoord(n, z, p, q, filter_k_);
  }

public:

  CUTLASS_HOST_DEVICE
  Conv3dDgradOutputGradientTileAccessIteratorAnalytic(
    Params const &params, 
    ConvProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()     // threadblock offset - units are whole CTA tiles
  ):
    params_(params), 
    problem_size_(problem_size), 
    pointer_(reinterpret_cast<char const *>(ptr)), 
    filter_k_(0), 
    filter_t_(0),
    filter_r_(0), 
    filter_s_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.column() + thread_coord.contiguous();

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      int offset_ndhw = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;

      offset_n_[s] = offset_ndhw / (problem_size_.D * problem_size_.H * problem_size_.W);
      int residual = offset_ndhw % (problem_size_.D * problem_size_.H * problem_size_.W);

      offset_d_[s] = residual / (problem_size_.H * problem_size_.W);
      residual     = residual % (problem_size_.H * problem_size_.W);

      offset_h_[s] = residual / problem_size_.W;
      offset_w_[s] = residual % problem_size_.W;
    }
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
    // move to the next tile
    ++filter_s_;
    if (filter_s_ < problem_size_.S) {
      return;
    }
    filter_s_  = 0;
    ++filter_r_;
    if (filter_r_ < problem_size_.R) {
      return;
    }
    filter_r_ = 0;
    ++filter_t_;
    if (filter_t_ < problem_size_.T) {
      return;
    }
    filter_t_ = 0;

    filter_k_ += Shape_::kColumn * problem_size_.split_k_slices;
  }

  /// Returns the coordinate in the output tensor Dy that is currently pointed to
  /// by the iterator.
  CUTLASS_HOST_DEVICE
  TensorCoord at() const {

    TensorCoord coord = unscaled_at_();

    return TensorCoord(
      coord.n(), 
      coord.d() / problem_size_.stride_d, 
      coord.h() / problem_size_.stride_h, 
      coord.w() / problem_size_.stride_w, 
      coord.c());
  }


  /// Returns true if the current coordinate is within the output tensor Dy
  CUTLASS_HOST_DEVICE
  bool valid() const {

    TensorCoord unscaled_coord = unscaled_at_();
    TensorCoord coord = at();

    return 
      !(unscaled_coord.d() % problem_size_.stride_d) &&
      !(unscaled_coord.h() % problem_size_.stride_h) && 
      !(unscaled_coord.w() % problem_size_.stride_w) &&
      coord.n() < problem_size_.N &&
      coord.d() >= 0 && coord.d() < problem_size_.Z &&
      coord.h() >= 0 && coord.h() < problem_size_.P &&
      coord.w() >= 0 && coord.w() < problem_size_.Q &&
      coord.c() < problem_size_.K;
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
  Conv3dDgradOutputGradientTileAccessIteratorAnalytic &operator++() {
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
  static Status can_implement(ConvProblemSize const &problem_size) {

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
