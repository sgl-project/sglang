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

    This iterator assumes TensorNHWC layout of tensors in Global Memory.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/threadblock/depthwise_direct_conv_params.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Shape_,
          typename OutputTileShape_,
          typename StrideShape_,
          typename DilationShape_,
          typename ActivationShape_,
          typename Element_,
          typename Layout_,
          typename ThreadMap_,
          typename AccessType_ = cutlass::AlignedArray<Element_, ThreadMap_::kElementsPerAccess> >
class DepthwiseFpropActivationDirect2dConvTileAccessIteratorFixedStrideDilation {
 public:
  //
  // Types
  //

  using Shape = Shape_;
  using OutputTileShape = OutputTileShape_;
  using Element = Element_;
  using Layout = Layout_;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;

  // Compilation value of stride , dialtion and activation shape
  using StrideShape = StrideShape_;
  using DilationShape = DilationShape_;
  using ActivationShape = ActivationShape_;


  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
  static int const kActivationSize = ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess * ThreadMap::kThreads *
           sizeof_bits<Element>::value / 8;


  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
                "Vectors implied by the thread map must be divisible by the access type.");

  //
  // Simplifying assertions
  //
  static_assert(ThreadMap::Iterations::kContiguous == 1, "Require Iterations::kContiguous == 1");
  
  static_assert(OutputTileShape::kN == 1, "Require OutputTileShape::kN == 1");
  static_assert(OutputTileShape::kC == Shape::kColumn, "Require OutputTile shape == channels per threadblock");

  //
  // Parameters structure
  //

  using Params = Depthwise2dFpropDirectConvActivationIteratorFixedStrideDilationParams<Layout>;

 private:
  Conv2dProblemSize const &problem_size_;
  Params const &params_;
  char const *pointer_;

  // Base channels for current threadblock
  int base_c_;
  // Base activation index for current threadblock
  int offset_intial_npq_;
  // Base activation coord for current threadblock
  TensorCoord activatioin_base_;
  // Intial thread positioin
  int offset_initial_hwc_;
  // Overall load instruction per thread.
  int iterator_load_;
  // thread loading position.
  int iterator_hwc_;
  // activation N is inside the Tensor or not
  bool valid_n_;

 public:


  CUTLASS_HOST_DEVICE
  DepthwiseFpropActivationDirect2dConvTileAccessIteratorFixedStrideDilation(
      Params const &params,
      Conv2dProblemSize const &problem_size,
      Element const *ptr,
      int thread_idx,
      MatrixCoord const &threadblock_offset =
          MatrixCoord()
      )
      : params_(params),
        problem_size_(problem_size),
        pointer_(reinterpret_cast<char const *>(ptr)),
        offset_intial_npq_(threadblock_offset.row()),
        offset_initial_hwc_(thread_idx),
        iterator_load_(0) {
    
    base_c_ = threadblock_offset.column();

    set_iteration_index(0);

    set_activation_coord(offset_intial_npq_);

  }

  CUTLASS_HOST_DEVICE
  void set_activation_coord(int offset_npq) {
    int offset_inital_n, offset_inital_p, offset_inital_q;
    int residual;

    params_.pq_divmod(offset_inital_n, residual, offset_npq);
    params_.q_divmod(offset_inital_p, offset_inital_q, residual);

    int base_n = offset_inital_n;

    int base_h =
        offset_inital_p * OutputTileShape::kH * StrideShape::kRow - problem_size_.pad_h;

    int base_w =
        offset_inital_q * OutputTileShape::kW * StrideShape::kColumn - problem_size_.pad_w;

    activatioin_base_ = TensorCoord(base_n, base_h, base_w, base_c_);

    valid_n_ = activatioin_base_.n() < problem_size_.N;
  }

  CUTLASS_HOST_DEVICE
  static Params getParams(Conv2dProblemSize const &problem_size, Layout const &layout) {
    return Params(
        problem_size,
        layout,
        {Shape::kRow, Shape::kColumn},
        {OutputTileShape::kN, OutputTileShape::kH, OutputTileShape::kW, OutputTileShape::kC},
        kActivationSize);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iterator_hwc_ = offset_initial_hwc_ + index * ThreadMap::kThreads;
    iterator_load_ = index;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_HOST_DEVICE
  void advance() {
    // Go to next threadblock
    offset_intial_npq_ += problem_size_.split_k_slices;

    set_iteration_index(0);

    set_activation_coord(offset_intial_npq_);
  }

  /// Returns the coordinate in the activations tensor X that is currently pointed to
  /// by the iterator.
  CUTLASS_HOST_DEVICE
  TensorCoord at() const {
    int c = iterator_hwc_ %  ThreadMap::Detail::ShapeVec::kContiguous ;
    int next = iterator_hwc_ /  ThreadMap::Detail::ShapeVec::kContiguous ;
    int h = next / ActivationShape::kW;
    int w = next % ActivationShape::kW;

    c = c * AccessType::kElements;

    return activatioin_base_ + TensorCoord(0, h, w, c);
  }

  /// Returns true if the current coordinate is within the activations tensor X
  CUTLASS_HOST_DEVICE
  bool valid() const {
    TensorCoord coord = at();
    bool valid_c = coord.c() < problem_size_.C;
    bool valid_h = coord.h() >= 0 && coord.h() < problem_size_.H;
    bool valid_w = coord.w() >= 0 && coord.w() < problem_size_.W;
    return valid_n_ ? valid_c & valid_h & valid_w : 0;
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    TensorCoord coord = at();
    LongIndex offset = params_.layout(coord);

    AccessType const *ptr =
        reinterpret_cast<AccessType const *>(pointer_ + offset * sizeof_bits<Element>::value / 8);

    return ptr;
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  DepthwiseFpropActivationDirect2dConvTileAccessIteratorFixedStrideDilation &operator++() {

    ++iterator_load_;
    iterator_hwc_ += ThreadMap::kThreads;

    if (iterator_load_ < ThreadMap::Iterations::kCount) {
       return *this;
    }
    
    iterator_load_ = 0;
    iterator_hwc_ = offset_initial_hwc_;

    return *this;
  }

  /// Determines the activation size loaded by iterator
  CUTLASS_HOST_DEVICE
  int get_load_size() {
    return kActivationSize;
  }

  /// Determines the iterations needed
  CUTLASS_HOST_DEVICE
  int get_iteration_num() {
    return ThreadMap::Iterations::kCount;
  }

  /// Determines whether the Depthwise fprop can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check stride and dilation constraint
    if (problem_size.stride_h != StrideShape::kRow || problem_size.stride_w != StrideShape::kColumn) {
      return Status::kErrorInvalidProblem;
    }

    if (problem_size.dilation_h != DilationShape::kRow || problem_size.dilation_w != DilationShape::kColumn) {
      return Status::kErrorInvalidProblem;
    }

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.C % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
