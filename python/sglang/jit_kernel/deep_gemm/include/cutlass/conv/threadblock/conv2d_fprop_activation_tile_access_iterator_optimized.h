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
  typename AccessType_ = cutlass::AlignedArray<Element_, ThreadMap_::kElementsPerAccess>
>
class Conv2dFpropActivationTileAccessIteratorOptimized {
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
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;
  
  using Mask = uint64_t;

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

  using Params = Conv2dFpropActivationIteratorOptimizedParams<Layout>;

private:

  Params const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;

  // One pointer per access
  char const *pointer_[ThreadMap::Iterations::kStrided];

  // current filter position (r, s)
  int filter_r_;
  int filter_s_;
  int filter_c_;

  Index masks_[ThreadMap::Iterations::kStrided][kAccessesPerVector][2];

public:

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationTileAccessIteratorOptimized(
    Params const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params), 
    problem_size_(problem_size),
    filter_c_(0), 
    filter_r_(0), 
    filter_s_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_c_ = threadblock_offset.column() + thread_coord.contiguous();

    int offset_n[ThreadMap::Iterations::kStrided];
    int offset_p[ThreadMap::Iterations::kStrided];
    int offset_q[ThreadMap::Iterations::kStrided];

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      pointer_[s] = reinterpret_cast<char const *>(ptr);
 
      int offset_npq = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;

      // The subseqnet fast_divmod() operations are equivalent to the following logical computation:
      //
      //
      //  offset_n[s] = offset_npq / (problem_size_.P * problem_size_.Q);
      //  int residual = offset_npq % (problem_size_.P * problem_size_.Q);
      //
      //  offset_p[s] = residual / problem_size_.Q;
      //  offset_q[s] = residual % problem_size_.Q;
      //

      int residual;

      params.pq_divmod(offset_n[s], residual, offset_npq);
      params.q_divmod(offset_p[s], offset_q[s], residual);

      TensorCoord coord = at_(offset_n[s], offset_p[s], offset_q[s], 0, 0);

      pointer_[s] += params_.layout(coord) * sizeof_bits<Element>::value / 8;
    }

    clear_mask();

    CUTLASS_PRAGMA_NO_UNROLL
    for (int r = 0; r < problem_size_.R; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {

        int r_ = r;
        if (problem_size_.mode == Mode::kConvolution) {
          r_ = problem_size_.R - 1 - r;
        }

        int h = offset_p[s_idx] * problem_size_.stride_h - problem_size_.pad_h + r_ * problem_size_.dilation_h;

        bool pred = (offset_n[s_idx] < problem_size_.N && h >= 0 && h < problem_size_.H);

        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          masks_[s_idx][v_idx][0] |= (pred << r);
        }
      }
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for (int s = 0; s < problem_size_.S; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {

        int s_ = s;
        if (problem_size_.mode == Mode::kConvolution) {
          s_ = problem_size_.S - 1 - s;
        }

        int w = offset_q[s_idx] * problem_size_.stride_w - problem_size_.pad_w + s_ * problem_size_.dilation_w;

        bool pred = (w >= 0 && w < problem_size_.W);

        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          masks_[s_idx][v_idx][1] |= (pred << s);
        }
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      clear_mask(v_idx, filter_c_ + v_idx * AccessType::kElements >= problem_size_.C);
    }

    set_iteration_index(0);
  }

  CUTLASS_HOST_DEVICE
  static Params getParams(Conv2dProblemSize const &problem_size, Layout const &layout) {
    return Params(problem_size,
                  layout,
                  sizeof_bits<Element>::value,
                  {Shape::kRow, Shape::kColumn},
                  ThreadMap::kThreads,
                  ThreadMap::kElementsPerAccess,
                  {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
                  {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided});
  }

private:

  /// Returns the coordinate in the activations tensor X that is correspoinding to 
  // output npq and filter position r, s
  CUTLASS_HOST_DEVICE
  TensorCoord at_(int n, int p, int q, int r, int s) const {

    if (problem_size_.mode == Mode::kConvolution) {
      r = problem_size_.R - 1 - r;
      s = problem_size_.S - 1 - s;
    }

    int h = p * problem_size_.stride_h - problem_size_.pad_h + r * problem_size_.dilation_h;
    int w = q * problem_size_.stride_w - problem_size_.pad_w + s * problem_size_.dilation_w;

    return TensorCoord(n, h, w, filter_c_);
  }
  
  /// Adds a pointer offset in units of element
  CUTLASS_HOST_DEVICE
  void add_byte_offset_(LongIndex byte_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      pointer_[s] += byte_offset;
    }
  }

public:

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;

    iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    add_byte_offset_(pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_HOST_DEVICE
  void advance() { 

    int next_idx = 0;
 
    // moves to the next tile
    ++filter_s_;
    if (filter_s_ == problem_size_.S) {
      filter_s_ = 0;
      ++filter_r_;
 
      if (filter_r_ < problem_size_.R) {
        next_idx = 1;
      }
      else {
        filter_r_ = 0;
        next_idx = 2;
      }
    }
    
    add_byte_offset_(params_.inc_next[next_idx]);
      
    if (next_idx == 2) {  
      filter_c_ += params_.filter_c_delta;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      clear_mask(v_idx, filter_c_ + v_idx * AccessType::kElements >= problem_size_.C);
    }
  }
   
  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask(bool clear = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kAccessesPerVector; ++v) {
        masks_[s][v][0] = clear ? 0 : masks_[s][v][0];
        masks_[s][v][1] = clear ? 0 : masks_[s][v][1];
      }
    }
  } 
   
  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask(int v, bool clear = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      masks_[s][v][0] = clear ? 0 : masks_[s][v][0];
      masks_[s][v][1] = clear ? 0 : masks_[s][v][1];
    }
  }

  CUTLASS_HOST_DEVICE
  bool valid() {

    return 
      (masks_[iteration_strided_][iteration_vector_][0] & (Index(1) << filter_r_)) &&
      (masks_[iteration_strided_][iteration_vector_][1] & (Index(1) << filter_s_));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {

    return reinterpret_cast<AccessType const *>(pointer_[iteration_strided_]) + iteration_vector_;
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationTileAccessIteratorOptimized &operator++() {

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

    // Conv2dFpropActivationTileAccessIteratorOptimized has constraint on filter positions 
    // due to the number of mask bits.
    if (problem_size.R > 32 || problem_size.S > 32) {
      return Status::kErrorNotSupported;
    }
    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
