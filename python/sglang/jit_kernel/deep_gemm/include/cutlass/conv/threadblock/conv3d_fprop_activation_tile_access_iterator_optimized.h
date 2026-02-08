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

    This iterator assumes TensorNDHWC layout of tensors in Global Memory.
    
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
  typename Layout_,
  typename ThreadMap_
>
class Conv3dFpropActivationTileAccessIteratorOptimized {
public:

  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = ThreadMap_;
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 3;
  using ConvProblemSize = typename conv::Conv3dProblemSize;
  static int const kAccessesPerVector = 1;  
  using Mask = uint64_t;

  //
  // Simplifying assertions
  //
  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "Require Iterations::kContiguous == 1");

  //
  // Parameters structure
  //

  using Params = Conv3dFpropActivationIteratorOptimizedParams<Layout>;

private:

  Conv3dFpropActivationIteratorOptimizedParams<Layout> const &params_;
  Conv3dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;

  // One pointer per access
  char const *pointer_[ThreadMap::Iterations::kStrided];

  // current filter position (t, r, s)
  int filter_t_;
  int filter_r_;
  int filter_s_;
  int filter_c_;

  // mask for t, r, and s
  Index masks_[ThreadMap::Iterations::kStrided][3];

public:

  CUTLASS_HOST_DEVICE
  Conv3dFpropActivationTileAccessIteratorOptimized(
    Conv3dFpropActivationIteratorOptimizedParams<Layout> const &params,
    Conv3dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ) :    
  params_(params), 
  problem_size_(problem_size),
  filter_t_(0), 
  filter_r_(0), 
  filter_s_(0),
  filter_c_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_c_ = threadblock_offset.column() + thread_coord.contiguous();

    int offset_n[ThreadMap::Iterations::kStrided];
    int offset_z[ThreadMap::Iterations::kStrided];
    int offset_p[ThreadMap::Iterations::kStrided];
    int offset_q[ThreadMap::Iterations::kStrided];

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      pointer_[s] = reinterpret_cast<char const *>(ptr);
 
      int offset_nzpq = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;

      // The subseqnet fast_divmod() operations are equivalent to the following logical computation:
      //
      //
      //  offset_n[s] = offset_nzpq / (problem_size_.Z * problem_size_.P * problem_size_.Q);
      //  int residual = offset_nzpq % (problem_size_.Z * problem_size_.P * problem_size_.Q);
      //
      //  offset_z[s] = residual / (problem_size_.P * problem_size_.Q);
      //  residual = residual % (problem_size_.P * problem_size_.Q);
      //
      //  offset_p[s] = residual / problem_size_.Q;
      //  offset_q[s] = residual % problem_size_.Q;
      //

      int residual;

      // input: (nzpq offset) output: (n offset and resudial (zpq offset))
      params.zpq_divmod(offset_n[s], residual, offset_nzpq);
      // input: (zpq offset) output: (z offset and resudial (pq))
      params.pq_divmod(offset_z[s], residual, residual);
      // input: (pq offset) output: (p offset and resudial (q offset))
      params.q_divmod(offset_p[s], offset_q[s], residual);

      TensorCoord coord = at_(offset_n[s], offset_z[s], offset_p[s], offset_q[s], 0, 0, 0);

      pointer_[s] += params_.layout(coord) * sizeof_bits<Element>::value / 8;
    }

    clear_mask();

    // mask predicates for filter position T
    CUTLASS_PRAGMA_NO_UNROLL
    for (int t = 0; t < problem_size_.T; ++t) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {

        int t_ = t;
        if (problem_size_.mode == Mode::kConvolution) {
          t_ = problem_size_.T - 1 - t;
        }

        int d = offset_z[s_idx] * problem_size_.stride_d - problem_size_.pad_d + t_ * problem_size_.dilation_d;

        bool pred = (offset_n[s_idx] < problem_size_.N && d >= 0 && d < problem_size_.D);
        masks_[s_idx][0] |= (pred << t);
      }
    }   

    // mask predicates for filter position R
    CUTLASS_PRAGMA_NO_UNROLL
    for (int r = 0; r < problem_size_.R; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {

        int r_ = r;
        if (problem_size_.mode == Mode::kConvolution) {
          r_ = problem_size_.R - 1 - r;
        }

        int h = offset_p[s_idx] * problem_size_.stride_h - problem_size_.pad_h + r_ * problem_size_.dilation_h;

        bool pred = (h >= 0 && h < problem_size_.H);
        masks_[s_idx][1] |= (pred << r);
      }
    }  

    // mask predicates for filter position S
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
        masks_[s_idx][2] |= (pred << s);
      }
    }

    if (filter_c_ >= problem_size.C) {
      clear_mask();
    }

    set_iteration_index(0);
  }

  CUTLASS_HOST_DEVICE
  static Params getParams(Conv3dProblemSize const &problem_size, Layout const &layout) {
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
  // output nzpq and filter position t, r, s
  CUTLASS_HOST_DEVICE
  TensorCoord at_(int n, int z, int p, int q, int t, int r, int s) const {

    if (problem_size_.mode == Mode::kConvolution) {
      t = problem_size_.T - 1 - t;
      r = problem_size_.R - 1 - r;
      s = problem_size_.S - 1 - s;
    }

    int d = z * problem_size_.stride_d - problem_size_.pad_d + t * problem_size_.dilation_d;
    int h = p * problem_size_.stride_h - problem_size_.pad_h + r * problem_size_.dilation_h;
    int w = q * problem_size_.stride_w - problem_size_.pad_w + s * problem_size_.dilation_w;

    return TensorCoord(n, d, h, w, filter_c_);
  }

  /// Adds a pointer offset in units of element
  CUTLASS_HOST_DEVICE
  void add_byte_offset_(LongIndex byte_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      pointer_[s] += byte_offset;
    }
  }


  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask_(bool clear) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      // We are using inline PTX assembly here to avoid an CUDA C++ compilation
      // artifact in which control flow instructions are generated. Instead, our
      // intent is to predicate the mov instructions.
      #if defined(__CUDA_ARCH__)
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  .reg .u32  m;"
          "  mov.u32 m, %2;"
          "  setp.ne.b32 p, %1, 0;\n"
          "  @p mov.u32 m, 0;\n"
          "  mov.u32 %0, m;\n"
          "}\n" 
        :
          "=r"(masks_[s][0])
       : 
          "r"((int)clear),
          "r"(masks_[s][0])
      );
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  .reg .u32  m;"
          "  mov.u32 m, %2;"
          "  setp.ne.b32 p, %1, 0;\n"
          "  @p mov.u32 m, 0;\n"
          "  mov.u32 %0, m;\n"
          "}\n" 
        :
          "=r"(masks_[s][1])
       : 
          "r"((int)clear),
          "r"(masks_[s][1])
      );
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  .reg .u32  m;"
          "  mov.u32 m, %2;"
          "  setp.ne.b32 p, %1, 0;\n"
          "  @p mov.u32 m, 0;\n"
          "  mov.u32 %0, m;\n"
          "}\n" 
        :
          "=r"(masks_[s][2])
       : 
          "r"((int)clear),
          "r"(masks_[s][2])
      );
      #else
        if (clear) {
          masks_[s][0] = 0;
          masks_[s][1] = 0;
          masks_[s][2] = 0;
        }
      #endif
    }
  }

public:

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
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
      next_idx = 1;

      if (filter_r_ == problem_size_.R) {
        filter_r_ = 0;
        ++filter_t_;

        if (filter_t_ < problem_size_.T) {
          next_idx = 2;
        } 
        else {
          filter_t_ = 0;
          next_idx = 3;
        } 
      }
    }

    add_byte_offset_(params_.inc_next[next_idx]);
      
    if (next_idx == 3) {  
      filter_c_ += params_.filter_c_delta;
    }

    clear_mask_(filter_c_ >= problem_size_.C);
  }

  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      masks_[s][0] = Mask(0);
      masks_[s][1] = Mask(0);
      masks_[s][2] = Mask(0);
    }
  }

  CUTLASS_HOST_DEVICE
  bool valid() {

    return 
      (masks_[iteration_strided_][0] & (Index(1) << filter_t_)) &&
      (masks_[iteration_strided_][1] & (Index(1) << filter_r_)) &&
      (masks_[iteration_strided_][2] & (Index(1) << filter_s_));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {

    return reinterpret_cast<AccessType const *>(pointer_[iteration_strided_]);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv3dFpropActivationTileAccessIteratorOptimized &operator++() {

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
    if (problem_size.C % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    // Conv3dFpropActivationTileAccessIteratorOptimized has constraint on filter positions 
    // due to the number of mask bits.
    if (problem_size.T > 32 || problem_size.R > 32 || problem_size.S > 32) {
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
