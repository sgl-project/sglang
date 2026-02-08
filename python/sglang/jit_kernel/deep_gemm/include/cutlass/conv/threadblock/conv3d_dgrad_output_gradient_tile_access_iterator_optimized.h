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
  typename ThreadMap_,
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kUnity
>
class Conv3dDgradOutputGradientTileAccessIteratorOptimized {
public:

  static_assert(StrideSupport_ == conv::StrideSupport::kUnity,
    "Only unit-stride dgrad is supported at this time.");

  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNDHWC;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = ThreadMap_;
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kUnity;
  static int const kConvDim = 3;
  using ConvProblemSize = typename conv::Conv3dProblemSize;
  using Coord3D = Coord<3>;
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

  using Params = Conv3dDgradOutputGradientIteratorOptimizedParams;

private:

  Params const &params_;
  ConvProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;


  // One pointer per access
  char const *pointer_[ThreadMap::Iterations::kStrided];

  // current filter position (t, r, s)
  int filter_t_;
  int filter_r_;
  int filter_s_;
  int filter_k_;

  Index masks_[ThreadMap::Iterations::kStrided][3];

public:

  CUTLASS_HOST_DEVICE
  Conv3dDgradOutputGradientTileAccessIteratorOptimized(
    Params const &params,
    ConvProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params), 
    problem_size_(problem_size),
    filter_k_(0), 
    filter_t_(0),
    filter_r_(0), 
    filter_s_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.column() + thread_coord.contiguous();

    int offset_n[ThreadMap::Iterations::kStrided];
    int offset_d[ThreadMap::Iterations::kStrided];
    int offset_h[ThreadMap::Iterations::kStrided];
    int offset_w[ThreadMap::Iterations::kStrided];

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      pointer_[s] = reinterpret_cast<char const *>(ptr);
 
      int offset_ndhw = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;

      // The subseqnet fast_divmod() operations are equivalent to the following logical computation:
      //
      //
      //  offset_n[s] = offset_ndhw / (problem_size_.D * problem_size_.H * problem_size_.W);
      //  int residual = offset_ndhw % (problem_size_.D * problem_size_.H * problem_size_.W);
      //
      //
      //  offset_d[s] = residual / (problem_size_.H * problem_size_.W);
      //  residual    = residual % (problem_size_.H * problem_size_.W);
      //
      //  offset_h[s] = residual / problem_size_.W;
      //  offset_w[s] = residual % problem_size_.W;
      //

      int residual;

      // input: (ndhw offset) output: (n offset and resudial (dhw offset))
      params_.dhw_divmod(offset_n[s], residual, offset_ndhw);
      // input: (dhw offset) output: (d offset and resudial (hw))
      params_.hw_divmod(offset_d[s], residual, residual);
      // input: (hw offset) output: (h offset and resudial (w offset))
      params_.w_divmod(offset_h[s], offset_w[s], residual);

      TensorCoord coord = at_(offset_n[s], offset_d[s], offset_h[s], offset_w[s], 0, 0, 0);

      pointer_[s] += params_.layout(coord) * sizeof_bits<Element>::value / 8;
    }

    clear_mask();

    CUTLASS_PRAGMA_NO_UNROLL
    for (int t = 0; t < problem_size_.T; ++t) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {

        int t_ = t;
        if (problem_size_.mode == Mode::kConvolution) {
          t_ = problem_size_.T - 1 - t;
        }

        int z = offset_d[s_idx] + problem_size_.pad_d - t_ * problem_size_.dilation_d;

        bool pred = (offset_n[s_idx] < problem_size_.N && z >= 0 && z < problem_size_.Z);
        masks_[s_idx][0] |= (pred << t);
      }
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for (int r = 0; r < problem_size_.R; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {

        int r_ = r;
        if (problem_size_.mode == Mode::kConvolution) {
          r_ = problem_size_.R - 1 - r;
        }

        int p = offset_h[s_idx] + problem_size_.pad_h - r_ * problem_size_.dilation_h;

        bool pred = (p >= 0 && p < problem_size_.P);
        masks_[s_idx][1] |= (pred << r);
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

        int q = offset_w[s_idx] + problem_size_.pad_w - s_ * problem_size_.dilation_w;

        bool pred = (q >= 0 && q < problem_size_.Q);
        masks_[s_idx][2] |= (pred << s);
      }
    }

    if (filter_k_ >= problem_size.K) {
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


  /// Returns the coordinate in the output gradient tensor dy that is correspoinding to 
  // activation ndhw and filter position k, t, r, s
  CUTLASS_HOST_DEVICE
  TensorCoord at_(int n, int d, int h, int w, int t, int r, int s) const {

    if (problem_size_.mode == Mode::kConvolution) {
      t = problem_size_.T - 1 - t;
      r = problem_size_.R - 1 - r;
      s = problem_size_.S - 1 - s;
    }

    int z = d + problem_size_.pad_d - t * problem_size_.dilation_d;
    int p = h + problem_size_.pad_h - r * problem_size_.dilation_h;
    int q = w + problem_size_.pad_w - s * problem_size_.dilation_w;

    return TensorCoord(n, z, p, q, filter_k_);
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
      filter_k_ += params_.filter_k_delta;
    }

    clear_mask_(filter_k_ >= problem_size_.K);
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
  Conv3dDgradOutputGradientTileAccessIteratorOptimized &operator++() {

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

    // This is specialized for unit stride
    if (problem_size.stride() != Coord3D({1, 1, 1})) {
      return Status::kErrorNotSupported;
    }

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.K % AccessType::kElements) {
      return Status::kErrorNotSupported;
    }

    // Limit on filter size
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


