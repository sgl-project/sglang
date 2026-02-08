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

    This iterator assumes TensorNHWC layout of tensors in Global Memory.

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
  typename ThreadMap_,
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kUnity,
  typename AccessType_ = cutlass::AlignedArray<Element_, ThreadMap_::kElementsPerAccess>
>
class Conv2dDgradOutputGradientTileAccessIteratorOptimized;
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2dDgradOutputGradientTileAccessIteratorOptimized strided dgrad needs special handling 
// to skip MMAs (Dx = Dy * w) on invalid filter positions
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_,
  typename AccessType_
>
class Conv2dDgradOutputGradientTileAccessIteratorOptimized <
  Shape_,
  Element_,
  ThreadMap_,
  conv::StrideSupport::kStrided,
  AccessType_
> {
public:

  //
  // Types
  //
  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNHWC;
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
  
  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements), 
    "Vectors implied by the thread map must be divisible by the access type.");
 
  using Mask = uint64_t;
  
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

  using Params = Conv2dStridedDgradOutputGradientIteratorOptimizedParams;

private:

  Params const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  
  // One pointer per access
  char const *pointer_[ThreadMap::Iterations::kStrided];
  
  int filter_k_;
  int filter_r_;
  int filter_s_;
  int start_r_;
  int start_s_;
  int64_t reset_bytes_s_;
  int64_t reset_bytes_r_;

  Index masks_[ThreadMap::Iterations::kStrided][kAccessesPerVector][2];

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientTileAccessIteratorOptimized(
    Params const &params, 
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    FastDivmod const &stride_h_divmod, FastDivmod const &stride_w_divmod,
    int start_r, int start_s,
    MatrixCoord const &threadblock_offset = MatrixCoord()     // threadblock offset - units are whole CTA tiles
  ):
    params_(params), 
    problem_size_(problem_size), 
    filter_k_(0),
    filter_r_(start_r),
    filter_s_(start_s),
    start_r_(start_r),
    start_s_(start_s) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.column() + thread_coord.contiguous();

    reset_bytes_s_ = (problem_size_.num_gemm_k_filter_s(start_s_) - 1) * params_.inc_next[0];

    reset_bytes_r_ = (problem_size_.num_gemm_k_filter_s(start_s_) - 1) * params_.inc_next[0] +
                      (problem_size_.num_gemm_k_filter_r(start_r_) - 1) * params_.inc_next[1];

    int offset_n[ThreadMap::Iterations::kStrided];
    int offset_p[ThreadMap::Iterations::kStrided];
    int offset_q[ThreadMap::Iterations::kStrided];

    int filter_r = filter_r_;
    int filter_s = filter_s_;

    if (problem_size_.mode == Mode::kConvolution) {
      filter_r = (problem_size_.R - 1 - filter_r);
      filter_s = (problem_size_.S - 1 - filter_s);
    }

    // Starting h, w positions for filter position in gemm_k=0
    int start_h, start_w;
    strided_dgrad_starting_coords(
      problem_size_, 
      stride_h_divmod, stride_w_divmod, 
      filter_r, filter_s, 
      start_h, start_w);


    // Effective starting P and Q for filter position required for remapping NHW rows
    int P = (problem_size_.H - start_h + problem_size_.stride_h - 1) / problem_size_.stride_h;
    int Q = (problem_size_.W - start_w + problem_size_.stride_w - 1) / problem_size_.stride_w;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      pointer_[s] = reinterpret_cast<char const *>(ptr);      

      int offset_npq = (threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided) % params_.tiled_rows_per_filter;

      // (STEP 1) [reorder NHW rows to start with same filter positions]
      offset_n[s] = offset_npq / (P * Q);
      int residual = offset_npq % (P * Q);

      int p = (residual / Q);
      int q = (residual % Q);

      int mapped_h = (start_h + p * problem_size_.stride_h);
      int mapped_w = (start_w + q * problem_size_.stride_w);
      
      // Access (p, q) coordinates for Dy tensor for filter position in gemm_k=0
      // note that (h + pad_h - filter_r) and (w + pad_w - filter_s) are ensured to be 
      // divisible by stride_h and stride_w
      offset_p[s] = (mapped_h + problem_size_.pad_h - filter_r) / problem_size_.stride_h;
      offset_q[s] = (mapped_w + problem_size_.pad_w - filter_s) / problem_size_.stride_w;

      // Initialize pointers for gemm_k=0
      TensorCoord coord{offset_n[s], offset_p[s], offset_q[s], filter_k_};

      pointer_[s] += params_.layout(coord) * sizeof_bits<Element>::value / 8;
    }

    //
    // Precompute mask predicates
    //
    clear_mask();

    CUTLASS_PRAGMA_NO_UNROLL
    for (int r = start_r; r < problem_size_.R; r += problem_size_.stride_h) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {

        int p = offset_p[s_idx] ;

        p += (params_.conv_sign * (r / problem_size_.stride_h));

        bool pred = (offset_n[s_idx] < problem_size_.N && p >= 0 && p < problem_size_.P);

        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          masks_[s_idx][v_idx][0] |= (pred << r);
        }
      }
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for(int s = start_s; s < problem_size_.S; s += problem_size_.stride_w) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {

        int q = offset_q[s_idx];
        q += (params_.conv_sign * (s / problem_size_.stride_w));

        bool pred = (q >=0 && q < problem_size_.Q);

        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          masks_[s_idx][v_idx][1] |= (pred << s);
        }
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      clear_mask(v_idx, (filter_k_ + v_idx * AccessType::kElements) >= problem_size.K);
    }

    set_iteration_index(0);
  }

  CUTLASS_HOST_DEVICE
  static Params getParams(Conv2dProblemSize const &problem_size, Layout const &layout) {
    return Params(problem_size, 
                  layout,
                  sizeof_bits<Element>::value,
                  {Shape::kRow, Shape::kColumn});
  }

private:

  /// Adds a pointer offset in units of element
  CUTLASS_HOST_DEVICE
  void add_byte_offset_(LongIndex byte_offset, LongIndex byte_reset = 0) {

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      pointer_[s] += byte_offset - byte_reset;
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

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    add_byte_offset_(pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void advance() {

    int next_idx = 0;
    int64_t reset_bytes = 0;

    // Move filter_s by stride_w
    filter_s_ +=  problem_size_.stride_w;
    if (filter_s_ >= problem_size_.S) {
      
      // Restore filter_s
      filter_s_ = start_s_;

      // Move filter_r by stride_h
      filter_r_ += problem_size_.stride_h;
#if 0
      if (filter_r_ < problem_size_.R) {

        next_idx = 1;

        // Restore bytes in q coordinate (Mma in filter s dimension)
        reset_bytes = reset_bytes_s_;

      } else {

        // Restore filter_r
        filter_r_ = start_r_;

        next_idx = 2;

        // Restore bytes in p and q coordinate (Mma in filter s and r dimension)
        reset_bytes = reset_bytes_r_;
      }
#else
      asm volatile(
          "{\n\t"
          " .reg .pred %%p;\n\t"
          " setp.lt.s32 %%p, %3, %4;\n\t"
          " selp.s32 %0, %3, %5, %%p;\n\t"
          " selp.s32 %1, 1, 2, %%p;\n\t"
          " selp.s64 %2, %6, %7, %%p;\n\t"
          "}\n"
          : "=r"(filter_r_), "=r"(next_idx), "=l"(reset_bytes)
          : "r"(filter_r_), "r"(problem_size_.R), "r"(start_r_),
            "l"(reset_bytes_s_), "l"(reset_bytes_r_));
#endif
    }

    // offset pointers by offset_bytes
    add_byte_offset_(params_.inc_next[next_idx] - reset_bytes);

    if (next_idx == 2) {  
      filter_k_ += params_.filter_k_delta;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      clear_mask(v_idx, (filter_k_ + v_idx * AccessType::kElements) >= problem_size_.K);
    }
  }

  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask(bool clear = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kAccessesPerVector; ++v) {
        masks_[s][v][0] = clear ? Mask(0) : masks_[s][v][0];
        masks_[s][v][1] = clear ? Mask(0) : masks_[s][v][1];
      }
    }
  }

  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask(int v, bool clear = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      masks_[s][v][0] = clear ? Mask(0) : masks_[s][v][0];
      masks_[s][v][1] = clear ? Mask(0) : masks_[s][v][1];
    }
  }

  /// Returns true if the current coordinate is within the output tensor Dy
  CUTLASS_HOST_DEVICE
  bool valid() const {
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
  Conv2dDgradOutputGradientTileAccessIteratorOptimized &operator++() {
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
    if (problem_size.K % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    // Limit on filter size
    if (problem_size.R > 32 || problem_size.S > 32) {
      return Status::kErrorNotSupported;
    }
    
    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Conv2dDgradOutputGradientTileAccessIteratorOptimized unity stride dgrad is optimized for dgrad
// with problem stride = {1x1}
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_,
  typename AccessType_
>
class Conv2dDgradOutputGradientTileAccessIteratorOptimized <
  Shape_,
  Element_,
  ThreadMap_,
  conv::StrideSupport::kUnity,
  AccessType_
> {
public:
  
  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNHWC;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kUnity;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;
 
  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
  
  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements), 
    "Vectors implied by the thread map must be divisible by the access type.");
 
  using Mask = uint64_t;

  //
  // Simplifying assertions
  //
  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "Require Iterations::kContiguous == 1");

  //
  // Parameters structure
  //

  using Params = Conv2dDgradOutputGradientIteratorOptimizedParams;

private:

  Conv2dDgradOutputGradientIteratorOptimizedParams const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;

  // One pointer per access
  char const *pointer_[ThreadMap::Iterations::kStrided];

  // current filter position (r, s)
  int filter_r_;
  int filter_s_;
  int filter_k_;

  Index masks_[ThreadMap::Iterations::kStrided][kAccessesPerVector][2];

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientTileAccessIteratorOptimized(
    Conv2dDgradOutputGradientIteratorOptimizedParams const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params), 
    problem_size_(problem_size),
    filter_k_(0), 
    filter_r_(0), 
    filter_s_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.column() + thread_coord.contiguous();

    int offset_n[ThreadMap::Iterations::kStrided];
    int offset_h[ThreadMap::Iterations::kStrided];
    int offset_w[ThreadMap::Iterations::kStrided];

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      pointer_[s] = reinterpret_cast<char const *>(ptr);
 
      int offset_nhw = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;

      // The subseqnet fast_divmod() operations are equivalent to the following logical computation:
      //
      //
      //  offset_n[s] = offset_nhw / (problem_size_.H * problem_size_.W);
      //  int residual = offset_nhw % (problem_size_.H * problem_size_.W);
      //
      //  offset_h[s] = residual / problem_size_.W;
      //  offset_w[s] = residual % problem_size_.W;
      //

      int residual;

      params_.hw_divmod(offset_n[s], residual, offset_nhw);
      params_.w_divmod(offset_h[s], offset_w[s], residual);

      TensorCoord coord = at_(offset_n[s], offset_h[s], offset_w[s], 0, 0);

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

        int p = offset_h[s_idx] + problem_size_.pad_h - r_ * problem_size_.dilation_h;

        bool pred = (offset_n[s_idx] < problem_size_.N && p >= 0 && p < problem_size_.P);

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

        int q = offset_w[s_idx] + problem_size_.pad_w - s_ * problem_size_.dilation_w;

        bool pred = (q >= 0 && q < problem_size_.Q);

        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          masks_[s_idx][v_idx][1] |= (pred << s);
        }
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      clear_mask(v_idx, filter_k_ + v_idx * AccessType::kElements >= problem_size.K);
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

  /// Returns the coordinate in the output gradient tensor dy that is correspoinding to 
  // activation nhw and filter position k, r, s
  CUTLASS_HOST_DEVICE
  TensorCoord at_(int n, int h, int w, int r, int s) const {

    if (problem_size_.mode == Mode::kConvolution) {
      r = problem_size_.R - 1 - r;
      s = problem_size_.S - 1 - s;
    }

    int p = h + problem_size_.pad_h - r * problem_size_.dilation_h;
    int q = w + problem_size_.pad_w - s * problem_size_.dilation_w;

    return TensorCoord(n, p, q, filter_k_);
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
      filter_k_ += params_.filter_k_delta;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      clear_mask(v_idx, (filter_k_ + v_idx * AccessType::kElements) >= problem_size_.K);
    }
  }

  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask(bool clear = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kAccessesPerVector; ++v) {
        masks_[s][v][0] = clear ? Mask(0) : masks_[s][v][0];
        masks_[s][v][1] = clear ? Mask(0) : masks_[s][v][1];
      }
    }
  }

  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask(int v, bool clear = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      masks_[s][v][0] = clear ? Mask(0) : masks_[s][v][0];
      masks_[s][v][1] = clear ? Mask(0) : masks_[s][v][1];
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
  Conv2dDgradOutputGradientTileAccessIteratorOptimized &operator++() {
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

    // This is specialized for unit stride
    if (problem_size.stride() != MatrixCoord({1, 1})) {
      return Status::kErrorNotSupported;
    }

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.K % AccessType::kElements) {
      return Status::kErrorNotSupported;
    }

    // Limit on filter size
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
