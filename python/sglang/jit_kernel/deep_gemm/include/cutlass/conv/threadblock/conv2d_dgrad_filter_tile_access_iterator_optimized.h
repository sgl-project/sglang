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
  typename ThreadMap_,
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kUnity,
  typename AccessType_ = cutlass::AlignedArray<Element_, ThreadMap_::kElementsPerAccess>
>
class Conv2dDgradFilterTileAccessIteratorOptimized;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Conv2dDgradFilterTileAccessIteratorOptimized unity strided dgrad is more performant for  dgrad
// on problem sizes with stride = {1x1}
template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_,
  typename AccessType_
>
class Conv2dDgradFilterTileAccessIteratorOptimized <
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
 
  //
  // Parameters structure
  //

  struct Params : Conv2dStridedDgradFilterIteratorOptimizedParams {

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Conv2dStridedDgradFilterIteratorOptimizedParams const &base): 
      Conv2dStridedDgradFilterIteratorOptimizedParams(base) { }
      
    CUTLASS_HOST_DEVICE
    Params(
      Conv2dProblemSize const &problem_size, 
      Layout const &layout
    ):
      Conv2dStridedDgradFilterIteratorOptimizedParams(
        problem_size,
        layout,
        sizeof_bits<Element>::value,
        {Shape::kRow, Shape::kColumn},
        ThreadMap::kThreads,
        ThreadMap::kElementsPerAccess,
        {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
        {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided}
      ) { }

  };

private:

  Conv2dStridedDgradFilterIteratorOptimizedParams const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  char const *pointer_;

  uint32_t predicates_[kAccessesPerVector];
  int filter_k_;
  int filter_r_;
  int filter_s_;

  int start_r_;
  int start_s_;

  int64_t reset_bytes_s_;
  int64_t reset_bytes_r_;

  //
  // Assertions
  //

  // We map predicates into bits packed in this uint32_t container
  static_assert(ThreadMap::Iterations::kStrided *
    ThreadMap::Iterations::kContiguous < sizeof(predicates_) * 8,
    "Currently, the number of loads per iteration is limited by the size of the predicates container.");

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterTileAccessIteratorOptimized(
    Conv2dStridedDgradFilterIteratorOptimizedParams const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    int start_r, int start_s,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params), 
    problem_size_(problem_size),
    pointer_(reinterpret_cast<char const *>(ptr)),
    predicates_{0},
    filter_r_(start_r),
    filter_s_(start_s),
    start_r_(start_r),
    start_s_(start_s) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.row() + thread_coord.strided();
    Index column = threadblock_offset.column() + thread_coord.contiguous();

    reset_bytes_s_ = (problem_size_.num_gemm_k_filter_s(start_s_) - 1) * params_.inc_next[0];
    reset_bytes_r_ = reset_bytes_s_ +
                      (problem_size_.num_gemm_k_filter_r(start_r_) - 1) * params_.inc_next[1];

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int filter_k = filter_k_ + s * ThreadMap::Delta::kStrided;
        int filter_c = column + c * ThreadMap::Delta::kContiguous;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          uint32_t pred = ((filter_k < problem_size_.K && (filter_c + v * AccessType::kElements) < problem_size_.C) ? 1u : 0);
  
          int pred_idx = c + s * ThreadMap::Iterations::kContiguous;
          
          predicates_[v] |= (pred << pred_idx);
        }
      }
    }

    TensorCoord coord{filter_k_, filter_r_, filter_s_, column};

    pointer_ += params_.layout(coord) * sizeof_bits<Element>::value / 8;

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

    pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_DEVICE
  void advance() {

    int next_idx = 0;
    LongIndex reset_bytes = params_.reset_bytes;

    // Move filter_s by stride_w
    filter_s_ +=  problem_size_.stride_w;
    if (filter_s_ >= problem_size_.S) {
      
      // Restore filter_s
      filter_s_ = start_s_;

      // Move filter_r by stride_h
      filter_r_ += problem_size_.stride_h;
#if 0
      bool check = (filter_r_ < problem_size_.R);

      filter_r_ = check ? filter_r_ : start_r_;
      next_idx = check ? 1 : 2;
      reset_bytes += (check ? reset_bytes_s_ : reset_bytes_r_);
#else
    asm volatile(
        "{\n\t"
        " .reg .pred %%p;\n\t"
        " .reg .s64 t1;\n\t"
        " setp.lt.s32 %%p, %3, %4;\n\t"
        " selp.s32 %0, %3, %5, %%p;\n\t"
        " selp.s32 %1, 1, 2, %%p;\n\t"
        " selp.s64 t1, %6, %7, %%p;\n\t"
        " add.s64 %2, %8, t1;\n\t"
        "}\n"
        : "=r"(filter_r_), "=r"(next_idx), "=l"(reset_bytes)
        : "r"(filter_r_), "r"(problem_size_.R), "r"(start_r_),
          "l"(reset_bytes_s_), "l"(reset_bytes_r_), "l"(reset_bytes));
#endif
    }

    // offset pointers by offset_bytes
    pointer_ += (params_.inc_next[next_idx] - reset_bytes);

    if (next_idx == 2) {
      filter_k_ += params_.filter_k_delta;
    }

    // Clear predicates if needed
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      if (filter_k_ + s * ThreadMap::Delta::kStrided >= problem_size_.K) {
        uint32_t kClearMask = ((1u << ThreadMap::Iterations::kContiguous) - 1) << (s * ThreadMap::Iterations::kContiguous);

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          predicates_[v] = (predicates_[v] & (~kClearMask));
        }
      }
    }
  }

  /// Returns true if the current coordinate is within the filter tensor W
  CUTLASS_HOST_DEVICE
  bool valid() {
    LongIndex pred_idx = iteration_contiguous_ + iteration_strided_ * ThreadMap::Iterations::kContiguous;
    return (predicates_[iteration_vector_] & (1u << pred_idx));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    return reinterpret_cast<AccessType const *>(pointer_ + 
      iteration_contiguous_ * ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value / 8) + iteration_vector_;
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterTileAccessIteratorOptimized &operator++() {
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

      // Move to the next K coordinate within the tile
      pointer_ += params_.inc_next_strided;

      return *this;
    }
    iteration_strided_ = 0;
 
    return *this;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.C % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Conv2dDgradFilterTileAccessIteratorOptimized unity strided dgrad is more performant for  dgrad
// on problem sizes with stride = {1x1}
template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_,
  typename AccessType_
>
class Conv2dDgradFilterTileAccessIteratorOptimized <
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
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kUnity;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;
 
  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
  
  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements), 
    "Vectors implied by the thread map must be divisible by the access type.");
 
  //
  // Parameters structure
  //

  struct Params : Conv2dDgradFilterIteratorOptimizedParams {

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Conv2dDgradFilterIteratorOptimizedParams const &base): 
      Conv2dDgradFilterIteratorOptimizedParams(base) { }
      
    CUTLASS_HOST_DEVICE
    Params(
      Conv2dProblemSize const &problem_size, 
      Layout const &layout
    ):
      Conv2dDgradFilterIteratorOptimizedParams(
        problem_size,
        layout,
        sizeof_bits<Element>::value,
        {Shape::kRow, Shape::kColumn},
        ThreadMap::kThreads,
        ThreadMap::kElementsPerAccess,
        {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
        {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided}
      ) { }

  };

private:

  Conv2dDgradFilterIteratorOptimizedParams const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  char const *pointer_;

  uint32_t predicates_[kAccessesPerVector];
  int filter_rs_;
  int filter_k_;

  //
  // Assertions
  //

  // We map predicates into bits packed in this uint32_t container
  static_assert(ThreadMap::Iterations::kStrided *
    ThreadMap::Iterations::kContiguous < sizeof(predicates_) * 8,
    "Currently, the number of loads per iteration is limited by the size of the predicates container.");

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterTileAccessIteratorOptimized(
    Conv2dDgradFilterIteratorOptimizedParams const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params), 
    problem_size_(problem_size),
    pointer_(reinterpret_cast<char const *>(ptr)),
    predicates_{0},
    filter_rs_(0),
    filter_k_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.row() + thread_coord.strided();
    Index column = threadblock_offset.column() + thread_coord.contiguous();

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int filter_k = filter_k_ + s * ThreadMap::Delta::kStrided;
        int filter_c = column + c * ThreadMap::Delta::kContiguous;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          uint32_t pred = ((filter_k < problem_size_.K && (filter_c + v * AccessType::kElements) < problem_size_.C) ? 1u : 0);
  
          int pred_idx = c + s * ThreadMap::Iterations::kContiguous;
          
          predicates_[v] |= (pred << pred_idx);
        }
      }
    }

    pointer_ += (
      filter_k_ * params.layout.stride()[2] + column
    ) * sizeof_bits<Element>::value / 8;

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

    pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_HOST_DEVICE
  void advance() {

    LongIndex next = params_.inc_next_rs;

    // moves to the next tile
    ++filter_rs_;
    if (filter_rs_ == params_.RS) {

      filter_rs_ = 0;
      next = params_.inc_next_k;
      filter_k_ += params_.filter_k_delta;
    }

    // Clear predicates if needed
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      if (filter_k_ + s * ThreadMap::Delta::kStrided >= problem_size_.K) {
        uint32_t kClearMask = ((1u << ThreadMap::Iterations::kContiguous) - 1) << (s * ThreadMap::Iterations::kContiguous); 

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          predicates_[v] = (predicates_[v] & (~kClearMask));
        }
      }
    }
      
    pointer_ += next;
  }

  /// Returns true if the current coordinate is within the filter tensor W
  CUTLASS_HOST_DEVICE
  bool valid() {
    LongIndex pred_idx = iteration_contiguous_ + iteration_strided_ * ThreadMap::Iterations::kContiguous;
    return (predicates_[iteration_vector_] & (1u << pred_idx));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    return reinterpret_cast<AccessType const *>(pointer_ + 
      iteration_contiguous_ * ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value / 8) + iteration_vector_;
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterTileAccessIteratorOptimized &operator++() {
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

      // Move to the next K coordinate within the tile
      pointer_ += params_.inc_next_strided;

      return *this;
    }
    iteration_strided_ = 0;
 
    return *this;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.C % AccessType::kElements) {
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
