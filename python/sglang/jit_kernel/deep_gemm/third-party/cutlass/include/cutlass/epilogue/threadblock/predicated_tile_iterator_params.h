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
  \brief 
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/conv/conv2d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct OutputTileShapeDesc {

  int column;
  int row;
  int group;
  int cluster;
  int tile;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  OutputTileShapeDesc(): column(0), row(0), group(0), cluster(0), tile(0) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  OutputTileShapeDesc(
    int column_,
    int row_,
    int group_,
    int cluster_,
    int tile_
  ):
    column(column_),
    row(row_),
    group(group_),
    cluster(cluster_),
    tile(tile_) { }

  /// Total number of points in the 5D space
  CUTLASS_HOST_DEVICE
  int count() const {
    return column * row * group * cluster * tile;
  }

  #if 0
  CUTLASS_HOST_DEVICE
  void print() const {
    printf("{%d, %d, %d, %d, %d}", column, row, group, cluster, tile);
  }
  #endif
};

/// Helper template to construct an OutputTileShapeDesc from a OutputTileShape template.
template <typename Shape>
CUTLASS_HOST_DEVICE
OutputTileShapeDesc make_OutputTileShapeDesc() {
  return OutputTileShapeDesc(
    Shape::kColumn,
    Shape::kRow,
    Shape::kGroup,
    Shape::kCluster,
    Shape::kTile
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Thread map description
struct OutputTileThreadMapDesc {

  int threads;
  int elements_per_access;
  OutputTileShapeDesc shape;
  OutputTileShapeDesc iterations;
  OutputTileShapeDesc delta;
  OutputTileShapeDesc count;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  OutputTileThreadMapDesc() { }

  CUTLASS_HOST_DEVICE
  OutputTileThreadMapDesc(
    int threads_,
    int elements_per_access_,
    OutputTileShapeDesc shape_,
    OutputTileShapeDesc iterations_,
    OutputTileShapeDesc delta_,
    OutputTileShapeDesc count_
  ):
    threads(threads_), 
    elements_per_access(elements_per_access_),
    shape(shape_),
    iterations(iterations_),
    delta(delta_),
    count(count_) 
  {
    
  }
};

/// Helper template to construct an OutputTileShapeDesc from a OutputTileThreadMap template.
template <typename ThreadMap>
CUTLASS_HOST_DEVICE
OutputTileThreadMapDesc make_OutputTileThreadMapDesc() {
  return OutputTileThreadMapDesc(
    ThreadMap::kThreads,
    ThreadMap::kElementsPerAccess,
    make_OutputTileShapeDesc<typename ThreadMap::Shape>(),
    make_OutputTileShapeDesc<typename ThreadMap::Iterations>(),
    make_OutputTileShapeDesc<typename ThreadMap::Delta>(),
    make_OutputTileShapeDesc<typename ThreadMap::Count>()
  );
}
///////////////////////////////////////////////////////////////////////////////

//
// Parameters struct for PredicatedTileIterator
//

struct PredicatedTileIteratorParams {

  using Index = int32_t;
  using LongIndex = int64_t;

  //
  // Data members
  //

  LongIndex stride;               ///< stride in bytes between rows

  LongIndex increment_row;        ///< increment quantity (in bytes) to advance when moving between rows
  LongIndex increment_group;      ///< increment quantity (in bytes) to advance when moving to the next group
  LongIndex increment_cluster;    ///< increment quantity (in bytes) to advance when moving to the next cluster

  LongIndex advance_row;          ///< amount to add to move to the next 'row' position
  LongIndex advance_group;        ///< amount to add to move to the next 'group' position
  LongIndex advance_cluster;      ///< amount to add to move to the next 'cluster' position
  LongIndex advance_tile;         ///< amount to add to move to the next 'tile'

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Status initialize(LongIndex stride_, OutputTileThreadMapDesc thread_map) {
    
    stride = stride_;

    increment_row = stride * thread_map.delta.row;

    increment_group = stride * thread_map.delta.group
      - stride * thread_map.delta.row * (thread_map.iterations.row - 1);

    increment_cluster = stride * thread_map.delta.cluster
      - stride * thread_map.delta.group * (thread_map.iterations.group - 1)
      - stride * thread_map.delta.row * (thread_map.iterations.row - 1);

    advance_row = stride * thread_map.shape.row;

    advance_group = 
      stride * 
      (thread_map.shape.group - 1) * thread_map.shape.row * thread_map.count.row;
    
    advance_cluster = 
      stride * 
      thread_map.count.group * 
      thread_map.shape.group * 
      thread_map.count.row * 
      thread_map.shape.row;
    
    advance_tile =
      stride * 
      thread_map.shape.group * 
      thread_map.shape.row * 
      thread_map.shape.cluster * 
      thread_map.shape.tile;

    return Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Status initialize(Index stride_, OutputTileThreadMapDesc thread_map) {
    return initialize(LongIndex(stride_), thread_map); 
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorParams() {
    initialize(LongIndex(0), OutputTileThreadMapDesc());
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorParams(Index stride, OutputTileThreadMapDesc thread_map) {
    initialize(stride, thread_map);
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorParams(LongIndex stride, OutputTileThreadMapDesc thread_map) {
    initialize(stride, thread_map);
  }
};

///////////////////////////////////////////////////////////////////////////////

//
// Parameters struct for PredicatedTileIteratorDirect2dConv
//

struct PredicatedTileIteratorDirect2dConvParams{
  using Index = int32_t;
  using LongIndex = int64_t;

  //
  // Data members
  //
  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  LongIndex stride;
  LongIndex stride_n;
  LongIndex stride_p;

  int N;
  int P;
  int Q;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Status initialize(LongIndex stride_,
                    cutlass::conv::Conv2dProblemSize const &problem_size,
                    MatrixCoord threadblock_output_shape) {
    stride = stride_; // The stride per row of output tensor (bytes)
    stride_n = problem_size.P * problem_size.Q;
    stride_p = problem_size.Q ;

    N = problem_size.N;
    P = problem_size.P;
    Q = problem_size.Q;

    // Fastdivmod for output O, P, Q
    if(threadblock_output_shape.row() != 0 && threadblock_output_shape.column() !=0 ){
      // MSVC emits a "potential divide by 0" warning as error
      // if the code just divides without a check and substitution.

      CUTLASS_ASSERT(threadblock_output_shape.row() != 0);
      const auto row_denom = threadblock_output_shape.row() != 0 ?
        threadblock_output_shape.row() : cutlass::MatrixCoord::Index(1);
      int tiles_p =
          (problem_size.P + (threadblock_output_shape.row() - 1)) / row_denom;

      CUTLASS_ASSERT(threadblock_output_shape.column() != 0);
      const auto col_denom = threadblock_output_shape.column() != 0 ?
        threadblock_output_shape.column() : cutlass::MatrixCoord::Index(1);
      int tiles_q = (problem_size.Q + (threadblock_output_shape.column() - 1)) /
                    col_denom;

      pq_divmod = FastDivmod(tiles_p * tiles_q);
      q_divmod = FastDivmod(tiles_q);
    }

    return Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Status initialize(
      Index stride_,
      cutlass::conv::Conv2dProblemSize const &problem_size = cutlass::conv::Conv2dProblemSize(),
      MatrixCoord threadblock_output_shape = MatrixCoord()) {
    return initialize(LongIndex(stride_), problem_size, threadblock_output_shape);
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorDirect2dConvParams() { initialize(LongIndex(0)); }

  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorDirect2dConvParams(Index stride,
                               cutlass::conv::Conv2dProblemSize const &problem_size,
                               MatrixCoord threadblock_output_shape) {
    initialize(stride, problem_size, threadblock_output_shape);
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorDirect2dConvParams(LongIndex stride,
                               cutlass::conv::Conv2dProblemSize const &problem_size,
                               MatrixCoord threadblock_output_shape) {
    initialize(stride, problem_size, threadblock_output_shape);
  }
};

///////////////////////////////////////////////////////////////////////////////
//  InterleavedPredicatedTileIterator
///////////////////////////////////////////////////////////////////////////////


/// Predicated tile access iterator descriptor object containing template dependent state
struct InterleavedPredicatedTileIteratorDesc {

  int element_size_bits;
  int elements_per_access;
  int threadmap_warp_size;
  layout::PitchLinearCoord threadmap_iterations;
  layout::PitchLinearCoord threadmap_delta;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  InterleavedPredicatedTileIteratorDesc() { }

  CUTLASS_HOST_DEVICE
  InterleavedPredicatedTileIteratorDesc(
    int element_size_bits_,
    int elements_per_access_,
    int threadmap_warp_size_,
    layout::PitchLinearCoord threadmap_iterations_,
    layout::PitchLinearCoord threadmap_delta_
  ):
    element_size_bits(element_size_bits_),
    elements_per_access(elements_per_access_),
    threadmap_warp_size(threadmap_warp_size_),
    threadmap_iterations(threadmap_iterations_),
    threadmap_delta(threadmap_delta_) { }
};

//
// Parameters struct InterleavedPredicatedTileIterator
//

struct InterleavedPredicatedTileIteratorParams {

  using Index = int32_t;
  using LongIndex = int64_t;

  //
  // Data members
  //

  LongIndex stride;               ///< stride in bytes between rows
  LongIndex advance_row;          ///< amount to add to move to the next 'row' position
  LongIndex advance_column;       ///< amount to add to move to the next 'column' position

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Status initialize(LongIndex stride_, InterleavedPredicatedTileIteratorDesc desc) {
    
    stride = stride_;

    advance_row = desc.threadmap_delta.contiguous() * desc.element_size_bits / 8;

    advance_column = stride_ - desc.threadmap_iterations.contiguous() *
                               desc.elements_per_access *
                               desc.element_size_bits *
                               desc.threadmap_warp_size / 8;

    return Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  InterleavedPredicatedTileIteratorParams() {
    initialize(LongIndex(0), InterleavedPredicatedTileIteratorDesc());
  }

  CUTLASS_HOST_DEVICE
  InterleavedPredicatedTileIteratorParams(Index stride, InterleavedPredicatedTileIteratorDesc desc) {
    initialize(stride, desc);
  }

  CUTLASS_HOST_DEVICE
  InterleavedPredicatedTileIteratorParams(LongIndex stride, InterleavedPredicatedTileIteratorDesc desc) {
    initialize(stride, desc);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper template to construct an OutputTileShapeDesc from a OutputTileThreadMap template.
template <typename Element, typename ThreadMap>
CUTLASS_HOST_DEVICE
InterleavedPredicatedTileIteratorDesc make_InterleavedPredicatedTileIteratorDesc() {
  return InterleavedPredicatedTileIteratorDesc(
    sizeof_bits<Element>::value,
    ThreadMap::kElementsPerAccess,
    ThreadMap::kWarpSize,
    {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
    {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided}
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper template to construct an MakePredicatedTileIteratorDesc from a template 
// dependent state
template <typename Element, typename Layout,
   typename ThreadMap>
  struct MakePredicatedTileIteratorDesc;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for layout::RowMajor output data.
template <typename Element, typename ThreadMap>
struct MakePredicatedTileIteratorDesc <
    Element, layout::RowMajor, ThreadMap> {

  CUTLASS_HOST_DEVICE
  OutputTileThreadMapDesc operator()() {

    return make_OutputTileThreadMapDesc<ThreadMap>();
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for layout::ColumnMajorInterleaved<InterleavedN> output data.
template <typename Element, typename ThreadMap, int InterleavedN>
struct MakePredicatedTileIteratorDesc <
    Element, layout::ColumnMajorInterleaved<InterleavedN>, ThreadMap> {

  CUTLASS_HOST_DEVICE
  InterleavedPredicatedTileIteratorDesc operator()() {

    return make_InterleavedPredicatedTileIteratorDesc<Element, ThreadMap>();
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
