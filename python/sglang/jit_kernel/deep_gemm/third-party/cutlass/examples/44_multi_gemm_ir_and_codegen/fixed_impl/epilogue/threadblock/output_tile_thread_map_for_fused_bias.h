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
  \brief Metaprogram for determining the mapping of output elements to threads for epilogue tiles.

  
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/fast_math.h"

#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// RowArrangement determines how one or more warps cover a region of consecutive rows.
template <
  typename Shape,
  int WarpsRemaining,
  int ElementsPerAccess,
  int ElementSize,
  bool Is2dTile
>
struct RowArrangementBiasAct;

/// RowArrangement in which each warp's access is a 1D tiled arrangement.
template <
  typename Shape,
  int WarpsRemaining,
  int ElementsPerAccess,
  int ElementSize
>
struct RowArrangementBiasAct<Shape, WarpsRemaining, ElementsPerAccess, ElementSize, false> {
  static int const kWarpSize = 32;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kElementSize = ElementSize;

  static int const kIterationsRow = 1;
  static int const kDeltaRow = 1;
  static int const kIterationsColumn = Shape::kColumn / kElementsPerAccess / kWarpSize;
  static int const kDeltaColumn = kWarpSize * kElementsPerAccess;

  static int const kAccessWidth = kWarpSize;
  static int const kAccessRows = 1;
  static int const kWarpPartitionsRow = 1;
  static int const kWarpPartitionsColumn = WarpsRemaining;
};

/// RowArrangement in which each warp's access is a 2D tiled arrangement.
template <
  typename Shape,
  int WarpsRemaining,
  int ElementsPerAccess,
  int ElementSize
>
struct RowArrangementBiasAct<Shape, WarpsRemaining, ElementsPerAccess, ElementSize, true> {

  static int const kMemoryAccessSize = 4;//128;
  static int const kWarpSize = 32;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kElementSize = ElementSize;

  struct Detail {
    static int const kShapeRow = Shape::kRow / WarpsRemaining;
    static int const kShapeWidth = Shape::kColumn / kElementsPerAccess;

    static int const kTargetMemoryAccessWidth = 
      kMemoryAccessSize / (kElementsPerAccess * kElementSize / 8);

    static int const kTargetAccessRows = kWarpSize / kTargetMemoryAccessWidth;
  };

  static int const kAccessWidth = 
    (Detail::kTargetAccessRows > Detail::kShapeRow ?
      kWarpSize / Detail::kShapeRow
      : const_min(
          Detail::kShapeWidth,
        const_min(kWarpSize, kMemoryAccessSize / (kElementsPerAccess * kElementSize / 8))
        ));

  static int const kAccessRows =
    (Detail::kTargetAccessRows > Detail::kShapeRow ?
      Detail::kShapeRow
      : const_min(Shape::kRow, kWarpSize / kAccessWidth));

  static int const kIterationsRow = Detail::kShapeRow / kAccessRows;
  static int const kDeltaRow = kAccessRows;

  static int const kIterationsColumn = Detail::kShapeWidth / kAccessWidth;
  static int const kDeltaColumn = kAccessWidth * kElementsPerAccess;

  static_assert( kAccessWidth * kElementsPerAccess <= Shape::kColumn, "Accessing too many elements per access");
  static_assert( kIterationsColumn > 0, "Iteration Count Column must be > 0" );
  static_assert( kIterationsRow > 0, "Iteration Count Row must be > 0" );

  static int const kWarpPartitionsRow = 1;
  static int const kWarpPartitionsColumn = 1;
};

}

////////////////////////////////////////////////////////////////////////////////

/// Template metaprogram for partitioning a 4D space across warps to achieve several performance
/// objectives:
///
///   - coalesced memory accesses in units of 16 Byte lines
///   - minimal address arithmetic
///   - minimal predicate calculations
///
template <
  typename Shape_,
  typename Count_,
  int Threads,
  int ElementsPerAccess,
  int ElementSize
>
struct OutputTileOptimalThreadMapBiasAct {

  using Shape = Shape_;
  using Count = Count_;

  static int const kWarpSize = 32;
  static int const kThreads = Threads;
  static int const kWarpCount = kThreads / kWarpSize;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kElementSize = ElementSize;

  //
  // Metaprogram computation
  //

  struct Detail {

    // Clusters
    static int const kIterationsCluster = 
      ((Shape::kCluster > kWarpCount) ?
        Shape::kCluster / kWarpCount
        : 1);

    static int const kDeltaCluster =
      ((Shape::kCluster > kWarpCount) ?
        Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup * Shape::kCluster / kIterationsCluster
        : 1);

    static int const kCompactedDeltaCluster =
      ((Shape::kCluster > kWarpCount) ?
        Shape::kRow * Shape::kGroup * Shape::kCluster / kIterationsCluster
        : 1);

    static int const kWarpPartitionsCluster =
      ((Shape::kCluster > kWarpCount) ?
        kWarpCount
        : kWarpCount / Shape::kCluster);

    static int const kWarpsRemainingForGroups =
      ((Shape::kCluster > kWarpCount) ? 1 : kWarpCount / Shape::kCluster);

    // Groups
    static int const kIterationsGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kGroup / kWarpsRemainingForGroups
        : 1);

    static int const kDeltaGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kRow * Count::kRow * Shape::kGroup / kIterationsGroup
        : 1);

    static int const kCompactedDeltaGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        Shape::kRow * Shape::kGroup / kIterationsGroup
        : 1);

    static int const kWarpPartitionsGroup =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        1
        : kWarpsRemainingForGroups / Shape::kGroup);

    static int const kWarpsRemainingForRows =
      ((Shape::kGroup > kWarpsRemainingForGroups) ?
        1
        : kWarpsRemainingForGroups / Shape::kGroup);
    
    // Rows
    using RowArrangement = detail::RowArrangementBiasAct<
      Shape,
      kWarpsRemainingForRows,
      kElementsPerAccess,
      kElementSize,
      (Shape::kRow > kWarpsRemainingForRows)
    >;

    // Warp partitions
    using WarpPartitions = OutputTileShape<
      RowArrangement::kWarpPartitionsColumn,
      RowArrangement::kWarpPartitionsRow,
      kWarpPartitionsGroup,
      kWarpPartitionsCluster,
      1>;

    static int const kAccessWidth = RowArrangement::kAccessWidth;
    static int const kAccessRows = RowArrangement::kAccessRows;
  };

  //
  // Output
  //

  using Iterations = OutputTileShape<
    Detail::RowArrangement::kIterationsColumn, 
    Detail::RowArrangement::kIterationsRow, 
    Detail::kIterationsGroup, 
    Detail::kIterationsCluster, 
    1>;

  using Delta = OutputTileShape<
    Detail::RowArrangement::kDeltaColumn,
    Detail::RowArrangement::kDeltaRow,
    Detail::kDeltaGroup,
    Detail::kDeltaCluster,
    1>;

  /// Initial offset function
  CUTLASS_HOST_DEVICE
  static MatrixCoord initial_offset(int thread_idx) {

    int warp_idx = thread_idx / kWarpSize;
    int lane_idx = thread_idx % kWarpSize;

    // Compute warp location
    int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
    int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

    int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
    int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

    int row_idx = residual_group / Detail::WarpPartitions::kRow;
    int col_idx = residual_group % Detail::WarpPartitions::kRow;

    // Compute per-lane offset
    int lane_row_offset = lane_idx / Detail::kAccessWidth;
    int lane_col_offset = lane_idx % Detail::kAccessWidth;

    // Compute coordinate in output space
    int cluster_offset = cluster_idx * Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup;
    int group_offset = group_idx * Shape::kRow * Count::kRow;
    int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
    int column_offset = col_idx * Iterations::kColumn * Detail::kAccessWidth * kElementsPerAccess;

    return MatrixCoord(
      cluster_offset + group_offset + row_offset + lane_row_offset,
      (column_offset + lane_col_offset) * kElementsPerAccess
    );
  }

};


////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass
