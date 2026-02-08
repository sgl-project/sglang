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
    \brief Unit tests for thread-level GEMM
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"
#include "cutlass/platform/platform.h"

#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"

#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Prototype algorithm for partitioning a 4D space across warps to achieve several performance
/// objectives:
///
///   - coalesced memory accesses in units of 128 Byte lines
///   - minimal address arithmetic
///   - minimal predicate calculations
///
struct OutputTileThreadMapExpr {

  struct Shape {
    int column;
    int row;
    int group;
    int cluster;

    Shape(int col = 1, int r = 1, int g = 1, int c = 1): 
      column(col), row(r), group(g), cluster(c) { }
  };

  int const kWarpSize = 32;
  int const kMemoryAccessSize = 256;  // size in bytes of the preferred memory access size

  //
  // Data members
  //

  Shape shape;
  Shape count;
  int threads;
  int warp_count;
  int elements_per_access;
  int element_size;

  Shape iterations;
  Shape delta;
  Shape warp_partitions;

  int access_width_in_vectors;
  int access_rows;

  //
  // Methods
  //

  OutputTileThreadMapExpr(
    Shape shape_,
    Shape count_,
    int threads_,
    int elements_per_access_,
    int element_size_
  ):
    shape(shape_), 
    count(count_), 
    threads(threads_), 
    warp_count(threads_ / kWarpSize),
    elements_per_access(elements_per_access_), 
    element_size(element_size_) {

    int warps_remaining = warp_count;

    // clusters
    if (shape.cluster > warp_count) {
      iterations.cluster = shape.cluster / warp_count;
      delta.cluster = shape.row * count.row * shape.group * count.group * shape.cluster / iterations.cluster;
      warps_remaining = 1;
      warp_partitions.cluster = warp_count;
    }
    else {
      iterations.cluster = 1;
      delta.cluster = 1;
      warps_remaining = warp_count / shape.cluster;
      warp_partitions.cluster = warps_remaining;
    }

    // group size
    if (shape.group > warps_remaining) {
      iterations.group = shape.group / warps_remaining;
      delta.group = shape.row * count.row * shape.group / iterations.group;
      warps_remaining = 1;
      warp_partitions.group = warps_remaining;
    }
    else {
      iterations.group = 1;
      delta.group = 1;
      warps_remaining = warps_remaining / shape.group;
      warp_partitions.group = warps_remaining;
    }

    // Number of rows in a group
    if (shape.row > warps_remaining) {
      
      // We must cover this shape within a warp
      int shape_row = shape.row / warps_remaining;
      int shape_width_vectors = shape.column / elements_per_access;

      // We would still like to minimize the number of strided increments. We can accomplish this
      // by arranging the memory instructions as 2D, 128B wide accesses.

      int target_memory_access_width = kMemoryAccessSize / (elements_per_access * element_size / 8);
      int target_rows_per_access = kWarpSize / target_memory_access_width;

      if (target_rows_per_access > shape_row) {
        access_rows = shape_row;
        access_width_in_vectors = kWarpSize / access_rows;
      }
      else {

        access_width_in_vectors = cutlass::platform::min(
          shape_width_vectors, 
          cutlass::platform::min(kWarpSize, kMemoryAccessSize / (elements_per_access * element_size / 8)));

        access_rows = cutlass::platform::min(shape_row, kWarpSize / access_width_in_vectors);
      }

      iterations.row = shape_row / access_rows;
      delta.row = access_rows;

      iterations.column = shape_width_vectors / access_width_in_vectors;
      delta.column = access_width_in_vectors * elements_per_access;
      
      warp_partitions.column = 1;
      warp_partitions.row = 1;
    }
    else {
      iterations.row = 1;
      delta.row = 1;
      iterations.column = (shape.column / elements_per_access) / kWarpSize;
      delta.column = kWarpSize * elements_per_access;

      access_width_in_vectors = kWarpSize;
      access_rows = 1;

      warp_partitions.row = 1;
      warp_partitions.column = warps_remaining;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream & operator<<(std::ostream &out, OutputTileThreadMapExpr::Shape const &shape) {
  out << "col: " << shape.column << ", r: " << shape.row << ", g: " << shape.group << ", c: " << shape.cluster;
  return out;
}

std::ostream & operator<<(std::ostream &out, OutputTileThreadMapExpr const &map) {
  out 
    << "  shape(" << map.shape  << ")\n"
    << "  count(" << map.count << ")\n"
    << "  iterations(" << map.iterations << ")\n"
    << "  delta(" << map.delta << ")\n"
    << "  warps(" << map.warp_partitions << ")\n"
    << "  access(width: " << map.access_width_in_vectors 
      << ", rows: " << map.access_rows
      << ") x v" << map.elements_per_access
      << ".b" << map.element_size << "\n";

  return out;
}


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape,
  typename Count,
  int Threads,
  int ElementsPerAccess,
  int ElementSize
>
struct ThreadMapTestbed {
  ThreadMapTestbed() {
    OutputTileThreadMapExpr map(
      { Shape::kColumn, Shape::kRow, Shape::kGroup, Shape::kCluster },
      { Count::kColumn, Count::kRow, Count::kGroup, Count::kCluster },
      Threads,
      ElementsPerAccess,
      ElementSize
    );

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
      Shape,
      Count,
      Threads,
      ElementsPerAccess,
      ElementSize
    >;

    using CompactThreadmap = typename ThreadMap::CompactedThreadMap;
    
    bool const kVerbose = false;

    if (kVerbose) {

      std::cout << map << std::endl;

      std::cout << "ThreadMap::warps remaining:\n"
        << "  for groups: " << ThreadMap::Detail::kWarpsRemainingForGroups << "\n"
        << "    for rows: " << ThreadMap::Detail::kWarpsRemainingForRows << "\n";

      std::cout << "ThreadMap::Access:\n"
        << " width: " << ThreadMap::Detail::kAccessWidth << "\n"
        << "  rows: " << ThreadMap::Detail::kAccessRows << "\n";

      std::cout << "ThreadMap::RowArrangement::Iterations:\n"
        << "  row: " << int(ThreadMap::Detail::RowArrangement::kIterationsRow) << "\n";
    }

    EXPECT_EQ(int(ThreadMap::Delta::kCluster), map.delta.cluster);
    EXPECT_EQ(int(ThreadMap::Delta::kGroup), map.delta.group);
    EXPECT_EQ(int(ThreadMap::Delta::kRow), map.delta.row);
    EXPECT_EQ(int(ThreadMap::Delta::kColumn), map.delta.column);

    EXPECT_EQ(int(ThreadMap::Iterations::kCluster), map.iterations.cluster);
    EXPECT_EQ(int(ThreadMap::Iterations::kGroup), map.iterations.group);
    EXPECT_EQ(int(ThreadMap::Iterations::kRow), map.iterations.row);
    EXPECT_EQ(int(ThreadMap::Iterations::kColumn), map.iterations.column);

    if (kVerbose) {
      std::cout << "Iterations(col: " << ThreadMap::Iterations::kColumn
        << ", r: " << ThreadMap::Iterations::kRow
        << ", g: " << ThreadMap::Iterations::kGroup
        << ", c: " << ThreadMap::Iterations::kCluster << ")\n";

      std::cout << "Delta(col: " << ThreadMap::Delta::kColumn
        << ", r: " << ThreadMap::Delta::kRow
        << ", g: " << ThreadMap::Delta::kGroup
        << ", c: " << ThreadMap::Delta::kCluster << ")\n";

      for (int tid = 0; tid < Threads; ++tid) {
        auto output_coord = ThreadMap::initial_offset(tid);
        auto source_coord = CompactThreadmap::initial_offset(tid);

        std::cout << "T" << tid << " - output: " << output_coord << ", source: " << source_coord << "\n";
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(ThreadMap, f16_tensor_op_64x64_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<64, 8, 1, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 1, 1, 1>;
  int const kThreads = 32;
  int const kElementsPerAccess = 8;
  int const kElementSize = 16;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}


TEST(ThreadMap, f16_tensor_op_128x128_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 8, 2, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 1, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 8;
  int const kElementSize = 16;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f16_tensor_op_256x128_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 8, 4, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 256;
  int const kElementsPerAccess = 8;
  int const kElementSize = 16;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f16_tensor_op_128x256_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<256, 8, 2, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 256;
  int const kElementsPerAccess = 8;
  int const kElementSize = 16;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f16_tensor_op_128x64_64x32x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<64, 8, 2, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 8;
  int const kElementSize = 16;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f16_tensor_op_64x128_128x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 8, 1, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 8;
  int const kElementSize = 16;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_tensor_op_64x64_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<64, 8, 1, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 1, 1, 1>;
  int const kThreads = 32;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_tensor_op_128x128_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 8, 2, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_tensor_op_256x128_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 8, 4, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 256;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_tensor_op_128x256_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<256, 8, 2, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 256;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_tensor_op_128x64_64x32x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<64, 8, 2, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_tensor_op_64x128_128x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 8, 1, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 8, 2, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(ThreadMap, f32_volta_tensor_op_64x64_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<64, 2, 4, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 32;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_volta_tensor_op_64x128_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 2, 4, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 64;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_volta_tensor_op_128x64_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<64, 2, 4, 2, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 64;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_volta_tensor_op_128x64_64x32x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<64, 2, 4, 2, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_volta_tensor_op_128x128_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 2, 4, 2, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_volta_tensor_op_128x256_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<256, 2, 4, 2, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 256;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, f32_volta_tensor_op_256x128_64x64x8) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 2, 4, 4, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 256;
  int const kElementsPerAccess = 4;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(ThreadMap, simt_32x64_32x64x1) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<64, 1, 4, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 32;
  int const kElementsPerAccess = 1;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, simt_32x128_32x64x1) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 1, 4, 1, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 64;
  int const kElementsPerAccess = 1;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, simt_64x128_32x64x1) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 1, 4, 2, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 128;
  int const kElementsPerAccess = 1;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

TEST(ThreadMap, simt_128x128_32x64x1) {

  using Shape = cutlass::epilogue::threadblock::OutputTileShape<128, 1, 4, 4, 1>;
  using Count = cutlass::epilogue::threadblock::OutputTileShape<1, 4, 2, 1, 1>;
  int const kThreads = 256;
  int const kElementsPerAccess = 1;
  int const kElementSize = 32;

  ThreadMapTestbed<Shape, Count, kThreads, kElementsPerAccess, kElementSize>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
