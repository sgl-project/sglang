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

#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the optimal thread map for TensorOp accumulator layouts
template <
  typename ThreadblockShape,
  typename WarpShape,
  int PartitionsK,
  typename ElementOutput,
  int ElementsPerAccess,
  typename ElementAccumulator
>
struct DefaultThreadMapVoltaTensorOp;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the optimal thread map for TensorOp accumulator layouts
template <
  typename ThreadblockShape_,
  typename WarpShape_,
  int PartitionsK,
  typename ElementOutput_,
  int ElementsPerAccess
>
struct DefaultThreadMapVoltaTensorOp<
  ThreadblockShape_, 
  WarpShape_, 
  PartitionsK, 
  ElementOutput_, 
  ElementsPerAccess, 
  half_t> {

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  static int const kPartitionsK = PartitionsK;
  using ElementOutput = ElementOutput_;
  static int const kElementsPerAccess = ElementsPerAccess;
  using ElementAccumulator = half_t;

  //
  // Definitions
  //

  struct Detail {

    static int const kTensorOpRows = 16;
    static int const kWarpSize = 32;
    static int const kInterleavedTilesM = WarpShape::kM / 32;

    static_assert(
      !(ThreadblockShape::kM % WarpShape::kM) &&
      !(ThreadblockShape::kN % WarpShape::kN), "Divisibility");

    /// Number of warps
    using WarpCount = gemm::GemmShape<
      ThreadblockShape::kM / WarpShape::kM,
      ThreadblockShape::kN / WarpShape::kN,
      kPartitionsK
    >;

    /// Number of participating threads
    static int const kThreads = WarpCount::kCount * kWarpSize;

    using Shape = cutlass::epilogue::threadblock::OutputTileShape<
      ThreadblockShape::kN,   // column
      4,                      // row
      4,                      // group
      WarpCount::kM,          // cluster
      1                       // tile
    >;
    
    /// Number of iterations per subspace
    using Count = cutlass::epilogue::threadblock::OutputTileShape<
      1,                                // column
      2,                                // row
      kInterleavedTilesM,               // group
      1,                                // cluster
      WarpShape::kM / kTensorOpRows     // iterations
    >;
  };

  //
  // ThreadMap
  //
  
  /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying concept OutputTileThreadMap
  using Type = OutputTileOptimalThreadMap <
    typename Detail::Shape,
    typename Detail::Count,
    Detail::kThreads,
    kElementsPerAccess,
    sizeof_bits<ElementOutput>::value
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the optimal thread map for TensorOp accumulator layouts
template <
  typename ThreadblockShape_,
  typename WarpShape_,
  int PartitionsK,
  typename ElementOutput_,
  int ElementsPerAccess
>
struct DefaultThreadMapVoltaTensorOp<
  ThreadblockShape_,
  WarpShape_,
  PartitionsK,
  ElementOutput_,
  ElementsPerAccess,
  float> {

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  static int const kPartitionsK = PartitionsK;
  using ElementOutput = ElementOutput_;
  static int const kElementsPerAccess = ElementsPerAccess;
  using ElementAccumulator = float;

  //
  // Definitions
  //

  struct Detail {

    static int const kTensorOpRows = 16;
    static int const kWarpSize = 32;
    static int const kInterleavedTilesM = WarpShape::kM / 32;

    static_assert(
      !(ThreadblockShape::kM % WarpShape::kM) &&
      !(ThreadblockShape::kN % WarpShape::kN), "Divisibility");

    /// Number of warps
    using WarpCount = gemm::GemmShape<
      ThreadblockShape::kM / WarpShape::kM,
      ThreadblockShape::kN / WarpShape::kN,
      kPartitionsK
    >;

    /// Number of participating threads
    static int const kThreads = WarpCount::kCount * kWarpSize;

    using Shape = cutlass::epilogue::threadblock::OutputTileShape<
      ThreadblockShape::kN,   // column
      4,                      // row
      4,                      // group
      WarpCount::kM,          // cluster
      1                       // tile
    >;
    
    /// Number of iterations per subspace
    using Count = cutlass::epilogue::threadblock::OutputTileShape<
      1,                                // column
      2,                                // row
      kInterleavedTilesM,               // group
      1,                                // cluster
      WarpShape::kM / kTensorOpRows     // iterations
    >;
  };

  //
  // ThreadMap
  //
  
  /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying concept OutputTileThreadMap
  using Type = OutputTileOptimalThreadMap <
    typename Detail::Shape,
    typename Detail::Count,
    Detail::kThreads,
    kElementsPerAccess,
    sizeof_bits<ElementOutput>::value
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
