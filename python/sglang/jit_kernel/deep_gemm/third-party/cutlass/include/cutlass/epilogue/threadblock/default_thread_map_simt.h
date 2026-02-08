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

/// Defines the optimal thread map for SIMT accumulator layouts
template <
  typename ThreadblockShape_,
  typename WarpShape_,
  typename MmaSimtPolicy_,
  int PartitionsK,
  typename Element_,
  int ElementsPerAccess
>
struct DefaultThreadMapSimt {

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using MmaSimtPolicy = MmaSimtPolicy_;
  static int const kPartitionsK = PartitionsK;
  using Element = Element_;
  static int const kElementsPerAccess = ElementsPerAccess;

  //
  // Definitions
  //

  struct Detail {

    static int const kWarpSize = 32;

    static_assert(
      !(ThreadblockShape::kM % WarpShape::kM) &&
      !(ThreadblockShape::kN % WarpShape::kN), "Divisibility");

    /// Number of warps
    using WarpCount = gemm::GemmShape<
      ThreadblockShape::kM / WarpShape::kM,
      ThreadblockShape::kN / WarpShape::kN,
      kPartitionsK
    >;

    /// Computes number of thread-level matrix multiplies are needed to span a warp
    static int const kGroupCount =
      WarpShape::kM / (MmaSimtPolicy::WarpShape::kRow * MmaSimtPolicy::LaneMmaShape::kM);

    /// Number of participating threads
    static int const kThreads = WarpCount::kCount * kWarpSize;

    /// Number of iterations
    static int const kIterations = MmaSimtPolicy::LaneMmaShape::kM * kGroupCount;
  };

  //
  // ThreadMap
  //
  
  /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying concept OutputTileThreadMap
  using Type = OutputTileOptimalThreadMap<
    OutputTileShape<                          // Shape
      ThreadblockShape::kN, 
      1, 
      MmaSimtPolicy::WarpShape::kRow, 
      Detail::WarpCount::kM, 
      1>,
    OutputTileShape<                          // Count
      1, 
      MmaSimtPolicy::LaneMmaShape::kM, 
      Detail::kGroupCount, 
      1, 
      Detail::kIterations>,
    Detail::kThreads,
    kElementsPerAccess,
    sizeof_bits<Element>::value
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
