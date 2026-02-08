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
#include "cutlass/layout/pitch_linear.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Defines the optimal thread map for TensorOp accumulator layouts
template <
  typename ThreadblockShape_,
  typename WarpShape_,
  int PartitionsK,
  typename Element_,
  int ElementsPerAccess
>
struct DefaultThreadMapTensorOpForFusedBias {

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  static int const kPartitionsK = PartitionsK;
  using Element = Element_;
  static int const kElementsPerAccess = ElementsPerAccess;

  //
  // Definitions
  //

  struct Detail {

    /// Tensor Operations fundamentally perform operations on 8 rows
    static int const kTensorOpRows = 8;
    static int const kWarpSize = 32;

    static_assert(
      !(ThreadblockShape::kM % WarpShape::kM) &&
      !(ThreadblockShape::kM % WarpShape::kM), "Divisibility");

    /// Number of warps
    using WarpCount = gemm::GemmShape<
      ThreadblockShape::kM / WarpShape::kM,
      ThreadblockShape::kN / WarpShape::kN,
      kPartitionsK
    >;

    /// Number of participating threads
    static int const kThreads = WarpCount::kCount * kWarpSize;
  };

  //
  // ThreadMap
  //
  
  /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying concept OutputTileThreadMap
  using Type = OutputTileOptimalThreadMapBiasAct <
    OutputTileShape<ThreadblockShape::kN, Detail::kTensorOpRows, Detail::WarpCount::kM, 1, 1>,
    OutputTileShape<1, WarpShape::kM / Detail::kTensorOpRows, 1, 1, WarpShape::kM / Detail::kTensorOpRows>,
    Detail::kThreads,
    kElementsPerAccess,
    sizeof_bits<Element>::value
  >;
};

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
