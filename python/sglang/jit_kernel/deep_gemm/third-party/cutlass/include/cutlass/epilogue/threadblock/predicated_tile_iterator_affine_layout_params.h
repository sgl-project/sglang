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
#include "cutlass/layout/matrix.h"
#include "cutlass/fast_math.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Rank
>
struct PredicatedTileIteratorAffineLayoutRankNParams {
  using Layout = layout::AffineRankN<Rank>;
  using TensorCoord = typename Layout::TensorCoord;

  static bool const kBigEndian = false;
  
  //
  // Data members
  //

  Layout layout;

  /// Stride in units of bytes along M modes
  Coord<Layout::kRank/2, typename Layout::LongIndex> stride_m;

  /// Stride in units of bytes along N modes
  Coord<Layout::kRank/2, typename Layout::LongIndex> stride_n;

  /// Fast divmod objects divided by tensor extents
  FastDivmod divmod_m[(Layout::kRank == 2) ? 1 : (Layout::kRank/2 - 1)];

  /// Fast divmod objects divided by tensor extents
  FastDivmod divmod_n[(Layout::kRank == 2) ? 1 : (Layout::kRank/2 - 1)];

  int64_t rank2_inc_col;
  int64_t rank2_inc_row;

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorAffineLayoutRankNParams() { }

  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorAffineLayoutRankNParams(TensorCoord const &extent, 
                                                Layout const &layout_,
                                                int64_t element_sizeof_bits)
  : layout(layout_) 
  {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Layout::kRank / 2; ++i) {
      stride_m[i] = OffsetBytes(layout_.stride()[i], element_sizeof_bits);
      stride_n[i] = OffsetBytes(layout_.stride()[i + Layout::kRank / 2], element_sizeof_bits);
    }

    if (kBigEndian) {
      // "Big Endian" scheme
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Layout::kRank / 2 - 1; ++i) {
        divmod_m[i] = FastDivmod(extent[i + 1]);
        divmod_n[i] = FastDivmod(extent[i + Layout::kRank / 2 + 1]);
      }
    }
    else {
      // "Little Endian" scheme
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Layout::kRank / 2 - 1; ++i) {
        divmod_m[i] = FastDivmod(extent[i]);
        divmod_n[i] = FastDivmod(extent[i + Layout::kRank / 2]);
      }
    }

    #if 0
    //
    // Debug print statements to verify extents and strides are passed correctly.
    //
    printf("PredicatedTileIteratorAffine::Params() entered\n");

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Layout::kRank; ++i) {
      printf("  extent[%d]: %d\n", i, extent[i]);
    }
    for (int i = 0; i < Layout::kRank; ++i) {
      printf("  stride[%d]: %ld\n", i, layout_.stride()[i]);
    }
    printf("PredicatedTileIteratorAffine::Params() returning\n");
    #endif
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorAffineLayoutRankNParams(Layout const &layout_,
                                                int32_t threadmap_delta_kColumn,
                                                int32_t threadmap_delta_kRow,
                                                int64_t element_sizeof_bits)
  : layout(layout_) 
  {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Layout::kRank / 2; ++i) {
      stride_m[i] = OffsetBytes(layout_.stride()[i], element_sizeof_bits);
      stride_n[i] = OffsetBytes(layout_.stride()[i + Layout::kRank / 2], element_sizeof_bits);
    }

    rank2_inc_col = threadmap_delta_kColumn * stride_n[0];
    rank2_inc_row = threadmap_delta_kRow * stride_m[0];
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
