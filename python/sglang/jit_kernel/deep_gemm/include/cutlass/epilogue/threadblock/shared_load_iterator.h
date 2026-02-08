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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  int MaxAlignment = ThreadMap_::kElementsPerAccess * sizeof_bits<Element_>::value / 8
>
class SharedLoadIterator {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::TileShape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

  static int const kMinAlignment = ThreadMap_::kElementsPerAccess * sizeof_bits<Element_>::value / 8;

  static int const kAlignment = (MaxAlignment < kMinAlignment ? MaxAlignment : kMinAlignment);

  static int const kThreads = ThreadMap::kThreads;

  /// Fragment object
  using Fragment = Array<
    Element, 
    ThreadMap::Iterations::kColumn * 
    ThreadMap::Iterations::kRow * 
    ThreadMap::Iterations::kGroup * 
    ThreadMap::Iterations::kCluster * 
    ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<
    Element, 
    ThreadMap::kElementsPerAccess, 
    kAlignment>;

  /// Vector type used for SMEM loads
  using LoadType = AlignedArray<
    Element,
    const_min(128 / sizeof_bits<Element>::value, ThreadMap::kElementsPerAccess),
    const_min(16, kAlignment)
  >;

  static int const kLoadsPerAccess = AccessType::kElements / LoadType::kElements;

private:

  //
  // Data members
  //

  /// Byte-level pointer
  uint8_t *byte_pointer_;

  /// Stride along adjacent rows
  int stride_;

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  SharedLoadIterator(
    TensorRef ref,
    int thread_idx
  ):
    byte_pointer_(reinterpret_cast<uint8_t *>(ref.data())),
    stride_((ref.stride(0) * sizeof_bits<Element>::value) / 8) {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

    // Initialize pointer
    byte_pointer_ +=
      thread_offset.row() * stride_ + 
      thread_offset.column() * sizeof(AccessType) / kElementsPerAccess;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &offset) {
    byte_pointer_ += 
      offset.row() * Shape::kRow * stride_ + 
      offset.column() * Shape::kColumn * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {


    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          uint8_t const *byte_pointer = byte_pointer_ + 
            row * ThreadMap::Delta::kRow * stride_ + 
            group * ThreadMap::Delta::kGroup* stride_ + 
            cluster * ThreadMap::Delta::kCluster * stride_ +
            pointer_offset * sizeof_bits<Element>::value / 8;

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          LoadType *frag_ptr = reinterpret_cast<LoadType *>(&frag);
          LoadType const *memory_pointer = reinterpret_cast<LoadType const *>(byte_pointer);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            
            int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn + column;

            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kLoadsPerAccess; ++v) {
              frag_ptr[frag_idx * kLoadsPerAccess + v] = 
                memory_pointer[(column * ThreadMap::Delta::kColumn / kElementsPerAccess) * kLoadsPerAccess + v];
            }
          }
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void set_smem_base_address(Index address) {
  }

  /// Loads a fragment
  CUTLASS_DEVICE
  void load(Fragment &frag) const {

    load_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
