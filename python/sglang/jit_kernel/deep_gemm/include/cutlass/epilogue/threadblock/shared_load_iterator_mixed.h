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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops optimized for mixed-precision.

  This assumes the shared memory tile is in a permuted layout which avoids bank conflicts on loading.

  When the fragment is loaded into registers, it matches the row-major thread map assumed by
  the predicated tile iterator writing to global memory.

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
  typename Element_,         ///< Accumulator data type
  int ElementSizeBits_,      ///< Size of accumulator in bits
  int OutputSizeBits_,       ///< Size of output element in bits
  int ElementsPerAccess,     ///< Vector length of output vector
  int ContiguousLanes,       ///< Number of lanes in the warp writing to contiguous elements
                             ///  in the global memory tensor
  bool EightBitsOutputOrLess = (OutputSizeBits_ <= 8)
>
class SharedLoadIteratorMixed;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_          ///< Accumulator data type
>
class SharedLoadIteratorMixed<ThreadMap_, Element_, 32, 16, 8, 8, false> {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

  static int const kAlignment = ThreadMap::kElementsPerAccess * sizeof_bits<Element_>::value / 8;

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
  LoadType const *pointers_[kLoadsPerAccess];

  /// Stride along adjacent rows in units of LoadType
  int stride_;

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  SharedLoadIteratorMixed(
    TensorRef ref,
    int thread_idx
  ):
    stride_((ref.stride(0) / LoadType::kElements)) {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

    // Initialize pointers
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      pointers_[i] = reinterpret_cast<LoadType const *>(ref.data());

      int col_idx = (thread_offset.column() / kElementsPerAccess) * kLoadsPerAccess;
      int bank_offset = (col_idx * int(sizeof(LoadType)) / 128) % kLoadsPerAccess;

      col_idx += (bank_offset + i) % kLoadsPerAccess;

      pointers_[i] += thread_offset.row() * stride_ + col_idx;
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      pointers_[i] += pointer_offset / LoadType::kElements;
    }
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &offset) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      pointers_[i] += 
        offset.row() * Shape::kRow * stride_ + 
        offset.column() * Shape::kColumn / LoadType::kElements;
    }
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

          int row_ptr_offset =
            row * ThreadMap::Delta::kRow * stride_ + 
            group * ThreadMap::Delta::kGroup* stride_ + 
            cluster * ThreadMap::Delta::kCluster * stride_ +
            pointer_offset / LoadType::kElements;

          int frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          LoadType *frag_ptr = reinterpret_cast<LoadType *>(&frag);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            
            int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn + column;

            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kLoadsPerAccess; ++v) {
           
              int vector_idx = (column * ThreadMap::Delta::kColumn / kElementsPerAccess * kLoadsPerAccess); 

              LoadType const *memory_pointer = pointers_[v] + row_ptr_offset;
            
              frag_ptr[frag_idx * kLoadsPerAccess + v] = memory_pointer[vector_idx];
            }
          }
        }
      }
    }
  }

  /// Set base smem address
  CUTLASS_DEVICE
  void set_smem_base_address(Index address) {}

  /// Loads a fragment
  CUTLASS_DEVICE
  void load(Fragment &frag) const {

    load_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for
///   int32_t x 16 => int8_t/int4b_t x 16 and
///   float x 16 => float_e4m3_t/float_e5m2_t x 16
template <
  typename ThreadMap_,      ///< Thread map (concept: OutputTileThreadMap)
  typename Element_,
  int OutputSizeBits_       ///< Size of output element in bits
>
class SharedLoadIteratorMixed<ThreadMap_, Element_, 32, OutputSizeBits_, 16, 8, true> {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;
  static_assert(sizeof_bits<Element>::value == 32, "Element size in bits must be 32.");

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

  static int const kAlignment = 16;

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
    16, 
    kAlignment>;

  /// Vector type used for SMEM loads
  using LoadType = AlignedArray<
    Element,
    4,
    16
  >;

  static int const kLoadsPerAccess = 4;

private:

  //
  // Data members
  //

  /// Byte-level pointer
  LoadType const *pointers_[kLoadsPerAccess];

  /// Stride along adjacent rows in units of LoadType
  int stride_;

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  SharedLoadIteratorMixed(
    TensorRef ref,
    int thread_idx
  ):
    stride_((ref.stride(0) / LoadType::kElements)) {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
    
    // Initialize pointers
    LoadType const *base_ptr = reinterpret_cast<LoadType const *>(ref.data()) + thread_offset.row() * stride_;
      
    int lane_col_idx = thread_offset.column() / 16;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      int lane_offset = (lane_col_idx % 2) * 4 | ((lane_col_idx / 2) * 8) | ((lane_col_idx / 2) ^ i);
 
      pointers_[i] = base_ptr + lane_offset;
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      pointers_[i] += pointer_offset / LoadType::kElements;
    }
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &offset) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      pointers_[i] += 
        offset.row() * Shape::kRow * stride_ + 
        offset.column() * Shape::kColumn / LoadType::kElements;
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int row_ptr_offset =
            row * ThreadMap::Delta::kRow * stride_ + 
            group * ThreadMap::Delta::kGroup* stride_ + 
            cluster * ThreadMap::Delta::kCluster * stride_ +
            pointer_offset / LoadType::kElements;

          int frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          LoadType *frag_ptr = reinterpret_cast<LoadType *>(&frag);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            
            int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn + column;

            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kLoadsPerAccess; ++v) {
           
              LoadType const *memory_pointer = pointers_[v];
            
              frag_ptr[frag_idx * kLoadsPerAccess + v] = memory_pointer[row_ptr_offset];
            }
          }
        }
      }
    }
  }

  /// Set base smem address
  CUTLASS_DEVICE
  void set_smem_base_address(Index address) {}

  /// Loads a fragment
  CUTLASS_DEVICE
  void load(Fragment &frag) {

    load_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for:
///   int32_t x 8 => int8_t/int4b_t x 8 and
///   float x 8 => float_e4m3_t/float_e5m2_t x 8
template <
  typename ThreadMap_,      ///< Thread map (concept: OutputTileThreadMap)
  typename Element_,
  int OutputSizeBits_
>
class SharedLoadIteratorMixed<ThreadMap_, Element_, 32, OutputSizeBits_, 8, 8, true> {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;
  static_assert(sizeof_bits<Element>::value == 32, "Element size in bits must be 32.");

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

  static int const kAlignment = 8;

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
    8, 
    kAlignment>;

  /// Vector type used for SMEM loads
  using LoadType = AlignedArray<
    Element,
    4,
    16
  >;

  static int const kLoadsPerAccess = 2;

private:

  //
  // Data members
  //

  /// Byte-level pointer
  LoadType const *pointers_[kLoadsPerAccess];

  /// Stride along adjacent rows in units of LoadType
  int stride_;

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  SharedLoadIteratorMixed(
    TensorRef ref,
    int thread_idx
  ):
    stride_((ref.stride(0) / LoadType::kElements)) {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
    
    // Initialize pointers
    LoadType const *base_ptr = reinterpret_cast<LoadType const *>(ref.data()) + thread_offset.row() * stride_;
      
    int lane_col_idx = thread_offset.column() / 8;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      int lane_offset = (lane_col_idx % 8) * 2 | ((lane_col_idx / 4) ^ i);

      pointers_[i] = base_ptr + lane_offset;
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      pointers_[i] += pointer_offset / LoadType::kElements;
    }
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &offset) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoadsPerAccess; ++i) {
      pointers_[i] += 
        offset.row() * Shape::kRow * stride_ + 
        offset.column() * Shape::kColumn / LoadType::kElements;
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int row_ptr_offset =
            row * ThreadMap::Delta::kRow * stride_ + 
            group * ThreadMap::Delta::kGroup* stride_ + 
            cluster * ThreadMap::Delta::kCluster * stride_ +
            pointer_offset / LoadType::kElements;

          int frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          LoadType *frag_ptr = reinterpret_cast<LoadType *>(&frag);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            
            int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn + column;

            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kLoadsPerAccess; ++v) {
           
              LoadType const *memory_pointer = pointers_[v];
            
              frag_ptr[frag_idx * kLoadsPerAccess + v] = memory_pointer[row_ptr_offset];
            }
          }
        }
      }
    }
  }

  /// Set base smem address
  CUTLASS_DEVICE
  void set_smem_base_address(Index address) {}

  /// Loads a fragment
  CUTLASS_DEVICE
  void load(Fragment &frag) {

    load_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
