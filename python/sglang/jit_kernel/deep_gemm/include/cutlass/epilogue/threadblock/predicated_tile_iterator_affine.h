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
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator | ForwardTileIterator
///
/// It provides a fast path for the case Rank = 2 which does not need div/rem to 
/// calculate modes.

template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  int Rank
>
class PredicatedTileIteratorAffineRankN {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::AffineRankN<Rank>;
  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = typename Layout::TensorCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup > 0,"ThreadMap::Iterations::kGroup must be > 0");
  static_assert( ThreadMap::Iterations::kCluster > 0,"ThreadMap::Iterations::kCluster must be > 0");
  static_assert( ThreadMap::Iterations::kColumn > 0,"ThreadMap::Iterations::kColumn must be > 0");
  static_assert( !(Layout::kRank % 2), 
    "Layout rank must be even. This assumes the first half of the modes correspond to the 'row' "
    "and the second half of the modes correspond to the 'column'");

  static bool const kBigEndian = false;

  /// Fragment object
  using Fragment = Array<
    Element, 
    ThreadMap::Iterations::kColumn * 
    ThreadMap::Iterations::kRow * 
    ThreadMap::Iterations::kGroup * 
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Parameters structure
  struct Params {

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
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(TensorCoord const &extent, Layout const &layout_): layout(layout_) {

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Layout::kRank / 2; ++i) {
        stride_m[i] = OffsetBytes<Element>(layout_.stride()[i]);
        stride_n[i] = OffsetBytes<Element>(layout_.stride()[i + Layout::kRank / 2]);
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
    Params(Layout const &layout_): layout(layout_) {

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Layout::kRank / 2; ++i) {
        stride_m[i] = OffsetBytes<Element>(layout_.stride()[i]);
        stride_n[i] = OffsetBytes<Element>(layout_.stride()[i + Layout::kRank / 2]);
      }

      rank2_inc_col = ThreadMap::Delta::kColumn * stride_n[0];
      rank2_inc_row = ThreadMap::Delta::kRow * stride_m[0];
    }
  };

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

private:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  Params params_;

  /// Byte-level pointer
  uint8_t *byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in columns
  Index extent_col_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column position (assuming steady-state predicates have been computed)
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];

  /// Offsets in columns, cached for performance
  int64_t offset_modes_n_[ThreadMap::Iterations::kColumn];
 
  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");

private:

  //
  // Methods
  //

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  PredicatedTileIteratorAffineRankN(
    Params const & params,
    Element *pointer,
    MatrixCoord extent,
    int thread_idx,
    MatrixCoord threadblock_offset = MatrixCoord(),
    int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
  ): 
    params_(params)
  {

    MatrixCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    extent_col_ = extent.column();

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    if (Layout::kRank > 2) {
      // Initialize predicates
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {

        // 
        // Compute coordinate and decompose into N modes
        //

        int coord_n = thread_start_column_ + c * ThreadMap::Delta::kColumn;

        mask_.predicates[c] = coord_n < extent.column();
        
        Coord<Layout::kRank / 2, Index> modes_n;

        int64_t offset_modes_n = 0;

        if (kBigEndian) {
          modes_n = CoordinateDecomposition<Layout::kRank / 2>(coord_n, params_.divmod_n);

          offset_modes_n = dot(modes_n, params_.stride_n);
        }
        else {
          modes_n = CoordinateDecompositionLittleEndian<Layout::kRank / 2>(coord_n, params_.divmod_n);

          offset_modes_n = dot(modes_n, params_.stride_n);
        }

        offset_modes_n_[c] = offset_modes_n;

      }

      if (!pointer) {
        mask_.clear();
      }
    }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer);

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) {
    uint8_t const *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        int row_begin = thread_start_row_ + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;
        int64_t offset_modes_m = row_begin * params_.stride_m[0];

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          // 
          // Compute coordinate and decompose into M modes
          //

          int coord_m = row * ThreadMap::Delta::kRow + row_begin;

          Coord<Layout::kRank / 2, Index> modes_m;

          if (Layout::kRank > 2) {
            if (kBigEndian) {
              modes_m = CoordinateDecomposition<Layout::kRank / 2>(coord_m, params_.divmod_m);
            } else {
              modes_m = CoordinateDecompositionLittleEndian<Layout::kRank / 2>(coord_m, params_.divmod_m);
            }

            offset_modes_m = dot(modes_m, params_.stride_m);
          }

          //
          // Compute the offset due to modes M
          //

          bool row_guard = (coord_m < extent_row_);
          int64_t offset_modes_n = thread_start_column_ * params_.stride_n[0];

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            // 
            // Compute coordinate and decompose into N modes
            //
            
            if (Layout::kRank > 2) {
              offset_modes_n = offset_modes_n_[column];
            }

            //
            // Compute the pointer and access
            //
            bool guard;

            if (Layout::kRank > 2) {
              guard = row_guard && mask_.predicates[column];
            } else {
              guard = (coord_m < extent_row_) && 
              ((thread_start_column_ + ThreadMap::Delta::kColumn * column) < extent_col_);
            }

            cutlass::arch::global_load<
              AccessType, 
              sizeof(AccessType)
            >(
              frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
              (void *)(byte_pointer + offset_modes_m + offset_modes_n + byte_offset),
              guard
            );

            if (Layout::kRank == 2) {
              offset_modes_n += params_.rank2_inc_col;
            }
          }

          if (Layout::kRank == 2) {
            offset_modes_m += params_.rank2_inc_row;
          }
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {

    load_with_byte_offset(frag, 0);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) {
    uint8_t *byte_pointer = byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        int row_begin = thread_start_row_ + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;
        int64_t offset_modes_m = row_begin * params_.stride_m[0];

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          // 
          // Compute coordinate and decompose into M modes
          //

          int coord_m = row * ThreadMap::Delta::kRow + row_begin;

          Coord<Layout::kRank / 2, Index> modes_m;

          if (Layout::kRank > 2) {
            if (kBigEndian) {
              modes_m = CoordinateDecomposition<Layout::kRank / 2>(coord_m, params_.divmod_m);
            } else {
              modes_m = CoordinateDecompositionLittleEndian<Layout::kRank / 2>(coord_m, params_.divmod_m);
            }

            offset_modes_m = dot(modes_m, params_.stride_m);
          }

          //
          // Compute the offset due to modes M
          //

          bool row_guard = (coord_m < extent_row_);
          int64_t offset_modes_n = thread_start_column_ * params_.stride_n[0];

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            // 
            // Compute coordinate and decompose into N modes
            //
            
            if (Layout::kRank > 2) {
              offset_modes_n = offset_modes_n_[column];
            } 

            //
            // Compute the pointer and access
            //
            bool guard;
            if (Layout::kRank > 2) {            
              guard = row_guard && mask_.predicates[column];
            } else {
              guard = (coord_m < extent_row_) && ((thread_start_column_ + ThreadMap::Delta::kColumn * column) < extent_col_);
            }

            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                (void *)(byte_pointer + offset_modes_m + offset_modes_n + byte_offset),
                guard);

            if (Layout::kRank == 2) {
              offset_modes_n += params_.rank2_inc_col;
            }
          }

          if (Layout::kRank == 2) {
            offset_modes_m += params_.rank2_inc_row;
          }
        }
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {

    store_with_byte_offset(frag, 0);
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorAffineRankN &operator++() {

    ++state_[0];
    thread_start_row_ += ThreadMap::Shape::kRow;
    
    if (state_[0] == ThreadMap::Count::kRow) {

      state_[0] = 0;
      ++state_[1];

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * 
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];

        thread_start_row_ += ThreadMap::Count::kGroup * 
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
        }
      }
    }

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) {
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
