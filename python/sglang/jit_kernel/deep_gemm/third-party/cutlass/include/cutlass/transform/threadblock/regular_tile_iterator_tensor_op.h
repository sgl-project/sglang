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
    \brief Templates implementing storing of tiles from pitch-linear rank=2 tensors. 
*/

#pragma once

#include "cutlass/transform/threadblock/regular_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                          Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// This iterator is specialized for an access size that is 128 bits in length.
    static int const kAccessSizeInBits = 128;

    static_assert(
      sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits,
      "This iterator requires a policy whose access size is 128bs");
  };

private:

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = RegularTileAccessIterator<Shape, Element, Layout,
                                                       kAdvanceRank, ThreadMap>;

private:

  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : address_iterator_(ref, thread_id) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    address_iterator_.add_tile_offset({0, 1});
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    address_iterator_.add_tile_offset(coord);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, Index byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char const *byte_ptr = reinterpret_cast<char const *>(address_iterator_.get()) + byte_offset;
        AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);

        frag_ptr[access_idx] = *access_ptr;
        ++address_iterator_;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, Index byte_offset) {  
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
        AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

        *access_ptr = frag_ptr[access_idx];
        ++address_iterator_;
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::ColumnMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::RowMajorTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                  Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }
  
  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for crosswise arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<Shape_, Element_,
                          layout::TensorOpMultiplicandCrosswise<
                              sizeof_bits<Element_>::value, Crosswise>,
                          AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Crosswise>;

  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess ==
                      kAccessSizeInBits,
                  "This iterator requires a policy whose access size is 128bs");
  };

 private:
  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  /// Fragment object to be loaded or stored
  using Fragment =
      Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = RegularTileAccessIterator<Shape, Element, Layout,
                                                       kAdvanceRank, ThreadMap>;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : address_iterator_(ref, thread_id) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    address_iterator_.add_tile_offset({1, 0});
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    address_iterator_.add_tile_offset(coord);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;
        frag_ptr[access_idx] = *(address_iterator_.get() + pointer_offset);
        ++address_iterator_;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, Index byte_offset) {  
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
        AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

        *access_ptr = frag_ptr[access_idx];
        ++address_iterator_;
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<Shape_, Element_,
                          layout::ColumnMajorTensorOpMultiplicandCrosswise<
                              sizeof_bits<Element_>::value, Crosswise>,
                          AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for column-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

 public:
  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<Shape_, Element_,
                          layout::RowMajorTensorOpMultiplicandCrosswise<
                              sizeof_bits<Element_>::value, Crosswise>,
                          AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for row-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

 public:
  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for k interleaved arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank, typename ThreadMap_, int InterleavedK, int Alignment>
class RegularTileIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandRowMajorInterleaved<sizeof_bits<Element_>::value,
                                                    InterleavedK>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandRowMajorInterleaved<sizeof_bits<Element_>::value,
                                                      InterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess ==
                      kAccessSizeInBits,
                  "This iterator requires a policy whose access size is 128bs");
  };

 private:

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  /// Fragment object to be loaded or stored
  using Fragment =
      Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = RegularTileAccessIterator<Shape, Element, Layout,
                                                       kAdvanceRank, ThreadMap>;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
       : address_iterator_(ref, thread_id) {}
 
  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    address_iterator_.add_pointer_offset(Shape::kCount);
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    address_iterator_.add_pointer_offset(coord.contiguous() * Shape::kCount);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;
        frag_ptr[access_idx] = *(address_iterator_.get() + pointer_offset);
        ++address_iterator_;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;
        *(address_iterator_.get() + pointer_offset) = frag_ptr[access_idx];
        ++address_iterator_;
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for k interleaved arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///

template <typename Shape_, typename Element_, int AdvanceRank, typename ThreadMap_, int InterleavedK, int Alignment>
class RegularTileIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandColumnMajorInterleaved<sizeof_bits<Element_>::value,
                                             InterleavedK>,
    AdvanceRank, ThreadMap_, Alignment> {

 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandColumnMajorInterleaved<sizeof_bits<Element_>::value,
                                                         InterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    cutlass::MatrixShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::TensorOpMultiplicandRowMajorInterleaved<sizeof_bits<Element_>::value, InterleavedK>,
    (kAdvanceRank == 1 ? 0 : 1),
    ThreadMap
  >;

 public:
  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

 private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
       : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.strided(), coord.contiguous()});
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
