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
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last "residue" tile
    first, with the objective of minimizing predicate mask updates during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"

#include "cutlass/transform/threadblock/regular_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

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

    ///< Number of pointers
    static int const kPointerCount = (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);
  };


private:

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType * pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0) {

    layout::PitchLinearCoord thread_offset_base = ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {

      // This is the offset of a thread within a threadblock tile for a specific pointer
      // (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
        thread_offset_base + layout::PitchLinearCoord{0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data() + ref.offset(thread_offset_in_threadblock_tile));
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {

    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    add_pointer_offset((kAdvanceRank ? Shape::kStrided * stride_ * Layout::kElementsPerAccess : Shape::kContiguous));

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
    add_pointer_offset(
      coord.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess +
      coord.strided() * Shape::kStrided * stride_ * Layout::kElementsPerAccess
    );
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    Index vec_pointer_offset = pointer_offset / ThreadMap::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[s & 1];
      int stride_idx = (s & ~1);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
            c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess +
            vec_pointer_offset;

        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char const *access_byte_ptr = reinterpret_cast<char const *>(access_ptr + access_offset);

        frag_ptr[access_idx] = *reinterpret_cast<AccessType const *>(access_byte_ptr + byte_offset_);
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
  void store_with_pointer_offset(
    Fragment const &frag,
    Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    Index vec_pointer_offset = pointer_offset / ThreadMap::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[s & 1];
      int stride_idx = (s & ~1);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
          c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess +
          vec_pointer_offset;

        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char *access_byte_ptr = reinterpret_cast<char *>(access_ptr + access_offset);

        *reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_) = frag_ptr[access_idx];
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Tile Iterator specialized for column-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap_>;

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


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::RowMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap_>;

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
/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

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

    ///< Number of pointers
    static int const kPointerCount = (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);
  };


private:

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType * pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

public:

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0) {

    layout::PitchLinearCoord thread_offset_base = ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {

      // This is the offset of a thread within a threadblock tile for a specific pointer
      // (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
        thread_offset_base + layout::PitchLinearCoord{0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data() + ref.offset(thread_offset_in_threadblock_tile));
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {

    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    add_pointer_offset((kAdvanceRank ? Shape::kStrided * stride_ * Layout::kElementsPerAccess : Shape::kContiguous));

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
    add_pointer_offset(
      coord.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess +
      coord.strided() * Shape::kStrided * stride_ * Layout::kElementsPerAccess
    );
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    Index vec_pointer_offset = pointer_offset / ThreadMap::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[s & 1];
      int stride_idx = (s & ~1);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
            c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess +
            vec_pointer_offset;

        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char const *access_byte_ptr = reinterpret_cast<char const *>(access_ptr + access_offset);

        frag_ptr[access_idx] = *reinterpret_cast<AccessType const *>(access_byte_ptr + byte_offset_);
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
  void store_with_pointer_offset(
    Fragment const &frag,
    Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    Index vec_pointer_offset = pointer_offset / ThreadMap::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[s & 1];
      int stride_idx = (s & ~1);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
          c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess +
          vec_pointer_offset;

        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char *access_byte_ptr = reinterpret_cast<char *>(access_ptr + access_offset);

        *reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_) = frag_ptr[access_idx];
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::ColumnMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap_>;

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


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
  Shape_,
  Element_,
  layout::RowMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
  AdvanceRank,
  ThreadMap_,
  Alignment> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap_>;

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


/// Tile iterator specialized for crosswise arrangements for TensorOps.
///
/// Volta TN SMEM layout is a little diffrent:
/// Crosseised elements will be stored in a line, while contiguous elements
/// sre stored in line-by-line.
/// Padding is used to reduce SMEM bank conflicts.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<
    Shape_, Element_,
    layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                               Shape_::kContiguous>,
    AdvanceRank, ThreadMap_, Alignment> {

 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                 Shape::kContiguous>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {

    ///< Number of pointers
    static int const kPointerCount = (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);

    /// Iterations for the kElementsPerAccess of ThreadMap
    static int const kIterarionsPerAccess =
        ThreadMap::kElementsPerAccess / Layout::kElementsPerAccess;

    /// Contiguous elements per line
    static int const kContiguousElementsPerLine = 4;
  };

 private:
  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  /// Fragment object to be loaded or stored
  using Fragment =
      Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// The crosswised elements will be stored in a line.
  /// line_size is size of crosswised dimension plus padding.
  /// in units of AccessType
  Index line_size;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;


 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : line_size(ref.stride(0) * Detail::kContiguousElementsPerLine / Layout::kElementsPerAccess),
        byte_offset_(0) {

    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(
          ref.data() + ref.offset(thread_offset_in_threadblock_tile));
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    // (Shape::kContiguous/Layout::kElementsPerAccess)*
    //   line_size * Layout::kElementsPerAccess
    add_pointer_offset(Shape::kContiguous * line_size);
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
    add_pointer_offset((coord.contiguous() * (Shape::kContiguous / Layout::kElementsPerAccess) *
                       line_size + coord.strided() * Shape::kStrided) *
                       Layout::kElementsPerAccess);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    Index vec_pointer_offset = pointer_offset / Layout::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      AccessType *access_ptr = pointer_[(s & 1) ^ (s / 2)];

      access_ptr += 16 * (s / 2);

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < Detail::kIterarionsPerAccess; ++i) {

          int access_offset = 
            c * ThreadMap::Delta::kContiguous / Detail::kContiguousElementsPerLine * line_size +
            vec_pointer_offset + i * line_size;

          int access_idx = (c + s * ThreadMap::Iterations::kContiguous) *
            Detail::kIterarionsPerAccess + i;

          char const *access_byte_ptr = reinterpret_cast<char const*>(access_ptr + access_offset);

          frag_ptr[access_idx] = *reinterpret_cast<AccessType const *>(
              access_byte_ptr + byte_offset_);
        }
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

    Index vec_pointer_offset = pointer_offset / Layout::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      AccessType *access_ptr = pointer_[(s & 1) ^ ((s >> 1) & 1)];

      access_ptr += 16 * (s / 2) + vec_pointer_offset;

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < Detail::kIterarionsPerAccess; ++i) {

          int access_offset = 
            c * ThreadMap::Delta::kContiguous / Detail::kContiguousElementsPerLine * line_size + i * line_size;

          int access_idx = (c + s * ThreadMap::Iterations::kContiguous) *
            Detail::kIterarionsPerAccess + i;

          char *access_byte_ptr = reinterpret_cast<char *>(access_ptr + access_offset);

          *reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_) =
              frag_ptr[access_idx];
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<Shape_, Element_,
                          layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
                              sizeof_bits<Element_>::value, Shape_::kRow>,
                          AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for column-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Shape::kRow>;
  static int const kAdvanceRank = AdvanceRank;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Shape::kRow>,
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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,  
  int Alignment
>
class RegularTileIterator<Shape_, Element_,
                          layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
                              sizeof_bits<Element_>::value, Shape_::kColumn>,
                          AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for row-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Shape::kColumn>;
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
      layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                 Shape::kColumn>,
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


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass
