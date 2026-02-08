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
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"

#include "cutlass/platform/platform.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads>
class MmaVoltaTensorOpMultiplicandTileIterator;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand::kA, Element_,
    cutlass::layout::VoltaTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value>,
    InstructionShape_, OpDelta_, 32> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Shape of one individual LDS.128
    using LdsShape = layout::PitchLinearShape<
      32,
      4
    >;

    // LdsShapes are arranged in the strided direction in SMEM
    using LdsIterations = layout::PitchLinearShape<
      InstructionShape::kStrided / LdsShape::kStrided,
      Shape::kContiguous / LdsShape::kContiguous
    >;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount = 2;

  /// Pointer type used for accesses
  using AccessType = AlignedArray<Element, Layout::kElementsPerAccess>;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment = Array<Element, Shape::kContiguous *
                                     InstructionShape::kStrided / kThreads * 2>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0) {
    // swizzle patterns for operandA LDS are
    // 1. (tid[4] << 3) | (tid[2:0] ^ tid[4])
    // 2. (tid[4] << 3) | (tid[2:0] ^ tid[4] ^ 0b10010)

    int vec_row = (lane_id >> 4); // tid[4]
    int vec_col = ((lane_id & 4) >> 2); // tid[2]

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {

      if(i == 1) {
        vec_row |= 2;
      }
      int access_contiguous_idx = (vec_col << 2) | ((lane_id & 3) ^ vec_row);
      int access_contiguous = access_contiguous_idx;

      int access_strided = vec_row;
      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
        access_contiguous + access_strided * stride_;
    }

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    int strided_offset = tile_offset.strided();

    // To support 32x32 tile size
    if (Shape::kContiguous == Policy::LdsShape::kContiguous) {
      if (contiguous_offset % 2) {
        AccessType const *tmp_pointer = pointer_[0];
        pointer_[0] = pointer_[1];
        pointer_[1] = tmp_pointer;
      }
      contiguous_offset = contiguous_offset / 2 * 2;
    }

    int offset = (strided_offset * InstructionShape::kStrided) * stride_ *
                     Layout::kElementsPerAccess +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator++() {
    byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    AccessType * fetch_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsIterations::kContiguous;

        AccessType const *source_ptr = pointer_[s & 1] +
          Policy::LdsShape::kContiguous * c +
          Policy::LdsShape::kStrided * (s / 2) * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;
        fetch_ptr[access_idx] = *(reinterpret_cast<AccessType const*> (source_byte_ptr));
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no operation here
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_>

class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand::kB, Element_,
    cutlass::layout::VoltaTensorOpMultiplicandBCongruous<
        sizeof_bits<Element_>::value>,
    InstructionShape_, OpDelta_, 32> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kB;

    /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Shape of one individual LDS
    using LdsShape = layout::PitchLinearShape<
      32,
      4
    >;

    using LdsIterations = layout::PitchLinearShape<
      Shape::kContiguous / LdsShape::kContiguous,
      InstructionShape::kStrided / LdsShape::kStrided
    >;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = AlignedArray<Element, Layout::kElementsPerAccess>;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile, needs on more time number of registers
 using Fragment = Array<Element, Shape::kContiguous *
                                     InstructionShape::kStrided / kThreads * 2>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0) {

    // swizzle pattern is (tid & (3 << 3) | (tid[1:0] ^ tid[4:3]))
    int access_strided = (lane_id >> 3) & 0x3;
    int access_contiguous = ((lane_id ^ (lane_id >> 3)) & 0x3);

    pointer_ = reinterpret_cast<AccessType const *>(ref.data()) +
                access_contiguous + access_strided * stride_;

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    int strided_offset = tile_offset.strided();

    int offset = (strided_offset * InstructionShape::kStrided) * stride_ *
                     Layout::kElementsPerAccess +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator++() {
    byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    AccessType * fetch_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsIterations::kContiguous;

        AccessType const *source_ptr = pointer_ +
          Policy::LdsShape::kContiguous / Layout::kElementsPerAccess * c +
          Policy::LdsShape::kStrided * s * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;
        fetch_ptr[access_idx] = *(reinterpret_cast<AccessType const*> (source_byte_ptr));
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no operation here
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand::kA, Element_,
    cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value>,
    InstructionShape_, OpDelta_, 32> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaVoltaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::VoltaTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand::kB, Element_,
    cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
        sizeof_bits<Element_>::value>,
    InstructionShape_, OpDelta_, 32> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kB;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaVoltaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::VoltaTensorOpMultiplicandBCongruous<sizeof_bits<Element_>::value>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaVoltaTensorOpAccumulatorTileIterator {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {

    /// Volta Tensor Op uses 32x32 interleaved tile
    using InterleavedTile = MatrixShape<32, 32>;

    static_assert(!(Shape::kRow % InterleavedTile::kRow) && !(Shape::kColumn % InterleavedTile::kColumn),
      "Shape of warp-level Mma must be divisible by operator shape.");

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using TileIterations = MatrixShape<
      Shape::kRow / InterleavedTile::kRow,
      Shape::kColumn / InterleavedTile::kColumn
    >;

    using MmaIterations =
        MatrixShape<InterleavedTile::kRow / InstructionShape::kM,
                    InterleavedTile::kColumn / InstructionShape::kN>;
  };

private:

  // Assume accumulator tile is multipile interleaved 32x32 tile.
  static int const kElementsPerPartial = 4;
  using EleShapePerPatial = typename platform::conditional<
                              platform::is_same<Element, float>::value,
                              MatrixShape<2, 2>,
                              MatrixShape<1, 4> >::type;
  static int const kElementsPerMma = 8;
  static int const kAccumulatorPatials = 2;
  using QuadShapePerPatialMma = MatrixShape<4, 4>;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
    int accum_m, accum_n;

    if (platform::is_same<Element, float>::value) {
      // (quad[2],quad[0])+lane_in_quad[0]
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 + (lane_in_quad & 1);
      // (quad[1])+lane_in_quad[1]
      accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials +
                  (lane_in_quad & 2);
    } else {
      accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 + lane_in_quad; // (quad[2],quad[0])
      accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial * kAccumulatorPatials;
    }
    MatrixCoord lane_offset(accum_m, accum_n);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_PRAGMA_UNROLL
    for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn; ++tile_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int tile_m = 0; tile_m < Policy::TileIterations::kRow; ++tile_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
          CUTLASS_PRAGMA_UNROLL
          for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

            int mma_accum_start =
                (((tile_n * Policy::TileIterations::kRow + tile_m) *
                    Policy::MmaIterations::kColumn + mma_n) *
                     Policy::MmaIterations::kRow + mma_m) * 
                    kElementsPerMma;

           CUTLASS_PRAGMA_UNROLL
            for (int p = 0; p < kAccumulatorPatials; ++p) {
              CUTLASS_PRAGMA_UNROLL
              for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < EleShapePerPatial::kColumn; ++n) {
                  int accum_m = tile_m * Policy::InterleavedTile::kRow +
                                mma_m * QuadShapePerPatialMma::kRow + m * 2;
                  int accum_n = tile_n * Policy::InterleavedTile::kColumn + 
                                mma_n * QuadShapePerPatialMma::kColumn +
                                p * Policy::InterleavedTile::kColumn/2 + n;
                  int idx = mma_accum_start + p * kElementsPerPartial + 
                            m * EleShapePerPatial::kColumn + n;
                frag[idx] = offset_ref.at({accum_m, accum_n});
                }
              }
            }
          }
        }
      }
    }
  }
  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_HOST_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_HOST_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset

    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_PRAGMA_UNROLL
    for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn; ++tile_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int tile_m = 0; tile_m < Policy::TileIterations::kRow; ++tile_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
          CUTLASS_PRAGMA_UNROLL
          for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {

            int mma_accum_start =
                (((tile_n * Policy::TileIterations::kRow + tile_m) *
                    Policy::MmaIterations::kColumn + mma_n) *
                     Policy::MmaIterations::kRow + mma_m) * 
                    kElementsPerMma;

            CUTLASS_PRAGMA_UNROLL
            for (int p = 0; p < kAccumulatorPatials; ++p) {
              CUTLASS_PRAGMA_UNROLL
              for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < EleShapePerPatial::kColumn; ++n) {
                  int accum_m = tile_m * Policy::InterleavedTile::kRow +
                                mma_m * QuadShapePerPatialMma::kRow + m * 2;
                  int accum_n = tile_n * Policy::InterleavedTile::kColumn + 
                                mma_n * QuadShapePerPatialMma::kColumn +
                                p * Policy::InterleavedTile::kColumn/2 + n;
                  int idx = mma_accum_start + p * kElementsPerPartial + 
                            m * EleShapePerPatial::kColumn + n;
                  offset_ref.at({accum_m, accum_n}) = frag[idx];
                }
              }
            }
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_HOST_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  CUTLASS_HOST_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_HOST_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDS to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// KBlock size (in units of elements)
    int KBlock>
class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::VoltaTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, KBlock>,
    InstructionShape_, OpDelta_, 32> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaVoltaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// KBlock size
  static int const kKBlock = KBlock;

  /// Layout of source tile
  using Layout = cutlass::layout::VoltaTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kKBlock>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {

    /// Shape of one individual LDS instruction
    using LdsShape = layout::PitchLinearShape<1, 32>;

    /// Number and arrangement of LDSM instructions
    using LdsIterations = layout::PitchLinearShape<1, Shape::kStrided / 32>;

    /// Using LDS.128
    static int const kElementsPerAccess = 8;

    /// Contiguous elements per line
    static int const kContiguousElementsPerLine = 4;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
      Array<Element,
            Shape::kStrided * InstructionShape::kContiguous / kThreads * 2>;

 private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Crosswised elements are arranged in a SMEM line
  /// in units of AccessType
  Index line_size;

  /// Internal counter used to determine load addr offset 
  /// and when to swap higher 64bit with lower 64bit
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        stride_(0),
        line_size(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        stride_(ref.stride(0) * Policy::kElementsPerAccess),
        line_size((ref.stride(0) * Policy::kContiguousElementsPerLine) /
                  Policy::kElementsPerAccess),
        k_group_idx_(0),
        byte_offset_(0) {

    int quad = (lane_id / 4);
    int lane_in_quad = (lane_id % 4);
    int access_contiguous;

    if(kOperand == Operand::kA) {

      // swizzle id: tid[4]|tid[1:0]|(tid[2]^tid[4])
      access_contiguous = ((quad & 0x4) << 1) + ((lane_in_quad) << 1) +
                            ((quad & 0x1) ^ ((quad & 0x4) >> 2));
    } else {

      // swizzle id: tid[4]|tid[1:0]|tid[3]
      access_contiguous = ((quad & 0x4) << 1) + (lane_in_quad << 1) +
                            ((quad & 0x2) >> 1 ^ ((quad & 0x4) >> 2));
    }

    byte_offset_ = access_contiguous *
                   sizeof(Element) * Policy::kElementsPerAccess;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    int strided_offset = tile_offset.strided();
    k_group_idx_ = 0;

    pointer_ += contiguous_offset *
                    (InstructionShape::kContiguous /
                     Policy::kContiguousElementsPerLine) *
                    line_size +
                strided_offset * Shape::kStrided / 2;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator++() {
    k_group_idx_ = (k_group_idx_ + 1) % 8;

    if (k_group_idx_ == 4 || k_group_idx_ == 0) {
      byte_offset_ ^= 1 * sizeof(Element) * Policy::kElementsPerAccess;
    }

    pointer_ += line_size;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    AccessType * fetch_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsIterations::kContiguous;

        AccessType const *source_ptr = pointer_ +
          Policy::LdsShape::kContiguous * c * line_size +
          Policy::LdsShape::kStrided * s / 2;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;
        fetch_ptr[access_idx] = *(reinterpret_cast<AccessType const*> (source_byte_ptr));

        // swap higher 64bit and lower 64bit
        if (k_group_idx_ &  0x2) {
            uint64_t *low = reinterpret_cast<uint64_t *>(&frag) + access_idx * 2;
            uint64_t *high = reinterpret_cast<uint64_t *>(&frag) + access_idx * 2 + 1;
            uint64_t tmp = *low;
            *low = *high;
            *high = tmp;
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Policy::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group;
  }
};

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDS to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// KBlock size (in units of elements)
    int KBlock>
class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, KBlock>,
    InstructionShape_, OpDelta_, 32> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// KBlock size
  static int const kKBlock = KBlock;


  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kKBlock>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaVoltaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                 kKBlock>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDS to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// KBlock size (in units of elements)
    int KBlock>
class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, KBlock>,
    InstructionShape_, OpDelta_, 32> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// KBlock size
  static int const kKBlock = KBlock;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kKBlock>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaVoltaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::VoltaTensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                 kKBlock>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
  }
  
  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for 'TN' arrangement
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand_,
    /// Data type of A elements
    typename Element_,
    /// Layout of matrix operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads = 32,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1>
class MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  /// Basic check
  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaVoltaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Number of elements accessed per Shared Memory load
  static int const kElementsPerAccess = 4;

private:

  static int const kInterleavedTileRows = 32;
  static int const kInterleavedTileColumns = 32;
  static int const kInstructionsPerTile = 2;
  
  /// Rounded up instruction counts
  using TileCount = MatrixShape<
    Shape::kRow / kInterleavedTileRows,
    Shape::kColumn / kInterleavedTileColumns
  >;

  using FragmentCount = MatrixShape<
    TileCount::kRow * kInstructionsPerTile,
    TileCount::kColumn * kInstructionsPerTile
  >;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element, 
    (kOperand == Operand::kA ? FragmentCount::kRow : FragmentCount::kColumn) * kElementsPerAccess
  >;

  /// Memory access type
  using AccessType = AlignedArray<Element, kElementsPerAccess>;

private:

  /// Underlying tensor reference
  TensorRef ref_;

  /// Extent of tensor
  MatrixCoord extent_;

  /// Origin
  MatrixCoord origin_;

  /// Used to conditionally enable extents checking
  bool divisible_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner(): divisible_(true) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner(
    TensorRef const &ref, 
    int lane_id
  ): 
    ref_(ref), extent_(Shape::kRow, Shape::kColumn), divisible_(true) {

    int quad_id = lane_id / 4;
    int lane_in_quad = (lane_id % 4);
  
    if (kOperand == Operand::kA) {
      
      int row_idx = ((quad_id & 1) + ((quad_id & 4) / 2)) * 4 * kInstructionsPerTile + lane_in_quad;
      int col_idx = 0;

      origin_ = MatrixCoord(row_idx, col_idx);
    }
    else {

      int row_idx = 0;
      int col_idx = (quad_id / 2) * 4 * kInstructionsPerTile  + lane_in_quad;

      origin_ = MatrixCoord(row_idx, col_idx); 
    }

    ref_.add_coord_offset(origin_);
  }
  
  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner(
    TensorRef const &ref, 
    TensorCoord extent,
    int lane_id
  ): ref_(ref), extent_(extent), divisible_(false) {
  
    int quad_id = lane_id / 4;
    int lane_in_quad = (lane_id % 4);
  
    if (kOperand == Operand::kA) {
      
      int row_idx = ((quad_id & 1) + ((quad_id & 4) / 2)) * 4 * kInstructionsPerTile  + lane_in_quad;
      int col_idx = 0;

      origin_ = MatrixCoord(row_idx, col_idx);
    }
    else {

      int row_idx = 0;
      int col_idx = (quad_id / 2) * 4 * kInstructionsPerTile  + lane_in_quad;

      origin_ = MatrixCoord(row_idx, col_idx); 
    }

    #if defined(__CUDA_ARCH__)
    __syncthreads();
    #endif

    ref_.add_coord_offset(origin_);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner &add_pointer_offset(LongIndex offset) {

    ref_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner &add_tile_offset(TensorCoord const &tile_offset) {

    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
    origin_ += coord_offset;

    ref_.add_coord_offset(coord_offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner & operator++() {

    if (kOperand == Operand::kA) {
      add_tile_offset({0, 1});
    }
    else {
      add_tile_offset({1, 0});
    }    

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner & operator--() {
    
    if (kOperand == Operand::kA) {
      add_tile_offset({0, -1});
    }
    else {
      add_tile_offset({-1, 0});
    }    

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    AccessType const *access_ptr = reinterpret_cast<AccessType const *>(ref_.data());
    int ldm = ref_.stride()[0];

    if (kOperand == Operand::kA) {

      CUTLASS_PRAGMA_UNROLL
      for (int idx = 0; idx < FragmentCount::kRow; ++idx) {
        
        int tile_idx = idx / 2;
        int quad_idx = idx % 2;

        int row_offset = tile_idx * kInterleavedTileRows + quad_idx * 4;
        frag_ptr[idx] = access_ptr[row_offset * ldm / kElementsPerAccess];
      } 
    }
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int idx = 0; idx < FragmentCount::kColumn; ++idx) {

        int tile_idx = idx / 2;
        int quad_idx = idx % 2;

        int col_offset = tile_idx * kInterleavedTileColumns + quad_idx * 4;
        frag_ptr[idx] = access_ptr[col_offset * ldm / kElementsPerAccess];
      } 
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {

    load_with_pointer_offset(frag, byte_offset * 8 / sizeof_bits<Element>::value);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    
    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
  
    load_with_pointer_offset(frag, ref_.offset(coord_offset));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {

    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
  
    load_with_pointer_offset(frag, ref_.offset(coord_offset) + pointer_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {

    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
  
    load_with_pointer_offset(frag, ref_.offset(coord_offset) + byte_offset * 8 / sizeof_bits<Element>::value);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no operation
  }
};


/// Tile iterator specialized for 'NT' arrangement
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand_,
    /// Data type of A elements
    typename Element_,
    /// Layout of matrix operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads = 32,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1>
class MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  /// Basic check
  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaVoltaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Number of elements accessed per Shared Memory load
  static int const kElementsPerAccess = 4;

private:

  static int const kInterleavedTileRows = 32;
  static int const kInterleavedTileColumns = 32;
  static int const kInstructionsPerTile = 2;
  
  /// Rounded up instruction counts
  using TileCount = MatrixShape<
    Shape::kRow / kInterleavedTileRows,
    Shape::kColumn / kInterleavedTileColumns
  >;

  using FragmentCount = MatrixShape<
    TileCount::kRow * kInstructionsPerTile,
    TileCount::kColumn * kInstructionsPerTile
  >;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element, 
    (kOperand == Operand::kA ? FragmentCount::kRow : FragmentCount::kColumn) * kElementsPerAccess
  >;

  /// Memory access type
  using AccessType = AlignedArray<Element, kElementsPerAccess>;

private:

  /// Underlying tensor reference
  TensorRef ref_;

  /// Extent of tensor
  MatrixCoord extent_;

  /// Origin
  MatrixCoord origin_;

  /// Used to conditionally enable extents checking
  bool divisible_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter(): divisible_(true) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter(
    TensorRef const &ref, 
    int lane_id
  ): 
    ref_(ref), extent_(Shape::kRow, Shape::kColumn), divisible_(true) {

    int quad_id = lane_id / 4;
    int lane_in_quad = (lane_id % 4);
  
    if (kOperand == Operand::kA) {
      
      int row_idx = ((quad_id & 1) + ((quad_id & 4) / 2)) * 4 * kInstructionsPerTile;
      int col_idx = lane_in_quad;

      origin_ = MatrixCoord(row_idx, col_idx);
    }
    else {

      int row_idx = lane_in_quad;
      int col_idx = (quad_id / 2) * 4 * kInstructionsPerTile;

      origin_ = MatrixCoord(row_idx, col_idx); 
    }

    ref_.add_coord_offset(origin_);
  }
  
  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter(
    TensorRef const &ref, 
    TensorCoord extent,
    int lane_id
  ): ref_(ref), extent_(extent), divisible_(false) {
  
    int quad_id = lane_id / 4;
    int lane_in_quad = (lane_id % 4);
  
    if (kOperand == Operand::kA) {
      
      int row_idx = ((quad_id & 1) + ((quad_id & 4) / 2)) * 4 * kInstructionsPerTile;
      int col_idx = lane_in_quad;

      origin_ = MatrixCoord(row_idx, col_idx);
    }
    else {

      int row_idx = lane_in_quad;
      int col_idx = (quad_id / 2) * 4 * kInstructionsPerTile;

      origin_ = MatrixCoord(row_idx, col_idx); 
    }

    #if defined(__CUDA_ARCH__)
    __syncthreads();
    #endif

    ref_.add_coord_offset(origin_);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter &add_pointer_offset(LongIndex offset) {

    ref_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter &add_tile_offset(TensorCoord const &tile_offset) {

    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
    origin_ += coord_offset;

    ref_.add_coord_offset(coord_offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter & operator++() {

    if (kOperand == Operand::kA) {
      add_tile_offset({0, 1});
    }
    else {
      add_tile_offset({1, 0});
    }    

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter & operator--() {
    
    if (kOperand == Operand::kA) {
      add_tile_offset({0, -1});
    }
    else {
      add_tile_offset({-1, 0});
    }    

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    AccessType const *access_ptr = reinterpret_cast<AccessType const *>(ref_.data());
    int ldm = ref_.stride()[0];

    if (kOperand == Operand::kA) {

      CUTLASS_PRAGMA_UNROLL
      for (int idx = 0; idx < FragmentCount::kRow; ++idx) {
        
        int tile_idx = idx / 2;
        int quad_idx = idx % 2;

        int row_offset = tile_idx * kInterleavedTileRows;
        frag_ptr[idx] = access_ptr[row_offset / kElementsPerAccess + quad_idx];
      }
    }
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int idx = 0; idx < FragmentCount::kColumn; ++idx) {

        int tile_idx = idx / 2;
        int quad_idx = idx % 2;

        int col_offset = tile_idx * kInterleavedTileColumns;
        frag_ptr[idx] = access_ptr[col_offset / kElementsPerAccess + quad_idx];
      } 
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {

    load_with_pointer_offset(frag, byte_offset * 8 / sizeof_bits<Element>::value);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    
    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
  
    load_with_pointer_offset(frag, ref_.offset(coord_offset));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {

    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
  
    load_with_pointer_offset(frag, ref_.offset(coord_offset) + pointer_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {

    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
  
    load_with_pointer_offset(frag, ref_.offset(coord_offset) + byte_offset * 8 / sizeof_bits<Element>::value);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no operation
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
  Shape_, 
  Operand::kA, 
  Element_,
  cutlass::layout::RowMajor,
  InstructionShape_, 
  OpDelta_,
  32
> : public MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner<
  Shape_, Operand::kA, Element_, cutlass::layout::RowMajor, InstructionShape_, OpDelta_> {

public:
  using Base = MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner<
  Shape_, Operand::kA, Element_, cutlass::layout::RowMajor, InstructionShape_, OpDelta_> ;

  using TensorRef = typename Base::TensorRef;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): Base(ref, lane_id) { }

};

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
  Shape_, 
  Operand::kA, 
  Element_,
  cutlass::layout::ColumnMajor,
  InstructionShape_, 
  OpDelta_,
  32
> : public MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter<
  Shape_, Operand::kA, Element_, cutlass::layout::ColumnMajor, InstructionShape_, OpDelta_> {

public:
  using Base = MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter<
  Shape_, Operand::kA, Element_, cutlass::layout::ColumnMajor, InstructionShape_, OpDelta_> ;

  using TensorRef = typename Base::TensorRef;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): Base(ref, lane_id) { }

};

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand::kB, Element_,
    cutlass::layout::ColumnMajor,
    InstructionShape_, OpDelta_, 32
> : public MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner<
  Shape_, Operand::kB, Element_, cutlass::layout::ColumnMajor, InstructionShape_, OpDelta_> {

public:
  using Base = MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner<
  Shape_, Operand::kB, Element_, cutlass::layout::ColumnMajor, InstructionShape_, OpDelta_>;

  using TensorRef = typename Base::TensorRef;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): Base(ref, lane_id) { }
};

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
    Shape_, Operand::kB, Element_,
    cutlass::layout::RowMajor,
    InstructionShape_, OpDelta_, 32
> : public MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter<
  Shape_, Operand::kB, Element_, cutlass::layout::RowMajor, InstructionShape_, OpDelta_> {

public:
  using Base = MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter<
  Shape_, Operand::kB, Element_, cutlass::layout::RowMajor, InstructionShape_, OpDelta_>;

  using TensorRef = typename Base::TensorRef;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaVoltaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): Base(ref, lane_id) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
