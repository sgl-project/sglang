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
    \brief Defines iterators to load sparse meta data used by warp-level matrix multiply operations
   targeting Sparse Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
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
    int Threads,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1>
class SparseMmaTensorOpMetaTileIterator {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  static int const kSparse = 2;

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
    static_assert(
        !(Shape::kColumn % InstructionShape::kColumn),
        "Shape of warp-level Mma must be divisible by operator shape.");
    
    static int const kElementsPerAccess = 128 / sizeof_bits<Element>::value;

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = InstructionShape::kColumn;
    static int const kLdsmOpInner = 8 * kElementsPerAccess / kLdsmOpOuter;

    static_assert(!(Shape::kColumn % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kRow % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeColumn =
        InstructionShape::kColumn / kLdsmOpOuter;
    static int const LdsmShapeRow =
        ((4 / LdsmShapeColumn * kLdsmOpInner) > Shape::kRow)
            ? (Shape::kRow / kLdsmOpInner)
            : (4 / LdsmShapeColumn);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeRow, LdsmShapeColumn>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kRow / kLdsmOpInner / LdsmShapeRow,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kColumn / InstructionShape::kColumn;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  using AccessType = Array<Element, Policy::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
      Array<Element, Shape::kRow * InstructionShape::kColumn / kThreads>;

 private:

  /// Layout object storing stride values
  Index stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  SparseMmaTensorOpMetaTileIterator()
      : pointer_(nullptr),
        stride_(0),
        byte_offset_(0),
        k_group_idx_(0) {}

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  SparseMmaTensorOpMetaTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        stride_(ref.stride(0) / Policy::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {

    int access_contiguous = (lane_id % (Shape::kRow / Policy::kElementsPerAccess));
    int access_strided = (lane_id / (Shape::kRow / Policy::kElementsPerAccess));

    byte_offset_ = (access_contiguous + access_strided * stride_) *
                   sizeof_bits<Element>::value * Policy::kElementsPerAccess / 8;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  SparseMmaTensorOpMetaTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  SparseMmaTensorOpMetaTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int offset = tile_offset.row() * Shape::kRow +
                 tile_offset.column() * InstructionShape::kColumn * stride_ *
                     Policy::kElementsPerAccess;

    add_pointer_offset(offset);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  SparseMmaTensorOpMetaTileIterator &operator++() {
    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  SparseMmaTensorOpMetaTileIterator &operator--(){
    byte_offset_ -= stride_ * InstructionShape::kColumn *
                    sizeof_bits<Element>::value * Policy::kElementsPerAccess /
                    8;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE SparseMmaTensorOpMetaTileIterator &
  operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  SparseMmaTensorOpMetaTileIterator &operator-=(
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
    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
        reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_ +
            Policy::LdsmShape::kContiguous * Policy::kLdsmOpInner * c +
            Policy::LdsmShape::kStrided * s * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) +
                                      byte_offset + byte_offset_;

        cutlass::arch::ldsm<layout::RowMajor, Policy::LdsmShape::kCount>(
            fetch_ptr[access_idx], source_byte_ptr);
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
      tile_offset.contiguous() * Shape::kRow / Layout::kElementsPerAccess + 
      tile_offset.strided() * InstructionShape::kColumn * stride_;

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
    // no op
  }
};

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
