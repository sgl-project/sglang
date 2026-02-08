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
    \brief Describes the lane policy used by warp-level matrix multiply operators targeting SIMT
      instructions
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"

#include "cutlass/layout/matrix.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterates over operands to warp-level matrix multiply operations targeting SIMT instructions
///
/// concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Operand identity
  Operand Operand,
  /// Data type of A elements
  typename Element_,
  /// Layout of operand
  typename Layout_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension - used in sliced-K
  int PartitionsK = 1,
  /// Group Size along kPartition - used in sliced-K
  int PartitionGroupSize = 1
>
class MmaSimtTileIterator;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for A operands of column-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension - used in sliced-K
  int PartitionsK,
  /// Group Size along kPartition - used in sliced-K
  int PartitionGroupSize
>
class MmaSimtTileIterator<Shape_, Operand::kA, Element_, layout::ColumnMajor, Policy_, PartitionsK, PartitionGroupSize> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of policy
  using Layout = layout::ColumnMajor;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  //
  // Derived quantities
  //

  static_assert(!(Shape::kRow % Policy::WarpShape::kRow), 
    "The warp-level GEMM M size must be divisible by the number of threads arranged along the M dimension.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kRow > 0, "Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Shape::kRow / Policy::WarpShape::kRow > 0, "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow / Policy::WarpShape::kRow,
    Shape::kColumn
  >;

  static_assert(!(ThreadShape::kRow % Policy::LaneMmaShape::kM), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kM,
    ThreadShape::kColumn
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

private:

  /// Internal reference
  cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kM>, layout::ColumnMajor> ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(Policy::LaneMmaShape::kM, 0);

    ref.add_coord_offset(lane_offset);

    ref_.reset(
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(ref.data()),
      ref.stride(0) / Policy::LaneMmaShape::kM);
  }
  

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    ref_.add_coord_offset({
      coord.row() * Shape::kRow / Policy::LaneMmaShape::kM, 
      coord.column() * Shape::kColumn});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator++() {

    ref_.add_coord_offset({0, Shape::kColumn});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator--() {

    ref_.add_coord_offset({0, -Shape::kColumn});

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator. (vector loads)
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
    Array<Element, Policy::LaneMmaShape::kM> *dst_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {

        // This logic has been replaced with calls to inline PTX to guarantee vectorization.
        #if 0
        dst_ptr[m + k * Iterations::kRow] = 
          *(ref_.data() + ref_.offset({m * Policy::WarpShape::kRow, k}) + pointer_offset / Policy::LaneMmaShape::kM);
        #endif

        auto ptr = ref_.data() + ref_.offset({m * Policy::WarpShape::kRow, k}) + pointer_offset / Policy::LaneMmaShape::kM;
        arch::shared_load(dst_ptr[m + k * Iterations::kRow], ptr);
      }
    }
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
    
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {
    
    Array<Element, Policy::LaneMmaShape::kM> const *src_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kN; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kM; ++m) {
        *(ref_.data() + ref_.offset(m * Policy::WarpShape::kM, k) + pointer_offset / Policy::LaneMmaShape::kM) = 
          src_ptr[m + k * Iterations::kM];
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for A operands of row-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension - used in sliced-K
  int PartitionsK,
  /// Group Size along kPartition - used in sliced-K
  int PartitionGroupSize
>
class MmaSimtTileIterator<Shape_, Operand::kA, Element_, layout::RowMajor, Policy_, PartitionsK, PartitionGroupSize> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of policy
  using Layout = layout::RowMajor;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  //
  // Derived quantities
  //

  static_assert(!(Shape::kRow % Policy::WarpShape::kRow), 
    "The warp-level GEMM M size must be divisible by the number of threads arranged along the M dimension.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kRow > 0, "Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Shape::kRow / Policy::WarpShape::kRow > 0, "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow / Policy::WarpShape::kRow,
    Shape::kColumn
  >;

  static_assert(!(ThreadShape::kRow % Policy::LaneMmaShape::kM), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads (scalar loads)
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kM,
    ThreadShape::kColumn
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

private:

  /// Internal reference
  cutlass::TensorRef<Element, layout::RowMajor> ref_;

  /// Extent of tensor
  MatrixCoord extent_;

  /// Origin
  MatrixCoord origin_;

  /// Used to conditionally enable extents checking
  bool divisible_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator() : divisible_(true) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ) : extent_(Shape::kRow, Shape::kColumn), divisible_ (true) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(Policy::LaneMmaShape::kM, 0);

    origin_ = lane_offset;

    ref.add_coord_offset(lane_offset);

    ref_.reset(ref.data(), ref.stride(0));

  }
  
  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref,
    TensorCoord extent, 
    int lane_id
  ) : extent_(extent), divisible_ (false) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(Policy::LaneMmaShape::kM, 0);

    origin_ = lane_offset;
    
    ref.add_coord_offset(lane_offset);

    ref_.reset(ref.data(), ref.stride(0));

  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    TensorCoord coord_offset(
      coord.row() * Shape::kRow, 
      coord.column() * Shape::kColumn);
    
    origin_ += coord_offset;

    ref_.add_coord_offset(coord_offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator++() {

    ref_.add_coord_offset({0, Shape::kColumn});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator--() {

    ref_.add_coord_offset({0, -Shape::kColumn});

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator. (scalar loads)
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Policy::LaneMmaShape::kM; i++) {
          
          MatrixCoord offset(m * Policy::WarpShape::kRow * Policy::LaneMmaShape::kM + i, k);
            
          MatrixCoord access_coord = origin_ + offset;

          int frag_idx = m * Policy::LaneMmaShape::kM + i + k * Iterations::kRow;

          if (divisible_ || 
              (access_coord.row() < extent_.row() && access_coord.column() < extent_.column())) {
          
            frag[frag_idx] = *(ref_.data() + ref_.offset(offset) + pointer_offset);
          }
          else {
            frag[frag_idx] = Element();
          }
        }
      }
    }
  }
  /// Loads a fragment from memory at the location pointed to by the iterator. 
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
    
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Policy::LaneMmaShape::kM; i++) {

          *(ref_.data() + ref_.offset(m * Policy::WarpShape::kM * Policy::LaneMmaShape::kM + i, k) + pointer_offset) = 
            frag[m * Policy::LaneMmaShape::kM + i + k * Iterations::kM];
        }
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for B operands of row-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK,
  /// Group Size along kPartition - used in sliced-K
  int PartitionGroupSize
>
class MmaSimtTileIterator<Shape_, Operand::kB, Element_, layout::RowMajor, Policy_, PartitionsK, PartitionGroupSize> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of policy
  using Layout = layout::RowMajor;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  //
  // Derived quantities
  //

  static_assert(!(Shape::kColumn % Policy::WarpShape::kColumn), 
    "The warp-level GEMM N size must be divisible by the number of threads arranged along the N dimension.");
  
  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kColumn > 0, "Policy::WarpShape::kColumn must be greater than zero.");
  static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0, "Shape::kColumn / Policy::WarpShape::kColumn must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow,
    Shape::kColumn / Policy::WarpShape::kColumn
  >;

  static_assert(!(ThreadShape::kColumn % Policy::LaneMmaShape::kN), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow,
    ThreadShape::kColumn / Policy::LaneMmaShape::kN
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

protected:

  /// Internal reference
  cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kN>, layout::RowMajor> ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(0, Policy::LaneMmaShape::kN);

    ref.add_coord_offset(lane_offset);

    ref_.reset(
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(ref.data()),
      ref.stride(0) / Policy::LaneMmaShape::kN);
  }
  
  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    ref_.add_coord_offset({
      coord.row() * Shape::kRow, 
      coord.column() * Shape::kColumn / Policy::LaneMmaShape::kN});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator++() {

    ref_.add_coord_offset({Shape::kRow, 0});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator--() {

    ref_.add_coord_offset({-Shape::kRow, 0});

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator. (vector loads)
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    Array<Element, Policy::LaneMmaShape::kN> *dst_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kRow; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kColumn; ++n) {

        #if 0
        dst_ptr[n + k * Iterations::kColumn] = 
          *(ref_.data() + ref_.offset({k, n * Policy::WarpShape::kColumn}) + pointer_offset / Policy::LaneMmaShape::kN);
        #endif

        void const *ptr = ref_.data() + ref_.offset({k, n * Policy::WarpShape::kColumn}) + pointer_offset / Policy::LaneMmaShape::kN;
        arch::shared_load(dst_ptr[n + k * Iterations::kColumn], ptr);
      }
    }
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
  
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {

    Array<Element, Policy::LaneMmaShape::kN> const *src_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kM; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kN; ++n) {
        *(ref_.data() + ref_.offset({k, n * Policy::WarpShape::kN}) + pointer_offset / Policy::LaneMmaShape::kN) = 
          src_ptr[n + k * Iterations::kN];
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, Index pointer_offset) const {
    store_with_pointer_offset(frag, 0);
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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for B operands of column-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK,
  /// Group Size along kPartition - used in sliced-K
  int PartitionGroupSize
>
class MmaSimtTileIterator<Shape_, Operand::kB, Element_, layout::ColumnMajor, Policy_, PartitionsK, PartitionGroupSize> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of policy
  using Layout = layout::ColumnMajor;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  //
  // Derived quantities
  //

  static_assert(!(Shape::kColumn % Policy::WarpShape::kColumn), 
    "The warp-level GEMM N size must be divisible by the number of threads arranged along the N dimension.");
  
  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kColumn > 0, "Policy::WarpShape::kColumn must be greater than zero.");
  static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0, "Shape::kColumn / Policy::WarpShape::kColumn must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow,
    Shape::kColumn / Policy::WarpShape::kColumn
  >;

  static_assert(!(ThreadShape::kColumn % Policy::LaneMmaShape::kN), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow,
    ThreadShape::kColumn / Policy::LaneMmaShape::kN
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

private:

  /// Internal reference
  cutlass::TensorRef<Element, layout::ColumnMajor> ref_;

  /// Extent of tensor
  MatrixCoord extent_;

  /// Origin
  MatrixCoord origin_;

  /// Used to conditionally enable extents checking
  bool divisible_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(): divisible_(true) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ): extent_(Shape::kRow, Shape::kColumn), divisible_(true) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(0, Policy::LaneMmaShape::kN);

    origin_ = lane_offset;

    ref.add_coord_offset(lane_offset);

    ref_.reset(ref.data(), ref.stride(0));
  }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref,
    TensorCoord extent, 
    int lane_id
  ): extent_(extent), divisible_(false) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(0, Policy::LaneMmaShape::kN);

    origin_ = lane_offset;

    ref.add_coord_offset(lane_offset);

    ref_.reset(ref.data(), ref.stride(0));
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    TensorCoord coord_offset(
      coord.row() * Shape::kRow, 
      coord.column() * Shape::kColumn);

    origin_ += coord_offset;

    ref_.add_coord_offset(coord_offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator++() {

    ref_.add_coord_offset({Shape::kRow, 0});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator--() {

    ref_.add_coord_offset({-Shape::kRow, 0});

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator. (scalar loads)
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kRow; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kColumn; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Policy::LaneMmaShape::kN; ++i) {

          MatrixCoord offset(k, n * Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN + i);
            
          MatrixCoord access_coord = origin_ + offset;

          int frag_idx = n * Policy::LaneMmaShape::kN + i + k * Iterations::kColumn;

          if (divisible_ || 
              (access_coord.row() < extent_.row() && access_coord.column() < extent_.column())) {

            frag[frag_idx] = *(ref_.data() + ref_.offset(offset) + pointer_offset);
          }
          else {
            frag[frag_idx] = Element();
          }
        }
      }
    }
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
  
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {

    Array<Element, Policy::LaneMmaShape::kN> const *src_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kM; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kN; ++n) {
        *(ref_.data() + ref_.offset({k, n * Policy::WarpShape::kN}) + pointer_offset / Policy::LaneMmaShape::kN) = 
          src_ptr[n + k * Iterations::kN];
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, Index pointer_offset) const {
    store_with_pointer_offset(frag, 0);
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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for C operands of column-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_
>
class MmaSimtTileIterator<Shape_, Operand::kC, Element_, layout::ColumnMajor, Policy_> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of accumulators in memory
  using Layout = layout::ColumnMajor;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  //
  // Derived quantities
  //

  static_assert(
    (!(Shape::kRow % Policy::WarpShape::kRow)) && (!(Shape::kColumn % Policy::WarpShape::kColumn)),
    "Warp-level GEMM shape must be divisible by the arrangement of threads in the warp.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kRow > 0, "Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Policy::WarpShape::kColumn > 0, "Policy::WarpShape::kColumn must be greater than zero.");
  static_assert(Shape::kRow / Policy::WarpShape::kRow > 0, "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0, "Shape::kColumn / Policy::WarpShape::kColumn must be greater than zero.");

  /// Thraed-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow / Policy::WarpShape::kRow,
    Shape::kColumn / Policy::WarpShape::kColumn
  >;

  static_assert(
    (!(ThreadShape::kRow % Policy::LaneMmaShape::kM)) && (!(ThreadShape::kColumn % Policy::LaneMmaShape::kN)),
    "Warp-level GEMM shape must be divisible by the arrangement of threads in the warp.");
  
  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kM,
    ThreadShape::kColumn / Policy::LaneMmaShape::kN
  >;

  using Delta = MatrixShape<
    Policy::WarpShape::kRow * Policy::LaneMmaShape::kM,
    Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

private:

  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(Policy::LaneMmaShape::kM, Policy::LaneMmaShape::kN);

    ref_.add_coord_offset(lane_offset);
  }
  
  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    ref_.add_coord_offset({
      coord.row() * Shape::kRow, 
      coord.column() * Shape::kColumn});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator++() {

    ref_.add_coord_offset({Shape::kRow, 0});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator--() {

    ref_.add_coord_offset({-Shape::kRow, 0});

    return *this;
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to be loaded from memory
    Index pointer_offset) const {               ///< linear offset (in units of Element) when loading

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Iterations::kN; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {

        Array<Element, Policy::LaneMmaShape::kM> const *src_ptr = 
          reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> const *>(
            ref_.data() + pointer_offset + ref_.offset({0, mma_n * Delta::kN + n}));

        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < Iterations::kM; ++mma_m) {

          Array<Element, Policy::LaneMmaShape::kM> *dst_ptr = 
            reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(&frag) + 
            mma_m + Iterations::kM * (n + mma_n * Policy::LaneMmaShape::kN);

          *dst_ptr = src_ptr[mma_m * Policy::WarpShape::kM];
        }
      }
    }
  }
    
  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {
    
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {

        Array<Element, Policy::LaneMmaShape::kM> *dst_ptr= 
          reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(
            ref_.data() + pointer_offset + ref_.offset({0, mma_n * Delta::kColumn + n}));

        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {

          Array<Element, Policy::LaneMmaShape::kM> const *src_ptr = 
            reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> const *>(&frag) + 
            mma_m + Iterations::kRow * (n + mma_n * Policy::LaneMmaShape::kN);

          dst_ptr[mma_m * Policy::WarpShape::kRow] = *src_ptr;
        }
      }
    }
  }
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for C operands of row-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_
>
class MmaSimtTileIterator<Shape_, Operand::kC, Element_, layout::RowMajor, Policy_> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of accumulators in memory
  using Layout = layout::RowMajor;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  //
  // Derived quantities
  //

  static_assert(
    (!(Shape::kRow % Policy::WarpShape::kRow)) && (!(Shape::kColumn % Policy::WarpShape::kColumn)),
    "Warp-level GEMM shape must be divisible by the arrangement of threads in the warp.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kRow > 0, "Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Policy::WarpShape::kColumn > 0, "Policy::WarpShape::kColumn must be greater than zero.");
  static_assert(Shape::kRow / Policy::WarpShape::kRow > 0, "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0, "Shape::kColumn / Policy::WarpShape::kColumn must be greater than zero.");

  /// Thraed-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow / Policy::WarpShape::kRow,
    Shape::kColumn / Policy::WarpShape::kColumn
  >;

  static_assert(
    (!(ThreadShape::kRow % Policy::LaneMmaShape::kM)) && (!(ThreadShape::kColumn % Policy::LaneMmaShape::kN)),
    "Warp-level GEMM shape must be divisible by the arrangement of threads in the warp.");
  
  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kM,
    ThreadShape::kColumn / Policy::LaneMmaShape::kN
  >;

  using Delta = MatrixShape<
    Policy::WarpShape::kRow * Policy::LaneMmaShape::kM,
    Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

private:

  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(Policy::LaneMmaShape::kM, Policy::LaneMmaShape::kN);
    
    ref_.add_coord_offset(lane_offset);
  }
  
  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    ref_.add_coord_offset({
      coord.row() * Shape::kRow, 
      coord.column() * Shape::kColumn});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator++() {

    ref_.add_coord_offset({Shape::kRow, 0});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator--() {

    ref_.add_coord_offset({-Shape::kRow, 0});

    return *this;
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to be loaded from memory
    Index pointer_offset) const {               ///< linear offset (in units of Element) when loading

    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {

        Array<Element, Policy::LaneMmaShape::kN> const *src_ptr = 
          reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> const *>(
            ref_.data() + pointer_offset + ref_.offset({mma_m * Delta::kRow + m, 0}));

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {

          Array<Element, Policy::LaneMmaShape::kN> *dst_ptr = 
            reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(&frag) + 
            mma_n + Iterations::kColumn * (m + mma_m * Policy::LaneMmaShape::kM);

          *dst_ptr = src_ptr[mma_n * Policy::WarpShape::kColumn];
        }
      }
    }
  }
    
  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {
    
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {

        Array<Element, Policy::LaneMmaShape::kN> *dst_ptr = 
          reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(
            ref_.data() + pointer_offset + ref_.offset({mma_m * Delta::kRow + m, 0}));

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {

          Array<Element, Policy::LaneMmaShape::kN> const *src_ptr = 
            reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> const *>(&frag) + 
            mma_n + Iterations::kColumn * (m + mma_m * Policy::LaneMmaShape::kM);

          dst_ptr[mma_n * Policy::WarpShape::kColumn] = *src_ptr;
        }
      }
    }
  }
  
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for A operands of column-major-K interleaved layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK,
  /// Number of KGroups per kPartition
  int PartitionGroupSize
>
class MmaSimtTileIterator<Shape_, Operand::kA, Element_, layout::ColumnMajorInterleaved<4>, Policy_, PartitionsK, PartitionGroupSize> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of policy
  using Layout = layout::ColumnMajorInterleaved<4> ;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Iterleave factor
  static const int kInterleave = 4;
  
  /// Number of partitions along K dimension
  static const int kPartitionsK = PartitionsK;

  /// Number of KGroups per kPartition
  static const int kGroupPerTile = PartitionGroupSize / Shape::kColumn;

  //
  // Derived quantities
  //

  static_assert(!(Shape::kRow % Policy::WarpShape::kRow), 
    "The warp-level GEMM M size must be divisible by the number of threads arranged along the M dimension.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kRow > 0, "Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Shape::kRow / Policy::WarpShape::kRow > 0, "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow / Policy::WarpShape::kRow,
    Shape::kColumn
  >;

  static_assert(!(ThreadShape::kRow % Policy::LaneMmaShape::kM) && !(ThreadShape::kColumn % Policy::LaneMmaShape::kK), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kM,
    ThreadShape::kColumn / Policy::LaneMmaShape::kK
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

private:

  /// Internal reference
  cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kMK>, layout::ColumnMajorInterleaved<4>> ref_;

  /// group index within tile
  int k_group_idx_;

public:
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(Policy::LaneMmaShape::kM, 0);

    ref.add_coord_offset(lane_offset);

    k_group_idx_ = 0;
    ref_.reset(reinterpret_cast<Array<Element, Policy::LaneMmaShape::kMK> *>(ref.data()), ref.stride(0)/Policy::LaneMmaShape::kMK);
  }
  

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    ref_.add_coord_offset({
      coord.row() * Shape::kRow / Policy::LaneMmaShape::kMK, 
      coord.column() * Shape::kColumn});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator++() {

    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == kGroupPerTile) {
        k_group_idx_ = 0;
        add_tile_offset({0, kGroupPerTile * (kPartitionsK-1)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator--() {

    ref_.add_coord_offset({0, -Shape::kColumn});

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    Array<Element, Policy::LaneMmaShape::kMK > *dst_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kMK> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {

      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {

        dst_ptr[m + k * Iterations::kRow] = 
          *((ref_.data() + ref_.offset({m * Policy::WarpShape::kRow / kInterleave, 
                  k*Policy::LaneMmaShape::kK}) + pointer_offset / Policy::LaneMmaShape::kM));
      }
    }
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
    
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {
    
    Array<Element, Policy::LaneMmaShape::kMK> const *src_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kMK > *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kN; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kM; ++m) {
        *(ref_.data() + ref_.offset(m * Policy::WarpShape::kM, k) + pointer_offset / Policy::LaneMmaShape::kM) = 
          src_ptr[m + k * Iterations::kM];
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for B operands of row-major k-interleaved layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK,
  /// Number of KGroups per kPartition
  int PartitionGroupSize
>
class MmaSimtTileIterator<Shape_, Operand::kB, Element_, layout::RowMajorInterleaved<4>, Policy_, PartitionsK, PartitionGroupSize> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of policy
  using Layout = layout::RowMajorInterleaved<4>;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Interleave factor
  static const int kInterleave = 4;

  /// Number of partitions along K dimension
  static const int kPartitionsK = PartitionsK;

  /// Number of KGroups per kPartition
  static const int kGroupPerTile = PartitionGroupSize / Shape::kRow;

  //
  // Derived quantities
  //

  static_assert(!(Shape::kColumn % Policy::WarpShape::kColumn), 
    "The warp-level GEMM N size must be divisible by the number of threads arranged along the N dimension.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kColumn > 0, "Policy::WarpShape::kColumn must be greater than zero.");
  static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0, "Shape::kColumn / Policy::WarpShape::kColumn must be greater than zero.");

  /// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    Shape::kRow,
    Shape::kColumn / Policy::WarpShape::kColumn
  >;

  static_assert(!(ThreadShape::kColumn % Policy::LaneMmaShape::kN) && !(ThreadShape::kRow % Policy::LaneMmaShape::kK), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow / Policy::LaneMmaShape::kK,
    ThreadShape::kColumn / Policy::LaneMmaShape::kN
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;


private:

  /// Internal reference
  cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kKN>, layout::RowMajorInterleaved<4>> ref_;

  /// group index within tile
  int k_group_idx_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id) * 
      MatrixCoord(0, Policy::LaneMmaShape::kN);

    ref.add_coord_offset(lane_offset);

    k_group_idx_ = 0;

    ref_.reset(
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kKN> *>(ref.data()),
      ref.stride(0) / Policy::LaneMmaShape::kKN);
  }
  
  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    ref_.add_coord_offset({
      coord.row() * Shape::kRow, 
      coord.column() * Shape::kColumn / Policy::LaneMmaShape::kKN});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator++() {

    add_tile_offset({1, 0});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == kGroupPerTile) {
        k_group_idx_ = 0;
        add_tile_offset({kGroupPerTile * (kPartitionsK-1), 0});
      }
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaSimtTileIterator & operator--() {

    ref_.add_coord_offset({-Shape::kRow, 0});

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    Array<Element, Policy::LaneMmaShape::kKN> *dst_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kKN> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kRow; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kColumn; ++n) {
        dst_ptr[n + k * Iterations::kColumn] = 
          *(ref_.data() + ref_.offset({k * Policy::LaneMmaShape::kK, 
                n * Policy::WarpShape::kColumn / kInterleave}) + pointer_offset / Policy::LaneMmaShape::kN);
      }
    }
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
  
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {

    Array<Element, Policy::LaneMmaShape::kN> const *src_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kM; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kN; ++n) {
        *(ref_.data() + ref_.offset({k, n * Policy::WarpShape::kN}) + pointer_offset / Policy::LaneMmaShape::kN) = 
          src_ptr[n + k * Iterations::kN];
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, Index pointer_offset) const {
    store_with_pointer_offset(frag, 0);
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

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass
