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

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/epilogue/warp/simt_policy.h"

#define CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES 1

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape,     ///< shape of warp-level GEMM (concept: MatrixShape)
  typename Operator,      ///< matrix multiply operation (concept: arch::Mma)
  typename Element,       ///< data type of element to be written
  typename Layout,        ///< target shared memory layout
  typename MmaSimtPolicy          ///< policy defining lane arrangement (concept: MmaSimtPolicy)
>
class TileIteratorSimt;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
  typename Operator_,      ///< matrix multiply operation (concept: arch::Mma)
  typename Element_,       ///< data type of element to be written
  typename MmaSimtPolicy_         ///< policy defining lane arrangement (concept: MmaSimtPolicy)
>
class TileIteratorSimt<WarpShape_, Operator_, Element_, layout::RowMajor, MmaSimtPolicy_> {
public:

  using WarpShape = WarpShape_;
  using Operator = Operator_;
  using Element = Element_;
  using Layout = layout::RowMajor;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = SimtPolicy<WarpShape, Operator, Layout, MmaSimtPolicy_>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    typename Operator::ElementC, 
    Policy::kElementsPerIteration>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<
    typename Operator::ElementC, 
    Policy::kAccumulatorElementCount>;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    4 * Policy::kElementsPerAccess
#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
    + 1
#endif
  >;

private:

#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
  /// Storage type for accessing memory
  using AccessType = AlignedArray<
    Element, 
    1
  >;

#else
  /// Storage type for accessing memory
  using AccessType = AlignedArray<
    Element, 
    Policy::kElementsPerAccess
  >;
#endif

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorSimt(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorSimt(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / AccessType::kElements) { 

    auto lane_layout = Policy::MmaSimtPolicy::get_lane_layout();
    MatrixCoord lane_offset = lane_layout.inverse(lane_id);

    pointer_ += layout_({
      lane_offset.row(),
      lane_offset.column() * Policy::kElementsPerAccess / int(AccessType::kElements)
    });
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorSimt & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / AccessType::kElements;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorSimt & add_tile_offset(TensorCoord const &tile_offset) {

    pointer_ += layout_({
      tile_offset.row() * Shape::kRow, 
      (tile_offset.column() * Shape::kColumn / int(AccessType::kElements))
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorSimt & operator+=(TensorCoord const &tile_offset) {

    add_tile_offset(tile_offset);
    
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
      // de-vectorized stores
      using ScalarAccessType = AlignedArray<Element, 1>;
      ScalarAccessType const *scalarFragPtr = reinterpret_cast<ScalarAccessType const *>(&frag);
      ScalarAccessType *scalarPointer = reinterpret_cast<ScalarAccessType *>(pointer_) + pointer_offset;

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Policy::kAccessesPerIteration; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::kElementsPerAccess; s++) {
          scalarPointer[n * Policy::MmaSimtPolicy::WarpShape::kColumn * Policy::kElementsPerAccess + s] = scalarFragPtr[n * Policy::kElementsPerAccess + s];
        }
      }
#else
    // original vector stores
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::kAccessesPerIteration; ++n) {
      pointer_[n * Policy::MmaSimtPolicy::WarpShape::kColumn + pointer_offset / int(AccessType::kElements)] = frag_ptr[n];
    }
#endif
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::kAccessesPerIteration; ++n) {
      frag_ptr[n] = pointer_[n * Policy::MmaSimtPolicy::WarpShape::kColumn + pointer_offset / int(AccessType::kElements)];
    }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
          typename Operator_,      ///< matrix multiply operation (concept: arch::Mma)
          typename Element_,       ///< data type of element to be written
          typename Layout_,         ///< target shared memory layout
          typename MmaSimtPolicy_  ///< policy defining lane arrangement (concept: MmaSimtPolicy)
          >
class TileIteratorSimtDirectConv {
 public:

  using WarpShape = WarpShape_;
  using Operator = Operator_;
  using Element = Element_;
  using Layout = layout::RowMajor;

  using TensorRef = TensorRef<Element, Layout>;  ///< Tensor Reference object
  using TensorCoord = MatrixCoord;               ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = SimtPolicy<WarpShape, Operator, Layout, MmaSimtPolicy_>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<Policy::kRowsPerIteration, WarpShape::kN>;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<typename Operator::ElementC, Policy::kElementsPerIteration>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<typename Operator::ElementC, Policy::kAccumulatorElementCount>;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Padding quantity
  using Padding = MatrixShape<0,
                              0
                              >;

private:
  /// Storage type for accessing memory
  using AccessType = AlignedArray<
    Element, 
    Policy::kElementsPerAccess
  >;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Base smem offset;
  Index base_smem_address_;

 public:
  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirectConv() : pointer_(nullptr) {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirectConv(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / AccessType::kElements) {

    auto lane_layout = Policy::MmaSimtPolicy::get_lane_layout();
    MatrixCoord lane_offset = lane_layout.inverse(lane_id);

    pointer_ += layout_({
      lane_offset.row(),
      lane_offset.column() * Policy::kElementsPerAccess / int(AccessType::kElements)
    });
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirectConv & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / AccessType::kElements;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirectConv & add_tile_offset(TensorCoord const &tile_offset) {

    pointer_ += layout_({
      tile_offset.row() * Shape::kRow, 
      (tile_offset.column() * Shape::kColumn / int(AccessType::kElements))
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirectConv & operator+=(TensorCoord const &tile_offset) {

    add_tile_offset(tile_offset);
    
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    // original vector stores
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
    AccessType * load_pointer_ = reinterpret_cast<AccessType *>(reinterpret_cast<uint8_t *>(pointer_) + base_smem_address_);
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::kAccessesPerIteration; ++n) {
      load_pointer_[n * Policy::MmaSimtPolicy::WarpShape::kColumn + pointer_offset / int(AccessType::kElements)] = frag_ptr[n];
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::kAccessesPerIteration; ++n) {
      frag_ptr[n] = pointer_[n * Policy::MmaSimtPolicy::WarpShape::kColumn + pointer_offset / int(AccessType::kElements)];
    }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address){
    base_smem_address_ = address;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Template for reading and writing tiles of accumulators to shared memory
template <typename WarpShape_,               ///< shape of warp-level GEMM (concept: GemmShape)
          typename ThreadOutputShape_,       /// Size of the matrix to load (concept: TensorNHWC)
          typename ThreadBlockOutputShape_,  /// Size of the matrix to load (concept: TensorNHWC)
          typename Operator_,                ///< matrix multi ply operation (concept: arch::Mma)
          typename Element_,                 ///< data type of element to be written
          typename Layout_,                  ///< target shared memory layout
          typename MmaSimtPolicy_            ///< policy defining lane arrangement (concept: MmaSimtPolicy)
          >
class TileIteratorSimtDirect2dConv {
 public:
  using WarpShape = WarpShape_;
  using ThreadOutputShape = ThreadOutputShape_;
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;
  using Operator = Operator_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  using MmaSimtPolicy = MmaSimtPolicy_;

  using TensorRef = TensorRef<Element, Layout>;  ///< Tensor Reference object
  using TensorCoord = MatrixCoord;               ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  // Thread-level shape of a fragment
  using ThreadShape = MatrixShape<ThreadOutputShape::kNHW, ThreadOutputShape::kC>;

  static_assert(!(ThreadShape::kColumn % MmaSimtPolicy::LaneMmaShape::kN),
                "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  using ThreadTileCount = MatrixShape<ThreadBlockOutputShape::kH / ThreadOutputShape::kH,
                                      ThreadBlockOutputShape::kW / ThreadOutputShape::kW>;

  using Iterations =
      MatrixShape<ThreadShape::kRow, ThreadShape::kColumn / MmaSimtPolicy::LaneMmaShape::kN>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = typename Operator::FragmentC;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = AccumulatorTile;

  /// Padding quantity
  using Padding = MatrixShape<0, 0>;

 private:
  // Storage type for accessing memory
  using AccessType = AlignedArray<Element, MmaSimtPolicy::LaneMmaShape::kN>;
  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Base smem offset;
  Index base_smem_address_;

 public:
  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirect2dConv() : pointer_(nullptr) {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirect2dConv(TensorRef const &ref, unsigned thread_id, unsigned lane_id)
      : pointer_(reinterpret_cast<AccessType *>(ref.data())),
        layout_(ref.stride()[0] / AccessType::kElements) {
  
    auto lane_layout = MmaSimtPolicy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id);

    // Get base HW offset of current threads
    const int threadgroup = thread_id / (ThreadBlockOutputShape::kC / ThreadOutputShape::kC);
    const int base_p = (threadgroup / (ThreadTileCount::kColumn)) * ThreadOutputShape::kH;
    const int base_q = (threadgroup % (ThreadTileCount::kColumn)) * ThreadOutputShape::kW;

    const int row_offset = base_p * ThreadBlockOutputShape::kW + base_q;

    pointer_ += layout_(
        {row_offset,
         lane_offset.column() * MmaSimtPolicy::LaneMmaShape::kN / int(AccessType::kElements)});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirect2dConv &add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / AccessType::kElements;
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    AccessType *storer_pointer_ =
        reinterpret_cast<AccessType *>(reinterpret_cast<uint8_t *>(pointer_) + base_smem_address_);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < ThreadOutputShape::kH; ++h) {
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < ThreadOutputShape::kW; ++w) {
        CUTLASS_PRAGMA_UNROLL
        for (int col = 0; col < Iterations::kColumn; ++col) {
          int offset = (w + h * ThreadBlockOutputShape::kW) *
                           (ThreadBlockOutputShape::kC / AccessType::kElements) +
                       col;
          storer_pointer_[offset + pointer_offset / int(AccessType::kElements)] =
              frag_ptr[w + h * ThreadOutputShape::kW + col];
        }
      }
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }

  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) { base_smem_address_ = address; }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_,        ///< shape of warp-level GEMM (concept: GemmShape)
  typename Operator_,         ///< matrix multiply operation (concept: arch::Mma)
  typename Element_,          ///< data type of element to be written
  typename Layout_,            ///< target shared memory layout
  typename MmaSimtPolicy_     ///< policy defining lane arrangement (concept: MmaSimtPolicy)
>
class TileIteratorSimtCanonical {
public:

  using WarpShape = WarpShape_;
  using Operator = Operator_;
  using Element = Element_;
  using Layout = Layout_;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = SimtPolicy<WarpShape, Operator, Layout, MmaSimtPolicy_>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    typename Operator::ElementC, 
    Policy::kElementsPerIteration>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<
    typename Operator::ElementC, 
    Policy::kAccumulatorElementCount>;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    4 * Policy::kElementsPerAccess + 1
  >;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<
    Element, 
    1
  >;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Guard to indicate whether the shape is divisible
  bool divisible_;

  /// Extent of the output tensor
  MatrixCoord extent_;

  /// Thread offset
  MatrixCoord thread_offset_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorSimtCanonical(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorSimtCanonical(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / AccessType::kElements),
    divisible_(true),
    extent_(WarpShape::kM, WarpShape::kN) { 

    auto lane_layout = Policy::MmaSimtPolicy::get_lane_layout();
    MatrixCoord lane_offset = lane_layout.inverse(lane_id);

    thread_offset_ = {
      lane_offset.row() * Shape::kRow, 
      lane_offset.column() * Policy::kElementsPerAccess
    };

    pointer_ += layout_({
      lane_offset.row() * Shape::kRow,
      lane_offset.column() * Policy::kElementsPerAccess / int(AccessType::kElements)
    });
  }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorSimtCanonical(
    TensorRef const &ref,
    TensorCoord const &extent,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / AccessType::kElements),
    divisible_(false),
    extent_(extent) { 

    auto lane_layout = Policy::MmaSimtPolicy::get_lane_layout();
    MatrixCoord lane_offset = lane_layout.inverse(lane_id);

    thread_offset_ = {
      lane_offset.row() * Shape::kRow, 
      lane_offset.column() * Policy::kElementsPerAccess
    };

    pointer_ += layout_({
      lane_offset.row() * Shape::kRow,
      lane_offset.column() * Policy::kElementsPerAccess / int(AccessType::kElements)
    });
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorSimtCanonical & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / AccessType::kElements;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorSimtCanonical & add_tile_offset(TensorCoord const &tile_offset) {

    MatrixCoord coord_offset(
      tile_offset.row(), 
      tile_offset.column() * Shape::kColumn
    );

    thread_offset_ += coord_offset;

    pointer_ += layout_({
      coord_offset.row(), 
      coord_offset.column()
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorSimtCanonical & operator+=(TensorCoord const &tile_offset) {

    add_tile_offset(tile_offset);
    
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    // de-vectorized stores
    using ScalarAccessType = AlignedArray<Element, 1>;
    ScalarAccessType const *scalarFragPtr = reinterpret_cast<ScalarAccessType const *>(&frag);
    ScalarAccessType *scalarPointer = reinterpret_cast<ScalarAccessType *>(pointer_) + pointer_offset;

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::kAccessesPerIteration; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < Policy::kElementsPerAccess; s++) {
        
        int ptr_idx = n * Policy::MmaSimtPolicy::WarpShape::kColumn * Policy::kElementsPerAccess + s;
        int frag_idx = n * Policy::kElementsPerAccess + s;
        
        int col = thread_offset_.column() + ptr_idx;

        if (divisible_ || (thread_offset_.row() < extent_.row() && col < extent_.column())) {
          scalarPointer[ptr_idx] = scalarFragPtr[frag_idx];
        }
      }
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

      // de-vectorized loads
      using ScalarAccessType = AlignedArray<Element, 1>;
      ScalarAccessType *scalarFragPtr = reinterpret_cast<ScalarAccessType *>(&frag);
      ScalarAccessType const *scalarPointer = reinterpret_cast<ScalarAccessType const*>(pointer_) + pointer_offset;

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Policy::kAccessesPerIteration; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::kElementsPerAccess; s++) {
          
          int ptr_idx = n * Policy::MmaSimtPolicy::WarpShape::kColumn * Policy::kElementsPerAccess + s;
          int frag_idx = n * Policy::kElementsPerAccess + s;
          
          int col = thread_offset_.column() + ptr_idx;

          if (divisible_ || (thread_offset_.row() < extent_.row() && col < extent_.column())) {
            scalarFragPtr[frag_idx] = scalarPointer[ptr_idx];
          }
        }
      }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  TileIteratorSimtCanonical & operator++() {
    return add_tile_offset({1, 0});
  }

  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};


} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
