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
#include "cutlass/wmma_array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/epilogue/warp/wmma_tensor_op_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape,           ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape,       ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorFragment,    ///< wmma fragment to be written (concept: nvcuda::wmma::fragment)
  typename Layout               ///< target shared memory layout
>
class TileIteratorWmmaTensorOp;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_,          ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,      ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorFragment_    ///< wmma fragment to be written (concept: nvcuda::wmma::fragment)
>
class TileIteratorWmmaTensorOp<WarpShape_, OperatorShape_, OperatorFragment_, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using OperatorFragment = OperatorFragment_;
  using Layout = layout::RowMajor;

  //
  // Derived types
  //
  using WmmaDataType = typename OperatorFragment::element_type;
  using Element = typename cutlass::arch::WmmaToCutlassDataType<WmmaDataType>::Type; ///< Data Type of element stored in nvcuda::wmma::frament         
  using TensorRef = TensorRef<Element, Layout>;                                      ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                                                   ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = WmmaTensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = WmmaFragmentArray<OperatorFragment, Policy::OperatorCount::kColumn * Policy::kWmmaFragmentsPerAccess>;


  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;


  /// Padding quantity 
  // (Epilogue shared memory padding for WMMA Gemm kernel is set to run optimaly on Turing)
  using Padding = MatrixShape<
    0,
    4 * Policy::kElementsPerAccess
  >;

private:

  /// Storage type for accessing memory
  //using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

  //
  // Data members
  //

  /// Internal pointer to shared memory
  TensorRef ref_;


public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorWmmaTensorOp(): ref_(nullptr) { 

  }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorWmmaTensorOp(
    TensorRef const &ref,
    unsigned lane_id
  ): ref_(ref) {
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorWmmaTensorOp & add_pointer_offset(Index pointer_offset) {
    ref_.add_pointer_offset(pointer_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorWmmaTensorOp & add_tile_offset(TensorCoord const &tile_offset) {
    ref_.add_coord_offset({tile_offset.row() * OperatorShape::kM, tile_offset.column() * WarpShape::kN});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorWmmaTensorOp & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    for(int n=0; n < Policy::OperatorCount::kColumn; n++) {
      
      WmmaDataType* ptr = reinterpret_cast<WmmaDataType*> (ref_.data() + ref_.offset({0, n * OperatorShape::kN}) + pointer_offset);

      nvcuda::wmma::store_matrix_sync(
        ptr, 
        frag[n], 
        ref_.stride()[0], 
        nvcuda::wmma::layout_t::mem_row_major
      ); 
    
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
 
    for(int n=0; n < Policy::OperatorCount::kColumn; n++) {

      WmmaDataType* ptr = reinterpret_cast<WmmaDataType*> (ref_.data() + ref_.offset({0, n * OperatorShape::kN}) + pointer_offset);

      nvcuda::wmma::load_matrix_sync(         
        frag[n], 
        ptr,
        ref_.stride()[0], 
        nvcuda::wmma::layout_t::mem_row_major
      ); 
    
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

} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


