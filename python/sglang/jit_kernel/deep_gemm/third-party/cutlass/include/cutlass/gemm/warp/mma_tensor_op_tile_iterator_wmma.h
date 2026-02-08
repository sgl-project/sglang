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
#include "cutlass/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include "cutlass/wmma_array.h"
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
    ///< Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity (A or B)
    Operand Operand,
    /// Data type of operand
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Delta between *MMA operations (in units of *WMMA operations, concept:MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads,
    /// Shape of the warp in units of thread (concept: MmaTensorOpPolicy)
    typename Policy_>
class MmaTensorOpWmmaMultiplicandTileIterator;


////////////////////////////////////////////////////////////////////////////////
/// This tile iterator is specialized for 32-thread WMMA operation. 
/// It uses nvcuda::wmma::load_matrix_sync to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
////////////////////////////////////////////////////////////////////////////////
template <
    ///< Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Interval between adjacent *WMMA instructions (in units of WMMA instructions)
    int OpDelta_,    
    /// Shape of the warp in units of thread (concept: MmaTensorOpPolicy)
    typename Policy_>
class MmaTensorOpWmmaMultiplicandTileIterator<
    Shape_, Operand::kA, Element_, Layout_,
    OpDelta_, 32, Policy_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Delta between *WMMA operations
  static int const kOpDelta = OpDelta_;

  /// Wmma Operator information and operation delta
  using Policy = Policy_;


  //
  // Derived quantities
  //
  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Stride Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Native Wmma shape for operand A (concept MatrixShape)
  using WmmaShape = MatrixShape<
    Policy::Operator::Shape::kM, 
    Policy::Operator::Shape::kK
  >;

  /// Map cutlass dataype to nvcuda::wmma datatype
  using WmmaDataType = typename cutlass::arch::CutlassToWmmaDataType<Element>::Type;

  /// Shape of individual WMMA load / stores for operand A
  using Iterations = MatrixShape<
    Shape::kRow / WmmaShape::kRow,
    1 
  >;

  /// Fragment object holding a warps part 
  using Fragment = WmmaFragmentArray<typename Policy::Operator::FragmentA, Iterations::kCount>;


  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// statically assert this specialization
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  /// This iterator is specalized for Operand A
  static_assert(kOperand == Operand::kA,
    "MmaTensorOpWmmaMultiplicandTileIterator may only be instantiated for A operands to warp-level Mma.");

  /// Supported memory layouts
  static_assert(
    platform::is_same<cutlass::layout::RowMajor, Layout>::value ||
    platform::is_same<cutlass::layout::ColumnMajor, Layout>::value,
    "Supported list of memory layouts for WMMA are: RowMajor, ColumnMajor");

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /////////////////////////////////////////////////////////////////////////////////////////////////////

private:

  /// Shared memory base pointers - not advanced
  char const *pointer_;
  
  /// Byte offset into shared memory - advanced
  Index byte_offset_;
  
  /// Stride in units of number of elements
  StrideIndex stride_;

  /// Layout of shared memory
  Layout layout_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): pointer_(reinterpret_cast<char const*>(ref.data())), byte_offset_(0), stride_(ref.stride(0)), layout_(ref.stride(0)) { 
  
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += (offset * sizeof_bits<Element>::value) / 8;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    Index elements_offset = layout_({tile_offset.row() * Shape::kRow, tile_offset.column() * WmmaShape::kColumn});
    
    byte_offset_ += (elements_offset * sizeof_bits<Element>::value) / 8;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator & operator++() {
    
    Index elements_offset = layout_({0, WmmaShape::kColumn});

    byte_offset_ += (elements_offset * sizeof_bits<Element>::value) / 8;

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator & operator--() {
    
    Index elements_offset = layout_({0, WmmaShape::kColumn});

    byte_offset_ -= (elements_offset * sizeof_bits<Element>::value) / 8;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load_with_byte_offset(Fragment &frag, Index byte_offset) const {

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {

        Index load_byte_offset = layout_({m * WmmaShape::kRow, k * WmmaShape::kColumn}) * sizeof_bits<Element>::value / 8;

        const WmmaDataType *ptr = reinterpret_cast<const WmmaDataType *>(pointer_ + byte_offset_ + load_byte_offset + byte_offset); 

        nvcuda::wmma::load_matrix_sync(frag[m], ptr, stride_); 
      
      }
    }
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_byte_offset(frag, 0);
  }
    
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_byte_offset(Fragment const &frag, Index byte_offset) const {
    
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kColumn; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < Iterations::kRow; ++m) {

        Index store_byte_offset = layout_({m * WmmaShape::kRow, k * WmmaShape::kColumn}) * sizeof_bits<Element>::value / 8;

        WmmaDataType *ptr = reinterpret_cast<WmmaDataType *>(pointer_ + byte_offset_ + store_byte_offset + byte_offset);

        nvcuda::wmma::store_matrix_sync(ptr, frag[m], stride_); 
      
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_byte_offset(frag, 0);
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


////////////////////////////////////////////////////////////////////////////////
/// This tile iterator is specialized for 32-thread WMMA operation. 
/// It uses nvcuda::wmma::load_matrix_sync to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
////////////////////////////////////////////////////////////////////////////////

template <
    ///< Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Interval between adjacent *WMMA instructions (in units of WMMA instructions)
    int OpDelta_,    
    /// Shape of the warp in units of thread (concept: MmaTensorOpPolicy)
    typename Policy_>
class MmaTensorOpWmmaMultiplicandTileIterator<
    Shape_, Operand::kB, Element_, Layout_,
    OpDelta_, 32, Policy_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Delta between *WMMA operations
  static int const kOpDelta = OpDelta_;

  /// Wmma Operator information and operation delta
  using Policy = Policy_;


  //
  // Derived quantities
  //

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Stride Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Native Wmma shape (concept MatrixShape)
  using WmmaShape = MatrixShape<
    Policy::Operator::Shape::kK, 
    Policy::Operator::Shape::kN
  >;

  /// Map cutlass dataype to nvcuda::wmma datatype
  using WmmaDataType = typename cutlass::arch::CutlassToWmmaDataType<Element>::Type;

  /// Shape of individual WMMA load / stores for operand B
  using Iterations = MatrixShape<
    1,
    Shape::kColumn / WmmaShape::kColumn
  >;

  /// Fragment object holding a warps part
  using Fragment = WmmaFragmentArray<typename Policy::Operator::FragmentB, Iterations::kCount>;


  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// statically asserts this specialization
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  /// This iterator is specalized for Operand B
  static_assert(kOperand == Operand::kB,
    "MmaTensorOpWmmaMultiplicandTileIterator may only be instantiated for B operands to warp-level Mma.");

  /// Supported memory layouts
  static_assert(
    platform::is_same<cutlass::layout::RowMajor, Layout>::value ||
    platform::is_same<cutlass::layout::ColumnMajor, Layout>::value,
    "Supported list of memory layouts for WMMA are: RowMajor, ColumnMajor");

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /////////////////////////////////////////////////////////////////////////////////////////////////////

private:

  /// Shared memory base pointers - not advanced
  char const *pointer_;
  
  /// Byte offset into shared memory - advanced
  Index byte_offset_;
  
  /// Stride in units of number of elements
  StrideIndex stride_;

  /// Layout of shared memory
  Layout layout_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): pointer_(reinterpret_cast<char const*>(ref.data())), byte_offset_(0), stride_(ref.stride(0)), layout_(ref.stride(0)) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    
    byte_offset_ += (offset * sizeof_bits<Element>::value) / 8;

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    
    Index elements_offset = layout_({tile_offset.row() * WmmaShape::kRow, tile_offset.column() * Shape::kColumn});
    
    byte_offset_ += (elements_offset * sizeof_bits<Element>::value) / 8;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator & operator++() {
    
    Index elements_offset = layout_({WmmaShape::kRow, 0});

    byte_offset_ += (elements_offset * sizeof_bits<Element>::value) / 8;
    
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator & operator--() {

    Index elements_offset = layout_({WmmaShape::kRow, 0});

    byte_offset_ -= (elements_offset * sizeof_bits<Element>::value) / 8;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpWmmaMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load_with_byte_offset(Fragment &frag, Index byte_offset) const {

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kRow; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kColumn; ++n) {
        
        Index load_byte_offset = layout_({k * WmmaShape::kRow, n * WmmaShape::kColumn}) * sizeof_bits<Element>::value / 8;

        const WmmaDataType *ptr = reinterpret_cast<const WmmaDataType *>(pointer_ + byte_offset_ + load_byte_offset + byte_offset);

        nvcuda::wmma::load_matrix_sync(frag[n], ptr, stride_);        
      }
    }
  }
  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_byte_offset(frag, 0);
  }
    
  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_byte_offset(Fragment const &frag, Index byte_offset) const {
    
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Iterations::kRow; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kColumn; ++n) {

        Index store_byte_offset = layout_({k * WmmaShape::kRow, n * WmmaShape::kColumn}) * sizeof_bits<Element>::value / 8;

        WmmaDataType *ptr = reinterpret_cast<WmmaDataType *>(pointer_ + byte_offset_ + store_byte_offset + byte_offset);
        
        nvcuda::wmma::store_matrix_sync(ptr, frag[n], stride_);        
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_byte_offset(frag, 0);
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

////////////////////////////////////////////////////////////////////////////////
template <
    ///< Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Interval between adjacent *WMMA instructions (in units of WMMA instructions, concept: MatrixShape)
    typename OpDelta_,
    /// Shape of the warp in units of thread (concept: MmaTensorOpPolicy)
    typename Policy_>
class MmaTensorOpWmmaAccumulatorTileIterator;

////////////////////////////////////////////////////////////////////////////////
/// This tile iterator is specialized for 32-thread WMMA operation. 
/// It uses nvcuda::wmma::store_matrix_sync to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
////////////////////////////////////////////////////////////////////////////////

template <
    ///< Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Interval between adjacent *WMMA instructions (in units of WMMA instructions)
    typename OpDelta_,    
    /// Shape of the warp in units of thread (concept: MmaTensorOpPolicy)
    typename Policy_>
class MmaTensorOpWmmaAccumulatorTileIterator
{
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Wmma Operator information and operation delta
  using Policy = Policy_;


  //
  // Derived quantities
  //
  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Native Wmma shape (concept MatrixShape)
  using WmmaShape = MatrixShape<
    Policy::Operator::Shape::kM, 
    Policy::Operator::Shape::kN
  >;
  
  /// Map cutlass dataype to nvcuda::wmma datatype
  using WmmaDataType = typename cutlass::arch::CutlassToWmmaDataType<Element>::Type;

  /// Map cutlass::layout to nvuda::wmma::layout_t enum
  static nvcuda::wmma::layout_t const WmmaLayout = cutlass::arch::CutlassToWmmaLayout<Layout>::value;

  /// Shape of individual WMMA load / stores for accumulator
  using Iterations = MatrixShape<
    Shape::kRow / WmmaShape::kRow,
    Shape::kColumn / WmmaShape::kColumn
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = WmmaFragmentArray<typename Policy::Operator::FragmentC, Iterations::kCount>;

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// statically asserts this specialization
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Supported layouts
  static_assert(
    platform::is_same<cutlass::layout::RowMajor, Layout>::value ||
    platform::is_same<cutlass::layout::ColumnMajor, Layout>::value,
    "Supported list of memory layouts for WMMA are: RowMajor, ColumnMajor");

private:
  
  /// Internal reference
  cutlass::TensorRef<Element, Layout> ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpWmmaAccumulatorTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): ref_(ref) { }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpWmmaAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    ref_.add_coord_offset({tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpWmmaAccumulatorTileIterator & operator++() {
    ref_.add_coord_offset({Shape::kRow, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpWmmaAccumulatorTileIterator & operator--() {
    ref_.add_coord_offset({-Shape::kRow, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpWmmaAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpWmmaAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
    
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < Iterations::kRow; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kColumn; ++n) {

        const WmmaDataType * ptr = reinterpret_cast<const WmmaDataType*> (ref_.data() + ref_.offset({m * WmmaShape::kRow, n * WmmaShape::kColumn}) + pointer_offset);
        
        nvcuda::wmma::load_matrix_sync(frag[m * Iterations::kColumn + n], ptr, ref_.stride()[0], WmmaLayout); 

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
    for (int m = 0; m < Iterations::kRow; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Iterations::kColumn; ++n) {

        WmmaDataType * ptr = reinterpret_cast<WmmaDataType*> (ref_.data() + ref_.offset({m * WmmaShape::kRow, n * WmmaShape::kColumn}) + pointer_offset);

        nvcuda::wmma::store_matrix_sync(ptr, frag[m * Iterations::kColumn + n], ref_.stride()[0], WmmaLayout); 
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



} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

#endif // if defined(CUTLASS_ARCH_WMMA_ENABLED)


