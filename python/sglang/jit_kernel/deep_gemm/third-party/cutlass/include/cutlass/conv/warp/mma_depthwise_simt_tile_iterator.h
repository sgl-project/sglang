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

#include "cutlass/conv/convolution.h"

#include "cutlass/arch/memory_sm75.h"

#include "cutlass/layout/matrix.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
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
  cutlass::gemm::Operand Operand,
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
class DepthwiseMmaSimtTileIterator;

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
    int PartitionGroupSize>
class DepthwiseMmaSimtTileIterator<Shape_,
                                   cutlass::gemm::Operand::kB,
                                   Element_,
                                   layout::RowMajor,
                                   Policy_,
                                   PartitionsK,
                                   PartitionGroupSize>
    : public cutlass::gemm::warp::MmaSimtTileIterator<Shape_,
                                               cutlass::gemm::Operand::kB,
                                               Element_,
                                               layout::RowMajor,
                                               Policy_,
                                               PartitionsK,
                                               PartitionGroupSize> {

  using Base = cutlass::gemm::warp::MmaSimtTileIterator<Shape_,
                                               cutlass::gemm::Operand::kB,
                                               Element_,
                                               layout::RowMajor,
                                               Policy_,
                                               PartitionsK,
                                               PartitionGroupSize>;
 public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static cutlass::gemm::Operand const kOperand = cutlass::gemm::Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of policy
  using Layout = layout::RowMajor;

  /// Decomposition of elements among threads
  using Policy = Policy_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = typename Base::TensorRef;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Thread-level shape of a fragment
  using ThreadShape = typename Base::ThreadShape;

  /// Number of individual loads
  using Iterations =  typename Base::Iterations;

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

  static_assert(Policy::LaneMmaShape::kN == 1, "Each thread should be 1 element per LDS along the k-dim");
  
private:

  MatrixCoord lane_offset_;
  int channel_idx_;
  int base_channel_idx_;
  int warps_n_;

 public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  DepthwiseMmaSimtTileIterator():Base() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  DepthwiseMmaSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ) : Base(ref, lane_id) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    warps_n_ = -1;
    channel_idx_ = 0;
    base_channel_idx_ = 0;
    lane_offset_ = lane_layout.inverse(lane_id) * MatrixCoord(0, Policy::LaneMmaShape::kN);
  }
  
  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  DepthwiseMmaSimtTileIterator &add_tile_offset(TensorCoord const &coord) {

    if(warps_n_ == -1){
        warps_n_ = coord.column();
    }
    
    Base::add_tile_offset(coord);
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

        void const *ptr = this->ref_.data() +
                          this->ref_.offset({-(channel_idx_ - base_channel_idx_),
                                             n * Policy::WarpShape::kColumn}) +
                          pointer_offset / Policy::LaneMmaShape::kN;

        // Base_k of a warp +  Base_k of current threads.
        int thread_k_base_idx =
            warps_n_ * Shape::kColumn / Policy::LaneMmaShape::kN + lane_offset_.column();

        if (channel_idx_ + k == thread_k_base_idx + n * Policy::WarpShape::kColumn) {
          // Depthwise kernel would only do computation when channel == k.
          // Loads an element when the current computation channel == the k corresponding to this thread.
          arch::shared_load(dst_ptr[n + k * Iterations::kColumn], ptr);
        } else {
          // Reduce SMEM load
          dst_ptr[n + k * Iterations::kColumn].fill(Element(0));
        }
      }
    }
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
  
  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    if(k_group % PartitionGroupSize == 0 && k_group != 0){
      base_channel_idx_ = k_group;
    }
    channel_idx_ = k_group;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Size of filter (concept: gemm::GemmShape<Depth, Height, Width>)
    typename FilterShape_,
    /// Size of the matrix to load (concept: MatrixShape)
    typename ThreadOutputShape_,
    /// Size of the matrix to load (concept: MatrixShape)
    typename ThreadBlockOutputShape_,
    /// Operand identity
    cutlass::gemm::Operand Operand,
    /// Data type of A elements
    typename Element_,
    /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
    typename Policy_,
    /// Iterator algo type
    conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kAnalytic,
    /// Stride ( MatrixShape<Height, Width> )
    typename StrideShape = cutlass::MatrixShape<-1, -1>,   
    /// Dilation ( MatrixShape<Height, Width> )
    typename DilationShape =  cutlass::MatrixShape<-1, -1>,
    /// Activation Shape loaded by threadblock
    typename ActivationShape = cutlass::conv::TensorNHWCShape<-1,-1,-1,-1>,
    /// Number of partitions along K dimension - used in sliced-K
    int PartitionsK = 1,
    /// Group Size along kPartition - used in sliced-K
    int PartitionGroupSize = 1>
class DepthwiseDirect2dConvSimtTileIterator;


/// Specialization for A operands of row-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Size of filter (concept: gemm::GemmShape<Depth, Height, Width>)
    typename FilterShape_,
    /// Size of the matrix to load (concept: TensorNHWC)
    typename ThreadOutputShape_,
    /// Size of the matrix to load (concept: TensorNHWC)
    typename ThreadBlockOutputShape_,
    /// Data type of A elements
    typename Element_,
    /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
    typename Policy_,
    /// Iterator algo type
    conv::IteratorAlgorithm IteratorAlgorithm,
    /// Stride ( MatrixShape<Height, Width> )
    typename StrideShape,   
    /// Dilation ( MatrixShape<Height, Width> )
    typename DilationShape,
    /// Activation Shape loaded by threadblock
    typename ActivationShape,
    /// Number of partitions along K dimension - used in sliced-K
    int PartitionsK,
    /// Group Size along kPartition - used in sliced-K
    int PartitionGroupSize>
class DepthwiseDirect2dConvSimtTileIterator<Shape_,
                                            FilterShape_,
                                            ThreadOutputShape_,
                                            ThreadBlockOutputShape_,
                                            cutlass::gemm::Operand::kA,
                                            Element_,
                                            Policy_,
                                            IteratorAlgorithm,
                                            StrideShape,   
                                            DilationShape,
                                            ActivationShape,
                                            PartitionsK,
                                            PartitionGroupSize> {
 public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Shape of filter (concept: gemm::GemmShape<Depth, Height, Width>)
  using FilterShape = FilterShape_;

  /// Shape of tile to load (concept: TensorNHWC)
  using ThreadOutputShape = ThreadOutputShape_;

  /// Shape of tile to load (concept: TensorNHWC)
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;

  /// Operand tag
  static cutlass::gemm::Operand const kOperand = cutlass::gemm::Operand::kA;

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

// Thread-level shape of a fragment
  using ThreadShape = MatrixShape<
    ThreadOutputShape::kNHW, // Output tile shape Computed by current threads
    ThreadOutputShape::kC
  >;

  static_assert(!(ThreadShape::kColumn % Policy::LaneMmaShape::kN), 
    "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations = MatrixShape<
    ThreadShape::kRow,
    ThreadShape::kColumn / Policy::LaneMmaShape::kN
  >;

  using ThreadTileCount = MatrixShape<
    ThreadBlockOutputShape::kH / ThreadOutputShape::kH,
    ThreadBlockOutputShape::kW / ThreadOutputShape::kW
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

protected:

  /// Internal reference
  cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kN>, layout::RowMajor> ref_;

  int activation_offset[ThreadOutputShape::kH][ThreadOutputShape::kW][Iterations::kColumn];
  int iterator_r_;
  int iterator_s_;
  int iterator_offset_;

  int inc_next_s_ ;
  int inc_next_r_ ;
  
  MatrixCoord lane_offset_;
public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator(
    TensorRef ref, 
    int lane_id
  ) {

    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    // Set channel offset
    lane_offset_ = lane_layout.inverse(lane_id) * MatrixCoord(0, Policy::LaneMmaShape::kN);

    ref.add_coord_offset(lane_offset_);

    ref_.reset(reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(ref.data()),
               ref.stride(0) / Policy::LaneMmaShape::kN);

    iterator_r_ = 0;
    iterator_s_ = 0;
    iterator_offset_ = 0;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  template<typename Params>
  CUTLASS_HOST_DEVICE
  void setup_initial_status(Params const& params)  {
  
    inc_next_s_ = params.inc_next[0];
    inc_next_r_ = params.inc_next[1];

    // Get base HW offset of current threads
    int threadgroup = threadIdx.x / (ThreadBlockOutputShape::kC / ThreadOutputShape::kC);
    int base_p_ =
        (threadgroup / (ThreadTileCount::kColumn)) * ThreadOutputShape::kH;
    int base_q_ =
        (threadgroup % (ThreadTileCount::kColumn)) * ThreadOutputShape::kW;

    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < ThreadOutputShape::kH; ++p) {
      CUTLASS_PRAGMA_UNROLL
      for (int q = 0; q < ThreadOutputShape::kW; ++q) {
        CUTLASS_PRAGMA_UNROLL
        for (int col = 0; col < Iterations::kColumn; ++col) {
          int base_w = (base_q_ + q) * params.stride[0];
          int base_h = (base_p_ + p) * params.stride[1];

          int offset = base_h * params.activation_tile_w + base_w;
          activation_offset[p][q][col] = offset;
        }
      }
    }
  }


  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator &add_tile_offset(TensorCoord const &coord) {
    // Set warp row and col start
    lane_offset_ = MatrixCoord({lane_offset_.row() + coord.row() * Shape::kRow, lane_offset_.column()});
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  void advance(int32_t pointer_offset) {
    ref_.reset(ref_.data() + pointer_offset / sizeof(Element) / Policy::LaneMmaShape::kN);
    iterator_s_ = 0;
    iterator_r_ = 0;
    iterator_offset_ = 0;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator &operator++() {
    ++iterator_s_;
    if (iterator_s_ < FilterShape::kColumn) {
      iterator_offset_ += inc_next_s_;

      return *this;
    }

    iterator_s_ = 0;

    ++iterator_r_;
    if (iterator_r_ < FilterShape::kRow) {
      iterator_offset_ += inc_next_r_;
      return *this;
    }

    iterator_r_ = 0;
    iterator_offset_ = 0;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator & operator--() {
    // Do nothing
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator. (vector loads)
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    Array<Element, Policy::LaneMmaShape::kN> *dst_ptr = 
      reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(&frag);


    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < ThreadOutputShape::kH; ++p) {
      CUTLASS_PRAGMA_UNROLL
      for (int q = 0; q < ThreadOutputShape::kW; ++q) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Iterations::kColumn; ++n) {
          void const *ptr = ref_.data() +
                            ref_.offset({activation_offset[p][q][n] + (iterator_offset_),
                                         n * Policy::WarpShape::kColumn}) +
                            pointer_offset / Policy::LaneMmaShape::kN;
          arch::shared_load(dst_ptr[n + q + p * ThreadOutputShape::kW], ptr);
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
    // Do nothing at present.
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, Index pointer_offset) const {
    store_with_pointer_offset(frag, 0);
  }

  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no operation here
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Specialization for A operands of row-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Size of filter (concept: gemm::GemmShape<Depth, Height, Width>)
    typename FilterShape_,
    /// Size of the matrix to load (concept: TensorNHWC)
    typename ThreadOutputShape_,
    /// Size of the matrix to load (concept: TensorNHWC)
    typename ThreadBlockOutputShape_,
    /// Data type of A elements
    typename Element_,
    /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
    typename Policy_,
    /// Stride ( MatrixShape<Height, Width> )
    typename StrideShape_,
    /// Dilation ( MatrixShape<Height, Width> )
    typename DilationShape_,
    /// Activation Shape loaded by threadblock
    typename ActivationShape_,
    /// Number of partitions along K dimension - used in sliced-K
    int PartitionsK,
    /// Group Size along kPartition - used in sliced-K
    int PartitionGroupSize>
class DepthwiseDirect2dConvSimtTileIterator<Shape_,
                                            FilterShape_,
                                            ThreadOutputShape_,
                                            ThreadBlockOutputShape_,
                                            cutlass::gemm::Operand::kA,
                                            Element_,
                                            Policy_,
                                            IteratorAlgorithm::kFixedStrideDilation,
                                            StrideShape_,
                                            DilationShape_,
                                            ActivationShape_,
                                            PartitionsK,
                                            PartitionGroupSize> {
 public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Shape of filter (concept: gemm::GemmShape<Depth, Height, Width>)
  using FilterShape = FilterShape_;

  /// Shape of tile to load (concept: TensorNHWC)
  using ThreadOutputShape = ThreadOutputShape_;

  /// Shape of tile to load (concept: TensorNHWC)
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;

  /// Stride ( MatrixShape<Height, Width> )
  using StrideShape = StrideShape_;

  /// Dilation ( MatrixShape<Height, Width> )
  using DilationShape = DilationShape_;

  /// Activation Shape loaded by threadblock
  using ActivationShape = ActivationShape_;

  /// Operand tag
  static cutlass::gemm::Operand const kOperand = cutlass::gemm::Operand::kA;

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
                "The warp-level GEMM M size must be divisible by the number of threads arranged "
                "along the M dimension.");

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Policy::WarpShape::kRow > 0, "Policy::WarpShape::kRow must be greater than zero.");
  static_assert(Shape::kRow / Policy::WarpShape::kRow > 0,
                "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

  // Activations loaded by threadblock
  static int const ThreadActivationShapeH = (ThreadOutputShape::kH - 1) * StrideShape::kRow +
                                            (FilterShape::kRow - 1) * DilationShape::kRow + 1;

  static int const ThreadActivationShapeW = (ThreadOutputShape::kW - 1) * StrideShape::kColumn +
                                            (FilterShape::kColumn - 1) * DilationShape::kColumn + 1;

  using ThreadActivationShape = cutlass::conv::
      TensorNHWCShape<1, ThreadActivationShapeH, ThreadActivationShapeW, ThreadOutputShape::kC>;

  // Thread-level shape of a fragment
  using ThreadShape =
      MatrixShape<ThreadOutputShape::kNHW,
                  ThreadOutputShape::kC>;

  static_assert(!(ThreadShape::kColumn % Policy::LaneMmaShape::kN),
                "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  /// Number of individual loads
  using Iterations =
      MatrixShape<ThreadShape::kRow, ThreadShape::kColumn / Policy::LaneMmaShape::kN>;

  using ThreadTileCount = MatrixShape<ThreadBlockOutputShape::kH / ThreadOutputShape::kH,
                                      ThreadBlockOutputShape::kW / ThreadOutputShape::kW>;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount>;

 protected:
  /// Internal reference
  cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kN>, layout::RowMajor> ref_;

  Array<Element, Policy::LaneMmaShape::kN>
      activation[ThreadActivationShape::kH][ThreadActivationShape::kW][Iterations::kColumn];
  int iterator_r_;
  int iterator_s_;


  MatrixCoord lane_offset_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator() {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator(TensorRef ref, int lane_id) {
    // compute offset based on thread ID and lane layout
    typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

    // Set channel offset
    lane_offset_ = lane_layout.inverse(lane_id) * MatrixCoord(0, Policy::LaneMmaShape::kN);

    ref.add_coord_offset(lane_offset_);

    ref_.reset(reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(ref.data()),
               ref.stride(0) / Policy::LaneMmaShape::kN);

    iterator_r_ = 0;
    iterator_s_ = 0;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  template <typename Params>
  CUTLASS_HOST_DEVICE void setup_initial_status(
      Params const &params) {

    // Get base HW offset of current threads
    int threadgroup = threadIdx.x / (ThreadBlockOutputShape::kC / ThreadOutputShape::kC);
    int base_h =
        (threadgroup / (ThreadTileCount::kColumn)) * ThreadOutputShape::kH * StrideShape::kRow;
    int base_w =
        (threadgroup % (ThreadTileCount::kColumn)) * ThreadOutputShape::kW * StrideShape::kColumn;

    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < ThreadActivationShape::kH; ++h) {
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < ThreadActivationShape::kW; ++w) {
        CUTLASS_PRAGMA_UNROLL
        for (int col = 0; col < Iterations::kColumn; ++col) {
          int offset = (base_h + h) * ActivationShape::kW + (base_w + w);

          void const *ptr = ref_.data() + ref_.offset({offset, col * Policy::WarpShape::kColumn});
          arch::shared_load(activation[h][w][col], ptr);
        }
      }
    }
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator &add_tile_offset(TensorCoord const &coord) {
    // Set warp row and col start
    lane_offset_ =
        MatrixCoord({lane_offset_.row() + coord.row() * Shape::kRow, lane_offset_.column()});
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  void advance(int32_t pointer_offset) {
    ref_.reset(ref_.data() + pointer_offset / sizeof(Element) / Policy::LaneMmaShape::kN);
    iterator_s_ = 0;
    iterator_r_ = 0;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator &operator++() {
    ++iterator_s_;
    if (iterator_s_ < FilterShape::kColumn) {
      return *this;
    }

    iterator_s_ = 0;

    ++iterator_r_;
    if (iterator_r_ < FilterShape::kRow) {
      return *this;
    }

    iterator_r_ = 0;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvSimtTileIterator &operator--() {
    // Do nothing
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator. (vector loads)
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
    Array<Element, Policy::LaneMmaShape::kN> *dst_ptr =
        reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < ThreadOutputShape::kH; ++p) {
      CUTLASS_PRAGMA_UNROLL
      for (int q = 0; q < ThreadOutputShape::kW; ++q) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Iterations::kColumn; ++n) {
          const int h = p * StrideShape::kRow + iterator_r_ * DilationShape::kRow;
          const int w = q * StrideShape::kColumn + iterator_s_ * DilationShape::kColumn;

          dst_ptr[n + q + p * ThreadOutputShape::kW] = activation[h][w][n];
        }
      }
    }
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_pointer_offset(frag, 0); }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {
    // Do nothing at present.
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, Index pointer_offset) const {
    store_with_pointer_offset(frag, 0);
  }

  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no operation here
  }
};

} // namespace warp
} // namespace conv
} // namespace cutlass
