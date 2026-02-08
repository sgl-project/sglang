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
#if !defined(__CUDACC_RTC__)
#include <type_traits>
#include <utility>
#endif
#include CUDA_STD_HEADER(cassert)

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

//
// This is used for metaprogramming epilogue functors. If they define 
// `static bool const kIsHeavy = true;`, then the epilogue functor itself is
// not inlined. This results in smaller code and is advantageous if the epilogue
// functor consists of many instructions.
//
// If the epilogue functor does not define `kIsHeavy` or if it is `false`, then
// the behavior from CUTLASS 2.5 and before is retained. The epilogue is fully
// unrolled and inlined.
//

template<class> 
struct TypeSink {  typedef void type; };

template<class T> using TypeSinkT = typename TypeSink<T>::type;

template<class T, class=void> struct IsEpilogueFunctorHeavy {
  static bool const value = false;
};

template<class T> struct IsEpilogueFunctorHeavy<T, TypeSinkT< decltype( T::kIsHeavy ) > > {
  static bool const value = T::kIsHeavy;
};

////////////////////////////////////////////////////////////////////////////////

/// Base class for epilogues defining warp-level 
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpShape_,                      ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerIteration = 1
>
class EpilogueBase {
public:

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  static int const kPartitionsK = PartitionsK;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using Padding = Padding_;

  /// Output layout is always row-major
  using Layout = layout::RowMajor;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename AccumulatorTile::Element;

  /// Number of warps
  using WarpCount = gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    kPartitionsK
  >;

  /// Use this to control the granularity of one epilogue 'iteration'
  static int const kFragmentsPerIteration = FragmentsPerIteration;

public:

  /// Shared storage allocation needed by the epilogue
  struct SharedStorage {
    
    //
    // Type definitions
    //

    /// Element type of shared memory
    using Element = typename WarpTileIterator::Element;

    /// Tensor reference to shared memory allocation
    using TensorRef = typename WarpTileIterator::TensorRef;

    /// Layout of shared memory allocation
    using Layout = typename WarpTileIterator::Layout;
    
    /// Logical shape of the shared memory tile written to by all warps.
    using Shape = MatrixShape<
      WarpCount::kM * WarpTileIterator::Shape::kRow * WarpCount::kK,
      WarpCount::kN * WarpTileIterator::Shape::kColumn
    >;

    /// Shape of the shared memory allocation for the epilogue    
    using StorageShape = MatrixShape<
      (Shape::kRow + Padding::kRow) * kFragmentsPerIteration, 
      Shape::kColumn + Padding::kColumn
    >;

    //
    // Data members
    //

    AlignedBuffer<Element, StorageShape::kCount> storage;

    //
    // Methods
    //

    /// Returns a pointer to the shared memory buffer
    CUTLASS_DEVICE
    Element *data() {
      return storage.data();
    }

    /// Returns a tensor reference to the shared memory buffer
    CUTLASS_DEVICE
    TensorRef reference() {
      return TensorRef(
        storage.data(), 
        Layout::packed({StorageShape::kRow, StorageShape::kColumn}));
    }
  };

protected:

  //
  // Data members
  //

  SharedStorage &shared_storage_;

  /// Stores a warp's fragment of accumulators to SMEM
  WarpTileIterator warp_tile_iterator_;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueBase(
    SharedStorage &shared_storage,    ///< Shared storage object    
    int thread_idx,                   ///< ID of a thread within the threadblock
    int warp_idx,                     ///< ID of warp within threadblock
    int lane_idx                      ///< Id of thread within warp
  ):
    shared_storage_(shared_storage),
    warp_tile_iterator_(shared_storage.reference(), lane_idx) {

    // Compute warp location within threadblock tile by mapping the warp_id to three coordinates:
    //
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int warp_m = warp_mn % WarpCount::kM;
    int warp_n = warp_mn / WarpCount::kM;

    MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m, warp_n};

    warp_tile_iterator_.add_tile_offset(warp_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
