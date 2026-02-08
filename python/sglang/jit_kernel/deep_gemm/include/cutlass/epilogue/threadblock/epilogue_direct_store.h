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
  \brief Epilogue for threadblock scoped GEMMs and convolution using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/reduction_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_                        ///< Output operator
>
class EpilogueDirectStore {
public:

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  using WarpShape = typename WarpMmaOperator_::Shape;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using OutputOp = OutputOp_;
  using Padding = MatrixShape<0, 0>;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename WarpTileIterator::Element;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Array type used to output
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Array type used by output functor
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>; 
  
  /// Number of warps
  using WarpCount = gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    kPartitionsK
  >;

  /// Use this to control the granularity of one epilogue 'iteration'
  static int const kFragmentsPerIteration = 1;

  static int constexpr kSmemTiles = 1;
  static int constexpr kSmemPointerOffset = 0;

  /// Shared storage allocation needed by the epilogue
  struct SharedStorage { } ;

private:

  // Assume accumulator tile is multipile interleaved 32x32 tile.
  static int const kElementsPerPartial = 4;
  using EleShapePerPatial = typename platform::conditional<
                              platform::is_same<ElementAccumulator, float>::value,
                              MatrixShape<2, 2>,
                              MatrixShape<1, 4> >::type;
  static int const kElementsPerMma = 8;
  static int const kAccumulatorPatials = 2;
  using QuadShapePerPatialMma = MatrixShape<4, 4>;

  static_assert(OutputOp::kCount >= 2, 
    "The direct store epilogue for Tensor Ops requires the output functor have kCount >= 2.");

private:

  LongIndex warp_offset;
  int thread_idx;
  int warp_idx;
  int lane_idx;
  int warp_m, warp_n; // warp coordinates within a cta
  int tid_m, tid_n;   // thread coordinates within a warp

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueDirectStore(
    SharedStorage &shared_storage,    ///< Shared storage object    
    int thread_idx_,                   ///< ID of a thread within the threadblock
    int warp_idx_,                     ///< ID of warp within threadblock
    int lane_idx_                     ///< Id of thread within warp
  ):
    thread_idx(thread_idx_), 
    warp_idx(warp_idx_), 
    lane_idx(lane_idx_) 
  {
    
    // warp offsetting calculations
    warp_offset = warp_idx * WarpShape::kM * WarpShape::kN;
    int warp_id_mn = warp_idx % (WarpCount::kM * WarpShape::kN);
    warp_m = warp_id_mn % WarpCount::kM;
    warp_n = warp_id_mn / WarpCount::kM;
    MatrixCoord warp_offset_coord(warp_m*WarpShape::kM, warp_n*WarpShape::kN);
    
    // thread offsetting calculations
    int quad = (lane_idx >> 2);
    int lane_in_quad = (lane_idx & 3);

    // this seems to be te correct layout
    tid_m = quad;
    tid_n = 2 * lane_in_quad;
  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(output_op, destination_iterator, accumulators);
    }
    else {
      compute_source_needed_(output_op, destination_iterator, accumulators, source_iterator);
    }
  }

private:

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_needed_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 2;
    const int kThreadsM = 8;
    const int kThreadsN = 4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    CUTLASS_PRAGMA_UNROLL
    for (int accum_m_idx = 0; accum_m_idx < WarpShape::kM / kThreadsM; accum_m_idx++) {

      int accum_m = kThreadsM * accum_m_idx;
      int mL = destination_iterator.threadblock_offset.row() + WarpShape::kM * warp_m + tid_m + accum_m;
      int nL_base = destination_iterator.threadblock_offset.column() + WarpShape::kN * warp_n + tid_n;

      ElementOutput *output_ptr = destination_iterator.pointer + mL * destination_iterator.stride;
      ElementOutput *source_ptr = source_iterator.pointer + mL * source_iterator.stride;

      int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;

      CUTLASS_PRAGMA_UNROLL
      for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {

        int accum_idx = accum_m_idx + kBlockM * accum_n_idx;
        int accum_n = kThreadsM * accum_n_idx;
        
        // mL and nL are logical coordinate in 2D mapping of epilogue's 4D output 
        int nL = nL_base + accum_n;
          
        bool guard = (mL < destination_iterator.extent.row()) && (nL < destination_iterator.extent.column());

        AccumulatorFragmentType accum_fragment;
        reinterpret_cast<AccumulatorAccessType &>(accum_fragment) = accumulator_pair[accum_idx];

        OutputFragmentType output_fragment;

        if(guard) {
          reinterpret_cast<OutputAccessType &>(output_fragment) = 
            *reinterpret_cast<OutputAccessType const *>(source_ptr + nL);
        }

        // Perform output operator
        output_fragment = output_op(accum_fragment, output_fragment);

        if(guard) {
          // Store
          *reinterpret_cast<OutputAccessType *>(output_ptr + nL) = reinterpret_cast<OutputAccessType const &>(output_fragment);
        }
      }
    }
  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_not_needed_(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    const int kAccumBlockN = 2;
    const int kThreadsM = 8;
    const int kThreadsN = 4;
    const int kBlockM = WarpShape::kM / kThreadsM;

    /// Array type used to output
    using OutputAccessType = AlignedArray<ElementOutput, kAccumBlockN>;

    /// Array type passed to the output operator - unused elements are optimized away
    using OutputFragmentType = Array<ElementOutput, OutputOp::kCount>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<ElementAccumulator, kAccumBlockN>;

    /// Array type used by output functor
    using AccumulatorFragmentType = Array<ElementAccumulator, OutputOp::kCount>;

    AccumulatorAccessType const *accumulator_pair = reinterpret_cast<AccumulatorAccessType const *>(&accumulators);

    CUTLASS_PRAGMA_UNROLL
    for (int accum_m_idx = 0; accum_m_idx < WarpShape::kM / kThreadsM; accum_m_idx++) {

      int accum_m = kThreadsM * accum_m_idx;
      int mL = destination_iterator.threadblock_offset.row() + WarpShape::kM * warp_m + tid_m + accum_m;
      int nL_base = destination_iterator.threadblock_offset.column() + WarpShape::kN * warp_n + tid_n;

      ElementOutput *output_ptr = destination_iterator.pointer + mL * destination_iterator.stride;

      int const kIterationsN = WarpShape::kN / kThreadsN / kAccumBlockN;

      CUTLASS_PRAGMA_UNROLL
      for (int accum_n_idx = 0; accum_n_idx < kIterationsN; accum_n_idx++) {

        int accum_idx = accum_m_idx + kBlockM * accum_n_idx;
        int accum_n = kThreadsM * accum_n_idx;
        
        // mL and nL are logical coordinate in 2D mapping of epilogue's 4D output 
        int nL = nL_base + accum_n;
          
        bool guard = (mL < destination_iterator.extent.row()) && (nL < destination_iterator.extent.column());
                   
        AccumulatorFragmentType accum_fragment;
        reinterpret_cast<AccumulatorAccessType &>(accum_fragment) = accumulator_pair[accum_idx];

        OutputFragmentType output_fragment;

        // Perform output operator
        output_fragment = output_op(accum_fragment);

        if(guard) { 

          // Store
          *reinterpret_cast<OutputAccessType *>(output_ptr + nL) = 
            reinterpret_cast<OutputAccessType const &>(output_fragment);      
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
