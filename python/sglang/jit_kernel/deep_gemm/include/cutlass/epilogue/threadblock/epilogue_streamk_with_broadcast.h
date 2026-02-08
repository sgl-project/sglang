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

#include CUDA_STD_HEADER(cassert)

#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(utility)
#else
#include <utility>
#endif

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This base class is meant to define the concept required of the
/// EpilogueStreamkWithBroadcast::OutputOp
template <
  typename ElementC_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementZ_,
  typename ElementT_,
  int ElementsPerAccess,
  bool StoreZ = true,
  bool StoreT = true
>
struct EpilogueStreamkWithBroadcastOpBase : EpilogueWithBroadcastOpBase<
                                            ElementC_,
                                            ElementAccumulator_,
                                            ElementCompute_,
                                            ElementZ_,
                                            ElementT_,
                                            ElementsPerAccess,
                                            StoreZ,
                                            StoreT
                                            > 
{

  /// Parameters structure - required
  struct Params { };

  //
  // Methods
  //

  /// Constructor from Params
  EpilogueStreamkWithBroadcastOpBase(Params const &params_) { }
};

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator with bias vector broadcast over columns.
///
/// Computes the following:
///
///
///  Z, T = OutputOp(AB, C, Broadcast)
///
///  if (ElementwiseOp::kStoreZ) {
///    store(converted_u);
///  }  
///
///  if (ElementwiseOp::kStoreT) {
///    store(v);
///  }  
///
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors (z)
  typename TensorTileIterator_,             ///< Additional tile iterator for tensor-valued operands (t)
  typename ElementVector_,                  ///< Pointer to broadcast vector
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator - concept is EpilogueWithBroadcastOp
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value),
  bool IsSingleSource = OutputOp_::kIsSingleSource
>
class EpilogueStreamkWithBroadcast;


/////////////////////////////////////////////////////////////////////////////////////////////////

/// EpilogueStreamkWithBroadcast: Two sources

template <
  typename Shape_,
  typename WarpMmaOperator_,
  int PartitionsK,
  typename OutputTileIterator_,
  typename TensorTileIterator_,
  typename ElementVector_,
  typename AccumulatorFragmentIterator_,
  typename WarpTileIterator_,
  typename SharedLoadIterator_,
  typename OutputOp_,
  typename Padding_,
  int FragmentsPerPartition,
  int IterationsUnroll
>
class EpilogueStreamkWithBroadcast<
  Shape_,
  WarpMmaOperator_,
  PartitionsK,
  OutputTileIterator_,
  TensorTileIterator_,
  ElementVector_,
  AccumulatorFragmentIterator_,
  WarpTileIterator_,
  SharedLoadIterator_,
  OutputOp_,
  Padding_,
  FragmentsPerPartition,
  IterationsUnroll,
  false
> : 
  public EpilogueWithBroadcast<
    Shape_,
    WarpMmaOperator_,
    PartitionsK,
    OutputTileIterator_,
    TensorTileIterator_,
    ElementVector_,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    SharedLoadIterator_,
    OutputOp_,
    Padding_,
    FragmentsPerPartition,
    IterationsUnroll,
    false>,
  public EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>
{

public:

  using Base = EpilogueWithBroadcast<
    Shape_,
    WarpMmaOperator_,
    PartitionsK,
    OutputTileIterator_,
    TensorTileIterator_,
    ElementVector_,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    SharedLoadIterator_,
    OutputOp_,
    Padding_,
    FragmentsPerPartition,
    IterationsUnroll,
    false>;

  using BaseStreamK = EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>;

  using Shape = Shape_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using TensorTileIterator = TensorTileIterator_;
  using ElementVector = ElementVector_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename Base::AccumulatorFragmentIterator::Fragment;

  /// Shared storage structure (shadows base) with additional SMEM buffer for reduction
  using SharedStorage = typename Base::SharedStorage;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueStreamkWithBroadcast(
    SharedStorage &shared_storage,                    ///< Shared storage object    
    int thread_idx,                                   ///< ID of a thread within the threadblock
    int warp_idx,                                     ///< ID of warp within threadblock
    int lane_idx                                      ///< Id of thread within warp
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    BaseStreamK(thread_idx)
  { }


  /// Aggregates the accumulator sets shared by peer blocks in the global workspace,
  /// performing epilogue computations, writing to output
  CUTLASS_DEVICE
  void reduce(
      int peer_idx_begin,
      int peer_idx_end,
      int reduce_fragment_idx,
      void *element_workspace,
      OutputOp const &output_op,                      ///< Output operator
      ElementVector const * broadcast_ptr,            ///< Broadcast vector
      OutputTileIterator destination_iterator,        ///< Tile iterator for destination
      OutputTileIterator source_iterator1,            ///< Tile iterator for first  source accumulator matrix
      OutputTileIterator source_iterator2,            ///< Tile iterator for second source accumulator matrix
      TensorTileIterator tensor_iterator,             ///< Threadblock tile iterator for additional tensor operand
      MatrixCoord const &problem_size =               ///< Problem size needed to guard against out-of-bounds accesses
          MatrixCoord(Shape::kM, Shape::kN),
      MatrixCoord const &threadblock_offset =         ///< Threadblock's initial offset within the problem size space
          MatrixCoord()) 
  {
    // Reduce peer accumulator fragments into one fragment
    AccumulatorFragment accum_fragment;
    BaseStreamK::reduce(accum_fragment, peer_idx_begin, peer_idx_end, reduce_fragment_idx, element_workspace);

    // Store fragment to shared memory
    this->warp_tile_iterator_.store(accum_fragment);

    __syncthreads();

    Base::reduce(reduce_fragment_idx, output_op, broadcast_ptr, destination_iterator, source_iterator1, source_iterator2, tensor_iterator, problem_size, threadblock_offset);
    
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// EpilogueStreamkWithBroadcast: Single source

template <
  typename Shape_,
  typename WarpMmaOperator_,
  int PartitionsK,
  typename OutputTileIterator_,
  typename TensorTileIterator_,
  typename ElementVector_,
  typename AccumulatorFragmentIterator_,
  typename WarpTileIterator_,
  typename SharedLoadIterator_,
  typename OutputOp_,
  typename Padding_,
  int FragmentsPerPartition,
  int IterationsUnroll
>
class EpilogueStreamkWithBroadcast<
  Shape_,
  WarpMmaOperator_,
  PartitionsK,
  OutputTileIterator_,
  TensorTileIterator_,
  ElementVector_,
  AccumulatorFragmentIterator_,
  WarpTileIterator_,
  SharedLoadIterator_,
  OutputOp_,
  Padding_,
  FragmentsPerPartition,
  IterationsUnroll,
  true
> : 
  public EpilogueWithBroadcast<
    Shape_,
    WarpMmaOperator_,
    PartitionsK,
    OutputTileIterator_,
    TensorTileIterator_,
    ElementVector_,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    SharedLoadIterator_,
    OutputOp_,
    Padding_,
    FragmentsPerPartition,
    IterationsUnroll,
    true>,
  public EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>
{

public:

  using Base = EpilogueWithBroadcast<
    Shape_,
    WarpMmaOperator_,
    PartitionsK,
    OutputTileIterator_,
    TensorTileIterator_,
    ElementVector_,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    SharedLoadIterator_,
    OutputOp_,
    Padding_,
    FragmentsPerPartition,
    IterationsUnroll,
    true>;

  using BaseStreamK = EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>;

  using Shape = Shape_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using TensorTileIterator = TensorTileIterator_;
  using ElementVector = ElementVector_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename Base::AccumulatorFragmentIterator::Fragment;

  /// Shared storage structure (shadows base) with additional SMEM buffer for reduction
  using SharedStorage = typename Base::SharedStorage;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueStreamkWithBroadcast(
    SharedStorage &shared_storage,                    ///< Shared storage object    
    int thread_idx,                                   ///< ID of a thread within the threadblock
    int warp_idx,                                     ///< ID of warp within threadblock
    int lane_idx                                      ///< Id of thread within warp
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    BaseStreamK(thread_idx)
  { }


  /// Aggregates the accumulator sets shared by peer blocks in the global workspace,
  /// performing epilogue computations, writing to output
  CUTLASS_DEVICE
  void reduce(
      int peer_idx_begin,
      int peer_idx_end,
      int reduce_fragment_idx,
      void *element_workspace,
      OutputOp const &output_op,                      ///< Output operator
      ElementVector const * broadcast_ptr,            ///< Broadcast vector
      OutputTileIterator destination_iterator,        ///< Tile iterator for destination
      OutputTileIterator source_iterator,             ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
      TensorTileIterator tensor_iterator,             ///< Threadblock tile iterator for additional tensor operand
      MatrixCoord const &problem_size =               ///< Problem size needed to guard against out-of-bounds accesses
          MatrixCoord(Shape::kM, Shape::kN),
      MatrixCoord const &threadblock_offset =         ///< Threadblock's initial offset within the problem size space
          MatrixCoord()) 
  {
    // Reduce peer accumulator fragments into one fragment
    AccumulatorFragment accum_fragment;
    BaseStreamK::reduce(accum_fragment, peer_idx_begin, peer_idx_end, reduce_fragment_idx, element_workspace);

    // Store fragment to shared memory
    this->warp_tile_iterator_.store(accum_fragment);

    __syncthreads();

    Base::reduce(reduce_fragment_idx, output_op, broadcast_ptr, destination_iterator, source_iterator, tensor_iterator, problem_size, threadblock_offset);
    
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
