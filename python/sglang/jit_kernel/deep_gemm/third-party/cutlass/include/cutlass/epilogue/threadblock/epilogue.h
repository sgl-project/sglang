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

  The shared memory resource is time-sliced across warps.
*/

#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)

#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

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
  typename OutputOp_,                       ///< Output operator
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)
>
class Epilogue :
  public EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::Shape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>,
  public EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>
{

public:

  using Base = EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::Shape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>;

  using BaseStreamK = EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// Number of warps per block
  using WarpCount = typename Base::WarpCount;

  /// Number of threads per block
  static int const kBlockThreads = 32 * WarpCount::kCount;

  /// Per-thread accumulator tile type
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Numerical accumulation element type
  using ElementAccumulator = typename WarpMmaOperator::ElementC;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

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

  /// Vector type used by the global output iterator
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Vector type used by the shared output iterator
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;

  static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;


public:

  static_assert(SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
    "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), 
    "Divisibility");

  static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");


public:

  /// Aspect for when epilogue source is not needed
  struct SourceAspectNotNeeded
  {
    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNotNeeded()
    {}

    // No-op
    CUTLASS_DEVICE
    void load() { }

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
      }
    }
  };


  /// Aspect for when epilogue source is needed
  struct SourceAspectNeeded
  {
    OutputTileIterator source_iterator;

    typename OutputTileIterator::Fragment source_fragment;

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    static void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
      typename OutputTileIterator::Fragment const &source_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      OutputAccessType const *source_frag_ptr =
        reinterpret_cast<OutputAccessType const *>(&source_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
      }
    }

    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNeeded(OutputTileIterator source_iterator) :
      source_iterator(source_iterator)
    {
      source_fragment.clear();
    }

    // Load addend source fragment from global memory
    CUTLASS_DEVICE
    void load() {
      source_iterator.load(source_fragment);
      ++source_iterator;
    }

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment)
    {
      apply_output_operator(output_fragment, output_op, aligned_accum_fragment, source_fragment);
    }
  };


private:

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

  /// Thread index in the threadblock
  int thread_idx;

  /// Warp index in the threadblock
  int warp_idx;

public:

  /// Constructor
  CUTLASS_DEVICE
  Epilogue(
      typename Base::SharedStorage &shared_storage,   ///< Shared storage object
      int thread_idx,                                 ///< ID of a thread within the threadblock
      int warp_idx,                                   ///< ID of warp within threadblock
      int lane_idx)                                   ///< Id of thread within warp
  :
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      BaseStreamK(thread_idx),
      shared_load_iterator_(shared_storage.reference(), thread_idx),
      thread_idx(thread_idx),
      warp_idx(warp_idx)
  {}


  /// Aggregates the accumulator sets shared by peer blocks in the global workspace,
  /// performing epilogue computations, writing to output
  CUTLASS_DEVICE
  void reduce(
      int peer_idx_begin,
      int peer_idx_end,
      int reduce_fragment_idx,
      void *element_workspace,
      OutputOp const &output_op,                      ///< Output operator
      OutputTileIterator destination_iterator,        ///< Tile iterator for destination
      OutputTileIterator source_iterator)             ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
  {
    // Reduce peer accumulator fragments into one fragment
    AccumulatorFragment accum_fragment;
    BaseStreamK::reduce(accum_fragment, peer_idx_begin, peer_idx_end, reduce_fragment_idx, element_workspace);

    // Store fragment to shared memory
    this->warp_tile_iterator_.store(accum_fragment);

    __syncthreads();

    // Initialize/load source-fragment data
    typename OutputTileIterator::Fragment source_fragment;
    source_fragment.clear();

    if (output_op.is_source_needed())
    {
      source_iterator += reduce_fragment_idx;
      source_iterator.load(source_fragment);
    }

    // Load fragment from shared memory
    typename SharedLoadIterator::Fragment aligned_accum_fragment;
    shared_load_iterator_.load(aligned_accum_fragment);

    // Add fragments shared by other k partitions
    if (kPartitionsK > 1)
    {
      plus <typename SharedLoadIterator::Fragment> add_fragments;

      CUTLASS_PRAGMA_UNROLL
      for ( int i = 1; i < kPartitionsK; ++i) {
        typename SharedLoadIterator::Fragment aligned_addend_fragment;
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        shared_load_iterator_.load(aligned_addend_fragment);
        aligned_accum_fragment = add_fragments(aligned_accum_fragment, aligned_addend_fragment);
      }
    }

    // Compute the output result
    typename OutputTileIterator::Fragment output_fragment;

    // Apply the output operator
    SourceAspectNeeded::apply_output_operator(
        output_fragment,
        output_op,
        aligned_accum_fragment,
        source_fragment);

    // Store the final result
    destination_iterator += reduce_fragment_idx;
    destination_iterator.store(output_fragment);
  }


  /// Perform the epilogue computations and stream the result to global memory.
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators)            ///< Complete warp-level accumulator tile
  {
    operator()(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
  }


  /// Perform the epilogue computations and stream the result to global memory.  Implements
  /// two alternative codepaths, depending on whether the output op requires addend data to be loaded.
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator )            ///< Tile iterator for addend source
  {
    if (output_op.is_source_needed())
    {
      operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
    }
    else
    {
      operator()(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
    }
  }


  /// Perform the epilogue computations and stream the result to global memory.  Implements a
  /// single codepath, regardless of whether the output op requires addend data to be loaded
  CUTLASS_DEVICE
  void unified(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator )            ///< Tile iterator for addend source
  {
    if (!output_op.is_source_needed())
    {
      source_iterator.clear_mask();
      __syncthreads();  // Dummy (CUDA 11.0)
    }

    operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
  }

  template<class Seq>
  struct acc2smem;

  template <size_t... Seq>
  struct acc2smem<cutlass::index_sequence<Seq...>> {
    template<int Advance>
    CUTLASS_DEVICE
    static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                      WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                    AccumulatorFragmentIterator const &iterator_begin,
                    WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
    }
  };


  /// Streams the result to global memory
  template <typename SourceAspect>
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    SourceAspect source)
  {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

    #ifdef __clang__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wcuda-compat"
    // Turn off clangs warning about loop unroll argument using parens.
    #endif

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
    {
      //
      // Load the source
      //

        source.load();
      //
      // Convert and store fragment
      //

      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
        iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);

      if (kPartitionsK > 1) {
        plus <typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for ( int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }

        shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
      }

      //
      // Compute the output result
      //

      typename OutputTileIterator::Fragment output_fragment;
      source.apply_output_operator(output_fragment, output_op, aligned_accum_fragment[0]);

      //
      // Store the final result
      //

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
    
    #ifdef __clang__
    #pragma clang diagnostic pop
    #endif
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
