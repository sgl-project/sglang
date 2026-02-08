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
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator without splitk
template <
    /// Shape of threadblock tile (concept: GemmShape)
    typename Shape_,
    /// Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
    typename WarpMmaOperator_,
    /// Number of partitions of the K dimension
    int PartitionsK,
    /// Tile iterator reading and writing output tensors
    typename OutputTileIterator_,
    /// Fragment iterator selecting accumulators
    typename AccumulatorFragmentIterator_,
    /// Output operator
    typename OutputOp_,
    /// Number of interleaved k
    int InterleavedK>
class InterleavedEpilogue :
  public EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>
{
public:

  using BaseStreamK = EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using OutputTileIterator = OutputTileIterator_;
  using OutputOp = OutputOp_;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

  /// Accumulator element
  using ElementAccumulator = typename AccumulatorTile::Element;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef =
      typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Array type used to output
  using OutputAccessType = Array<typename OutputTileIterator::Element,
                                 OutputTileIterator::kElementsPerAccess>;

  /// Array type used by output functor
  using AccumulatorAccessType =
      Array<ElementAccumulator, OutputTileIterator::kElementsPerAccess>;

  /// Number of warps
  using WarpCount =
      gemm::GemmShape<Shape::kM / WarpMmaOperator::Shape::kM,
                      Shape::kN / WarpMmaOperator::Shape::kN, kPartitionsK>;

public:

  static_assert(OutputTileIterator::kElementsPerAccess,
                "This must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements %
                  OutputTileIterator::kElementsPerAccess),
                "Divisibility");

public:

  /// Aspect for when epilogue source is not needed
  struct SourceAspectNotNeeded
  {
    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNotNeeded()
    {}

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename AccumulatorFragmentIterator::Fragment const &aligned_accum_fragment)
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
      typename AccumulatorFragmentIterator::Fragment const &aligned_accum_fragment,
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

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename AccumulatorFragmentIterator::Fragment const &aligned_accum_fragment)
    {
      // Load addend source fragment from global memory
      source_iterator.load(source_fragment);
      ++source_iterator;

      apply_output_operator(output_fragment, output_op, aligned_accum_fragment, source_fragment);
    }
  };


  /// Shared storage allocation needed by the epilogue
  struct SharedStorage {};


public:

  /// Constructor
  CUTLASS_DEVICE
  InterleavedEpilogue(
      SharedStorage &shared_storage,  ///< Shared storage object
      int thread_idx,                 ///< ID of a thread within the threadblock
      int warp_idx,                   ///< ID of warp within threadblock
      int lane_idx)                   ///< Id of thread within warp
  :
      BaseStreamK(thread_idx)
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
    // Redcuce peer accumulator fragments into one fragment
    AccumulatorFragment accum_fragment;
    BaseStreamK::reduce(accum_fragment, peer_idx_begin, peer_idx_end, reduce_fragment_idx, element_workspace);

    // Source-fragment data (zero-initialized for scenarios where the
    // output operator allows us to skip loading it from global input)
    typename OutputTileIterator::Fragment source_fragment;
    source_fragment.clear();

    if (output_op.is_source_needed())
    {
      source_iterator += reduce_fragment_idx;
      source_iterator.load(source_fragment);
    }

    // Compute the output result
    typename OutputTileIterator::Fragment output_fragment;

    // Apply the output operator
    SourceAspectNeeded::apply_output_operator(output_fragment, output_op, accum_fragment, source_fragment);

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


  /// Streams the result to global memory
  template <typename SourceAspect>
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    SourceAspect source)
  {
    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

      //
      // Convert fragment
      //

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;

      //
      // Compute the output result
      //

      typename OutputTileIterator::Fragment output_fragment;
      source.apply_output_operator(output_fragment, output_op, accum_fragment);

      //
      // Store the final result
      //

      destination_iterator.set_iteration_index(iter);
      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
