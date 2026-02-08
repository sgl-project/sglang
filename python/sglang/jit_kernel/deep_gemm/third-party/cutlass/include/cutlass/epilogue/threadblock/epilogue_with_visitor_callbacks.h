/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/epilogue/threadblock/epilogue_base.h"


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {
namespace detail {

struct EVT2xBase { };

template <class T>
static constexpr bool is_2x_evt_v = platform::is_base_of<EVT2xBase, T>::value;

} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename DefaultEpilogue,                 ///< Default Epilogue Descriptor
  typename FusionCallbacks_,                ///< The called fusion callbacks
  int Stages = 2,                           ///< Software pipeline stages for epilogue
  int IterationsUnroll = true               ///< Used to reduce binary size when epilogue op is large
>
class EpilogueWithVisitorCallbacks :
  public EpilogueBase<
    typename DefaultEpilogue::Shape,
    typename DefaultEpilogue::WarpMmaOperator::Shape,
    DefaultEpilogue::kPartitionsK,
    typename DefaultEpilogue::AccumulatorFragmentIterator,
    typename DefaultEpilogue::WarpTileIterator,
    typename DefaultEpilogue::Padding,
    DefaultEpilogue::kFragmentsPerIteration>,
  public EpilogueBaseStreamK<
    typename DefaultEpilogue::Shape,
    DefaultEpilogue::kPartitionsK,
    typename DefaultEpilogue::WarpMmaOperator,
    typename DefaultEpilogue::AccumulatorFragmentIterator>,
  public detail::EVT2xBase
   {

public:

  static_assert(Stages <= 2, "Sm80 EVT only support upto 2 Stages.");

  // Whether the epilogue is pipelined
  static bool constexpr Pipelined = Stages > 1;

  using FusionCallbacks = FusionCallbacks_;

  using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;
  // Number of epilogue iterations. 
  // Each iteration processes a 8xThreadblockTile::kN output tile
  static const int kIterations = OutputTileIterator::kIterations;

  using Base = EpilogueBase<
    typename DefaultEpilogue::Shape,
    typename DefaultEpilogue::WarpMmaOperator::Shape,
    DefaultEpilogue::kPartitionsK,
    typename DefaultEpilogue::AccumulatorFragmentIterator,
    typename DefaultEpilogue::WarpTileIterator,
    typename DefaultEpilogue::Padding,
    DefaultEpilogue::kFragmentsPerIteration>;
  
  using BaseStreamK = EpilogueBaseStreamK<
    typename DefaultEpilogue::Shape,
    DefaultEpilogue::kPartitionsK,
    typename DefaultEpilogue::WarpMmaOperator,
    typename DefaultEpilogue::AccumulatorFragmentIterator>;

  static int const kPartitionsK = DefaultEpilogue::kPartitionsK;

  using AccumulatorFragmentIterator = typename DefaultEpilogue::AccumulatorFragmentIterator;
  using WarpTileIterator = typename DefaultEpilogue::WarpTileIterator;
  using SharedLoadIterator = typename DefaultEpilogue::SharedLoadIterator;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename WarpTileIterator::Element;

  struct OutputOp{
    using ElementAccumulator = ElementAccumulator;
    using Params = typename FusionCallbacks::Arguments;
  };

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

  // Output access size
  static int const kElementsPerAccess = DefaultEpilogue::kElementsPerAccess;

  /// Array type used by output functor
  using AccumulatorAccessType = Array<
    typename WarpTileIterator::Element, kElementsPerAccess>;

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;
  static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;

  using Params = typename FusionCallbacks::Params;

  static size_t constexpr kSmemStageOffset = sizeof(Base::SharedStorage) / sizeof(ElementAccumulator);
  static int constexpr kAccumulatorFragmentCount = AccumulatorTile::kElements / (kIterations * AccumulatorAccessType::kElements) / kPartitionsK;

  struct SharedStorage {
    typename Base::SharedStorage acc_smem[Stages];
    typename FusionCallbacks::SharedStorage callback_smem;
  };

private:

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;
  FusionCallbacks fusion_callbacks;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueWithVisitorCallbacks(
    const Params &params_callbacks,   ///< Epilogue Visitor params
    SharedStorage &shared_storage,    ///< Shared storage object
    int thread_idx,                   ///< ID of a thread within the threadblock
    int warp_idx,                     ///< ID of warp within threadblock
    int lane_idx                      ///< Id of thread within warp
  ):
    Base(shared_storage.acc_smem[0], thread_idx, warp_idx, lane_idx),
    BaseStreamK(thread_idx),
    shared_load_iterator_(shared_storage.acc_smem[0].reference(), thread_idx),
    fusion_callbacks(params_callbacks, shared_storage.callback_smem)
  { }

  /// Aggregates the accumulator sets shared by peer blocks in the global workspace,
  /// performing epilogue computations, writing to output
  template <class ProblemShape>
  CUTLASS_DEVICE
  void reduce(
      int peer_idx_begin,
      int peer_idx_end,
      int reduce_fragment_idx,
      void *element_workspace,
      cutlass::gemm::GemmCoord threadblock_tile_offset,
      ProblemShape problem_shape,
      int thread_idx) 
  {
    auto callbacks = fusion_callbacks.get_callbacks(
      threadblock_tile_offset,
      thread_idx,
      problem_shape
    );

    callbacks.begin_epilogue();
    // Reduce peer accumulator fragments into one fragment
    AccumulatorFragment accum_fragment;
    BaseStreamK::reduce(accum_fragment, peer_idx_begin, peer_idx_end, reduce_fragment_idx, element_workspace);

    // Store fragment to shared memory
    this->warp_tile_iterator_.store(accum_fragment);

    __syncthreads();

    callbacks.begin_step(reduce_fragment_idx);

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

    //
    // Iterate over output fragment
    //

    AccumulatorAccessType const *accum_frag_ptr =
      reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < kAccumulatorFragmentCount; ++idx) {
      int row_idx = idx / SharedLoadIterator::ThreadMap::Iterations::kColumn;
      int col_idx = idx % SharedLoadIterator::ThreadMap::Iterations::kColumn;

      // Start a new row of the output fragment
      if (!col_idx) {
        callbacks.begin_row(row_idx);
      }

      callbacks.visit(
        reduce_fragment_idx,
        row_idx,
        col_idx,
        idx,
        accum_frag_ptr[idx]
      );

      // End the row of the output fragment
      if (col_idx + 1 == SharedLoadIterator::ThreadMap::Iterations::kColumn) {
        callbacks.end_row(row_idx);
      }
    }

    callbacks.end_step(reduce_fragment_idx);
    callbacks.end_epilogue();
  }

  /// Streams the result to global memory
  template <class ProblemShape>
  CUTLASS_DEVICE
  void operator()(
    AccumulatorTile const &accumulators,
    cutlass::gemm::GemmCoord threadblock_tile_offset,
    ProblemShape problem_shape,
    int thread_idx
    ) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    auto callbacks = fusion_callbacks.get_callbacks(
      threadblock_tile_offset,
      thread_idx,
      problem_shape
    );

    callbacks.begin_epilogue();

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

    if constexpr(Pipelined){
      __syncthreads();

      //
      // Pipeline Prologue
      //
      size_t warp_iterator_offset = kSmemStageOffset;
      size_t smem_iterator_offset = kSmemStageOffset;
      callbacks.begin_step(0);
    
      acc2smem_source_needed<cutlass::make_index_sequence<kIterations>>::push(
            0, accum_fragment_iterator, this->warp_tile_iterator_);
      
      this->warp_tile_iterator_.add_pointer_offset(warp_iterator_offset);
      warp_iterator_offset = -warp_iterator_offset;

      //
      // Pipeline Loop
      //

      #ifdef __clang__
      #pragma clang diagnostic push
      #pragma clang diagnostic ignored "-Wcuda-compat"
      // Turn off clang warning about loop unroll argument using parens.
      #endif

      #pragma unroll(IterationsUnroll ? kIterations : 1)
      for (int iter_idx = 1; iter_idx < kIterations + 1; ++iter_idx) {

        __syncthreads();

        // Skip the load for epilogue
        if (iter_idx < kIterations) {
          callbacks.begin_step(iter_idx);

          acc2smem_source_needed<cutlass::make_index_sequence<kIterations>>::push(
              iter_idx, accum_fragment_iterator, this->warp_tile_iterator_);

          this->warp_tile_iterator_.add_pointer_offset(warp_iterator_offset);
          warp_iterator_offset = -warp_iterator_offset;
        }
        
        typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];

        shared_load_iterator_.load(aligned_accum_fragment[0]);
        // If the number of k-slices is > 1 - perform a reduction amongst the k-slices
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
        shared_load_iterator_.add_pointer_offset(smem_iterator_offset);
        smem_iterator_offset = -smem_iterator_offset;
        
        //
        // Iterate over output fragments
        //

        AccumulatorAccessType const *accum_frag_ptr =
          reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < kAccumulatorFragmentCount; ++idx) {

          int row_idx = idx / SharedLoadIterator::ThreadMap::Iterations::kColumn;
          int col_idx = idx % SharedLoadIterator::ThreadMap::Iterations::kColumn;

          // Start a new row of the output fragment
          if (!col_idx) {
            callbacks.begin_row(row_idx);
          }

          callbacks.visit(
            iter_idx-1,
            row_idx,
            col_idx,
            idx,
            accum_frag_ptr[idx]
          );

          // End the row of the output fragment
          if (col_idx + 1 == SharedLoadIterator::ThreadMap::Iterations::kColumn) {
            callbacks.end_row(row_idx);
          }
        }

        //
        // Conclude the step
        //

        callbacks.end_step(iter_idx-1);
      }

      #ifdef __clang__
      #pragma clang diagnostic pop
      #endif

    } else {

      #ifdef __clang__
      #pragma clang diagnostic push
      #pragma clang diagnostic ignored "-Wcuda-compat"
      // Turn off clang warning about loop unroll argument using parens.
      #endif

      #pragma unroll(IterationsUnroll ? kIterations : 1)
      for (int iter_idx = 0; iter_idx < kIterations; ++iter_idx) {

        //
        // Load the source
        //

        callbacks.begin_step(iter_idx);

        //
        // Convert and store fragment
        //

        __syncthreads();

        acc2smem_source_needed<cutlass::make_index_sequence<kIterations>>::push(
            iter_idx, accum_fragment_iterator, this->warp_tile_iterator_);

        __syncthreads();

        //
        // Load fragments from shared memory
        //

        typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];

        shared_load_iterator_.load(aligned_accum_fragment[0]);
        // If the number of k-slices is > 1 - perform a reduction amongst the k-slices
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
        // Iterate over output fragments
        //

        AccumulatorAccessType const *accum_frag_ptr =
          reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment[0]);

        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < kAccumulatorFragmentCount; ++idx) {

          int row_idx = idx / SharedLoadIterator::ThreadMap::Iterations::kColumn;
          int col_idx = idx % SharedLoadIterator::ThreadMap::Iterations::kColumn;

          // Start a new row of the output fragment
          if (!col_idx) {
            callbacks.begin_row(row_idx);
          }

          callbacks.visit(
            iter_idx,
            row_idx,
            col_idx,
            idx,
            accum_frag_ptr[idx]
          );

          // End the row of the output fragment
          if (col_idx + 1 == SharedLoadIterator::ThreadMap::Iterations::kColumn) {
            callbacks.end_row(row_idx);
          }
        }

        //
        // Conclude the step
        //

        callbacks.end_step(iter_idx);
      }

      #ifdef __clang__
      #pragma clang diagnostic pop
      #endif

    }

    callbacks.end_epilogue();
  }

private:


  template<class Seq>
  struct acc2smem_source_needed;

  template <size_t... Seq>
  struct acc2smem_source_needed<cutlass::index_sequence<Seq...>> {
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
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                     AccumulatorFragmentIterator const &iterator_begin,
                     WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
    }
  };
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
