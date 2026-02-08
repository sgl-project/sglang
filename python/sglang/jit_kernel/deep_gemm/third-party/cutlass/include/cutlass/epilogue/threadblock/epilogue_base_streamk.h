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
  \brief Basic subset of epilogue functionality for supporting StreamK decompositions
*/


#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/block_striped.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////


/// StreamK epilogue functionality for cross-block accumulator fragment reduction
template <
  typename Shape,                          ///< Shape of threadblock tile (concept: GemmShape)
  int PartitionsK,
  typename WarpMmaOperator,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  typename AccumulatorFragmentIterator>    ///< Iterator for enumerating fragments within the per-thread tile of raw accumulators
class EpilogueBaseStreamK
{

protected:

  /// The per-thread tile of raw accumulators
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Number of warps
  using WarpCount = gemm::GemmShape<
                        Shape::kM / WarpMmaOperator::Shape::kM,
                        Shape::kN / WarpMmaOperator::Shape::kN,
                        PartitionsK>;

  /// Number of threads per block
  static int const kBlockThreads = 32 * WarpCount::kCount;

  /// Numerical accumulation element type
  using ElementAccumulator = typename WarpMmaOperator::ElementC;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

public:

  /// Number of AccumulatorTile fragments per thread
  static int const kAccumulatorFragments = AccumulatorFragmentIterator::Policy::kIterations;

protected:

  /// Number of AccumulatorTile fragments per block output tile
  static int const kOutputTileFragments = kBlockThreads * kAccumulatorFragments;

  /// Block-striped transfer utility for sharing AccumulatorFragment
  using BlockStripedT = BlockStriped<kBlockThreads, AccumulatorFragment>;

  /// AccumulatorFragment stride in the shared workspace between different peer blocks (each thread block can share accumulators for up to two block output tiles)
  static const int kPeerFragmentStride = kOutputTileFragments * 2;

public:

  /// Workspace bytes per thread block
  static size_t const kWorkspaceBytesPerBlock =sizeof(AccumulatorFragment) * kPeerFragmentStride;

public:

  /// Thread index in the threadblock
  int thread_idx;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueBaseStreamK(
      int thread_idx)                                       ///< ID of a thread within the threadblock
  :
      thread_idx(thread_idx)
  {}


  /// Aggregates the accumulator sets shared by peer blocks in the global workspace
  CUTLASS_DEVICE
  void reduce(
      AccumulatorFragment &accum_fragment,                  ///< [out] sum of all shared accumulator fragments for these peer partials
      int peer_idx_begin,
      int peer_idx_end,
      int reduce_fragment_idx,
      void *workspace_ptr)
  {
    plus<AccumulatorFragment> add_fragments;

    AccumulatorFragment *fragment_workspace = reinterpret_cast<AccumulatorFragment *>(workspace_ptr);

    int fragment_offset = (peer_idx_begin * kPeerFragmentStride) + (reduce_fragment_idx * kBlockThreads);

    // Load first peer fragment
    BlockStripedT::load(accum_fragment, fragment_workspace + fragment_offset, this->thread_idx);

    fragment_offset += kPeerFragmentStride;         // Move to next peer
    fragment_offset += kOutputTileFragments;        // Move to the set of fragments for this peer's "non-started" output tile

    // Reduce fragments from additional peers
    #pragma unroll 2
    for (; fragment_offset < peer_idx_end * kPeerFragmentStride; fragment_offset += kPeerFragmentStride)
    {
      // Load peer fragment
      AccumulatorFragment addend_fragment;
      BlockStripedT::load(addend_fragment, fragment_workspace + fragment_offset, this->thread_idx);

      // Add peer fragment
      accum_fragment = add_fragments(accum_fragment, addend_fragment);
    }
  }


  /// Shares the accumulator set with peers in the global workspace
  CUTLASS_DEVICE
  void share(
      int peer_idx,
      void *workspace_ptr,
      AccumulatorTile const &accumulators,
      bool started_tile)                      ///< Whether this thread block computed the first work volume for the current output tile
  {
    AccumulatorFragment *fragment_workspace = reinterpret_cast<AccumulatorFragment *>(workspace_ptr);

    int fragment_offset = peer_idx * kPeerFragmentStride;

    if (!started_tile) {
      // Move to the set of fragments for the "non-started" output tile
      fragment_offset += kOutputTileFragments;
    }

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    // Convert raw accumulator tile to fragments and store
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < kAccumulatorFragments; ++iter)
    {
      // Acquire reordered accumulator fragment
      AccumulatorFragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;

      // Store accumulator fragment
      BlockStripedT::store(fragment_workspace + fragment_offset, accum_fragment, this->thread_idx);

      fragment_offset += kBlockThreads;
    }
  }

};



////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
