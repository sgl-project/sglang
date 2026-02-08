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
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator without splitk
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename OutputOp_                        ///< Output operator
>
class FusedBiasActEpilogue {

public:

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using OutputOp = OutputOp_;

  /// Output layout is always row-major
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  
public:


  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), 
    "Divisibility");

public:

  /// Constructor
  CUTLASS_DEVICE
  FusedBiasActEpilogue(
  ){ }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    AccumulatorTile &accumulators,          ///< Complete warp-level accumulator tile
    AccumulatorTile & fused_bias_act_accumlators,
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
    
    bool need_bias = output_op.is_source_needed();

    if (need_bias)
      compute_source_needed_(output_op, accumulators, fused_bias_act_accumlators, source_iterator);
    else
      compute_source_no_needed_(output_op, accumulators, fused_bias_act_accumlators);


  }

  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    AccumulatorTile &accumulators,          ///< Complete warp-level accumulator tile
    AccumulatorTile & fused_bias_act_accumlators) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
    
    compute_source_no_needed_(output_op, accumulators, fused_bias_act_accumlators);
  }

  CUTLASS_DEVICE
  void compute_source_needed_(
    OutputOp const &output_op,                    ///< Output operator
    AccumulatorTile &accumulators,          ///< Complete warp-level accumulator tile
    AccumulatorTile & fused_bias_act_accumlators,
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
    
    typename OutputTileIterator::Fragment source_fragment;


    source_fragment.clear();

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    AccumulatorFragmentIterator fused_bias_act_fragment_iterator(fused_bias_act_accumlators);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

      source_iterator.load(source_fragment);
      ++source_iterator;

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;

      typename AccumulatorFragmentIterator::Fragment fused_bias_act_fragment;
      fused_bias_act_fragment = output_op(accum_fragment, source_fragment);

      fused_bias_act_fragment_iterator.store(fused_bias_act_fragment);
      ++fused_bias_act_fragment_iterator;
    }
  }

  CUTLASS_DEVICE
  void compute_source_no_needed_(
    OutputOp const &output_op,                    ///< Output operator
    AccumulatorTile &accumulators,          ///< Complete warp-level accumulator tile
    AccumulatorTile & fused_bias_act_accumlators) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
    

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    AccumulatorFragmentIterator fused_bias_act_fragment_iterator(fused_bias_act_accumlators);



    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < AccumulatorFragmentIterator::kIterations; ++iter) {

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;

      typename AccumulatorFragmentIterator::Fragment fused_bias_act_fragment;
      fused_bias_act_fragment = output_op(accum_fragment);

      fused_bias_act_fragment_iterator.store(fused_bias_act_fragment);
      ++fused_bias_act_fragment_iterator;
    }
  }

};




////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
