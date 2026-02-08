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
  \brief Epilogue for threadblock scoped GEMM/CONV to store accumulator in shared memory after
    applying scale, bias loaded from global memory and element-wise operations.

    This Epilogue is typically used in fused GEMM/CONV to stage the intermediate accumulator.

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

#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename SmemTileIterator_,               ///< Shared memory Tile iterator to output to shared memory
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename ScaleBiasIterator_,              ///< Iterator to load scale and bias from global memory
  typename OutputOp_                        ///< Output operator
>
class EpilogueSmemAccumulator {

public:

  using SmemTileIterator = SmemTileIterator_;

  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;

  using ScaleBiasIterator = ScaleBiasIterator_;

  using OutputOp = OutputOp_;

  /// Fragment of accumulator tile
  using FragmentAccumulator = typename AccumulatorFragmentIterator::Fragment;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Fragment of Scale and Bias loaded from global memory
  using FragmentScaleBias = typename ScaleBiasIterator::Fragment;

  static const bool PerChannelScale = (OutputOp::kScale ==
      epilogue::thread::ScaleType::OnlyAlphaPerChannelScaling);

  /// Constructor
  CUTLASS_DEVICE
  EpilogueSmemAccumulator() {}

  /// Streams the result to shared memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    SmemTileIterator smem_iterator,               ///< Tile iterator for destination in shared memory
    AccumulatorTile const &accumulator,          ///< Complete warp-level accumulator tile
    ScaleBiasIterator scale_iterator,             ///< iterator for scale vector in global memory
    ScaleBiasIterator bias_iterator) {            ///< iterator for bias vector in global memory
 
  
    // Fragment to load scale bias from global memory
    FragmentScaleBias tb_frag_scale;
    FragmentScaleBias tb_frag_bias;
      
    /// Fragment Iterator to load slice of accumulator tile
    AccumulatorFragmentIterator frag_iterator_accum(accumulator);
    FragmentAccumulator tb_frag_accum;
  
    /// Epilogue output fragment
    typename SmemTileIterator::Fragment tb_frag_smem;
  
    /// Load scale and bias from global memory
  
    if(PerChannelScale)
        scale_iterator.load(tb_frag_scale);
  
    bias_iterator.load(tb_frag_bias);
  
    /// Iterate over the accumulator tile and store to shared memory
    CUTLASS_PRAGMA_UNROLL
    for (int rid = 0; rid < AccumulatorFragmentIterator::TileIterations::kRow; ++rid) {
    
      CUTLASS_PRAGMA_UNROLL
      for (int cid = 0; cid < AccumulatorFragmentIterator::TileIterations::kColumn; ++cid) {
  
        using AccumulatorAccessType = typename OutputOp::FragmentAccumulator;
        using ScaleBiasAccessType = typename OutputOp::FragmentScaleBias;
        using FragmentSmemAccessType = typename OutputOp::FragmentOutput;
  
  
        ScaleBiasAccessType const * scale_frag_ptr =
          reinterpret_cast<ScaleBiasAccessType const *>(&tb_frag_scale);
        ScaleBiasAccessType const * bias_frag_ptr =
          reinterpret_cast<ScaleBiasAccessType const *>(&tb_frag_bias);
   
        FragmentSmemAccessType * smem_frag_ptr =  
          reinterpret_cast<FragmentSmemAccessType *>(&tb_frag_smem);
  
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < AccumulatorFragmentIterator::kIterationsPerTile; ++idx) {
          frag_iterator_accum.load(tb_frag_accum);
          ++frag_iterator_accum;
  
          AccumulatorAccessType const * accumulator_frag_ptr = 
            reinterpret_cast<AccumulatorAccessType const *>(&tb_frag_accum);
          const int kOutputIterations = FragmentAccumulator::kElements / OutputOp::kCount;
  
          CUTLASS_PRAGMA_UNROLL
          for (int it = 0; it < kOutputIterations; it++) {
            smem_frag_ptr[idx * kOutputIterations + it] = output_op(accumulator_frag_ptr[it],
                scale_frag_ptr[cid * kOutputIterations + it], bias_frag_ptr[cid * kOutputIterations + it]);
          }
        }
  
        smem_iterator.store(tb_frag_smem);
        ++smem_iterator;
  
      }
    }
  }

  /// Streams the result to shared memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    SmemTileIterator smem_iterator,               ///< Tile iterator for destination in shared memory
    AccumulatorTile const &accumulator) {          ///< Complete warp-level accumulator tile
 
    /// Fragment Iterator to load slice of accumulator tile
    AccumulatorFragmentIterator frag_iterator_accum(accumulator);
    FragmentAccumulator tb_frag_accum;
  
    /// Epilogue output fragment
    typename SmemTileIterator::Fragment tb_frag_smem;
  
    /// Iterate over the accumulator tile and store to shared memory
    CUTLASS_PRAGMA_UNROLL
    for (int rid = 0; rid < AccumulatorFragmentIterator::TileIterations::kRow; ++rid) {
    
      CUTLASS_PRAGMA_UNROLL
      for (int cid = 0; cid < AccumulatorFragmentIterator::TileIterations::kColumn; ++cid) {
  
        using AccumulatorAccessType = typename OutputOp::FragmentAccumulator;
        using FragmentSmemAccessType = typename OutputOp::FragmentOutput;
  
        FragmentSmemAccessType * smem_frag_ptr =  
          reinterpret_cast<FragmentSmemAccessType *>(&tb_frag_smem);
  
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < AccumulatorFragmentIterator::kIterationsPerTile; ++idx) {
          frag_iterator_accum.load(tb_frag_accum);
          ++frag_iterator_accum;
  
          AccumulatorAccessType const * accumulator_frag_ptr = 
            reinterpret_cast<AccumulatorAccessType const *>(&tb_frag_accum);
          const int kOutputIterations = FragmentAccumulator::kElements / OutputOp::kCount;
  
          CUTLASS_PRAGMA_UNROLL
          for (int it = 0; it < kOutputIterations; it++) {
            smem_frag_ptr[idx * kOutputIterations + it] = output_op(accumulator_frag_ptr[it]);
          }
        }
  
        smem_iterator.store(tb_frag_smem);
        ++smem_iterator;
  
      }
    }
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
 
