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
#include "cutlass/numeric_types.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename ElementAccumulator_,
  typename ElementOutput_,
  typename ThreadBlockShape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  bool ReduceKForA_
>
class EpilogueGemmKReduction {

public:

  using ThreadBlockShape = ThreadBlockShape_;
  using WarpMmaOperator = WarpMmaOperator_;
  using WarpShape = typename WarpMmaOperator::Shape;
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// Accumulator element
  using ElementAccumulator = ElementAccumulator_;

  /// Output element
  using ElementOutput = ElementOutput_;

  /// Output access size
  static int const kElementsPerAccess = 1;

  static bool const kReduceKForA = ReduceKForA_;

  static int const kThreadBlockSize = kReduceKForA ? ThreadBlockShape::kM : ThreadBlockShape::kN;

  static int const kWarpSize = kReduceKForA ? WarpShape::kM : WarpShape::kN;

  static int const kIterations = kWarpSize / 8;

  using FragmentAccumulator = Array<ElementAccumulator, kIterations>;

private:

  int thread_offset_;
  ElementOutput* pointer_;
  int col_;
public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueGemmKReduction(
    int thread_idx,                   ///< ID of a thread within the threadblock
    int warp_idx,                     ///< ID of warp within threadblock
    int lane_idx,                     ///< Id of thread within warp
    int threadblock_offset,
    ElementOutput* pointer 
  )
  {
     col_ = lane_idx % 4;
     thread_offset_ = threadblock_offset * kThreadBlockSize
                    + warp_idx * kWarpSize 
                    + lane_idx / 4 + col_ * 8;

     pointer_ = pointer + LongIndex(thread_offset_);
  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    int size,
    FragmentAccumulator &gemm_k_with_reduction_accumulation,
    bool LoadForSerialSplitK
  ) {
      bool guard[kIterations / 4];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kIterations / 4; ++i) {
        guard[i] = ((thread_offset_ + i * 32) < size);
      }

      Array<ElementOutput, kIterations / 4> source;
      source.clear();

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kIterations / 4; ++i) {
        ElementOutput *source_ptr = reinterpret_cast<ElementOutput *>(&source);
        cutlass::arch::global_load<ElementOutput, sizeof(ElementOutput)>(
                                                  source_ptr[i],
                                                  (void *)(pointer_ + i * 32),
                                                  guard[i] && LoadForSerialSplitK);

      }

      FragmentAccumulator sum = gemm_k_with_reduction_accumulation;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kIterations; ++i) {
        sum[i] += __shfl_xor_sync(0xffffffff, sum[i], 1);
        sum[i] += __shfl_xor_sync(0xffffffff, sum[i], 2);
      }

      Array<ElementAccumulator, kIterations / 4> intermediate;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kIterations / 4; ++i) {
        if (col_ == 0) {
          intermediate[i] = sum[0 + i * 4];
        }
  
        if (col_ == 1) {
          intermediate[i] = sum[1 + i * 4];
        }
  
        if (col_ == 2) {
          intermediate[i] = sum[2 + i * 4];
        }
  
        if (col_ == 3) {
          intermediate[i] = sum[3 + i * 4];
        }
      }

      NumericArrayConverter<ElementAccumulator, ElementOutput, kIterations / 4> source_converter;
      Array<ElementAccumulator, kIterations / 4> converted_source = source_converter(source);

      plus<Array<ElementAccumulator, kIterations / 4>> plus_source;
      intermediate = plus_source(intermediate, converted_source);

      NumericArrayConverter<ElementOutput, ElementAccumulator, kIterations / 4> converter;
      Array<ElementOutput, kIterations / 4> result = converter(intermediate);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kIterations / 4; ++i) {
        cutlass::arch::global_store<ElementOutput, sizeof(ElementOutput)>(result[i], 
                                                (void *)(pointer_ + i * 32), guard[i]);
      }
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
