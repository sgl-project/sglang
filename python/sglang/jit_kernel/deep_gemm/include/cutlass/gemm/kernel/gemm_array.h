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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmArray {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::Element const * const * ptr_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::Element const * const * ptr_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Element const * const * ptr_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::Element * const * ptr_D;
    int64_t stride_D;
    typename OutputOp::Params epilogue;
    int batch_count;
    int gemm_k_iterations;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : 
      swizzle_log_tile(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size_,
      cutlass::gemm::GemmCoord const & grid_tiled_shape_,
      typename Mma::IteratorA::Element const * const * ptr_A_,
      typename Mma::IteratorA::Layout layout_A,
      typename Mma::IteratorB::Element const * const * ptr_B_,
      typename Mma::IteratorB::Layout layout_B,
      typename Epilogue::OutputTileIterator::Element const * const * ptr_C_,
      typename Epilogue::OutputTileIterator::Layout layout_C,
      typename Epilogue::OutputTileIterator::Element * const * ptr_D_,
      typename Epilogue::OutputTileIterator::Layout layout_D,
      typename OutputOp::Params epilogue_,
      int batch_count_
    ):
      problem_size(problem_size_),
      grid_tiled_shape(grid_tiled_shape_),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(layout_A),
      ptr_A(ptr_A_),
      params_B(layout_B),
      ptr_B(ptr_B_),
      params_C(layout_C),
      ptr_C(ptr_C_),
      params_D(layout_D),
      ptr_D(ptr_D_),
      epilogue(epilogue_),
      batch_count(batch_count_),
      gemm_k_iterations((problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK) {

    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  GemmArray() { } 

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }


    // Each CTA handles multiple batch indices to accommodate limited range of CUDA grid's Z dimension
    for (int batch_idx = threadblock_swizzle.get_batch_idx(); 
      batch_idx < params.batch_count; 
      batch_idx += gridDim.z) {

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        0
      };

      cutlass::MatrixCoord tb_offset_B{
        0,
        threadblock_tile_offset.n() * Mma::Shape::kN
      };

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        params.params_A,
        const_cast<typename Mma::IteratorA::Element *>(params.ptr_A[batch_idx]),
        params.problem_size.mk(),
        thread_idx,
        tb_offset_A);

      typename Mma::IteratorB iterator_B(
        params.params_B,
        const_cast<typename Mma::IteratorB::Element *>(params.ptr_B[batch_idx]),
        params.problem_size.kn(),
        thread_idx,
        tb_offset_B);

      //
      // Main loop
      //
      
      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = canonical_warp_idx_sync();

      int lane_idx = threadIdx.x % 32;
      
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentC accumulators;

      accumulators.clear();


      // Compute threadblock-scoped matrix multiply-add
      mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

      //
      // Epilogue
      //

      OutputOp output_op(params.epilogue);

      //
      // Masked tile iterators constructed from members
      //

      threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

      //assume identity swizzle
      MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.n() * Mma::Shape::kN
      );

      // Tile iterator writing to output tile
      typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        const_cast<typename Epilogue::OutputTileIterator::Element *>(params.ptr_C[batch_idx]),
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset
      );

      // Tile iterator writing to output tile
      typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        params.ptr_D[batch_idx],
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset
      );

      Epilogue epilogue(
        shared_storage.epilogue, 
        thread_idx, 
        warp_idx, 
        lane_idx);

      // run efficient epilogue
      epilogue(output_op, iterator_D, accumulators, iterator_C);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

