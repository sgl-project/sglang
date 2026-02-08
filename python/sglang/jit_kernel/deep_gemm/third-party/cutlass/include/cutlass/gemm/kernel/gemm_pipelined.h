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

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma, typename Epilogue, typename ThreadblockSwizzle>
CUTLASS_GLOBAL void GemmPipelined(
  cutlass::gemm::GemmCoord problem_size,
  cutlass::gemm::GemmCoord grid_tiled_shape,
  typename Mma::IteratorA::Params params_A,
  typename Mma::IteratorA::TensorRef ref_A,
  typename Mma::IteratorB::Params params_B,
  typename Mma::IteratorB::TensorRef ref_B,
  typename Epilogue::Params params_epilogue
  ) {

  // Shared storage needed by threadblock-scoped matrix multiply-accumulate
  __shared__ union {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  } shared_storage;

  // Compute threadblock location
  ThreadblockSwizzle threadblock_swizzle;

  int swizzle_log_tile = ThreadblockSwizzle().get_log_tile(grid_tiled_shape);

  cutlass::gemm::GemmCoord tb_tile_offset = threadblock_swizzle.get_tile_offset(swizzle_log_tile);

  if (grid_tiled_shape.m() <= tb_tile_offset.m() ||
    grid_tiled_shape.n() <= tb_tile_offset.n()) {

    return;
  }

  // Compute initial location in logical coordinates
  cutlass::MatrixCoord tb_offset_A{
    tb_tile_offset.m() * Mma::Shape::kM,
    tb_tile_offset.k()
  };

  cutlass::MatrixCoord tb_offset_B{
    tb_tile_offset.k(),
    tb_tile_offset.n() * Mma::Shape::kN
  };

  // Compute position within threadblock
  int tb_thread_id = threadIdx.x;

  // Construct iterators to A and B operands
  typename Mma::IteratorA iterator_A(
    params_A,
    ref_A.data(),
    {problem_size.m(), problem_size.k()},
    tb_thread_id,
    tb_offset_A);

  typename Mma::IteratorB iterator_B(
    params_B,
    ref_B.data(),
    {problem_size.k(), problem_size.n()},
    tb_thread_id,
    tb_offset_B);

  int warp_id = canonical_warp_idx_sync();
  int lane_id = threadIdx.x % 32;

  //
  // Main loop
  //

  // Construct thread-scoped matrix multiply
  Mma mma(shared_storage.main_loop, tb_thread_id, warp_id, lane_id);

  typename Mma::FragmentC accumulators;

  accumulators.clear();

  // Compute threadblock-scoped matrix multiply-add
  mma(problem_size, accumulators, iterator_A, iterator_B, accumulators);

  //
  // Epilogue
  //

  Epilogue epilogue(
    params_epilogue, 
    shared_storage.epilogue, 
    tb_thread_id, 
    warp_id, 
    lane_id);

  tb_tile_offset = threadblock_swizzle.get_tile_offset(swizzle_log_tile);

  //assume identity swizzle
  MatrixCoord threadblock_offset(
    tb_tile_offset.m() * Mma::Shape::kM,
    tb_tile_offset.n() * Mma::Shape::kN
  );

  // run efficient epilogue
  epilogue({problem_size.m(), problem_size.n()}, accumulators, threadblock_offset);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass
