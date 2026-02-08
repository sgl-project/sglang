
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
    \brief Sparse GEMM with visitor.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/kernel/sparse_gemm.h"
#include "cutlass/gemm/kernel/params_sparse_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Sparse Gemm that compute the epilogue visitor functor
template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct SparseGemmWithEpilogueVisitor : public SparseGemm<Mma_, Epilogue_, ThreadblockSwizzle_, false>  {

  using Base = SparseGemm<Mma_, Epilogue_, ThreadblockSwizzle_, false>;

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using FusionCallbacks = typename Epilogue::FusionCallbacks;

  using ParamsA = typename Mma::IteratorA::Params;
  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using ParamsB = typename Mma::IteratorB::Params;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using ParamsE = typename Mma::IteratorE::Params;
  using TensorRefE = typename Mma::IteratorE::TensorRef;

  static int const kSparse = Base::kSparse;
  static int const kElementsPerElementE = Base::kElementsPerElementE;
  using SharedStorage = typename Base::SharedStorage;

  /// Parameters structure
  struct Params : public SparseParamsBase<
      ThreadblockSwizzle, ParamsA, TensorRefA, ParamsB, TensorRefB,
      ParamsE, TensorRefE> {

    using Base = SparseParamsBase<
        ThreadblockSwizzle, ParamsA, TensorRefA, ParamsB, TensorRefB,
        ParamsE, TensorRefE>;

    //
    // Data members
    //

    typename FusionCallbacks::Params output_op;
    cute::Shape<int32_t,int32_t,int32_t> problem_shape;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Mma::IteratorE::TensorRef ref_E,
      typename FusionCallbacks::Arguments output_op = typename FusionCallbacks::Arguments()
    ):
      Base(problem_size, grid_tiled_shape, ref_A, ref_B, ref_E, Mma::Shape::kK),
      output_op(FusionCallbacks::to_underlying_arguments(problem_size, output_op, nullptr /*workspace*/)),
      problem_shape(problem_size.m(), problem_size.n(), 1) {
    }
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  SparseGemmWithEpilogueVisitor() { }

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

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size / kSparse,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    cutlass::MatrixCoord tb_offset_E{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size / kSparse,
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(), 
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_B.row() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A, B, and E operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k / kSparse},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    typename Mma::IteratorE iterator_E(
        params.params_E, params.ref_E.data(),
        {params.problem_size.m(),
         problem_size_k / kSparse / kElementsPerElementE},
        thread_idx, tb_offset_E);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    if (gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E, accumulators);
    }

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    //
    // Epilogue
    //

    Epilogue epilogue(
      params.output_op,
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Execute the epilogue operator to update the destination tensor.
    epilogue(accumulators, threadblock_tile_offset, params.problem_shape, thread_idx);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass
