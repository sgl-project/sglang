/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../collective/fmha_common.hpp"

namespace cutlass::fmha::collective {

template<class Element, class ElementAccumulator, class TileShape_WG>
struct FmhaFwdEpilogue {

  static constexpr int Alignment = 16 / sizeof(Element);

  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<Element, ElementAccumulator, void>;
  using CollectiveEpilogueTMA = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_WG, Shape<_1,_1,_1>, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      void, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment,
      Element, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment,
      cutlass::epilogue::TmaWarpSpecialized,
      DefaultOperation
  >::CollectiveOp;

  struct Arguments {
    Element* ptr_O;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> dO;

    ElementAccumulator* ptr_LSE;
    cute::tuple<cute::_1, cute::tuple<int, int>> dLSE;
  };

  struct Params {
    ElementAccumulator* ptr_LSE;
    cute::tuple<cute::_1, cute::tuple<int, int>> dLSE;
  
    typename CollectiveEpilogueTMA::Params epilogue_TMA;
  };

  using TensorStorage = typename CollectiveEpilogueTMA::TensorStorage;
  using PipelineStorage = typename CollectiveEpilogueTMA::PipelineStorage;
  using LoadPipeline = typename CollectiveEpilogueTMA::LoadPipeline;
  static constexpr int TmaTransactionBytes = CollectiveEpilogueTMA::TmaTransactionBytes;

  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size, Arguments const& args, void* workspace = nullptr) {
    auto problem_size_o = make_shape(get<2>(problem_size), get<4>(problem_size), 1,
              make_shape(get<0>(problem_size), get<1>(problem_size)));
    typename CollectiveEpilogueTMA::Arguments args_tma{{}, args.ptr_O, args.dO, args.ptr_O, args.dO};
    return Params{
      args.ptr_LSE, args.dLSE,
      CollectiveEpilogueTMA::to_underlying_arguments(problem_size_o, args_tma, workspace)
    };
  }

  template<class TileShape, class BlkCoord, class ResultTuple, class TiledMma, class ProblemShape>
  CUTLASS_DEVICE void operator()(
      TileShape const& tile_shape, BlkCoord const& blk_coord,
      ResultTuple const& result, TiledMma const& tiled_mma,
      ProblemShape const& problem_size, Params const& params,
      LoadPipeline epi_load_pipeline,
      TensorStorage& epi_tensor_storage)
  {
    using X = Underscore;

    auto acc = get<0>(result);
    auto lse = get<1>(result);
  
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  
    int seqlen_q = get<2>(problem_size);
    int num_batch = get<0>(problem_size);
    int num_heads = get<1>(problem_size);
    // Epilogue for lse
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE),
        make_shape(seqlen_q, get<1>(tile_shape), make_shape(num_batch, num_heads)),
        make_stride(_1{}, _0{}, get<1>(params.dLSE)));
    Tensor gLSE_full = local_tile(mLSE, tile_shape, make_coord(_, _, _), Step<_1, _1, X>{});
    Tensor gLSE = gLSE_full(_, _, get<0>(blk_coord), get<1>(blk_coord), get<2>(blk_coord));
    Tensor tOgLSE = thr_mma.partition_C(gLSE);
    Tensor cO = make_identity_tensor(take<0,2>(tile_shape));
    Tensor tOcO = thr_mma.partition_C(cO);
    if (get<1>(tOcO(_0{})) == 0) {
      auto tOgLSE_mn = make_tensor(tOgLSE.data(), layout_acc_mn(tiled_mma, tOgLSE.layout()));
      auto tOcO_mn = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma, tOcO.layout()));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<0>(tOgLSE_mn); i++) {
        if (get<0>(tOcO_mn(i)) + get<0>(blk_coord) * get<0>(tile_shape) < get<2>(problem_size)) {
          tOgLSE_mn(i, _0{}) = lse(i);
        }
      }
    }
    auto problem_size_o = make_shape(get<2>(problem_size), get<4>(problem_size), _,
              make_shape(get<0>(problem_size), get<1>(problem_size)));

    CollectiveEpilogueTMA epilogue_tma(params.epilogue_TMA, epi_tensor_storage);

    using EpiStorePipeline = typename CollectiveEpilogueTMA::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename CollectiveEpilogueTMA::LoadPipelineState epi_load_pipe_consumer_state;
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
    epilogue_tma.store(
      epi_load_pipeline, epi_load_pipe_consumer_state,
      epi_store_pipeline, epi_store_pipe_producer_state,
      problem_size_o, tile_shape, make_coord(get<0>(blk_coord), _0{}, _, get<2>(blk_coord)),
      acc, tiled_mma, threadIdx.x % cutlass::NumThreadsPerWarpGroup,
      epi_tensor_storage
    );

    epilogue_tma.store_tail(
      epi_load_pipeline, epi_load_pipe_consumer_state_next,
      epi_store_pipeline, epi_store_pipe_producer_state_next
    );
  }
};

}  // namespace cutlass::fmha::collective
