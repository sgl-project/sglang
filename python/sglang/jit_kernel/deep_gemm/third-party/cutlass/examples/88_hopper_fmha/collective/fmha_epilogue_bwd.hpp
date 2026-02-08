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
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "../collective/fmha_epilogue.hpp"

namespace cutlass::fmha::collective {

template<class Element, class ElementAccumulator, class TileShape_WG>
struct FmhaBwdEpilogueKV {

  static constexpr int Alignment = 16 / sizeof(Element);

  struct Arguments {
    Element* ptr_K;
    cute::tuple<int, int, int, cute::_1> dK;

    Element* ptr_V;
    cute::tuple<int, int, int, _1> dV;
  };

  //using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<Element, ElementAccumulator, void>;
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using DefaultOperation = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::first, Element, ElementAccumulator, RoundStyle>,
    cutlass::epilogue::fusion::Sm90AccFetch
  >;
  using CollectiveEpilogueTMA = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_WG, Shape<_1,_1,_1>, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        void, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment,
        Element, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment,
        cutlass::epilogue::TmaWarpSpecialized,
        DefaultOperation
    >::CollectiveOp;

  struct Params {
    typename CollectiveEpilogueTMA::Params epilogue_K;
    typename CollectiveEpilogueTMA::Params epilogue_V;
  };

  
  using TensorStorage = typename CollectiveEpilogueTMA::TensorStorage[2];
  using PipelineStorage = typename CollectiveEpilogueTMA::PipelineStorage;
  using LoadPipeline = typename CollectiveEpilogueTMA::LoadPipeline;
  static constexpr int TmaTransactionBytes = CollectiveEpilogueTMA::TmaTransactionBytes;

  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size, Arguments const& args, void* workspace = nullptr) {
    auto dK = make_stride(get<2>(args.dK), get<3>(args.dK),
        make_stride(get<0>(args.dK), get<1>(args.dK)));
    auto dV = make_stride(get<2>(args.dV), get<3>(args.dV),
        make_stride(get<0>(args.dV), get<1>(args.dV)));
      
    auto problem_size_kv = make_shape(get<3>(problem_size), get<4>(problem_size), 1,
              make_shape(get<0>(problem_size), get<1>(problem_size)));
    typename CollectiveEpilogueTMA::Arguments args_k{{}, args.ptr_K, dK, args.ptr_K, dK};
    typename CollectiveEpilogueTMA::Arguments args_v{{}, args.ptr_V, dV, args.ptr_V, dV};
    return Params{
      CollectiveEpilogueTMA::to_underlying_arguments(problem_size_kv, args_k, nullptr),
      CollectiveEpilogueTMA::to_underlying_arguments(problem_size_kv, args_v, nullptr)
    };
  }

  template<class TileShape, class BlkCoord, class ResultTuple, class TiledMma, class ProblemShape>
  CUTLASS_DEVICE void operator()(
      TileShape const& tile_shape, BlkCoord const& blk_coord,
      ResultTuple const& result, TiledMma const& tiled_mma,
      ProblemShape const& problem_size, Params const& params,
      LoadPipeline epi_load_pipeline, TensorStorage& epi_tensor_storage)
  {
    auto acc_k = get<0>(result);
    auto acc_v = get<1>(result);
  
    auto problem_size_kv = make_shape(get<3>(problem_size), get<4>(problem_size), _,
              make_shape(get<0>(problem_size), get<1>(problem_size)));
  
    using EpiStorePipeline = typename CollectiveEpilogueTMA::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename CollectiveEpilogueTMA::LoadPipelineState epi_load_pipe_consumer_state;
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    CollectiveEpilogueTMA epilogue_k{params.epilogue_K, epi_tensor_storage[0]};
    CollectiveEpilogueTMA epilogue_v{params.epilogue_V, epi_tensor_storage[1]};

    {
      auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
      epilogue_k.store(
        epi_load_pipeline, epi_load_pipe_consumer_state,
        epi_store_pipeline, epi_store_pipe_producer_state,
        problem_size_kv, tile_shape, make_coord(get<1>(blk_coord), _0{}, _, get<2>(blk_coord)),
        acc_k, tiled_mma, threadIdx.x % cutlass::NumThreadsPerWarpGroup,
        epi_tensor_storage[0]
      );

    }

    {
      auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
      epilogue_v.store(
        epi_load_pipeline, epi_load_pipe_consumer_state,
        epi_store_pipeline, epi_store_pipe_producer_state,
        problem_size_kv, tile_shape, make_coord(get<1>(blk_coord), _0{}, _, get<2>(blk_coord)),
        acc_v, tiled_mma, threadIdx.x % cutlass::NumThreadsPerWarpGroup,
        epi_tensor_storage[1]
      );

      epilogue_k.store_tail(
        epi_load_pipeline, epi_load_pipe_consumer_state_next,
        epi_store_pipeline, epi_store_pipe_producer_state_next
      );

      epilogue_v.store_tail(
        epi_load_pipeline, epi_load_pipe_consumer_state_next,
        epi_store_pipeline, epi_store_pipe_producer_state_next
      );
    }
  }
};

}  // namespace cutlass::fmha::collective
