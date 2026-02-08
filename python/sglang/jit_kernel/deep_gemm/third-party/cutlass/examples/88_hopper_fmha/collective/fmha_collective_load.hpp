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
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {

enum class LoadKind {
  kQ, kK, kV,
  kBwdN, kBwdM, kBwdScalar
};

template<
  LoadKind kKind,
  class Pipeline,
  class Element,
  class SmemLayout,
  class TMA
>
struct CollectiveLoadTma {

  using Params = TMA;
  using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayout>>;
  using PipelineState  = typename cutlass::PipelineState<Pipeline::Stages>;

  Params const& params;
  Pipeline& pipeline;
  SharedStorage& storage;

  CUTLASS_DEVICE
  CollectiveLoadTma(Params const& params, Pipeline& pipeline, SharedStorage& storage)
    : params(params), pipeline(pipeline), storage(storage) {}

  template<class ProblemSize, class TileShape, class BlockCoord>
  CUTLASS_DEVICE auto init_g(ProblemSize const& problem_size, TileShape const& tile_shape,
      BlockCoord const& blk_coord, int loop_count
  ) {
    using X = Underscore;
    if constexpr (kKind == LoadKind::kK) {
      Tensor mK_full = params.get_tma_tensor(make_shape(get<3>(problem_size), get<4>(problem_size), select<0,1>(problem_size)));
      Tensor gK_full = local_tile(mK_full, tile_shape, make_coord(_, _, _), Step<X, _1, _1>{});
      Tensor gK = gK_full(_, _, _, _0{}, get<2>(blk_coord));
      return gK;
    } else if constexpr (kKind == LoadKind::kQ) {
      Tensor mQ_full = params.get_tma_tensor(make_shape(get<2>(problem_size), get<4>(problem_size), select<0,1>(problem_size)));
      Tensor gQ_full = local_tile(mQ_full, tile_shape, make_coord(_, _, _), Step<_1, X, _1>{});
      Tensor gQ = gQ_full(_, _, _, _0{}, get<2>(blk_coord));
      return make_tensor(gQ.data() + loop_count * get<0>(blk_coord) * stride<2>(gQ), gQ.layout());
    } else if constexpr (kKind == LoadKind::kV) {
      Tensor mV_full = params.get_tma_tensor(make_shape(get<4>(problem_size), get<3>(problem_size), select<0,1>(problem_size)));
      Tensor gV_full = local_tile(mV_full, tile_shape, make_coord(_, _, _), Step<X, _1, _1>{});
      Tensor gV = gV_full(_, _, _0{}, _, get<2>(blk_coord));
      return gV;
    } else if constexpr (kKind == LoadKind::kBwdN) {
      Tensor m_full = params.get_tma_tensor(make_shape(get<3>(problem_size), get<4>(problem_size), select<0,1>(problem_size)));
      Tensor g_full = local_tile(m_full, tile_shape, make_coord(_, _, _), Step<_1, X, _1>{});
      Tensor g = g_full(_, _, _, _0{}, get<2>(blk_coord));
      return make_tensor(g.data() + loop_count * get<1>(blk_coord) * stride<2>(g), g.layout());
    } else if constexpr (kKind == LoadKind::kBwdM) {
      Tensor m_full = params.get_tma_tensor(make_shape(get<2>(problem_size), get<4>(problem_size), select<0,1>(problem_size)));
      Tensor g_full = local_tile(m_full, tile_shape, make_coord(_, _, _), Step<X, _1, _1>{});
      Tensor g = g_full(_, _, _, _0{}, get<2>(blk_coord));
      return g;
    } else if constexpr (kKind == LoadKind::kBwdScalar) {
      Tensor m_full = params.get_tma_tensor(select<2,0,1>(problem_size));
      Tensor g_full = local_tile(m_full, tile_shape, make_coord(_, _, _), Step<X, _1, X>{});
      Tensor g = g_full(_, _, get<2,0>(blk_coord), get<2,1>(blk_coord));
      return g;
    }
  }

  template<class ClusterRank, class ProblemSize, class TileShape, class BlockCoord>
  CUTLASS_DEVICE auto init_state(ClusterRank const& block_rank_in_cluster,
      ProblemSize const& problem_size, TileShape const& tile_shape,
      BlockCoord const& block_coord, int loop_count
  ) {
    Tensor g = init_g(problem_size, tile_shape, block_coord, loop_count);
    Tensor s = make_tensor(make_smem_ptr(storage.data()), SmemLayout{});
  
    auto block_tma = params.get_slice(block_rank_in_cluster);
    Tensor ts = block_tma.partition_D(s);
    Tensor tg = block_tma.partition_S(g);

    return make_tuple(tg, ts);
  }

  template<bool kAdvanceIterator=true, bool kAdvancePipe=true, bool kAcquireBarrier=true, class TileIterator, class State>
  CUTLASS_DEVICE void step(TileIterator& tile_iter, State const& state,
      PipelineState& smem_pipe_write,
      int lane_predicate, int& tile_count, uint16_t mcast_mask = 0
  ) {
    if ((lane_predicate == 1) && (tile_count > 0)) {
      if constexpr (kAcquireBarrier) pipeline.producer_acquire(smem_pipe_write);
      using BarrierType = typename Pipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      if constexpr (kKind == LoadKind::kBwdScalar) {
        copy(params.with(*tma_barrier, mcast_mask), get<0>(state)(_,_,*tile_iter), get<1>(state)(_,_,smem_pipe_write.index()));
      } else {
        copy(params.with(*tma_barrier, mcast_mask), get<0>(state)(_,_,_,*tile_iter), get<1>(state)(_,_,_,smem_pipe_write.index()));
      }
      if constexpr (kAdvancePipe) ++smem_pipe_write;
      if constexpr (kAdvanceIterator) ++tile_iter;
    }
    --tile_count;
  }
};

}  // namespace cutlass::fmha::collective
