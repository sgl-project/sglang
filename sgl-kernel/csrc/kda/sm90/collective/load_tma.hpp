// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cutlass/cutlass.h>

#include <cute/tensor.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include "kda/sm90/utils/debug.hpp"

namespace kda::sm90::collective {

using namespace cute;

enum class LoadKind {
  kQ,
  kK,
  kV,
  kAlpha,
};

CUTE_HOST_DEVICE constexpr char const* to_string(LoadKind kind) {
  if (kind == LoadKind::kQ) {
    return "Q";
  } else if (kind == LoadKind::kK) {
    return "K";
  } else if (kind == LoadKind::kV) {
    return "V";
  } else if (kind == LoadKind::kAlpha) {
    return "Alpha";
  } else {
    return "unknown loadkind";
  }
}

template <LoadKind kKind, class Pipeline, class Element, class SmemLayout, class TMA>
struct CollectiveLoadTma {
  using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayout>>;
  using PipelineState = typename cutlass::PipelineState<Pipeline::Stages>;

  static constexpr LoadKind kind = kKind;
  TMA const& tma_load;
  Pipeline& pipeline;
  SharedStorage& storage;

  CUTE_DEVICE
  CollectiveLoadTma(TMA const& tma_load, Pipeline& pipeline, SharedStorage& storage)
      : tma_load(tma_load), pipeline(pipeline), storage(storage) {}

  template <class ProblemSize, class TileShape, class WorkDesc>
  CUTE_DEVICE auto
  partition_SD(ProblemSize const& problem_size, TileShape const& tile_shape, WorkDesc const& work_desc) {
    constexpr auto BlkSeqQ = decltype(get<0>(tile_shape))::value;
    constexpr auto BlkSeqKV = decltype(get<1>(tile_shape))::value;
    constexpr auto HeadSize = decltype(get<2>(tile_shape))::value;

    Tensor g = [&] {
      if constexpr (kind == LoadKind::kQ) {
        DPRINTF0_W(
            "slice view GMEM %s: seq_idx:%d head_idx:%d tok_offset:%lld\n",
            to_string(kind),
            work_desc.seq_idx,
            work_desc.q_head_idx(),
            work_desc.tok_offset);
        Tensor m_varlen_head = tma_load.get_tma_tensor(make_shape(
            problem_size.total_seqlen,
            problem_size.head_size,
            problem_size.num_heads));                                   // global view to the packed varlen sequence
        Tensor m_varlen = m_varlen_head(_, _, work_desc.q_head_idx());  // slice into current head_idx
        Tensor m_offset = domain_offset(
            make_coord(work_desc.tok_offset, _0{}),
            m_varlen);  // offset to start of the current sequence
        Tensor g_full = local_tile(m_offset, make_tile(BlkSeqQ, HeadSize), make_coord(_, _0{}));  // (blk, d, iter_blk)
        return g_full;
      } else if constexpr (kind == LoadKind::kAlpha) {  // same as Q currently
        DPRINTF0_W(
            "slice view GMEM %s: seq_idx:%d head_idx:%d tok_offset:%lld\n",
            to_string(kind),
            work_desc.seq_idx,
            work_desc.q_head_idx(),
            work_desc.tok_offset);
        Tensor m_varlen_head = tma_load.get_tma_tensor(make_shape(
            problem_size.total_seqlen,
            problem_size.head_size,
            problem_size.num_heads));                                   // global view to the packed varlen sequence
        Tensor m_varlen = m_varlen_head(_, _, work_desc.q_head_idx());  // slice into current head_idx
        Tensor m_offset = domain_offset(
            make_coord(work_desc.tok_offset, _0{}),
            m_varlen);  // offset to start of the current sequence
        Tensor g_full = local_tile(m_offset, make_tile(BlkSeqQ, HeadSize), make_coord(_, _0{}));  // (blk, d, iter_blk)
        return g_full;
      } else {
        auto head_idx = (kind == LoadKind::kK ? work_desc.k_head_idx() : work_desc.v_head_idx());
        DPRINTF0_W(
            "slice view GMEM %s: seq_idx:%d head_idx:%d tok_offset:%lld\n",
            to_string(kind),
            work_desc.seq_idx,
            head_idx,
            work_desc.tok_offset);
        Tensor m_varlen_head = tma_load.get_tma_tensor(make_shape(
            problem_size.head_size,
            problem_size.total_seqlen,
            problem_size.num_heads));                     // global view to the packed varlen sequence
        Tensor m_varlen = m_varlen_head(_, _, head_idx);  // slice into current head_idx
        Tensor m_offset = domain_offset(
            make_coord(_0{}, work_desc.tok_offset),
            m_varlen);  // offset to start of the current sequence
        Tensor g_full = local_tile(m_offset, make_tile(HeadSize, BlkSeqKV), make_coord(_0{}, _));  // (d, blk, iter_blk)
        return g_full;
      }
    }();
    Tensor s = make_tensor(make_smem_ptr(storage.data()), SmemLayout{});

    auto block_tma = tma_load.get_slice(_0{});  // do not support cluster
    return make_tuple(block_tma.partition_S(g), block_tma.partition_D(s));
  }

  template <bool kAcquireBarrier = true, class SrcDst>
  CUTE_DEVICE void step(SrcDst const& src_dst, int src_iter, PipelineState& dst_pipe, uint32_t lane_predicate) {
    if (lane_predicate == 1) {
      DPRINTF_WG("%s pipeline.producer_acquire smem_pipe_write:%d\n", to_string(kind), dst_pipe.index());
      if constexpr (kAcquireBarrier) {
        pipeline.producer_acquire(dst_pipe);
      }
      using BarrierType = typename Pipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(dst_pipe);

      auto src = get<0>(src_dst);
      auto dst = get<1>(src_dst);

      copy(tma_load.with(*tma_barrier), src(_, _, _, src_iter), dst(_, _, _, dst_pipe.index()));
      ++dst_pipe;
    }
  }
};

}  // namespace kda::sm90::collective
