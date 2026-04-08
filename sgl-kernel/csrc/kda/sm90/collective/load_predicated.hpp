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
#include "kda/sm90/utils/unused.hpp"

namespace kda::sm90::collective {

using namespace cute;

// Wraps a callable into a predicate "tensor" usable by CuTe's copy_if.
// copy_if calls pred(i) with a linear index; the wrapped function maps that index to bool.
template <class Fn>
struct FunctionPredTensor {
  Fn fn_;
  CUTE_HOST_DEVICE
  FunctionPredTensor(Fn fn) : fn_(fn) {}
  template <class Idx>
  CUTE_HOST_DEVICE bool operator()(Idx const& i) const {
    return fn_(i);
  }
};

enum class LoadKindVector {
  kAlpha,
  kBeta,
};

CUTE_HOST_DEVICE constexpr char const* to_string(LoadKindVector kind) {
  if (kind == LoadKindVector::kAlpha) {
    return "alpha";
  } else if (kind == LoadKindVector::kBeta) {
    return "beta";
  } else {
    return "unknown loadkind";
  }
}

template <
    LoadKindVector kKind,
    class Pipeline,
    class ElementSrc,
    class GmemLayout,
    class ElementDst,
    class SmemLayout,
    class VectorProcessor_ = Unused>
struct CollectiveLoadVector {
  using SharedStorage = cute::array_aligned<ElementDst, cute::cosize_v<SmemLayout>>;
  using PipelineState = typename cutlass::PipelineState<Pipeline::Stages>;

  using VectorProcessor = VectorProcessor_;

  static_assert(rank_v<SmemLayout> == 2 || rank_v<SmemLayout> == 3);

  static constexpr LoadKindVector kind = kKind;
  static constexpr int VectorSize = size<0>(SmemLayout{});

  CUTE_DEVICE
  CollectiveLoadVector(
      ElementSrc const* src, GmemLayout layout, ElementSrc oob_value, Pipeline& pipeline, SharedStorage& storage)
      : src_(src), src_layout_(layout), src_oob_value_(oob_value), pipeline_(pipeline), storage_(storage) {}

  template <class ProblemSize, class TileShape, class WorkDesc>
  CUTE_DEVICE auto
  partition_SD(ProblemSize const& problem_size, TileShape const& tile_shape, WorkDesc const& work_desc) {
    constexpr auto BlkSeqQ = decltype(get<0>(tile_shape))::value;

    Tensor g = [&] {
      auto head_idx = work_desc.o_head_idx();
      DPRINTF0_W(
          "slice view GMEM %s: seq_idx:%d head_idx:%d tok_offset:%lld\n",
          to_string(kind),
          work_desc.seq_idx,
          head_idx,
          work_desc.tok_offset);
      Tensor m_varlen_head = make_tensor(make_gmem_ptr(src_), src_layout_);

      Tensor m_varlen = m_varlen_head(_, head_idx);  // slice into current head_idx
      Tensor m_offset =
          domain_offset(make_coord(work_desc.tok_offset), m_varlen);  // offset to start of the current sequence
      Tensor g_full = flat_divide(m_offset, BlkSeqQ);                 // (blk, iter_blk)
      return g_full;
    }();
    // (blk, pipe) or (blk, pipe, N), N for feature rich preprocess, data will be stored at 0
    Tensor s = make_tensor(make_smem_ptr(storage_.data()), SmemLayout{});

    auto thr_layout = Layout<_32>{};
    auto val_layout = Layout<_1>{};
    auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<ElementSrc>, ElementDst>{}, thr_layout, val_layout);
    auto thr_copy = tiled_copy.get_thread_slice(cutlass::canonical_lane_idx());

    auto coord = thr_copy.partition_S(make_identity_tensor(Shape<Int<BlkSeqQ>, _1>{}));
    int seq_len = work_desc.chunk_len();
    auto len_of_last_blk = seq_len - (ceil_div(seq_len, BlkSeqQ) - 1) * BlkSeqQ;

    auto mask = FunctionPredTensor([coord, len_of_last_blk](auto frag_coord) {
      auto coord_in_blk = get<0>(coord(frag_coord));
      return coord_in_blk < len_of_last_blk;
    });

    auto src = thr_copy.partition_S(g);  // (cpy, iter_cpy, iter_blk)
    auto dst = thr_copy.partition_D(s);  // (cpy, iter_cpy, pipe)

    return make_tuple(src, dst, mask);
  }

  template <bool IsTail, class SrcDst>
  CUTE_DEVICE void
  step(SrcDst const& src_dst, int src_iter, PipelineState& dst_pipe, int num_iters, VectorProcessor processor = {}) {
    auto src = get<0>(src_dst);
    auto dst = get<1>(src_dst);

    auto regs = make_fragment_like<ElementSrc>(take<0, 2>(shape(dst)));
    if constexpr (!IsTail) {
      copy(src(_, _, src_iter), regs);
    } else {
      auto mask = get<2>(src_dst);
      fill(regs, src_oob_value_);
      copy_if(mask, src(_, _, src_iter), regs);
    }

    int dst_pipe_idx = dst_pipe.index();

    DPRINTF0_WG("%s pipeline.producer_acquire smem_pipe_write:%d\n", to_string(kind), dst_pipe_idx);
    pipeline_.producer_acquire(dst_pipe);
    cutlass::arch::fence_view_async_shared();

    if constexpr (rank_v<SmemLayout> == 3) {
      copy(regs, dst(_, _, _0{}, dst_pipe_idx));
    } else {
      copy(regs, dst(_, _, dst_pipe_idx));
    }

    Tensor s = make_tensor(make_smem_ptr(storage_.data()), SmemLayout{});
    if constexpr (!std::is_same_v<VectorProcessor, Unused>) {
      if constexpr (rank_v<SmemLayout> == 3) {
        processor(s(_, _, dst_pipe_idx));
      } else {
        processor(s(_, dst_pipe_idx));
      }
    }

    cutlass::arch::fence_view_async_shared();
    pipeline_.producer_commit(dst_pipe);
    ++dst_pipe;
  }

 private:
  ElementSrc const* src_;
  GmemLayout src_layout_;  // in (packed_seq, H) coordinate
  ElementSrc src_oob_value_;
  Pipeline& pipeline_;
  SharedStorage& storage_;
};

}  // namespace kda::sm90::collective
