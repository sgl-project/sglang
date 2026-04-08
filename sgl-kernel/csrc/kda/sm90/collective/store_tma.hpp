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

#include <cuda/ptx>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>

#include "kda/sm90/utils/debug.hpp"
#include "kerutils/kerutils.cuh"

namespace kda::sm90::collective {

using ku::alignment_for_swizzle;
using namespace cute;

/*
NOTE: what we need is as follows

  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementO, ElementAccumulatorO, void>;
  using CollectiveStoreO = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeO1, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulatorO, ElementAccumulatorO,
      void, LayoutO, Alignment,                                 // C, not exists
      ElementO, decltype(select<1,0,2>(LayoutO{})), Alignment,  // D
      cutlass::epilogue::TmaWarpSpecializedCooperative, DefaultOperation>::CollectiveOp;

but unfortunately the required type alias is only useful for our purpose is private so we roll out our own.
*/

CUTE_DEVICE uint32_t smid() {
#ifdef __CUDA_ARCH__
  uint32_t virtual_smid;
  asm("mov.u32 %0, %%smid;" : "=r"(virtual_smid));
  return virtual_smid;
#else
  return 0;
#endif
}

template <
    typename TileShape_MNK_,
    typename ClusterShape,
    typename ElementO,
    typename ElementAccumulator,
    typename SmemElementO,
    typename StrideO,
    int Stages>
struct CollectiveStoreTma {
  static_assert(size_v<ClusterShape> == 1);
  using TileShape_MNK = TileShape_MNK_;
  using TileShape_MN =
      decltype(select<0, 1>(TileShape_MNK{}));      // Collective work on TileShape_MN, it is also the OutputTile
  using SizeM = decltype(get<0>(TileShape_MNK{}));  // head_size
  using SizeN = decltype(get<1>(TileShape_MNK{}));  // seqlen

  constexpr static bool is_m_major_O = cutlass::epilogue::collective::detail::is_m_major<StrideO>();

#if 0
  // NOTE: the following derived layout is a bit slower than the manual one, will evaluate it later
  using SmemLayoutAtom = decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
                                  StrideO, ElementO, TileShape_MN>());
#else
  static_assert(sizeof(SmemElementO) == 2);
  using SmemLayoutAtom = GMMA::Layout_MN_SW32_Atom<SmemElementO>;
#endif

  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(SizeM{}, SizeN{}, Int<Stages>{}),
      cute::conditional_t<is_m_major_O, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

  constexpr static uint32_t TmaTransactionBytes =
      (size(take<0, 2>(SmemLayoutO{})) * static_cast<uint32_t>(sizeof_bits<SmemElementO>::value)) / 8;

  using CopyOpR2S = decltype(cutlass::epilogue::collective::detail::
                                 sm90_get_smem_store_op_for_accumulator<StrideO, ElementO, TileShape_MN>());
  using CopyAtomR2S = Copy_Atom<CopyOpR2S, SmemElementO>;

  using CopyOpS2G = SM90_TMA_STORE;

  using SharedStorage =
      cute::array_aligned<SmemElementO, cute::cosize_v<SmemLayoutO>, alignment_for_swizzle(SmemLayoutO{})>;
  using Pipeline = cutlass::PipelineAsync<Stages>;  // NOT PipelineTmaStore!
  using PipelineState = cutlass::PipelineState<Stages>;

  struct Arguments {
    ElementO* ptr_O;
    StrideO dO;
    void* workspace;
  };

  struct Params {
    using TMA_O = decltype(make_tma_copy(
        CopyOpS2G{},
        make_tensor(make_gmem_ptr<ElementO>(nullptr), repeat_like(StrideO{}, int32_t(0)), StrideO{}),
        take<0, 2>(SmemLayoutO{}),
        TileShape_MN{},
        _1{}));

    TMA_O tma_store_o;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* tensormaps;
  };
  using TMA = typename Params::TMA_O;

  CUTE_DEVICE
  CollectiveStoreTma(TMA const& tma_store, Pipeline& pipeline, SharedStorage& storage, void* tensormaps)
      : tma_store_(tma_store), pipeline_(pipeline), storage_(storage), tensormaps_(tensormaps) {}

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args, void* workspace) {
    auto problem_size_MNKL = append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_size_MNKL;

    Tensor tensor_o = make_tensor(make_gmem_ptr<ElementO>(args.ptr_O), make_layout(make_shape(M, N, L), args.dO));
    TMA tma_store_o = make_tma_copy_C_sm90(CopyOpS2G{}, tensor_o, take<0, 2>(SmemLayoutO{}), TileShape_MN{});

    return {
        .tma_store_o = tma_store_o,
        .tma_transaction_bytes = TmaTransactionBytes,
        .tensormaps = workspace,
    };
  }

  static size_t get_workspace_size(/*Arguments const& args,*/ int sm_count) {
    // only use additional TMA desc for output tail tiles
    size_t num_bytes = sizeof(cute::TmaDescriptor) * sm_count;
    DPRINTF("workspace num_bytes:%zu\n", num_bytes);
    return num_bytes;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape,
      /*Arguments const& args,*/ void* workspace,
      cudaStream_t stream) {
    return cutlass::Status::kSuccess;
  }

  CUTE_DEVICE static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
  }

  template <class ProblemSize, class TileShape, class WorkDesc>
  CUTE_DEVICE auto
  partition_SD(ProblemSize const& problem_size, TileShape const& tile_shape, WorkDesc const& work_desc) {
    constexpr auto BlkSeqQ = decltype(get<0>(tile_shape))::value;
    constexpr auto HeadSize = decltype(get<2>(tile_shape))::value;

    Tensor g = [&] {
      DPRINTF0_W(
          "slice view GMEM O: seq_idx:%d head_idx:%d tok_offset:%lld\n",
          work_desc.seq_idx,
          work_desc.o_head_idx(),
          work_desc.tok_offset);
      Tensor m_varlen_head = tma_store_.get_tma_tensor(make_shape(
          problem_size.head_size,
          problem_size.total_seqlen,
          problem_size.num_heads));                                   // global view to the packed varlen sequence
      Tensor m_varlen = m_varlen_head(_, _, work_desc.o_head_idx());  // slice into current head_idx
      Tensor m_offset = domain_offset(
          make_coord(_0{}, work_desc.tok_offset),
          m_varlen);  // offset to start of the current sequence
      Tensor g_full = local_tile(m_offset, make_tile(HeadSize, BlkSeqQ), make_coord(_0{}, _));  // (d, blk, iter_blk)
      return g_full;
    }();
    Tensor s = make_tensor(make_smem_ptr(storage_.data()), SmemLayoutO{});

    auto block_tma = tma_store_.get_slice(_0{});  // do not support cluster
    return make_tuple(block_tma.partition_S(s), block_tma.partition_D(g));
  }

  template <typename ProblemSize, typename WorkDesc>
  CUTE_DEVICE static bool
  can_process(ProblemSize const& problem_size, WorkDesc const& work_desc, int blk, int num_blocks) {
    if (blk < num_blocks - 1) {
      // intermediate full tiles, always use TMA
      return true;
    } else if (work_desc.seq_len % SizeN{} == 0 || work_desc.seq_idx == problem_size.num_seqs - 1) {
      // 1. last tile but full, also use TMA
      // 2. last tile but last seq, oob can be handled by TMA
      return true;
    } else {
      return false;
    }
  }

  template <bool kAcquireBarrier = true, typename ProblemSize, typename WorkDesc, typename SrcDst>
  CUTE_DEVICE void step(
      ProblemSize const& problem_size,
      WorkDesc const& work_desc,
      SrcDst const& src_dst,
      PipelineState& src_pipe,
      int dst_iter,
      int num_iters,
      uint32_t lane_predicate) {
    auto src = get<0>(src_dst);
    auto dst = get<1>(src_dst);

    if (dst_iter == 0) {
      bool can_process_tail = can_process(problem_size, work_desc, num_iters - 1, num_iters);
      if (!can_process_tail) {
        create_tensormap_for_tail(work_desc, lane_predicate);
      }
    }

    DPRINTF0_WG("pipeline.producer_acquire smem_pipe_read:%d\n", src_pipe.index());
    if constexpr (kAcquireBarrier) {
      pipeline_.consumer_wait(src_pipe);
    }

    if (can_process(problem_size, work_desc, dst_iter, num_iters)) {
      DPRINTF0_W("store src_pipe:%d -> blk:%d\n", src_pipe.index(), dst_iter);
      if (lane_predicate == 1) {
        copy(tma_store_, src(_, _, _, src_pipe.index()), dst(_, _, _, dst_iter));
      }
    } else {
      cute::TmaDescriptor* tensormap = acquire_tensormap_for_tail();
      DPRINTF0_W("store tail with tensormap:%p src_pipe:%d -> blk:%d\n", tensormap, src_pipe.index(), dst_iter);
      if (lane_predicate == 1) {
        copy(tma_store_.with(tensormap), src(_, _, _, src_pipe.index()), dst(_, _, _, dst_iter));
      }
    }

    if constexpr (kAcquireBarrier) {
      pipeline_.consumer_release(src_pipe);
    }
    ++src_pipe;
  }

  template <typename WorkDesc>
  CUTE_DEVICE void create_tensormap_for_tail(WorkDesc const& work_desc, uint32_t lane_predicate) {
    namespace ptx = cuda::ptx;
    constexpr int num_of_16B = sizeof(cute::TmaDescriptor) / sizeof(uint128_t);

    cute::TmaDescriptor* tensormap = static_cast<cute::TmaDescriptor*>(tensormaps_) + smid();

    auto lane_idx = cutlass::canonical_lane_idx();
    if (lane_idx < num_of_16B) {
      auto src = reinterpret_cast<uint128_t const*>(tma_store_.get_tma_descriptor());
      auto dst = reinterpret_cast<uint128_t*>(tensormap);

      dst[lane_idx] = src[lane_idx];
    }
    __syncwarp();

    if (lane_predicate == 1) {
      uint32_t new_total_seqlen = work_desc.tok_offset + work_desc.seq_len;
      ptx::tensormap_replace_global_dim(ptx::space_global, tensormap, /*ord=*/ptx::n32_t<1>{}, new_total_seqlen);
    }
    __syncwarp();

    ptx::fence_proxy_tensormap_generic(ptx::sem_release, ptx::scope_cta);
  }

  CUTE_DEVICE cute::TmaDescriptor* acquire_tensormap_for_tail() {
    namespace ptx = cuda::ptx;
    cute::TmaDescriptor* tensormap = static_cast<cute::TmaDescriptor*>(tensormaps_) + smid();
    ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_cta, tensormap, /*size=*/ptx::n32_t<128>{});
    return tensormap;
  }

 private:
  TMA const& tma_store_;
  Pipeline& pipeline_;
  SharedStorage& storage_;
  void* tensormaps_;
};

}  // namespace kda::sm90::collective
