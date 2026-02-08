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
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template<
  class Element,
  class StrideQ,
  class StrideNewK,
  class StrideNewV,
  class StrideCacheK,
  class StrideCacheV,
  class TensorStorage,
  class CollectiveMmaQK,
  class CollectiveMmaPV,
  class SmemLayoutQ,
  class SmemLayoutK,
  class SmemLayoutV,
  class PipelineQ,
  class PipelineKV,
  class TileShape,
  class Mask
>
struct Sm100FmhaLoadCpAsyncWarpspecialized {

  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;

  struct Arguments {
  
    const int* cache_batch_idx;

    const Element* ptr_q;
    StrideQ dQ;

    const Element* ptr_new_k;
    StrideNewK dNewK;
    const Element* ptr_new_v;
    StrideNewV dNewV;

    Element* ptr_cache_k;
    StrideCacheK dCacheK;
    Element* ptr_cache_v;
    StrideCacheV dCacheV;
  };

  using Params = Arguments;

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    return args;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
  }

  template<class TEngine, class TLayout>
  CUTLASS_DEVICE auto constexpr transpose(Tensor<TEngine, TLayout> const& t) {
    CUTE_STATIC_ASSERT_V(rank(t) == _2{});
    return t.compose(make_layout(make_shape(size<1>(t), size<0>(t)), make_stride(size<0>(t), _1{})));
  }

  template<
    class CAtom, class TA, class TB,
    class CountTensor, class CountLimit,
    class SrcTensor, class DstTensor
  >
  CUTLASS_DEVICE void copy_with_limit(
      TiledCopy<CAtom, TA, TB> const& tiled_copy,
      CountTensor const& c, CountLimit const& l,
      SrcTensor const& src, DstTensor&& dst) {

    //copy(tiled_copy, src, dst);
#if 1
    auto c_f = make_tensor(c.data(), flatten(c.layout()));
    auto src_f = make_tensor(src.data(), flatten(src.layout()));
    auto dst_f = make_tensor(dst.data(), flatten(dst.layout()));
    auto c_v = group_modes<1,rank_v<decltype(c_f)>>(c_f);
    auto src_v = group_modes<1,rank_v<decltype(src_f)>>(src_f);
    auto dst_v = group_modes<1,rank_v<decltype(dst_f)>>(dst_f);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(src_v); i++) {
      if (elem_less(c_v(_0{}, i), l)) {
        copy(CAtom{}, src_v(_, i), dst_v(_, i));
      }
      else {
        clear(dst_v(_, i));
      }
    }
#endif
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state) {

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    mask_tile_count *= 2;

    int warp_idx = (threadIdx.x / 32) % 2;
    int thread_idx = warp_idx * 32 + (threadIdx.x % 32);

    using X = Underscore;

    // this one is only executed by one thread, no need to elect_one
    auto blk_coord_cache = blk_coord;
    if (params.cache_batch_idx != nullptr) {
      get<2,1>(blk_coord_cache) = params.cache_batch_idx[get<2,1>(blk_coord_cache)];
    }

    // Q1, K1, K2, V1, K3, V2, ... Kn, Vn-1, Vn
    // two pipes: Q and KV
    auto cQ = make_identity_tensor(select<0,2>(TileShape{}));
    auto mQ = make_tensor(make_gmem_ptr(params.ptr_q), append<3>(select<0,2>(TileShapeQK{}), get<3>(problem_shape)), params.dQ);
    auto gQ = mQ(_, _, get<2>(blk_coord));
    auto sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(0);
    auto tSgQ = thr_mma_qk.partition_A(gQ);
    auto tScQ = thr_mma_qk.partition_A(cQ);

    auto atom_q_tv = Layout<Shape<Shape<_2, _32>, _16>, Stride<Stride<_16, _32>, _1>>{};
    auto atom_kv_tv = Layout<Shape<Shape<_2, _32>, _16>, Stride<Stride<_16, _32>, _1>>{};

    auto tiled_copy_q = make_cotiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
        atom_q_tv,
        make_layout(shape(tSgQ), replace<0>(stride(tSgQ), replace<0>(stride<0>(tSgQ), get<2>(TileShape{})))));

    auto thr_copy_q = tiled_copy_q.get_slice(thread_idx);

    auto tQsQ = thr_copy_q.partition_D(sQ);
    auto tQgQ = thr_copy_q.partition_S(tSgQ);
    auto tQcQ = thr_copy_q.partition_S(tScQ);

    auto limitQ = append<2>(get<0>(problem_shape), _128{});

    // Q1
    int q0_index = get<0>(blk_coord);

    auto load_q = [&](int q_index, auto& state) {
      pipeline_q.producer_acquire(state);

      // q is always loaded masked
      using Vec = uint128_t;
      Vec vzero = uint128_t(0, 0);
      auto src = recast<Vec>(tQgQ(_, _, _, _));
      auto dst = recast<Vec>(tQsQ(_, _, _, _, state.index()));
      auto c = tQcQ(_, _, _, _);
      int vlen = sizeof(Vec) / sizeof(Element);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src); i++) {
        auto cc = c(vlen*i);
        Vec* dst_ptr = &dst(i);
        const Vec* src_ptr = &src(i);
        bool guard = elem_less(cc, limitQ);
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
          dst_ptr, src_ptr, guard
        );
      }
    
      pipeline_q.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
    };

    load_q(q0_index, pipeline_q_producer_state);
    ++pipeline_q_producer_state;

    auto cK_t = make_identity_tensor(select<1,2>(TileShapeQK{}));
    auto cK = make_tensor(cK_t.data(), make_layout(get<0>(cK_t.layout()), get<1>(cK_t.layout()), make_layout(_2{}, get<1>(TileShapeQK{}) * stride<0>(cK_t))));
    auto mK = make_tensor(make_gmem_ptr(params.ptr_cache_k), select<1,2,3>(problem_shape), params.dCacheK);
    auto gK = local_tile(mK(_, _, get<2>(blk_coord_cache)), TileShapeQK{}, make_coord(_, _, _0{}), Step<X, _1, _1>{});
    auto sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});

    auto tSgK = thr_mma_qk.partition_B(gK);
    auto tScK = thr_mma_qk.partition_B(cK);

    auto tSlK = thr_mma_qk.partition_B(make_tensor((Element*) nullptr, make_ordered_layout(select<1,2>(TileShapeQK{}), Step<_1, _0>{})));
    auto tiled_copy_k = make_cotiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
        atom_kv_tv,
        tSlK.layout());

    auto thr_copy_k = tiled_copy_k.get_slice(thread_idx);

    auto tKsK = thr_copy_k.partition_D(sK);
    auto tKgK = thr_copy_k.partition_S(tSgK);
    auto tKcK = thr_copy_k.partition_S(tScK);

    int seqlen_cache_kv = get<1>(problem_shape) - ((params.ptr_new_k != nullptr) ? 1 : 0);
    auto limitK = append<2>(seqlen_cache_kv, _128{});

    auto cV_t = make_identity_tensor(select<1,2>(TileShapePV{}));
    auto cV = make_tensor(cV_t.data(), make_layout(get<0>(cV_t.layout()), get<1>(cV_t.layout()), make_layout(_2{}, get<2>(TileShapePV{}) * stride<1>(cV_t))));
    auto mV = make_tensor(make_gmem_ptr(params.ptr_cache_v), select<2,1,3>(problem_shape), select<1,0,2>(params.dCacheV));
    auto gV = local_tile(mV(_, _, get<2>(blk_coord_cache)), TileShapePV{}, make_coord(_, _0{}, _), Step<X, _1, _1>{});
    auto sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    typename CollectiveMmaPV::TiledMma mma_pv;
    ThrMMA thr_mma_pv = mma_pv.get_slice(0);
    auto tOgV = thr_mma_pv.partition_B(gV);
    auto tOcV = thr_mma_pv.partition_B(cV);
    auto tOlV = thr_mma_pv.partition_B(make_tensor((Element*) nullptr, make_layout(select<1,2>(TileShapePV{}))));

    auto tiled_copy_v = make_cotiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
        atom_kv_tv,
        tOlV.layout());

    auto thr_copy_v = tiled_copy_v.get_slice(thread_idx);

    auto tVsV = thr_copy_v.partition_D(sV);
    auto tVgV = thr_copy_v.partition_S(tOgV);
    auto tVcV = thr_copy_v.partition_S(tOcV);

    auto limitV = select<1,0>(limitK);

    int full_tiles_cache = seqlen_cache_kv / get<1>(TileShapeQK{});

    bool has_new = params.ptr_new_k != nullptr;
    Tensor mNewK = make_tensor(make_gmem_ptr(params.ptr_new_k), select<1,2,3>(problem_shape), params.dNewK);
    Tensor mNewV = make_tensor(make_gmem_ptr(params.ptr_new_v), select<1,2,3>(problem_shape), params.dNewV);
    Tensor gNewK = mNewK(_, _, get<2>(blk_coord));
    Tensor gNewV = mNewV(_, _, get<2>(blk_coord));

    auto load_k = [&](int k_index, auto& state) {
      pipeline_kv.producer_acquire(state);

      if (k_index < full_tiles_cache) {
        copy(tiled_copy_k, tKgK(_, _, _, _, k_index), tKsK(_, _, _, _, state.index()));
        pipeline_kv.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
      } else {
        using Vec = uint128_t;
        Vec vzero = uint128_t(0, 0);
        auto src = recast<Vec>(tKgK(_, _, _, _, k_index));
        auto dst = recast<Vec>(tKsK(_, _, _, _, state.index()));
        auto src2 = recast<Vec>(gNewK);
        auto c = tKcK(_, _, _, _, k_index);
        int vlen = sizeof(Vec) / sizeof(Element);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src); i++) {
          auto cc = c(vlen*i);
          Vec* dst_ptr = &dst(i);
          const Vec* src_ptr = &src(i);
          bool guard = elem_less(cc, limitK);
          if (get<0>(cc) == seqlen_cache_kv && has_new) {
            src_ptr = &src2(_0{}, get<1>(cc) / vlen);
            guard = true;
          }
          cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
            dst_ptr, src_ptr, guard
          );
        }
      
        pipeline_kv.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
      }
    };

    auto load_v = [&](int v_index, auto& state) {
      pipeline_kv.producer_acquire(state);

      if (v_index < full_tiles_cache) {
        copy(tiled_copy_v, tVgV(_, _, _, _, v_index), tVsV(_, _, _, _, state.index()));
        pipeline_kv.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
      } else {
        using Vec = uint128_t;
        Vec vzero = uint128_t(0, 0);
        auto src = recast<Vec>(tVgV(_, _, _, _, v_index));
        auto dst = recast<Vec>(tVsV(_, _, _, _, state.index()));
        auto src2 = recast<Vec>(gNewV);
        int vlen = sizeof(Vec) / sizeof(Element);
        auto c = tVcV(_, _, _, _, v_index);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src); i++) {
          auto cc = c(vlen*i);
          Vec* dst_ptr = &dst(i);
          const Vec* src_ptr = &src(i);
          bool guard = elem_less(cc, limitV);
          if (get<1>(cc) == seqlen_cache_kv && has_new) {
            src_ptr = &src2(_0{}, get<0>(cc) / vlen);
            guard = true;
          }
          cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
            dst_ptr, src_ptr, guard
          );
        }

        pipeline_kv.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
      }
    };

    // K1
    int k_index = 0;
    int v_index = 0;

    load_k(k_index, pipeline_kv_producer_state);

    ++pipeline_kv_producer_state;
    k_index += 1;

    mask_tile_count -= 1;

    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      load_k(k_index, pipeline_kv_producer_state);

      ++pipeline_kv_producer_state;
      k_index += 1;
  
      load_v(v_index, pipeline_kv_producer_state);

      ++pipeline_kv_producer_state;
      v_index += 1;
    }

    // V1

    load_v(v_index, pipeline_kv_producer_state);

    ++pipeline_kv_producer_state;
    v_index += 1;
  
    if (has_new) {
      for (int i = thread_idx; i < get<2>(TileShape{}); i += 64) {
        gK(seqlen_cache_kv, i, 0) = gNewK(0, i);
        gV(i, seqlen_cache_kv, 0) = gNewV(0, i);
      }
    }
  }

};

}  // namespace cutlass::fmha::collective
