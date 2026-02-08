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
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "../collective/fmha_common.hpp"
#include "../collective/fmha_collective_load.hpp"
#include "../collective/fmha_collective_softmax.hpp"
#include "../kernel/fmha_options.hpp"

namespace cutlass::fmha::collective {

template<
  typename Element_,
  typename ElementAccumulator_,
  typename TileShape_, // BlockQO, BlockKV, BlockHead
  class Fusion,
  class... Options
>
struct FmhaBwdMainloopTmaWarpSpecialized {

  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using TileShape = TileShape_;

  static constexpr bool kIsPersistent = false;

  static const int NumLoadWarpGroups = 1;
  static constexpr int NumMmaWarpGroups = 2;
  static constexpr int StageCountQ = 2 /*K, V*/ * NumMmaWarpGroups;
  static constexpr int StageCount = 2 /*Q, dO*/ * 2 /* actual stages */;

  static const int kOuterLoads = 2;
  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using Stages = cutlass::gemm::collective::StageCount<StageCount>;
  using ClusterShape = Shape<_1, _1, _1>;
  static_assert(StagesQ::value >= 2);
  static_assert(Stages::value >= 2 * NumMmaWarpGroups);

  // 16B alignment lets us use TMA
  static constexpr int Alignment = 16 / sizeof(Element);

  using TileShapeNM = Shape<  // (N,M,D)
    decltype(tuple_element_t<1, TileShape>{} / Int<NumMmaWarpGroups>{}),
    tuple_element_t<0, TileShape>,
    tuple_element_t<2, TileShape>>;

  using TileShapeND = decltype(select<0,2,1>(TileShapeNM{}));  // (N,D,M)

  using TileShapeMD = decltype(select<2,1,0>(TileShapeND{}));  // (M,D,N)

  using CollectiveMmaNM = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment,
      Element, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment,
      ElementAccumulator,
      TileShapeNM, ClusterShape, Stages,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using CollectiveMmaND = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment, // from register, doesn't matter
      Element, cute::tuple<_1, int, cute::tuple<int, int>>, Alignment,
      ElementAccumulator,
      TileShapeND, ClusterShape, Stages,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using CollectiveMmaND_SS = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment, // from register, doesn't matter
      Element, cute::tuple<_1, int, cute::tuple<int, int>>, Alignment,
      ElementAccumulator,
      TileShapeND, ClusterShape, Stages,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;


  using CollectiveMmaMD = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, cute::tuple<_1, int, cute::tuple<int, int>>, Alignment, // from smem, might matter (?)
      Element, cute::tuple<_1, int, cute::tuple<int, int>>, Alignment,
      ElementAccumulator,
      TileShapeMD, ClusterShape, Stages,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using TiledMmaNM = typename CollectiveMmaNM::TiledMma;
  using TiledMmaND_SS = typename CollectiveMmaND_SS::TiledMma;
  using TiledMmaND_RS = decltype(convert_to_gmma_rs(typename CollectiveMmaND::TiledMma{}));
  using TiledMmaND = TiledMmaND_RS;
  using TiledMmaMD = typename CollectiveMmaMD::TiledMma;

  using SmemLayoutQ = typename CollectiveMmaNM::SmemLayoutB;
  using SmemLayoutK = typename CollectiveMmaNM::SmemLayoutA;
  using SmemLayoutV = typename CollectiveMmaNM::SmemLayoutA;
  using SmemLayoutDO = typename CollectiveMmaNM::SmemLayoutB;

  //using SmemLayoutDQ = Layout<
  //    Shape<
  //      tuple_element_t<0, TileShapeMD>,
  //      Shape<_2, _4, decltype(tuple_element_t<1, TileShapeMD>{} / _8{})>,
  //      _2
  //    >,
  //    Stride<
  //      _4,
  //      Stride<decltype(tuple_element_t<0, TileShapeMD>{} * _4{}), _1, decltype(tuple_element_t<0, TileShapeMD>{} * _8{})>,
  //      decltype(tuple_element_t<0, TileShapeMD>{} * tuple_element_t<1, TileShapeMD>{})
  //    >>;

  using SmemLayoutDQ_0 = Layout<
      Shape<
        tuple_element_t<0, TileShapeMD>,
        tuple_element_t<1, TileShapeMD>,
        _2
      >,
      Stride<
        tuple_element_t<1, TileShapeMD>,
        _1,
        decltype(tuple_element_t<0, TileShapeMD>{} * tuple_element_t<1, TileShapeMD>{})
      >>;

  using SmemAtomDQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<                 
                cute::GMMA::Major::K, ElementAccumulator, tuple_element_t<0, TileShapeMD>, tuple_element_t<1, TileShapeMD>>()); 
  using SmemLayoutDQ_1 = decltype(tile_to_shape(SmemAtomDQ{}, make_shape(get<0>(TileShapeMD{}), get<1>(TileShapeMD{}), _2{}), Step<_2, _1, _3>{}));
  using SmemLayoutDQ = SmemLayoutDQ_1;


  using PipelineDQ = cutlass::PipelineAsync<2>;

  
  using SmemLayoutDS_0 = decltype(unstageSmemLayout(typename CollectiveMmaMD::SmemLayoutA{}, Int<NumMmaWarpGroups>{}));

  using SmemLayoutDS = decltype(tile_to_shape(GMMA::Layout_MN_INTER_Atom<Element>{}, make_shape(size<0>(SmemLayoutDS_0{}), size<1>(SmemLayoutDS_0{}), size<2>(SmemLayoutDS_0{})), Step<_1, _2, _3>{}));
  using SmemLayoutKp = typename CollectiveMmaMD::SmemLayoutB;
  
  using SmemLayoutQp = typename CollectiveMmaND::SmemLayoutB;
  using SmemLayoutDOp = typename CollectiveMmaND::SmemLayoutB;

  using SmemLayoutLSE = Layout<Shape<tuple_element_t<1, TileShapeNM>, Int<StageCount>>>;

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages::value>;
  using MainloopPipelineQ = cutlass::PipelineTmaAsync<StagesQ::value>;

  using PipelineState  = typename cutlass::PipelineState<MainloopPipeline::Stages>;
  using PipelineStateQ  = typename cutlass::PipelineState<MainloopPipelineQ::Stages>;

  using TileShapePV = TileShapeND;  // To work with the kernel level
  using TiledMmaPV = TiledMmaND;

  static constexpr int kInnerLoadBytes = size(SmemLayoutQ{}(_,_,_0{})) * sizeof(Element) + size(SmemLayoutLSE{}(_,_0{})) * sizeof(ElementAccumulator);
  static constexpr int kOuterLoadBytes = size(SmemLayoutK{}(_,_,_0{})) * sizeof(Element);

  struct SharedStorage {
    // One for each consumer WG
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutKp>> smem_kp;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    };

    cute::array_aligned<Element, cute::cosize_v<SmemLayoutDS>> smem_ds;

    // Loaded by producer, consumed by both WGs
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutDO>> smem_do;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQp>> smem_qp;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutDOp>> smem_dop;
    };
    
    // Accumulated into by both consumers, potentially loaded, potentially written
    cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayoutDQ>> smem_dq;

    union {
      cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayoutLSE>> smem_lse;
      cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayoutLSE>> smem_sumOdO;
    };
  };

  struct Arguments {
    const Element* ptr_Q;
    cute::tuple<int, int, int, _1> dQ;
    const Element* ptr_K;
    cute::tuple<int, int, int, _1> dK;
    const Element* ptr_V;
    cute::tuple<int, int, int, _1> dV;

    const Element* ptr_dO;
    cute::tuple<int, int, int, _1> dDO;

    const ElementAccumulator* ptr_LSE;
    cute::tuple<int, int, _1> dLSE;
    const ElementAccumulator* ptr_sum_OdO;
    cute::tuple<int, int, _1> dSumOdO;

    ElementAccumulator* ptr_dQ;
    cute::tuple<int, int, int, _1> dDQ;
  };

  using TMA_Q = typename CollectiveMmaNM::Params::TMA_B;
  using TMA_K = typename CollectiveMmaNM::Params::TMA_A;
  using TMA_V = typename CollectiveMmaNM::Params::TMA_A;
  using TMA_DO = typename CollectiveMmaNM::Params::TMA_B;

  using TMA_LSE = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor((const ElementAccumulator*)nullptr, make_shape(1, 1, 1), make_stride(_1{}, 0, 0)), SmemLayoutLSE{}(_,_0{})));
  using TMA_ODO = TMA_LSE;

  using TMA_DQ = decltype(make_tma_copy(SM90_TMA_REDUCE_ADD{}, make_tensor((const ElementAccumulator*)nullptr, make_shape(1, 1, 1, 1), make_stride(0, _1{}, 0, 0)), SmemLayoutDQ{}(_,_,_0{})));

  using LoadQ = CollectiveLoadTma<
    LoadKind::kBwdM,
    MainloopPipeline,
    Element,
    SmemLayoutQ,
    TMA_Q
  >;

  using LoadK = CollectiveLoadTma<
    LoadKind::kBwdN,
    MainloopPipelineQ,
    Element,
    SmemLayoutK,
    TMA_K
  >;

  using LoadV = CollectiveLoadTma<
    LoadKind::kBwdN,
    MainloopPipelineQ,
    Element,
    SmemLayoutV,
    TMA_V
  >;

  using LoadDO = CollectiveLoadTma<
    LoadKind::kBwdM,
    MainloopPipeline,
    Element,
    SmemLayoutDO,
    TMA_DO
  >;

  using LoadLSE = CollectiveLoadTma<
    LoadKind::kBwdScalar,
    MainloopPipeline,
    ElementAccumulator,
    SmemLayoutLSE,
    TMA_LSE
  >;

  using LoadODO = CollectiveLoadTma<
    LoadKind::kBwdScalar,
    MainloopPipeline,
    ElementAccumulator,
    SmemLayoutLSE,
    TMA_ODO
  >;

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    TMA_DO tma_load_do;

    TMA_LSE tma_load_lse;
    TMA_ODO tma_load_odo;

    TMA_DQ tma_red_dq;

    float scale_softmax;
    float scale_softmax_log2;
  };

  static_assert(size(TiledMmaNM{}) == size(TiledMmaND{}));
  static_assert(size(TiledMmaNM{}) == size(TiledMmaMD{}));

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_size, Arguments const& args) {
    return true
      && (get<4>(problem_size) <= get<2>(TileShape{}))
      && ((get<4>(problem_size) % Alignment) == 0)
      && ((get<2>(problem_size) % Alignment) == 0)
    ;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size, Arguments const& args, void* workspace) {
    auto problem_shape_nm = make_shape(get<3>(problem_size), get<2>(problem_size), get<4>(problem_size), make_shape(get<0>(problem_size), get<1>(problem_size)));

    auto dK = make_stride(get<2>(args.dK), get<3>(args.dK), make_stride(get<0>(args.dK), get<1>(args.dK)));
    auto dQ = make_stride(get<2>(args.dQ), get<3>(args.dQ), make_stride(get<0>(args.dQ), get<1>(args.dQ)));
    auto params_nm_kq = CollectiveMmaNM::to_underlying_arguments(problem_shape_nm,
        typename CollectiveMmaNM::Arguments {
            args.ptr_K, dK,
            args.ptr_Q, dQ,
        }, /*workspace=*/ nullptr);

    auto dV = make_stride(get<2>(args.dV), get<3>(args.dV), make_stride(get<0>(args.dV), get<1>(args.dV)));
    auto dDO = make_stride(get<2>(args.dDO), get<3>(args.dDO), make_stride(get<0>(args.dDO), get<1>(args.dDO)));
    auto params_nm_vdo = CollectiveMmaNM::to_underlying_arguments(problem_shape_nm,
        typename CollectiveMmaNM::Arguments {
            args.ptr_V, dV,
            args.ptr_dO, dDO,
        }, /*workspace=*/ nullptr);


    TMA_LSE tma_load_lse = make_tma_copy(SM90_TMA_LOAD{}, make_tensor(args.ptr_LSE, select<2,0,1>(problem_size), select<2,0,1>(args.dLSE)), SmemLayoutLSE{}(_,_0{}));
    TMA_ODO tma_load_odo = make_tma_copy(SM90_TMA_LOAD{}, make_tensor(args.ptr_sum_OdO, select<2,0,1>(problem_size), select<2,0,1>(args.dSumOdO)), SmemLayoutLSE{}(_,_0{}));

    TMA_DQ tma_red_dq = make_tma_copy(SM90_TMA_REDUCE_ADD{}, make_tensor(args.ptr_dQ, select<2,4,0,1>(problem_size), select<2,3,0,1>(args.dDQ)), SmemLayoutDQ{}(_,_,_0{}));

    return Params{
        params_nm_kq.tma_load_b,
        params_nm_kq.tma_load_a,
        params_nm_vdo.tma_load_a,
        params_nm_vdo.tma_load_b,
        tma_load_lse, tma_load_odo,
        tma_red_dq,
        1.0f / (float) std::sqrt(get<4>(problem_size)),
        (float) (std::log2(std::exp(1.0)) / std::sqrt(get<4>(problem_size)))
    };
  }

  template<class BlkCoord, class ProblemSize>
  CUTLASS_DEVICE
  auto
  get_inner_tile_count(BlkCoord const& blk_coord, ProblemSize const& problem_size) {
    return Fusion{}.get_trip_count(blk_coord, TileShape{}, problem_size);
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_do.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_odo.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_lse.get_tma_descriptor());
  }

  template<bool kLoadOuter, class BlkCoord, class ProblemShape, class LoadWarpBarrier>
  CUTLASS_DEVICE void
  load_kv_maybe_q(
      int block_rank_in_cluster,
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_size,
      MainloopPipeline& pipeline_inner, PipelineState& smem_pipe_write_inner, 
      MainloopPipelineQ& pipeline_outer, PipelineStateQ& smem_pipe_write_outer, 
      SharedStorage& storage,
      LoadWarpBarrier& load_warp_barrier, bool do_barrier)
  {
    // Load pattern:
    // K0    V0 K1     V1
    //    Q0       DO0    Q1 DO1 Q2 DO2 ...
    // K0 Q0 V0 K1 DO0 V1 ...
    int lane_predicate = cute::elect_one_sync();

    int outer_tile_count = NumMmaWarpGroups;
    int inner_tile_count = get_inner_tile_count(blk_coord, problem_size);

    auto outer_tile_iter = cute::make_coord_iterator(outer_tile_count);
    auto inner_tile_iter = cute::make_coord_iterator(inner_tile_count);

    uint16_t mcast_mask_b = 0;
    
    LoadQ load_q{params.tma_load_q, pipeline_inner, storage.smem_q};
    auto load_state_q = load_q.init_state(block_rank_in_cluster, problem_size, TileShapeNM{}, blk_coord, inner_tile_count);

    LoadDO load_do{params.tma_load_do, pipeline_inner, storage.smem_do};
    auto load_state_do = load_do.init_state(block_rank_in_cluster, problem_size, TileShapeNM{}, blk_coord, inner_tile_count);

    LoadK load_k{params.tma_load_k, pipeline_outer, storage.smem_k};
    auto load_state_k = load_k.init_state(_0{}, problem_size, TileShapeNM{}, blk_coord, outer_tile_count);

    LoadV load_v{params.tma_load_v, pipeline_outer, storage.smem_v};
    auto load_state_v = load_v.init_state(_0{}, problem_size, TileShapeNM{}, blk_coord, outer_tile_count);

    LoadLSE load_lse{params.tma_load_lse, pipeline_inner, storage.smem_lse};
    auto load_state_lse = load_lse.init_state(_0{}, problem_size, TileShapeNM{}, blk_coord, outer_tile_count);

    LoadODO load_odo{params.tma_load_odo, pipeline_inner, storage.smem_sumOdO};
    auto load_state_odo = load_odo.init_state(_0{}, problem_size, TileShapeNM{}, blk_coord, outer_tile_count);

    outer_tile_count *= 2; // K & V
    inner_tile_count *= 4; // Q & dO & LSE & sumOdO

    while (inner_tile_count > 0) { 
      if (Fusion{}.is_contributing(make_coord(*inner_tile_iter, get<1>(blk_coord)), TileShape{}, problem_size)) {
        break;
      }
      inner_tile_count -= 4;
      ++inner_tile_iter;
    }

    if constexpr (kLoadOuter) {
      load_k.template step<false>(outer_tile_iter, load_state_k, smem_pipe_write_outer, lane_predicate, outer_tile_count);
    }

    load_q.template step<false,false,true>(inner_tile_iter, load_state_q, smem_pipe_write_inner, lane_predicate, inner_tile_count, mcast_mask_b);
    load_lse.template step<false,true,false>(inner_tile_iter, load_state_lse, smem_pipe_write_inner, lane_predicate, inner_tile_count, mcast_mask_b);

    if constexpr (! kLoadOuter) {
      if (do_barrier) {
        load_warp_barrier.arrive();
        load_warp_barrier.wait(/*phase=*/ 0);
        do_barrier = false;
      }
    }

    if constexpr (kLoadOuter) {
      load_v.template step<true>(outer_tile_iter, load_state_v, smem_pipe_write_outer, lane_predicate, outer_tile_count);
      load_k.template step<false>(outer_tile_iter, load_state_k, smem_pipe_write_outer, lane_predicate, outer_tile_count);
    }

    load_do.template step<false,false,true>(inner_tile_iter, load_state_do, smem_pipe_write_inner, lane_predicate, inner_tile_count, mcast_mask_b);
    load_odo.template step<true,true,false>(inner_tile_iter, load_state_odo, smem_pipe_write_inner, lane_predicate, inner_tile_count, mcast_mask_b);

    if constexpr (kLoadOuter) {
      load_v.template step<true>(outer_tile_iter, load_state_v, smem_pipe_write_outer, lane_predicate, outer_tile_count);
    }

    if constexpr (kLoadOuter) {
      while (outer_tile_count > 0) {
        load_k.template step<false>(outer_tile_iter, load_state_k, smem_pipe_write_outer, lane_predicate, outer_tile_count);
        load_v.template step<true>(outer_tile_iter, load_state_v, smem_pipe_write_outer, lane_predicate, outer_tile_count);
      }
    }

    CUTLASS_PRAGMA_NO_UNROLL
    while (inner_tile_count > 0) {
      while (inner_tile_count > 0) { 
        if (Fusion{}.is_contributing(make_coord(*inner_tile_iter, get<1>(blk_coord)), TileShape{}, problem_size)) {
          break;
        }
        inner_tile_count -= 4;
        ++inner_tile_iter;
      }
      load_q.template step<false,false,true>(inner_tile_iter, load_state_q, smem_pipe_write_inner, lane_predicate, inner_tile_count, mcast_mask_b);
      load_lse.template step<false,true,false>(inner_tile_iter, load_state_lse, smem_pipe_write_inner, lane_predicate, inner_tile_count, mcast_mask_b);

      load_do.template step<false,false,true>(inner_tile_iter, load_state_do, smem_pipe_write_inner, lane_predicate, inner_tile_count, mcast_mask_b);
      load_odo.template step<true,true,false>(inner_tile_iter, load_state_odo, smem_pipe_write_inner, lane_predicate, inner_tile_count, mcast_mask_b);
    }
  }

  template<class BlkCoord, class ProblemShape, class LoadWarpBarrier>
  CUTLASS_DEVICE void
  load_maybe_q(
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_size,
      MainloopPipelineQ& pipeline_outer, PipelineStateQ& smem_pipe_write_outer, 
      SharedStorage& storage,
      LoadWarpBarrier& load_warp_barrier, bool do_barrier)
  {
    // Load pattern:
    // K0    V0 K1     V1
    //    Q0       DO0    Q1 DO1 Q2 DO2 ...
    // K0 Q0 V0 K1 DO0 V1 ...
    int lane_predicate = cute::elect_one_sync();

    int outer_tile_count = NumMmaWarpGroups;

    auto outer_tile_iter = cute::make_coord_iterator(outer_tile_count);

    LoadK load_k{params.tma_load_k, pipeline_outer, storage.smem_k};
    auto load_state_k = load_k.init_state(_0{}, problem_size, TileShapeNM{}, blk_coord, outer_tile_count);

    LoadV load_v{params.tma_load_v, pipeline_outer, storage.smem_v};
    auto load_state_v = load_v.init_state(_0{}, problem_size, TileShapeNM{}, blk_coord, outer_tile_count);

    outer_tile_count *= 2; // K & V

    load_k.template step<false>(outer_tile_iter, load_state_k, smem_pipe_write_outer, lane_predicate, outer_tile_count);

    if (do_barrier) {
      load_warp_barrier.arrive();
      load_warp_barrier.wait(/*phase=*/ 0);
      do_barrier = false;
    }

    load_v.template step<true>(outer_tile_iter, load_state_v, smem_pipe_write_outer, lane_predicate, outer_tile_count);

    while (outer_tile_count > 0) {
      load_k.template step<false>(outer_tile_iter, load_state_k, smem_pipe_write_outer, lane_predicate, outer_tile_count);
      load_v.template step<true>(outer_tile_iter, load_state_v, smem_pipe_write_outer, lane_predicate, outer_tile_count);
    }
  }

  template<class BlkCoord, class ProblemShape, class MainloopPipelineReducer, class PipelineStateReducer>
  CUTLASS_DEVICE void
  reduce(
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_size,
      MainloopPipelineReducer& pipeline_reducer, PipelineStateReducer& smem_pipe_read_reducer, 
      SharedStorage& storage)
  {
    int lane_predicate = cute::elect_one_sync();

    Tensor mDQ_full = params.tma_red_dq.get_tma_tensor(select<2,4,0,1>(problem_size));
    Tensor gDQ_full = local_tile(mDQ_full, TileShapeMD{}, make_coord(_, _, _), Step<_1, _1, Underscore>{});
    Tensor gDQ = gDQ_full(_, _, _, _0{}, get<2,0>(blk_coord), get<2,1>(blk_coord));
    Tensor sDQ = make_tensor(make_smem_ptr(storage.smem_dq.data()), SmemLayoutDQ{});

    auto block_tma = params.tma_red_dq.get_slice(_0{});

    Tensor tDQsDQ = block_tma.partition_S(sDQ);
    Tensor tDQgDQ = block_tma.partition_D(gDQ);

    int inner_tile_count = get_inner_tile_count(blk_coord, problem_size);
    int g_index = 0;

    auto smem_pipe_release_reducer = smem_pipe_read_reducer;
    bool first = true;
    while (inner_tile_count > 0) {
      while (inner_tile_count > 0) { 
        if (Fusion{}.is_contributing(make_coord(g_index, get<1>(blk_coord)), TileShape{}, problem_size)) {
          break;
        }
        inner_tile_count -= 1;
        ++g_index;
      }
      if (inner_tile_count == 0) break;

      pipeline_reducer.consumer_wait(smem_pipe_read_reducer);
      if (lane_predicate == 1) {
        tma_store_wait<1>();
      }
      if (! first) {
        pipeline_reducer.consumer_release(smem_pipe_release_reducer);
        ++smem_pipe_release_reducer;
      } else {
        first = false;
      }
      if (lane_predicate == 1) {
        copy(params.tma_red_dq, tDQsDQ(_,_,_,smem_pipe_read_reducer.index()), tDQgDQ(_,_,_,g_index));
        tma_store_arrive();
      }
      ++smem_pipe_read_reducer;
      --inner_tile_count;
      ++g_index;
    }
    if (lane_predicate) {
      tma_store_wait<0>();
    }
    pipeline_reducer.consumer_release(smem_pipe_release_reducer);
    ++smem_pipe_release_reducer;
  }

  template<class BlkCoord, class ProblemShape, class MainloopPipelineReducer, class PipelineStateReducer, class MathWgOrderBarrier>
  CUTLASS_DEVICE auto
  compute(
      BlkCoord const& blk_coord, BlkCoord const& wg_coord,
      Params const& params, ProblemShape const& problem_size,
      MainloopPipeline& pipeline_inner, PipelineState& smem_pipe_read_inner, 
      MainloopPipelineQ& pipeline_outer, PipelineStateQ& smem_pipe_read_outer, 
      MainloopPipelineReducer& pipeline_reducer, PipelineStateReducer& smem_pipe_write_reducer,
      SharedStorage& storage,
      MathWgOrderBarrier& math_wg_order_barrier)
  {
    TiledMmaND tiled_mma_nd;

    Tensor acc_DV = partition_fragment_C(tiled_mma_nd, take<0,2>(TileShapeND{}));
    clear(acc_DV);

    Tensor acc_DK = partition_fragment_C(tiled_mma_nd, take<0,2>(TileShapeND{}));
    clear(acc_DK);

    int thread_idx = int(threadIdx.x) % cutlass::NumThreadsPerWarpGroup;

    PipelineState smem_pipe_release_inner = smem_pipe_read_inner;

    pipeline_outer.consumer_wait(smem_pipe_read_outer);
    PipelineStateQ smem_pipe_read_k = smem_pipe_read_outer;
    ++smem_pipe_read_outer;
    pipeline_outer.consumer_wait(smem_pipe_read_outer);
    PipelineStateQ smem_pipe_read_v = smem_pipe_read_outer;

    int inner_tile_count = get_inner_tile_count(wg_coord, problem_size);
                                                                                
    TiledMmaNM tiled_mma_nm;
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    auto thr_mma_nm = tiled_mma_nm.get_thread_slice(thread_idx);
    Tensor tSsK = thr_mma_nm.partition_A(sK);
    Tensor tSsQ = thr_mma_nm.partition_B(sQ);
    Tensor tSrK = thr_mma_nm.make_fragment_A(tSsK);
    Tensor tSrQ = thr_mma_nm.make_fragment_B(tSsQ);

    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    Tensor sDO = make_tensor(make_smem_ptr(storage.smem_do.data()), SmemLayoutDO{});

    Tensor tDPsV = thr_mma_nm.partition_A(sV);
    Tensor tDPsDO = thr_mma_nm.partition_B(sDO);
    Tensor tDPrV = thr_mma_nm.make_fragment_A(tDPsV);
    Tensor tDPrDO = thr_mma_nm.make_fragment_B(tDPsDO);

    auto thr_mma_nd = tiled_mma_nd.get_thread_slice(thread_idx);

    Tensor sDOp = make_tensor(make_smem_ptr(storage.smem_dop.data()), SmemLayoutDOp{});
    Tensor tDV_sDO = thr_mma_nd.partition_B(sDOp);
    Tensor tDVrDO = thr_mma_nd.make_fragment_B(tDV_sDO);


    Tensor sQp = make_tensor(make_smem_ptr(storage.smem_qp.data()), SmemLayoutQp{});
    Tensor tDK_sQ = thr_mma_nd.partition_B(sQp);
    Tensor tDKrQ = thr_mma_nd.make_fragment_B(tDK_sQ);


    int wg_idx = __shfl_sync(0xffffffff, get<1>(wg_coord) % NumMmaWarpGroups, 0);

    TiledMmaMD tiled_mma_md;
    auto thr_mma_md = tiled_mma_md.get_thread_slice(thread_idx);
    Tensor sDS = make_tensor(make_smem_ptr(storage.smem_ds.data()), SmemLayoutDS{});
    Tensor tDQsDS = thr_mma_md.partition_A(sDS);
    Tensor tDQrDS_full = thr_mma_md.make_fragment_A(tDQsDS);
    Tensor tDQrDS = tDQrDS_full(_,_,_,_);
    Tensor sKp = make_tensor(make_smem_ptr(storage.smem_kp.data()), SmemLayoutKp{});
    Tensor tDQsK = thr_mma_md.partition_B(sKp);
    Tensor tDQrK = thr_mma_md.make_fragment_B(tDQsK);

    Tensor sLSE = make_tensor(make_smem_ptr(storage.smem_lse.data()), make_shape(get<0>(TileShapeNM{}), get<1>(TileShapeNM{}), Int<StageCount>{}), make_stride(_0{}, _1{}, get<1>(TileShapeNM{})));
    Tensor tSsLSE = thr_mma_nm.partition_C(sLSE);

    Tensor sODO = make_tensor(make_smem_ptr(storage.smem_sumOdO.data()), make_shape(get<0>(TileShapeNM{}), get<1>(TileShapeNM{}), Int<StageCount>{}), make_stride(_0{}, _1{}, get<1>(TileShapeNM{})));
    Tensor tDPsODO = thr_mma_nm.partition_C(sODO);

    Tensor cS = make_identity_tensor(take<0,2>(TileShapeNM{}));
    Tensor tScS = thr_mma_nm.partition_C(cS);
    int n_block = get<1>(wg_coord);
    tScS.data() = tScS.data() + E<0>{} * n_block * get<0>(TileShapeNM{});


    // Transpose
    Tensor sDSp_full = sDS.compose(make_layout(make_shape(size<1>(sDS), size<0>(sDS), size<2>(sDS)), make_stride(size<0>(sDS), _1{}, size<1>(sDS) * size<0>(sDS))));
    Tensor sDSp = sDSp_full(_,_,_);
    Tensor tDPsDS = thr_mma_nm.partition_C(sDSp);

    auto thr_mma_nd_ss = TiledMmaND_SS{}.get_thread_slice(thread_idx);
    Tensor tDKsDSp = thr_mma_nd_ss.partition_A(sDSp);

    Tensor tDKrDSp = thr_mma_nd_ss.make_fragment_A(tDKsDSp);

    Tensor sDQ = make_tensor(make_smem_ptr(storage.smem_dq.data()), SmemLayoutDQ{});
    auto tDQsDQ_full = thr_mma_md.partition_C(sDQ);


    auto smem_pipe_read_k_other = smem_pipe_read_k;
    smem_pipe_read_k_other.advance(2);

    int k_index = 0;

    while (inner_tile_count > 0) {
      while (inner_tile_count > 0) { 
        if (Fusion{}.is_contributing(make_coord(k_index, get<1>(blk_coord)), TileShape{}, problem_size)) {
          break;
        }
        inner_tile_count -= 1;
        tScS.data() = tScS.data() + E<1>{} * get<1>(TileShapeNM{});
        k_index += 1;
      }
      if (inner_tile_count == 0) break;

      pipeline_inner.consumer_wait(smem_pipe_read_inner);
      PipelineState smem_pipe_read_q = smem_pipe_read_inner;
      ++smem_pipe_read_inner;
      PipelineState smem_pipe_read_do = smem_pipe_read_inner;
      ++smem_pipe_read_inner;

      // GEMM KQ -> S
      Tensor acc_S = partition_fragment_C(tiled_mma_nm, take<0,2>(TileShapeNM{}));

      warpgroup_fence_operand(acc_S);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_nm, tSrK(_,_,_,smem_pipe_read_k.index()), tSrQ(_,_,_,smem_pipe_read_q.index()), acc_S);
      warpgroup_commit_batch();
      
      pipeline_inner.consumer_wait(smem_pipe_read_do);

      // GEMM VdO -> dP
      Tensor acc_DP = partition_fragment_C(tiled_mma_nm, take<0,2>(TileShapeNM{}));

      warpgroup_fence_operand(acc_DP);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_nm, tDPrV(_,_,_,smem_pipe_read_v.index()), tDPrDO(_,_,_,smem_pipe_read_do.index()), acc_DP);
      warpgroup_commit_batch();

      Tensor reg_LSE = make_fragment_like<ElementAccumulator>(acc_S);
      for (int i = 0; i < size(reg_LSE); i++) {
        reg_LSE(i) = ((ElementAccumulator)std::log2(std::exp(1.0))) * tSsLSE(_,_,_,smem_pipe_read_q.index())(i);
      }

      Tensor reg_ODO = make_fragment_like<ElementAccumulator>(acc_S);
      if constexpr (decltype(get<0>(TileShape{}) != _128{})::value) {
        for (int i = 0; i < size(reg_ODO); i++) {
          reg_ODO(i) = tDPsODO(_,_,_,smem_pipe_read_do.index())(i);
        }
      }

      warpgroup_wait<1>();
      warpgroup_fence_operand(acc_S);

      math_wg_order_barrier.wait();
      // Compute S -> P
      Fusion{}.before_softmax(acc_S, tScS, problem_size);
      auto acc_P = make_fragment_like<ElementAccumulator>(acc_S);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_P); i++) {
        acc_P(i) = ::exp2f(params.scale_softmax_log2 * acc_S(i) - reg_LSE(i));
      }
      math_wg_order_barrier.arrive();

      if constexpr (decltype(get<0>(TileShape{}) == _128{})::value) {
        for (int i = 0; i < size(reg_ODO); i++) {
          reg_ODO(i) = tDPsODO(_,_,_,smem_pipe_read_do.index())(i);
        }
      }

      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_DP);
      
      // Compute dP P -> dS
      auto acc_DS = make_fragment_like<Element>(acc_DP);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_DS); i++) {
        // We could move the scale out and into the respective epilogues (or a final scaling step)
        acc_DS(i) = acc_P(i) * params.scale_softmax * (acc_DP(i) - reg_ODO(i));
      }

      // GEMM PdO -> dV
      auto op_P = make_acc_into_op<Element>(acc_P, typename TiledMmaND::LayoutA_TV{});
      warpgroup_fence_operand(acc_DV);
      warpgroup_fence_operand(op_P);
      warpgroup_arrive();
      cute::gemm(tiled_mma_nd, op_P, tDVrDO(_,_,_,smem_pipe_read_do.index()), acc_DV);
      warpgroup_commit_batch();

      // Store dS to smem dS'
      if (wg_idx == 0) math_wg_order_barrier.wait();

      auto recast_bits = [](auto sz, auto t) {
        return recast<uint_bit_t<decltype(sz)::value>>(t);
      };
      auto tDPsDS_v = recast_bits(Int<sizeof_bits_v<Element> * 2>{}, tDPsDS);
      auto acc_DS_v = recast_bits(Int<sizeof_bits_v<Element> * 2>{}, acc_DS);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_DS_v); i++) {
        tDPsDS_v(_,_,_,wg_idx)(i) = acc_DS_v(i);
      }

      cutlass::arch::fence_view_async_shared();
      if (wg_idx == 0) math_wg_order_barrier.arrive();

      // GEMM dS Q -> dK
      if (wg_idx == 1) {

        math_wg_order_barrier.wait();

        // GEMM dS' K -> dQ
        Tensor acc_DQ = partition_fragment_C(tiled_mma_md, take<0,2>(TileShapeMD{}));
  
        warpgroup_fence_operand(acc_DQ);
        warpgroup_arrive();
        gemm_zero_acc(tiled_mma_md, tDQrDS(_,_,_,0), tDQrK(_,_,_,smem_pipe_read_k_other.index()), acc_DQ);
        cute::gemm(tiled_mma_md, tDQrDS(_,_,_,1), tDQrK(_,_,_,smem_pipe_read_k.index()), acc_DQ);
        warpgroup_commit_batch();

        warpgroup_fence_operand(acc_DK);
        warpgroup_arrive();
        cute::gemm(TiledMmaND_SS{}, tDKrDSp(_,_,_,wg_idx), tDKrQ(_,_,_,smem_pipe_read_q.index()), acc_DK);
        warpgroup_commit_batch();

        warpgroup_wait<1>();
        warpgroup_fence_operand(acc_DK);
  
        warpgroup_wait<1>();
        warpgroup_fence_operand(acc_DQ);

        math_wg_order_barrier.arrive();
  
        pipeline_reducer.producer_acquire(smem_pipe_write_reducer);
        auto tDQsDQ = tDQsDQ_full(_,_,_,smem_pipe_write_reducer.index());
  
        // Store dQ to smem dQ'
        // Invoke TMA reduce on dQ'
        using Vec = uint_bit_t<sizeof_bits_v<ElementAccumulator> * 2>;
        auto tDQsDQ_v = recast<Vec>(tDQsDQ);
        auto acc_DQ_v = recast<Vec>(acc_DQ);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_DQ_v); i++) {
          tDQsDQ_v(i) = acc_DQ_v(i);
        }
  
        cutlass::arch::fence_view_async_shared();
        
        pipeline_reducer.producer_commit(smem_pipe_write_reducer);
        ++smem_pipe_write_reducer;
      } else {

        warpgroup_fence_operand(acc_DK);
        warpgroup_arrive();
        cute::gemm(TiledMmaND_SS{}, tDKrDSp(_,_,_,wg_idx), tDKrQ(_,_,_,smem_pipe_read_q.index()), acc_DK);
        warpgroup_commit_batch();

        warpgroup_wait<1>();
        warpgroup_fence_operand(acc_DK);

        pipeline_reducer.producer_acquire(smem_pipe_write_reducer);
        pipeline_reducer.producer_commit(smem_pipe_write_reducer);
        ++smem_pipe_write_reducer;
      }

      --inner_tile_count;

      pipeline_inner.consumer_release(smem_pipe_release_inner);  
      ++smem_pipe_release_inner;
      pipeline_inner.consumer_release(smem_pipe_release_inner);  
      ++smem_pipe_release_inner;

      tScS.data() = tScS.data() + E<1>{} * get<1>(TileShapeNM{});
      k_index += 1;
    }

    pipeline_outer.consumer_release(smem_pipe_read_k);           
    pipeline_outer.consumer_release(smem_pipe_read_outer);
    pipeline_reducer.producer_tail(smem_pipe_write_reducer);
    ++smem_pipe_read_outer;

    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_DK);
    warpgroup_fence_operand(acc_DV);
    
    return make_tuple(acc_DK, acc_DV);
  }
};

}  // namespace cutlass::fmha::collective
