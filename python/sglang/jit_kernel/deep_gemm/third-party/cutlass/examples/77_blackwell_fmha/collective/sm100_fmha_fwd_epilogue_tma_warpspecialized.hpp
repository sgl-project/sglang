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
#include "cute/layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

namespace cutlass::fmha::collective {

template<
  class Element,
  class ElementAcc,
  class TileShape,  // Q, D, _
  class StrideO,    // Q, D, B
  class StrideLSE_,   // Q, B
  class OrderLoadEpilogue = cute::false_type
>
struct Sm100FmhaFwdEpilogueTmaWarpspecialized {
    
  using Pipeline = cutlass::PipelineAsync<2>;

//  using SmemLayoutO = decltypa(make_layout(append<3>(select<0,1>(TileShape_WG{}), _2{})));
  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        cute::UMMA::Major::K, Element, tuple_element_t<0, TileShape>, tuple_element_t<1, TileShape>>());
//  using SmemLayoutAtomO = decltype(make_ordered_layout(select<0,1>(TileShape{}), Step<_1, _0>{}));
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, replace<2>(TileShape{}, _2{}), Step<_2, _1, _3>{}));
  using SmemLayoutO_ = SmemLayoutO;
  using StrideLSE = StrideLSE_;
  using ElementOut = Element;

  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  struct TensorStorage {

    using SmemLayoutO = SmemLayoutO_;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;

  };

  struct Arguments {
    Element* ptr_O;
    StrideO dO;

    ElementAcc* ptr_LSE;
    StrideLSE dLSE;
  };

  using TMA_O = decltype(make_tma_copy(
    SM90_TMA_STORE{},
    make_tensor((Element*) nullptr, repeat_like(StrideO{}, 0), StrideO{}),
    SmemLayoutO{}(_,_,_0{})
  ));


  struct Params {
    TMA_O tma_store_o;

    ElementAcc* ptr_LSE;
    StrideLSE dLSE;
  };

  // FMHA and MLA have different input ProblemShapes; 
  // get problem_shape_O according to the input ProblemShape.
  template<class ProblemShape>
  CUTLASS_DEVICE static constexpr
  auto get_problem_shape_O (
    ProblemShape const& problem_shape) {
    if constexpr (rank_v<decltype(get<2>(ProblemShape{}))> == 2) {
      return replace<1>(select<0,2,3>(problem_shape), get<2, 0>(problem_shape));
    } else {
      return select<0,2,3>(problem_shape);
    }
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace = nullptr) {

    auto ptr_O = args.ptr_O;
    StrideO dO = args.dO;

    auto problem_shape_O = get_problem_shape_O(problem_shape);

    if constexpr (is_variable_length_v<tuple_element_t<0, ProblemShape>>) {
      auto cumulative_length_q = get<0>(problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
          int max_length_q = get<0>(problem_shape).max_length;
          // for variable sequence lenght, the batch is in units of row_stride
          get<2,1>(dO) = get<0>(dO);
          get<2,1>(problem_shape_O) = max_length_q * (1 + get<2,1>(problem_shape_O));
          // offset ptr by the amount we add back in later
          ptr_O -= max_length_q * get<0>(dO);
      }
    }

    auto tma_store_o = make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(ptr_O, problem_shape_O, dO),
      SmemLayoutO{}(_,_,_0{})
    );

    return {
      tma_store_o,
      args.ptr_LSE,
      args.dLSE
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
  }

  const Params& params;

  CUTLASS_DEVICE Sm100FmhaFwdEpilogueTmaWarpspecialized(const Params& params) : params(params) {}

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE auto
  store(
      BlkCoord const& blk_coord_in, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& shared_storage,
      Pipeline& pipeline, typename Pipeline::PipelineState& pipeline_consumer_state) {

    BlkCoord blk_coord = blk_coord_in;
    uint32_t lane_predicate = cute::elect_one_sync();

    using X = Underscore;

    int o0_index = 2 * get<0>(blk_coord);
    int o1_index = 2 * get<0>(blk_coord) + 1;

    Tensor mO_qdl_p = params.tma_store_o.get_tma_tensor(get_problem_shape_O(problem_shape));
    // offset mode 0 by (max_length - real_length)
    // offset mode 3,1 by cumulative_length + real_length
    // the ptr is already offset by - max_length
    // so in total this achieves 
    int offs_0 = 0;
    int offs_2_1 = 0;

    if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
      auto cumulative_length_q = get<0>(params_problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
        int max_length_q = get<0>(params_problem_shape).max_length;
        offs_0 = max_length_q - get<0>(problem_shape);
        offs_2_1 = cumulative_length_q[get<2,1>(blk_coord)] + get<0>(problem_shape);
        get<2,1>(blk_coord) = 0;
      }
    }

    Tensor mO_qdl = domain_offset(make_coord(offs_0, _0{}, make_coord(_0{}, offs_2_1)), mO_qdl_p);

    Tensor gO_qdl = local_tile(mO_qdl, TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{});
    Tensor gO = gO_qdl(_, _, _, _0{}, get<2>(blk_coord));
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto block_tma = params.tma_store_o.get_slice(0);
    Tensor tOsO = block_tma.partition_S(sO);
    Tensor tOgO = block_tma.partition_D(gO);

    auto pipeline_release_state = pipeline_consumer_state;

    // O1 O2
    // one pipeline: O
    // wait from corr, issue tma store on smem
    pipeline.consumer_wait(pipeline_consumer_state);
    ++pipeline_consumer_state;

    if (lane_predicate) {
      copy(params.tma_store_o, tOsO(_,_,_,_0{}), tOgO(_,_,_,o0_index));
    }
    tma_store_arrive();

    pipeline.consumer_wait(pipeline_consumer_state);
    ++pipeline_consumer_state;

    if (lane_predicate) {
      copy(params.tma_store_o, tOsO(_,_,_,_1{}), tOgO(_,_,_,o1_index));
    }
    tma_store_arrive();

    tma_store_wait<1>();

    pipeline.consumer_release(pipeline_release_state);
    ++pipeline_release_state;

    tma_store_wait<0>();

    if constexpr (cute::is_same_v<OrderLoadEpilogue, cute::true_type>) {
      cutlass::arch::NamedBarrier::arrive((NumWarpsLoad + NumWarpsEpilogue) * NumThreadsPerWarp, 
                                          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }

    pipeline.consumer_release(pipeline_release_state);
    ++pipeline_release_state;

  } 

};

}  // namespace cutlass::fmha::collective
