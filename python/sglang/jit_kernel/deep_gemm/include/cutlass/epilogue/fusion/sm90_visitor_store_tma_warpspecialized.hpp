/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Visitor tree store operations for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"

#include "cute/tensor.hpp"
#include "sm90_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class EpilogueTile,
  class Element,
  FloatRoundStyle RoundStyle,
  class StrideMNL,
  class SmemLayoutAtom,
  class CopyOpR2S,
  int Alignment = 128 / sizeof_bits_v<Element>,
  bool EnableNullptr = true // Noop on nullptr params
>
struct Sm90AuxStore {
  using ElementAux = Element;
  static_assert(Alignment * sizeof_bits_v<Element> % 128 == 0, "sub-16B alignment not supported yet");

  constexpr static bool is_m_major = epilogue::collective::detail::is_m_major<StrideMNL>();
  // Find the max contiguous layout usable by TMA (if EpilogueTile is a non-compact tiler)
  using SmemShapeTma = decltype(make_shape(
      max_common_vector(make_layout(get<0>(EpilogueTile{})),make_layout(get<0>(EpilogueTile{}))),
      max_common_vector(make_layout(get<1>(EpilogueTile{})),make_layout(get<1>(EpilogueTile{})))));
  using SmemLayoutTma = decltype(tile_to_shape(
      SmemLayoutAtom{}, SmemShapeTma{},
      cute::conditional_t<is_m_major, Step<_2,_1>, Step<_1,_2>>{} ));
  using SmemLayout = decltype(tile_to_shape(
      SmemLayoutTma{},
      make_shape(size<0>(shape(EpilogueTile{})), size<1>(shape(EpilogueTile{})), Int<Stages>{}),
      cute::conditional_t<is_m_major, Step<_2,_1,_3>, Step<_1,_2,_3>>{} ));

  struct SharedStorage {
    alignas(cutlass::detail::alignment_for_swizzle(SmemLayout{}))
    array_aligned<Element, size(SmemLayout{})> smem_aux;
  };

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
  };

  struct Params {
    using TMA_Aux = decltype(make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(static_cast<Element*>(nullptr), repeat_like(StrideMNL{}, int32_t(0)), StrideMNL{}),
        SmemLayoutTma{}));
    TMA_Aux tma_store_aux;
    bool is_nullptr = false;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;

    bool is_nullptr = false;
    if constexpr (EnableNullptr) {
      is_nullptr = args.ptr_aux == nullptr;
    }

    typename Params::TMA_Aux tma_store_aux;
    if (not is_nullptr) {
      Tensor tensor_aux = make_tensor(args.ptr_aux, make_layout(make_shape(M,N,L), args.dAux));
      tma_store_aux = make_tma_copy(SM90_TMA_STORE{}, tensor_aux, SmemLayoutTma{});
    }

    return {tma_store_aux, is_nullptr};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90AuxStore() { }

  CUTLASS_HOST_DEVICE
  Sm90AuxStore(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params),
        smem_aux(const_cast<Element*>(shared_storage.smem_aux.data())) { }

  Params const* params_ptr;
  Element* smem_aux;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <
    class RTensor,
    class TiledR2S,
    class STensorR2S,
    class STensorS2G,
    class GTensorS2G
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
          RTensor&& tC_rAux,
          TiledR2S tiled_r2s,
          STensorR2S&& tRS_sAux,
          STensorS2G&& bSG_sAux,
          GTensorS2G&& bSG_gAux,
          Params const* params_ptr)
      : tiled_r2s(tiled_r2s),
        tC_rAux(cute::forward<RTensor>(tC_rAux)),
        tRS_sAux(cute::forward<STensorR2S>(tRS_sAux)),
        bSG_sAux(cute::forward<STensorS2G>(bSG_sAux)),
        bSG_gAux(cute::forward<GTensorS2G>(bSG_gAux)),
        params_ptr(params_ptr) {}

    TiledR2S tiled_r2s;
    RTensor tC_rAux;                                                                   // (CPY,CPY_M,CPY_N)
    STensorR2S tRS_sAux;                                                               // (R2S,R2S_M,R2S_N,PIPE)
    STensorS2G bSG_sAux;                                                               // (S2G,S2G_M,S2G_N,PIPE)
    GTensorS2G bSG_gAux;                                                               // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)
    Params const* params_ptr;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));                          // (EPI_V)
      tC_rAux_frg(epi_v) = convert_input(frg_input);

      return frg_input;
    }

    CUTLASS_DEVICE void
    postreduce(int epi_m, int epi_n, int store_iteration, bool issue_smem_store) {
      if constexpr (EnableNullptr) {
        if (params_ptr->is_nullptr) {
          return;
        }
      }

      using RLayoutR2S = decltype(cute::layout(TiledR2S{}.get_slice(0).retile_S(RTensor{})));
      Tensor tRS_rAux = make_tensor(tC_rAux.data(), RLayoutR2S{});                                 // (R2S,R2S_M,R2S_N)

      if (issue_smem_store) {
        int store_pipe_index = store_iteration % Stages;
        copy(tiled_r2s, tRS_rAux, tRS_sAux(_,_,_,store_pipe_index));
      }
    }

    CUTLASS_DEVICE void
    tma_store(int epi_m, int epi_n, int store_iteration, bool issue_tma_store) {
      if constexpr (EnableNullptr) {
        if (params_ptr->is_nullptr) {
          return;
        }
      }

      if (issue_tma_store) {
        // Issue the TMA store
        int store_pipe_index = store_iteration % Stages;
        copy(params_ptr->tma_store_aux, bSG_sAux(_,_,_,store_pipe_index), bSG_gAux(_,_,_,epi_m,epi_n));
      }
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    Tensor mAux = params_ptr->tma_store_aux.get_tma_tensor(make_shape(M,N,L));                               // (M,N,L)
    Tensor gAux = local_tile(mAux, take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));                 // (CTA_M,CTA_N)

    Tensor tC_gAux = sm90_partition_for_epilogue<ReferenceSrc>(                        // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
                      gAux, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tC_rAux = make_tensor<Element>(take<0,3>(shape(tC_gAux)));                  // (CPY,CPY_M,CPY_N)

    Tensor sAux_epi = cute::as_position_independent_swizzle_tensor(
                        make_tensor(make_smem_ptr(smem_aux), SmemLayout{}));     // (EPI_TILE_M,EPI_TILE_N,PIPE)
    Tensor gAux_epi = flat_divide(gAux, args.epi_tile);                          // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    auto tiled_r2s = conditional_return<ReferenceSrc>(
      make_tiled_copy_S(Copy_Atom<CopyOpR2S,Element>{}, args.tiled_copy),
      make_tiled_copy_D(Copy_Atom<CopyOpR2S,Element>{}, args.tiled_copy)
    );
    auto tRS_sAux = tiled_r2s.get_slice(args.thread_idx).partition_D(sAux_epi);               // (R2S,R2S_M,R2S_N,PIPE)

    ThrCopy thrblk_s2g = params_ptr->tma_store_aux.get_slice(_0{});
    Tensor bSG_sAux = thrblk_s2g.partition_S(sAux_epi);                                // (TMA,TMA_M,TMA_N,PIPE)
    Tensor bSG_gAux = thrblk_s2g.partition_D(gAux_epi);                                // (TMA,TMA_M,TMA_N,EPI_M,EPI_N)

    return ConsumerStoreCallbacks<decltype(tC_rAux), decltype(tiled_r2s), decltype(tRS_sAux), decltype(bSG_sAux), decltype(bSG_gAux)>(
            cute::move(tC_rAux),
            tiled_r2s,
            cute::move(tRS_sAux),
            cute::move(bSG_sAux),
            cute::move(bSG_gAux),
            params_ptr);
  }
};

template <
  class Element,
  class EpilogueTile,   // Unused
  FloatRoundStyle RoundStyle,
  class LayoutOrStrideMNL,
  class SmemLayoutAtom, // Unused
  class CopyOpR2S,      // Unused
  int Alignment,
  bool EnableNullptr
>
struct Sm90AuxStore<
  0, EpilogueTile, Element, RoundStyle, LayoutOrStrideMNL,
  SmemLayoutAtom, CopyOpR2S, Alignment, EnableNullptr
> {
  using ElementAux = Element;
  using StrideMNL = cutlass::gemm::TagToStrideC_t<LayoutOrStrideMNL>;

  struct SharedStorage { };

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90AuxStore() { }

  CUTLASS_HOST_DEVICE
  Sm90AuxStore(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<
    class GTensorR2G,
    class RTensor,
    class CTensorR2G,
    class ProblemShapeMNL
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        GTensorR2G&& tC_gAux,
        RTensor&& tC_rAux,
        CTensorR2G&& tC_cAux,
        ProblemShapeMNL problem_shape_mnl,
        Params const* params_ptr)
      : tC_gAux(cute::forward<GTensorR2G>(tC_gAux)),
        tC_rAux(cute::forward<RTensor>(tC_rAux)),
        tC_cAux(cute::forward<CTensorR2G>(tC_cAux)),
        problem_shape_mnl(problem_shape_mnl),
        params_ptr(params_ptr) {}

    GTensorR2G tC_gAux;
    RTensor tC_rAux;
    CTensorR2G tC_cAux;
    ProblemShapeMNL problem_shape_mnl;
    Params const* params_ptr;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(epi_v) = convert_input(frg_input);

      return frg_input;
    }

    CUTLASS_DEVICE void
    end_loop(int epi_m, int epi_n) {
      if constexpr (EnableNullptr) {
        if (params_ptr->ptr_aux == nullptr) {
          return;
        }
      }

      constexpr auto MCL = decltype(max_common_layout(tC_gAux(_,_,_,_0{},_0{}), tC_rAux)){};
      constexpr int V = cute::min(Alignment, size(MCL));

      Tensor tC_gAux_vec = recast<Array<Element, V>>(coalesce(tC_gAux(_,_,_,epi_m,epi_n)));
      Tensor tC_rAux_vec = recast<Array<Element, V>>(coalesce(tC_rAux));

      Tensor tC_cAux_vec = tensor<1>(zipped_divide(coalesce(tC_cAux(_,_,_,epi_m,epi_n)), MCL.compose(Int<V>{})));
      Tensor tC_pAux_vec = cute::lazy::transform(tC_cAux_vec, [&](auto const& c){ return elem_less(c, problem_shape_mnl); });

      copy_if(tC_pAux_vec, tC_rAux_vec, tC_gAux_vec);
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    auto problem_shape_mnl = make_shape(M,N,L);

    // Gmem Tensor
    Tensor mAux = make_tensor(
      make_gmem_ptr(params_ptr->ptr_aux), make_shape(M,N,L), params_ptr->dAux
    );
    Tensor tC_gAux = sm90_partition_for_epilogue<ReferenceSrc>(
                      mAux, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);

    // Register Tensor
    Tensor tC_rAux = make_tensor<Element>(take<0,3>(shape(tC_gAux)));

    // Predication support
    Tensor coordAux = make_identity_tensor(shape(mAux));
    Tensor tC_cAux = sm90_partition_for_epilogue<ReferenceSrc>(
                      coordAux, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);

    return ConsumerStoreCallbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux), decltype(problem_shape_mnl)>(
      cute::move(tC_gAux),
      cute::move(tC_rAux),
      cute::move(tC_cAux),
      problem_shape_mnl,
      params_ptr
    );

  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Reduction Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Scalar reduction
template <
  template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_0,_0>,
  bool EnableNullptr = true // Noop on nullptr params
>
struct Sm90ScalarReduction {
private:
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))>); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_0>{});
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(IsAtomic, "non-atomic scalar reduction not supported yet");

public:
  struct SharedStorage { };

  struct Arguments {
    ElementOutput* ptr_scalar = nullptr;
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dScalar = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
  #if !defined(CUTLASS_SKIP_REDUCTION_INIT)
    if constexpr (IsAtomic) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      Layout mScalar_layout = make_layout(make_shape(M,N,L), args.dScalar);
      if (args.ptr_scalar != nullptr) {
        return fill_workspace(args.ptr_scalar, ElementOutput(args.reduction_identity), cosize(mScalar_layout), stream, cuda_adapter);
      }
    }
  #endif

    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90ScalarReduction() { }

  CUTLASS_HOST_DEVICE
  Sm90ScalarReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params const params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class CTensor, class ThrResidue>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        int l_coord,
        CTensor tCcScalar,
        ThrResidue residue_tCcScalar,
        Params const& params)
      : scalar(params.reduction_identity),
        l_coord(l_coord),
        tCcScalar(tCcScalar),
        residue_tCcScalar(residue_tCcScalar),
        params(params) {}

    ElementCompute scalar;
    int l_coord;
    CTensor tCcScalar;                                                                 // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    ThrResidue residue_tCcScalar;
    Params params;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      if constexpr (EnableNullptr) {
        if (params.ptr_scalar == nullptr) {
          return frg_input;
        }
      }

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ReduceInput = RegReduceFn<ElementCompute>;
      ConvertInput convert_input{};
      ReduceInput reduce_input{};

      Array frg_I = convert_input(frg_input);
      Tensor tCcScalar_mn = tCcScalar(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        if (elem_less(tCcScalar_mn(epi_v * FragmentSize + i), residue_tCcScalar)) {
          scalar = reduce_input(scalar, frg_I[i]);
        }
      }

      return frg_input;
    }

    CUTLASS_DEVICE void
    end() {
      if constexpr (EnableNullptr) {
        if (params.ptr_scalar == nullptr) {
          return;
        }
      }

      using ConvertI = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
      using ReduceInput = GmemReduceFn<ElementOutput>;

      ConvertI convert_I{};
      ReduceInput reduce_input{};

      ElementOutput* ptr_scalar = params.ptr_scalar + l_coord * get<2>(params.dScalar);
      reduce_input(ptr_scalar, convert_I(scalar));
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks<decltype(args.tCcD), decltype(args.residue_tCcD)>(
      get<3>(args.tile_coord_mnkl), args.tCcD, args.residue_tCcD, params);
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////

// Row vector reduction
template <
  template <class> class RegReduceFn,
  template <class> class ShuffleReduceFn,
  template <class> class GmemReduceFn,
  int Stages,
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementOutput>,
  bool EnableNullptr = true, // Noop on nullptr params
  // If this is false, ptr_row is assumed to point to a compact n-major (ceil_div(M,CTA_M), round_nearest(N,CTA_N), L)
  // tensor of ElementCompute. It is the user's responsibility to reduce this to a (N, L) tensor of ElementOutput
  bool FinalReduction = true,
  // False means skip OOB predication if OOB inputs are known to be the reduction identity
  bool VisitCheckOOB = true,
  // Indicate the parameter order when calling RegReduceFn
  // Seq length equals the number of RegReduceFn parameters
  // No.0 represents tCrRow; No.1 and subsequent numbers sequentially represent frg_inputs in `visit`
  class RegReduceSeq = cute::seq<0, 1>
>
struct Sm90RowReduction {
private:
  static_assert(Stages == 0, "Smem usage not supported yet");
  static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))>); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{});
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(not (IsAtomic && not FinalReduction), "atomic reduction must be final");

public:
  struct SharedStorage { };

  struct Arguments {
    void* ptr_row = nullptr; // ElementOutput* if FinalReduction, else ElementCompute*
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dRow = {};
  };

  struct Params {
    void* ptr_row = nullptr;
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dRow = {};
    ElementCompute* reduction_buffer = nullptr;
    int* tile_counters = nullptr;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    ElementCompute* reduction_buffer;
    int* tile_counters = nullptr;
    if constexpr (IsAtomic) {
      reduction_buffer = nullptr;
    }
    else if constexpr (FinalReduction) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
      size_t tile_counters_offset = product(ceil_div(make_shape(size<>(M), size<>(N), L), make_shape(tile_M, tile_N))) * tile_N * sizeof(ElementCompute);
      tile_counters_offset = round_nearest(tile_counters_offset, MinWorkspaceAlignment);

      reduction_buffer = reinterpret_cast<ElementCompute*>(workspace);
      tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
    }
    else {
      reduction_buffer = reinterpret_cast<ElementCompute*>(args.ptr_row);
    }

    return {
      args.ptr_row,
      args.reduction_identity,
      args.dRow,
      reduction_buffer,
      tile_counters
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    if constexpr (IsAtomic || not FinalReduction) {
      return 0;
    }

    size_t workspace_size = 0;
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
    // Increment by size of reduction buffer
    workspace_size += product(ceil_div(make_shape(size<>(M),size<>(N),L), make_shape(tile_M, tile_N))) * tile_N * sizeof(ElementCompute);
    // Align and increment by size of tile counters
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);
    workspace_size += cute::ceil_div(size<>(N), tile_N) * sizeof(int);
    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    if constexpr (IsAtomic) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      Layout mRow_layout = make_layout(make_shape(size<>(M),size<>(N),size<>(L)), args.dRow);
      if (args.ptr_row != nullptr) {
        return fill_workspace(args.ptr_row, ElementOutput(args.reduction_identity), cosize(mRow_layout), stream, cuda_adapter);
      }
      return Status::kSuccess;
    }
    else if constexpr (FinalReduction) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
      size_t tile_counters_offset = product(ceil_div(make_shape(size<>(M),size<>(N),L), make_shape(tile_M, tile_N))) * tile_N * sizeof(ElementCompute);
      tile_counters_offset = round_nearest(tile_counters_offset, MinWorkspaceAlignment);

      int* tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
      size_t tile_counters_size = cute::ceil_div(size<>(N), tile_N) * sizeof(int);
      return zero_workspace(tile_counters, tile_counters_size, stream, cuda_adapter);
    }
    else {
      return Status::kSuccess;
    }
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90RowReduction() { }

  CUTLASS_HOST_DEVICE
  Sm90RowReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class ArgsTuple>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ArgsTuple&& args_tuple, Params const& params)
      : args_tuple(cute::forward<ArgsTuple>(args_tuple)),
        params(params) {}

    ArgsTuple args_tuple;
    Params const& params;
    bool do_final_reduction = false;

    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) {
      if constexpr (EnableNullptr) {
        if (params.ptr_row == nullptr) {
          return cute::get<0>(cute::make_tuple(frg_inputs...));
        }
      }

      auto& [ref_src, tCrRow, tCcRow, gRow_l, cRow, gBuf_ml, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        tile_coord_mnkl, residue_cRow, residue_tCcRow, epi_tile, tiled_copy, thread_idx] = args_tuple;
      Tensor tCrRow_mn = tCrRow(_,_,_,epi_m,epi_n);
      Tensor tCcRow_mn = tCcRow(_,_,_,epi_m,epi_n);

      if constexpr (VisitCheckOOB) {
        using ReduceInput = RegReduceFn<ElementCompute>;
        ReduceInput reduce_input{};

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          if (elem_less(tCcRow_mn(epi_v * FragmentSize + i), residue_tCcRow)) {
            ElementCompute& tCrRow_vmn = tCrRow_mn(epi_v * FragmentSize + i);
            tCrRow_vmn = transform_apply(cute::make_tuple(frg_inputs...),
                [&] (auto&& frg_input) {
                  return ElementCompute(frg_input[i]);
                },
                [&] (auto&&... cvt_frg_inputs) {
                  auto frg_compute_tuple = cute::make_tuple(tCrRow_vmn, cvt_frg_inputs...);
                  return cute::detail::apply(frg_compute_tuple, reduce_input, RegReduceSeq{});
                });
          }
        }
      }
      else {
        constexpr int RegFragSize = cute::max(1, static_cast<int>(sizeof(uint32_t) / sizeof(ElementCompute)));
        using ReduceInput = RegReduceFn<Array<ElementCompute, RegFragSize>>;
        ReduceInput reduce_input{};
        Tensor tCrRow_mn_frg = recast<Array<ElementCompute, RegFragSize>>(tCrRow_mn);

        constexpr int RegFragArraySize = FragmentSize / RegFragSize;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < RegFragArraySize; ++i) {
          Array<ElementCompute, RegFragSize>& tCrRow_vmn_frg = tCrRow_mn_frg(epi_v * RegFragArraySize + i);
          tCrRow_vmn_frg = transform_apply(cute::make_tuple(frg_inputs...),
              [&] (auto&& frg_input) {
                using ElementInput = typename cute::remove_cvref_t<decltype(frg_input)>::Element;
                using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, RegFragSize, RoundStyle>;
                using RegFragArr = Array<Array<ElementCompute, RegFragSize>, RegFragArraySize>;
                ConvertInput convert_input{};
                return convert_input(reinterpret_cast<RegFragArr&>(frg_input)[i]);
              },
              [&] (auto&&... cvt_frg_inputs) {
                auto frg_compute_tuple = cute::make_tuple(tCrRow_vmn_frg, cvt_frg_inputs...);
                return cute::detail::apply(frg_compute_tuple, reduce_input, RegReduceSeq{});
              });
        }
      }
      return cute::get<0>(cute::make_tuple(frg_inputs...));
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      if (not is_last_iteration) {
        return;
      }

      auto& [ref_src, tCrRow, tCcRow, gRow_l, cRow, gBuf_ml, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        tile_coord_mnkl, residue_cRow, residue_tCcRow, epi_tile, tiled_copy, thread_idx] = args_tuple;
      auto [m, n, k, l] = tile_coord_mnkl;
      constexpr bool ReferenceSrc = decltype(ref_src)::value;
      if constexpr (EnableNullptr) {
        if (params.ptr_row == nullptr) {
          return;
        }
      }

      // fully OOB CTA in partially OOB cluster
      if (not elem_less(cRow(_0{},_0{}), residue_cRow)) {
        return;
      }

      int lane_m = get<0>(lane_mn);
      [[maybe_unused]] bool is_reduced_lane = lane_m == 0;

      //
      // 1. Warp shuffle reduction
      //
      using FragmentShuffle = Array<ElementCompute, sizeof(uint64_t) / sizeof(ElementCompute)>;
      Tensor tCrRow_frg = recast<FragmentShuffle>(filter(tCrRow));
      using ReduceShuffle = ShuffleReduceFn<FragmentShuffle>;
      ReduceShuffle reduce_shuffle{};

      auto FrgSizePerLaneM = size(tCrRow_frg) / size<0>(lane_layout_MN);
      constexpr bool SwapShuffle = FrgSizePerLaneM > 0;

      //
      // Swap Shuffle
      //
      // The normal way to reduction among threads:
      // use shuffle to let *** the first half of threads *** have *** whole data *** from the second half of threads.
      // After each step of reduction, a half of threads won't work in the following steps.
      // That is, as the reduction progresses, the efficiency of shuffle & reduction instructions gradually change from 1/2, 1/4 to 1/32 (the worst case).
      //
      // To overcome this shortcoming, for a NxN matrix to be reduced among N threads as a 1XN vectors,
      // we use swap & shuffle aiming to let *** each half of threads *** have *** a half of data *** from the other half of threads.
      // After reduction, each half of threads should deal with a (N/2)x(N/2) sub-matrix independently in the following step.
      // We can recursively do this until the problem size is 1.
      //
      if constexpr (SwapShuffle) { // for a NxN matrix to be reduced among N threads as a 1XN vectors
        Tensor tCrRow_frg_ = logical_divide(tCrRow_frg, FrgSizePerLaneM);                       // (FrgSizePerLaneM, M)
        CUTLASS_PRAGMA_UNROLL
        for (int m = size<1>(tCrRow_frg_) / 2; m > 0; m /= 2) {
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < m; ++r) {
            auto frg_A = tCrRow_frg_(_,r);
            auto frg_B = tCrRow_frg_(_,r + m);
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < size(frg_A); ++v) {
              // Step1: swap
              if (not (lane_m & m)) { // the first half of threads swap fragments from the first half of data to the second
                cutlass::swap(frg_A(v), frg_B(v));
              }

              // Step2: shuffle
              uint64_t frg_shfl = reinterpret_cast<uint64_t&>(frg_A(v));
              // each half of threads get a half of data from the other half of threads
              frg_shfl = __shfl_xor_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(m, _0{}));

              // Step3: reduction
              frg_A(v) = reduce_shuffle(frg_B(v), reinterpret_cast<FragmentShuffle&>(frg_shfl));
            }
          }
        }
      }
      else {
        CUTLASS_PRAGMA_UNROLL
        for (int reduction_rows = size<0>(lane_layout_MN) / 2; reduction_rows > 0; reduction_rows /= 2) {
          CUTLASS_PRAGMA_UNROLL
          for (int frg_idx = 0; frg_idx < size(tCrRow_frg); ++frg_idx) {
            uint64_t frg_shfl = reinterpret_cast<uint64_t&>(tCrRow_frg(frg_idx));
            frg_shfl = __shfl_down_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(reduction_rows, _0{}));
            tCrRow_frg(frg_idx) = reduce_shuffle(tCrRow_frg(frg_idx), reinterpret_cast<FragmentShuffle&>(frg_shfl));
          }
        }
      }

      //
      // 2. Atomic reduction
      //
      if constexpr (IsAtomic) {
        // Filter so we don't issue redunant copies over stride-0 modes
        Tensor tCrRow_flt = filter_zeros(tCrRow);
        Tensor tCcRow_flt = make_tensor(tCcRow.data(), make_layout(tCrRow_flt.shape(), tCcRow.stride()));
        auto FltFrgSizePerLaneM = size(tCrRow_flt) / size<0>(lane_layout_MN);

        Tensor tCgRow = sm90_partition_for_epilogue<ReferenceSrc>(gRow_l(_,_,l), epi_tile, tiled_copy, thread_idx);
        Tensor tCgRow_flt = filter_zeros(tCgRow);
        // NOTE: atomic reduction is performed in the output type
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        using ReduceOutput = GmemReduceFn<ElementOutput>;
        ConvertOutput convert_output{};
        ReduceOutput reduce_output{};

        if constexpr (SwapShuffle) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < FltFrgSizePerLaneM; ++i) {
            int idx = lane_m * FltFrgSizePerLaneM + i;
            // Only care about OOB for N mode
            if (get<1>(tCcRow_flt(idx)) < get<1>(residue_tCcRow)) {
              reduce_output(&tCgRow_flt(idx), convert_output(tCrRow_flt(i)));
            }
          }
        }
        else {
          if (is_reduced_lane) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tCrRow_flt); ++i) {
              if (elem_less(tCcRow_flt(i), residue_tCcRow)) {
                reduce_output(&tCgRow_flt(i), convert_output(tCrRow_flt(i)));
              }
            }
          }
        }
        sync_fn();
      }

      //
      // 2. One warp in M, skip threadblock smem reduction
      //
      else if constexpr (decltype(size<0>(warp_layout_MN))::value <= 1) {
        // Dump warp reduction to gmem workspace
        using ElementGmem = cute::conditional_t<FinalReduction, ElementCompute volatile, ElementCompute>;
        Tensor tCgBuf = sm90_partition_for_epilogue<ReferenceSrc>(gBuf_ml(_,_,m,l), epi_tile, tiled_copy, thread_idx);

        if constexpr (SwapShuffle) {
          Tensor tCrRow_flt = filter(tCrRow);
          Tensor tCgBuf_flt = recast<ElementGmem>(filter(tCgBuf));
          auto FltFrgSizePerLaneM = size(tCrRow_flt) / size<0>(lane_layout_MN);
          Tensor tCgBuf_flt_ = logical_divide(tCgBuf_flt, FltFrgSizePerLaneM);               // (FltFrgSizePerLaneM, M)
          Tensor tCrRow_flt_ = logical_divide(tCrRow_flt, FltFrgSizePerLaneM);               // (FltFrgSizePerLaneM, M)
          copy_aligned(tCrRow_flt_(_,_0{}), tCgBuf_flt_(_,lane_m));
        }
        else {
          if (is_reduced_lane) {
            copy_aligned(tCrRow, recast<ElementGmem>(tCgBuf));
          }
        }
        sync_fn();
      }

      //
      // 2. Multiple warps in M, do threadblock smem reduction
      //
      else {
        Tensor sBuf = make_tensor(make_smem_ptr<ElementCompute>(raw_pointer_cast(smem_buffer.data())), sBuf_layout);
        static_assert(decltype(cosize(sBuf.layout()))::value * sizeof(ElementCompute) <=
                      decltype(cosize(smem_buffer.layout()))::value * sizeof(typename remove_cvref_t<STensor>::value_type),
                      "smem reduction buffer not large enough, use a larger epilogue tile");
        sync_fn();

        // Dump warp reduction to smem workspace
        Tensor tCsBuf = sm90_partition_for_epilogue<ReferenceSrc>(sBuf(_,_,get<0>(warp_mn)), epi_tile, tiled_copy, thread_idx);

        if constexpr (SwapShuffle) {
          Tensor tCrRow_flt = filter(tCrRow);
          Tensor tCsBuf_flt = filter(tCsBuf);
          auto FltFrgSizePerLaneM = size(tCrRow_flt) / size<0>(lane_layout_MN);
          Tensor tCsBuf_flt_ = logical_divide(tCsBuf_flt, FltFrgSizePerLaneM);               // (FltFrgSizePerLaneM, M)
          Tensor tCrRow_flt_ = logical_divide(tCrRow_flt, FltFrgSizePerLaneM);               // (FltFrgSizePerLaneM, M)
          copy_aligned(tCrRow_flt_(_,_0{}), tCsBuf_flt_(_,lane_m));
        }
        else {
          if (is_reduced_lane) {
            copy_aligned(tCrRow, tCsBuf);
          }
        }
        sync_fn();

        constexpr int SmemFragSize = cute::max(size_t{1}, sizeof(uint32_t) / sizeof(ElementCompute));
        using FragmentSmem = Array<ElementCompute, SmemFragSize>;
        using VectorSmem = uint_bit_t<sizeof_bits_v<FragmentSmem>>;
        using ReduceSmem = GmemReduceFn<FragmentSmem>;
        ReduceSmem reduce_smem{};

        Tensor sBuf_frg = recast<FragmentSmem>(filter_zeros(sBuf));
        Tensor sBuf_vec = recast<VectorSmem>(filter_zeros(sBuf));
        constexpr int FragsPerRow = decltype(size<1>(sBuf_frg))::value;

        constexpr int RowNum = decltype(size<0>(warp_layout_MN))::value;
        using FragmentSmemArray = Array<FragmentSmem, RowNum>;

        // Do the threadblock smem reduction
        using VectorGmem = cute::conditional_t<FinalReduction, VectorSmem volatile, VectorSmem>;
        Tensor gBuf_vec = recast<VectorGmem>(filter(gBuf_ml(_,_,m,l)));
        CUTLASS_PRAGMA_UNROLL
        for (int frg_idx = thread_idx; frg_idx < FragsPerRow; frg_idx += size(tiled_copy)) {
          FragmentSmemArray frg_smem;

          CUTLASS_PRAGMA_UNROLL
          for (int reduction_rows = 0; reduction_rows < RowNum; ++reduction_rows) {
            int FragsCurrRows = reduction_rows * FragsPerRow;
            frg_smem[reduction_rows] = sBuf_frg(FragsCurrRows + frg_idx);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int reduction_rows = RowNum / 2; reduction_rows > 0; reduction_rows /= 2) {
            CUTLASS_PRAGMA_UNROLL
            for (int row_idx = 0; row_idx < reduction_rows; ++row_idx) {
              frg_smem[row_idx] = reduce_smem(frg_smem[row_idx], frg_smem[row_idx + reduction_rows]);
            }
          }
          gBuf_vec(frg_idx) = reinterpret_cast<VectorSmem&>(frg_smem[0]);
        }
        sync_fn();
      }

      //
      // 3. Increment atomic counters to signal final gmem reduction
      //
      if constexpr (not IsAtomic && FinalReduction) {
        // Ensure gmem writes are visible to other threads before incrementing counter
        __threadfence();
        sync_fn();
        // Collective thread 0 increments atomic tile counter and copies value to smem
        int* prev_tile_count = reinterpret_cast<int*>(raw_pointer_cast(smem_buffer.data()));
        if (thread_idx == 0) {
          *prev_tile_count = atomicAdd(&params.tile_counters[n], 1);
        }
        sync_fn();
        // Broadcast tile count to other threads in CTA and determine final reduction status
        do_final_reduction = *prev_tile_count == size<2>(gBuf_ml) * size<3>(gBuf_ml) - 1;
        sync_fn();
      }
    }

    CUTLASS_DEVICE void
    end() {
      //
      // 4. Do final gmem reduction if necessary
      //
      if constexpr (not IsAtomic && FinalReduction) {
        if (not do_final_reduction) {
          return;
        }

        auto& [ref_src, tCrRow, tCcRow, gRow_l, cRow, gBuf_ml, sBuf_layout,
          lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
          tile_coord_mnkl, residue_cRow, residue_tCcRow, epi_tile, tiled_copy, thread_idx] = args_tuple;

        using ReduceOutput = GmemReduceFn<ElementCompute>;
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        ReduceOutput reduce_output{};
        ConvertOutput convert_output{};

        // Reduction over batches
        if (size<2>(stride(gRow_l)) == 0) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int n = thread_idx; n < size<1>(gBuf_ml); n += size(tiled_copy)) {
            Tensor tRgBuf_ml = gBuf_ml(_0{},n,_,_);
            ElementCompute output = tRgBuf_ml(_0{});
            CUTLASS_PRAGMA_NO_UNROLL
            for (int ml = 1; ml < size(tRgBuf_ml); ++ml) {
              output = reduce_output(output, tRgBuf_ml(ml));
            }
            if (elem_less(cRow(_0{},n), residue_cRow)) {
              gRow_l(_0{},n,_0{}) = convert_output(output);
            }
          }
        }
        // No reduction over batches
        else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int n = thread_idx; n < size<1>(gBuf_ml); n += size(tiled_copy)) {
            bool do_store = elem_less(cRow(_0{},n), residue_cRow);
            CUTLASS_PRAGMA_NO_UNROLL
            for (int l = 0; l < size<3>(gBuf_ml); ++l) {
              Tensor tRgBuf_m = gBuf_ml(_0{},n,_,l);
              ElementCompute output = tRgBuf_m(_0{});
              CUTLASS_PRAGMA_NO_UNROLL
              for (int m = 1; m < size(tRgBuf_m); ++m) {
                output = reduce_output(output, tRgBuf_m(m));
              }
              if (do_store) {
                gRow_l(_0{},n,l) = convert_output(output);
              }
            }
          }
        }

      }
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Layout ref_layout_MN = [&] () {
      auto mn_shape = shape(typename decltype(args.tiled_copy)::Tiler_MN{});
      if constexpr (ReferenceSrc) { return right_inverse(args.tiled_copy.get_layoutS_TV()).with_shape(mn_shape); }
      else                        { return right_inverse(args.tiled_copy.get_layoutD_TV()).with_shape(mn_shape); }
    }();                                                                                         // tile_mn -> tv_idx

    // Get the MN layout + coord of lanes to determine shuffle reduction iterations
    using _W = Int<decltype(args.tiled_copy)::TiledNumThr::value / NumThreadsPerWarp>;
    Layout tv2lane = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_1,_0,_0>>{};            //   tv_idx -> lane_idx
    Layout ref2lane = composition(tv2lane, ref_layout_MN);                                      //  tile_mn -> lane_idx
    Layout lane_layout_MN = make_layout(filter(get<0>(ref2lane)), filter(get<1>(ref2lane)));    //  lane_mn -> lane_idx
    Layout inv_lane_layout_MN = right_inverse(lane_layout_MN);                                  // lane_idx -> lane_mn
    int lane_idx = canonical_lane_idx();
    auto lane_mn = idx2crd(inv_lane_layout_MN(lane_idx), shape(lane_layout_MN));

    // Get the MN layout + coord of warps to determine smem reduction iterations
    Layout tv2warp = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_0,_1,_0>>{};            //   tv_idx -> warp_idx
    Layout ref2warp = composition(tv2warp, ref_layout_MN);                                      //  tile_mn -> warp_idx
    Layout warp_layout_MN = make_layout(filter(get<0>(ref2warp)), filter(get<1>(ref2warp)));    //  warp_mn -> warp_idx
    Layout inv_warp_layout_MN = right_inverse(warp_layout_MN);                                  // warp_idx -> warp_mn

    int warp_idx = args.thread_idx / NumThreadsPerWarp;
    auto warp_mn = idx2crd(inv_warp_layout_MN(warp_idx), shape(warp_layout_MN));

    // Partition output gmem and register tensors
    auto [tile_M, tile_N, tile_K] = args.tile_shape_mnk;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    Tensor mRow = make_tensor(make_gmem_ptr<ElementOutput>(params.ptr_row), make_shape(M,N,L), params.dRow); // (M,N,L)
    Tensor gRow_l = local_tile(mRow, take<0,2>(args.tile_shape_mnk), make_coord(m,n,_));             // (CTA_M,CTA_N,L)
    Tensor tCgRow = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      gRow_l(_,_,l), args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrRow = make_tensor_like<ElementCompute>(tCgRow);                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    fill(tCrRow, params.reduction_identity);

    // Partition gmem+smem reduction buffer tensors
    Layout gBuf_layout = make_layout(take<0,2>(args.tile_shape_mnk), make_stride(_0{}, _1{}));
    auto block_shape = ceil_div(make_shape(M,N,L), shape(gBuf_layout)); // (M_CNT, N_CNT, L_CNT)

    // Let the M_CNT (the num of partial reduction results) become the outer mode
    Layout block_layout = make_layout(block_shape, make_stride(get<1>(block_shape), _1{}, get<0>(block_shape) * get<1>(block_shape)));
    Layout mBuf_layout = blocked_product(gBuf_layout, block_layout);
    Tensor mBuf = make_tensor(make_gmem_ptr(params.reduction_buffer), mBuf_layout);                // (ceil_M,ceil_N,L)
    Tensor gBuf_ml = local_tile(mBuf, take<0,2>(args.tile_shape_mnk), make_coord(_,n,_));     // (CTA_M,CTA_N,REST_M,L)
    Layout sBuf_layout = blocked_product(gBuf_layout,                                          // (CTA_M,CTA_N,WARPS_M)
      make_layout(make_shape(_1{},_1{},size<0>(warp_layout_MN))));

    auto args_tuple = make_tuple(
        bool_constant<ReferenceSrc>{}, cute::move(tCrRow), args.tCcD, gRow_l, args.cD, gBuf_ml, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        args.tile_coord_mnkl, args.residue_cD, args.residue_tCcD, args.epi_tile, args.tiled_copy, args.thread_idx);
    return ConsumerStoreCallbacks<decltype(args_tuple)>(cute::move(args_tuple), params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Col vector reduction
template <
  template <class> class RegReduceFn,
  template <class> class ShuffleReduceFn,
  template <class> class GmemReduceFn,
  int Stages,
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_1,_0,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementOutput>,
  bool EnableNullptr = true, // Noop on nullptr params
  // If this is false, ptr_col is assumed to point to a compact m-major (round_nearest(M,CTA_M), ceil_div(N,CTA_N), L)
  // tensor of ElementCompute. It is the user's responsibility to reduce this to a (M, L) tensor of ElementOutput
  bool FinalReduction = true,
  // False means skip OOB predication if OOB inputs are known to be the reduction identity
  bool VisitCheckOOB = true
>
struct Sm90ColReduction {
private:
  static_assert(Stages == 0, "Smem usage not supported yet");
  static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))>); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_1,_0>{});
  static constexpr bool IsAtomic = is_atomic<GmemReduceFn<ElementCompute>>::value;
  static_assert(not (IsAtomic && not FinalReduction), "atomic reduction must be final");

public:
  struct SharedStorage { };

  struct Arguments {
    void* ptr_col = nullptr; // ElementOutput* if FinalReduction, else ElementCompute*
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dCol = {};
  };

  struct Params {
    void* ptr_col = nullptr;
    ElementCompute reduction_identity = ElementCompute(0);
    StrideMNL dCol = {};
    ElementCompute* reduction_buffer = nullptr;
    int* tile_counters = nullptr;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    ElementCompute* reduction_buffer;
    int* tile_counters = nullptr;
    if constexpr (IsAtomic) {
      reduction_buffer = nullptr;
    }
    else if constexpr (FinalReduction) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
      size_t tile_counters_offset = product(ceil_div(make_shape(M,N,L), make_shape(tile_M, tile_N))) * tile_M * sizeof(ElementCompute);
      tile_counters_offset = round_nearest(tile_counters_offset, MinWorkspaceAlignment);

      reduction_buffer = reinterpret_cast<ElementCompute*>(workspace);
      tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
    }
    else {
      reduction_buffer = reinterpret_cast<ElementCompute*>(args.ptr_col);
    }

    return {
      args.ptr_col,
      args.reduction_identity,
      args.dCol,
      reduction_buffer,
      tile_counters
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    if constexpr (IsAtomic || not FinalReduction) {
      return 0;
    }

    size_t workspace_size = 0;
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};

    // Increment by size of reduction buffer
    workspace_size += product(ceil_div(make_shape(M,N,L), make_shape(tile_M, tile_N))) * tile_M * sizeof(ElementCompute);
    // Align and increment by size of tile counters
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);
    workspace_size += cute::ceil_div(M, tile_M) * sizeof(int);

    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    if constexpr (IsAtomic) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      Layout mCol_layout = make_layout(make_shape(size<>(M),size<>(N),size<>(L)), args.dCol);
      if (args.ptr_col != nullptr) {
        return fill_workspace(args.ptr_col, ElementOutput(args.reduction_identity), cosize(mCol_layout), stream, cuda_adapter);
      }
      return Status::kSuccess;
    }
    else if constexpr (FinalReduction) {
      auto problem_shape_mnkl = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_mnkl;
      auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
      size_t tile_counters_offset = product(ceil_div(make_shape(M,N,L), make_shape(tile_M, tile_N))) * tile_M * sizeof(ElementCompute);
      tile_counters_offset = round_nearest(tile_counters_offset, MinWorkspaceAlignment);

      int* tile_counters = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(workspace) + tile_counters_offset);
      size_t tile_counters_size = cute::ceil_div(M, tile_M) * sizeof(int);
      return zero_workspace(tile_counters, tile_counters_size, stream, cuda_adapter);
    }
    else {
      return Status::kSuccess;
    }
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90ColReduction() { }

  CUTLASS_HOST_DEVICE
  Sm90ColReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class ArgsTuple>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ArgsTuple&& args_tuple, Params const& params)
      : args_tuple(cute::forward<ArgsTuple>(args_tuple)),
        params(params) {}

    ArgsTuple args_tuple;
    Params const& params;
    bool do_final_reduction = false;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      if constexpr (EnableNullptr) {
        if (params.ptr_col == nullptr) {
          return frg_input;
        }
      }

      auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
              lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
              tile_coord_mnkl, residue_cCol, residue_tCcCol, epi_tile, tiled_copy, thread_idx] = args_tuple;
      Tensor tCrCol_mn = tCrCol(_,_,_,epi_m,epi_n);
      Tensor tCcCol_mn = tCcCol(_,_,_,epi_m,epi_n);

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ReduceInput = RegReduceFn<ElementCompute>;
      ConvertInput convert_input{};
      ReduceInput reduce_input{};

      Array frg_I = convert_input(frg_input);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        if (!VisitCheckOOB || elem_less(tCcCol_mn(epi_v * FragmentSize + i), residue_tCcCol)) {
          ElementCompute& tCrCol_vmn = tCrCol_mn(epi_v * FragmentSize + i);
          tCrCol_vmn = reduce_input(tCrCol_vmn, frg_I[i]);
        }
      }

      return frg_input;
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      if (not is_last_iteration) {
        return;
      }

      auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
              lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
              tile_coord_mnkl, residue_cCol, residue_tCcCol, epi_tile, tiled_copy, thread_idx] = args_tuple;
      auto [m, n, k, l] = tile_coord_mnkl;
      constexpr bool ReferenceSrc = decltype(ref_src)::value;

      // Runtime nullptr is noop
      if constexpr (EnableNullptr) {
        if (params.ptr_col == nullptr) {
          return;
        }
      }

      // fully OOB CTA in partially OOB cluster
      if (not elem_less(cCol(_0{},_0{}), residue_cCol)) {
        return;
      }

      //
      // 1. Warp shuffle reduction
      //
      using FragmentShuffle = Array<ElementCompute, sizeof(uint64_t) / sizeof(ElementCompute)>;
      using ReduceShuffle = ShuffleReduceFn<FragmentShuffle>;
      ReduceShuffle reduce_shuffle{};
      Tensor tCrCol_frg = recast<FragmentShuffle>(filter(tCrCol));
      CUTLASS_PRAGMA_UNROLL
      for (int reduction_cols = size<1>(lane_layout_MN) / 2; reduction_cols > 0; reduction_cols /= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int frg_idx = 0; frg_idx < size(tCrCol_frg); ++frg_idx) {
          uint64_t frg_shfl = reinterpret_cast<uint64_t&>(tCrCol_frg(frg_idx));
          frg_shfl = __shfl_down_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(_0{},reduction_cols));
          tCrCol_frg(frg_idx) = reduce_shuffle(tCrCol_frg(frg_idx), reinterpret_cast<FragmentShuffle&>(frg_shfl));
        }
      }
      bool is_reduced_lane = get<1>(lane_mn) == 0;

      //
      // 2. Atomic reduction
      //
      if constexpr (IsAtomic) {
        // Filter so we don't issue redunant copies over stride-0 modes
        Tensor tCrCol_flt = filter_zeros(tCrCol);
        Tensor tCcCol_flt = make_tensor(tCcCol.data(), make_layout(tCrCol_flt.shape(), tCcCol.stride()));

        Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(gCol_l(_,_,l), epi_tile, tiled_copy, thread_idx);
        Tensor tCgCol_flt = filter_zeros(tCgCol);

        // NOTE: atomic reduction is performed in the output type
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        using ReduceOutput = GmemReduceFn<ElementOutput>;
        ConvertOutput convert_output{};
        ReduceOutput reduce_output{};

        if (is_reduced_lane) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrCol_flt); ++i) {
            if (elem_less(tCcCol_flt(i), residue_tCcCol)) {
              reduce_output(&tCgCol_flt(i), convert_output(tCrCol_flt(i)));
            }
          }
        }
        sync_fn();
      }

      //
      // 2. One warp in N, skip threadblock smem reduction
      //
      else if constexpr (decltype(size<1>(warp_layout_MN))::value <= 1) {
        // Dump warp reduction to gmem workspace
        using ElementGmem = cute::conditional_t<FinalReduction, ElementCompute volatile, ElementCompute>;
        Tensor tCgBuf = sm90_partition_for_epilogue<ReferenceSrc>(gBuf_nl(_,_,n,l), epi_tile, tiled_copy, thread_idx);
        if (is_reduced_lane) {
          copy_aligned(tCrCol, recast<ElementGmem>(tCgBuf));
        }
        sync_fn();
      }

      //
      // 2. Multiple warps in N, do threadblock smem reduction
      //
      else {
        Tensor sBuf = make_tensor(make_smem_ptr<ElementCompute>(raw_pointer_cast(smem_buffer.data())), sBuf_layout);
        static_assert(decltype(cosize(sBuf.layout()))::value * sizeof(ElementCompute) <=
                      decltype(cosize(smem_buffer.layout()))::value * sizeof(typename remove_cvref_t<STensor>::value_type),
                      "smem reduction buffer not large enough, use a larger epilogue tile");
        sync_fn();

        // Dump warp reduction to smem workspace
        Tensor tCsBuf = sm90_partition_for_epilogue<ReferenceSrc>(sBuf(_,_,get<1>(warp_mn)), epi_tile, tiled_copy, thread_idx);
        if (is_reduced_lane) {
          copy_aligned(tCrCol, tCsBuf);
        }
        sync_fn();

        constexpr int SmemFragSize = cute::max(size_t{1}, sizeof(uint32_t) / sizeof(ElementCompute));
        using FragmentSmem = Array<ElementCompute, SmemFragSize>;
        using VectorSmem = uint_bit_t<sizeof_bits_v<FragmentSmem>>;
        using ReduceSmem = GmemReduceFn<FragmentSmem>;
        ReduceSmem reduce_smem{};

        Tensor sBuf_frg = recast<FragmentSmem>(filter_zeros(sBuf));
        Tensor sBuf_vec = recast<VectorSmem>(filter_zeros(sBuf));
        constexpr int FragsPerCol = decltype(size<0>(sBuf_frg))::value;

        // Do the threadblock smem reduction
        CUTLASS_PRAGMA_UNROLL
        for (int reduction_cols = size<1>(warp_layout_MN) / 2; reduction_cols > 1; reduction_cols /= 2) {
          int FragsPerReduction = reduction_cols * FragsPerCol;
          CUTLASS_PRAGMA_NO_UNROLL
          for (int frg_idx = thread_idx; frg_idx < FragsPerReduction; frg_idx += size(tiled_copy)) {
            FragmentSmem frg_smem = reduce_smem(sBuf_frg(frg_idx), sBuf_frg(frg_idx + FragsPerReduction));
            sBuf_vec(frg_idx) = reinterpret_cast<VectorSmem&>(frg_smem);
          }
          sync_fn();
        }

        // Do final smem reduction and dump to gmem workspace
        using VectorGmem = cute::conditional_t<FinalReduction, VectorSmem volatile, VectorSmem>;
        Tensor gBuf_vec = recast<VectorGmem>(filter(gBuf_nl(_,_,n,l)));
        CUTLASS_PRAGMA_NO_UNROLL
        for (int frg_idx = thread_idx; frg_idx < FragsPerCol; frg_idx += size(tiled_copy)) {
          FragmentSmem frg_smem = reduce_smem(sBuf_frg(frg_idx), sBuf_frg(frg_idx + FragsPerCol));
          gBuf_vec(frg_idx) = reinterpret_cast<VectorSmem&>(frg_smem);
        }
        sync_fn();
      }

      //
      // 3. Increment atomic counters to signal final gmem reduction
      //
      if constexpr (not IsAtomic && FinalReduction) {
        // Ensure gmem writes are visible to other threads before incrementing counter
        __threadfence();
        sync_fn();
        // Collective thread 0 increments atomic tile counter and copies value to smem
        int* prev_tile_count = reinterpret_cast<int*>(raw_pointer_cast(smem_buffer.data()));
        if (thread_idx == 0) {
          *prev_tile_count = atomicAdd(&params.tile_counters[m], 1);
        }
        sync_fn();
        // Broadcast tile count to other threads in CTA and determine final reduction status
        do_final_reduction = *prev_tile_count == size<2>(gBuf_nl) * size<3>(gBuf_nl) - 1;
        sync_fn();
      }
    }

    CUTLASS_DEVICE void
    end() {
      //
      // 4. Do final gmem reduction if necessary
      //
      if constexpr (not IsAtomic && FinalReduction) {
        if (not do_final_reduction) {
          return;
        }

        auto& [ref_src, tCrCol, tCcCol, gCol_l, cCol, gBuf_nl, sBuf_layout,
                lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
                tile_coord_mnkl, residue_cCol, residue_tCcCol, epi_tile, tiled_copy, thread_idx] = args_tuple;

        using ReduceOutput = GmemReduceFn<ElementCompute>;
        using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
        ReduceOutput reduce_output{};
        ConvertOutput convert_output{};

        // Reduction over batches
        if (size<2>(stride(gCol_l)) == 0) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int m = thread_idx; m < size<0>(gBuf_nl); m += size(tiled_copy)) {
            Tensor tRgBuf_nl = gBuf_nl(m,_0{},_,_);
            ElementCompute output = tRgBuf_nl(_0{});
            CUTLASS_PRAGMA_NO_UNROLL
            for (int nl = 1; nl < size(tRgBuf_nl); ++nl) {
              output = reduce_output(output, tRgBuf_nl(nl));
            }
            if (elem_less(cCol(m,_0{}), residue_cCol)) {
              gCol_l(m,_0{},_0{}) = convert_output(output);
            }
          }
        }
        // No reduction over batches
        else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int m = thread_idx; m < size<0>(gBuf_nl); m += size(tiled_copy)) {
            bool do_store = elem_less(cCol(m,_0{}), residue_cCol);
            CUTLASS_PRAGMA_NO_UNROLL
            for (int l = 0; l < size<3>(gBuf_nl); ++l) {
              Tensor tRgBuf_n = gBuf_nl(m,_0{},_,l);
              ElementCompute output = tRgBuf_n(_0{});
              CUTLASS_PRAGMA_NO_UNROLL
              for (int n = 1; n < size(tRgBuf_n); ++n) {
                output = reduce_output(output, tRgBuf_n(n));
              }
              if (do_store) {
                gCol_l(m,_0{},l) = convert_output(output);
              }
            }
          }
        }

      }
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Layout ref_layout_MN = [&] () {
      auto mn_shape = shape(typename decltype(args.tiled_copy)::Tiler_MN{});
      if constexpr (ReferenceSrc) { return right_inverse(args.tiled_copy.get_layoutS_TV()).with_shape(mn_shape); }
      else                        { return right_inverse(args.tiled_copy.get_layoutD_TV()).with_shape(mn_shape); }
    }();                                                                                         // tile_mn -> tv_idx

    // Get the MN layout + coord of lanes to determine shuffle reduction iterations
    using _W = Int<decltype(args.tiled_copy)::TiledNumThr::value / NumThreadsPerWarp>;
    Layout tv2lane = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_1,_0,_0>>{};            //   tv_idx -> lane_idx
    Layout ref2lane = composition(tv2lane, ref_layout_MN);                                      //  tile_mn -> lane_idx
    Layout lane_layout_MN = make_layout(filter(get<0>(ref2lane)), filter(get<1>(ref2lane)));    //  lane_mn -> lane_idx
    Layout inv_lane_layout_MN = right_inverse(lane_layout_MN);                                  // lane_idx -> lane_mn
    int lane_idx = canonical_lane_idx();
    auto lane_mn = idx2crd(inv_lane_layout_MN(lane_idx), shape(lane_layout_MN));

    // Get the MN layout + coord of warps to determine smem reduction iterations
    Layout tv2warp = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_0,_1,_0>>{};            //   tv_idx -> warp_idx
    Layout ref2warp = composition(tv2warp, ref_layout_MN);                                      //  tile_mn -> warp_idx
    Layout warp_layout_MN = make_layout(filter(get<0>(ref2warp)), filter(get<1>(ref2warp)));    //  warp_mn -> warp_idx
    Layout inv_warp_layout_MN = right_inverse(warp_layout_MN);                                  // warp_idx -> warp_mn
    int warp_idx = args.thread_idx / NumThreadsPerWarp;
    auto warp_mn = idx2crd(inv_warp_layout_MN(warp_idx), shape(warp_layout_MN));

    // Partition output gmem and register tensors
    auto [tile_M, tile_N, tile_K] = args.tile_shape_mnk;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    Tensor mCol = make_tensor(make_gmem_ptr<ElementOutput>(params.ptr_col), make_shape(M,N,L), params.dCol); // (M,N,L)
    Tensor gCol_l = local_tile(mCol, take<0,2>(args.tile_shape_mnk), make_coord(m,n,_));             // (CTA_M,CTA_N,L)
    Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
                      gCol_l(_,_,l), args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrCol = make_tensor_like<ElementCompute>(tCgCol);                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    fill(tCrCol, params.reduction_identity);

    // Partition gmem+smem reduction buffer tensors
    Layout gBuf_layout = make_layout(take<0,2>(args.tile_shape_mnk), make_stride(_1{}, _0{}));
    Layout mBuf_layout = blocked_product(gBuf_layout, make_layout(ceil_div(make_shape(M,N,L), shape(gBuf_layout))));
    Tensor mBuf = make_tensor(make_gmem_ptr(params.reduction_buffer), mBuf_layout);                // (ceil_M,ceil_N,L)
    Tensor gBuf_nl = local_tile(mBuf, take<0,2>(args.tile_shape_mnk), make_coord(m,_,_));     // (CTA_M,CTA_N,REST_N,L)
    Layout sBuf_layout = blocked_product(gBuf_layout,make_layout(make_shape(_1{},_1{},size<1>(warp_layout_MN)))); // (CTA_M,CTA_N,WARPS_N)

    auto args_tuple = make_tuple(
        bool_constant<ReferenceSrc>{}, cute::move(tCrCol), args.tCcD, gCol_l, args.cD, gBuf_nl, sBuf_layout,
        lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
        args.tile_coord_mnkl, args.residue_cD, args.residue_tCcD, args.epi_tile, args.tiled_copy, args.thread_idx);
    return ConsumerStoreCallbacks<decltype(args_tuple)>(std::move(args_tuple), params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Batch matrix reduction
template <
  int Stages,
  class EpilogueTile,
  class Element,
  class StrideMNL,
  class CopyOpR2S,
  class SmemLayoutAtom,
  int Alignment = 128 / sizeof_bits_v<Element>,
  bool EnableNullptr = true // Noop on nullptr params
>
struct Sm90MatrixReduction;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
