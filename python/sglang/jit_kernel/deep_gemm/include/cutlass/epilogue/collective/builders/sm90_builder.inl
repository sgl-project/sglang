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
#pragma once

#include "cute/atom/mma_traits_sm90.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/atom/copy_traits_sm90.hpp"

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(type_traits)
#else
#include <type_traits>
#endif

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

///////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the parameterized dispatch policy for the TMA epilogue
template<class TileShapeMNK, class EpilogueTileMN, class ElementC, class ElementD, class Schedule>
constexpr auto
sm90_get_tma_dispatch_policy() {
  using namespace cute;

  constexpr int EpiTiles = size(shape_div(take<0,2>(TileShapeMNK{}), EpilogueTileMN{}));
  constexpr int FragmentSize = size(EpilogueTileMN{}) / (detail::sm90_is_cooperative_v<Schedule> ? 256 : 128);
  // 8b residuals load fast and consume little smem, so the perf cost of waiting on stores to finish outweighs the cost of extra allocation
  constexpr bool ReuseSmem = (sizeof_bits_v<ElementC> == sizeof_bits_v<ElementD>) && (sizeof_bits_v<ElementD> > 8);
  // TMA store delay performs worse with residual loads and compilicates tensormap updates for Ptr-Array GEMMs
  constexpr bool DelayTmaStore = is_void_v<ElementC> && !detail::sm90_is_ptr_array_tma_v<Schedule>;
  constexpr int StagesD = cute::min(EpiTiles, 2);
  constexpr int StagesC = ReuseSmem ? cute::max(cute::min(EpiTiles, 4), StagesD+1)
                                    : cute::min(EpiTiles, 4);

  if constexpr (detail::sm90_is_ptr_array_tma_v<Schedule>) {
      return Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, 
                                            DelayTmaStore, Schedule::NumEpilogueWarpGroups>{};
  } 
  else {
    return Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>{};
  }
}

// Returns the smem layout atom to be used for C or D matrix
template<class GmemStrideType, class Element, class EpilogueTile_MN>
constexpr auto
sm90_get_epilogue_smem_swizzle_layout_atom() {
  using namespace cute;

  // ColMajor C/D (M-major)
  if constexpr (cutlass::gemm::detail::is_major<0>(GmemStrideType{})) {
    return cutlass::gemm::collective::detail::ss_smem_selector<
      cute::GMMA::Major::MN, Element, decltype(get<0>(EpilogueTile_MN{})), decltype(get<1>(EpilogueTile_MN{}))
    >();
  }
  // RowMajor C/D (N-major)
  else if constexpr (cutlass::gemm::detail::is_major<1>(GmemStrideType{})) {
    return cutlass::gemm::collective::detail::ss_smem_selector<
      cute::GMMA::Major::K , Element, decltype(get<0>(EpilogueTile_MN{})), decltype(get<1>(EpilogueTile_MN{}))
    >();
  }
  else {
    static_assert(cutlass::detail::dependent_false<GmemStrideType>, "Unsupported gmem layout.");
  }
}

// Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.
template <class ElementD, class EpilogueTileType, class Schedule, class TileShape_MNK>
constexpr auto
sm90_compute_tile_shape_or_override() {
  if constexpr (cute::is_same_v<EpilogueTileType, EpilogueTileAuto>) {
    auto epi_tile = [&] () {
      if constexpr (detail::sm90_is_cooperative_v<Schedule>) {

        auto tile_m = cute::min(_128{}, size<0>(TileShape_MNK{}));
        auto tile_n = cute::gcd(cute::min(_32{}, size<1>(TileShape_MNK{})), size<1>(TileShape_MNK{}));
        return make_shape(tile_m, tile_n);
      }
      else if constexpr (detail::sm90_is_warp_specialized_v<Schedule>) {
        constexpr int N_perf = (sizeof_bits_v<ElementD> == 8) && (size<1>(TileShape_MNK{}) % 64 == 0) ? 64 : 32;
        auto tile_m = cute::min(_64{}, size<0>(TileShape_MNK{}));
        auto tile_n = cute::gcd(cute::min(Int<N_perf>{}, size<1>(TileShape_MNK{})), size<1>(TileShape_MNK{}));
        return make_shape(tile_m, tile_n);
      }
      else {
        static_assert(cutlass::detail::dependent_false<Schedule>, "Unsupported schedule.");
      }
    }();

    return cute::transform(epi_tile, seq<0,1>{},
      [] (auto epi_tiler, auto I) {
        auto cta_tiler = make_layout(get<I>(TileShape_MNK{}));
        // This is a multimodal CTA tiler, transform before returning
        if constexpr (depth(cta_tiler) > 0) {
          // This is an implicit multimodal tiler, match profile and return
          if constexpr (tuple_size_v<decltype(shape(cta_tiler))> == 1) {
            return make_tile(epi_tiler);
          }
          // This is an explicit multimodal tiler, compose out epi tiler
          else {
            return composition(cta_tiler, epi_tiler);
          }
        }
        // This is a flat CTA tiler, no need for transformation
        else {
          return epi_tiler;
        }
      });
  }
  else if constexpr (cute::is_tuple<EpilogueTileType>::value) {
    EpilogueTileType epi_tile;
    constexpr int M = size<0>(shape(epi_tile));
    constexpr int N = size<1>(shape(epi_tile));

    static_assert(!is_layout<EpilogueTileType>::value, "EpilogueTile must be a cute::Tile or cute::Shape");
    static_assert(M ==  64 && detail::sm90_is_warp_specialized_v<Schedule> ||
                  M == 128 && detail::sm90_is_cooperative_v<Schedule>, "Unsupported tile shape");
    static_assert(N % 16 == 0, "Unsupported tile shape");

    return epi_tile;
  }
  else {
    static_assert(cutlass::detail::dependent_false<EpilogueTileType>, "Invalid type for EpilogueTileType.");
  }
}

// aux fusion callbacks builder for sm90 tma epilogue
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class FusionOp,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class AccLoadOp,
  class ElementAccumulator
>
struct CallbacksBuilder<
  Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  FusionOp,
  TileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && not cute::is_subbyte_v<typename FusionOp::ElementAux>> // aux subbyte tensor doesn't use smem
> {
  using GmemStrideTypeAux = gemm::TagToStrideC_t<typename FusionOp::GmemLayoutTagAux>;
  using SmemLayoutAtomAux = decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());
  using CopyOpR2S = decltype(detail::sm90_get_smem_store_op_for_accumulator<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());
  using CopyOpS2R = decltype(detail::sm90_get_smem_load_op_for_source<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());
  using SmemCopyOpAux = cute::conditional_t<FusionOp::IsAuxOutSupported, CopyOpR2S, CopyOpS2R>;

  using Callbacks = fusion::FusionCallbacks<
    Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOp, TileShape_MNK, EpilogueTile_MN,
    SmemLayoutAtomAux, SmemCopyOpAux
  >;
};

// ptr array aux fusion callbacks builder for sm90 tma epilogue
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class FusionOp,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class AccLoadOp,
  class ElementAccumulator
>
struct CallbacksBuilder<
  Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumEpilogueWarpGroups>,
  FusionOp,
  TileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && not cute::is_subbyte_v<typename FusionOp::ElementAux>> // aux subbyte tensor doesn't use smem
> {
  using GmemStrideTypeAux = gemm::TagToStrideC_t<typename FusionOp::GmemLayoutTagAux>;
  using SmemLayoutAtomAux = decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());
  using CopyOpR2S = decltype(detail::sm90_get_smem_store_op_for_accumulator<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());
  using CopyOpS2R = decltype(detail::sm90_get_smem_load_op_for_source<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());
  using SmemCopyOpAux = cute::conditional_t<FusionOp::IsAuxOutSupported, CopyOpR2S, CopyOpS2R>;

  using Callbacks = fusion::FusionCallbacks<
    Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumEpilogueWarpGroups>,
    FusionOp, TileShape_MNK, EpilogueTile_MN,
    SmemLayoutAtomAux, SmemCopyOpAux
  >;
};

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class FusionOp,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class AccLoadOp,
  class ElementAccumulator
>
struct CallbacksBuilder<
  Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  FusionOp,
  TileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && sizeof_bits_v<typename FusionOp::ElementAux> == 1>
> {
  using Callbacks = fusion::FusionCallbacks<
    Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOp, TileShape_MNK, EpilogueTile_MN,
    Layout<_1,_0>, DefaultCopy // aux bit tensor doesn't use smem
  >;
};

// Helper for building TMA warp-specialized collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template <
  class TileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD_,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOpOrCallbacks,
  class DispatchPolicy
>
struct Sm90TmaBuilderImpl {
  // C/D should meet TMA alignment requirement if not void
  static_assert(detail::is_aligned<ElementC_, AlignmentC, ElementD_, AlignmentD>(),
                "C/D Should meet TMA alignment requirement\n");
  // Passing void D disables destination store + smem allocation
  using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                     fusion::get_element_aux_t<FusionOpOrCallbacks>, ElementD_>;

  // Passing void C disables source load + smem allocation
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,ElementD,ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,GmemLayoutTagD,GmemLayoutTagC_>;

  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;
  
  using UnderlyingGmemStrideTypeC = cute::remove_pointer_t<GmemStrideTypeC>;
  using UnderlyingGmemStrideTypeD = cute::remove_pointer_t<GmemStrideTypeD>;

  using CopyOpS2G = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagD>,
      SM90_TMA_STORE_IM2COL,
      SM90_TMA_STORE
    >;
  using CopyOpG2S = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagC>,
      SM90_TMA_LOAD_IM2COL,
      SM90_TMA_LOAD
    >;

  // Get the smallest tiled copy we can use to retile the accumulators
  // using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
  using CopyAtomC = cute::conditional_t<
    size<1>(EpilogueTile_MN{}) % 16 == 0,
    Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>,
    cute::conditional_t<
      size<1>(EpilogueTile_MN{}) % 8 == 0,
      Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>,
      void
    >
  >;
  static_assert(!cute::is_same_v<CopyAtomC, void>, "CopyAtomC can't be void, divisiblity check for EpilogueTile_MN failed");
  // Get register to register tiled copy that happen before shared memory store.
  // Apply void as no register transform op needed currently.
  using CopyOpR2R = void;

  // TMA builder allows for passing callbacks directly, which is either a fusion::FusionCallbacks
  // instance or a direct visitor implementation, e.g. fusion::Sm90LinearCombination
  using FusionCallbacks = 
    typename CallbacksBuilder<
      DispatchPolicy,
      FusionOpOrCallbacks,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator
    >::Callbacks;

  using CollectiveOp = cutlass::epilogue::collective::CollectiveEpilogue<
      DispatchPolicy,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD_,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_load_op_for_source<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      CopyOpS2G,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_store_op_for_accumulator<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      CopyAtomC,
      CopyOpR2R
    >;
};

///////////////////////////////////////////////////////////////////////////////
// Descriptor classes for defining EVT nodes
// Some of the epilogue visitor nodes require non-intuitive template arguments
// such as CopyOpS2R for AuxLoad node. Traditionaly, these are resolved by the
// builder classes. Here we provide a set of descriptor classes that resolve
// these template arguments from more intuitive types such as Stride, Layout

// Get TileShape, EpilogueTile, Dispatch Policy, StagesC, and STagesD
template<
  typename TileShape_MNK,
  typename EpilogueTileType, 
  typename ElementC,
  typename ElementD,
  typename Schedule
>
struct EpilogueDescriptor {
  using TileShape = TileShape_MNK;
  using EpilogueTile = 
    decltype(
      detail::sm90_compute_tile_shape_or_override<
        ElementD, EpilogueTileType, Schedule, TileShape_MNK
      >()
    );
  using DispatchPolicy = 
    decltype(
      detail::sm90_get_tma_dispatch_policy<
        TileShape_MNK, EpilogueTile, 
        ElementC, ElementD, Schedule
      >()
    );
  constexpr static int StagesC = DispatchPolicy::StagesC;
  constexpr static int StagesD = DispatchPolicy::StagesD;
};

// Get Stride, SmemLayout, and CopyOpS2R for AuxLoad node
template<
  typename EpilogueDescriptor,
  typename StrideOrLayoutTag,
  typename ElementAux
>
struct AuxLoadDescriptor {
  constexpr static int Stages = EpilogueDescriptor::StagesC;
  using EpilogueTile = typename EpilogueDescriptor::EpilogueTile;
  using Element = ElementAux;
  using Stride = cutlass::detail::TagToStrideC_t<StrideOrLayoutTag>;
  using SmemLayoutAtom =
    decltype(
      detail::sm90_get_epilogue_smem_swizzle_layout_atom<
        Stride, ElementAux, typename EpilogueDescriptor::EpilogueTile
      >()
    );
  using CopyOpS2R =
    decltype(detail::sm90_get_smem_load_op_for_source<Stride, ElementAux, EpilogueTile>());
};

// Get Stride, SmemLayout, and CopyOpS2R for AuxStore node
template<
  typename EpilogueDescriptor,
  typename StrideOrLayoutTag,
  typename ElementAux
>
struct AuxStoreDescriptor {
  constexpr static int Stages = EpilogueDescriptor::StagesD;
  using EpilogueTile = typename EpilogueDescriptor::EpilogueTile;
  using Element = ElementAux;
  using Stride = cutlass::detail::TagToStrideC_t<StrideOrLayoutTag>;
  using SmemLayoutAtom =
    decltype(
      detail::sm90_get_epilogue_smem_swizzle_layout_atom<
        Stride, ElementAux, typename EpilogueDescriptor::EpilogueTile
      >()
    );
  using CopyOpR2S =
    decltype(detail::sm90_get_smem_store_op_for_accumulator<Stride, ElementAux, EpilogueTile>());
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////

// No-smem builder
template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute,RoundStyle>,
    cute::enable_if_t<cute::is_same_v<Schedule, NoSmemWarpSpecialized> ||
                      cute::is_same_v<Schedule, PtrArrayNoSmemWarpSpecialized> ||
                      cute::is_same_v<Schedule, PtrArrayNoSmemWarpSpecializedTransposed> >> {

  // Passing void C disables source load
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
      ElementD, ElementC_>; // prevents cute breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  static constexpr int FragmentSize = 1;
  using ThreadOp = thread::LinearCombination<
    ElementD, FragmentSize, ElementAccumulator, ElementCompute,
    ScaleType, RoundStyle, ElementC>;

  using CollectiveOp = cute::conditional_t<
    cute::is_same_v<Schedule, NoSmemWarpSpecialized>,
    cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
      cutlass::epilogue::collective::DefaultEpilogue<
        ElementC_,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
        ThreadOp,
        cutlass::gemm::EpilogueDefault>>,
    // Epilogue for Ptr-Array and Grouped Gemm
    cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
      cutlass::epilogue::collective::DefaultEpilogueArray<
        ElementC_,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
        ThreadOp,
        Schedule>>
    >;
};

// Tma warp-specialized builder
template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD_,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOperation
>
struct CollectiveBuilder<
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD_,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    FusionOperation,
    cute::enable_if_t<cute::is_same_v<Schedule, TmaWarpSpecialized> ||
                      cute::is_same_v<Schedule, TmaWarpSpecializedCooperative> ||
                      detail::sm90_is_ptr_array_tma_v<Schedule>>> {
private:
  using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                     fusion::get_element_aux_t<FusionOperation>, ElementD_>;
  using EpilogueTile_MN =
    decltype(detail::sm90_compute_tile_shape_or_override<ElementD, EpilogueTileType, Schedule, TileShape_MNK>());
  using DispatchPolicy =
    decltype(detail::sm90_get_tma_dispatch_policy<TileShape_MNK,EpilogueTile_MN,ElementC,ElementD,Schedule>());

public:
  using CollectiveOp =
    typename detail::Sm90TmaBuilderImpl<
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD_,
      GmemLayoutTagD,
      AlignmentD,
      FusionOperation,
      DispatchPolicy
    >::CollectiveOp;
};

// Auto builder
template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOperation
>
struct CollectiveBuilder<
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleAuto,
    FusionOperation,
    void> {
private:
  static_assert(cute::is_same_v<FusionOperation, fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>>,
                "Auto schedule doesn't support fusion. Use one of the TmaWarpSpecialized schedules instead.");

  // Pick No-Smem epilogue as the Auto Epilogue Schedule (Auto schedules do not guarantee best performance) 
  // since TMA epilogues are not compatible with non-TMA non-WS mainloops
  using EpilogueSchedule = NoSmemWarpSpecialized;
  using _CollectiveBuilder = CollectiveBuilder<
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >;

public:
  using CollectiveOp = typename _CollectiveBuilder::CollectiveOp;
};

// DEPRECATED Tma warp-specialized builder for elementwise fusion
template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class UnusedFusionOp
>
struct [[deprecated("Use TmaWarpSpecialized with fusion::LinCombEltAct instead")]]
CollectiveBuilder<
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    UnusedFusionOp,
    cute::enable_if_t<cute::is_base_of_v<TmaWarpSpecializedElementwiseBase, Schedule> ||
                      cute::is_base_of_v<TmaWarpSpecializedCooperativeElementwiseBase, Schedule> >> {
private:
  using FusionOp =
    fusion::LinCombEltAct<Schedule::template ActivationFunctor, ElementD, ElementCompute, ElementC, ElementCompute, Schedule::Round>;
  using ImplSchedule =
    cute::conditional_t<cute::is_base_of_v<TmaWarpSpecializedElementwiseBase, Schedule>,
      TmaWarpSpecialized, TmaWarpSpecializedCooperative>;

public:
  using CollectiveOp =
    typename CollectiveBuilder<
      arch::Sm90,
      OpClass,
      TileShape_MNK,
      ClusterShape_MNK,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD,
      GmemLayoutTagD,
      AlignmentD,
      ImplSchedule,
      FusionOp
    >::CollectiveOp;
};

// DEPRECATED Tma warp-specialized builder for bias + elementwise fusion
template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class UnusedFusionOp
>
struct [[deprecated("Use TmaWarpSpecialized with fusion::LinCombPerRowBiasEltAct or fusion::LinCombPerRowBiasEltActAux instead")]]
CollectiveBuilder<
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    UnusedFusionOp,
    cute::enable_if_t<cute::is_base_of_v<TmaWarpSpecializedBiasElementwiseBase, Schedule> ||
                      cute::is_base_of_v<TmaWarpSpecializedCooperativeBiasElementwiseBase, Schedule> >> {
private:
  using EpilogueTile_MN = decltype(detail::sm90_compute_tile_shape_or_override<
    ElementD, EpilogueTileType, Schedule, TileShape_MNK>());
  // MSVC doesn't seem to be able to deduce DispatchPolicy correctly if it's
  // defined as decltype of a detail::sm90_get_tma_dispatch_policy call.
  // Instead, we paste in the contents of that function.  A natural refactoring
  // would be to create a type alias in the detail namespace.
  using DispatchPolicy = Sm90TmaWarpSpecialized<
    /* StagesC = */ size(shape_div(take<0, 2>(TileShape_MNK{}), EpilogueTile_MN{})),
    /* StagesD = */ 2,
    /* FragmentSize = */ size(EpilogueTile_MN{}) / (detail::sm90_is_cooperative_v<Schedule> ? 256 : 128),
    /* ReuseSmemC = */ sizeof_bits_v<ElementC_> == sizeof_bits_v<ElementD>,
    false
  >;

  using GmemStrideTypeAux = gemm::TagToStrideC_t<GmemLayoutTagD>;
  using SmemLayoutAtomAux = decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<
    GmemStrideTypeAux, typename Schedule::ElementT, EpilogueTile_MN>());
  using SmemCopyOpAux = decltype(detail::sm90_get_smem_store_op_for_accumulator<
    GmemStrideTypeAux, typename Schedule::ElementT, EpilogueTile_MN>());
  using FusionOperationAux = fusion::LinCombPerRowBiasEltActAux<
    GmemLayoutTagD, Schedule::template ActivationFunctor, ElementD, ElementCompute,
    typename Schedule::ElementT, typename Schedule::ElementBias, ElementC_, ElementCompute
  >;
  using FusionCallbacksAux = fusion::FusionCallbacks<
    DispatchPolicy, FusionOperationAux, TileShape_MNK, EpilogueTile_MN, SmemLayoutAtomAux, SmemCopyOpAux
  >;

  using FusionOperationNoAux = fusion::LinCombPerRowBiasEltAct<
    Schedule::template ActivationFunctor, ElementD, ElementCompute,
    typename Schedule::ElementBias, ElementC_, ElementCompute
  >;
  using FusionCallbacksNoAux = fusion::FusionCallbacks<
    DispatchPolicy, FusionOperationNoAux, TileShape_MNK, EpilogueTile_MN
  >;

  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,ElementD,ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,GmemLayoutTagD,GmemLayoutTagC_>;

  using GmemStrideTypeC = gemm::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = gemm::TagToStrideC_t<GmemLayoutTagD>;

  // Get the smallest tiled copy we can use to retile the accumulators
  using CopyAtomC = cute::conditional_t<
    size<1>(EpilogueTile_MN{}) % 16 == 0,
    Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>,
    cute::conditional_t<
      size<1>(EpilogueTile_MN{}) % 8 == 0,
      Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>,
      void
    >
  >;
  static_assert(!cute::is_same_v<CopyAtomC, void>, "CopyAtomC can't be void, divisiblity check for EpilogueTile_MN failed");

  // Get register to register tiled copy that happen before shared memory store.
  // Apply void as no register transform op needed.
  using CopyOpR2R = void;

public:
  using CollectiveOp = cutlass::epilogue::collective::Sm90EpilogueTmaWarpSpecializedBiasElementwise<
      DispatchPolicy::StagesC,
      DispatchPolicy::StagesD,
      DispatchPolicy::FragmentSize,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      cute::conditional_t<Schedule::StoreT, FusionCallbacksAux, FusionCallbacksNoAux>,
      SM90_TMA_LOAD,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<GmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_load_op_for_source<GmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      SM90_TMA_STORE,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<GmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_store_op_for_accumulator<GmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      CopyAtomC,
      CopyOpR2R
    >;
};

// CollectiveBuilder that transposed epilogue below is used for sm90 gmma RS TT kernels
// since swapping NNN kernels input matrix and transposing its output at the same time then
// we can get TTN kernel.
template <
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOperation
>
struct CollectiveBuilder<
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    cutlass::gemm::EpilogueTransposed,
    FusionOperation,
    void> {
private:
  static_assert(cute::is_same_v<FusionOperation, fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute>>,
                "EpilogueTransposed schedule doesn't support fusion.");
  // Passing void C disables source load
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
      ElementD, ElementC_>; // prevents cute breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  static constexpr int FragmentSize = 1;
  using ThreadOp = thread::LinearCombination<
    ElementD, FragmentSize, ElementAccumulator, ElementCompute,
    ScaleType, cutlass::FloatRoundStyle::round_to_nearest, ElementC>;

public:
  using CollectiveOp = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::DefaultEpilogue<
      ElementC_,
      cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
      cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
      ThreadOp,
      cutlass::gemm::EpilogueTransposed>
    >;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective
