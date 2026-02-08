/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/detail/collective.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/collective/builders/sm120_common.inl"
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

// Helper structs for getting the SF vector size used by the epilogue, if one is used
template <class FusionOp, class = void>
struct EpilogueSFVecSize {
  static constexpr int value = 0;
};

template <class FusionOp>
struct EpilogueSFVecSize<FusionOp, cute::void_t<decltype(FusionOp::SFVecSize)>> {
  static constexpr int value = FusionOp::SFVecSize;
};

// Helper to deduce NumEpilogueWarpGroups based on Schedule
template <class Schedule, class = void>
struct GetNumEpilogueWarpGroups {
  static constexpr int value = 2;
};

template <class Schedule>
struct GetNumEpilogueWarpGroups<Schedule, cute::void_t<decltype(Schedule::NumEpilogueWarpGroups)>> {
  static constexpr int value = Schedule::NumEpilogueWarpGroups;
};

// Returns the parameterized dispatch policy for the TMA epilogue
template<class TileShapeMNK, class EpilogueTileMN, class ElementC, class ElementD, class GmemLayoutTagD, class Schedule>
constexpr auto
sm120_get_tma_dispatch_policy() {
  using namespace cute;

  constexpr int EpiTiles = size(shape_div(take<0,2>(TileShapeMNK{}), EpilogueTileMN{}));
  using StrideD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;
  using InternalStrideD  = cute::remove_pointer_t<StrideD>;
  constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideD, StrideD>;

  // For 120, a FragmentSize of 4 is used to match the
  // output per thread from each MMA. Epilogue subtiles iterate over multiple of these
  // fragments before storing the subtile's outputs to shared memory.
  constexpr int FragmentSize = 4;

  // 8b residuals load fast and consume little smem, so the perf cost of waiting on stores to finish outweighs the cost of extra allocation
  constexpr bool ReuseSmem = (sizeof_bits_v<ElementC> == sizeof_bits_v<ElementD>) && (sizeof_bits_v<ElementD> > 8);
  constexpr bool DelayTmaStore = is_void_v<ElementC>; // TMA store delay performs worse with residual loads

  constexpr bool IsFP6 = cute::is_same_v<ElementD, cutlass::float_e3m2_t> || cute::is_same_v<ElementD, cutlass::float_e2m3_t>;
  constexpr bool IsRowMajorD = cutlass::gemm::detail::is_major<1, StrideD>();
  constexpr int StagesD = (IsFP6 && IsRowMajorD) ? 1 : cute::min(EpiTiles, 2);

  // SM120 epilogues use smaller stage counts in order to fit within the limited shared memory capacity.
  constexpr int StagesC = ReuseSmem ? cute::max(cute::min(EpiTiles, 2), StagesD+1)
                                    : StagesD;  

  constexpr int NumEpilogueWarpGroups = GetNumEpilogueWarpGroups<Schedule>::value;

  if constexpr (IsGroupedGemmKernel) {
    return Sm120PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, 
                                          DelayTmaStore, NumEpilogueWarpGroups>{};
  } 
  else {
    return Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>{};
  }
}

// Returns the smem layout atom to be used for C or D matrix
template<class GmemStrideType, class Element_, class EpilogueTile_MN>
constexpr auto
sm120_get_epilogue_smem_swizzle_layout_atom() {
  using namespace cute;

  // FP6 data is always stored in 8-bit containers in the epilogue
  using Element = cute::conditional_t<
    cute::is_same_v<Element_, cutlass::float_e3m2_t> || cute::is_same_v<Element_, cutlass::float_e2m3_t>,
    uint8_t,  Element_
  >;

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

template <class ElementC, class ElementD, class EpilogueTileType, class Schedule, class TileShape_MNK, class StrideD, class FusionOp>
constexpr auto
sm120_compute_tile_shape_or_override() {

  constexpr int CTA_M = size<0>(TileShape_MNK{});
  constexpr int CTA_N = size<1>(TileShape_MNK{});

  constexpr bool IsFP6 = cute::is_same_v<ElementD, cutlass::float_e3m2_t> || cute::is_same_v<ElementD, cutlass::float_e2m3_t>;
  
  constexpr bool IsColMajorD = cutlass::gemm::detail::is_major<0, StrideD>();
  constexpr bool IsRowMajorD = cutlass::gemm::detail::is_major<1, StrideD>();
  static_assert(IsColMajorD || IsRowMajorD, "SM120 LayoutD must be either row or column major.");
  
  static_assert(!IsFP6 ||
              (CTA_M % 128 == 0 && IsColMajorD) ||
              (CTA_N % 128 == 0 && IsRowMajorD),
              "CTA tile for FP6 ElementD must have a contiguous extent that is a multiple of 128.");

  if constexpr (cute::is_same_v<EpilogueTileType, EpilogueTileAuto>) {
    // If ElementD is FP6, use an epilogue subtile with an extent
    // of 128 along the continuous dimension to meet TMA requirements.
    if constexpr (IsFP6) {
      if constexpr (IsRowMajorD) {
        return Shape<_64, _128>{};
      }
      else {
        return Shape<_128, _32>{};
      }
    }
    else {
      if constexpr (cute::is_same_v<Schedule, SparseTmaWarpSpecializedCooperativeSm120>) {
        // sm120 sparse kernels require more shared memory budget than dense kernels in the mainloop 
        // so selecting a smaller EpilogueTileN (16) for some cases.
        if constexpr (FusionOp::SFVecSize == 64 && IsRowMajorD) {
          return Shape<_32, _64>{};
        }
        else {
          constexpr int M = 64;
          constexpr int N = cute::is_void_v<ElementC>
            // When C is void, let N = 16 when D is fp32 for lesser SMEM consumption, otherwise 32.
            ? cute::sizeof_bits_v<ElementD> == 32 ? 16 : 32 
            // When C is not void
            : cute::sizeof_bits_v<ElementC> <= 16
              ? 32 // 16-bit or smaller C needs lesser SMEM for epilogue so we keep N = 32
              : 16; // 32-bit needs to let N = 16
          return Shape<Int<M>, Int<N>>{};
        }
      }
      else {
        return Shape<_64, _32>{};
      }
    }
  } // EpilogueTileAuto
  else if constexpr (cute::is_tuple<EpilogueTileType>::value) {
    static_assert(!is_layout<EpilogueTileType>::value, "EpilogueTile must be a cute::Tile or cute::Shape");

    EpilogueTileType epi_tile;
    constexpr int M = size<0>(shape(epi_tile));
    constexpr int N = size<1>(shape(epi_tile));

    static_assert(!IsFP6 ||
                  (M % 128 == 0 && IsColMajorD) ||
                  (N % 128 == 0 && IsRowMajorD),
                  "EpilogueTile for narrow ElementD must have a contiguous extent that is a multiple of 128.");

    static_assert(CTA_M % M == 0 && CTA_N % N == 0, "EpilogueTile must evenly divide CTA tile");

    return epi_tile;
  }
  else {
    static_assert(cutlass::detail::dependent_false<EpilogueTileType>, "Invalid type for EpilogueTileType.");
  }
}

template <class GmemStrideTypeD, class ElementD>
constexpr auto
sm120_get_register_transform_op() {
  using namespace cute;

  [[maybe_unused]] constexpr bool is_m_major = cutlass::detail::is_major<0>(GmemStrideTypeD{});
  [[maybe_unused]] constexpr bool is_n_major = cutlass::detail::is_major<1>(GmemStrideTypeD{});
  static_assert(is_m_major || is_n_major, "Unsupported gmem layout");

  if constexpr (sizeof_bits_v<ElementD> == 4 && is_m_major) {
    // Before store fp4 along M major, row0 column{0,1} is kept in one thread, and row1 column{0,1}
    // is kept in another thread. It is expected to have row{0,1} column0 in one thread,
    // while row{0,1} column1 in another thread, so that the store could keep granularity
    // 8bits at least. The shuffle is a 2x2 transpose, like below diagram, switching N major to
    // M major from a register view.
    //
    // Before                            After
    //         Column0   Column1                 Column0   Column1
    //  Row0    d0(t0)   d1(t0)           Row0    d0(t0)    d0(t4)
    //  Row1    d0(t4)   d1(t4)           Row1    d1(t0)    d1(t4)
    //
    return SM50_Shuffle_U32_2x2Trans_XOR4{};
  }
  else {
    return; // void
  }
}

// Overload CallbacksBuilder to pick the correct copy atoms
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
  Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  FusionOp,
  TileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && not cute::is_subbyte_v<typename FusionOp::ElementAux>>
> {
  using GmemStrideTypeAux = gemm::TagToStrideC_t<typename FusionOp::GmemLayoutTagAux>;
  using SmemLayoutAtomAux = decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());

  using CopyOpR2S = decltype(detail::sm120_get_smem_store_op_for_accumulator<GmemStrideTypeAux, typename FusionOp::ElementAux>());

  using CopyOpS2R = decltype(detail::sm120_get_smem_load_op_for_source<GmemStrideTypeAux, typename FusionOp::ElementAux>());
  
  using SmemCopyOpAux = cute::conditional_t<FusionOp::IsAuxOutSupported, CopyOpR2S, CopyOpS2R>;

  using Callbacks = fusion::FusionCallbacks<
    Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOp, TileShape_MNK, EpilogueTile_MN,
    SmemLayoutAtomAux, SmemCopyOpAux
  >;
};

// Overload CallbacksBuilder to pick the correct copy atoms for PtrArray epilogue fusions
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpgroups,
  class FusionOp,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class AccLoadOp,
  class ElementAccumulator
>
struct CallbacksBuilder<
  Sm120PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumEpilogueWarpgroups>,
  FusionOp,
  TileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && not cute::is_subbyte_v<typename FusionOp::ElementAux>>
> {
  using GmemStrideTypeAux = gemm::TagToStrideC_t<typename FusionOp::GmemLayoutTagAux>;
  using SmemLayoutAtomAux = decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());

  using CopyOpR2S = decltype(detail::sm120_get_smem_store_op_for_accumulator<GmemStrideTypeAux, typename FusionOp::ElementAux>());

  using CopyOpS2R = decltype(detail::sm120_get_smem_load_op_for_source<GmemStrideTypeAux, typename FusionOp::ElementAux>());
  
  using SmemCopyOpAux = cute::conditional_t<FusionOp::IsAuxOutSupported, CopyOpR2S, CopyOpS2R>;

  using Callbacks = fusion::FusionCallbacks<
    Sm120PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumEpilogueWarpgroups>,
    FusionOp, TileShape_MNK, EpilogueTile_MN,
    SmemLayoutAtomAux, SmemCopyOpAux
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
struct Sm120TmaBuilderImpl {
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

  using CopyOpS2G =
    cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagD>,
      SM90_TMA_STORE_IM2COL,
      SM90_TMA_STORE
    >;

  using CopyOpG2S =
    cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagC>,
      SM90_TMA_LOAD_IM2COL,
      SM90_TMA_LOAD
    >;

  // Get the smallest tiled copy we can use to retile the accumulators
  using CopyAtomC = Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>;

  using SmemLayoutAtomC = decltype(detail::sm120_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>());
  using SmemLayoutAtomD = decltype(detail::sm120_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>());

  using CopyOpS2R = decltype(detail::sm120_get_smem_load_op_for_source<UnderlyingGmemStrideTypeC, ElementC>());

  using CopyOpR2S = decltype(detail::sm120_get_smem_store_op_for_accumulator<UnderlyingGmemStrideTypeD, ElementD>());

  // Get register to register tiled copy that happen before shared memory store.
  using CopyOpR2R = decltype(detail::sm120_get_register_transform_op<UnderlyingGmemStrideTypeD, ElementD>());

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

  // Re-use Sm90 collective epilogue implementation
  constexpr static int StagesC = DispatchPolicy::StagesC;
  constexpr static int  StagesD = DispatchPolicy::StagesD;
  constexpr static int  FragmentSize = DispatchPolicy::FragmentSize;
  constexpr static bool ReuseSmemC = DispatchPolicy::ReuseSmemC;
  constexpr static bool DelayTmaStore = DispatchPolicy::DelayTmaStore;

  //Helper to deduce BaseDispatchPolicy based on DispatchPolicy
  template<class T>
  struct GetBaseDispatchPolicy {
    using Type = T;
  };

  template<int StagesC_, int StagesD_, int FragmentSize_, bool ReuseSmemC_, 
           bool DelayTmaStore_, int NumEpilogueWarpGroups_>
  struct GetBaseDispatchPolicy<Sm120PtrArrayTmaWarpSpecialized<StagesC_, StagesD_, 
    FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumEpilogueWarpGroups_>> {
    using Type = typename cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC_, StagesD_, 
      FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumEpilogueWarpGroups_>;
  };

  template<int StagesC_, int StagesD_, int FragmentSize_, bool ReuseSmemC_, 
           bool DelayTmaStore_>
  struct GetBaseDispatchPolicy<Sm120TmaWarpSpecialized<StagesC_, StagesD_, 
    FragmentSize_, ReuseSmemC_, DelayTmaStore_>> {
    using Type = typename cutlass::epilogue::Sm90TmaWarpSpecialized<StagesC_, StagesD_, 
      FragmentSize_, ReuseSmemC_, DelayTmaStore_>;
  };

  using BaseDispatchPolicy = typename GetBaseDispatchPolicy<DispatchPolicy>::Type;
  
  using CollectiveOp = cutlass::epilogue::collective::CollectiveEpilogue<
      BaseDispatchPolicy,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD_,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      SmemLayoutAtomC,
      CopyOpS2R,
      CopyOpS2G,
      SmemLayoutAtomD,
      CopyOpR2S,
      CopyAtomC,
      CopyOpR2R
    >;
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////

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
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOperation
>
struct CollectiveBuilder<
    arch::Sm120,
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
    FusionOperation,
    cute::enable_if_t<cute::is_same_v<Schedule, EpilogueScheduleAuto> ||
                      cute::is_same_v<Schedule, TmaWarpSpecialized> ||
                      cute::is_same_v<Schedule, TmaWarpSpecializedCooperative> ||
                      cute::is_same_v<Schedule, PtrArrayTmaWarpSpecializedPingpong> ||
                      cute::is_same_v<Schedule, PtrArrayTmaWarpSpecializedCooperative> ||
                      cute::is_same_v<Schedule, SparseTmaWarpSpecializedCooperativeSm120>
                     >> {
private:
  using EpilogueTile_MN =
    decltype(detail::sm120_compute_tile_shape_or_override<ElementC, ElementD, EpilogueTileType, Schedule, TileShape_MNK, cutlass::detail::TagToStrideC_t<GmemLayoutTagD>, FusionOperation>());
  using DispatchPolicy =
    decltype(detail::sm120_get_tma_dispatch_policy<TileShape_MNK,EpilogueTile_MN,ElementC,ElementD, GmemLayoutTagD, Schedule>());


public:
  using CollectiveOp =
    typename detail::Sm120TmaBuilderImpl<
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD,
      GmemLayoutTagD,
      AlignmentD,
      FusionOperation,
      DispatchPolicy
    >::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective
