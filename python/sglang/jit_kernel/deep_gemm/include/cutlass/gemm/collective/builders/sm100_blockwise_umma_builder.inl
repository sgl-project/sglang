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

#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/gemm/collective/builders/sm100_pipeline_carveout.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class ElementScalar,
  class ScaleShapeMNK,
  class TileShapeMNK,
  class MainloopABPipelineStorage,
  class MainloopSFPipelineStorage,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_blockwise(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class ElementScalar,
  class ScaleShapeMNK,
  class TileShapeMNK,
  class MainloopABPipelineStorage,
  class MainloopSFPipelineStorage,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_blockwise(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class ElementScalar,
  class ScaleShapeMNK,
  class TileShapeMNK,
  class MainloopABPipelineStorage,
  class MainloopSFPipelineStorage,
  int carveout_bytes>
constexpr int
sm100_compute_stage_count_or_override_blockwise(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For F8/F6/F4 sub-bytes, ElementA/B will be passed in as uint8_t
  // For Planar Complex, ElementA/B will be passed in as cutlass::complex<ElementARaw>
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A and smem for B (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one of each of the pipelines
  constexpr auto pipeline_bytes = sizeof(MainloopABPipelineStorage) + 
      sizeof(MainloopSFPipelineStorage);

  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto scale_bits = cute::sizeof_bits_v<ElementScalar>;

  constexpr int stage_bytes =
    cutlass::round_nearest(
      cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      cutlass::bits_to_bytes(scale_bits * size<0>(ScaleShapeMNK{}) * size<2>(ScaleShapeMNK{})) +
      cutlass::bits_to_bytes(scale_bits * size<1>(ScaleShapeMNK{}) * size<2>(ScaleShapeMNK{})),
      128) +
    static_cast<int>(pipeline_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

template<class Element, typename LayoutSFA, class CtaShape_MNK>
auto sm100_make_simt_gmem_tiled_copy_SFA() {

  // we have at most a warp to perform the loads

  constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
  constexpr int ScaleMsPerTile = size<0>(CtaShape_MNK{}) / ScaleGranularityM;
  constexpr int ScaleGranularityK = size<1,0>(LayoutSFA{});
  constexpr int ScaleKsPerTile = size<2>(CtaShape_MNK{}) / ScaleGranularityK;

  if constexpr (size<0,1>(LayoutSFA{}.stride()) == 1) {
    constexpr int LeadingScalesPerTileSFA = ScaleMsPerTile;
    if constexpr (LeadingScalesPerTileSFA >= 32) {
      constexpr int Alignment = cute::min(static_cast<int>(LeadingScalesPerTileSFA * sizeof(Element)) / 32, 16);
      using ScaleCopyTypeA = cute::uint_byte_t<Alignment>; 
      using SmemScalingCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ScaleCopyTypeA>, Element>;
      constexpr int ElementsPerSFACopy = static_cast<int>(sizeof(ScaleCopyTypeA) / sizeof(Element));
      return make_tiled_copy(
          SmemScalingCopyAtomA{}, 
          Layout<Shape<_32>>{},  // 32 threads
          Layout<Shape<Int<ElementsPerSFACopy>>>{});
    } 
    else {
      using SmemScalingCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<Element>, Element>;
      return make_tiled_copy(SmemScalingCopyAtomA{}, Layout<Shape<Int<LeadingScalesPerTileSFA>>>{}, Layout<Shape<_1>>{});
    }
  } 
  else {
    // we expect scale Ks per tile to be small
    using SmemScalingCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<Element>, Element>;
    return make_tiled_copy(SmemScalingCopyAtomA{}, Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
  }
}

template<class Element, typename LayoutSFB, class CtaShape_MNK>
auto sm100_make_simt_gmem_tiled_copy_SFB() {

  // we have at most a warp to perform the loads

  constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
  constexpr int ScaleNsPerTile = size<1>(CtaShape_MNK{}) / ScaleGranularityN;
  constexpr int ScaleGranularityK = size<1,0>(LayoutSFB{});
  constexpr int ScaleKsPerTile = size<2>(CtaShape_MNK{}) / ScaleGranularityK;

  if constexpr (size<0,1>(LayoutSFB{}.stride()) == 1) {
    constexpr int LeadingScalesPerTileSFB = ScaleNsPerTile;
    if constexpr (LeadingScalesPerTileSFB >= 32) {
      constexpr int Alignment = cute::min(static_cast<int>(LeadingScalesPerTileSFB * sizeof(Element)) / 32, 16);
      using ScaleCopyTypeB = cute::uint_byte_t<Alignment>; 
      using SmemScalingCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ScaleCopyTypeB>, Element>;
      constexpr int ElementsPerSFBCopy = static_cast<int>(sizeof(ScaleCopyTypeB) / sizeof(Element));
      return make_tiled_copy(
          SmemScalingCopyAtomB{}, 
          Layout<Shape<_32>>{},  // 32 threads
          Layout<Shape<Int<ElementsPerSFBCopy>>>{});
    } 
    else {
      using SmemScalingCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<Element>, Element>;
      return make_tiled_copy(SmemScalingCopyAtomB{}, Layout<Shape<Int<LeadingScalesPerTileSFB>>>{}, Layout<Shape<_1>>{});
    }
  } 
  else {
    // we expect scale Ks per tile to be small
    using SmemScalingCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<Element>, Element>;
    return make_tiled_copy(SmemScalingCopyAtomB{}, Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
  }
}

// For new MMA construction and partitioning that supports both dynamic and static cluster shape.
// Used in conjunction with make_tma_atom_(A|B)_sm100
// TileShape_MNK is always static and has shape (MmaAtomShapeM, MmaAtomShapeN, TileK)
// ClusterShape_MNK can be dynamic or static.
template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  class BuilderScheduleTag,
  UMMA::ScaleIn ANeg = UMMA::ScaleIn::One,
  UMMA::ScaleIn BNeg = UMMA::ScaleIn::One
>
constexpr auto
sm100_make_trivial_tiled_mma_blockwise() {
  // MMA_2SM requested
  if constexpr (cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> ) {
    return sm100_make_2sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                    TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
  }
  // MMA_1SM requested
  else if constexpr (cute::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag> ) {
    return sm100_make_1sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                    TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
  }
  // Auto scheduling requested
  else if constexpr (cute::is_same_v<BuilderScheduleTag, KernelScheduleSm100Blockwise>) {
    // Static cluster
    if constexpr (cute::is_static_v<ClusterShape_MNK>) {
      // For MMA_2SM we need a cluster shape that is multiple of 2x1
      // and only M=128 and M=256 are supported, otherwise, fall back to MMA_1SM
      if constexpr (cute::size<0>(ClusterShape_MNK{}) % 2 == 0 &&
                    cute::size<0>(TileShape_MNK{}) % 128 == 0) {
        return sm100_make_2sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                        TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
      }
      else {
        return sm100_make_1sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                        TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
      }
    // Dynamic cluster shape means we cannot assume we can use 2SM MMA 
    }
    else {
        return sm100_make_1sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                        TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
    }
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementA,
  class GmemLayoutATagPair,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTagPair,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATagPair,
    AlignmentA,
    ElementB,
    GmemLayoutBTagPair,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      not cute::is_tuple_v<ElementA>   && not cute::is_tuple_v<ElementB> &&
      not cute::is_complex_v<ElementA> && not cute::is_complex_v<ElementB> &&
      cute::is_tuple_v<GmemLayoutATagPair>   && cute::is_tuple_v<GmemLayoutBTagPair> &&
      // Dense Gemm
      cute::is_base_of_v<KernelScheduleSm100Blockwise, BuilderScheduleTag> &&
      // Alignment check
      detail::sm1xx_gemm_is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, BuilderScheduleTag>()>>
{
  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(detail::check_input_datatypes<ElementA, ElementB>(), "Incorrect input types");

  using GmemLayoutATag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutATagPair{}))>;
  using GmemLayoutSFATag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutATagPair{}))>;
  using GmemLayoutBTag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutBTagPair{}))>;
  using GmemLayoutSFBTag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutBTagPair{}))>;

  static_assert(cute::depth(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 2 and 
                cute::depth(cute::remove_pointer_t<GmemLayoutSFBTag>{}) == 2, 
      "Expect SFA and SFB layout to be depth of two with shape ((SFVecMN, restMN),(SFVecK, restK), L)");
  static_assert(size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 
                size<1,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{}), 
      "SFA and SFB must have equivalent SF vector sizes along K");

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  // Data type used by MMA instruction
  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB>());

  static constexpr bool is_2sm = cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> ||
                        (not cute::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag> &&
                          not cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> &&
                          cute::is_static_v<ClusterShape_MNK> &&
                          cute::get<0>(ClusterShape_MNK{}) % 2 == 0 );

  static_assert(detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMma, ElementBMma,
                                                                      TileShape_MNK, ClusterShape_MNK,
                                                                      GmemLayoutATag, GmemLayoutBTag, false /*is_sparse*/, is_2sm>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );
  using TiledMma =  decltype(detail::sm100_make_trivial_tiled_mma_blockwise<
      ElementAMma, ElementBMma, ElementAccumulator,
      decltype(cute::product_each(TileShape_MNK{})), ClusterShape_MNK,
      UmmaMajorA, UmmaMajorB, BuilderScheduleTag>());

  using ElementAMma_SmemAllocType = cute::conditional_t<cute::sizeof_bits_v<ElementAMma> < 8, uint8_t, ElementAMma>;
  using ElementBMma_SmemAllocType = cute::conditional_t<cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  using AtomThrID = typename TiledMma::AtomThrID;

  using AtomThrShapeMNK = cute::Shape<decltype(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  using CtaTileShape_MNK = decltype(cute::shape_div(TileShape_MNK{}, AtomThrShapeMNK{}));

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));

  static_assert(BlockTileA_K{} == BlockTileB_K{}, "Block tile Ks should be equal");

  using SmemShape_M = decltype(shape_div(shape<0>(TileShape_MNK{}), shape_div(shape<0>(TileShape_MNK{}), size<0>(TileShape_MNK{}) / size(AtomThrID{}))));
  using SmemShape_N = decltype(shape_div(shape<1>(TileShape_MNK{}), shape_div(shape<1>(TileShape_MNK{}), size<1>(TileShape_MNK{}) / size(AtomThrID{}))));
  using SmemShape_K = decltype(cute::get<2>(TileShape_MNK{}));

  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
    ClusterShape_MNK{}, AtomThrID{}));
  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(
      ClusterShape_MNK{}, AtomThrID{}));

  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma_SmemAllocType, SmemShape_M, SmemShape_K>());
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma_SmemAllocType, SmemShape_N, SmemShape_K>());
  static constexpr uint32_t TotalTmemRows = 128;
  static constexpr uint32_t Sm100TmemCapacityColumns = 512;
  static constexpr uint32_t TotalTmem = TotalTmemRows * Sm100TmemCapacityColumns;
  static constexpr uint32_t AccumulatorPipelineStageCount = (is_2sm || (!is_2sm && size(shape<0,0>(MmaShapeA_MK{}) > 64))) ? 
                                                              TotalTmem / (cute::size<0>(CtaTileShape_MNK{}) * cute::size<1>(CtaTileShape_MNK{}))
                                                            : (Sm100TmemCapacityColumns / cute::size<1>(CtaTileShape_MNK{})) * 2;                       // 1SM MMA_M = 64 case
  static_assert(AccumulatorPipelineStageCount > 0, "Accumulator pipeline stage count must be positive.  This error probably means that TileShape_MNK and/or TiledMma::ThrLayoutVMNK are wrong.");

  // Calculate scheduler pipeline stages. Having one more stage than the accumulator allows more latency hiding.
  using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayoutATag>;
  using InternalStrideA  = cute::remove_pointer_t<StrideA>;
  // Grouped GEMM (where Stride type is Stride*) does not use CLC based scheduler.
  // SchedulerPipelineStageCount could be set to zero for Grouped GEMM, but we shouldn't define CLC Pipeline's barrier arrays of size zero.
  static constexpr uint32_t SchedulerPipelineStageCount = cute::is_same_v<InternalStrideA, StrideA> ? (AccumulatorPipelineStageCount + 1) : 1;

  static constexpr bool IsArrayOfPointersGemm = (cute::is_base_of_v<KernelScheduleSm100PtrArrayBlockwise, BuilderScheduleTag>);

  static constexpr uint32_t KernelSmemCarveout = detail::Sm100DenseGemmTmaUmmaCarveout<
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount,
      detail::CLCResponseSize,
      IsArrayOfPointersGemm
    >::KernelSmemCarveout;
  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;
  using MainloopABPipelineStorage = typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage;
  using MainloopSFPipelineStorage = typename cutlass::PipelineAsync<1>::SharedStorage;

  static constexpr int ScaleGranularityM = size<0,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});
  static constexpr int ScaleGranularityN = size<0,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{});
  static constexpr int ScaleGranularityK = size<1,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{});

  static_assert(size<0>(CtaTileShape_MNK{}) >= ScaleGranularityM, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<1>(CtaTileShape_MNK{}) >= ScaleGranularityN, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<2>(CtaTileShape_MNK{}) >= ScaleGranularityK, "Scale Granularity must be smaller than or equal to the tile shape");

  using GmemTiledCopySFA = decltype(detail::sm100_make_simt_gmem_tiled_copy_SFA<
      ElementAccumulator,
      cute::remove_pointer_t<GmemLayoutSFATag>,
      CtaTileShape_MNK>());
  using GmemTiledCopySFB = decltype(detail::sm100_make_simt_gmem_tiled_copy_SFB<
      ElementAccumulator,
      cute::remove_pointer_t<GmemLayoutSFBTag>,
      CtaTileShape_MNK>());

  using BlockTileScale_M = Int<size<0>(TileShape_MNK{}) / ScaleGranularityM>;
  using BlockTileScale_N = Int<size<1>(TileShape_MNK{}) / ScaleGranularityN>;
  using BlockTileScale_K = Int<size<2>(TileShape_MNK{}) / ScaleGranularityK>;

  using ScaleTileShape = cute::Shape<BlockTileScale_M, BlockTileScale_N, BlockTileScale_K>;

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_blockwise<
      Sm100ReducedSmemCapacityBytes, ElementAMma_SmemAllocType, ElementBMma_SmemAllocType, 
      ElementAccumulator, ScaleTileShape, SmemTileShape, MainloopABPipelineStorage,
      MainloopSFPipelineStorage>(StageCountType{});
  static_assert(PipelineStages > 0, "Smem usage is too high. Can't create any SMEM buffers for A, B, and scales.");

  using DispatchPolicy = cute::conditional_t<
    IsArrayOfPointersGemm,
    cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling<
      PipelineStages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape_MNK>,
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling<
      PipelineStages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape_MNK>>;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      cute::tuple<cutlass::gemm::TagToStrideA_t<GmemLayoutATag>, cutlass::gemm::TagToStrideA_t<GmemLayoutSFATag>>,
      ElementB,
      cute::tuple<cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>, cutlass::gemm::TagToStrideB_t<GmemLayoutSFBTag>>,
      TiledMma,
      cute::tuple<GmemTiledCopyA, GmemTiledCopySFA>,
      SmemLayoutAtomA,
      void,
      cute::identity,
      cute::tuple<GmemTiledCopyB, GmemTiledCopySFB>,
      SmemLayoutAtomB,
      void,
      cute::identity
    >;
};


} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
