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

#include "cutlass/gemm/collective/builders/sm120_common.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class ElementScalar,
  class TileShapeMNK,
  class ScaleShapeMNK,
  class MainloopPipelineStorage,
  int stages
>
constexpr int
sm120_compute_stage_count_or_override_blockwise(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity.
template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class ElementScalar,
  class TileShapeMNK,
  class ScaleShapeMNK,
  class MainloopPipelineStorage,
  int carveout_bytes
>
constexpr auto
sm120_compute_stage_count_or_override_blockwise(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For F6/F4 sub-bytes, ElementA/B will be passed in as uint8_t

  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto scale_bits = cute::sizeof_bits_v<ElementScalar>;
  constexpr auto mainloop_pipeline_bytes = sizeof(MainloopPipelineStorage);

  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(scale_bits * size<0>(ScaleShapeMNK{}) * size<2>(ScaleShapeMNK{})) +
    cutlass::bits_to_bytes(scale_bits * size<1>(ScaleShapeMNK{}) * size<2>(ScaleShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes);


  return (CapacityBytes - carveout_bytes) / stage_bytes;
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
    arch::Sm120,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATagPair,
    AlignmentA,
    ElementB,
    GmemLayoutBTagPair,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      not cute::is_tuple_v<ElementA> && not cute::is_tuple_v<ElementB> && 
      not cute::is_complex_v<ElementA> && not cute::is_complex_v<ElementB> &&
      cute::is_tuple_v<GmemLayoutATagPair> && cute::is_tuple_v<GmemLayoutBTagPair> &&
      (cute::is_base_of_v<KernelScheduleSm120Blockwise, BuilderScheduleTag> ||
       cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>) &&
      detail::sm1xx_gemm_is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, BuilderScheduleTag>()>>
{

  static_assert(detail::is_sm10x_f8f6f4_element<ElementA>() && detail::is_sm10x_f8f6f4_element<ElementB>(),
                "SM120 TmaWarpSpecialized blockwise scaling builder currently only supports F8F6F4 MMA.");
  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(cute::is_static_v<ClusterShape_MNK>, "Cluster has to be static");

  using GmemLayoutATag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutATagPair{}))>;
  using GmemLayoutSFATag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutATagPair{}))>;
  using GmemLayoutBTag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutBTagPair{}))>;
  using GmemLayoutSFBTag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutBTagPair{}))>;

  static_assert(cute::depth(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 2 and 
                cute::depth(cute::remove_pointer_t<GmemLayoutSFBTag>{}) == 2, 
      "Expect SFA and SFB layout to be depth of two with shape ((SFVecMN, restMN),(SFVecK, restK), L)");
  static_assert(size<1, 0>(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 
                size<1, 0>(cute::remove_pointer_t<GmemLayoutSFBTag>{}), 
      "SFA and SFB must have equivalent SF vector sizes along K");

  static constexpr cute::UMMA::Major UmmaMajorA = detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = detail::tag_to_umma_major_B<GmemLayoutBTag>();
  static_assert((UmmaMajorA == UMMA::Major::K && UmmaMajorB == UMMA::Major::K), "Only TN layout is supported.");

  using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
  using PermTileN = decltype(cute::min(size<1>(TileShape_MNK{}),  _32{}));

  static constexpr bool IsCooperative = !cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag>;
  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_4,_2,_1>>, Layout<Shape<_2,_2,_1>>>;

  // Data type used by MMA instruction
  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB>());

  static_assert(detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMma, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, false /*IsSparse*/>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  // Setup TiledMma
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::rr_op_selector_sm120<ElementA, ElementB, ElementAccumulator>(),
    AtomLayoutMNK{},
    Tile<PermTileM, PermTileN, _32>{}
  ));

  // DType check
  static constexpr bool UseF8f6f4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();
  static_assert(UseF8f6f4, "Non-blockscaled collective builder only supports F8F6F4 MMA.\n");

  // Element type
  using SmemAllocTypeA = cute::conditional_t<UseF8f6f4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<UseF8f6f4, uint8_t, typename TiledMma::ValTypeB>;

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::sm120_rr_smem_selector<SmemAllocTypeA, decltype(size<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::sm120_rr_smem_selector<SmemAllocTypeB, decltype(size<2>(TileShape_MNK{}))>());

  using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayoutATag>;
  using StrideB = cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>;
  using StrideSFA = cutlass::gemm::TagToStrideA_t<GmemLayoutSFATag>;
  using StrideSFB = cutlass::gemm::TagToStrideB_t<GmemLayoutSFBTag>;

  static constexpr int ScaleGranularityM = size<0,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});
  static constexpr int ScaleGranularityN = size<0,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{});
  static constexpr int ScaleGranularityK = size<1,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{});

  static_assert(size<0>(TileShape_MNK{}) % ScaleGranularityM == 0, "Scale Granularity M must evenly divide the tile shape M.");
  static_assert(size<1>(TileShape_MNK{}) % ScaleGranularityN == 0, "Scale Granularity N must evenly divide the tile shape N.");
  static_assert(size<2>(TileShape_MNK{}) == ScaleGranularityK    , "Scale Granularity K must be equal to the tile shape K.");

  using BlockTileScale_M = Int<size<0>(TileShape_MNK{}) / ScaleGranularityM>;
  using BlockTileScale_N = Int<size<1>(TileShape_MNK{}) / ScaleGranularityN>;
  using BlockTileScale_K = Int<size<2>(TileShape_MNK{}) / ScaleGranularityK>;

  using ScaleTileShape = cute::Shape<BlockTileScale_M, BlockTileScale_N, BlockTileScale_K>;


  // Setup Stages and DispatchPolicy
  using MainloopPipelineStorage = typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage;

  static constexpr int PipelineStages = detail::sm120_compute_stage_count_or_override_blockwise<
      detail::sm120_smem_capacity_bytes, SmemAllocTypeA,
      SmemAllocTypeB, ElementAccumulator, 
      TileShape_MNK, ScaleTileShape, MainloopPipelineStorage>(StageCountType{});
  static constexpr uint32_t SchedulerPipelineStageCount = 2;
  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<cute::remove_pointer_t<StrideA>, StrideA>;
  using KernelSchedule = cute::conditional_t<IsGroupedGemmKernel,
                                              // PtrArray
                                              cute::conditional_t<IsCooperative,
                                                KernelPtrArrayTmaWarpSpecializedCooperativeBlockwiseScalingSm120<SchedulerPipelineStageCount>,
                                                KernelPtrArrayTmaWarpSpecializedPingpongBlockwiseScalingSm120<SchedulerPipelineStageCount>>,
                                              // Non-PtrArray
                                              cute::conditional_t<IsCooperative,
                                                KernelTmaWarpSpecializedCooperativeBlockwiseScalingSm120<SchedulerPipelineStageCount>,
                                                KernelTmaWarpSpecializedPingpongBlockwiseScalingSm120<SchedulerPipelineStageCount>>>;

  using DispatchPolicy = cute::conditional_t<IsGroupedGemmKernel,
                                              MainloopSm120ArrayTmaWarpSpecializedBlockwiseScaling<PipelineStages,
                                                                        SchedulerPipelineStageCount,
                                                                        ClusterShape_MNK,
                                                                        KernelSchedule>,
                                              MainloopSm120TmaWarpSpecializedBlockwiseScaling<PipelineStages,
                                                                        SchedulerPipelineStageCount,
                                                                        ClusterShape_MNK,
                                                                        KernelSchedule>>;

  using SmemCopyAtomA = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_A<ElementA, ElementB, UseF8f6f4>()), SmemAllocTypeA>;
  using SmemCopyAtomB = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_B<ElementA, ElementB, UseF8f6f4>()), SmemAllocTypeB>;


  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      cute::tuple<StrideA, StrideSFA>,
      ElementB,
      cute::tuple<StrideB, StrideSFB>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
