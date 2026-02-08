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


template <
  class ElementPairA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementPairB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    arch::Sm120,
    arch::OpClassBlockScaledTensorOp,
    ElementPairA,
    GmemLayoutATag,
    AlignmentA,
    ElementPairB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      // Blockscaled Gemm
      (cute::is_base_of_v<KernelScheduleBlockScaledGemmSm120, BuilderScheduleTag> ||
       cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag> ||
       cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, BuilderScheduleTag> ||
       cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, BuilderScheduleTag> ||
       cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, BuilderScheduleTag> ||
       cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>)
       &&
      // Alignment check
      detail::sm1xx_blockscaled_gemm_is_aligned<typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type,
                                                AlignmentA,
                                                typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type,
                                                AlignmentB,
                                                BuilderScheduleTag>()>>
{
  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
  static_assert(cute::is_same_v<ElementSFA, ElementSFB>, "Scale factor types for A and B must be the same.");
  using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
  using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
  using ElementSF = ElementSFA;
  static constexpr auto SFVectorSize = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::SfVectorSize;
  static_assert((SFVectorSize == detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::SfVectorSize),  
                "Scale factor vector size for A and B must be the same.");

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();
  static_assert((UmmaMajorA == UMMA::Major::K && UmmaMajorB == UMMA::Major::K), "Only TN layout is supported.");

  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(cute::is_static_v<ClusterShape_MNK>, "Cluster has to be static");
  static_assert(detail::blockscaled::check_input_datatypes<BuilderScheduleTag, ElementPairA, ElementPairB, UmmaMajorA, UmmaMajorB>(), "Incorrect input types");
  static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch");
  static_assert(size<1>(TileShape_MNK{}) >= 32, "Invalid tile shape N.");

  static constexpr auto Instr = detail::blockscaled::select_instr<ElementPairA,
                                                                  ElementPairB,
                                                                  ElementAccumulator,
                                                                  UmmaMajorA,
                                                                  UmmaMajorB,
                                                                  BuilderScheduleTag>();
  static constexpr bool UseMxf8f6f4 = Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8;
  using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
  using PermTileN = decltype(detail::sm120_tile_n_permute_selector<SFVectorSize>());
  using PermTileK = cute::conditional_t<(UseMxf8f6f4
                                        ), _32, _64>;

  static constexpr bool IsCooperative = !(cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag> ||
                                          cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, BuilderScheduleTag>);
  // Data type used by MMA instruction
  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB>());

  static_assert(!cute::is_base_of_v<KernelScheduleMxf8f6f4Sm120, BuilderScheduleTag> ||
                detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMma, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, false /*IsSparse*/>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_4,_2,_1>>, Layout<Shape<_2,_2,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(
    cute::rr_blockscaled_op_selector_sm120<ElementA,
                                           ElementB,
                                           ElementAccumulator,
                                           ElementSF,
                                           SFVectorSize,
                                           UseMxf8f6f4
                                           >(),
    AtomLayoutMNK{},
    Tile<PermTileM, PermTileN, PermTileK>{}
  ));

  static constexpr int MMA_NSF = size<2>(typename TiledMma::AtomShape_MNK{}) / SFVectorSize;

  using SmemAllocTypeA = cute::conditional_t<UseMxf8f6f4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<UseMxf8f6f4, uint8_t, typename TiledMma::ValTypeB>;
  using SmemAllocTypeSF = ElementSF;

  using GmemTiledCopyA = SM90_TMA_LOAD;
  using GmemTiledCopyB = SM90_TMA_LOAD;

  using GmemTiledCopySFA = SM90_TMA_LOAD;
  using GmemTiledCopySFB = SM90_TMA_LOAD;

  using GmemTiledCopyPairA = decltype(cute::make_tuple(GmemTiledCopyA{}, GmemTiledCopySFA{}));
  using GmemTiledCopyPairB = decltype(cute::make_tuple(GmemTiledCopyB{}, GmemTiledCopySFB{}));

  // Setup Config
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

  using SmemLayoutAtomA = decltype(detail::sm120_rr_smem_selector<SmemAllocTypeA, decltype(size<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::sm120_rr_smem_selector<SmemAllocTypeB, decltype(size<2>(TileShape_MNK{}))>());

  using SmemCopyAtomA = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_A<ElementA,
                                                                                 ElementB,
                                                                                 UseMxf8f6f4
                                                                                 >()), SmemAllocTypeA>;
  using SmemCopyAtomB = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_B<ElementA,
                                                                                 ElementB,
                                                                                 UseMxf8f6f4
                                                                                >()), SmemAllocTypeB>;

  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<SmemAllocTypeSF>, SmemAllocTypeSF>; // auto-vectorized LDS
  using SmemCopyAtomSFA = SmemCopyAtomSF;
  using SmemCopyAtomSFB = SmemCopyAtomSF;

  using SmemCopyAtomsA = decltype(cute::make_tuple(SmemCopyAtomA{}, SmemCopyAtomSFA{}));
  using SmemCopyAtomsB = decltype(cute::make_tuple(SmemCopyAtomB{}, SmemCopyAtomSFB{}));

  // Construct SMEM layout for SF
  // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
  // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size 
  using Blk_MN    = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF    = typename Sm1xxBlkScaledConfig::Blk_SF; 
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

  // Basic storage block for new Scaling Factor Layouts
  using mnBasicBlockShape  =  Shape<_32,_4>;
  using mnBasicBlockStride = Stride<_16,_4>;
  using kBasicBlockShape  = Shape<Int<SFVectorSize>, Int<MMA_NSF>>;
  using kBasicBlockStride = Stride<_0, _1>;
  
  using sSFA_shapeM       = decltype(prepend(size<0>(TileShape_MNK{}) / Blk_MN{},   mnBasicBlockShape{}));
  using sSF_strideMN      = decltype(prepend(                        Blk_Elems{},  mnBasicBlockStride{}));
  using sSFA_strideM      = sSF_strideMN;
  using sSF_shapeK        = decltype(prepend(make_shape( Blk_SF{}/Int<MMA_NSF>{},   size<2>(TileShape_MNK{}) / Int<SFVectorSize>{} / Blk_SF{}),  kBasicBlockShape{}));
  
  using sSFA_strideK      = decltype(prepend(make_stride(         Int<MMA_NSF>{},   size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
  using sSFA_shape        = decltype(make_shape(  sSFA_shapeM{},   sSF_shapeK{}));
  using sSFA_stride       = decltype(make_stride(sSFA_strideM{}, sSFA_strideK{}));
  using SmemLayoutAtomSFA = decltype(make_layout(  sSFA_shape{},  sSFA_stride{}));

  using sSFB_shapeN       = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{},   mnBasicBlockShape{}));
  using sSFB_strideN      = sSF_strideMN;
  using sSFB_strideK      = decltype(prepend(make_stride(Int<MMA_NSF>{},   size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
  using sSFB_shape        = decltype(make_shape(  sSFB_shapeN{},   sSF_shapeK{}));
  using sSFB_stride       = decltype(make_stride(sSFB_strideN{}, sSFB_strideK{}));
  using SmemLayoutAtomSFB = decltype(make_layout(  sSFB_shape{},  sSFB_stride{}));

  using SmemLayoutAtomsA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));
  using SmemLayoutAtomsB = decltype(cute::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_blockscaled<
    detail::sm120_smem_capacity_bytes, SmemAllocTypeA, SmemAllocTypeB, TileShape_MNK, SmemLayoutAtomSFA, SmemLayoutAtomSFB>(StageCountType{});

  static constexpr uint32_t SchedulerPipelineStageCount = 3;

  using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayoutATag>;
  using StrideB = cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>;
  using InternalStrideA  = cute::remove_pointer_t<StrideA>;
  using InternalStrideB  = cute::remove_pointer_t<StrideB>;
  using InternalLayoutSFA = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFA());
  using InternalLayoutSFB = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFB());
  using LayoutSFA = cute::conditional_t<cute::is_same_v<InternalStrideA, StrideA>, InternalLayoutSFA, InternalLayoutSFA *>;
  using LayoutSFB = cute::conditional_t<cute::is_same_v<InternalStrideB, StrideB>, InternalLayoutSFB, InternalLayoutSFB *>;
  using StridePairA = decltype(cute::make_tuple(StrideA{}, LayoutSFA{}));
  using StridePairB = decltype(cute::make_tuple(StrideB{}, LayoutSFB{}));

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;
  static_assert(!IsGroupedGemmKernel || 
                cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, BuilderScheduleTag> ||
                cute::is_base_of_v<KernelScheduleAuto, BuilderScheduleTag> ||
                cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, BuilderScheduleTag>,
                "Invalid builder schedule tag for grouped GEMM");

  using KernelSchedule = cute::conditional_t<IsGroupedGemmKernel, 
                                              // PtrArray
                                              cute::conditional_t<IsCooperative, 
                                                KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<SchedulerPipelineStageCount>, 
                                                KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120<SchedulerPipelineStageCount>>,
                                              // Non-PtrArray
                                              cute::conditional_t<IsCooperative, 
                                                KernelTmaWarpSpecializedCooperativeBlockScaledSm120<SchedulerPipelineStageCount>, 
                                                KernelTmaWarpSpecializedPingpongBlockScaledSm120<SchedulerPipelineStageCount>>>;

  using DispatchPolicy = cute::conditional_t<IsGroupedGemmKernel,
                                              MainloopSm120ArrayTmaWarpSpecializedBlockScaled<PipelineStages,
                                                                    SchedulerPipelineStageCount,
                                                                    ClusterShape_MNK,
                                                                    KernelSchedule>,
                                              MainloopSm120TmaWarpSpecializedBlockScaled<PipelineStages,
                                                                    SchedulerPipelineStageCount,
                                                                    ClusterShape_MNK,
                                                                    KernelSchedule>>;
                                                                    
  static_assert(cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, typename DispatchPolicy::Schedule> ||
                cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, typename DispatchPolicy::Schedule> ||
                cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, typename DispatchPolicy::Schedule> ||
                cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, typename DispatchPolicy::Schedule>, 
                "Unsupported kernel schedule by this collective mainloop dispatch policy.");

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      cute::tuple<ElementA, ElementSF>,
      StridePairA,
      cute::tuple<ElementB, ElementSF>,
      StridePairB,
      TiledMma,
      GmemTiledCopyPairA,
      SmemLayoutAtomsA,
      SmemCopyAtomsA,
      cute::identity,
      GmemTiledCopyPairB,
      SmemLayoutAtomsB,
      SmemCopyAtomsB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
