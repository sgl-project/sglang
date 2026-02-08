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

#include "cutlass/gemm/collective/collective_builder_decl.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_blockscaled_mixed_tma_cpasync(StageCount<stages> stage_count) {
  return stages;
}

template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int carveout_bytes
>
constexpr int
sm100_compute_stage_count_or_override_blockscaled_mixed_tma_cpasync(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For MXF8F6F4 MMA, ElementA/B will be passed in as uint8_t
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A and smem for B (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one MainloopPipeline = PipelineTmaUmmaAsync (CollectiveMma::SharedStorage::SharedStorage)
  // 3. smem for SFB and smem for SFB (CollectiveMma::SharedStorage::TensorStorage, independent of input size b.c. sizeof(sf) is fixed)
  constexpr auto mainloop_pipeline_bytes = 
    sizeof(typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage) + 
    sizeof(typename cutlass::PipelineUmmaConsumerAsync<1>::SharedStorage);
  
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto stage_sfa_bytes = size(filter_zeros(TileShapeSFA{}));
  constexpr auto stage_sfb_bytes = size(filter_zeros(TileShapeSFB{}));

  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes + stage_sfa_bytes + stage_sfb_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

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
    arch::Sm100,
    arch::OpClassBlockScaledTensorOp,
    ElementPairA,
    GmemLayoutATag,
    AlignmentA,
    ElementPairB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape (_1, _1, _1)
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<cute::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100, BuilderScheduleTag> >
>
{
  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
  using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
  using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
  using ElementSF = ElementSFA;

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(detail::blockscaled::check_input_datatypes<BuilderScheduleTag, ElementPairA, ElementPairB, UmmaMajorA, UmmaMajorB>(), "Incorrect input types");
  
  static constexpr bool is_2sm = false; // detail::blockscaled::is_2sm<TileShape_MNK, ClusterShape_MNK, BuilderScheduleTag>();
  static constexpr auto Instr = detail::blockscaled::select_instr<ElementPairA, ElementPairB, ElementAccumulator, UmmaMajorA, UmmaMajorB, BuilderScheduleTag>();

  using TiledMma = typename cutlass::gemm::collective::detail::TrivialBlockscaledMma<ElementPairA, ElementPairB, ElementAccumulator,
                                                                  TileShape_MNK, ClusterShape_MNK,
                                                                  UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag, is_2sm>::type;

  static constexpr bool UseMxf8f6f4 = Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8;

  static_assert(UseMxf8f6f4 || (cutlass::gemm::detail::is_k_major_A<GmemLayoutATag>() && cutlass::gemm::detail::is_k_major_B<GmemLayoutBTag>()), "Only MMA.MXF8F6F4 supports non-K major inputs");

  // Data type used by MMA instruction
  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA, UseMxf8f6f4>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB, UseMxf8f6f4>());

  static_assert(detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMma, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, false /*is_sparse*/, is_2sm>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  static constexpr uint32_t SFVectorSize = TiledMma::SFVecSize;

  using ElementAMma_SmemAllocType = cute::conditional_t<UseMxf8f6f4, uint8_t, ElementAMma>;
  using ElementBMma_SmemAllocType = cute::conditional_t<UseMxf8f6f4, uint8_t, ElementBMma>;

  // using TiledMma =  decltype(detail::sm100_make_trivial_tiled_mma<
  //     ElementAMma, ElementBMma, ElementAccumulator,
  //     decltype(cute::product_each(TileShape_MNK{})), ClusterShape_MNK,
  //     UmmaMajorA, UmmaMajorB, BuilderScheduleTag>());

  using AtomThrID = typename TiledMma::AtomThrID;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  // Assigning 4 warps for mainloop load of B
  static constexpr int NumLoadThreadsCpAsync = 128;

  
  using SmemShapeA_M = decltype(shape_div(shape<0>(TileShape_MNK{}), shape_div(shape<0>(TileShape_MNK{}), size<0>(TileShape_MNK{}) / size(AtomThrID{}))));
  using SmemShapeA_K = decltype(cute::get<2>(TileShape_MNK{}));


  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
    ClusterShape_MNK{}, AtomThrID{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma_SmemAllocType, SmemShapeA_M, SmemShapeA_K>());

  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(cutlass::sizeof_bits<ElementB>::value) * AlignmentB / 8>;
  using GmemCopyAtomB = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<AlignmentTypeB>, ElementB>;
  using GmemTiledCopyB = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomB, NumLoadThreadsCpAsync, AlignmentB, TagToStrideB_t<GmemLayoutBTag>,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma_SmemAllocType, BlockTileB_N, BlockTileB_K>());

  using GmemTiledCopySFA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));
  using GmemTiledCopySFB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_SFB(
      ClusterShape_MNK{}, AtomThrID{})); 

  using GmemTiledCopyPairA = decltype(cute::make_tuple(GmemTiledCopyA{}, GmemTiledCopySFA{}));
  using GmemTiledCopyPairB = decltype(cute::make_tuple(GmemTiledCopyB{}, GmemTiledCopySFB{}));

  using SmemLayoutAtomSFA = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomSFB = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomsA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));
  using SmemLayoutAtomsB = decltype(cute::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));


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

  static constexpr uint32_t AccumulatorPipelineStageCount = 2;
  // Calculate scheduler pipeline stages. Having one more stage than the accumulator allows more latency hiding.
  static constexpr uint32_t SchedulerPipelineStageCount = AccumulatorPipelineStageCount + 1;

  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLCPipeline = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * detail::CLCResponseSize;
  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( AccumulatorPipelineStorage +
                                                               CLCPipelineStorage +
                                                               CLCResponseStorage);
  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute::Shape<SmemShapeA_M, BlockTileB_N, SmemShapeA_K>;

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_blockscaled_mixed_tma_cpasync<
      Sm100ReducedSmemCapacityBytes, ElementAMma_SmemAllocType, ElementBMma_SmemAllocType, SmemTileShape, SmemLayoutAtomSFA, SmemLayoutAtomSFB>(StageCountType{});
  static_assert(PipelineStages > 0, "Smem usage is too high. Can't create any SMEM buffers for A, B, SFA, and SFB.");

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100UmmaMixedTmaCpAsyncWarpSpecializedBlockScaled<
        PipelineStages,
        SchedulerPipelineStageCount,
        AccumulatorPipelineStageCount,
        ClusterShape_MNK>,
      TileShape_MNK,
      cute::tuple<ElementA, ElementSF>,
      StridePairA,
      cute::tuple<ElementB, ElementSF>,
      StridePairB,
      TiledMma,
      GmemTiledCopyPairA,
      SmemLayoutAtomsA,
      void,
      cute::identity,
      GmemTiledCopyPairB,
      SmemLayoutAtomsB,
      void,
      cute::identity
    >;
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
