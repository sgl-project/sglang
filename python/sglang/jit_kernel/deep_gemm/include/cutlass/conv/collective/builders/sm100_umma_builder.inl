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
//

//
#pragma once

#include "cutlass/conv/collective/builders/sm100_common.inl"
#include "cutlass/conv/collective/builders/sm90_gmma_builder.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  conv::Operator ConvOp,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNKL,    // (MmaAtomShapeM, MmaAtomShapeN, TileK, optional: TileL)
  class ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassTensorOp,
    ConvOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNKL,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecialized1SmSm100> ||
       cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecialized2SmSm100> ||
       cute::is_same_v<KernelScheduleType, KernelStridedDgradTmaWs1SmSm100> ||
       cute::is_same_v<KernelScheduleType, KernelStridedDgradTmaWs2SmSm100> ||
       cute::is_same_v<KernelScheduleType, KernelScheduleAuto>) &&
      ((sizeof(ElementA) * AlignmentA) % cutlass::gemm::collective::detail::tma_alignment_bytes == 0) &&
      ((sizeof(ElementB) * AlignmentB) % cutlass::gemm::collective::detail::tma_alignment_bytes == 0)>> {
private:
  // For fprop, majorA = K,  major B = K;
  // For wgrad, majorA = MN, major B = MN;
  // For dgrad, majorA = K,  major B = MN;
  static constexpr cute::UMMA::Major UmmaMajorA =
    (ConvOp == conv::Operator::kWgrad) ? cute::UMMA::Major::MN : cute::UMMA::Major::K;
  static constexpr cute::UMMA::Major UmmaMajorB =
    (ConvOp == conv::Operator::kFprop) ? cute::UMMA::Major::K : cute::UMMA::Major::MN;

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TileShape_MNK = decltype(cute::take<0,3>(TileShape_MNKL{})); // (MmaAtomShapeM, MmaAtomShapeN, TileK)

  static constexpr auto
  get_tiled_mma_schedule() {
    if constexpr (cute::is_same_v<KernelScheduleType, KernelStridedDgradTmaWs1SmSm100>) {
      return KernelImplicitTmaWarpSpecialized1SmSm100{};
    }
    else if constexpr (cute::is_same_v<KernelScheduleType, KernelStridedDgradTmaWs2SmSm100>) {
      return KernelImplicitTmaWarpSpecialized2SmSm100{};
    }
    else {
      return KernelScheduleType{};
    }
  }

  using TiledMmaSchedule = decltype(get_tiled_mma_schedule());
  using TiledMma = decltype(detail::sm100_make_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                                         TileShape_MNK, ClusterShape_MNK,
                                                         UmmaMajorA, UmmaMajorB, TiledMmaSchedule>());

  using AtomThrID = typename TiledMma::AtomThrID;

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  static constexpr auto
  get_tma_atom_A() {
    if constexpr (cute::is_same_v<KernelScheduleType,KernelStridedDgradTmaWs1SmSm100> ||
                  cute::is_same_v<KernelScheduleType,KernelStridedDgradTmaWs2SmSm100>) {
      static_assert(ConvOp == conv::Operator::kDgrad, "Operator+Schedule mismatch");
      return cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(ClusterShape_MNK{}, AtomThrID{});
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(ClusterShape_MNK{}, AtomThrID{});
    }
    else {
      return cutlass::conv::collective::detail::sm100_cluster_shape_to_im2col_tma_atom_A(ClusterShape_MNK{}, AtomThrID{});
    }
  }

  static constexpr auto
  get_tma_atom_B() {
    if constexpr (cute::is_same_v<KernelScheduleType,KernelStridedDgradTmaWs1SmSm100> ||
                  cute::is_same_v<KernelScheduleType,KernelStridedDgradTmaWs2SmSm100>) {
      static_assert(ConvOp == conv::Operator::kDgrad, "Operator+Schedule mismatch");
      return cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(ClusterShape_MNK{}, AtomThrID{});
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return cutlass::conv::collective::detail::sm100_cluster_shape_to_im2col_tma_atom_B(ClusterShape_MNK{}, AtomThrID{});
    }
    else {
      return cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(ClusterShape_MNK{}, AtomThrID{});
    }
  }

  // For wgrad kernel, tensor A uses tma tiled mode and tensor B uses tma im2col mode.
  using GmemTiledCopyA = decltype(get_tma_atom_A());
  using GmemTiledCopyB = decltype(get_tma_atom_B());

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma, BlockTileA_M, BlockTileA_K>());

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma, BlockTileB_N, BlockTileB_K>());

  // Calculate SMEM matrix A and B buffers' pipeline stages
  static constexpr uint32_t AccumulatorPipelineStageCount = 2;
  static constexpr uint32_t SchedulerPipelineStageCount = 1;
  static constexpr uint32_t CLCResponseSize = 16;

  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLCPipeline = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // LoadOrderBarrier = OrderedSequenceBarrier<1,2>
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * CLCResponseSize;
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = SchedulerPipelineStageCount * sizeof(uint32_t);
  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( AccumulatorPipelineStorage +
                                                               CLCPipelineStorage +
                                                               LoadOrderBarrierStorage +
                                                               TmemDeallocStorage +
                                                               CLCResponseStorage +
                                                               TmemBasePtrsStorage);
  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      Sm100ReducedSmemCapacityBytes, ElementAMma, ElementBMma, SmemTileShape>(StageCountType{});

  constexpr static int NumSpatialDimensions = detail::gmem_layout_tags_to_spatial_dims<GmemLayoutA, GmemLayoutB>();

  using DispatchPolicy = cutlass::conv::MainloopSm100TmaUmmaWarpSpecializedImplicitGemm<
      ConvOp,
      PipelineStages,
      NumSpatialDimensions,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape_MNK>;

public:
  using CollectiveOp = cutlass::conv::collective::CollectiveConv<
      DispatchPolicy,
      TileShape_MNKL,
      ElementA,
      ElementB,
      TiledMma,
      detail::Sm100ImplicitGemmTileTraits<GmemTiledCopyA, SmemLayoutAtomA>,
      detail::Sm100ImplicitGemmTileTraits<GmemTiledCopyB, SmemLayoutAtomB>
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
