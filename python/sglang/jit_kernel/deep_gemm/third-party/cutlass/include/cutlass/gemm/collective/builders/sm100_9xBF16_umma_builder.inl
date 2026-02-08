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
//

//

#include "cutlass/gemm/collective/builders/sm100_common.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<
  int CapacityBytes,
  class CtaTileShape_MNK,
  class TiledMma,
  class BuilderScheduleTag,
  UMMA::Major UmmaMajorA,
  int ComplexComponent = 1,
  int NumComputeMtxs = 3,
  int carveout_bytes
>
constexpr cute::tuple<int, int, int>
sm100_compute_stage_count_or_override_fast_fp32(StageCountAutoCarveout<carveout_bytes> stage_count) {
  constexpr int CtaM = get<0>(CtaTileShape_MNK{});
  constexpr int CtaN = get<1>(CtaTileShape_MNK{});
  static_assert(CtaN <= 128, "Can't support CtaN>128 tiles");
  constexpr int CtaK = get<2>(CtaTileShape_MNK{});
  using AtomThrID = typename TiledMma::AtomThrID;
  constexpr int TmemColumns = 512;

  // Detect 2x2 TMEM layout
  constexpr int TmemAccWordsPerDP = (CtaM == 64 && size(AtomThrID{}) == 2) ? CtaN/2 : CtaN;
  constexpr int TmemAWordsPerDP = ComplexComponent * NumComputeMtxs * CtaK / 2;
  constexpr bool IsAComputeinTmem = UmmaMajorA == cute::UMMA::Major::K && !cute::is_base_of_v<KernelTmaWarpSpecializedFastFP32SmemSm100, BuilderScheduleTag>;
  constexpr bool IsAComputeinSmem = !IsAComputeinTmem;
  constexpr int AccumulatorStageCount = (IsAComputeinTmem) ? (((TmemAccWordsPerDP * ComplexComponent == 128) ? 2 : 3) * ComplexComponent) : (TmemColumns / TmemAccWordsPerDP);
  
  constexpr int SmemCapacityAfterMma2AccumCarveout = CapacityBytes - (carveout_bytes + AccumulatorStageCount * 32);

  constexpr int TmemInAStageCount_Potential = (IsAComputeinTmem) ? (TmemColumns - AccumulatorStageCount * TmemAccWordsPerDP) / TmemAWordsPerDP : 10000;
  
  constexpr auto load2transform_pipeline_bytes = sizeof(typename cutlass::PipelineTmaTransformAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<float> * ComplexComponent;
  constexpr auto b_bits = cute::sizeof_bits_v<float> * ComplexComponent;
  constexpr int ab_stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(CtaTileShape_MNK{}) * size<2>(CtaTileShape_MNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(CtaTileShape_MNK{}) / size(AtomThrID{}) * size<2>(CtaTileShape_MNK{})) +
    static_cast<int>(load2transform_pipeline_bytes);

  constexpr auto transform2mma_pipeline_bytes = sizeof(typename cutlass::PipelineUmmaConsumerAsync<1>::SharedStorage);
  constexpr auto a_compute_bits = cute::sizeof_bits_v<cutlass::bfloat16_t> * ComplexComponent;
  constexpr auto b_compute_bits = cute::sizeof_bits_v<cutlass::bfloat16_t> * ComplexComponent * ComplexComponent;
  constexpr int ab_compute_stage_bytes =
    cutlass::bits_to_bytes(NumComputeMtxs * a_compute_bits * int(IsAComputeinSmem) * size<0>(CtaTileShape_MNK{})  * size<2>(CtaTileShape_MNK{})) + // If ACompute is in TMEM, Acompute buffer has 0 bytes.
    cutlass::bits_to_bytes(NumComputeMtxs * b_compute_bits * size<1>(CtaTileShape_MNK{}) / size(AtomThrID{}) * size<2>(CtaTileShape_MNK{})) +
    static_cast<int>(transform2mma_pipeline_bytes);

  constexpr int ABComputeStageCount_Potential = SmemCapacityAfterMma2AccumCarveout / (ab_stage_bytes + ab_compute_stage_bytes);
  // The number of SMEM buffers for A, B. ACompute (if in SMEM), BCompute should be at least Transform2MmaStageCount
  constexpr int Transform2MmaStageCount = std::min(TmemInAStageCount_Potential, ABComputeStageCount_Potential);

  constexpr int SmemCapacityAfterABComputeCarveout = SmemCapacityAfterMma2AccumCarveout - (Transform2MmaStageCount * ab_compute_stage_bytes);
  // Can we boost the number of buffers for A and B?
  constexpr int Load2TransformStageCount = SmemCapacityAfterABComputeCarveout / ab_stage_bytes;

  static_assert(Load2TransformStageCount >= 2 && Transform2MmaStageCount >= 2 && AccumulatorStageCount >= 2, "Not enough SMEM or TMEM capacity for selected tile size");
  return cute::make_tuple(Load2TransformStageCount, Transform2MmaStageCount, AccumulatorStageCount);
}

} // namespace detail


// FastFP (9xBF16) MMA kernels builder
template <
  class GmemLayoutATag,
  int AlignmentA,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,  // The Cluster-level TileShape
  class ClusterShape_MNK,
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassTensorOp,
    float,           // ElementA
    GmemLayoutATag,  // LayoutA 
    AlignmentA,
    float,           // ElementB
    GmemLayoutBTag,  // LayoutB
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, int)
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      (not cute::is_tuple<GmemLayoutATag>::value && not cute::is_tuple<GmemLayoutBTag>::value) &&
      (cute::is_base_of_v<KernelScheduleSm100FastFP32Gemm, BuilderScheduleTag>) &&
      ((sizeof(float) * AlignmentA) % detail::tma_alignment_bytes == 0) &&
      ((sizeof(float) * AlignmentB) % detail::tma_alignment_bytes == 0)>>
{
  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  using ElementA = float;
  using ElementB = float;
  using ElementAMma = cutlass::bfloat16_t;
  using ElementBMma = cutlass::bfloat16_t;
  static constexpr int ScalingFactor = 8;

  using TiledMma = decltype(detail::sm100_make_trivial_fastFP32_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ScalingFactor, BuilderScheduleTag>());
  using AtomThrID = typename TiledMma::AtomThrID;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  using CtaTileShape_MNK = decltype(shape_div(TileShape_MNK{}, AtomThrShapeMNK{}));

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));

  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<UmmaMajorA, ElementA,
    BlockTileA_M, BlockTileA_K>());
  // Take 3 compute buffers into account for swizzle selection
  using SmemLayoutAtomACompute =  decltype(cutlass::gemm::collective::detail::sm100_smem_selector<UmmaMajorA, ElementAMma,
    BlockTileA_M, BlockTileA_K>());

  // Input transform kernel can not use TMA 2SM instructions.
  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(cute::size<1>(ClusterShape_MNK{})));
  using SmemLayoutAtomPairA = cutlass::gemm::collective::detail::CollectiveMmaEmulatedLayoutAtomType<
    SmemLayoutAtomA, SmemLayoutAtomACompute>;
  
  static constexpr int MMA_M = cute::size<0,0>(MmaShapeA_MK{});
  using CopyAtomPairA = cutlass::gemm::collective::detail::CollectiveMmaEmulatedCopyType<
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementA>,
    cute::conditional_t<(UmmaMajorA == cute::UMMA::Major::K && !cute::is_base_of_v<KernelTmaWarpSpecializedFastFP32SmemSm100, BuilderScheduleTag>),
                        cute::conditional_t<(MMA_M == 64 && size(AtomThrID{}) == 1), SM100_TMEM_STORE_16dp256b1x, SM100_TMEM_STORE_32dp32b8x>, // TS Implementation
                        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementA>>                                         // SS Implementation
  >;

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  
  // Input transform kernel can not use TMA 2SM instructions.
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(cute::size<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<UmmaMajorB, ElementB,
    BlockTileB_N, BlockTileB_K>());
  // Take 3 compute buffers into account for swizzle selection
  using SmemLayoutAtomBCompute = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<UmmaMajorB, ElementBMma,
    BlockTileB_N, BlockTileB_K>());
  
  using SmemLayoutAtomPairB = cutlass::gemm::collective::detail::CollectiveMmaEmulatedLayoutAtomType<
    SmemLayoutAtomB, SmemLayoutAtomBCompute>;
  using CopyAtomPairB = cutlass::gemm::collective::detail::CollectiveMmaEmulatedCopyType<
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementB>, 
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementBMma>
  >;

  // SmemCarveout
  static constexpr int NumBandsToCompute = 5;
  static constexpr int AccPromotionInterval = 1;
  static constexpr int SchedulerPipelineStageCount = 3;
  static constexpr bool IsArrayOfPointersGemm = (cute::is_base_of_v<KernelScheduleSm100PtrArrayFastFP32Gemm, BuilderScheduleTag>);

  // CLCPipeline = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * detail::CLCResponseSize;
  // CLC Throttle pipeline storage
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = sizeof(uint32_t);
  // Tensormap Storage
  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;

  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( CLCPipelineStorage +
                                                               CLCResponseStorage +
                                                               CLCThrottlePipelineStorage +
                                                               TmemDeallocStorage +
                                                               TmemBasePtrsStorage +
                                                               TensorMapStorage);

  // Reduce SMEM capacity available for buffers considering extra B smem and barrier smem allocations
  static constexpr int Sm100ReducedSmemCapacityBytes = detail::sm100_smem_capacity_bytes - KernelSmemCarveout;
  static constexpr auto stage_info = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_fast_fp32<
    Sm100ReducedSmemCapacityBytes, CtaTileShape_MNK, TiledMma, BuilderScheduleTag, UmmaMajorA>(StageCountType{});
  
  static constexpr int Load2TransformPipelineStageCount = get<0>(stage_info);
  static constexpr int Transform2MmaPipelineStageCount = get<1>(stage_info);
  static constexpr int AccumulatorPipelineStageCount = get<2>(stage_info);

  using AccumulatorCopyAtom = cute::SM100_TMEM_LOAD_32dp32b32x;

  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
    cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedFastF32<
      Load2TransformPipelineStageCount,
      Transform2MmaPipelineStageCount,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      NumBandsToCompute,
      ScalingFactor,
      AccPromotionInterval,
      ClusterShape_MNK,
      AccumulatorCopyAtom>,
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedFastF32<
      Load2TransformPipelineStageCount,
      Transform2MmaPipelineStageCount,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      NumBandsToCompute,
      ScalingFactor,
      AccPromotionInterval,
      ClusterShape_MNK,
      AccumulatorCopyAtom>
  >;
  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy,
    TileShape_MNK,
    ElementA,
    cutlass::gemm::TagToStrideA_t<GmemLayoutATag>,
    ElementB,
    cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>,
    TiledMma,
    GmemTiledCopyA,
    SmemLayoutAtomPairA,
    CopyAtomPairA,
    cute::identity,
    GmemTiledCopyB,
    SmemLayoutAtomPairB,
    CopyAtomPairB,
    cute::identity
  >;
};

} // namespace cutlass::gemm::collective
