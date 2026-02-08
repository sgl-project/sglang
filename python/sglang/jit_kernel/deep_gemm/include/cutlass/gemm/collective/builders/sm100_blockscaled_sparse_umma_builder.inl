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

#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/gemm/collective/builders/sm100_pipeline_carveout.inl"
#include "cutlass/gemm/collective/builders/sm1xx_sparse_config.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template <
  class ElementAMma,
  class ElementB,
  class ElementEMma,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  class ClusterShapeMNK,
  uint32_t AccumulatorPipelineStageCount,
  uint32_t SchedulerPipelineStageCount,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_blockscaled_sparse(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template <
  class ElementAMma,
  class ElementB,
  class ElementEMma,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  class ClusterShapeMNK,
  uint32_t AccumulatorPipelineStageCount,
  uint32_t SchedulerPipelineStageCount,
  int carveout_bytes
>
constexpr int
sm100_compute_stage_count_or_override_blockscaled_sparse(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For MXF8F6F4 and MXF4NVF4 kernels, ElementAMma will be passed in as sparse_elem<uint8_t, Sparsity>, and
  //    ElementB will be passed in as uint8_t

  constexpr uint32_t KernelSmemCarveout = detail::Sm100SparseGemmTmaUmmaCarveout<
                                            ClusterShapeMNK,
                                            AccumulatorPipelineStageCount,
                                            SchedulerPipelineStageCount,
                                            detail::CLCResponseSize>::KernelSmemCarveout;

  // * Compute Stage
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A,B,E (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one MainloopPipeline = (CollectiveMma::SharedStorage::PipelineStorage = PipelineTmaSparseUmmaAsync, two arrive-wait barrier)
  // 3. smem for SFB and smem for SFB (CollectiveMma::SharedStorage::TensorStorage, independent of input size b.c. sizeof(sf) is fixed)
  constexpr auto MainloopPipelineStorage_per_Stage_aligned = static_cast<int>(cutlass::round_up(sizeof(cutlass::arch::ClusterBarrier) * 2, 16));

  // a_bits, e_bits already consider sparsity through `sizeof_bits(ElementAMma)
  // NOTE: sizeof_bits<sparse_elem<>> return integral_ratio instead of size_t
  constexpr auto a_bits = cute::sizeof_bits_v<ElementAMma>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto e_bits = cute::sizeof_bits_v<ElementEMma>;
  constexpr auto stage_sfa_bytes = cute::size(filter_zeros(TileShapeSFA{}));
  constexpr auto stage_sfb_bytes = cute::size(filter_zeros(TileShapeSFB{}));

  constexpr int MainloopTensorStorage_per_Stage =
    cutlass::round_up(
      cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      cutlass::bits_to_bytes(e_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      stage_sfa_bytes +
      stage_sfb_bytes,
    128);

  constexpr auto EpilogueSharedStorage = carveout_bytes;

  constexpr auto Stages = (cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout - EpilogueSharedStorage) / 
    (MainloopTensorStorage_per_Stage + MainloopPipelineStorage_per_Stage_aligned);

  return Stages;
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
  class TileShape_MNK,        // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ClusterShape_MNK,     // Static cluster shape or dynamic (int, int, _1)
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassBlockScaledSparseTensorOp,
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
      // Blockscaled Sparse Gemm
      cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag>
       &&
      // Alignment check
      detail::sm1xx_blockscaled_sparse_gemm_is_aligned<typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type,
                                                       AlignmentA,
                                                       GmemLayoutATag,
                                                       typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type,
                                                       AlignmentB,
                                                       BuilderScheduleTag>()>>
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

  static constexpr bool is_2sm = detail::blockscaled::is_2sm<TileShape_MNK, ClusterShape_MNK, BuilderScheduleTag>();
  static constexpr auto Instr = detail::blockscaled::select_instr<ElementPairA,
                                                                  ElementPairB,
                                                                  ElementAccumulator,
                                                                  UmmaMajorA,
                                                                  UmmaMajorB,
                                                                  BuilderScheduleTag>();
  static constexpr bool UseMxf8f6f4 = Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8;
  static_assert(UseMxf8f6f4 || (cutlass::gemm::detail::is_k_major_A<GmemLayoutATag>() && cutlass::gemm::detail::is_k_major_B<GmemLayoutBTag>()), "Only MMA.MXF8F6F4 supports non-K major inputs");

  // Data type used by MMA instruction
  using ElementAMmaRaw = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA, UseMxf8f6f4>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB, UseMxf8f6f4>());

  static_assert(detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMmaRaw, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, true /*is_sparse*/, is_2sm>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  using TiledMma = typename cutlass::gemm::collective::detail::TrivialBlockscaledMma<ElementPairA, ElementPairB, ElementAccumulator,
                                                                  TileShape_MNK, ClusterShape_MNK,
                                                                  UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag, is_2sm>::type;

  static constexpr uint32_t SFVectorSize = TiledMma::SFVecSize;

  // Basic storage block for new Scaling Factor Layouts
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaSparsity = Int<ElementAMma::sparsity>;
  using ElementEMma = typename TiledMma::ValTypeE;
  using AtomThrID = typename TiledMma::AtomThrID;
  using Sm1xxSparseConfig = cutlass::Sm1xxGemmSparseConfig<ElementAMma, GmemLayoutATag, ElementEMma>;

  // For both MXF4NVF4 and MXF8F6F4 instructions underlying SMEM allocation type for A is uint8
  using ElementAMmaRaw_SmemAllocType = uint8_t;
  using ElementAMma_SmemAllocType = cute::sparse_elem<ElementAMmaSparsity{}, ElementAMmaRaw_SmemAllocType>;
  using ElementBMma_SmemAllocType = cute::conditional_t<UseMxf8f6f4, uint8_t, ElementB>;

  using LayoutA = decltype(Sm1xxSparseConfig::deduce_layoutA());
  using LayoutE = decltype(Sm1xxSparseConfig::deduce_layoutE());
  using LayoutPairAE = decltype(cute::make_tuple(LayoutA{}, LayoutE{}));

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopySFA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopySFB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_SFB(
      ClusterShape_MNK{}, AtomThrID{})); 

  using GmemTiledCopyPairA = decltype(cute::make_tuple(GmemTiledCopyA{}, GmemTiledCopySFA{}));
  using GmemTiledCopyPairB = decltype(cute::make_tuple(GmemTiledCopyB{}, GmemTiledCopySFB{}));

  //
  // Construct SMEM layout (SmemLayoutAtom) for A and SFA
  //
  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector_sparse<
      UmmaMajorA, ElementAMmaRaw_SmemAllocType, BlockTileA_M, BlockTileA_K, ElementAMmaSparsity>());

  // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
  // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size 
  using SmemLayoutAtomSFA = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomsA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));

  //
  // Construct SMEM layout (SmemLayoutAtom) for B and SFB
  //
  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma_SmemAllocType, BlockTileB_N, BlockTileB_K>());
  using SmemLayoutAtomSFB = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomsB = decltype(cute::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));

  //
  // Construct Strides for A, SFA, B, and SFB
  //
  using StridePairA = decltype(cute::make_tuple(LayoutA{}, LayoutE{}, Sm1xxBlkScaledConfig::deduce_layoutSFA()));
  using StridePairB = decltype(cute::make_tuple(cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>{}, Sm1xxBlkScaledConfig::deduce_layoutSFB()));

  // Calculate SMEM matrix A and B buffers' pipeline stages and the accumulator stages.
  static constexpr uint32_t AccumulatorNPerCta = cute::size<1>(TileShape_MNK{});
  static constexpr uint32_t AccumulatorPipelineStageCount = (AccumulatorNPerCta == 256) ? 1 : 2;
  static constexpr uint32_t SchedulerPipelineStageCount = 2;

  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_blockscaled_sparse<
      ElementAMma_SmemAllocType,
      ElementBMma_SmemAllocType,
      ElementEMma,
      SmemTileShape,
      SmemLayoutAtomSFA,
      SmemLayoutAtomSFB,
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount>(StageCountType{});

  using DispatchPolicy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledSparse<
        PipelineStages,
        SchedulerPipelineStageCount,
        AccumulatorPipelineStageCount,
        ClusterShape_MNK>;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
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

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
