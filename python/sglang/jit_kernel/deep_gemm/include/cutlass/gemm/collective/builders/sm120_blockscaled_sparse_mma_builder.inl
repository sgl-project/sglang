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
#include "cutlass/gemm/collective/builders/sm1xx_sparse_config.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template <
  int CapacityBytes,
  class ElementAMma,
  class ElementB,
  class ElementEMma,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int stages
>
constexpr int
sm120_compute_stage_count_or_override_blockscaled_sparse(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity.
// With limited SMEM capacity, F8/F6/F4 MMA with larger tiles, such as 128x128, cannot
// get 2 stages. Therefore, use 1.5 stages for B.
template <
  int CapacityBytes,
  class ElementAMma,
  class ElementB,
  class ElementEMma,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int carveout_bytes
>
constexpr auto
sm120_compute_stage_count_or_override_blockscaled_sparse(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For MXF8F6F4 & MXF4NVF4 sub-bytes, ElementAMma will be passed in as sparse_elem<uint8_t, Sparsity>, and
  //    ElementB will be passed in as uint8_t
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A/E and smem for B (CollectiveMma::SharedStorage::TensorStorage)
  // 2. smem for SFA and smem for SFB (CollectiveMma::SharedStorage::TensorStorage, independent of input size b.c. sizeof(sf) is fixed)
  // 3. one MainloopPipeline = (CollectiveMma::SharedStorage::PipelineStorage = PipelineTmaSparseUmmaAsync, three arrive-wait barrier)
  constexpr auto mainloop_pipeline_bytes = sizeof(cutlass::arch::ClusterBarrier) * 3;
  // a_bits, e_bits already consider sparsity through `sizeof_bits(ElementAMma)
  // NOTE: sizeof_bits<sparse_elem<>> return integral_ratio instead of size_t.
  constexpr auto a_bits = cute::sizeof_bits_v<ElementAMma>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto e_bits = cute::sizeof_bits_v<ElementEMma>;
  constexpr auto stage_sfa_bytes = size(filter_zeros(TileShapeSFA{}));
  constexpr auto stage_sfb_bytes = size(filter_zeros(TileShapeSFB{}));

  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(e_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes + stage_sfa_bytes + stage_sfb_bytes);

  // When stage >=2, it can be used directly.
  constexpr int stages = (CapacityBytes - carveout_bytes) / stage_bytes;
  if constexpr (stages >= 2) {
    return cute::make_tuple(stages, stages);
  }

  // When stage < 2, try to use half of TileK, aka size<2>(TileShapeMNK{}),
  //   and increase its stages. Eg. when input TileK is 256, instead of
  //   buffering K=256 with 2 stages, it uses K=128, with 3 stages. From the
  //   kernel's TileK view (K=256), B is 1.5 stages.
  // Note, if B apply 1.5 stages, metadata is kept in L2, as limited SMEM capacity.
  //  A/B is with asymmetric DMA and buffering, as they are with different
  //  TileK and buffer advance steps.
  constexpr int stage_bytes_b15 =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) * 3 / 4 +
    static_cast<int>(mainloop_pipeline_bytes + stage_sfa_bytes + stage_sfb_bytes);

  constexpr int stages_b15 = (CapacityBytes - carveout_bytes) / stage_bytes_b15;
  if constexpr (stages_b15 >= 2) {
    return cute::make_tuple(stages_b15, 3);
  }

  return cute::make_tuple(stages_b15, stages_b15);
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
    arch::Sm120,
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
      cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>
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
  static constexpr auto SFVectorSize = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::SfVectorSize;

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(cute::is_static_v<ClusterShape_MNK>, "Cluster has to be static");
  static_assert(detail::blockscaled::check_input_datatypes<BuilderScheduleTag, ElementPairA, ElementPairB, UmmaMajorA, UmmaMajorB>(), "Incorrect input types");
  static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch");

  static constexpr auto Instr = detail::blockscaled::select_instr<ElementPairA,
                                                                  ElementPairB,
                                                                  ElementAccumulator,
                                                                  UmmaMajorA,
                                                                  UmmaMajorB,
                                                                  BuilderScheduleTag>();
  static constexpr bool UseMxf8f6f4 = Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8;

  using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
  using PermTileN = decltype(cute::min(size<1>(TileShape_MNK{}),  _32{}));
  // For MXF8F6F4 MMA, FP4/FP6 types are in 8-bits containers in registers. For MXF4NVF4 MMA, it is 2 elements in one register.
  //     Operand A and B are coming from registers.
  using SizeOfBits_InRegs = cute::conditional_t<UseMxf8f6f4, _8, _4>;
  //  MXF4NVF4 MMA has 2:4 sparsity, while MXF8F6F4 MMA has 1:2. Therefore, we need to update K-dim bits from 256 to 512 to 
  //     find the logical K-dim of MMA.
  using PermTileK = decltype(_512{} / SizeOfBits_InRegs{});
  
  // Data type used by MMA instruction
  using ElementAMmaRaw = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB>());

  static_assert(!cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm120, BuilderScheduleTag> ||
                detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMmaRaw, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, true /*is_sparse*/>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  static constexpr bool Use2x4AtomLayout = cute::is_base_of_v<KernelScheduleAcc2x4Sm120, BuilderScheduleTag>;
  using AtomLayoutMNK = cute::conditional_t<Use2x4AtomLayout, 
                                            Layout<Shape<_2,_4,_1>>,
                                            Layout<Shape<_4,_2,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(
    cute::rr_blockscaled_sparse_op_selector_sm120<ElementA, ElementB, ElementAccumulator, ElementSF, SFVectorSize, UseMxf8f6f4>(),
    AtomLayoutMNK{},
    Tile<PermTileM, PermTileN, PermTileK>{}
  ));

  static constexpr int MMA_NSF = size<2>(typename TiledMma::AtomShape_MNK{}) / SFVectorSize;

  using SmemAllocTypeA = cute::conditional_t<UseMxf8f6f4, uint8_t, typename TiledMma::ValTypeA::raw_type>;
  using SmemAllocTypeB = cute::conditional_t<UseMxf8f6f4, uint8_t, typename TiledMma::ValTypeB>;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementASparsity = Int<ElementAMma::sparsity>;
  using SmemAllocTypeE = typename TiledMma::FrgTypeE::raw_type;
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementE = typename ElementEMma::raw_type;
  using ElementESparsity = Int<ElementEMma::sparsity>;
  using SmemAllocTypeSF = ElementSF;

  using GmemTiledCopyA = SM90_TMA_LOAD;
  using GmemTiledCopyB = SM90_TMA_LOAD;

  using GmemTiledCopySFA = SM90_TMA_LOAD;
  using GmemTiledCopySFB = SM90_TMA_LOAD;

  using GmemTiledCopyPairA = decltype(cute::make_tuple(GmemTiledCopyA{}, GmemTiledCopySFA{}));
  using GmemTiledCopyPairB = decltype(cute::make_tuple(GmemTiledCopyB{}, GmemTiledCopySFB{}));

  // Setup Config
  using Sm1xxSparseConfig = cutlass::Sm1xxGemmSparseConfig<ElementAMma, GmemLayoutATag, ElementEMma>;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

  using StrideA = TagToStrideA_t<GmemLayoutATag>;
  using LayoutA = decltype(Sm1xxSparseConfig::deduce_layoutA());
  using LayoutE = decltype(Sm1xxSparseConfig::deduce_layoutE());
  using LayoutSFA = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFA());
  using LayoutTupleA = decltype(cute::make_tuple(LayoutA{}, LayoutE{}, LayoutSFA{}, StrideA{}));

  using SmemLayoutAtomA = decltype(detail::sm120_rr_smem_selector_sparse<SmemAllocTypeA, decltype(size<2>(TileShape_MNK{})), ElementASparsity>());
  using SmemLayoutAtomB = decltype(detail::sm120_rr_smem_selector<SmemAllocTypeB, decltype(size<2>(TileShape_MNK{}))>());

  using SmemCopyAtomA = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_A<ElementA, ElementB, UseMxf8f6f4>()), sparse_elem<ElementAMma::sparsity,SmemAllocTypeA>>;
  using SmemCopyAtomB = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_B<ElementA, ElementB, UseMxf8f6f4>()), SmemAllocTypeB>;
  using SmemCopyAtomE = Copy_Atom<UniversalCopy<uint64_t>, sparse_elem<ElementEMma::sparsity,SmemAllocTypeE>>;
  using SmemCopyAtomAPair = decltype(make_tuple(SmemCopyAtomA{}, SmemCopyAtomE{}));

  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<SmemAllocTypeSF>, SmemAllocTypeSF>; // auto-vectorized LDS
  using SmemCopyAtomSFA = SmemCopyAtomSF;
  using SmemCopyAtomSFB = SmemCopyAtomSF;

  using SmemCopyAtomsA = decltype(cute::make_tuple(SmemCopyAtomA{}, SmemCopyAtomE{}, SmemCopyAtomSFA{}));
  using SmemCopyAtomsB = decltype(cute::make_tuple(SmemCopyAtomB{}, SmemCopyAtomSFB{}));

  // Construct SMEM layout for SF
  // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
  // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size 
  using Blk_MN    = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF    = typename Sm1xxBlkScaledConfig::Blk_SF; 
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

  // Construct TileShape for SFB load from GMEM to SMEM.
  // It is required to keep consistency with BlockScaled granularity defined in Sm1xxBlkScaledConfig.
  // So that TileShape for scaling factor needs to be defined as a multiple of Blk_MN.
  using TileShapeSf_MNK = decltype(make_shape(ceil_div(size<0>(TileShape_MNK{}), Blk_MN{}) * Blk_MN{},
                                           ceil_div(size<1>(TileShape_MNK{}), Blk_MN{}) * Blk_MN{}, 
                                           shape<2>(TileShape_MNK{})));

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

  using sSFB_shapeN       = decltype(prepend(size<1>(TileShapeSf_MNK{}) / Blk_MN{},   mnBasicBlockShape{}));
  using sSFB_strideN      = sSF_strideMN;
  using sSFB_strideK      = decltype(prepend(make_stride(Int<MMA_NSF>{},   size<1>(TileShapeSf_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
  using sSFB_shape        = decltype(make_shape(  sSFB_shapeN{},   sSF_shapeK{}));
  using sSFB_stride       = decltype(make_stride(sSFB_strideN{}, sSFB_strideK{}));
  using SmemLayoutAtomSFB = decltype(make_layout(  sSFB_shape{},  sSFB_stride{}));

  using SmemLayoutAtomsA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));
  using SmemLayoutAtomsB = decltype(cute::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));

  // Get stages from shared memory usage. 
  static constexpr auto PipelineStages = detail::sm120_compute_stage_count_or_override_blockscaled_sparse<
    detail::sm120_smem_capacity_bytes, sparse_elem<ElementAMma::sparsity, SmemAllocTypeA>,
    SmemAllocTypeB, ElementEMma, TileShape_MNK, SmemLayoutAtomSFA, SmemLayoutAtomSFB>(StageCountType{});
  static constexpr uint32_t PipelineStagesA = get<0>(PipelineStages);
  static constexpr uint32_t PipelineStagesB = get<1>(PipelineStages);
  // In normal case, when A/B with same stages, E can be kept in SMEM, with A/B stages.
  // When A/B is with different stages, it is design for keeping E in GMEM/L2 to saving
  //    SMEM usage. StageE is defined to 0 for this case.
  static constexpr uint32_t PipelineStagesE = PipelineStagesA == PipelineStagesB ? PipelineStagesA : 0;

  static constexpr uint32_t SchedulerPipelineStageCount = 2;

  // Choose dispatch policy based on different required stages.
  using DispatchPolicy = MainloopSm120TmaWarpSpecializedSparseBlockScaled<PipelineStagesA,
                                                                          PipelineStagesB,
                                                                          PipelineStagesE,
                                                                          SchedulerPipelineStageCount,
                                                                          ClusterShape_MNK>;

  using StridePairB = decltype(cute::make_tuple(cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>{}, Sm1xxBlkScaledConfig::deduce_layoutSFB()));

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      cute::tuple<ElementA, ElementSF>,
      LayoutTupleA,
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
