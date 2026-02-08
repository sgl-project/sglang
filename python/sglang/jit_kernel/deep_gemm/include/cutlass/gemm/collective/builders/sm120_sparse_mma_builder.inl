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
sm120_compute_stage_count_or_override_sparse(StageCount<stages> stage_count) {
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
  int carveout_bytes
>
constexpr auto
sm120_compute_stage_count_or_override_sparse(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For F6/F4 sub-bytes, ElementAMma will be passed in as sparse_elem<uint8_t, Sparsity>, and
  //    ElementB will be passed in as uint8_t
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A,B,E (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one MainloopPipeline = (CollectiveMma::SharedStorage::PipelineStorage = PipelineTmaSparseUmmaAsync, three arrive-wait barrier)
  constexpr auto mainloop_pipeline_bytes = sizeof(cutlass::arch::ClusterBarrier) * 3;
  // a_bits, e_bits already consider sparsity through `sizeof_bits(ElementAMma)
  // NOTE: sizeof_bits<sparse_elem<>> return integral_ratio instead of size_t
  constexpr auto a_bits = cute::sizeof_bits_v<ElementAMma>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto e_bits = cute::sizeof_bits_v<ElementEMma>;
  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(e_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes);

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
    static_cast<int>(mainloop_pipeline_bytes);

  constexpr int stages_b15 = (CapacityBytes - carveout_bytes) / stage_bytes_b15;
  if constexpr (stages_b15 >= 2) {
    return cute::make_tuple(stages_b15, 3);
  }

  return cute::make_tuple(stages_b15, stages_b15);
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
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
    arch::OpClassSparseTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      (not cute::is_tuple_v<ElementA>   && not cute::is_tuple_v<ElementB> &&
       not cute::is_complex_v<ElementA> && not cute::is_complex_v<ElementB> &&
       not cute::is_sparse_v<ElementA>) &&
      // Sparse Gemm
      (cute::is_base_of_v<KernelScheduleSparseGemmSm120, BuilderScheduleTag> ||
       cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>) &&
      // Alignment check
      detail::sm1xx_sparse_gemm_is_aligned<ElementA, AlignmentA, GmemLayoutATag, ElementB, AlignmentB, BuilderScheduleTag>()>>
{

  static_assert(detail::is_sm10x_f8f6f4_element<ElementA>() && detail::is_sm10x_f8f6f4_element<ElementB>(),
                "SM120 Sparse TmaWarpSpecialized builder currently only supports F8F6F4 MMA.");
  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(cute::is_static_v<ClusterShape_MNK>, "Cluster has to be static");
  static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch");

  static_assert(cute::is_same_v<GmemLayoutATag, cutlass::layout::RowMajor>, "LayoutA must be K major.");
  static_assert(cute::is_same_v<GmemLayoutBTag, cutlass::layout::ColumnMajor>, "LayoutB must be K major.");

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();

  using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
  using AtomLayoutMNK =  Layout<Shape<_4,_2,_1>>;
  using PermTileN = decltype(cute::min(size<1>(TileShape_MNK{}),  _32{}));

  // Data type used by MMA instruction
  using ElementAMmaRaw = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB>());

  static_assert(detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMmaRaw, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, true /*IsSparse*/>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  // Setup TiledMma
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::rr_sparse_op_selector_sm120<ElementA, ElementB, ElementAccumulator>(),
    AtomLayoutMNK{},
    Tile<PermTileM, PermTileN, _64>{}
  ));

  // DType check
  static constexpr bool UseF8f6f4 = cute::is_base_of_v<KernelScheduleSparseF8f6f4Sm120, BuilderScheduleTag> || 
                                    cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>;
  static_assert(UseF8f6f4, "Non-blockscaled collective builder only supports F8F6F4 MMA.\n");

  // Element type
  using SmemAllocTypeA = cute::conditional_t<UseF8f6f4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<UseF8f6f4, uint8_t, typename TiledMma::ValTypeB>;
  using SmemAllocTypeE = typename TiledMma::FrgTypeE::raw_type;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementASparsity = Int<ElementAMma::sparsity>;
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementESparsity = Int<ElementEMma::sparsity>;
  using ElementE = typename ElementEMma::raw_type;

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  // Setup SparseConfig
  using Sm1xxSparseConfig = cutlass::Sm1xxGemmSparseConfig<ElementAMma, GmemLayoutATag, ElementEMma>;
  using LayoutA = decltype(Sm1xxSparseConfig::deduce_layoutA());
  using LayoutE = decltype(Sm1xxSparseConfig::deduce_layoutE());
  using LayoutPairAE = decltype(cute::make_tuple(LayoutA{}, LayoutE{}, TagToStrideA_t<GmemLayoutATag>{}));

  // Setup LayoutAtom and CopyAtom
  using SmemLayoutAtomA = decltype(detail::sm120_rr_smem_selector_sparse<SmemAllocTypeA, decltype(size<2>(TileShape_MNK{})), ElementASparsity>());
  using SmemLayoutAtomB = decltype(detail::sm120_rr_smem_selector<SmemAllocTypeB, decltype(size<2>(TileShape_MNK{}))>());

  // Setup Stages and DispatchPolicy
  static constexpr uint32_t SchedulerPipelineStageCount = 2;
  static constexpr auto PipelineStages = detail::sm120_compute_stage_count_or_override_sparse<
                    detail::sm120_smem_capacity_bytes, sparse_elem<ElementAMma::sparsity, SmemAllocTypeA>, 
                    SmemAllocTypeB, ElementEMma, TileShape_MNK>(StageCountType{});
  static constexpr uint32_t PipelineStagesA = get<0>(PipelineStages);
  static constexpr uint32_t PipelineStagesB = get<1>(PipelineStages);
  // In normal case, when A/B with same stages, E can be kept in SMEM, with A/B stages.
  // When A/B is with different stages, it is design for keeping E in GMEM/L2 to saving
  //    SMEM usage. StageE is defined to 0 for this case.
  static constexpr uint32_t PipelineStagesE = PipelineStagesA == PipelineStagesB ? PipelineStagesA : 0;

  using DispatchPolicy = MainloopSm120TmaWarpSpecializedSparse<PipelineStagesA,
                                                               PipelineStagesB,
                                                               PipelineStagesE,
                                                               SchedulerPipelineStageCount,
                                                               ClusterShape_MNK>;

  using SmemCopyAtomA = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_A<ElementA, ElementB, UseF8f6f4>()),
                        sparse_elem<ElementAMma::sparsity,SmemAllocTypeA>>;
  using SmemCopyAtomB = Copy_Atom<decltype(detail::sm120_rr_smem_copy_selector_B<ElementA, ElementB, UseF8f6f4>()), SmemAllocTypeB>;
  using SmemCopyAtomE = Copy_Atom<UniversalCopy<uint64_t>,
                        sparse_elem<ElementEMma::sparsity,SmemAllocTypeE>>;
  using SmemCopyAtomAPair = decltype(make_tuple(SmemCopyAtomA{}, SmemCopyAtomE{}));

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      LayoutPairAE,
      ElementB,
      cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomAPair,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
