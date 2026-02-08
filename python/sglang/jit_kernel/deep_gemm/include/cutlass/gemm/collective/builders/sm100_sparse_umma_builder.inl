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
  class ClusterShapeMNK,
  uint32_t AccumulatorPipelineStageCount,
  uint32_t SchedulerPipelineStageCount,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_sparse(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template <
  class ElementAMma,
  class ElementB,
  class ElementEMma,
  class TileShapeMNK,
  class ClusterShapeMNK,
  uint32_t AccumulatorPipelineStageCount,
  uint32_t SchedulerPipelineStageCount,
  int carveout_bytes
>
constexpr int
sm100_compute_stage_count_or_override_sparse(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For F8/F6/F4 sub-bytes, ElementAMma will be passed in as sparse_elem<uint8_t, Sparsity>, and
  //    ElementB will be passed in as uint8_t

  constexpr uint32_t KernelSmemCarveout = detail::Sm100SparseGemmTmaUmmaCarveout<
                                            ClusterShapeMNK,
                                            AccumulatorPipelineStageCount,
                                            SchedulerPipelineStageCount,
                                            detail::CLCResponseSize>::KernelSmemCarveout;

  // * Compute Stage
  constexpr bool is_hmma_two_kphase = (cutlass::bits_to_bytes(cute::sizeof_bits_v<typename ElementAMma::raw_type> * size<2>(TileShapeMNK{})) == 128) && 
                                      (cute::sizeof_bits_v<typename ElementAMma::raw_type> == 16 ||
                                       cute::sizeof_bits_v<typename ElementAMma::raw_type> == 32);

  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A,B,E (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one MainloopPipeline = (CollectiveMma::SharedStorage::PipelineStorage = PipelineTmaSparseUmmaAsync, two arrive-wait barrier)
  constexpr auto MainloopPipelineStorage_per_Stage_aligned = static_cast<int>(cutlass::round_up(sizeof(cutlass::arch::ClusterBarrier) * 2, 16));

  // a_bits, e_bits already consider sparsity through `sizeof_bits(ElementAMma)
  // NOTE: sizeof_bits<sparse_elem<>> return integral_ratio instead of size_t
  constexpr auto a_bits = cute::sizeof_bits_v<ElementAMma>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto e_bits = cute::sizeof_bits_v<ElementEMma>;
  constexpr int MainloopTensorStorage_per_Stage =
    cutlass::round_up(
      cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      cutlass::bits_to_bytes(e_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{}) * (is_hmma_two_kphase ? 2 : 1)),
    128);

  constexpr auto EpilogueSharedStorage = carveout_bytes;

  constexpr auto Stages = (cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout - EpilogueSharedStorage) / 
    (MainloopTensorStorage_per_Stage + MainloopPipelineStorage_per_Stage_aligned);

  return Stages;
}

template<
  class ElementAMmaRaw,
  class ElementBMma,
  class ElementAccumulator,
  class TileShape_MNK, // (MmaAtomShape_M, MmaAtomShape_N, CtaTileShapeK)
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB
>
constexpr auto
sm100_make_sparse_1sm_trivial_tiled_mma() {

  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 64 || M == 128, "Invalid TileShape_M.");

  // Do not allow a tiled MMA N mode > 1, as that is not reasonable.
  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N % 8 == 0 && N <= 256, "Invalid TileShape_N.");

  if constexpr    (cute::is_same_v<ElementAMmaRaw, cutlass::tfloat32_t>) {
    static_assert(cute::is_same_v<ElementAMmaRaw, ElementBMma>, "ElementA and ElementBMma must match.");
    return make_tiled_mma(cute::SM100_MMA_TF32_SS_SPARSE<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                     M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else if constexpr (cute::is_same_v<ElementAMmaRaw,     cutlass::half_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::bfloat16_t>) {
    static_assert(cute::is_same_v<ElementAMmaRaw, ElementBMma>, "ElementA and ElementBMma must match.");
    return make_tiled_mma(cute::SM100_MMA_F16BF16_SS_SPARSE<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                     M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else if constexpr (cute::is_same_v<ElementAMmaRaw,  int8_t> ||
                     cute::is_same_v<ElementAMmaRaw, uint8_t>) {
    return make_tiled_mma(cute::SM100_MMA_S8_SS_SPARSE<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                     M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else if constexpr (cute::is_same_v<ElementAMmaRaw, cutlass::float_e4m3_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::float_e5m2_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::type_erased_dynamic_float8_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::float_e2m3_unpacksmem_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::float_e3m2_unpacksmem_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::float_e2m1_unpacksmem_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t>) {
    return make_tiled_mma(cute::SM100_MMA_F8F6F4_SS_SPARSE<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                     M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMmaRaw>,
        "Unsupported configuration for SM100 collective builder.");
  }
}

template<
  class ElementAMmaRaw,
  class ElementBMma,
  class ElementAccumulator,
  class TileShape_MNK, // (MmaAtomShape_M, MmaAtomShape_N, CtaTileShapeK)
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB
>
constexpr auto
sm100_make_sparse_2sm_trivial_tiled_mma() {

  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 128 || M == 256, "Invalid TileShape_M.");

  // Do not allow a tiled MMA N mode > 1, as that is not reasonable.
  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N % 8 == 0 && N <= 256, "Invalid TileShape_N.");

  if constexpr     (cute::is_same_v<ElementAMmaRaw, cutlass::tfloat32_t>) {
    static_assert(cute::is_same_v<ElementAMmaRaw, ElementBMma>, "ElementA and ElementBMma must match.");
    return make_tiled_mma(cute::SM100_MMA_TF32_2x1SM_SS_SPARSE<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                           M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else if constexpr (cute::is_same_v<ElementAMmaRaw,     cutlass::half_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::bfloat16_t>) {
    static_assert(cute::is_same_v<ElementAMmaRaw, ElementBMma>, "ElementA and ElementBMma must match.");
    return make_tiled_mma(cute::SM100_MMA_F16BF16_2x1SM_SS_SPARSE<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                           M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else if constexpr (cute::is_same_v<ElementAMmaRaw, int8_t> ||
                     cute::is_same_v<ElementAMmaRaw, uint8_t>) {
    return make_tiled_mma(cute::SM100_MMA_S8_2x1SM_SS_SPARSE<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                           M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else if constexpr (cute::is_same_v<ElementAMmaRaw, cutlass::float_e4m3_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::float_e5m2_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::type_erased_dynamic_float8_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::float_e2m3_unpacksmem_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::float_e3m2_unpacksmem_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::float_e2m1_unpacksmem_t> ||
                     cute::is_same_v<ElementAMmaRaw, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t> ) {
    return make_tiled_mma(cute::SM100_MMA_F8F6F4_2x1SM_SS_SPARSE<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                           M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMmaRaw>,
        "Unsupported configuration for SM100 collective builder.");
  }
}

// MMA construction and partitioning that supports both dynamic and static cluster shape.
// Used in conjunction with make_tma_atom_(A|B)_sm100
// TileShape_MNK is always static and has shape (MmaAtomShapeM, MmaAtomShapeN, TileShapeK)
// ClusterShape_MNK can be dynamic or static.
template<
  class ElementAMmaRaw,
  class ElementBMma,
  class ElementAccumulator,
  class TileShape_MNK, // (MmaAtomShape_M, MmaAtomShape_N, TileShapeK)
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  class BuilderScheduleTag
>
constexpr auto
sm100_make_trivial_tiled_mma_sparse() {
  // MMA_2SM requested
  if constexpr (cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag>) {
    return sm100_make_sparse_2sm_trivial_tiled_mma<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                    TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
  }
  // MMA_1SM requested
  else if constexpr (cute::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag>) {
    return sm100_make_sparse_1sm_trivial_tiled_mma<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                   TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
  }
  // Auto scheduling requested
  else if constexpr (cute::is_same_v<BuilderScheduleTag, KernelScheduleAuto>) {
    // Static cluster
    if constexpr (cute::is_static_v<ClusterShape_MNK>) {
      // For MMA_2SM we need a cluster shape that is multiple of 2x1
      // and only M=128 and M=256 are supported, otherwise, fall back to MMA_1SM
      if constexpr (cute::get<0>(ClusterShape_MNK{}) % 2 == 0 &&
                    cute::get<0>(TileShape_MNK{}) % 128 == 0) {
        return sm100_make_sparse_2sm_trivial_tiled_mma<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                       TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
      }
      else {
        return sm100_make_sparse_1sm_trivial_tiled_mma<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                       TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
      }
    // Dynamic cluster shape means we cannot assume we can use 2SM MMA 
    }
    else {
        return sm100_make_sparse_1sm_trivial_tiled_mma<ElementAMmaRaw, ElementBMma, ElementAccumulator,
                                                       TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
    }
  }
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
    arch::Sm100,
    arch::OpClassSparseTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      (not cute::is_tuple_v<ElementA>   && not cute::is_tuple_v<ElementB> &&
       not cute::is_complex_v<ElementA> && not cute::is_complex_v<ElementB> &&
       not cute::is_sparse_v<ElementA>) &&
      // Sparse Gemm
      (cute::is_base_of_v<KernelScheduleSparseGemmSm100, BuilderScheduleTag> ||
       cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>) &&
      // Alignment check
      detail::sm1xx_sparse_gemm_is_aligned<ElementA, AlignmentA, GmemLayoutATag, ElementB, AlignmentB, BuilderScheduleTag>()>>
{
  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  // Data type used by MMA instruction
  using ElementAMmaRaw = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB>());

  static constexpr bool is_2sm = cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> ||
                        (not cute::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag> &&
                          not cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> &&
                          cute::is_static_v<ClusterShape_MNK> &&
                          cute::get<0>(ClusterShape_MNK{}) % 2 == 0 );

  static_assert(detail::sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMmaRaw, ElementBMma,
                                                                        TileShape_MNK, ClusterShape_MNK,
                                                                        GmemLayoutATag, GmemLayoutBTag, true /*is_sparse*/, is_2sm>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );

  using TiledMma = decltype(cutlass::gemm::collective::detail::sm100_make_trivial_tiled_mma_sparse<
                              ElementAMmaRaw, ElementBMma, ElementAccumulator,
                              TileShape_MNK, ClusterShape_MNK,
                              UmmaMajorA, UmmaMajorB, BuilderScheduleTag>());

  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaSparsity = Int<ElementAMma::sparsity>;
  using ElementEMma = typename TiledMma::ValTypeE;
  using AtomThrID = typename TiledMma::AtomThrID;
  using Sm1xxSparseConfig = cutlass::Sm1xxGemmSparseConfig<ElementAMma, GmemLayoutATag, ElementEMma>;

  using ElementAMmaRaw_SmemAllocType = cute::conditional_t<cute::sizeof_bits_v<ElementAMmaRaw> < 8, uint8_t, ElementAMmaRaw>;
  using ElementAMma_SmemAllocType = cute::sparse_elem<ElementAMmaSparsity{}, ElementAMmaRaw_SmemAllocType>;
  using ElementBMma_SmemAllocType = cute::conditional_t<cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

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

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector_sparse<
      UmmaMajorA, ElementAMmaRaw_SmemAllocType, BlockTileA_M, BlockTileA_K, ElementAMmaSparsity>());

  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(
      ClusterShape_MNK{}, AtomThrID{}));

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma_SmemAllocType, BlockTileB_N, BlockTileB_K>());


  // Calculate SMEM matrix A and B buffers' pipeline stages and the accumulator stages.
  static constexpr uint32_t AccumulatorNPerCta = cute::size<1>(TileShape_MNK{});
  static constexpr uint32_t AccumulatorPipelineStageCount = AccumulatorNPerCta > 224 ? 1 : 2;
  static constexpr uint32_t SchedulerPipelineStageCount = 2;

  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_sparse<
      ElementAMma_SmemAllocType,
      ElementBMma_SmemAllocType,
      ElementEMma,
      SmemTileShape,
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount>(StageCountType{});

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedSparse<
        PipelineStages,
        SchedulerPipelineStageCount,
        AccumulatorPipelineStageCount,
        ClusterShape_MNK>,
      TileShape_MNK,
      ElementA,
      LayoutPairAE,
      ElementB,
      cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      void,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      void,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
