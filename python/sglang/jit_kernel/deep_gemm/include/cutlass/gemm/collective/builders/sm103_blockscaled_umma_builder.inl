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

#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/detail/sm103_blockscaled_layout.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

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
sm103_compute_stage_count_or_override_blockscaled(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int carveout_bytes
>
constexpr auto
sm103_compute_stage_count_or_override_blockscaled(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For F8F6F4 MMA sub-bytes, ElementA/B will be passed in as uint8_t
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A and smem for B (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one MainloopPipeline = PipelineTmaUmmaAsync (CollectiveMma::SharedStorage::SharedStorage)
  // 3. smem for SFB and smem for SFB (CollectiveMma::SharedStorage::TensorStorage, independent of input size b.c. sizeof(sf) is fixed)
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto stage_sfa_bytes = size(filter_zeros(TileShapeSFA{}));
  constexpr auto stage_sfb_bytes = size(filter_zeros(TileShapeSFB{}));

  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes * 2 + stage_sfa_bytes + stage_sfb_bytes);
  
  constexpr int ab_buffer = (CapacityBytes - carveout_bytes) / stage_bytes;
  constexpr int sb_buffer = ab_buffer + (CapacityBytes - carveout_bytes - ab_buffer * stage_bytes) /  (mainloop_pipeline_bytes + stage_sfa_bytes + stage_sfb_bytes);
  return make_tuple(ab_buffer, sb_buffer);
}

template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class ElementSF,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  int SFVectorSize
>
constexpr auto
sm103_make_blockscaled_1sm_tiled_mma() { 
  using AtomLayout_MNK = Layout<ClusterShape_MNK>;
  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 128 || M == 256, "Invalid TileShape_M.");

  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N % 64 == 0 && N <= 256, "Invalid TileShape_N.");

  if constexpr (cute::is_same_v<ElementAMma, cutlass::float_e2m1_t> ||
                cute::is_same_v<ElementAMma, cutlass::type_erased_dynamic_float4_t>) {
    return make_tiled_mma(cute::SM103::SM103_MXF4_ULTRA_SS_VS<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                 M, N, SFVectorSize, UmmaMajorA, UmmaMajorB>{});
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMma>,
        "Unsupported configuration for SM103 collective builder.");
  }
}

template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class ElementSF,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  int SFVectorSize
>
constexpr auto
sm103_make_blockscaled_2sm_tiled_mma() {
  using AtomLayout_MNK = Layout<decltype(shape_div(ClusterShape_MNK{}, Shape<_2,_1,_1>{}))>;

  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 128 || M == 256, "Invalid TileShape_M.");

  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N % 64 == 0 && N <= 256, "Invalid TileShape_N.");

  if constexpr (cute::is_same_v<ElementAMma, cutlass::float_e2m1_t> ||
                cute::is_same_v<ElementAMma, cutlass::type_erased_dynamic_float4_t>) {
    return make_tiled_mma(cute::SM103::SM103_MXF4_ULTRA_2x1SM_SS_VS<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                       M, N, SFVectorSize, UmmaMajorA, UmmaMajorB>{});
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMma>,
        "Unsupported configuration for SM103 collective builder.");
  }
}


template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class ElementSF,
  class ClusterTileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  class BuilderScheduleTag
>
constexpr auto
sm103_make_blockscaled_tiled_mma() {
  constexpr uint32_t SFVectorSize = find_vector_size<BuilderScheduleTag>();

  // MMA_2SM requested
  if constexpr (cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag>) {
    return sm103_make_blockscaled_2sm_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                ClusterTileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, SFVectorSize>();
  }
  // MMA_1SM requested
  else if constexpr (cute::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag>) {
    return sm103_make_blockscaled_1sm_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                ClusterTileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, SFVectorSize>();
  }
  // Auto scheduling requested
  else if constexpr (cute::is_same_v<BuilderScheduleTag, KernelScheduleAuto>) {
    if constexpr (cute::get<0>(ClusterShape_MNK{}) % 2 == 0) {
      return sm103_make_blockscaled_2sm_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                  ClusterTileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, SFVectorSize>();
    }
    else {
      return sm103_make_blockscaled_1sm_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                  ClusterTileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, SFVectorSize>();
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMma>,
      "Unsupported policy for SM103 collective builder.");
  }
}

template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class ElementSF,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  uint32_t SFVectorSize,
  class BuilderScheduleTag,
  bool Is2SM
>
struct Sm103TrivialBlockscaledMma {};

template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class ElementSF,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  uint32_t SFVectorSize,
  class BuilderScheduleTag
>
struct Sm103TrivialBlockscaledMma< ElementAMma,
  ElementBMma,
  ElementAccumulator,
  ElementSF,
  TileShape_MNK,
  ClusterShape_MNK,
  UmmaMajorA,
  UmmaMajorB,
  SFVectorSize,
  BuilderScheduleTag,
  true /*Is2SM*/> {
    using type = decltype(sm103_make_blockscaled_2sm_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                                       TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, SFVectorSize>());
  };

template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class ElementSF,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  uint32_t SFVectorSize,
  class BuilderScheduleTag
>
struct Sm103TrivialBlockscaledMma< ElementAMma,
  ElementBMma,
  ElementAccumulator,
  ElementSF,
  TileShape_MNK,
  ClusterShape_MNK,
  UmmaMajorA,
  UmmaMajorB,
  SFVectorSize,
  BuilderScheduleTag,
  false /*Is2SM*/> {
    using type = decltype(sm103_make_blockscaled_1sm_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                                       TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, SFVectorSize>());
};

template <class ElementType>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm103_block_scale_input() {
   // Allowed input element datatype for block-scaling GEMM
   return ( cute::is_same_v<ElementType, cutlass::type_erased_dynamic_float4_t> ||
            cute::is_same_v<ElementType, cutlass::float_e2m1_t>);
}

template <class BlockScaleConfig, class MmaShapeA_MK>
constexpr
auto sm103_sfa_smem_atom_layout() {
  constexpr int SF_BUFFERS_PER_TILE_K = BlockScaleConfig::SFVecSize == 16 ? 4 : 2;
  auto mma_sfa_tiler = make_shape(get<0,0>(MmaShapeA_MK{})*get<1>(MmaShapeA_MK{}), get<0,1>(MmaShapeA_MK{}) * get<2>(MmaShapeA_MK{}) / Int<SF_BUFFERS_PER_TILE_K>{});
  return tiled_product(typename BlockScaleConfig::SfAtom{},
                            make_layout(shape_div(mma_sfa_tiler, product_each(shape(typename BlockScaleConfig::SfAtom{})))));
}

template <class BlockScaleConfig, class MmaShapeB_NK, int SFVecSize>
constexpr
auto sm103_sfb_smem_atom_layout() {
auto sSFB = [&]() {
    constexpr int MMA_N = get<0>(MmaShapeB_NK{});
    constexpr int NonPow2N = 192;
    constexpr int NonPow2N_RoundUp = 256;
    // If MMA_N is 192, we need to operate at MMA_N = 256 granularity for UTCCP to work for ScaleFactorB.
    // Both TMA and UTCCP will transfer scale factor B as if we have 256 columns in B matrix.
    constexpr int MMA_N_SFB = (MMA_N == NonPow2N) ? NonPow2N_RoundUp : MMA_N;
    constexpr int SF_BUFFERS_PER_TILE_K = BlockScaleConfig::SFVecSize == 16 ? 4 : 2;
    auto mma_sfb_tiler = make_shape(Int<MMA_N_SFB>{}, get<1>(MmaShapeB_NK{}) / Int<SF_BUFFERS_PER_TILE_K>{});
    if constexpr(Int<MMA_N>{} == Int<128>{}) {
      return tiled_product(typename BlockScaleConfig::SfAtom{},
                            make_layout(shape_div(mma_sfb_tiler,product_each(shape(typename BlockScaleConfig::SfAtom{})))));

    }
    else {
      using SfKMajorAtom256  = Layout< Shape< Shape<_32,_4,                                         _2>, Shape<Int<SFVecSize>, _4>>, 
                                      Stride<Stride<_16,_4, Int<get<1>(mma_sfb_tiler)/SFVecSize/4*512>>, Stride<           _0, _1>>>;
      return tiled_product(SfKMajorAtom256{},
                            make_layout(shape_div(mma_sfb_tiler,product_each(shape(SfKMajorAtom256{})))));
    }
  }();
  return sSFB;
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
    arch::Sm103,
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
      // Not paired input, Not Complex input
      (cute::is_tuple_v<ElementPairA>       && cute::is_tuple_v<ElementPairB> &&
       not cute::is_complex_v<ElementPairA> && not cute::is_complex_v<ElementPairB>) &&
      // Blockscaled Gemm
      (cute::is_base_of_v<KernelScheduleSm103BlockScaledGemm, BuilderScheduleTag> ||
       cute::is_base_of_v<KernelSchedulePtrArraySm103BlockScaledGemm, BuilderScheduleTag> ||
       cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>) &&
      // Alignment check
      detail::sm1xx_blockscaled_gemm_is_aligned<remove_cvref_t<decltype(get<0>(ElementPairA{}))>,
                                                AlignmentA,
                                                remove_cvref_t<decltype(get<0>(ElementPairB{}))>,
                                                AlignmentB,
                                                BuilderScheduleTag>()>>
{
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;

  static_assert(cute::is_tuple<ElementPairA>::value, "Expecting ElementPairA to be a tuple.");
  static_assert(cute::is_tuple<ElementPairB>::value, "Expecting ElementPairB to be a tuple.");

  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(cute::size<2>(TileShape_MNK{}) == _768{}, "TileShape_K should 768 for MMA kernels");

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();
  static_assert(cutlass::gemm::detail::is_k_major_A<GmemLayoutATag>() && cutlass::gemm::detail::is_k_major_B<GmemLayoutBTag>(), "Only K major inputs are supported");

  static_assert(cutlass::gemm::collective::detail::is_sm103_block_scale_input<ElementA>(), "Incorrect type for A matrix");  
  static_assert(cutlass::gemm::collective::detail::is_sm103_block_scale_input<ElementB>(), "Incorrect type for B matrix");
  
  static_assert(cute::is_same_v<ElementSF, cutlass::float_ue8m0_t> ||
                cute::is_same_v<ElementSF, cutlass::float_ue4m3_t>, "Incorrect scale factor type");

  // Data type used by MMA instruction
  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA, false /*UseQmma=false*/>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB, false /*UseQmma=false*/>());

  static constexpr uint32_t SFVectorSize = detail::find_vector_size<BuilderScheduleTag>();

  static constexpr bool is_2sm = cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> ||
                                  (cute::is_same_v<BuilderScheduleTag, KernelScheduleAuto> && 
                                  (cute::is_static_v<ClusterShape_MNK> && cute::get<0>(ClusterShape_MNK{}) % 2 == 0));

  using TiledMma = typename cutlass::gemm::collective::detail::Sm103TrivialBlockscaledMma<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                                                          TileShape_MNK, ClusterShape_MNK,
                                                                                          UmmaMajorA, UmmaMajorB, SFVectorSize, BuilderScheduleTag, is_2sm>::type;

  using AtomThrID = typename TiledMma::AtomThrID;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm103BlockScaledConfig<SFVectorSize>;

  using ElementAMma_SmemAllocType = uint8_t;
  //  ElementAMma;
  using ElementBMma_SmemAllocType = uint8_t;
  //  ElementBMma;

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
  using SmemLayoutAtomA = UMMA::Layout_K_SW128_Atom<ElementAMma_SmemAllocType>;
  // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
  // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size 
  static constexpr int MMA_M = cute::size<0>(TileShape_MNK{}) / cute::size(AtomThrID{});
  using SmemLayoutAtomSFA = decltype(detail::sm103_sfa_smem_atom_layout<Sm1xxBlkScaledConfig,MmaShapeA_MK>());
  using SmemLayoutAtomsA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));

  //
  // Construct SMEM layout（SmemLayoutAtom）for B and SFB
  //

  using SmemLayoutAtomB = UMMA::Layout_K_SW128_Atom<ElementBMma_SmemAllocType>;
  static constexpr int MMA_N = cute::size<1>(TileShape_MNK{});
  // If MMA_N is 192, we need to operate at MMA_N = 256 granularity for UTCCP to work for ScaleFactorB.
  // Both TMA and UTCCP will transfer scale factor B as if we have 256 columns in B matrix.
  using SmemLayoutAtomSFB = decltype(detail::sm103_sfb_smem_atom_layout<Sm1xxBlkScaledConfig,decltype(select<1,2>(TileShape_MNK{})),SFVectorSize>());
  using SmemLayoutAtomsB = decltype(cute::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));

  //
  // Construct Strides for A, SFA, B, and SFB
  //
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

  //
  // Others
  //

  static constexpr cutlass::sm103::detail::KernelPrefetchType PrefetchType = cute::is_base_of_v<KernelScheduleSm103BlockScaledMxNvf4UltraDisablePrefetch, BuilderScheduleTag> 
                                                                             || cute::is_base_of_v<KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraDisablePrefetch, BuilderScheduleTag>
                                                                                                                                                                         ? cutlass::sm103::detail::KernelPrefetchType::Disable :
                                                                                                                                                                           cutlass::sm103::detail::KernelPrefetchType::TmaPrefetch;                                                                            

  static constexpr uint32_t AccumulatorPipelineStageCount = (MMA_N == 256) ? 1 : 2;
  static constexpr uint32_t SchedulerPipelineStageCount = 3;

  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLCPipeline = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount>::SharedStorage);
  // LoadOrderBarrier = OrderedSequenceBarrier<1,2>
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * detail::CLCResponseSize;
  // CLC Throttle pipeline storage
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = AccumulatorPipelineStageCount * sizeof(uint32_t);
  // Tensormap Storage
  static constexpr bool IsArrayOfPointersGemm = cute::is_base_of_v<KernelSchedulePtrArraySm103BlockScaledGemm, BuilderScheduleTag>;
  static constexpr auto TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 4 /* for A, B, SFA and SFB */ : 0;
  // TMA Load Prefetch Storage
  static constexpr auto TmaPrefetchStorage = 0;
  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( AccumulatorPipelineStorage +
                                                               CLCPipelineStorage +
                                                               LoadOrderBarrierStorage +
                                                               CLCResponseStorage +
                                                               CLCThrottlePipelineStorage +
                                                               TmemDeallocStorage +
                                                               TmemBasePtrsStorage +
                                                               TensorMapStorage +
                                                               TmaPrefetchStorage);
  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute::Shape<Int<MMA_M>, Int<MMA_N/cute::size(AtomThrID{})>, _128>; // SmemAllocTypes are uint8_t. We always allocate 128bytes
  static constexpr auto PipelineStages = cutlass::gemm::collective::detail::sm103_compute_stage_count_or_override_blockscaled<
      Sm100ReducedSmemCapacityBytes, ElementAMma_SmemAllocType, ElementBMma_SmemAllocType, SmemTileShape, SmemLayoutAtomSFA, SmemLayoutAtomSFB>(StageCountType{});

  using DispatchPolicy = typename cute::conditional_t<IsArrayOfPointersGemm,
    cutlass::gemm::MainloopSm103ArrayTmaUmmaWarpSpecializedBlockScaled<
      get<0>(PipelineStages),
      get<1>(PipelineStages),
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape_MNK,
      PrefetchType
    >,
    cutlass::gemm::MainloopSm103TmaUmmaWarpSpecializedBlockScaled<
      get<0>(PipelineStages),
      get<1>(PipelineStages),
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape_MNK,
      PrefetchType
    >
  >;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementPairA,
      StridePairA,
      ElementPairB,
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
