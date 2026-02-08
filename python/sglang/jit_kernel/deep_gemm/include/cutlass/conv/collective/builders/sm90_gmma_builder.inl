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

#include "cutlass/conv/collective/builders/sm90_common.inl"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

}

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS_FPROP
template <
  conv::Operator ConvOp,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ConvOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecializedSm90> ||
                      cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecializedSm90Cooperative> ||
                      cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecializedSm90Pingpong>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(cutlass::gemm::collective::detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, cutlass::gemm::collective::detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  // For fprop, majorA = K,  major B = K;
  // For wgrad, majorA = MN, major B = MN;
  // For dgrad, majorA = K,  major B = MN;
  static constexpr cute::GMMA::Major GmmaMajorA =
    (ConvOp == conv::Operator::kWgrad) ? cute::GMMA::Major::MN : cute::GMMA::Major::K;
  static constexpr cute::GMMA::Major GmmaMajorB =
    (ConvOp == conv::Operator::kFprop) ? cute::GMMA::Major::K : cute::GMMA::Major::MN;

  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecializedSm90Cooperative>,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  // For wgrad kernel, tensor A uses tma tiled mode and tensor B uses tma im2col mode.
  using GmemTiledCopyA = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
      decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(cute::shape<1>(ClusterShape_MNK{}))),
      decltype(cutlass::conv::collective::detail::sm90_cluster_shape_to_im2col_tma_atom(cute::shape<1>(ClusterShape_MNK{})))>;
  using GmemTiledCopyB = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
      decltype(cutlass::conv::collective::detail::sm90_cluster_shape_to_im2col_tma_atom(cute::shape<0>(ClusterShape_MNK{}))),
      decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(cute::shape<0>(ClusterShape_MNK{})))>;

  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<cutlass::gemm::collective::detail::sm90_smem_capacity_bytes,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<PipelineStages>{}),
      Step<_2,_1,_3>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<PipelineStages>{}),
      Step<_2,_1,_3>{}));

  constexpr static int NumSpatialDimensions = cutlass::conv::collective::detail::gmem_layout_tags_to_spatial_dims<GmemLayoutA, GmemLayoutB>();

  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedImplicitGemm<
      ConvOp, PipelineStages, NumSpatialDimensions, ClusterShape_MNK, KernelScheduleType>;

  using CollectiveOp = CollectiveConv<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      ElementB,
      TiledMma,
      detail::Sm90ImplicitGemmTileTraits<GmemTiledCopyA, SmemLayoutA>,
      detail::Sm90ImplicitGemmTileTraits<GmemTiledCopyB, SmemLayoutB>
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA auto kernel schedule
template <
  conv::Operator ConvOp,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ConvOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

/*
#if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 1)))
  // Cooperative schedule performs best for CUDA Toolkits with version >= 12.1

  // For TileShape_M == 64, choosing KernelTmaWarpSpecialized as the KernelSchedule
  // Since KernelTmaWarpSpecializedCooperative requires TileShape_M to be at least 128
  using KernelWarpSpecializedSchedule = cute::conditional_t<size<0>(TileShape_MNK{}) == Int<64>{},
      KernelImplicitTmaWarpSpecializedSm90PingPong, KernelImplicitTmaWarpSpecializedSm90Cooperative>;
#else
  using KernelWarpSpecializedSchedule = KernelImplicitTmaWarpSpecializedSm90;
#endif
*/
  using KernelWarpSpecializedSchedule = KernelImplicitTmaWarpSpecializedSm90;

  using CollectiveOp = typename CollectiveBuilder<
      arch::Sm90,
      arch::OpClassTensorOp,
      ConvOp,
      ElementA,
      GmemLayoutA,
      AlignmentA,
      ElementB,
      GmemLayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape_MNK,
      ClusterShape_MNK,
      StageCountType,
      KernelWarpSpecializedSchedule
    >::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
