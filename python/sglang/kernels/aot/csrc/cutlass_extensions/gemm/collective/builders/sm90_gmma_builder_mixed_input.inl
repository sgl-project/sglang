/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <class StageCountType>
struct stage_count_traits;

template <int Stages>
struct stage_count_traits<StageCount<Stages>> {
  static constexpr bool is_fixed = true;
  static constexpr int stages = Stages;
  static constexpr int carveout_bytes = 0;
};

template <int CarveoutBytes>
struct stage_count_traits<StageCountAutoCarveout<CarveoutBytes>> {
  static constexpr bool is_fixed = false;
  static constexpr int stages = 0;
  static constexpr int carveout_bytes = CarveoutBytes;
};

template <
    int CapacityBytes,
    class ElementA,
    class ElementB,
    class ElementScale,
    class ElementZero,
    class TileShapeMNK,
    int Alignment = 128,
    int CarveoutBytes>
constexpr int compute_stage_count_or_override_folded_weight_scale(StageCountAutoCarveout<CarveoutBytes>) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto s_bits = cute::sizeof_bits_v<ElementScale>;
  constexpr auto z_bits = get_bits_for_possibly_void_element<ElementZero>();
  constexpr int scale_group_size = DefaultWeightScaleGroupSize<ElementA>::value;

  static_assert(
      (size<2>(TileShapeMNK{}) % scale_group_size) == 0,
      "Folded weight-scale storage requires TileK to cover complete scale groups.");

  constexpr auto scale_bytes =
      cutlass::bits_to_bytes(s_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{}) / scale_group_size);
  constexpr auto zero_bytes = cutlass::bits_to_bytes(z_bits * size<0>(TileShapeMNK{}));
  static_assert(scale_bytes % 16 == 0, "Folded weight-scale bulk copy must be at least 16B aligned.");
  static_assert(zero_bytes % 128 == 0, "Zero bytes must be a multiple of 128");

  constexpr int stage_bytes_ = cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
                               cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
                               scale_bytes + zero_bytes;
  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, Alignment) + static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(CarveoutBytes, Alignment);
  constexpr int capacity_bytes = CapacityBytes / Alignment * Alignment;
  constexpr int computed_stage_count = (capacity_bytes - carveout_bytes) / stage_bytes;
  return computed_stage_count < 2 ? 2 : computed_stage_count;
}

}  // namespace detail

// GMMA_TMA_WS_RS
template <
    class ElementA_,
    class GmemLayoutATag_,
    int AlignmentA,
    class ElementB_,
    class GmemLayoutBTag_,
    int AlignmentB,
    class ElementAccumulator,
    class TileShape_MNK,
    class ClusterShape_MNK,
    class StageCountType,
    class KernelScheduleType,
    MixedInputScaleMode ScaleMode>
struct CollectiveBuilderMixedInput<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA_,
    GmemLayoutATag_,
    AlignmentA,
    ElementB_,
    GmemLayoutBTag_,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    ScaleMode,
    cute::enable_if_t<
        (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
         cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
         cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative> ||
         cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperative> ||
         cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedPingpong>) &&
        (detail::is_use_rmem_A<ElementA_, GmemLayoutATag_, ElementB_, GmemLayoutBTag_>() ||
         // ConvertAndScale and ConvertAndScaleWithZero
         cute::is_tuple<ElementA_>::value || cute::is_tuple<ElementB_>::value ||
         // DirectConvert
         sizeof_bits<ElementA_>::value != sizeof_bits<ElementB_>::value)>> {
 private:
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementA_>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementB_>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementA_>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementB_>;
  static constexpr bool NeitherIsTuple = !cute::is_tuple<ElementA_>::value && !cute::is_tuple<ElementB_>::value;
  // Determine if mixed input types.
  static constexpr bool IsMixedInput = cute::sizeof_bits_v<detail::deduce_mixed_width_dtype_t<0, ElementA_>> !=
                                       cute::sizeof_bits_v<detail::deduce_mixed_width_dtype_t<0, ElementB_>>;
  static constexpr bool IsArrayOfPointersGemm = cute::is_any_of_v<
      KernelScheduleType,
      KernelPtrArrayTmaWarpSpecializedCooperative,
      KernelPtrArrayTmaWarpSpecializedPingpong>;
  static_assert(IsMixedInput || !IsArrayOfPointersGemm, "Only mixed input grouped RS GEMM is supported.");

 public:
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementA_>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementB_>;

  static_assert(
      !IsMixedInput || (cute::is_tuple<ElementA_>::value ^ cute::is_tuple<ElementB_>::value ||
                        (NeitherIsTuple && (sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value))),
      "Either A OR B must be a tuple or the widths of A and B must be different.");

  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;

  template <class T>
  static auto get_stride(T const& t) {
    if constexpr (not cute::is_layout<cute::remove_pointer_t<T>>::value) {
      return t;
    } else {
      if constexpr (cute::is_pointer_v<T>) {
        return &cute::stride(*t);
      } else {
        return cute::stride(t);
      }
    }
  }

  using GmemLayoutATag = decltype(get_stride(GmemLayoutATag_{}));
  using GmemLayoutBTag = decltype(get_stride(GmemLayoutBTag_{}));

  using ElementPairA =
      cute::conditional_t<IsMixedInput && IsANarrow && NeitherIsTuple, cute::tuple<ElementA>, ElementA_>;
  using ElementPairB =
      cute::conditional_t<IsMixedInput && !IsANarrow && NeitherIsTuple, cute::tuple<ElementB>, ElementB_>;

  static constexpr bool IsATransformed = cute::is_tuple<ElementPairA>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(
      detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
      "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
  // If A is scaled, then we don't need to swap. Otherwise, we must ensure B goes to rmem and we must swap the
  // operands.
  static constexpr bool SwapAB =
      IsMixedInput ? !IsATransformed : detail::is_swapAB<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>();
  static constexpr bool IsWarpSpecializedTransposeB =
      detail::is_warpspecialized_transpose_B<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag, KernelScheduleType>();
  static_assert(!IsMixedInput || !IsWarpSpecializedTransposeB, "Mixed input GEMM does not support WS transpose B.");

  // When we relax the above assertion, we must handle setting the tile mma GmmaMajorB correctly.
  static constexpr cute::GMMA::Major TiledMmaGmmaMajorB = SwapAB ? GmmaMajorA : GmmaMajorB;

  // For fp32 types, map to tf32 MMA value type.
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  // Handle mixed dtypes and MMA.
  using RealElementA = cute::conditional_t<SwapAB, ElementBMma, ElementAMma>;
  using RealElementB = cute::conditional_t<SwapAB, ElementAMma, ElementBMma>;
  using RealElementAMma = cute::conditional_t<IsMixedInput, RealElementB, RealElementA>;
  // Always the same for element B.
  using RealElementBMma = RealElementB;

  static_assert(
      !IsMixedInput || TiledMmaGmmaMajorB == GMMA::Major::K || sizeof_bits<RealElementB>::value == 16,
      "Mixed input GEMM does not support MN major layout except for 16bit");

  using AtomLayoutMNK = cute::conditional_t<
      cute::is_any_of_v<
          KernelScheduleType,
          KernelTmaWarpSpecializedCooperative,
          KernelPtrArrayTmaWarpSpecializedCooperative>,
      Layout<Shape<_2, _1, _1>>,
      Layout<Shape<_1, _1, _1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<
          RealElementAMma,
          RealElementBMma,
          ElementAccumulator,
          TileShape_MNK,
          GMMA::Major::K,
          GMMA::Major::K>(),
      AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::rs_smem_selector<
                                   GmmaMajorA,
                                   ElementAMma,
                                   decltype(cute::get<0>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{})),
                                   IsWarpSpecializedTransposeB>());
  using SmemLayoutAtomB = decltype(detail::rs_smem_selector<
                                   GmmaMajorB,
                                   ElementBMma,
                                   decltype(cute::get<1>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{})),
                                   IsWarpSpecializedTransposeB>());

  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutAtomA{});
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutAtomB{});
  static constexpr int SmemAlignment = static_cast<int>(cute::max(SmemAlignmentA, SmemAlignmentB));

  // Handle mixed dtype array GEMM's size of tensor map storage.
  static constexpr size_t TensorMapStorage = sizeof(cute::TmaDescriptor) * size_t(IsMixedInput) * 4;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage);
  static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr bool UseFusedE8M0PreMmaScale = ScaleMode == MixedInputScaleMode::kPreMmaE8M0;

  static constexpr int PipelineStages = [] {
    if constexpr (detail::stage_count_traits<StageCountType>::is_fixed) {
      return detail::stage_count_traits<StageCountType>::stages;
    } else if constexpr (IsMixedInput) {
      if constexpr (IsArrayOfPointersGemm) {
        if constexpr (UseFusedE8M0PreMmaScale) {
          return detail::compute_stage_count_or_override_folded_weight_scale<
              Sm90ReducedSmemCapacityBytes,
              RealElementA,
              RealElementB,
              ElementScale,
              ElementZero,
              TileShape_MNK,
              SmemAlignment>(StageCountType{});
        } else {
          return detail::compute_stage_count_or_override_single_affine_transformed_input<
              Sm90ReducedSmemCapacityBytes,
              RealElementA,
              RealElementB,
              ElementScale,
              ElementZero,
              TileShape_MNK,
              detail::stage_count_traits<StageCountType>::carveout_bytes,
              SmemAlignment>(StageCountType{});
        }
      } else {
        return detail::compute_stage_count_or_override_single_affine_transformed_input<
            detail::sm90_smem_capacity_bytes,
            RealElementA,
            RealElementB,
            ElementScale,
            ElementZero,
            TileShape_MNK,
            detail::stage_count_traits<StageCountType>::carveout_bytes,
            SmemAlignment>(StageCountType{});
      }
    } else {
      return detail::compute_stage_count_or_override<
          detail::sm90_smem_capacity_bytes,
          ElementAMma,
          ElementBMma,
          TileShape_MNK,
          detail::stage_count_traits<StageCountType>::carveout_bytes,
          SmemAlignment>(StageCountType{});
    }
  }();

  static_assert(
      !UseFusedE8M0PreMmaScale ||
          (IsArrayOfPointersGemm && IsATransformed && cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
           cute::is_same_v<ElementB, cutlass::float_e4m3_t>),
      "Pre-MMA E8M0 scale mode is only implemented for grouped MXFP4 weight x FP8 activation.");

  using ArrayMixedInputDispatchPolicy = cute::conditional_t<
      UseFusedE8M0PreMmaScale,
      MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInputPreScale<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
      MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

  using DispatchPolicy = cute::conditional_t<
      IsMixedInput,
      cute::conditional_t<
          IsArrayOfPointersGemm,
          ArrayMixedInputDispatchPolicy,
          MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<PipelineStages, ClusterShape_MNK, KernelScheduleType>>,
      MainloopSm90TmaGmmaRmemAWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

  using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::AutoVectorizingCopy, ElementA>>;
  using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::AutoVectorizingCopy, ElementB>, void>;

  // We pack the scale data with the operand that will be optionally scaled and converted before MMA.
  using StrideA = cute::conditional_t<
      cute::is_layout<cute::remove_pointer_t<GmemLayoutATag_>>::value,
      GmemLayoutATag_,
      TagToStrideA_t<GmemLayoutATag>>;
  using StrideB = cute::conditional_t<
      cute::is_layout<cute::remove_pointer_t<GmemLayoutBTag_>>::value,
      GmemLayoutBTag_,
      TagToStrideB_t<GmemLayoutBTag>>;

  using CollectiveOp = CollectiveMmaArrayMixedInput<
      DispatchPolicy,
      TileShape_MNK,
      ElementPairA,
      StrideA,
      ElementPairB,
      StrideB,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity>;

  static_assert(
      SmemAlignment == static_cast<int>(cute::max(CollectiveOp::SmemAlignmentA, CollectiveOp::SmemAlignmentB)));
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
