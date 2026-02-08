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

#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes_, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity in gemm of blockwise/groupwise scale.
template<int capacity_bytes_, class ElementA, class ElementB, class ElementBlockScale, class TileShapeMNK, int ScaleMsPerTile, int ScaleNsPerTile, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_with_blockwise_scale(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto scale_bits = cute::sizeof_bits_v<ElementBlockScale>;
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(scale_bits * ScaleMsPerTile) + // scale of tensor A
    cutlass::bits_to_bytes(scale_bits * ScaleNsPerTile);  // scale of tensor B

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix), or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class ElementScale, class ElementZero, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input(StageCount<stages> stage_count) {
  return stages;
}

template <class Element>
constexpr int get_bits_for_possibly_void_element() {
  if constexpr (cute::is_same_v<Element, void>) {
    return 0;
  }
  else {
    return sizeof_bits<Element>::value;
  }
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix), or overrides with manual count.
template<int capacity_bytes_, class ElementA, class ElementB, class ElementScale, class ElementZero, class TileShapeMNK, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input(StageCountAutoCarveout<carveout_bytes_> stage_count) {

  // 32 bytes to account for barriers etc.
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr int scale_zero_k_tile = 1;
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto s_bits = get_bits_for_possibly_void_element<ElementScale>();
  constexpr auto z_bits = get_bits_for_possibly_void_element<ElementZero>();

  constexpr auto scale_bytes = cutlass::bits_to_bytes(s_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  constexpr auto zero_bytes  = cutlass::bits_to_bytes(z_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  static_assert(scale_bytes % 128 == 0, "Scale bytes must be a multiple of 128");
  static_assert(zero_bytes  % 128 == 0, "Zero bytes must be a multiple of 128");

  // When scales are void, s_bits will be 0 so no smem will be allocated for scales.
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    scale_bytes + zero_bytes;

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB>
constexpr bool
is_swapAB(){
  constexpr bool IsInputSizeTwoBytes = is_input_size_two_bytes<ElementA, ElementB>();
  constexpr bool IsLayoutAkBmn = cutlass::gemm::detail::is_k_major_A<LayoutA>() &&
                                 cutlass::gemm::detail::is_mn_major_B<LayoutB>();
  constexpr bool SwapAB = !IsInputSizeTwoBytes && IsLayoutAkBmn;
  return SwapAB;
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class KernelScheduleType>
constexpr bool
is_warpspecialized_transpose_B(){
  constexpr bool IsInputSizeTwoBytes = is_input_size_two_bytes<ElementA, ElementB>();
  constexpr bool IsLayoutAmnBmn = cutlass::gemm::detail::is_mn_major_A<LayoutA>() &&
                                  cutlass::gemm::detail::is_mn_major_B<LayoutB>();
  constexpr bool IsWarpSpecialized = cute::is_base_of_v<KernelTmaWarpSpecialized, KernelScheduleType>                ||
                                     cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, KernelScheduleType>        ||
                                     cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, KernelScheduleType>     ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecialized, KernelScheduleType>            ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecializedPingpong, KernelScheduleType>    ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecializedCooperative, KernelScheduleType>;
  constexpr bool IsWarpSpecializedTransposeB = !IsInputSizeTwoBytes && IsLayoutAmnBmn && IsWarpSpecialized;
  return IsWarpSpecializedTransposeB;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS
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
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
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
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_any_of_v<KernelScheduleType,
                         KernelTmaWarpSpecialized,
                         KernelTmaWarpSpecializedCooperative,
                         KernelTmaWarpSpecializedPingpong,
                         KernelPtrArrayTmaWarpSpecializedCooperative,
                         KernelPtrArrayTmaWarpSpecializedPingpong>) &&
       not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  static constexpr bool IsArrayOfPointersGemm = (cute::is_any_of_v<KernelScheduleType,
                                                                   KernelPtrArrayTmaWarpSpecializedCooperative,
                                                                   KernelPtrArrayTmaWarpSpecializedPingpong>);
  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  static constexpr bool IsCooperative = cute::is_any_of_v<KernelScheduleType,
                                                          KernelTmaWarpSpecializedCooperative,
                                                          KernelPtrArrayTmaWarpSpecializedCooperative>;
  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
 
  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;
  static constexpr size_t SchedulerPipelineStorage = cute::is_pointer_v<TagToStrideA_t<GmemLayoutATag>> ? 
      sizeof(cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>) : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage + SchedulerPipelineStorage);
  static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<Sm90ReducedSmemCapacityBytes,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  /* For FP8 use a separate mainloop compared to other datatypes */
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
      cute::conditional_t<IsFP8Input,
          MainloopSm90ArrayTmaGmmaWarpSpecializedFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
          MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>
      >,
      cute::conditional_t<IsFP8Input,
          MainloopSm90TmaGmmaWarpSpecializedFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
          MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>
      >
  >;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

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
  class KernelScheduleType
>
struct CollectiveBuilder<
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
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecializedPingpong> ||
       cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType,  KernelPtrArrayTmaWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType,  KernelPtrArrayTmaWarpSpecializedPingpong>) && 
      (detail::is_use_rmem_A<ElementA_, GmemLayoutATag_, ElementB_, GmemLayoutBTag_>() ||
       // ConvertAndScale and ConvertAndScaleWithZero 
       cute::is_tuple<ElementA_>::value || cute::is_tuple<ElementB_>::value || 
       // DirectConvert
       sizeof_bits<ElementA_>::value != sizeof_bits<ElementB_>::value)>
> {

private:
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementA_>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementB_>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementA_>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementB_>;
  static constexpr bool NeitherIsTuple = !cute::is_tuple<ElementA_>::value && !cute::is_tuple<ElementB_>::value;
  // Determine if mixed input types.
  static constexpr bool IsMixedInput =
    cute::sizeof_bits_v<detail::deduce_mixed_width_dtype_t<0, ElementA_>> != cute::sizeof_bits_v<detail::deduce_mixed_width_dtype_t<0, ElementB_>>;
  static constexpr bool IsArrayOfPointersGemm = cute::is_any_of_v<KernelScheduleType,
                                                                  KernelPtrArrayTmaWarpSpecializedCooperative,
                                                                  KernelPtrArrayTmaWarpSpecializedPingpong>;
  static_assert(IsMixedInput || !IsArrayOfPointersGemm, "Only mixed input grouped RS GEMM is supported.");

public:
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementA_>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementB_>;

  static_assert(!IsMixedInput || (cute::is_tuple<ElementA_>::value ^ cute::is_tuple<ElementB_>::value ||
               (NeitherIsTuple && (sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value))),
    "Either A OR B must be a tuple or the widths of A and B must be different.");

  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;

  template<class T>
  static auto get_stride(T const& t) {
    if constexpr (not cute::is_layout<cute::remove_pointer_t<T>>::value) {
      return t;
    }
    else {
      if constexpr (cute::is_pointer_v<T>) {
        return &cute::stride(*t);
      }
      else {
        return cute::stride(t);
      }
    }
  }

  using GmemLayoutATag = decltype(get_stride(GmemLayoutATag_{}));
  using GmemLayoutBTag = decltype(get_stride(GmemLayoutBTag_{}));

  using ElementPairA = cute::conditional_t<IsMixedInput && IsANarrow && NeitherIsTuple, cute::tuple<ElementA>, ElementA_>;
  using ElementPairB = cute::conditional_t<IsMixedInput && !IsANarrow && NeitherIsTuple, cute::tuple<ElementB>, ElementB_>;

  static constexpr bool IsATransformed = cute::is_tuple<ElementPairA>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
  // If A is scaled, then we don't need to swap. Otherwise, we must ensure B goes to rmem and we must swap the operands.
  static constexpr bool SwapAB = IsMixedInput ? !IsATransformed : detail::is_swapAB<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>();
  static constexpr bool IsWarpSpecializedTransposeB = detail::is_warpspecialized_transpose_B<
      ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag, KernelScheduleType>();
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

  static_assert(!IsMixedInput || TiledMmaGmmaMajorB == GMMA::Major::K || sizeof_bits<RealElementB>::value == 16, 
      "Mixed input GEMM does not support MN major layout except for 16bit");

  using AtomLayoutMNK = cute::conditional_t<
      cute::is_any_of_v<KernelScheduleType, 
                        KernelTmaWarpSpecializedCooperative,
                        KernelPtrArrayTmaWarpSpecializedCooperative>,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
      RealElementAMma, RealElementBMma, ElementAccumulator, TileShape_MNK, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::rs_smem_selector<GmmaMajorA, ElementAMma,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());
  using SmemLayoutAtomB = decltype(detail::rs_smem_selector<GmmaMajorB, ElementBMma,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());

  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutAtomA{}); 
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutAtomB{});
  static constexpr int    SmemAlignment  = static_cast<int>(cute::max(SmemAlignmentA, SmemAlignmentB));

  // Handle mixed dtype array GEMM's size of tensor map storage.
  static constexpr size_t TensorMapStorage = sizeof(cute::TmaDescriptor) * size_t(IsMixedInput) * 4;
  static constexpr size_t SchedulerPipelineStorage = cute::is_pointer_v<TagToStrideA_t<GmemLayoutATag_>> ? 
      sizeof(cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>) : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage + SchedulerPipelineStorage);
  static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr int PipelineStages = IsMixedInput ?
      ( IsArrayOfPointersGemm ? 
        detail::compute_stage_count_or_override_single_affine_transformed_input<Sm90ReducedSmemCapacityBytes,
          RealElementA, RealElementB, ElementScale, ElementZero, TileShape_MNK, StageCountType::bytes, SmemAlignment>(StageCountType{}) : 
        detail::compute_stage_count_or_override_single_affine_transformed_input<detail::sm90_smem_capacity_bytes,
          RealElementA, RealElementB, ElementScale, ElementZero, TileShape_MNK, StageCountType::bytes, SmemAlignment>(StageCountType{})
      ) 
      : detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes,
          ElementAMma, ElementBMma, TileShape_MNK, StageCountType::bytes, SmemAlignment>(StageCountType{});
      
  using DispatchPolicy = cute::conditional_t<IsMixedInput,
      cute::conditional_t<IsArrayOfPointersGemm,
        MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<PipelineStages, ClusterShape_MNK, KernelScheduleType>, 
        MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<PipelineStages, ClusterShape_MNK, KernelScheduleType>>, 
        MainloopSm90TmaGmmaRmemAWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

  using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::AutoVectorizingCopy, ElementA>>;
  using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::AutoVectorizingCopy, ElementB>, void>;
  
  // We pack the scale data with the operand that will be optionally scaled and converted before MMA.
  using StrideA = cute::conditional_t<cute::is_layout<cute::remove_pointer_t<GmemLayoutATag_>>::value, GmemLayoutATag_, TagToStrideA_t<GmemLayoutATag>>;
  using StrideB = cute::conditional_t<cute::is_layout<cute::remove_pointer_t<GmemLayoutBTag_>>::value, GmemLayoutBTag_, TagToStrideB_t<GmemLayoutBTag>>;

  using CollectiveOp = CollectiveMma<
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
      cute::identity
    >;

  static_assert(SmemAlignment == static_cast<int>(cute::max(CollectiveOp::SmemAlignmentA, CollectiveOp::SmemAlignmentB)));
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_FP8_FAST_ACCUM_SS
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
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
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
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_any_of_v<KernelScheduleType,
                        KernelTmaWarpSpecializedFP8FastAccum,
                        KernelTmaWarpSpecializedPingpongFP8FastAccum,
                        KernelTmaWarpSpecializedCooperativeFP8FastAccum,
                        KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
                        KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Not meet TMA alignment requirement yet\n");
  static_assert(detail::is_input_fp8<ElementA, ElementB>(),
                "Only FP8 datatypes are compatible with these kernel schedules\n");
  // Dispatch TN fp8 kernels only to TMA warp specialized FP8 builder
  static_assert(!detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>(),
                 "Not supported for fp8 non-TN warp specialized kernels yet\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementA, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementB, GmemLayoutBTag>();

  static constexpr bool IsArrayOfPointersGemm = cute::is_any_of_v<KernelScheduleType,
                                                                   KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
                                                                   KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum>;

  static constexpr bool IsCooperative = cute::is_any_of_v<KernelScheduleType,
                                                          KernelTmaWarpSpecializedCooperativeFP8FastAccum,
                                                          KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum>;

  using AtomLayoutMNK = cute::conditional_t<IsCooperative, Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementA, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementB, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;
  static constexpr size_t SchedulerPipelineStorage = cute::is_pointer_v<TagToStrideA_t<GmemLayoutATag>> ? 
      sizeof(cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>) : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage + SchedulerPipelineStorage);
  static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<Sm90ReducedSmemCapacityBytes,
      ElementA, ElementB, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
      MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
      MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_SS
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
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
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
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelTma> &&
                     not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>()));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = MainloopSm90TmaGmma<PipelineStages, ClusterShape_MNK>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_CpAsync
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
  class KernelScheduleType
>
struct [[deprecated("Use one of KernelCpAsyncWarpSpecialized schedules instead")]]
CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
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
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelMultistage>>
> {
  // Map to warp-specialized kernels for better performance
  using CollectiveOp = typename CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
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
    KernelCpAsyncWarpSpecialized
  >::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_CpAsync_WS_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int   AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int   AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
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
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong>) &&
      not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::cp_async_min_alignment_bytes>(),
                "Minimum alignment required for cp.async is 4B.");

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementA, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementB, GmemLayoutBTag>();

  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative>,
      Layout<Shape<cute::Int<(size<0>(TileShape_MNK{}) < 128) ? 1 : 2>,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  static constexpr int NumLoadWarpGroups = cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ? 2 : 1;

  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * AlignmentA>;
  using GmemCopyAtomA = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeA>, ElementA>;
  using GmemTiledCopyA = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomA, NumThreadsPerWarpGroup * NumLoadWarpGroups, AlignmentA, TagToStrideA_t<GmemLayoutATag>,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * AlignmentB>;
  using GmemCopyAtomB = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeB>, ElementB>;
  using GmemTiledCopyB = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomB, NumThreadsPerWarpGroup * NumLoadWarpGroups, AlignmentB, TagToStrideB_t<GmemLayoutBTag>,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      detail::sm90_smem_capacity_bytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using DispatchPolicy = MainloopSm90CpAsyncGmmaWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
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

// GMMA_CpAsync_WS_RS
template <
  class ElementA,
  class GmemLayoutATag,
  int   AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int   AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
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
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong>) &&
      detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::cp_async_min_alignment_bytes>(),
                "Minimum alignment required for cp.async is 4B.");

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
  static constexpr bool SwapAB = detail::is_swapAB<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>();
  static constexpr bool IsWarpSpecializedTransposeB = detail::is_warpspecialized_transpose_B<
      ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag, KernelScheduleType>();

  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative>,
      Layout<Shape<cute::Int<(size<0>(TileShape_MNK{}) < 128) ? 1 : 2>,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

  static constexpr int NumLoadWarpGroups = 1;

  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * AlignmentA>;
  using GmemCopyAtomA = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeA>, ElementA>;
  using GmemTiledCopyA = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomA, NumThreadsPerWarpGroup * NumLoadWarpGroups, AlignmentA, TagToStrideA_t<GmemLayoutATag>,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * AlignmentB>;
  using GmemCopyAtomB = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeB>, ElementB>;
  using GmemTiledCopyB = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomB, NumThreadsPerWarpGroup * NumLoadWarpGroups, AlignmentB, TagToStrideB_t<GmemLayoutBTag>,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::rs_smem_selector<GmmaMajorA, ElementAMma,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());
  using SmemLayoutAtomB = decltype(detail::rs_smem_selector<GmmaMajorB, ElementBMma,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      detail::sm90_smem_capacity_bytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using DispatchPolicy = MainloopSm90CpAsyncGmmaRmemAWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::AutoVectorizingCopy, ElementA>>;
  using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::AutoVectorizingCopy, ElementB>, void>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA auto kernel schedule
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
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
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
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

using ExtractedElementA = detail::deduce_mixed_width_dtype_t<0, ElementA>;
using ExtractedElementB = detail::deduce_mixed_width_dtype_t<0, ElementB>;

static constexpr bool IsTmaCompatible = detail::is_aligned<
    ExtractedElementA, AlignmentA, ExtractedElementB, AlignmentB, detail::tma_alignment_bytes>();

// Users opt into scales via the builder by passing a tuple of Elements for the input that will be scaled. We detect
// scale support if ONLY one of the inputs have tuples to describe them.
static constexpr bool OnlyOneIsTuple = cute::is_tuple<ElementA>::value ^ cute::is_tuple<ElementB>::value;
static constexpr bool IsDifferentWidth = sizeof_bits<ExtractedElementA>::value != sizeof_bits<ExtractedElementB>::value;
static constexpr bool IsMixedWidthInput = IsDifferentWidth || (IsDifferentWidth && OnlyOneIsTuple);

#if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 1)))
  // Persistent schedules perform best for CUDA Toolkits with version >= 12.1
  // KernelTmaWarpSpecializedCooperative requires TileShape_M to be at least 128
  using KernelTmaWarpSpecializedSchedule = cute::conditional_t<size<0>(TileShape_MNK{}) == Int<64>{},
      KernelTmaWarpSpecializedPingpong, KernelTmaWarpSpecializedCooperative>;
#else
  using KernelTmaWarpSpecializedSchedule = KernelTmaWarpSpecialized;
#endif

  // Non-persistent schedule is a safer choice for CpAsync kernels due to register pressure
  using KernelCpAsyncWarpSpecializedSchedule = KernelCpAsyncWarpSpecialized;
  using KernelSchedule = cute::conditional_t<IsTmaCompatible, KernelTmaWarpSpecializedSchedule, KernelCpAsyncWarpSpecializedSchedule>;
  static_assert((cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedSchedule> && IsMixedWidthInput) || !IsMixedWidthInput, "Only TMA warp specialized kernels are supported for mixed width input.");
  using CollectiveOp = typename CollectiveBuilder<
      arch::Sm90,
      arch::OpClassTensorOp,
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
      KernelSchedule
    >::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS (BlockScaled Builders)
template <
  class ElementA,
  class GmemLayoutPairA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutPairB,
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
    ElementA,
    GmemLayoutPairA,
    AlignmentA,
    ElementB,
    GmemLayoutPairB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeFP8Blockwise> or
       cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise> or
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpongFP8Blockwise> or
       cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise>) and
      not detail::is_use_rmem_A<ElementA, GmemLayoutPairA, ElementB, GmemLayoutPairB>()
    >
> {
  using GmemLayoutATag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutPairA{}))>;
  using GmemLayoutSFATag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutPairA{}))>;
  using GmemLayoutBTag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutPairB{}))>;
  using GmemLayoutSFBTag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutPairB{}))>;

  static_assert(cute::depth(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 2 and 
                cute::depth(cute::remove_pointer_t<GmemLayoutSFBTag>{}) == 2, 
      "Expect SFA and SFB layout to be depth of two with shape ((SFVecMN, restMN),(SFVecK, restK), L)");
  static_assert(size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 
                size<1,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{}), 
      "SFA and SFB must have equivalent SF vector sizes along K");

  static constexpr auto ScaleGranularityM = size<0,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});
  static constexpr auto ScaleGranularityN = size<0,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{});
  static constexpr auto ScaleGranularityK = size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  static constexpr bool IsArrayOfPointersGemm = (
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, KernelScheduleType> ||
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, KernelScheduleType>);

  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();
  static_assert(IsFP8Input, "Warp Specialized gemm with FP8 Blockwise (Software) Scaling is only compatible with FP8 inputs version right now.");

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;
  using ElementBlockScale = ElementAccumulator;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  static constexpr bool IsCooperative = cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, KernelScheduleType> ||
                                        cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, KernelScheduleType>;

  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;
  // Reserve 128B for 8 stages of tile scheduling
  static constexpr size_t SchedulerPipelineStorage = cute::is_pointer_v<TagToStrideA_t<GmemLayoutATag>> ? 
      sizeof(cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>) : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage + SchedulerPipelineStorage);

  static constexpr int ScaleMsPerTile = size<0>(TileShape_MNK{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape_MNK{}) / ScaleGranularityN;
  static_assert((size<0>(TileShape_MNK{}) % ScaleGranularityM) == 0, "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert((size<1>(TileShape_MNK{}) % ScaleGranularityN) == 0, "FP8 scaling granularity must evenly divide tile shape along N.");

  static constexpr int PipelineStages = detail::compute_stage_count_with_blockwise_scale<detail::sm90_smem_capacity_bytes - KernelSmemCarveout,
      ElementAMma, ElementBMma, ElementBlockScale, TileShape_MNK, ScaleMsPerTile, ScaleNsPerTile>(StageCountType{});
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
    MainloopSm90ArrayTmaGmmaWarpSpecializedBlockwise<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
    MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      cute::tuple<TagToStrideA_t<GmemLayoutATag>, TagToStrideA_t<GmemLayoutSFATag>>,
      ElementB,
      cute::tuple<TagToStrideB_t<GmemLayoutBTag>, TagToStrideB_t<GmemLayoutSFBTag>>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
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
