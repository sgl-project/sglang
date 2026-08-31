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

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass_extensions/detail/collective/mixed_input_utils.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <int Stages, class ClusterShape, class KernelSchedule_, class TileShape_,
          class ElementAOptionalTuple, class StrideA_, class ElementBOptionalTuple, class StrideB_,
          class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_, class SmemCopyAtomA_,
          class TransformA_, class GmemTiledCopyB_, class SmemLayoutAtomB_, class SmemCopyAtomB_,
          class TransformB_>
struct CollectiveMmaArrayMixedInput<MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInputPreScale<
                                        Stages, ClusterShape, KernelSchedule_>,
                                    TileShape_, ElementAOptionalTuple, StrideA_,
                                    ElementBOptionalTuple, StrideB_, TiledMma_, GmemTiledCopyA_,
                                    SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_,
                                    SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {
 public:
  enum class ConversionMode { DirectConvert, ConvertAndScale, ConvertAndScaleWithZero };

  //
  // Type Aliases
  //
  using DispatchPolicy =
      MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInputPreScale<Stages, ClusterShape,
                                                                KernelSchedule_>;
  using TileShape = TileShape_;
  using KernelSchedule = KernelSchedule_;

 private:
  template <class T>
  friend struct detail::MixedGroupedGemmInputUtils;
  using CollectiveType =
      CollectiveMmaArrayMixedInput<DispatchPolicy, TileShape_, ElementAOptionalTuple, StrideA_,
                                   ElementBOptionalTuple, StrideB_, TiledMma_, GmemTiledCopyA_,
                                   SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_,
                                   SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_>;
  using Utils = detail::MixedGroupedGemmInputUtils<CollectiveType>;

  //
  // Type Aliases
  //
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

 public:
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;
  static constexpr bool HasWeightScale = !cute::is_void_v<ScaleA>;
  static constexpr bool HasZeroB = !cute::is_void_v<ZeroB>;
  static_assert(
      IsANarrow,
      "SM90 mixed-input mainloop expects the first operand to be the narrow transformed weight.");
  static_assert(HasWeightScale, "The transformed weight operand must carry mixed-input scale.");
  static_assert(!HasZeroB, "Activation operand must not carry zero-point.");
  static constexpr bool IsATransformed = true;
  using ElementScale = ScaleA;
  using ElementZero = ZeroA;

  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  static constexpr bool IsMXFP4 = cute::is_same_v<ElementA, cutlass::float_e2m1_t>;
  static constexpr bool HasActivationScale = !cute::is_void_v<ScaleB> &&
                                             cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
                                             cute::is_same_v<ElementB, cutlass::float_e4m3_t>;
  using ElementActivationScale = cute::conditional_t<HasActivationScale, ScaleB, void>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when
  // the scale / zero is void.
  using NonVoidElementScale =
      cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;
  using NonVoidElementActivationScale =
      cute::conditional_t<cute::is_void_v<ElementActivationScale>, cutlass::float_ue8m0_t,
                          ElementActivationScale>;
  // The GEMM kernel consumes weight scales in Ktile-major, MN-contiguous form.
  // MXFP8 activation scales stay in their raw M-major, K-contiguous form and
  // are loaded with a separate TMA descriptor.
  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>,
                                                 cute::Stride<_1, int64_t, int64_t>, StrideScale>;
  using StrideActivationScale = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  static_assert(
      (IsATransformed && (cutlass::gemm::detail::is_k_major<StrideA>() ||
                          is_layout<StrideA>::value || is_layout<InternalStrideA>::value)) ||
          (!IsATransformed && (cutlass::gemm::detail::is_k_major<StrideB>() ||
                               is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),
      "The transformed type must be K-major.");

  static_assert((IsATransformed && (sizeof(ElementB) == 2)) ||
                    (!IsATransformed && (sizeof(ElementA) == 2)) ||
                    ((cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value ||
                      is_layout<InternalStrideA>::value) &&
                     (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value ||
                      is_layout<InternalStrideB>::value)),
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(),
                "Scale tensor consumed by the GEMM kernel must be MN major.");

  static constexpr int ScalingGroupSize = detail::DefaultWeightScaleGroupSize<ElementA>::value;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using WeightScaleRawElement = NonVoidElementScale;
  using SmemCopyAtomScale = Copy_Atom<cute::AutoVectorizingCopy, NonVoidElementScale>;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = !IsATransformed;
  using SwappedStrideA = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using SwappedStrideB = cute::conditional_t<!SwapAB, StrideB, StrideA>;
  using InternalSwappedStrideA = cute::conditional_t<!SwapAB, InternalStrideA, InternalStrideB>;
  using InternalSwappedStrideB = cute::conditional_t<!SwapAB, InternalStrideB, InternalStrideA>;
  using SwappedSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using SwappedSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using SwappedSmemCopyAtomA = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using SwappedSmemCopyAtomB = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using ConvertedElementA =
      cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB =
      cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using SwappedTransformA = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using SwappedTransformB = cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;
  using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int NumProducerThreadEvents = 1;

  static constexpr int RawActScaleChunksPerTileK = size<2>(TileShape{}) / ScalingGroupSize;
  // Keep the derived value nonzero so the modulo static_assert below remains
  // well-formed; RawActScaleChunksPerTileK carries the actual validity check.
  static constexpr int ActScaleChunksPerTileK =
      HasActivationScale ? ((RawActScaleChunksPerTileK > 0) ? RawActScaleChunksPerTileK : 1) : 1;
  static constexpr int ActScaleTmaAlignmentChunks =
      HasActivationScale ? (128 / cutlass::sizeof_bits<NonVoidElementActivationScale>::value) : 1;
  static constexpr bool ActScaleTmaUsesMinWindow =
      ActScaleChunksPerTileK <= ActScaleTmaAlignmentChunks;
  static constexpr int ActScaleTmaChunks =
      ActScaleTmaUsesMinWindow ? ActScaleTmaAlignmentChunks : ActScaleChunksPerTileK;
  static_assert(!HasActivationScale || RawActScaleChunksPerTileK > 0,
                "Activation scale TMA requires TileShapeK to cover at least one scale group.");
  static_assert(
      !HasActivationScale ||
          (ActScaleTmaUsesMinWindow ? (ActScaleTmaAlignmentChunks % ActScaleChunksPerTileK == 0)
                                    : (ActScaleChunksPerTileK % ActScaleTmaAlignmentChunks == 0)),
      "Activation scale TileShapeK must divide or be a multiple of the 16B TMA scale window.");
  static constexpr int WeightScaleLogicalMPerFoldBlock = 64;
  static constexpr int WeightScaleLogicalKPerFoldBlock = 128;
  static constexpr int WeightScaleFoldedMPerFoldBlock = 16;
  static constexpr int WeightScaleMSlicesPerFoldBlock =
      WeightScaleLogicalMPerFoldBlock / WeightScaleFoldedMPerFoldBlock;
  static constexpr int WeightScaleScaleGroupsPerFoldBlock =
      WeightScaleLogicalKPerFoldBlock / ScalingGroupSize;
  static constexpr int WeightScalePhysicalColsPerFoldBlock =
      WeightScaleMSlicesPerFoldBlock * WeightScaleScaleGroupsPerFoldBlock;
  static constexpr int WeightScaleMBlocksPerTile =
      size<0>(TileShape{}) / WeightScaleLogicalMPerFoldBlock;
  static constexpr int WeightScaleKBlocksPerTile =
      size<2>(TileShape{}) / WeightScaleLogicalKPerFoldBlock;
  static_assert(size<0>(TileShape{}) % WeightScaleLogicalMPerFoldBlock == 0,
                "Folded weight scale requires TileShapeM to be a multiple of 64.");
  static_assert(size<2>(TileShape{}) % WeightScaleLogicalKPerFoldBlock == 0,
                "Folded weight scale requires TileShapeK to be a multiple of 128.");
  static_assert(WeightScaleLogicalMPerFoldBlock % WeightScaleFoldedMPerFoldBlock == 0,
                "Folded weight scale M dimension must evenly divide the logical M block.");
  static_assert(WeightScalePhysicalColsPerFoldBlock *
                        cutlass::sizeof_bits<WeightScaleRawElement>::value ==
                    128,
                "Folded weight scale must expose 16B per folded-M coordinate.");
  static constexpr int ScaleNRawElementsPerStage = size<1>(TileShape{}) * ActScaleTmaChunks;
  static constexpr int ScaleNElementsPerStage = size<1>(TileShape{});
  static constexpr int WeightScaleRawElementsPerFoldBlock =
      WeightScaleLogicalMPerFoldBlock * WeightScaleLogicalKPerFoldBlock / ScalingGroupSize;
  static constexpr int WeightScaleRawElementsPerStage =
      WeightScaleRawElementsPerFoldBlock * WeightScaleMBlocksPerTile * WeightScaleKBlocksPerTile;
  static constexpr uint32_t WeightScaleFoldBlockBytes = cutlass::bits_to_bytes(
      WeightScaleRawElementsPerFoldBlock * cutlass::sizeof_bits<WeightScaleRawElement>::value);
  static constexpr uint32_t WeightScaleBulkCopyBytes =
      WeightScaleFoldBlockBytes * WeightScaleKBlocksPerTile;
  static constexpr uint32_t WeightScaleTransactionBytes = cutlass::bits_to_bytes(
      WeightScaleRawElementsPerStage * cutlass::sizeof_bits<WeightScaleRawElement>::value);
  static_assert(WeightScaleBulkCopyBytes % 16 == 0,
                "Folded weight-scale bulk copy size must be 16B aligned.");

  using SmemLayoutAtomScale =
      Layout<Shape<decltype(cute::shape<0>(SwappedSmemLayoutAtomA{})), cute::Int<1>>>;
  using ScaleTileShape =
      decltype(make_shape(shape<0>(TileShape{}), shape<1>(SmemLayoutAtomScale{})));

  static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2,
                "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SwappedSmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2,
                "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SwappedSmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0,
                "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0,
                "SmemLayoutAtomScale must evenly divide tile k shape.");

  /// Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomA{}, select<0, 2>(TileShape{}), InternalSwappedStrideA{}));
  using SmemLayoutB = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomB{}, select<1, 2>(TileShape{}), InternalSwappedStrideB{}));

  // It is assumed that weight scales and zero-points share the same smem layout.
  using SmemLayoutScale = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<Stages>{}),
      cute::conditional_t<::cutlass::gemm::detail::is_major<0, NonVoidStrideScale>(),
                          Step<_2, _1, _3>, Step<_1, _2, _3>>{}));
  using SmemLayoutWeightScaleRaw =
      Layout<Shape<Int<WeightScalePhysicalColsPerFoldBlock>, Int<WeightScaleFoldedMPerFoldBlock>,
                   Int<WeightScaleMBlocksPerTile>, Int<WeightScaleKBlocksPerTile>, Int<Stages>>,
             Stride<_1, Int<WeightScalePhysicalColsPerFoldBlock>,
                    Int<WeightScaleFoldedMPerFoldBlock * WeightScalePhysicalColsPerFoldBlock *
                        WeightScaleKBlocksPerTile>,
                    Int<WeightScaleFoldedMPerFoldBlock * WeightScalePhysicalColsPerFoldBlock>,
                    Int<WeightScaleRawElementsPerStage>>>;
  using SmemLayoutWeightScaleExpanded = Layout<
      Shape<Shape<Int<WeightScaleFoldedMPerFoldBlock>, Int<WeightScaleMSlicesPerFoldBlock>,
                  Int<WeightScaleMBlocksPerTile>>,
            Shape<Int<ScalingGroupSize>,
                  Shape<Int<WeightScaleScaleGroupsPerFoldBlock>, Int<WeightScaleKBlocksPerTile>>>,
            Int<Stages>>,
      Stride<
          Stride<Int<WeightScalePhysicalColsPerFoldBlock>, Int<WeightScaleScaleGroupsPerFoldBlock>,
                 Int<WeightScaleFoldedMPerFoldBlock * WeightScalePhysicalColsPerFoldBlock *
                     WeightScaleKBlocksPerTile>>,
          Stride<_0, Stride<_1, Int<WeightScaleFoldedMPerFoldBlock *
                                    WeightScalePhysicalColsPerFoldBlock>>>,
          Int<WeightScaleRawElementsPerStage>>>;
  // MXFP8 activation scales are independent from MXFP4 weight scales.  They are
  // stored in raw M-major, K-contiguous form and TMA-loaded into this raw scale
  // layout: (BLK_N, ActScaleTmaChunks, PIPE).  The TMA window is 16B-aligned:
  // smaller compute Ktiles reuse a subrange, while larger Ktiles must already
  // span a 16B-aligned activation-scale row.
  using SmemLayoutActivationScale =
      Layout<Shape<decltype(shape<1>(TileShape{})), Int<ActScaleTmaChunks>, Int<Stages>>,
             Stride<Int<ActScaleTmaChunks>, _1, Int<ScaleNRawElementsPerStage>>>;

  static_assert(DispatchPolicy::Stages >= 2,
                "Specialization requires Stages set to value 2 or more.");
  static_assert(
      not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
          cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
      "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> ||
                    cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> ||
                    cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // To relax them, we need to handle loading more than 1 row of scales for every main loop
  // iteration. We must also handle updating the pipeline transaction bytes on the fly.
  static_assert(size<1>(SmemLayoutAtomScale{}) == 1, "size<1>(SmemLayoutAtomScale) must be 1.");

 private:
  static constexpr ConversionMode get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    } else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

  int current_group_idx_ = 0;
  cute::TmaDescriptor const* current_tma_desc_b_ = nullptr;

 public:
  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  // MixedInputUtils consumes these traits for shared-memory sizing and layout
  // selection.  Prescale only supports the FP4->FP8 scale-table case below.
  static constexpr bool ModeHasScales =
      KernelConversionMode == ConversionMode::ConvertAndScale ||
      KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;
  static constexpr bool FusedE8M0PreMmaScale = true;
  static_assert(!HasActivationScale,
                "The prescale collective expects activation scale, if any, to be handled outside "
                "the mainloop.");
  static constexpr bool UseScaleLookupTable = false;
  static constexpr bool UseFP4ToBF16LookupTable =
      KernelConversionMode == ConversionMode::ConvertAndScale &&
      cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
      cute::is_same_v<ElementB, cutlass::bfloat16_t>;
  static constexpr bool UseFP4ToFP8LookupTable =
      KernelConversionMode == ConversionMode::ConvertAndScale &&
      cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
      cute::is_same_v<ElementB, cutlass::float_e4m3_t>;
  static constexpr bool UseInt4ToFP8LookupTable =
      KernelConversionMode == ConversionMode::ConvertAndScale &&
      cute::is_same_v<ElementA, cutlass::int4b_t> &&
      cute::is_same_v<ElementB, cutlass::float_e4m3_t>;
  static_assert(UseFP4ToFP8LookupTable && cute::is_same_v<ElementScale, cutlass::float_ue8m0_t>,
                "Fused e8m0 pre-MMA scale is only implemented for MXFP4 x FP8 with folded scalar "
                "e8m0 scales.");
  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{});
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});
  static constexpr size_t SmemAlignmentScale = cute::max(SmemAlignmentA, SmemAlignmentB);

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");

  struct SharedStorage {
    static constexpr int scale_elements = cute::cosize_v<SmemLayoutWeightScaleRaw>;
    static constexpr int zero_elements = 0;
    static constexpr int activation_scale_elements = 0;
    struct TensorStorage {
      CUTE_ALIGNAS(SmemAlignmentA)
      cute::ArrayEngine<RealSwappedElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      CUTE_ALIGNAS(SmemAlignmentB)
      cute::ArrayEngine<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      // Keep the member layout aligned with mixed_input.hpp for online collective
      // switching.  Prescale only stages weight e8m0 scale; zero and activation
      // scale storage are intentionally empty.
      cute::ArrayEngine<WeightScaleRawElement, scale_elements> smem_scale;
      cute::ArrayEngine<NonVoidElementActivationScale, activation_scale_elements>
          smem_activation_scale;
      cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
    } tensors;

    struct TensorMapStorage {};

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;
  static constexpr bool RequiresTensormapUpdateOnBatchChange = false;
  static constexpr bool RequiresPrebuiltTensormapAcquireOnBatchChange = IsGroupedGemmKernel;

  // Host side kernel arguments.  Keep this parameter surface aligned with
  // mixed_input.hpp so callers can switch collectives without rebuilding the
  // argument-preparation path; unsupported prescale paths are rejected in
  // can_implement instead of by deleting fields here.
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    ElementScale const** ptr_S = nullptr;
    NonVoidStrideScale const* dS{};
    int chunk_size = 0;
    ElementZero const** ptr_Z = nullptr;
    NonVoidElementActivationScale const** ptr_ActivationScale = nullptr;
    StrideActivationScale const* dActivationScale{};
    cute::TmaDescriptor const* ptr_A_prebuilt_tma_desc = nullptr;
    cute::TmaDescriptor const* ptr_B_prebuilt_tma_descs = nullptr;
    cute::TmaDescriptor const* ptr_ActivationScale_prebuilt_tma_descs = nullptr;
  };

  // Device side kernel params
  struct Params {
    // For grouped GEMM with non-layout stride: replace static-zero L stride (_0) with
    // a static non-zero value so the TMA descriptor includes the L dimension at creation.
    // Int<32> is the minimum static value that after subbyte upcast<2> (FP4→uint8_t)
    // produces Int<16> = 16 bytes, satisfying cuTensorMapEncodeTiled's 16-byte alignment.
    // Being fully static, all CuTe coordinate computations remain compile-time optimizable.
    using TmaStrideA =
        cute::conditional_t<IsGroupedGemmKernel && !cute::is_layout<InternalSwappedStrideA>::value,
                            decltype(cute::make_stride(cute::get<0>(InternalSwappedStrideA{}),
                                                       cute::get<1>(InternalSwappedStrideA{}),
                                                       cute::Int<32>{})),
                            InternalSwappedStrideA>;
    using LayoutA =
        decltype(detail::get_gmem_layout(repeat_like(TmaStrideA{}, int32_t(0)), TmaStrideA{}));
    using LayoutB = decltype(detail::get_gmem_layout(
        repeat_like(InternalSwappedStrideB{}, int32_t(0)), InternalSwappedStrideB{}));

    using TMA_A = decltype(make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementA const*>(nullptr)),
                    LayoutA{}),
        SmemLayoutA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementB const*>(nullptr)),
                    LayoutB{}),
        SmemLayoutB{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})));  // mcast along M mode for this N load, if any
    using LayoutActivationScale = decltype(detail::get_gmem_layout(
        repeat_like(StrideActivationScale{}, int32_t(0)), StrideActivationScale{}));
    using TMA_ActivationScale_ = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            detail::get_logical_ptr(static_cast<NonVoidElementActivationScale const*>(nullptr)),
            LayoutActivationScale{}),
        SmemLayoutActivationScale{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), Int<ActScaleTmaChunks>{}), Int<1>{}));
    using TMA_ActivationScale =
        cute::conditional_t<HasActivationScale, TMA_ActivationScale_, cute::tuple<>>;

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_ActivationScale tma_load_activation_scale;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    SwappedElementA const** ptr_A;
    SwappedStrideA ptr_dA;
    SwappedElementB const** ptr_B;
    SwappedStrideB ptr_dB;
    cute::TmaDescriptor const* ptr_A_prebuilt_tma_desc;
    cute::TmaDescriptor const* ptr_B_prebuilt_tma_descs;
    NonVoidElementScale const** ptr_S;
    NonVoidStrideScale const* dS;
    NonVoidElementActivationScale const** ptr_ActivationScale;
    StrideActivationScale const* dActivationScale;
    cute::TmaDescriptor const* ptr_ActivationScale_prebuilt_tma_descs;
    NonVoidElementZero const** ptr_Z;
    int64_t scale_k;
    int chunk_size;
    int reload_factor = (chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;
    int num_groups;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape problem_shapes,
                                                  Arguments const& args,
                                                  [[maybe_unused]] void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create
    // tensormap/tma desc. These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);

    if constexpr (SwapAB) {
      init_M = get<1>(init_shape);
      init_N = get<0>(init_shape);
    }
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t mock_L = 1;
    SwappedElementA const* ptr_A_first_batch;
    SwappedElementB const* ptr_B_first_batch;
    SwappedStrideA ptr_dA;
    SwappedStrideB ptr_dB;
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;

    if constexpr (not SwapAB) {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_A);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_B);
    } else {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_B);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_A);
    }

    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      if constexpr (not SwapAB) {
        ptr_dA = args.dA;
        ptr_dB = args.dB;
      } else {
        ptr_dA = args.dB;
        ptr_dB = args.dA;
      }
      dA = InternalSwappedStrideA{};
      if constexpr (is_layout<InternalSwappedStrideA>::value) {
        dA = make_layout(transform_leaf(dA.shape(),
                                        [](auto x) {
                                          if constexpr (not is_static_v<decltype(x)>) {
                                            return static_cast<decltype(x)>(1);
                                          } else {
                                            return x;
                                          }
                                        }),
                         dA.stride());
      }
      dB = InternalSwappedStrideB{};
    } else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      if constexpr (not SwapAB) {
        dA = args.dA;
        dB = args.dB;
      } else {
        dA = args.dB;
        dB = args.dA;
      }
      ptr_dA = SwappedStrideA{};
      ptr_dB = SwappedStrideB{};
    }
    // For grouped GEMM: use TmaStrideA (with static _1 L stride) so the TMA descriptor
    // is created as 3D, enabling coordinate-based group selection.
    typename Params::TmaStrideA tma_dA;
    if constexpr (!IsGroupedGemmKernel || cute::is_layout<InternalSwappedStrideA>::value) {
      tma_dA = dA;
    }
    Tensor tensor_a = make_tensor(
        ptr_A_first_batch, detail::get_gmem_layout(make_shape(init_M, init_K, mock_L), tma_dA));
    Tensor tensor_b = make_tensor(ptr_B_first_batch,
                                  detail::get_gmem_layout(make_shape(init_N, init_K, mock_L), dB));

    typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(
        GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{}));  // mcast along N mode for this M load, if any
    typename Params::TMA_B tma_load_b =
        make_tma_copy(GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_, _, cute::Int<0>{}),
                      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
                      size<0>(ClusterShape{}));  // mcast along M mode for this N load, if any
    typename Params::TMA_ActivationScale tma_load_activation_scale{};

    int num_groups_val = 1;
    if constexpr (IsGroupedGemmKernel) {
      num_groups_val = problem_shapes.groups();
    }
    auto args_setup = [&](auto ptr_A, auto ptr_B, int64_t scale_k = 0, int chunk_size = 0,
                          int reload_factor = 1) -> Params {
      return {tma_load_a,
              tma_load_b,
              tma_load_activation_scale,
              TmaTransactionBytes,
              reinterpret_cast<SwappedElementA const**>(ptr_A),
              ptr_dA,
              reinterpret_cast<SwappedElementB const**>(ptr_B),
              ptr_dB,
              args.ptr_A_prebuilt_tma_desc,
              args.ptr_B_prebuilt_tma_descs,
              reinterpret_cast<NonVoidElementScale const**>(args.ptr_S),
              args.dS,
              args.ptr_ActivationScale,
              args.dActivationScale,
              args.ptr_ActivationScale_prebuilt_tma_descs,
              reinterpret_cast<NonVoidElementZero const**>(args.ptr_Z),
              scale_k,
              chunk_size,
              reload_factor,
              dA,
              dB,
              num_groups_val};
    };

    // Prescale keeps the historical scale_k field in Params so the argument
    // surface stays aligned with mixed_input.hpp.  Runtime scale addressing is
    // computed directly from ptr_S, chunk_size, and the current K tile.
    int64_t scale_k_placeholder = 1;
    return SwapAB ? args_setup(args.ptr_B, args.ptr_A, scale_k_placeholder, args.chunk_size,
                               (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}))
                  : args_setup(args.ptr_A, args.ptr_B, scale_k_placeholder, args.chunk_size,
                               (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}));
  }

  template <class ProblemShape>
  static size_t get_workspace_size([[maybe_unused]] ProblemShape const& problem_shape,
                                   [[maybe_unused]] Arguments const& args,
                                   [[maybe_unused]] int sm_count) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      [[maybe_unused]] ProblemShape const& problem_shape, [[maybe_unused]] Arguments const& args,
      [[maybe_unused]] void* workspace, [[maybe_unused]] cudaStream_t stream,
      [[maybe_unused]] CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(ProblemShape problem_shapes,
                                                Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A =
        tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B =
        tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if constexpr (IsGroupedGemmKernel) {
      implementable = implementable && (args.ptr_A_prebuilt_tma_desc != nullptr);
      implementable = implementable && (args.ptr_B_prebuilt_tma_descs != nullptr);
    }
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M, N, K, L] = problem_shape_MNKL;
        if constexpr (!cute::is_pointer_v<cute::decay_t<decltype(args.dA)>>) {
          implementable =
              implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(
                                   detail::get_gmem_layout(cute::make_shape(M, K, L), args.dA));
        }
        if constexpr (!cute::is_pointer_v<cute::decay_t<decltype(args.dB)>>) {
          implementable =
              implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
                                   detail::get_gmem_layout(cute::make_shape(N, K, L), args.dB));
        }
        const int scale_mn = SwapAB ? N : M;
        if (args.chunk_size == 0) {
          implementable = false;
        } else {
          implementable = implementable && (args.chunk_size == ScalingGroupSize);
          implementable = implementable && ((scale_mn % size<0>(TileShape{})) == 0);
          implementable = implementable && ((K % size<2>(TileShape{})) == 0);
        }
        implementable = implementable && (args.ptr_S != nullptr);
        implementable = implementable && (args.ptr_Z == nullptr);
        implementable = implementable && (args.ptr_ActivationScale == nullptr);
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for "
          "TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK = Utils::compute_tma_transaction_bytes_mk();
  static constexpr uint32_t TmaTransactionBytesNK = Utils::compute_tma_transaction_bytes_nk();
  static constexpr uint32_t TmaTransactionBytesExtra = Utils::compute_tma_transaction_bytes_extra();
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesMK + TmaTransactionBytesNK + TmaTransactionBytesExtra;

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL,
                                Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    // In this mixed-input layout, the transformed A operand is the offline weight tensor and the B
    // operand is the activation tensor.  A keeps a grouped L dimension for the
    // fixed weight layout; B and activation scales are retargeted per expert.
    auto A_L = mainloop_params.num_groups;
    auto B_L = mock_L;
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(
        shape(detail::get_gmem_layout(make_shape(M, K, A_L), mainloop_params.dA)));  // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(
        shape(detail::get_gmem_layout(make_shape(N, K, B_L), mainloop_params.dB)));  // (n,k,l)
    int const scale_total_k128_blocks = int(K) / WeightScaleLogicalKPerFoldBlock;

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_, _, _),
                               Step<_1, X, _1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_, _, _),
                               Step<X, _1, _1>{});  // (BLK_N,BLK_K,n,k,l)
    return cute::make_tuple(gA_mkl, gB_nkl, scale_total_k128_blocks);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <class... Ts, class... TMs, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline,
                           PipelineState smem_pipe_write, cute::tuple<Ts...> const& load_inputs,
                           [[maybe_unused]] cute::tuple<TMs...> const& input_tensormaps,
                           BlockCoord const& blk_coord, KTileIterator k_tile_iter, int k_tile_count,
                           int thread_idx, uint32_t block_rank_in_cluster,
                           TensorStorage& shared_tensors) {
    static_assert(sizeof...(Ts) == 3,
                  "Fused pre-MMA scale needs three inputs (gA, gB, total_k128_blocks)");
    static_assert(sizeof...(TMs) == 2, "Only A and B tensormaps needed");

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                             SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                             SmemLayoutB{});                  // (BLK_N,BLK_K,PIPE)
    Tensor sA = as_position_independent_swizzle_tensor(sA_);  // (BLK_M,BLK_K,PIPE)
    Tensor sB = as_position_independent_swizzle_tensor(sB_);  // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and B
    //

    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                    block_rank_in_cluster / cluster_shape_x};

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);
    int const scale_total_k128_blocks = get<2>(load_inputs);

    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    auto a_l_coord = current_group_idx_;
    auto b_l_coord = cute::Int<0>{};
    Tensor gA = gA_mkl(_, _, m_coord, _, a_l_coord);  // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_, _, n_coord, _, b_l_coord);  // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);  // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);  // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);  // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);  // (TMA,TMA_N,TMA_K,PIPE)

    Tensor sSRaw = make_tensor(
        make_smem_ptr(reinterpret_cast<WeightScaleRawElement*>(shared_tensors.smem_scale.begin())),
        SmemLayoutWeightScaleRaw{});

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;

    // Issue TmaLoads
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, Int<0>{}));
      }
    }

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        // TMA for A and B
        copy(mainloop_params.tma_load_a.with(mainloop_params.ptr_A_prebuilt_tma_desc, *tma_barrier,
                                             mcast_mask_a),
             tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
        copy(mainloop_params.tma_load_b.with(current_tma_desc_b_, *tma_barrier, mcast_mask_b),
             tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
      }

      int const scale_k128_offset = int(*k_tile_iter) * WeightScaleKBlocksPerTile;
      int const scale_m64_offset =
          int(m_coord) * int(size<0>(TileShape{})) / WeightScaleLogicalMPerFoldBlock;
      auto* scale_base =
          reinterpret_cast<WeightScaleRawElement const*>(mainloop_params.ptr_S[current_group_idx_]);

      auto scale_gmem_fold_block = [&](int m64_block, int k128_block) {
        return int64_t(m64_block) * int64_t(scale_total_k128_blocks) + int64_t(k128_block);
      };
      auto issue_scale_bulk_copy = [&](int local_m64_block) {
        int const m64_block = scale_m64_offset + local_m64_block;
        int64_t const scale_gmem_offset = scale_gmem_fold_block(m64_block, scale_k128_offset) *
                                          int64_t(WeightScaleRawElementsPerFoldBlock);
        auto* scale_gmem_addr = reinterpret_cast<void const*>(scale_base + scale_gmem_offset);
        auto* scale_smem_addr = static_cast<void*>(&sSRaw(0, 0, local_m64_block, 0, write_stage));
        cute::SM90_BULK_COPY_G2S::copy(scale_gmem_addr, reinterpret_cast<uint64_t*>(tma_barrier),
                                       scale_smem_addr, WeightScaleBulkCopyBytes);
      };

      if (cute::elect_one_sync()) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_m64_block = 0; local_m64_block < WeightScaleMBlocksPerTile;
             ++local_m64_block) {
          issue_scale_bulk_copy(local_m64_block);
        }
      }
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }
  struct NoopReleasedStageProducer {
    CUTLASS_DEVICE void operator()() const {}
  };

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <class FrgTensorC>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                          FrgTensorC& accum, int k_tile_count, int thread_idx,
                          TensorStorage& shared_tensors, Params const& mainloop_params) {
    NoopReleasedStageProducer released_stage_producer;
    mma_with_released_stage_producer(pipeline, smem_pipe_read, accum, k_tile_count, thread_idx,
                                     shared_tensors, mainloop_params, released_stage_producer);
  }

  // The compact single-warpgroup kernel refills a stage immediately after the
  // current tile safely releases it. Regular kernels compile the no-op callback away.
  template <class FrgTensorC, class ReleasedStageProducer>
  CUTLASS_DEVICE void mma_with_released_stage_producer(
      MainloopPipeline pipeline, PipelineState smem_pipe_read, FrgTensorC& accum, int k_tile_count,
      int thread_idx, TensorStorage& shared_tensors, Params const& mainloop_params,
      ReleasedStageProducer& released_stage_producer) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2,
                  "SwappedSmemLayoutAtomA must be rank 2.");
    static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2,
                  "SwappedSmemLayoutAtomB must be rank 2.");
    static_assert(
        !cute::is_void_v<SwappedSmemCopyAtomA>,
        "SM90 GMMA mainloops must specify a non-void copy atom for smem sourced instructions.");
    static_assert(
        cute::is_void_v<SwappedSmemCopyAtomB>,
        "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // Obtain warp index
    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                             SmemLayoutA{});                  // (BLK_M,BLK_K,PIPE)
    Tensor sA = as_position_independent_swizzle_tensor(sA_);  // (BLK_M,BLK_K,PIPE)

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                            SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                      size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be "
                  "NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout =
        make_layout(Int<MmaWarpGroups>{}, Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = thread_idx / NumThreadsPerWarpGroup;

    TiledMma tiled_mma;
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = mma_thread_slice.partition_A(sA);
    auto mma_warpgroup_slice = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // Allocate fragments and descriptors
    Tensor tCrA_mma =
        mma_thread_slice.partition_fragment_A(sA(_, _, Int<0>{}));  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA_load = [&] {
      if constexpr (not is_layout<InternalSwappedStrideA>::value) {
        // Make register tensor with MMA layout
        return make_fragment_like<RealSwappedElementA>(tCrA_mma);
      } else {
        // Make register tensor matching smem layout, converter will take care of de-swizzling
        return make_tensor_like<RealSwappedElementA>(tCsA(_, _, _, Int<0>{}));
      }
    }();
    Tensor tCsB = mma_warpgroup_slice.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)
    // tCrB is just a view of the tensor tCsB
    Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)

    //
    // Copy Atom A retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SwappedSmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);

    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA_load);  // (CPY,CPY_M,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));      // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));      // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));           // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));               // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));  // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));  // PIPE

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    using SmemCopyAtomA_LDSM = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;

    auto smem_tiled_copy_A_LDSM = make_tiled_copy_A(SmemCopyAtomA_LDSM{}, tiled_mma);
    auto smem_thr_copy_A_LDSM = smem_tiled_copy_A_LDSM.get_thread_slice(thread_idx);

    Tensor sA_LDSM = recast<ElementB>(sA);
    auto tCsA_LDSM = smem_thr_copy_A_LDSM.partition_S(sA_LDSM);

    using ABBitWidthRatio = Int<sizeof_bits_v<ElementB> / sizeof_bits_v<ElementA>>;
    auto tCrA_load_LDSM_shape =
        replace<2>(tCrA_mma.shape(), size(get<2>(tCrA_mma.shape())) / ABBitWidthRatio{});
    Tensor tCrA_load_LDSM = make_fragment_like<ElementB>(tCrA_load_LDSM_shape);
    Tensor tCrA_copy_view_LDSM =
        smem_thr_copy_A_LDSM.retile_D(tCrA_load_LDSM);  // (CPY,CPY_M,CPY_K)

    auto ptr = recast_ptr<RealSwappedElementA>(tCrA_load_LDSM.data());
    auto old_shape = tCrA_load_LDSM.shape();
    // LDSM packs two 4-bit K sub-blocks before advancing to the next MMA_M
    // slice. Preserve that nested K order so MMA_M > 1 does not alias K.
    auto tCrA_load_4b_layout = make_layout(
        make_shape(size<0>(old_shape), get<1>(old_shape),
                   make_shape(ABBitWidthRatio{}, size<2>(old_shape))),
        make_stride(Int<1>{}, size<0>(old_shape) * ABBitWidthRatio{},
                    make_stride(size<0>(old_shape),
                                size<0>(old_shape) * ABBitWidthRatio{} * size<1>(old_shape))));
    Tensor tCrA_load_4b_packed = make_tensor(ptr, tCrA_load_4b_layout);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor sSRaw = make_tensor(
        make_smem_ptr(reinterpret_cast<WeightScaleRawElement*>(shared_tensors.smem_scale.begin())),
        SmemLayoutWeightScaleExpanded{});
    Tensor tCsSRaw = mma_thread_slice.partition_A(sSRaw);

    PipelineState smem_pipe_release = smem_pipe_read;

    constexpr int K_BLOCK_MAX = size<2>(tCrA_load);
    constexpr int K_COMMIT_GROUP_SIZE = 4;
    constexpr int K_COMMIT_GROUPS = (K_BLOCK_MAX + K_COMMIT_GROUP_SIZE - 1) / K_COMMIT_GROUP_SIZE;
    constexpr int K_WAIT_MAX = (K_COMMIT_GROUPS - 1 < 7) ? K_COMMIT_GROUPS - 1 : 7;
    // Large-N tiles expose scale smem->RF latency; small-N best configs keep
    // the rolling copy to avoid extending scale register lifetime.
    constexpr bool PreloadAllScaleKblocks = size<1>(TileShape{}) >= 128;
    static_assert(K_BLOCK_MAX >= 4, "Consider increasing TileShapeK");
    static_assert(ScalingGroupSize % cute::get<0, 1>(tCsB.shape())() == 0,
                  "Fused e8m0 pre-MMA scale requires scale groups to align to MMA K blocks.");
    Tensor tCrA_scale_probe = tCrA_load_4b_packed(_, _, Int<0>{});
    Tensor tCrA_scale_probe_vm =
        cute::group_modes<1, -1>(cute::zipped_divide(tCrA_scale_probe, Int<8>{}));
    constexpr int ScalePairCount = decltype(size<1>(tCrA_scale_probe_vm))::value / 2;
    static_assert(ScalePairCount > 0,
                  "Fused e8m0 pre-MMA scale cache expects at least one fp4x8 operand pair.");
    // TileM256 has enough scale pairs per K block that the compact offset
    // cache creates a longer dependency chain than keeping the expanded scale
    // tensor in RF.  Smaller M tiles still prefer the compact offset cache.
    constexpr bool UseExpandedScaleRFForLargeM = size<0>(TileShape{}) >= 256;
    Tensor tCrA_scale = make_fragment_like<WeightScaleRawElement>(tCrA_load_4b_packed);
    cute::array<uint32_t, K_BLOCK_MAX * ScalePairCount> lo_exp_offsets;
    cute::array<uint32_t, K_BLOCK_MAX * ScalePairCount> hi_exp_offsets;

    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};
    auto copy_scale_kblock = [&](auto k_block_c, int read_stage) {
      constexpr int k_block = decltype(k_block_c)::value;
      if constexpr (k_block < size<2>(tCsSRaw.shape())) {
        Tensor scales = tCsSRaw(_, _, k_block_c, read_stage);
        if constexpr (UseExpandedScaleRFForLargeM) {
          copy(scales, tCrA_scale(_, _, k_block_c));
        } else {
          Utils::cache_A_kblock_fused_e8m0_pre_mma_exp_offsets(
              scales, k_block_c, Int<ScalePairCount>{}, lo_exp_offsets, hi_exp_offsets);
        }
      }
    };
    auto copy_scale_for_mma = [&](auto k_block_c, int read_stage) {
      if constexpr (PreloadAllScaleKblocks) {
        if constexpr (decltype(k_block_c)::value == 0) {
          cute::for_each(cute::make_seq<K_BLOCK_MAX>{}, [&](auto preload_k_block_c) {
            copy_scale_kblock(preload_k_block_c, read_stage);
          });
        }
      } else {
        copy_scale_kblock(k_block_c, read_stage);
      }
    };
    auto convert_A_kblock_static = [&](auto k_block_c, int read_stage) {
      auto tCrA_mma_slot = tCrA_mma(_, _, k_block_c);
      if constexpr (UseExpandedScaleRFForLargeM) {
        Utils::convert_A_kblock_fused_e8m0_pre_mma_raw_scale_to_slot(
            tCrA_load_4b_packed, tCrA_mma_slot, tCrA_scale, k_block_c);
      } else {
        Utils::convert_A_kblock_fused_e8m0_pre_mma_exp_offsets_to_slot(
            tCrA_load_4b_packed, tCrA_mma_slot, k_block_c, Int<ScalePairCount>{}, lo_exp_offsets,
            hi_exp_offsets);
      }
    };
    auto commit_mma_group = [&] {
      warpgroup_commit_batch();
      // A operand slots are reused by the next K tile.  Commit four adjacent K
      // blocks as one group and keep only the tail groups outstanding.  The
      // wait is FIFO: after the last group of tile T, the first group of T is
      // retired before tile T+1 overwrites slots 0..3.  Subsequent commits in
      // tile T+1 keep retiring older tail groups before their slots are reused.
      warpgroup_wait<K_WAIT_MAX>();
    };
    auto maybe_commit_mma_group = [&](auto k_block_c) {
      constexpr int k_block = decltype(k_block_c)::value;
      if constexpr (((k_block + 1) % K_COMMIT_GROUP_SIZE == 0) || (k_block == K_BLOCK_MAX - 1)) {
        commit_mma_group();
      }
    };

    // First K tile.
    {
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();

      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

      Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0, read_stage);
      copy_scale_for_mma(cute::Int<0>{}, read_stage);
      if (K_BLOCK_MAX > 1) {
        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1,
                              read_stage);
        copy_scale_for_mma(cute::Int<1>{}, read_stage);
      }

      convert_A_kblock_static(cute::Int<0>{}, read_stage);

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA_mma(_, _, cute::Int<0>{}), tCrB(_, _, cute::Int<0>{}, read_stage),
                 accum);
      maybe_commit_mma_group(cute::Int<0>{});
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;

      Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 2, read_stage);
      copy_scale_for_mma(cute::Int<2>{}, read_stage);
      convert_A_kblock_static(cute::Int<1>{}, read_stage);

      cute::for_each(cute::make_seq<K_BLOCK_MAX - 1>{}, [&](auto i) {
        constexpr int k_block = decltype(i)::value + 1;
        warpgroup_arrive();
        cute::gemm(tiled_mma, tCrA_mma(_, _, cute::Int<k_block>{}),
                   tCrB(_, _, cute::Int<k_block>{}, read_stage), accum);
        maybe_commit_mma_group(cute::Int<k_block>{});

        if constexpr (k_block < K_BLOCK_MAX - 2) {
          Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, k_block + 2,
                                read_stage);
          copy_scale_for_mma(cute::Int<k_block + 2>{}, read_stage);
        }
        if constexpr (k_block < K_BLOCK_MAX - 1) {
          convert_A_kblock_static(cute::Int<k_block + 1>{}, read_stage);
        }
      });

      --k_tile_count;
      if (k_tile_count > 0) {
        pipeline.consumer_wait(smem_pipe_read, barrier_token);

        int const next_read_stage = smem_pipe_read.index();
        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0,
                              next_read_stage);
        copy_scale_for_mma(cute::Int<0>{}, next_read_stage);
        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1,
                              next_read_stage);
        copy_scale_for_mma(cute::Int<1>{}, next_read_stage);

        // The rolling wait after the last commit has retired the oldest group
        // from the previous tile, which is the group that reads A slots 0..3.
        convert_A_kblock_static(cute::Int<0>{}, next_read_stage);
      } else {
        warpgroup_wait<0>();
      }
    }

    if (k_tile_count == 0) {
      return;
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 1; --k_tile_count) {
      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;

      cute::for_each(cute::make_seq<K_BLOCK_MAX>{}, [&](auto i) {
        constexpr int k_block = decltype(i)::value;
        warpgroup_arrive();
        cute::gemm(tiled_mma, tCrA_mma(_, _, cute::Int<k_block>{}),
                   tCrB(_, _, cute::Int<k_block>{}, read_stage), accum);
        maybe_commit_mma_group(cute::Int<k_block>{});

        if constexpr (k_block == K_BLOCK_MAX - 1) {
          pipeline.consumer_release(smem_pipe_release);
          ++smem_pipe_release;
          released_stage_producer();
        }

        if constexpr (k_block == 0) {
          barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        }

        if constexpr (k_block == K_BLOCK_MAX - 1) {
          pipeline.consumer_wait(smem_pipe_read, barrier_token);
          int const next_read_stage = smem_pipe_read.index();
          Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0,
                                next_read_stage);
          copy_scale_for_mma(cute::Int<0>{}, next_read_stage);
          Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1,
                                next_read_stage);
          copy_scale_for_mma(cute::Int<1>{}, next_read_stage);

          // The rolling wait after the last commit has retired the previous
          // tile's first A-slot group.  Later A-slot groups are retired by
          // subsequent grouped commits before their convert points.
          convert_A_kblock_static(cute::Int<0>{}, next_read_stage);
        } else {
          if constexpr (k_block < K_BLOCK_MAX - 2) {
            Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM,
                                  k_block + 2, read_stage);
            copy_scale_for_mma(cute::Int<k_block + 2>{}, read_stage);
          }
          convert_A_kblock_static(cute::Int<k_block + 1>{}, read_stage);
        }
      });
    }

    {
      int read_stage = smem_pipe_read.index();

      cute::for_each(cute::make_seq<K_BLOCK_MAX>{}, [&](auto i) {
        constexpr int k_block = decltype(i)::value;
        warpgroup_arrive();
        cute::gemm(tiled_mma, tCrA_mma(_, _, cute::Int<k_block>{}),
                   tCrB(_, _, cute::Int<k_block>{}, read_stage), accum);
        maybe_commit_mma_group(cute::Int<k_block>{});

        if constexpr (k_block == K_BLOCK_MAX - 1) {
          pipeline.consumer_release(smem_pipe_release);
          ++smem_pipe_release;
          released_stage_producer();
        }

        if constexpr (k_block < K_BLOCK_MAX - 2) {
          Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, k_block + 2,
                                read_stage);
          copy_scale_for_mma(cute::Int<k_block + 2>{}, read_stage);
        }
        if constexpr (k_block < K_BLOCK_MAX - 1) {
          convert_A_kblock_static(cute::Int<k_block + 1>{}, read_stage);
        }
      });

      warpgroup_wait<0>();
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release,
                               int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = 1;
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(
          smem_pipe_release);  // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //
  CUTLASS_DEVICE auto tensormaps_init(Params const& mainloop_params,
                                      [[maybe_unused]] TensorMapStorage& shared_tensormaps,
                                      [[maybe_unused]] int32_t sm_count,
                                      [[maybe_unused]] int32_t sm_idx) {
    return cute::make_tuple(mainloop_params.ptr_A_prebuilt_tma_desc,
                            mainloop_params.ptr_B_prebuilt_tma_descs);
  }

  template <class... TMs, class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_perform_update(
      [[maybe_unused]] TensorMapStorage& shared_tensormaps,
      [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] cute::tuple<TMs...> const& input_tensormaps,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl, [[maybe_unused]] int32_t next_batch) {}

  template <class... TMs>
  CUTLASS_DEVICE void tensormaps_cp_fence_release(
      [[maybe_unused]] TensorMapStorage& shared_tensormaps,
      [[maybe_unused]] cute::tuple<TMs...> const& input_tensormaps) {}

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class... TMs>
  CUTLASS_DEVICE void tensormaps_fence_acquire(cute::tuple<TMs...> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(current_tma_desc_b_);
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE InputTensors tensors_perform_update(
      InputTensors const& input_tensors, [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl, [[maybe_unused]] int32_t next_batch) {
    current_group_idx_ = next_batch;
    current_tma_desc_b_ = mainloop_params.ptr_B_prebuilt_tma_descs + next_batch;
    return input_tensors;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
