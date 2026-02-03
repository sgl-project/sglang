// Support scale B from cutlass
#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/collective/mixed_input_utils.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
    int Stages,
    class ClusterShape,
    class KernelSchedule_,
    class TileShape_,
    class ElementAOptionalTuple,
    class StrideA_,
    class ElementBOptionalTuple,
    class StrideB_,
    class TiledMma_,
    class GmemTiledCopyA_,
    class SmemLayoutAtomA_,
    class SmemCopyAtomA_,
    class TransformA_,
    class GmemTiledCopyB_,
    class SmemLayoutAtomB_,
    class SmemCopyAtomB_,
    class TransformB_>
struct CollectiveMma_<
    MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>,
    TileShape_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
 public:
  //
  // Type Aliases
  //
  using ConversionMode = cutlass::detail::ConversionMode;
  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>;
  using TileShape = TileShape_;
  using KernelSchedule = KernelSchedule_;

 private:
  template <class T>
  friend struct detail::MixedInputUtils;
  using CollectiveType = CollectiveMma<
      DispatchPolicy,
      TileShape_,
      ElementAOptionalTuple,
      StrideA_,
      ElementBOptionalTuple,
      StrideB_,
      TiledMma_,
      GmemTiledCopyA_,
      SmemLayoutAtomA_,
      SmemCopyAtomA_,
      TransformA_,
      GmemTiledCopyB_,
      SmemLayoutAtomB_,
      SmemCopyAtomB_,
      TransformB_>;
  using Utils = detail::MixedInputUtils<CollectiveType>;

  //
  // Type Aliases
  //
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

 public:
  static_assert(
      cute::is_tuple<ElementAOptionalTuple>::value && cute::is_tuple<ElementBOptionalTuple>::value,
      "ElementA and ElementB must be tuples.");
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;

  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;

  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is
  // void.
  using NonVoidElementScaleA = cute::conditional_t<cute::is_void_v<ScaleA>, float, ScaleA>;
  using NonVoidElementScaleB = cute::conditional_t<cute::is_void_v<ScaleB>, float, ScaleB>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;

  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  using StrideScaleA = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using StrideScaleB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using NonVoidStrideScaleA =
      cute::conditional_t<cute::is_void_v<StrideScaleA>, cute::Stride<_1, int64_t, int64_t>, StrideScaleA>;
  using NonVoidStrideScaleB =
      cute::conditional_t<cute::is_void_v<StrideScaleB>, cute::Stride<int64_t, _1, int64_t>, StrideScaleB>;

  static_assert(
      (IsATransformed && (cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value ||
                          is_layout<InternalStrideA>::value)) ||
          (!IsATransformed && (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value ||
                               is_layout<InternalStrideB>::value)),
      "The transformed type must be K-major.");

  static_assert(
      (IsATransformed && (sizeof(ElementB) == 2)) || (!IsATransformed && (sizeof(ElementA) == 2)) ||
          ((cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value ||
            is_layout<InternalStrideA>::value) &&
           (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value ||
            is_layout<InternalStrideB>::value)),
      "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(
      cutlass::gemm::detail::is_mn_major<NonVoidStrideScaleA>() &&
          cutlass::gemm::detail::is_k_major<NonVoidStrideScaleB>(),
      "ScaleA must be MN major [Col Major if A is scaled]. ScaleB must be K major.");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = cute::SM90_TMA_LOAD;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using SmemCopyAtomScaleA = Copy_Atom<cute::AutoVectorizingCopy, NonVoidElementScaleA>;
  using SmemCopyAtomScaleB = Copy_Atom<cute::DefaultCopy, NonVoidElementScaleB>;

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
  using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using SwappedTransformA = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using SwappedTransformB = cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int NumProducerThreadEvents = 1;
  static constexpr int GroupSize = 128;
  using SmemLayoutAtomScaleA = Layout<Shape<decltype(cute::shape<0>(SwappedSmemLayoutAtomA{})), cute::Int<1>>>;

  static constexpr int TileK = cute::get<2>(TileShape{});

  // static constexpr int TileKGroup = TileK / GroupSize;
  static constexpr int TileKGroup = 4;  // K need padding to 4

  using SmemLayoutAtomScaleB = Layout<
      Shape<decltype(cute::shape<0>(SwappedSmemLayoutAtomB{})), cute::Int<TileKGroup>>,
      cute::Stride<cute::Int<TileKGroup>, cute::Int<1>>>;

  using ScaleATileShape = decltype(make_shape(shape<0>(TileShape{}), shape<1>(SmemLayoutAtomScaleA{})));
  using ScaleBTileShape = decltype(make_shape(shape<1>(TileShape{}), cute::Int<TileKGroup>{}));

  static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");

  static_assert(
      (size<0>(TileShape{}) % size<0>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(
      (size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(
      (size<1>(TileShape{}) % size<0>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(
      (size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScaleA{}) == 2, "SmemLayoutAtomScaleA must be rank 2");
  static_assert(rank(SmemLayoutAtomScaleB{}) == 2, "SmemLayoutAtomScaleB must be rank 2");

  static_assert(
      (size<0>(TileShape{}) % size<0>(SmemLayoutAtomScaleA{})) == 0, "SmemLayoutAtomScaleA must equal the tile shape.");
  static_assert(
      (size<2>(TileShape{}) % size<1>(SmemLayoutAtomScaleA{})) == 0,
      "SmemLayoutAtomScaleA must evenly divide tile k shape.");

  static_assert(
      (size<1>(TileShape{}) % size<0>(SmemLayoutAtomScaleB{})) == 0, "SmemLayoutAtomScaleB must equal the tile shape.");
  static_assert(
      (size<2>(TileShape{}) % size<1>(SmemLayoutAtomScaleB{})) == 0,
      "SmemLayoutAtomScaleB must evenly divide tile k shape.");

  /// Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomA{}, select<0, 2>(TileShape{}), InternalSwappedStrideA{}));
  using SmemLayoutB = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomB{}, select<1, 2>(TileShape{}), InternalSwappedStrideB{}));

  // It is assumed that the scales and zero-points share the same smem layout
  using SmemLayoutScaleA = decltype(tile_to_shape(
      SmemLayoutAtomScaleA{},
      make_shape(shape<0>(ScaleATileShape{}), shape<1>(ScaleATileShape{}), Int<Stages>{}),
      cute::conditional_t<
          ::cutlass::gemm::detail::is_major<0, NonVoidStrideScaleA>(),
          Step<_2, _1, _3>,
          Step<_1, _2, _3>>{}));

  using SmemLayoutScaleB = decltype(tile_to_shape(
      SmemLayoutAtomScaleB{},
      make_shape(shape<0>(ScaleBTileShape{}), shape<1>(ScaleBTileShape{}), Int<Stages>{}),
      cute::conditional_t<
          ::cutlass::gemm::detail::is_major<0, NonVoidStrideScaleB>(),
          Step<_2, _1, _3>,
          Step<_1, _2, _3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(
      not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
          cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
      "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");
  static_assert(
      cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(
      cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // To relax them, we need to handle loading more than 1 row of scales for every main loop iteration.
  // We must also handle updating the pipeline transaction bytes on the fly.
  static_assert(size<1>(SmemLayoutAtomScaleA{}) == 1, "size<1>(SmemLayoutAtomScaleA) must be 1.");

 private:
  static constexpr ConversionMode get_conversion_mode() {
    if constexpr (cute::is_void_v<ScaleA>) {
      return ConversionMode::DirectConvert;
    } else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    } else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

 public:
  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;
  static constexpr bool UseScaleLookupTable =
      KernelConversionMode == ConversionMode::ConvertAndScale && cutlass::detail::is_Array_v<ScaleA>;
  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{});
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});
  static constexpr size_t SmemAlignmentScale = cute::max(SmemAlignmentA, SmemAlignmentB);

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
