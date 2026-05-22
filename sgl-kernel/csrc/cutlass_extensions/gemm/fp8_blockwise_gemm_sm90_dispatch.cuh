// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass_extensions/common.hpp"
#include "cutlass_extensions/gemm/cutlass_gemm_caller.cuh"
#include "cutlass_extensions/gemm/dispatch_policy.hpp"

using namespace cute;

template <
    typename SchedulerType,
    typename OutType,
    int GroupSizeM_,
    int GroupSizeN_,
    int GroupSizeK_,
    int TileSizeM_ = 128,
    class ClusterShape = Shape<_1, _2, _1>>
struct cutlass_3x_gemm_fp8_blockwise {
  using GroupSizeM = Int<GroupSizeM_>;
  using GroupSizeN = Int<GroupSizeN_>;
  using GroupSizeK = Int<GroupSizeK_>;
  using TileSizeM = Int<TileSizeM_>;

  static_assert(TileSizeM_ % GroupSizeM_ == 0, "TileSizeM must be a multiple of GroupSizeM");

  using ElementAB = cutlass::float_e4m3_t;

  // A matrix configuration
  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  // B matrix configuration
  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  // C/D matrix configuration
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutType>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using ScaleTileShape = Shape<_1, _128, _128>;
  using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator = float;                            // Element type for internal accumulation
  using ElementCompute = float;                                // Element type for compute
  using TileShape = Shape<TileSizeM, GroupSizeN, GroupSizeK>;  // Threadblock-level tile size

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      EpilogueSchedule,
      StoreEpilogueCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      SchedulerType>;
};

template <typename Gemm>
void cutlass_gemm_caller_blockwise(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales) {
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementAB = typename Gemm::ElementAB;
  using ElementA = ElementAB;
  using ElementB = ElementAB;
  using ElementD = typename Gemm::ElementD;
  using ElementBlockScale = float;

  using ScaleTileShape = Shape<_1, _128, _128>;
  using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());
  auto b_ptr = static_cast<ElementB*>(b.data_ptr());

  auto a_s_ptr = static_cast<ElementBlockScale*>(a_scales.data_ptr());
  auto b_s_ptr = static_cast<ElementBlockScale*>(b_scales.data_ptr());

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  using StrideC = typename GemmKernel::StrideC;

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptr, a_stride, b_ptr, b_stride, a_s_ptr, layout_sfa, b_s_ptr, layout_sfb};
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::TileSchedulerArguments scheduler;

  static constexpr bool UsesStreamKScheduler =
      cute::is_same_v<typename GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;

  if constexpr (UsesStreamKScheduler) {
    using DecompositionMode =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    using ReductionMode =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;

    scheduler.decomposition_mode = DecompositionMode::StreamK;
    scheduler.reduction_mode = ReductionMode::Nondeterministic;
  }

  cutlass_gemm_caller<GemmKernel>(a.device(), {m, n, k, 1}, mainloop_args, epilogue_args, scheduler);
}

template <typename OutType>
void cutlass_gemm_blockwise_sm90_fp8_dispatch(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales) {
  auto k = a.size(1);
  auto n = b.size(1);

  if (k > 3 * n) {
    cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<cutlass::gemm::StreamKScheduler, OutType, 1, 128, 128>>(
        out, a, b, a_scales, b_scales);
  } else {
    cutlass_gemm_caller_blockwise<
        cutlass_3x_gemm_fp8_blockwise<cutlass::gemm::PersistentScheduler, OutType, 1, 128, 128>>(
        out, a, b, a_scales, b_scales);
  }
}
