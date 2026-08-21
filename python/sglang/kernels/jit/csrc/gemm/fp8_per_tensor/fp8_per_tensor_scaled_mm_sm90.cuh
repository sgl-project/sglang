/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include "fp8_per_tensor_common.cuh"

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

namespace sglang {

using namespace cute;

template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

namespace sm90_fp8 {

template <typename T, typename TileShape>
using ColLoad = cutlass::epilogue::fusion::
    Sm90ColBroadcast<0, TileShape, T, T, Stride<Int<1>, Int<0>, Int<0>>, 128 / sizeof_bits_v<T>, false>;

template <typename T, typename TileShape>
using RowLoad = cutlass::epilogue::fusion::
    Sm90RowBroadcast<0, TileShape, T, T, Stride<Int<0>, Int<1>, Int<0>>, 128 / sizeof_bits_v<T>, false>;

// One scale broadcast to every element. The AOT tree reads a per-row or
// scalar A scale through a vendored Sm90ColOrScalarBroadcast that switches on
// a runtime flag; upstream CUTLASS has no such visitor, so the JIT build picks
// the broadcast at compile time instead -- the same way the SM100/SM120 paths
// specialize on ScalarA.
template <typename T, typename TileShape>
using ScalarLoad = cutlass::epilogue::fusion::Sm90ScalarBroadcast<T>;

// Sm90ScalarBroadcast takes its value by pointer, in a different Arguments
// shape than the row/column broadcasts.
template <typename Descriptor, bool IsScalar>
static typename Descriptor::Arguments scale_args(tvm::ffi::TensorView scales) {
  auto* ptr = static_cast<const float*>(scales.data_ptr());
  if constexpr (IsScalar) {
    return typename Descriptor::Arguments{{}, {ptr}, {}};
  } else {
    return typename Descriptor::Arguments{ptr};
  }
}

template <bool ScalarA, bool ScalarB, typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogue {
 private:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = std::conditional_t<ScalarA, ScalarLoad<float, TileShape>, ColLoad<float, TileShape>>;
  using ScaleB = std::conditional_t<ScalarB, ScalarLoad<float, TileShape>, RowLoad<float, TileShape>>;

  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(
      tvm::ffi::TensorView a_scales, tvm::ffi::TensorView b_scales, tvm::ffi::Optional<tvm::ffi::TensorView> bias) {
    auto a_args = scale_args<ScaleA, ScalarA>(a_scales);
    auto b_args = scale_args<ScaleB, ScalarB>(b_scales);
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};

template <
    bool ScalarA,
    bool ScalarB,
    typename ElementAcc,
    typename ElementD,
    typename TileShape,
    template <typename, typename> typename BiasLoad>
struct ScaledEpilogueWithBias {
 private:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = std::conditional_t<ScalarA, ScalarLoad<float, TileShape>, ColLoad<float, TileShape>>;
  using ScaleB = std::conditional_t<ScalarB, ScalarLoad<float, TileShape>, RowLoad<float, TileShape>>;
  using Bias = BiasLoad<ElementD, TileShape>;

  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::homogeneous_multiply_add, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(
      tvm::ffi::TensorView a_scales, tvm::ffi::TensorView b_scales, tvm::ffi::Optional<tvm::ffi::TensorView> bias) {
    auto a_args = scale_args<ScaleA, ScalarA>(a_scales);
    auto b_args = scale_args<ScaleB, ScalarB>(b_scales);
    typename Bias::Arguments bias_args{static_cast<const ElementD*>(bias.value().data_ptr())};
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, bias_args, {}};
  }
};

template <bool ScalarA, bool ScalarB, typename ElementAcc, typename ElementD, typename TileShape>
using ScaledEpilogueBias = ScaledEpilogueWithBias<ScalarA, ScalarB, ElementAcc, ElementD, TileShape, RowLoad>;

template <bool ScalarA, bool ScalarB, typename ElementAcc, typename ElementD, typename TileShape>
using ScaledEpilogueColumnBias = ScaledEpilogueWithBias<ScalarA, ScalarB, ElementAcc, ElementD, TileShape, ColLoad>;

template <
    typename ElementAB_,
    typename ElementD_,
    template <typename, typename, typename> typename Epilogue_,
    typename TileShape,
    typename ClusterShape,
    typename KernelSchedule,
    typename EpilogueSchedule,
    bool swap_ab_ = false>
struct Gemm {
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc = float;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;
  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementD>::value;

  static constexpr bool swap_ab = swap_ab_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;

  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc,
      float,
      ElementC,
      conditional_t<swap_ab, LayoutC_Transpose, LayoutC>,
      AlignmentCD,
      ElementD,
      conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
      AlignmentCD,
      EpilogueSchedule,
      EVTCompute>::CollectiveOp;

  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          ElementAB,
          LayoutB_T,
          AlignmentAB,
          ElementAB,
          LayoutA_T,
          AlignmentAB,
          ElementAcc,
          TileShape,
          ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          ElementAB,
          LayoutA,
          AlignmentAB,
          ElementAB,
          LayoutB,
          AlignmentAB,
          ElementAcc,
          TileShape,
          ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp>;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

template <typename GemmType>
void launch(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView a_scales,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  static constexpr bool swap_ab = GemmType::swap_ab;
  using ElementAB = typename GemmType::ElementAB;
  using ElementD = typename GemmType::ElementD;
  using GemmKernel = typename GemmType::GemmKernel;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;

  const int32_t m = a.size(0);
  const int32_t n = b.size(1);
  const int32_t k = a.size(1);
  auto prob_shape = swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr, b_stride, a_ptr, a_stride}
              : typename GemmKernel::MainloopArguments{a_ptr, a_stride, b_ptr, b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      swap_ab ? GemmType::Epilogue::prepare_args(b_scales, a_scales, bias)
              : GemmType::Epilogue::prepare_args(a_scales, b_scales, bias),
      c_ptr,
      c_stride,
      c_ptr,
      c_stride};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.device().device_id;
  hw_info.sm_count = static_cast<int>(host::runtime::get_sm_count(hw_info.device_id));
  typename GemmKernel::TileSchedulerArguments scheduler = {};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args, hw_info, scheduler};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  const size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace_tensor = host::alloc_workspace_tensor(workspace_size, a.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm_op.run(args, workspace, stream));
}

}  // namespace sm90_fp8

template <
    typename OutType,
    bool WithBias,
    typename TileShape_,
    typename ClusterShape_,
    typename KernelSchedule_,
    bool SwapAb,
    bool ScalarAScale>
struct Sm90Fp8Config {
  // launch() hands the epilogue (b_scales, a_scales) when it swaps A and B, so
  // a scalar A scale sits in the ScaleB slot there and in ScaleA otherwise.
  static constexpr bool kScalarInA = ScalarAScale && !SwapAb;
  static constexpr bool kScalarInB = ScalarAScale && SwapAb;

  template <typename Acc, typename D, typename Tile>
  using Epilogue = std::conditional_t<
      !WithBias,
      sm90_fp8::ScaledEpilogue<kScalarInA, kScalarInB, Acc, D, Tile>,
      std::conditional_t<
          SwapAb,
          sm90_fp8::ScaledEpilogueColumnBias<kScalarInA, kScalarInB, Acc, D, Tile>,
          sm90_fp8::ScaledEpilogueBias<kScalarInA, kScalarInB, Acc, D, Tile>>>;

  using Gemm = sm90_fp8::Gemm<
      cutlass::float_e4m3_t,
      OutType,
      Epilogue,
      TileShape_,
      ClusterShape_,
      KernelSchedule_,
      cutlass::epilogue::TmaWarpSpecialized,
      SwapAb>;
};

template <typename OutType, bool WithBias, bool ScalarA>
void sm90_fp8_pertensor_dispatch_shape_impl(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  using PingpongFastAccum = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using FastAccum = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;

  using GemmDefault = typename Sm90Fp8Config<
      OutType,
      WithBias,
      Shape<_128, _128, _128>,
      Shape<_2, _1, _1>,
      PingpongFastAccum,
      false,
      ScalarA>::Gemm;
  using GemmM128LargeN = typename Sm90Fp8Config<
      OutType,
      WithBias,
      Shape<_64, _128, _128>,
      Shape<_2, _1, _1>,
      PingpongFastAccum,
      false,
      ScalarA>::Gemm;
  using GemmM128SmallN = typename Sm90Fp8Config<
      OutType,
      WithBias,
      Shape<_64, _64, _128>,
      Shape<_1, _1, _1>,
      PingpongFastAccum,
      false,
      ScalarA>::Gemm;
  using GemmM64SmallN =
      typename Sm90Fp8Config<OutType, WithBias, Shape<_64, _16, _256>, Shape<_1, _4, _1>, FastAccum, true, ScalarA>::
          Gemm;
  using GemmM64LargeN =
      typename Sm90Fp8Config<OutType, WithBias, Shape<_64, _64, _256>, Shape<_1, _1, _1>, FastAccum, true, ScalarA>::
          Gemm;
  using GemmM32LargeN =
      typename Sm90Fp8Config<OutType, WithBias, Shape<_64, _32, _256>, Shape<_1, _1, _1>, FastAccum, true, ScalarA>::
          Gemm;
  using GemmM16SmallN =
      typename Sm90Fp8Config<OutType, WithBias, Shape<_64, _16, _256>, Shape<_1, _2, _1>, FastAccum, true, ScalarA>::
          Gemm;
  using GemmM16LargeN =
      typename Sm90Fp8Config<OutType, WithBias, Shape<_64, _16, _256>, Shape<_1, _1, _1>, FastAccum, true, ScalarA>::
          Gemm;

  const uint32_t m = a.size(0);
  const uint32_t n = b.size(1);

  static constexpr uint32_t kNThreshold = 1280;

  static constexpr uint32_t kM128NThreshold = 4096;

  if (m <= 16) {
    if (n <= kNThreshold) {
      return sm90_fp8::launch<GemmM16SmallN>(out, a, b, scales_a, scales_b, bias, stream);
    }
    return sm90_fp8::launch<GemmM16LargeN>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (m <= 64) {
    if (n <= kNThreshold) {
      return sm90_fp8::launch<GemmM64SmallN>(out, a, b, scales_a, scales_b, bias, stream);
    }
    if (m <= 32) {
      return sm90_fp8::launch<GemmM32LargeN>(out, a, b, scales_a, scales_b, bias, stream);
    }
    return sm90_fp8::launch<GemmM64LargeN>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (m <= 128) {
    if (n <= kM128NThreshold) {
      return sm90_fp8::launch<GemmM128SmallN>(out, a, b, scales_a, scales_b, bias, stream);
    }
    return sm90_fp8::launch<GemmM128LargeN>(out, a, b, scales_a, scales_b, bias, stream);
  }
  return sm90_fp8::launch<GemmDefault>(out, a, b, scales_a, scales_b, bias, stream);
}

template <typename OutType>
void sm90_fp8_pertensor_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  const bool scalar_a = scales_a.numel() == 1;
  if (bias.has_value()) {
    if (scalar_a) {
      return sm90_fp8_pertensor_dispatch_shape_impl<OutType, true, true>(out, a, b, scales_a, scales_b, bias, stream);
    }
    return sm90_fp8_pertensor_dispatch_shape_impl<OutType, true, false>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (scalar_a) {
    return sm90_fp8_pertensor_dispatch_shape_impl<OutType, false, true>(out, a, b, scales_a, scales_b, bias, stream);
  }
  return sm90_fp8_pertensor_dispatch_shape_impl<OutType, false, false>(out, a, b, scales_a, scales_b, bias, stream);
}

}  // namespace sglang
