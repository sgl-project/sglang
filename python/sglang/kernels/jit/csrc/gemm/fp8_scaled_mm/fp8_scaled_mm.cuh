/* Copyright 2025 SGLang Team. All Rights Reserved.

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

// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_template.h
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_kernel_template_sm89.h
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_kernel_template_sm90.h

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/packed_stride.hpp>

#include <cudaTypedefs.h>

#if SGL_CUDA_ARCH >= 900 && SGL_CUDA_ARCH < 1000
#include "cutlass_extensions/gemm/fp8_gemm_sm90_dispatch.cuh"
#endif
#include "math.cuh"

namespace sglang {

using namespace host;
using namespace cute;

#if defined CUDA_VERSION && CUDA_VERSION >= 12040 && SGL_CUDA_ARCH == 890
template <
    typename ElementType,
    typename OutElementType,
    typename AccumElementType,
    typename CtaShape,
    typename WarpShape,
    int Stages,
    bool WithBias,
    typename FP8MathOperator = cutlass::arch::OpMultiplyAdd,
    template <typename...> typename EpilogueVisitor = cutlass::epilogue::threadblock::Sm80EVT,
    typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct DeviceGemmFp8RowwiseSm89 {
  static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

  using ElementA = ElementType;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementType;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = OutElementType;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementOutput = OutElementType;
  using LayoutOutput = cutlass::layout::RowMajor;
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ElementAccumulator = AccumElementType;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm89;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  // Number of epilogue stages in EVT
  static constexpr int EVTEpilogueStages = 1;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::
      OutputTileThreadLayout<CtaShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

  // Definition of EVT
  using accSrc = cutlass::epilogue::threadblock::VisitorAccFetch;

  using ComputeBScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using bScaleSrc = cutlass::epilogue::threadblock::
      VisitorRowBroadcast<OutputTileThreadMap, ElementComputeEpilogue, Stride<_0, _1, _0>>;
  using EpilogueBScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeBScale, accSrc, bScaleSrc>;

  using ComputeAScale = cutlass::epilogue::threadblock::
      VisitorCompute<cutlass::multiplies, ElementC, ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
  using aScaleSrc = cutlass::epilogue::threadblock::
      VisitorColBroadcast<OutputTileThreadMap, ElementComputeEpilogue, Stride<_1, _0, _0>>;
  using EpilogueAScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeAScale, EpilogueBScale, aScaleSrc>;

  // With bias
  using biasSrc =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ElementOutput, Stride<_0, _1, _0>>;
  using ComputeAScaleWithBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiply_add,
      ElementC,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EpilogueAScaleWithBias =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAScaleWithBias, EpilogueBScale, aScaleSrc, biasSrc>;

  using dTar = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap,
      ElementC,
      cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, _1, _0>>;
  using EpilogueStore = typename cutlass::platform::conditional<
      WithBias,
      cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScaleWithBias>,
      cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScale>>::type;

  using EpilogueOp = EpilogueStore;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      AlignmentA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      AlignmentB,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementAccumulator,
      ElementComputeEpilogue,
      OperatorClass,
      ArchTag,
      CtaShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      ThreadblockSwizzle,
      Stages,
      FP8MathOperator,
      EVTEpilogueStages>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm, bool WithBias>
typename Gemm::Arguments prepare_sm89_fp8_args(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  using ElementT = typename Gemm::ElementA;
  using ElementOutput = typename Gemm::ElementD;
  using ElementComputeEpilogue = float;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  ElementT const* ptr_a = reinterpret_cast<ElementT const*>(a.data_ptr());
  ElementT const* ptr_b = reinterpret_cast<ElementT const*>(b.data_ptr());
  ElementOutput const* ptr_bias = nullptr;
  if constexpr (WithBias) {
    RuntimeCheck(bias.has_value());
    ptr_bias = reinterpret_cast<ElementOutput const*>(bias.value().data_ptr());
  }
  ElementOutput* ptr_d = reinterpret_cast<ElementOutput*>(out.data_ptr());
  ElementComputeEpilogue const* ptr_scales_a = reinterpret_cast<ElementComputeEpilogue const*>(scales_a.data_ptr());
  ElementComputeEpilogue const* ptr_scales_b = reinterpret_cast<ElementComputeEpilogue const*>(scales_b.data_ptr());

  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,  // Mode
      {m, n, k},                                // Problem size
      1,                                        // Split-k factor
      {},                                       // Epilogue args
      ptr_a,                                    // a pointer
      ptr_b,                                    // b pointer
      nullptr,                                  // c pointer (unused)
      nullptr,                                  // d pointer (unused)
      m * k,                                    // batch stride a (unused)
      n * k,                                    // batch stride b (unused)
      m * n,                                    // batch stride c (unused)
      m * n,                                    // batch stride d (unused)
      lda,                                      // stride a
      ldb,                                      // stride b
      ldc,                                      // stride c (unused)
      ldc);                                     // stride d (unused)
  if constexpr (WithBias) {
    args.epilogue = {
        {
            {
                {},  // Accumulator
                {ptr_scales_b, ElementComputeEpilogue(0), {_0{}, _1{}, _0{}}},
                {}  // Multiplies
            },
            {ptr_scales_a, ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
            {ptr_bias, ElementOutput(0), {_0{}, _1{}, _0{}}},
            {}  // Multiplies
        },
        {ptr_d, {n, _1{}, _0{}}}};
  } else {
    args.epilogue = {
        {
            {
                {},  // Accumulator
                {ptr_scales_b, ElementComputeEpilogue(0), {_0{}, _1{}, _0{}}},
                {}  // Multiplies
            },
            {ptr_scales_a, ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
            {}  // Multiplies
        },
        {ptr_d, {n, _1{}, _0{}}}};
  }

  return args;
}

template <typename Gemm, bool WithBias>
void launch_sm89_fp8_scaled_mm(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  auto args = prepare_sm89_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);
  Gemm gemm_op;

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = alloc_workspace_tensor(workspace_size, a.device());
  auto stream = LaunchKernel::resolve_device(a.device());

  auto can_implement = gemm_op.can_implement(args);
  RuntimeCheck(can_implement == cutlass::Status::kSuccess);

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  RuntimeCheck(status == cutlass::Status::kSuccess);
}

template <typename OutType, typename CtaShape, typename WarpShape, int Stages>
void sm89_fp8_dispatch_bias(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = OutType;
  using AccumElementType = float;
  if (bias) {
    using Gemm = typename DeviceGemmFp8RowwiseSm89<
        ElementInput,
        ElementOutput,
        AccumElementType,
        CtaShape,
        WarpShape,
        Stages,
        true>::Gemm;
    return launch_sm89_fp8_scaled_mm<Gemm, true>(out, a, b, scales_a, scales_b, bias);
  } else {
    using Gemm = typename DeviceGemmFp8RowwiseSm89<
        ElementInput,
        ElementOutput,
        AccumElementType,
        CtaShape,
        WarpShape,
        Stages,
        false>::Gemm;
    return launch_sm89_fp8_scaled_mm<Gemm, false>(out, a, b, scales_a, scales_b, bias);
  }
}

template <typename OutType>
void sm89_fp8_dispatch_shape(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  uint32_t const m = a.size(0);
  uint32_t const n = out.size(1);

  if (m == 1) {
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 16) {
    // M in (1, 16]
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          4>(out, a, b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    // M in (16, 64]
    if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    // M in (64, 128]
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          4>(out, a, b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 256) {
    // M in (128, 256]
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 64, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<64, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 64, 128>,
          cutlass::gemm::GemmShape<64, 32, 128>,
          4>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 512) {
    // M in (256, 512)
    if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          2>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          4>(out, a, b, scales_a, scales_b, bias);
    }
  } else {
    // M in (512, inf)
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          3>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          2>(out, a, b, scales_a, scales_b, bias);
    }
  }
}
#endif

#if defined CUDA_VERSION && CUDA_VERSION >= 12080 && SGL_CUDA_ARCH >= 1000
template <
    typename ElementType,
    typename OutElementType,
    typename AccumElementType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename EpilogueScheduleType,
    typename TileSchedulerType = void,
    bool WithBias = false,
    bool ScalarA = false>
struct DeviceGemmFp8RowwiseSm100 {
  static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");
  using TileShape = CTAShape;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using ElementComputeEpilogue = float;
  using VectorScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;
  using ScalarScaleA = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
  using ScaleA = std::conditional_t<ScalarA, ScalarScaleA, VectorScaleA>;

  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      OutElementType,
      OutElementType,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;

  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;

  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using Compute1MulAdd = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiply_add, OutElementType, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using Compute1Mul = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, OutElementType, float, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute = typename std::conditional_t<
      WithBias,
      cutlass::epilogue::fusion::Sm90EVT<Compute1MulAdd, ScaleA, EVTCompute0, Bias>,
      cutlass::epilogue::fusion::Sm90EVT<Compute1Mul, ScaleA, EVTCompute0>>;
  using ArgumentType = typename EVTCompute::Arguments;
  // MMA type
  using ElementAccumulator = AccumElementType;

  // Epilogue types
  using ElementCompute = float;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      OutElementType,
      LayoutD,
      AlignmentD,
      EpilogueScheduleType,
      EVTCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      ElementType,
      LayoutA,
      AlignmentA,
      ElementType,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopScheduleType>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  template <typename Descriptor, typename T>
  static auto args_from_tensor(tvm::ffi::TensorView const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    static_assert(
        std::is_same_v<Descriptor, ScaleA> || std::is_same_v<Descriptor, ScaleB> || std::is_same_v<Descriptor, Bias>);
    if constexpr (std::is_same_v<Descriptor, ScalarScaleA>) {
      return Arguments{{}, {data_ptr}, {}};
    } else {
      return Arguments{data_ptr};
    }
  }

 public:
  static ArgumentType prepare_args(
      tvm::ffi::TensorView const& a_scales,
      tvm::ffi::TensorView const& b_scales,
      tvm::ffi::Optional<tvm::ffi::TensorView> const& bias) {
    auto a_args = args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = args_from_tensor<ScaleB, float>(b_scales);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};

    if constexpr (WithBias) {
      auto bias_args = args_from_tensor<Bias, OutElementType>(bias.value());
      return ArgumentType{a_args, evt0_args, bias_args, {}};
    } else {
      return ArgumentType{a_args, evt0_args, {}};
    }
  }
};

template <typename GemmType, bool WithBias>
typename GemmType::Gemm::Arguments prepare_sm100_fp8_args(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  using Gemm = typename GemmType::Gemm;
  using ElementT = typename Gemm::ElementA;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename Gemm::ElementD;
  using ElementComputeEpilogue = float;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = StrideC;
  using StrideAux = StrideC;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);

  ElementT const* ptr_a = reinterpret_cast<ElementT const*>(a.data_ptr());
  ElementT const* ptr_b = reinterpret_cast<ElementT const*>(b.data_ptr());

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  StrideAux aux_stride = stride_d;

  typename GemmKernel::MainloopArguments mainloop_args{ptr_a, stride_a, ptr_b, stride_b};

  typename GemmKernel::ProblemShape prob_shape = {m, n, k, 1};
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::TileSchedulerArguments scheduler = {};

  auto ptr_c = static_cast<ElementOutput*>(out.data_ptr());

  auto prepare_epilogue_args = [&](const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
    if constexpr (WithBias) {
      RuntimeCheck(bias.has_value(), "Bias tensor is required but not provided.");
      return typename GemmKernel::EpilogueArguments{
          GemmType::prepare_args(scales_a, scales_b, bias.value()), ptr_c, stride_c, ptr_c, stride_d};
    } else {
      return typename GemmKernel::EpilogueArguments{
          GemmType::prepare_args(scales_a, scales_b, bias), ptr_c, stride_c, ptr_c, stride_d};
    }
  };

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      prob_shape,
      mainloop_args,
      prepare_epilogue_args(bias),
      hw_info,
      scheduler};
  return args;
}

template <typename Gemm, bool WithBias>
void launch_sm100_fp8_scaled_mm(
    tvm::ffi::TensorView& out,
    tvm::ffi::TensorView const& a,
    tvm::ffi::TensorView const& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  auto args = prepare_sm100_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);

  typename Gemm::Gemm gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = alloc_workspace_tensor(workspace_size, a.device());
  auto stream = LaunchKernel::resolve_device(a.device());
  auto can_implement = gemm_op.can_implement(args);
  RuntimeCheck(can_implement == cutlass::Status::kSuccess);
  auto status = gemm_op.run(args, workspace.data_ptr(), stream);
  RuntimeCheck(status == cutlass::Status::kSuccess);
}

template <typename OutType, bool ScalarA>
void sm100_fp8_dispatch_bias(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  using CTAShapeDefault = Shape<_256, _128, _64>;
  using ClusterShapeDefault = Shape<_2, _2, _1>;

  using CTAShape256 = Shape<_128, _128, _128>;
  using ClusterShape256 = Shape<_2, _1, _1>;

  using CTAShape64 = Shape<_64, _64, _128>;
  using ClusterShape64 = Shape<_1, _1, _1>;

  using CTAShape16 = Shape<_64, _64, _128>;
  using ClusterShape16 = Shape<_1, _4, _1>;

  using MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileSchedulerType = void;

  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = OutType;
  using AccumElementType = float;

  // Gemm type with bias
  using BiasGemmDefault = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShapeDefault,
      ClusterShapeDefault,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      true,
      ScalarA>;
  using BiasGemm256 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape256,
      ClusterShape256,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      true,
      ScalarA>;
  using BiasGemm64 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape64,
      ClusterShape64,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      true,
      ScalarA>;
  using BiasGemm16 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape16,
      ClusterShape16,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      true,
      ScalarA>;

  // Gemm type without bias
  using GemmDefault = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShapeDefault,
      ClusterShapeDefault,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      false,
      ScalarA>;
  using Gemm256 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape256,
      ClusterShape256,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      false,
      ScalarA>;
  using Gemm64 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape64,
      ClusterShape64,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      false,
      ScalarA>;
  using Gemm16 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape16,
      ClusterShape16,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      false,
      ScalarA>;

  // next power of 2 (minimum 16)
  uint32_t const m = a.size(0);
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));

  if (bias) {
    if (mp2 <= 16) {
      // m in [1, 16]
      return launch_sm100_fp8_scaled_mm<BiasGemm16, true>(out, a, b, scales_a, scales_b, bias);
    } else if (mp2 <= 64) {
      // m in (16, 64]
      return launch_sm100_fp8_scaled_mm<BiasGemm64, true>(out, a, b, scales_a, scales_b, bias);
    } else if (mp2 <= 256) {
      // m in (64, 256]
      return launch_sm100_fp8_scaled_mm<BiasGemm256, true>(out, a, b, scales_a, scales_b, bias);
    } else {
      // m in (256, inf]
      return launch_sm100_fp8_scaled_mm<BiasGemmDefault, true>(out, a, b, scales_a, scales_b, bias);
    }
  } else {
    if (mp2 <= 16) {
      // m in [1, 16]
      return launch_sm100_fp8_scaled_mm<Gemm16, false>(out, a, b, scales_a, scales_b, bias);
    } else if (mp2 <= 64) {
      // m in (16, 64]
      return launch_sm100_fp8_scaled_mm<Gemm64, false>(out, a, b, scales_a, scales_b, bias);
    } else if (mp2 <= 256) {
      // m in (64, 256]
      return launch_sm100_fp8_scaled_mm<Gemm256, false>(out, a, b, scales_a, scales_b, bias);
    } else {
      return launch_sm100_fp8_scaled_mm<GemmDefault, false>(out, a, b, scales_a, scales_b, bias);
    }
  }
}

template <typename OutType>
void sm100_fp8_dispatch_shape(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  if (scales_a.numel() == 1) {
    return sm100_fp8_dispatch_bias<OutType, true>(out, a, b, scales_a, scales_b, bias);
  }
  return sm100_fp8_dispatch_bias<OutType, false>(out, a, b, scales_a, scales_b, bias);
}

template <
    typename ElementType,
    typename OutElementType,
    typename AccumElementType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename EpilogueScheduleType,
    typename TileSchedulerType = void,
    bool WithBias = false,
    bool ScalarA = false>
struct DeviceGemmFp8RowwiseSm120 {
  static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");
  using TileShape = CTAShape;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using ElementComputeEpilogue = float;
  using VectorScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;
  using ScalarScaleA = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>;
  using ScaleA = std::conditional_t<ScalarA, ScalarScaleA, VectorScaleA>;

  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      OutElementType,
      OutElementType,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;

  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;

  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using Compute1MulAdd = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiply_add, OutElementType, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using Compute1Mul = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, OutElementType, float, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute = typename std::conditional_t<
      WithBias,
      cutlass::epilogue::fusion::Sm90EVT<Compute1MulAdd, ScaleA, EVTCompute0, Bias>,
      cutlass::epilogue::fusion::Sm90EVT<Compute1Mul, ScaleA, EVTCompute0>>;
  using ArgumentType = typename EVTCompute::Arguments;
  // MMA type
  using ElementAccumulator = AccumElementType;

  // Epilogue types
  using ElementCompute = float;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      OutElementType,
      LayoutD,
      AlignmentD,
      EpilogueScheduleType,
      EVTCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      ElementType,
      LayoutA,
      AlignmentA,
      ElementType,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopScheduleType>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  template <typename Descriptor, typename T>
  static auto args_from_tensor(tvm::ffi::TensorView const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    static_assert(
        std::is_same_v<Descriptor, ScaleA> || std::is_same_v<Descriptor, ScaleB> || std::is_same_v<Descriptor, Bias>);
    if constexpr (std::is_same_v<Descriptor, ScalarScaleA>) {
      return Arguments{{}, {data_ptr}, {}};
    } else {
      return Arguments{data_ptr};
    }
  }

 public:
  static ArgumentType prepare_args(
      tvm::ffi::TensorView const& a_scales,
      tvm::ffi::TensorView const& b_scales,
      tvm::ffi::Optional<tvm::ffi::TensorView> const& bias) {
    auto a_args = args_from_tensor<ScaleA, float>(a_scales);
    auto b_args = args_from_tensor<ScaleB, float>(b_scales);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};

    if constexpr (WithBias) {
      auto bias_args = args_from_tensor<Bias, OutElementType>(bias.value());
      return ArgumentType{a_args, evt0_args, bias_args, {}};
    } else {
      return ArgumentType{a_args, evt0_args, {}};
    }
  }
};

template <typename GemmType, bool WithBias>
typename GemmType::Gemm::Arguments prepare_sm120_fp8_args(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  using Gemm = typename GemmType::Gemm;
  using ElementT = typename Gemm::ElementA;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename Gemm::ElementD;
  using ElementComputeEpilogue = float;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = StrideC;
  using StrideAux = StrideC;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);

  ElementT const* ptr_a = reinterpret_cast<ElementT const*>(a.data_ptr());
  ElementT const* ptr_b = reinterpret_cast<ElementT const*>(b.data_ptr());

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
  StrideAux aux_stride = stride_d;

  typename GemmKernel::MainloopArguments mainloop_args{ptr_a, stride_a, ptr_b, stride_b};

  typename GemmKernel::ProblemShape prob_shape = {m, n, k, 1};
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::TileSchedulerArguments scheduler = {};

  auto ptr_c = static_cast<ElementOutput*>(out.data_ptr());

  auto prepare_epilogue_args = [&](const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
    if constexpr (WithBias) {
      RuntimeCheck(bias.has_value(), "Bias tensor is required but not provided.");
      return typename GemmKernel::EpilogueArguments{
          GemmType::prepare_args(scales_a, scales_b, bias.value()), ptr_c, stride_c, ptr_c, stride_d};
    } else {
      return typename GemmKernel::EpilogueArguments{
          GemmType::prepare_args(scales_a, scales_b, bias), ptr_c, stride_c, ptr_c, stride_d};
    }
  };

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      prob_shape,
      mainloop_args,
      prepare_epilogue_args(bias),
      hw_info,
      scheduler};
  return args;
}

template <typename Gemm, bool WithBias>
void launch_sm120_fp8_scaled_mm(
    tvm::ffi::TensorView& out,
    tvm::ffi::TensorView const& a,
    tvm::ffi::TensorView const& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  auto args = prepare_sm120_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);

  typename Gemm::Gemm gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = alloc_workspace_tensor(workspace_size, a.device());
  auto stream = LaunchKernel::resolve_device(a.device());
  auto can_implement = gemm_op.can_implement(args);
  RuntimeCheck(can_implement == cutlass::Status::kSuccess);
  auto status = gemm_op.run(args, workspace.data_ptr(), stream);
  RuntimeCheck(status == cutlass::Status::kSuccess);
}

template <typename OutType, bool ScalarA>
void sm120_fp8_dispatch_bias(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  using CTAShapeDefault = Shape<_128, _128, _128>;
  using ClusterShapeDefault = Shape<_1, _1, _1>;

  using MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileSchedulerType = void;

  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = OutType;
  using AccumElementType = float;

  using BiasGemmDefault = DeviceGemmFp8RowwiseSm120<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShapeDefault,
      ClusterShapeDefault,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      true,
      ScalarA>;

  using GemmDefault = DeviceGemmFp8RowwiseSm120<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShapeDefault,
      ClusterShapeDefault,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      false,
      ScalarA>;

  if (bias) {
    return launch_sm120_fp8_scaled_mm<BiasGemmDefault, true>(out, a, b, scales_a, scales_b, bias);
  } else {
    return launch_sm120_fp8_scaled_mm<GemmDefault, false>(out, a, b, scales_a, scales_b, bias);
  }
}

template <typename OutType>
void sm120_fp8_dispatch_shape(
    tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  if (scales_a.numel() == 1) {
    return sm120_fp8_dispatch_bias<OutType, true>(out, a, b, scales_a, scales_b, bias);
  }
  return sm120_fp8_dispatch_bias<OutType, false>(out, a, b, scales_a, scales_b, bias);
}
#endif

void fp8_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias) {
  RuntimeCheck(mat_a.device().device_type == kDLCUDA, "mat_a must be a CUDA tensor");
  RuntimeCheck(mat_b.device().device_type == kDLCUDA, "mat_b must be a CUDA tensor");
  RuntimeCheck(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  RuntimeCheck(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  RuntimeCheck(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  RuntimeCheck(mat_b.stride(0) == 1, "mat_b must be a column major tensor");
  RuntimeCheck(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  RuntimeCheck(
      (mat_a.size(1) * (mat_a.dtype().bits / 8)) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");
  RuntimeCheck(
      (mat_b.size(0) * (mat_b.dtype().bits / 8)) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
  RuntimeCheck(is_type<fp8_e4m3_t>(mat_a.dtype()), "mat_a must be Float8_e4m3fn");
  RuntimeCheck(is_type<fp8_e4m3_t>(mat_b.dtype()), "mat_b must be Float8_e4m3fn");
  RuntimeCheck(is_type<fp16_t>(out.dtype()) || is_type<bf16_t>(out.dtype()), "out_dtype must be Half or BFloat16");

  RuntimeCheck(
      scales_a.numel() == 1 || scales_a.numel() == mat_a.size(0),
      "scales_a must contain either one scalar scale or one scale per row");
#if SGL_CUDA_ARCH == 890
  RuntimeCheck(scales_a.numel() != 1 || mat_a.size(0) == 1, "scalar scales_a with M > 1 is unsupported on SM89");
#endif
  RuntimeCheck(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  RuntimeCheck(scales_a.is_contiguous(), "scales_a must be contiguous");
  RuntimeCheck(scales_b.is_contiguous(), "scales_b must be contiguous");
  RuntimeCheck(is_type<float>(scales_a.dtype()), "scales_a must be Float32");
  RuntimeCheck(is_type<float>(scales_b.dtype()), "scales_b must be Float32");

  if (bias) {
    RuntimeCheck(bias.value().numel() == mat_b.size(1), "size of bias is not matched");
    RuntimeCheck(bias.value().is_contiguous(), "bias must be contiguous");
    RuntimeCheck(bias.value().dtype() == out.dtype(), "bias dtype must match output dtype");
    RuntimeCheck(bias.value().device() == mat_a.device(), "bias and mat_a must be on the same device");
  }

  RuntimeCheck(out.dim() == 2, "out must be a 2D tensor");
  RuntimeCheck(out.size(0) == mat_a.size(0) && out.size(1) == mat_b.size(1), "out shape is not matched");
  RuntimeCheck(out.is_contiguous(), "out must be contiguous");
  RuntimeCheck(out.device() == mat_a.device(), "out and mat_a must be on the same device");
  RuntimeCheck(mat_b.device() == mat_a.device(), "mat_b and mat_a must be on the same device");
  RuntimeCheck(scales_a.device() == mat_a.device(), "scales_a and mat_a must be on the same device");
  RuntimeCheck(scales_b.device() == mat_a.device(), "scales_b and mat_a must be on the same device");
  RuntimeCheck(
      (out.size(1) * (out.dtype().bits / 8)) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

#if SGL_CUDA_ARCH >= 1200 && defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (is_type<bf16_t>(out.dtype())) {
    sm120_fp8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    sm120_fp8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
#elif SGL_CUDA_ARCH >= 1000 && defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (is_type<bf16_t>(out.dtype())) {
    sm100_fp8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    sm100_fp8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
#elif SGL_CUDA_ARCH >= 900 && defined CUDA_VERSION && CUDA_VERSION >= 12000
  cutlass_scaled_mm_sm90_fp8(out, mat_a, mat_b, scales_a, scales_b, bias);
#elif SGL_CUDA_ARCH == 890 && defined CUDA_VERSION && CUDA_VERSION >= 12040
  if (is_type<bf16_t>(out.dtype())) {
    sm89_fp8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    sm89_fp8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
#else
  Panic("No fp8_scaled_mm implementation for the active CUDA architecture");
#endif
}

}  // namespace sglang
