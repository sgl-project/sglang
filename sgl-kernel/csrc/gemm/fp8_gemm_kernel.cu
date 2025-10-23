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

#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "math.hpp"
#include "utils.h"

using namespace cute;

#if defined CUDA_VERSION && CUDA_VERSION >= 12040
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
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
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
    TORCH_CHECK(bias.has_value())
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
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  auto args = prepare_sm89_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);
  Gemm gemm_op;

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess)
}

template <typename OutType, typename CtaShape, typename WarpShape, int Stages>
void sm89_fp8_dispatch_bias(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
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
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
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

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
template <
    typename ElementType,
    typename OutElementType,
    typename AccumElementType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename EpilogueScheduleType,
    typename TileSchedulerType = void,
    bool WithBias = false>
struct DeviceGemmFp8RowwiseSm90 {
  static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

  // A matrix configuration
  using ElementA = ElementType;               // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = ElementType;                  // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                    // matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementC = void;                      // Element type for C matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C matrix operands
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<OutElementType>::value;  // Memory access granularity/alignment of C matrices in
                                                          // units of elements (up to 16 bytes)

  // Output matrix configuration
  using ElementOutput = OutElementType;            // Element type for output matrix operands
  using LayoutOutput = cutlass::layout::RowMajor;  // Layout type for output matrix operands
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // // Auxiliary matrix configuration and other fusion types
  // using ElementBias = float;

  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator = AccumElementType;  // Element type for internal accumulation
  using ElementCompute = float;                 // Element type for compute
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = CTAShape;                            // Threadblock-level tile size

  static constexpr bool PONG = false;
  static constexpr bool FAST_ACCUM = true;
  static constexpr bool USE_BIAS = false;

  using StageCountType = cutlass::gemm::collective::StageCountAuto;      // Stage count maximized
                                                                         // based on the tile size
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;  // Kernel to launch based on the default
                                                                         // setting in the Collective Builder
  // Implement rowwise scaling epilogue.
  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementOutput,
      ElementOutput,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementComputeEpilogue,  // First stage output type.
      ElementComputeEpilogue,  // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementOutput,
      ElementComputeEpilogue,  // Second stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  // With bias
  using ComputeWithBias = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiply_add,
      ElementOutput,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeWithBias = cutlass::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

  using EpilogueEVT = typename cutlass::platform::conditional<WithBias, EVTComputeWithBias, EVTCompute1>::type;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementComputeEpilogue,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementOutput,
      LayoutOutput,
      AlignmentOutput,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueEVT>::CollectiveOp;

  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using FastDefaultSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using FastPongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;

  using SlowAccum = DefaultSchedule;
  using FastAccum = FastPongSchedule;  // Default apply Pingpong

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm, bool WithBias>
typename Gemm::Arguments prepare_sm90_fp8_args(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ElementT = typename Gemm::ElementA;
  using ElementOutput = typename Gemm::ElementD;
  using ElementComputeEpilogue = float;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  ElementT const* ptr_a = reinterpret_cast<ElementT const*>(a.data_ptr());
  ElementT const* ptr_b = reinterpret_cast<ElementT const*>(b.data_ptr());
  ElementOutput const* ptr_bias = nullptr;
  if constexpr (WithBias) {
    TORCH_CHECK(bias.has_value())
    ptr_bias = reinterpret_cast<ElementOutput const*>(bias.value().data_ptr());
  }
  ElementOutput* ptr_d = reinterpret_cast<ElementOutput*>(out.data_ptr());
  ElementComputeEpilogue const* ptr_scales_a = reinterpret_cast<ElementComputeEpilogue const*>(scales_a.data_ptr());
  ElementComputeEpilogue const* ptr_scales_b = reinterpret_cast<ElementComputeEpilogue const*>(scales_b.data_ptr());

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC stride_c;
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
  typename Gemm::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {ptr_a, stride_a, ptr_b, stride_b},
      {{},  // epilogue.thread
       nullptr,
       stride_c,
       ptr_d,
       stride_d}};
  if constexpr (WithBias) {
    args.epilogue.thread = {
        {ptr_scales_a},
        {
            {ptr_scales_b},
            {},  // Accumulator
            {}   // Multiplies
        },
        {ptr_bias},
        {},  // Multiplies
    };
  } else {
    args.epilogue.thread = {
        {ptr_scales_a},
        {
            {ptr_scales_b},
            {},  // Accumulator
            {}   // Multiplies
        },
        {},  // Multiplies
    };
  }

  return args;
}

template <typename Gemm, bool WithBias>
void launch_sm90_fp8_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  auto args = prepare_sm90_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);
  Gemm gemm_op;

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

  auto status = gemm_op.run(args, workspace.data_ptr(), stream);

  TORCH_CHECK(status == cutlass::Status::kSuccess)
}

template <
    typename OutType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename TileSchedulerType>
void sm90_fp8_dispatch_bias(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias,
    bool fast_accum = true,
    bool use_persistent = false) {
  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = OutType;
  using AccumElementType = float;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;

  if (bias) {
    using Gemm = typename DeviceGemmFp8RowwiseSm90<
        ElementInput,
        ElementOutput,
        AccumElementType,
        CTAShape,
        ClusterShape,
        MainloopScheduleType,
        EpilogueScheduleType,
        TileSchedulerType,
        true>::Gemm;
    return launch_sm90_fp8_scaled_mm<Gemm, true>(out, a, b, scales_a, scales_b, bias);
  } else {
    using Gemm = typename DeviceGemmFp8RowwiseSm90<
        ElementInput,
        ElementOutput,
        AccumElementType,
        CTAShape,
        ClusterShape,
        MainloopScheduleType,
        EpilogueScheduleType,
        TileSchedulerType,
        false>::Gemm;
    return launch_sm90_fp8_scaled_mm<Gemm, false>(out, a, b, scales_a, scales_b, bias);
  }
}

template <typename OutType>
void sm90_fp8_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  uint32_t const m = a.size(0);
  using FastPingpongScheduler = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using FastBasicScheduler = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using PersistentTileScheduler = cutlass::gemm::PersistentScheduler;
  using BasicTileScheduler = void;
  if (m <= 1) {
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_64, _64, _128>,
        Shape<_1, _8, _1>,
        FastBasicScheduler,
        BasicTileScheduler>(out, a, b, scales_a, scales_b, bias);
  }
  if (m <= 64) {
    // m in [1, 64]
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_64, _64, _128>,
        Shape<_1, _4, _1>,
        FastPingpongScheduler,
        PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
  } else if (m <= 256) {
    // m in (64, 256]
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_64, _64, _128>,
        Shape<_1, _1, _1>,
        FastPingpongScheduler,
        PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
  } else if (m <= 1024) {
    // m in (256, 1024]
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_128, _128, _128>,
        Shape<_1, _1, _1>,
        FastPingpongScheduler,
        PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
  } else {
    // m in (1024, inf)
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_128, _128, _128>,
        Shape<_2, _1, _1>,
        FastPingpongScheduler,
        PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
  }
}
#endif

#if defined CUDA_VERSION && CUDA_VERSION >= 12080
template <
    typename ElementType,
    typename OutElementType,
    typename AccumElementType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename EpilogueScheduleType,
    typename TileSchedulerType = void,
    bool WithBias = false>
struct DeviceGemmFp8RowwiseSm100 {
  static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");
  using TileShape = CTAShape;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using ElementComputeEpilogue = float;
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

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
  static auto args_from_tensor(torch::Tensor const& tensor) {
    using Arguments = typename Descriptor::Arguments;
    auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    static_assert(
        std::is_same_v<Descriptor, ScaleA> || std::is_same_v<Descriptor, ScaleB> || std::is_same_v<Descriptor, Bias>);
    return Arguments{data_ptr};
  }

 public:
  static ArgumentType prepare_args(
      torch::Tensor const& a_scales,
      torch::Tensor const& b_scales,
      std::optional<torch::Tensor> const& bias = std::nullopt) {
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
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
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

  auto prepare_epilogue_args = [&](const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    if constexpr (WithBias) {
      TORCH_CHECK(bias.has_value(), "Bias tensor is required but not provided.");
      return typename GemmKernel::EpilogueArguments{
          GemmType::prepare_args(scales_a, scales_b, bias.value()), ptr_c, stride_c, ptr_c, stride_d};
    } else {
      return typename GemmKernel::EpilogueArguments{
          GemmType::prepare_args(scales_a, scales_b), ptr_c, stride_c, ptr_c, stride_d};
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
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  auto args = prepare_sm100_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);

  typename Gemm::Gemm gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess)
  auto status = gemm_op.run(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess)
}

template <typename OutType>
void sm100_fp8_dispatch_bias(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
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
      true>;
  using BiasGemm256 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape256,
      ClusterShape256,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      true>;
  using BiasGemm64 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape64,
      ClusterShape64,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      true>;
  using BiasGemm16 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape16,
      ClusterShape16,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      true>;

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
      false>;
  using Gemm256 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape256,
      ClusterShape256,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      false>;
  using Gemm64 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape64,
      ClusterShape64,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      false>;
  using Gemm16 = DeviceGemmFp8RowwiseSm100<
      ElementInput,
      ElementOutput,
      AccumElementType,
      CTAShape16,
      ClusterShape16,
      MainloopScheduleType,
      EpilogueScheduleType,
      TileSchedulerType,
      false>;

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
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  return sm100_fp8_dispatch_bias<OutType>(out, a, b, scales_a, scales_b, bias);
}
#endif

torch::Tensor fp8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  TORCH_CHECK(
      (mat_a.size(1) * mat_a.element_size()) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(
      (mat_b.size(0) * mat_b.element_size()) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(mat_a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
  TORCH_CHECK(mat_b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

  TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
  TORCH_CHECK(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b msut be contiguous");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

  if (bias) {
    TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dtype() == out_dtype, "bias dtype must match output dtype");
  }

  torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));
  TORCH_CHECK((out.size(1) * out.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  auto sm_version = getSMVersion();

#if defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (sm_version == 100
#if CUDA_VERSION >= 12090
      || sm_version == 103
#endif
  ) {
    if (out_dtype == torch::kBFloat16) {
      sm100_fp8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm100_fp8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
    return out;
  }
#endif

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
  if (sm_version >= 90) {
    if (out_dtype == torch::kBFloat16) {
      sm90_fp8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm90_fp8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
    return out;
  }
#endif

#if defined CUDA_VERSION && CUDA_VERSION >= 12040
  if (sm_version == 89) {
    if (out_dtype == torch::kBFloat16) {
      sm89_fp8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm89_fp8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
    return out;
  }
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(false, "No implemented fp8_scaled_mm for current compute capability: ", sm_version);
}
