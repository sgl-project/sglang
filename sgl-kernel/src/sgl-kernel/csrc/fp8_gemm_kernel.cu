/*
 * Copyright (c) 2022-2024, Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC__

// #include "fp8_rowwise_gemm_kernel_template_sm89.h"
// #include "fp8_rowwise_gemm_kernel_template_sm90.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"


#include "utils.hpp"
using namespace cute;

template <typename ElementType, typename OutElementType, typename AccumElementType, typename CTAShape,
    typename ClusterShape, typename MainloopScheduleType, typename EpilogueScheduleType,
    typename TileSchedulerType = void>
struct DeviceGemmFp8RowwiseSm90
{
    static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

    // A matrix configuration
    using ElementA = ElementType;                      // Element type for A matrix operand
    using LayoutA = cutlass::layout::RowMajor;         // Layout type for A matrix operand
    static constexpr int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A
                                                       // matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = ElementType;                      // Element type for B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;      // Layout type for B matrix operand
    static constexpr int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B
                                                       // matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using ElementC = void;                                   // Element type for C matrix operands
    using LayoutC = cutlass::layout::RowMajor;               // Layout type for C matrix operands
    static constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<OutElementType>::value; // Memory access granularity/alignment of C matrices in
                                                             // units of elements (up to 16 bytes)

    // Output matrix configuration
    using ElementOutput = OutElementType;           // Element type for output matrix operands
    using LayoutOutput = cutlass::layout::RowMajor; // Layout type for output matrix operands
    static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    // Auxiliary matrix configuration and other fusion types
    using ElementBias = float;

    // Multiply-accumulate blocking/pipelining details
    using ElementAccumulator = AccumElementType; // Element type for internal accumulation
    using ElementCompute = float;                // Element type for compute
    using ElementComputeEpilogue = float;
    using ArchTag = cutlass::arch::Sm90;         // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
    using TileShape = CTAShape;                           // Threadblock-level tile size
    using TileScheduler = TileSchedulerType;

    static constexpr bool PONG = false;
    static constexpr bool FAST_ACCUM = true;
    static constexpr bool USE_BIAS = false;

    using StageCountType = cutlass::gemm::collective::StageCountAuto;     // Stage count maximized
                                                                          // based on the tile size
    using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto; // Kernel to launch based on the default
                                                                          // setting in the Collective Builder
    // Implement rowwise scaling epilogue.
    using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementComputeEpilogue, ElementComputeEpilogue,
        cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

    using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementComputeEpilogue, ElementComputeEpilogue,
        cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

    using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementBias, ElementBias,
        cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

    using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,
        ElementComputeEpilogue, // First stage output type.
        ElementComputeEpilogue, // First stage input types.
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

    using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementOutput,
        ElementComputeEpilogue, // Second stage input types.
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

    using ComputeBias = cutlass::epilogue::fusion::Sm90Compute<cutlass::plus,
        ElementOutput, // Final (optional) stage output type.
        ElementBias,   // Final stage input types.
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTComputeBias = cutlass::epilogue::fusion::Sm90EVT<ComputeBias, Bias, EVTCompute1>;

    using EpilogueEVT = EVTCompute1;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementComputeEpilogue, ElementC, LayoutC, AlignmentC, ElementOutput, LayoutOutput,
        AlignmentOutput, cutlass::epilogue::TmaWarpSpecialized, EpilogueEVT>::CollectiveOp;

    using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
    using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using FastDefaultSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using FastPongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;

    using SlowAccum = DefaultSchedule;
    using FastAccum = FastDefaultSchedule;
    using MainLoopSchedule = cute::conditional_t<FAST_ACCUM, FastAccum, SlowAccum>;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA,
        LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainLoopSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, // Indicates ProblemShape
        CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename ElementType, typename OutElementType, typename AccumElementType, typename CtaShape,
    typename WarpShape, int Stages>
// template <typename ElementType, typename OutElementType, typename AccumElementType, typename CtaShape,
//     typename WarpShape, int Stages, bool WithBias>
struct DeviceGemmFp8RowwiseSm89
{
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

    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<CtaShape, WarpShape, ElementC,
        AlignmentC, EVTEpilogueStages>;

    // Definition of EVT
    using accSrc = cutlass::epilogue::threadblock::VisitorAccFetch;

    using ComputeBScale = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
    using bScaleSrc = cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ElementComputeEpilogue,
        Stride<_0, _1, _0>>;
    using EpilogueBScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeBScale, accSrc, bScaleSrc>;

    using ComputeAScale = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementC,
        ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
    using aScaleSrc = cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, ElementComputeEpilogue,
        Stride<_1, _0, _0>>;
    using EpilogueAScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeAScale, EpilogueBScale, aScaleSrc>;

    // // With bias
    // using biasSrc = cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ElementOutput, Stride<_0, _1, _0>>;
    // using ComputeAScaleWithBias = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiply_add, ElementC,
    //     ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
    // using EpilogueAScaleWithBias = cutlass::epilogue::threadblock::Sm80EVT<ComputeAScaleWithBias, EpilogueBScale, aScaleSrc, biasSrc>;


    using dTar = cutlass::epilogue::threadblock::VisitorAuxStore<OutputTileThreadMap, ElementC,
        cutlass::FloatRoundStyle::round_to_nearest, Stride<int64_t, _1, _0>>;
    using EpilogueStore = cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScale>;
    // using EpilogueStore = cutlass::platform::conditional<WithBias, cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScaleWithBias>,
    //     cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScale>>::type;
    

    using EpilogueOp = EpilogueStore;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, AlignmentA, ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
        ElementC, LayoutC, AlignmentC, ElementAccumulator, ElementComputeEpilogue, OperatorClass, ArchTag, CtaShape,
        WarpShape, InstructionShape, EpilogueOp, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, Stages,
        cutlass::arch::OpMultiplyAdd, EVTEpilogueStages>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};


template <typename Gemm>
typename Gemm::Arguments prepare_sm89_fp8_args(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias)
{
    using ElementT = typename Gemm::ElementA;
    using ElementOutput = typename Gemm::ElementD;
    using ElementComputeEpilogue = float;

    // int const lda = k;
    // int const ldb = k;
    // int const ldc = n;
    int32_t m = a.size(0);
    int32_t n = b.size(1);
    int32_t k = a.size(1);

    int64_t lda = a.stride(0);
    int64_t ldb = b.stride(1);
    int64_t ldc = out.stride(0);

    ElementT const* ptr_a = reinterpret_cast<ElementT const*>(a.data_ptr());
    ElementT const* ptr_b = reinterpret_cast<ElementT const*>(b.data_ptr());
    ElementOutput* ptr_d = reinterpret_cast<ElementOutput*>(out.data_ptr());
    ElementComputeEpilogue const* ptr_scales_a = reinterpret_cast<ElementComputeEpilogue const*>(scales_a.data_ptr());
    ElementComputeEpilogue const* ptr_scales_b = reinterpret_cast<ElementComputeEpilogue const*>(scales_b.data_ptr());

    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, // Mode
        {m, n, k},                                                         // Problem size
        1,                                                                 // Split-k factor
        {},                                                                // Epilogue args
        ptr_a,                              // a pointer
        ptr_b,                              // b pointer
        nullptr,                                                           // c pointer (unused)
        nullptr,                                                           // d pointer (unused)
        m * k,                                                             // batch stride a (unused)
        n * k,                                                             // batch stride b (unused)
        m * n,                                                             // batch stride c (unused)
        m * n,                                                             // batch stride d (unused)
        lda,                                                               // stride a
        ldb,                                                               // stride b
        ldc,                                                               // stride c (unused)
        ldc);                                                              // stride d (unused)

    args.epilogue = {
        {
            {
                {}, // Accumulator
                {ptr_scales_b, ElementComputeEpilogue(0),
                    {_0{}, _1{}, _0{}}},
                {} // Multiplies
            },
            {ptr_scales_a, ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
            {} // Multiplies
        },
        {ptr_d, {n, _1{}, _0{}}}};
    return args;
}

template <typename OutType, typename CtaShape, typename WarpShape, int Stages>
void launch_sm89_fp8_scaled_mm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias)
{
    using ElementInput = cutlass::float_e4m3_t;
    using ElementOutput = OutType;
    using AccumElementType = float;

    using Gemm = typename DeviceGemmFp8RowwiseSm89<ElementInput, ElementOutput, AccumElementType, CtaShape, WarpShape,
        Stages>::Gemm;

    auto args = prepare_sm89_fp8_args<Gemm>(out, a, b, scales_a, scales_b, bias);
    Gemm gemm_op;
    // CUTLASS_CHECK(gemm_op.can_implement(args));

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto const workspace_options =
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(workspace_size, workspace_options);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto can_implement = gemm_op.can_implement(args);
    TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

    // auto status = gemm_op.run(args, workspace.data_ptr(), stream);
    auto status = gemm_op(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess)
    // return typedFp8RowwiseGemmKernelLauncher(
    //     Gemm{}, args, D, A, B, C_bias, workspace, workspaceBytes, stream, occupancy);
}


template <typename OutType>
void s89_dispatch_shape(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias) {
    uint32_t const m = a.size(0);
    uint32_t const mp2 =
        std::max(static_cast<uint32_t>(32), next_pow_2(m));  // next power of 2

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

  if (mp2 <= 16) {
    // M in [1, 16]
    if (np2 <= 8192) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<16, 64, 128>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 24576) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<16, 128, 64>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<32, 64, 128>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (mp2 <= 32) {
    // M in (16, 32]
    if (np2 <= 8192) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<32, 64, 128>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 16384) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<32, 128, 128>, cutlass::gemm::GemmShape<32, 64, 64>, 4>(out, a, b, scales_a, scales_b, bias);
    } else {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<32, 64, 128>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (mp2 <= 64) {
    // M in (32, 64]
    if (np2 <= 8192) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<32, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 16384) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<64, 128, 128>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    } else {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<32, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (mp2 <= 128) {
    // M in (64, 128]
    if (np2 <= 8192) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<64, 128, 128>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 16384) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<128, 64, 128>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (mp2 <= 256) {
    // M in (128, 256]
    if (np2 <= 4096) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<64, 128, 128>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    } else {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  } else {
    // M in (256, inf)
    if (np2 <= 4096) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 8192) {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    } else {
        return launch_sm89_fp8_scaled_mm<OutType, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  }
}

template <typename Gemm>
typename Gemm::Arguments prepare_sm90_fp8_args(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias)
{
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
    ElementOutput* ptr_d = reinterpret_cast<ElementOutput*>(out.data_ptr());
    ElementComputeEpilogue const* ptr_scales_a = reinterpret_cast<ElementComputeEpilogue const*>(scales_a.data_ptr());
    ElementComputeEpilogue const* ptr_scales_b = reinterpret_cast<ElementComputeEpilogue const*>(scales_b.data_ptr());

    // TODO: confirm correctess
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
    StrideC stride_c;
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
    typename Gemm::Arguments args
        = {cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1}, {ptr_a, stride_a, ptr_b, stride_b},
            {{}, // epilogue.thread
                nullptr, stride_c, ptr_d, stride_d}};
    args.epilogue.thread = {
        {ptr_scales_a},
        {
            {ptr_scales_b}, {}, // Accumulator
            {}                                                                             // Multiplies
        },
        {},                                                                                // Multiplies
    };
    return args;
}

template <typename OutType, typename CTAShape, typename ClusterShape>
void launch_sm90_fp8_scaled_mm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias)
{
    using ElementInput = cutlass::float_e4m3_t;
    using ElementOutput = OutType;
    using AccumElementType = float;
    using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedulerType = void;
    using Gemm = typename DeviceGemmFp8RowwiseSm90<ElementInput, ElementOutput, AccumElementType, CTAShape,
        ClusterShape, MainloopScheduleType, EpilogueScheduleType, TileSchedulerType>::Gemm;
    auto args = prepare_sm90_fp8_args<Gemm>(out, a, b, scales_a, scales_b, bias);

    // Launch the CUTLASS GEMM kernel.
    Gemm gemm_op;
    // CUTLASS_CHECK(gemm_op.can_implement(args));

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto const workspace_options =
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(workspace_size, workspace_options);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto can_implement = gemm_op.can_implement(args);
    TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

    auto status = gemm_op.run(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess)
    // cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
    // CUTLASS_CHECK(status);
//     return typedFp8RowwiseGemmKernelLauncher(
//         Gemm{}, args, D, A, B, C_bias, workspace, workspaceBytes, stream, occupancy);
// #else  // COMPILE_HOPPER_TMA_GEMMS
//     throw std::runtime_error(
//         "[TensorRT-LLm Error][Fp8RowwiseGemmKernelLauncherSm90] Please recompile with support for hopper by passing "
//         "90-real as an arch to build_wheel.py.");
// #endif // COMPILE_HOPPER_TMA_GEMMS
}

template <typename OutType>
void s90_dispatch_shape(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias) {
    uint32_t const m = a.size(0);
    uint32_t const mp2 =
        std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2

    if (mp2 <= 64) {
        // m in [1, 64]
        return launch_sm90_fp8_scaled_mm<OutType, Shape<_64, _64, _128>, Shape<_1, _8, _1>>(out, a, b, scales_a, scales_b, bias);
    } else if (mp2 <= 128) {
        // m in (64, 128]
        return launch_sm90_fp8_scaled_mm<OutType, Shape<_64, _128, _128>, Shape<_2, _1, _1>>(out, a, b, scales_a, scales_b, bias);
    } else {
        // m in (128, inf)
        return launch_sm90_fp8_scaled_mm<OutType, Shape<_128, _128, _128>, Shape<_2, _1, _1>>(out, a, b, scales_a, scales_b, bias);
    }
}

torch::Tensor fp8_scaled_mm(const torch::Tensor& mat_a, const torch::Tensor& mat_b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b, const torch::Dtype& out_dtype,
                             const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_a must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  TORCH_CHECK(mat_a.size(1) % 16 == 0, "mat_a.size(1) must be multiple of 16 for memory alignment");
//   TORCH_CHECK(mat_b.size(0) % 16 == 0, "mat_b.size(0) must be multiple of 16 for memory alignment");
//TODO: % 8
  TORCH_CHECK(mat_b.size(1) % 16 == 0, "mat_b.size(1) must be multiple of 16 for memory alignment");  // out.stride(0)
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

  auto sm_version = getSMVersion();

  if (sm_version >= 90) {
    if (out_dtype == torch::kBFloat16) {
        s90_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
        s90_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (sm_version == 89) {
    if (out_dtype == torch::kBFloat16) {
        s89_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
        s89_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "No implemented int8_scaled_mm for current compute capability: ", sm_version);
  }

  return out;
}
