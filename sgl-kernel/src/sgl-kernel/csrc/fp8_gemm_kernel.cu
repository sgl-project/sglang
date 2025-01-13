// Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_template.h
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_kernel_template_sm89.h
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm_kernel_template_sm90.h

#pragma once
#include <chrono>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

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

template <typename ElementType, typename OutElementType, typename AccumElementType, typename CtaShape,
    typename WarpShape, int Stages, bool WithBias,
    typename FP8MathOperator = cutlass::arch::OpMultiplyAdd,
    template <typename...> typename EpilogueVisitor = cutlass::epilogue::threadblock::Sm80EVT,
    typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
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

    // With bias
    using biasSrc = cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ElementOutput, Stride<_0, _1, _0>>;
    using ComputeAScaleWithBias = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiply_add, ElementC,
        ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
    using EpilogueAScaleWithBias = cutlass::epilogue::threadblock::Sm80EVT<ComputeAScaleWithBias, EpilogueBScale, aScaleSrc, biasSrc>;

    using dTar = cutlass::epilogue::threadblock::VisitorAuxStore<OutputTileThreadMap, ElementC,
        cutlass::FloatRoundStyle::round_to_nearest, Stride<int64_t, _1, _0>>;
    using EpilogueStore = typename cutlass::platform::conditional<WithBias, cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScaleWithBias>,
        cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScale>>::type;
    
    using EpilogueOp = EpilogueStore;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, AlignmentA, ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
        ElementC, LayoutC, AlignmentC, ElementAccumulator, ElementComputeEpilogue, OperatorClass, ArchTag, CtaShape,
        WarpShape, InstructionShape, EpilogueOp, ThreadblockSwizzle,
        Stages, FP8MathOperator, EVTEpilogueStages>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};


template <typename Gemm, bool WithBias>
typename Gemm::Arguments prepare_sm89_fp8_args(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias)
{
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


    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, // Mode
        {m, n, k},                                                         // Problem size
        1,                                                                 // Split-k factor
        {},                                                                // Epilogue args
        ptr_a,                                                             // a pointer
        ptr_b,                                                             // b pointer
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
    if constexpr (WithBias) {
        args.epilogue = {
        {
            {
                {}, // Accumulator
                {ptr_scales_b, ElementComputeEpilogue(0),
                    {_0{}, _1{}, _0{}}},
                {} // Multiplies
            },
            {ptr_scales_a, ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
            {ptr_bias, ElementOutput(0), {_0{}, _1{}, _0{}}},
            {} // Multiplies
        },
        {ptr_d, {n, _1{}, _0{}}}};
    } else {
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
    }

    return args;
}

template <typename Gemm, bool WithBias>
void launch_sm89_fp8_scaled_mm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias)
{
    auto args = prepare_sm89_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);
    Gemm gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto const workspace_options =
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(workspace_size, workspace_options);
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto can_implement = gemm_op.can_implement(args);
    TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

    auto status = gemm_op(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess)
}

template <typename OutType, typename CtaShape, typename WarpShape, int Stages>
void sm89_dispatch_bias(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias) {
    using ElementInput = cutlass::float_e4m3_t;
    using ElementOutput = OutType;
    using AccumElementType = float;
    if (bias) {
        using Gemm = typename DeviceGemmFp8RowwiseSm89<ElementInput, ElementOutput, AccumElementType, CtaShape, WarpShape,
            Stages, true>::Gemm;
        return launch_sm89_fp8_scaled_mm<Gemm, true>(out, a, b, scales_a, scales_b, bias);
    } else {
        using Gemm = typename DeviceGemmFp8RowwiseSm89<ElementInput, ElementOutput, AccumElementType, CtaShape, WarpShape,
            Stages, false>::Gemm;
        return launch_sm89_fp8_scaled_mm<Gemm, false>(out, a, b, scales_a, scales_b, bias);
    }
}


template <typename OutType>
void sm89_dispatch_shape(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
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
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<16, 64, 128>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 24576) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<16, 128, 64>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<32, 64, 128>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (mp2 <= 32) {
    // M in (16, 32]
    if (np2 <= 8192) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<32, 64, 128>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 16384) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<32, 128, 128>, cutlass::gemm::GemmShape<32, 64, 64>, 4>(out, a, b, scales_a, scales_b, bias);
    } else {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<32, 64, 128>, cutlass::gemm::GemmShape<16, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (mp2 <= 64) {
    // M in (32, 64]
    if (np2 <= 8192) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<32, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 16384) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<64, 128, 128>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    } else {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<32, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (mp2 <= 128) {
    // M in (64, 128]
    if (np2 <= 8192) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<64, 128, 128>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 16384) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<128, 64, 128>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (mp2 <= 256) {
    // M in (128, 256]
    if (np2 <= 4096) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<64, 128, 128>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    } else {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  } else {
    // M in (256, inf)
    if (np2 <= 4096) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    } else if (np2 <= 8192) {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 3>(out, a, b, scales_a, scales_b, bias);
    } else {
        return sm89_dispatch_bias<OutType, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 5>(out, a, b, scales_a, scales_b, bias);
    }
  }
}

template <typename ElementType, typename OutElementType, typename AccumElementType, typename CTAShape,
    typename ClusterShape, typename MainloopScheduleType, typename EpilogueScheduleType,
    typename TileSchedulerType = void, bool WithBias = false>
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

    // // Auxiliary matrix configuration and other fusion types
    // using ElementBias = float;

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

    using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementOutput, ElementOutput,
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

    // With bias
    using ComputeWithBias = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiply_add, ElementOutput,
        ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTComputeWithBias = cutlass::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

    using EpilogueEVT = typename cutlass::platform::conditional<WithBias, EVTComputeWithBias, EVTCompute1>::type;

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

template <typename Gemm, bool WithBias>
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
    ElementOutput const* ptr_bias = nullptr;
    if constexpr (WithBias) {
        TORCH_CHECK(bias.has_value())
        ptr_bias = reinterpret_cast<ElementOutput const*>(bias.value().data_ptr());
    }
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
    if constexpr (WithBias) {
        args.epilogue.thread = {
            {ptr_scales_a},
            {
                {ptr_scales_b}, {}, // Accumulator
                {}                                                                             // Multiplies
            },
            {ptr_bias},
            {},                                                                                // Multiplies
        };
    } else {
        args.epilogue.thread = {
            {ptr_scales_a},
            {
                {ptr_scales_b}, {}, // Accumulator
                {}                                                                             // Multiplies
            },
            {},                                                                                // Multiplies
        };
    }

    return args;
}

template <typename Gemm, bool WithBias>
void launch_sm90_fp8_scaled_mm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias)
{
    auto args = prepare_sm90_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);
    Gemm gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto const workspace_options =
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(workspace_size, workspace_options);
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto can_implement = gemm_op.can_implement(args);
    TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

    auto status = gemm_op.run(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess)
}


template <typename OutType, typename CTAShape, typename ClusterShape>
void sm90_dispatch_bias(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias) {
    using ElementInput = cutlass::float_e4m3_t;
    using ElementOutput = OutType;
    using AccumElementType = float;
    using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedulerType = void;
    if (bias) {
        using Gemm = typename DeviceGemmFp8RowwiseSm90<ElementInput, ElementOutput, AccumElementType, CTAShape,
            ClusterShape, MainloopScheduleType, EpilogueScheduleType, TileSchedulerType, true>::Gemm;
        return launch_sm90_fp8_scaled_mm<Gemm, true>(out, a, b, scales_a, scales_b, bias);
    } else {
        using Gemm = typename DeviceGemmFp8RowwiseSm90<ElementInput, ElementOutput, AccumElementType, CTAShape,
            ClusterShape, MainloopScheduleType, EpilogueScheduleType, TileSchedulerType, false>::Gemm;
        return launch_sm90_fp8_scaled_mm<Gemm, false>(out, a, b, scales_a, scales_b, bias);
    }
}

template <typename OutType>
void sm90_dispatch_shape(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias) {
    uint32_t const m = a.size(0);
    uint32_t const mp2 =
        std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2

    // if (mp2 <= 64) {
    //     // m in [1, 64]
    //     return sm90_dispatch_bias<OutType, Shape<_64, _64, _128>, Shape<_1, _8, _1>>(out, a, b, scales_a, scales_b, bias);
    // } else if (mp2 <= 128) {
    //     // m in (64, 128]
    //     return sm90_dispatch_bias<OutType, Shape<_64, _128, _128>, Shape<_2, _1, _1>>(out, a, b, scales_a, scales_b, bias);
    // } else {
    //     // m in (128, inf)
    //     return sm90_dispatch_bias<OutType, Shape<_128, _128, _128>, Shape<_2, _1, _1>>(out, a, b, scales_a, scales_b, bias);
    // }
}

#define DISPATCH_FP8_GEMM_CONFIG(TB_M, TB_N, TB_K, WP_M, WP_N, WP_K, STAGES) \
    sm89_dispatch_bias<ElementOutput, cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>, \
        cutlass::gemm::GemmShape<WP_M, WP_N, WP_K>, STAGES>(out, mat_a, mat_b, scales_a, scales_b, bias)
// generate all stages for a group of configs
#define DISPATCH_FP8_GEMM_GROUP(GROUP_ID, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, BASE_CASE) \
    case BASE_CASE:     DISPATCH_FP8_GEMM_CONFIG(CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 2); break; \
    case BASE_CASE + 1: DISPATCH_FP8_GEMM_CONFIG(CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 3); break; \
    case BASE_CASE + 2: DISPATCH_FP8_GEMM_CONFIG(CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 4); break; \
    case BASE_CASE + 3: DISPATCH_FP8_GEMM_CONFIG(CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 5); break; \
    case BASE_CASE + 4: DISPATCH_FP8_GEMM_CONFIG(CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 6); break; \
    case BASE_CASE + 5: DISPATCH_FP8_GEMM_CONFIG(CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 7); break;

template <typename ElementOutput>
void sm89_dispatch_shape_explicit(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                            const torch::Tensor& scales_a, const torch::Tensor& scales_b,
                            const c10::optional<torch::Tensor>& bias,
                            int config_id) {
#ifdef SGL_DEBUG_BUILD
    switch(config_id) {
        case 1:
            DISPATCH_FP8_GEMM_CONFIG(32, 64, 128, 16, 64, 64, 5);
            break;
        case 2:
            DISPATCH_FP8_GEMM_CONFIG(16, 64, 128, 16, 64, 64, 5);
            break;
        case 3:
            DISPATCH_FP8_GEMM_CONFIG(64, 64, 128, 32, 64, 64, 5);
            break;
        case 4:
            DISPATCH_FP8_GEMM_CONFIG(64, 128, 64, 32, 64, 64, 5);
            break;
        case 5:
            DISPATCH_FP8_GEMM_CONFIG(128, 128, 64, 64, 32, 64, 2);
            break;
        case 6:
            DISPATCH_FP8_GEMM_CONFIG(64, 128, 64, 32, 64, 64, 6);
            break;
        default:
            throw std::runtime_error("Invalid config_id in debug mode: " + std::to_string(config_id));
    }
#else
    switch(config_id) {
        // Group 1: CtaShape32x128x64_WarpShape32x32x64
        DISPATCH_FP8_GEMM_GROUP(1, 32, 128, 64, 32, 32, 64, 1);

        // Group 2: CtaShape64x128x64_WarpShape32x64x64
        DISPATCH_FP8_GEMM_GROUP(2, 64, 128, 64, 32, 64, 64, 7);

        // Group 3: CtaShape64x64x128_WarpShape32x64x64
        DISPATCH_FP8_GEMM_GROUP(3, 64, 64, 128, 32, 64, 64, 13);

        // Group 4: CtaShape64x128x64_WarpShape64x32x64
        DISPATCH_FP8_GEMM_GROUP(4, 64, 128, 64, 64, 32, 64, 19);

        // Group 5: CtaShape128x64x64_WarpShape64x32x64
        DISPATCH_FP8_GEMM_GROUP(5, 128, 64, 64, 64, 32, 64, 25);

        // Group 6: CtaShape128x128x64_WarpShape64x32x64
        DISPATCH_FP8_GEMM_GROUP(6, 128, 128, 64, 64, 32, 64, 31);

        // Group 7: CtaShape128x128x64_WarpShape64x64x64
        DISPATCH_FP8_GEMM_GROUP(7, 128, 128, 64, 64, 64, 64, 37);

        // Group 8: CtaShape128x128x64_WarpShape128x32x64
        DISPATCH_FP8_GEMM_GROUP(8, 128, 128, 64, 128, 32, 64, 43);

        // Group 9: CtaShape128x256x64_WarpShape64x64x64
        DISPATCH_FP8_GEMM_GROUP(9, 128, 256, 64, 64, 64, 64, 49);

        // Group 10: CtaShape256x128x64_WarpShape64x64x64
        DISPATCH_FP8_GEMM_GROUP(10, 256, 128, 64, 64, 64, 64, 55);

        // Group 11: CtaShape128x64x128_WarpShape64x32x128
        DISPATCH_FP8_GEMM_GROUP(11, 128, 64, 128, 64, 32, 128, 61);

        // Group 12: CtaShape16x256x128_WarpShape16x64x128
        DISPATCH_FP8_GEMM_GROUP(12, 16, 256, 128, 16, 64, 128, 67);

        // Group 13: CtaShape16x64x128_WarpShape16x64x64
        DISPATCH_FP8_GEMM_GROUP(13, 16, 64, 128, 16, 64, 64, 73);

        // Group 14: CtaShape16x128x64_WarpShape16x64x64
        DISPATCH_FP8_GEMM_GROUP(14, 16, 128, 64, 16, 64, 64, 79);

        // Group 15: CtaShape32x64x128_WarpShape16x64x64
        DISPATCH_FP8_GEMM_GROUP(15, 32, 64, 128, 16, 64, 64, 85);
        
        default:
            throw std::runtime_error("Invalid config_id: " + std::to_string(config_id));
    }
#endif
}

torch::Tensor fp8_scaled_mm(const torch::Tensor& mat_a, const torch::Tensor& mat_b, const torch::Tensor& scales_a,
                             const torch::Tensor& scales_b, const torch::Dtype& out_dtype,
                             const c10::optional<torch::Tensor>& bias, bool is_profile=false) {


  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_a must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  TORCH_CHECK((mat_a.size(1) * mat_a.element_size()) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK((mat_b.size(0) * mat_b.element_size()) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
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
  if (sm_version >= 90) {
        if (out_dtype == torch::kBFloat16) {
            sm90_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
        } else {
            sm90_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
        }
  } else if (sm_version == 89) {
        if (is_profile) {
            std::string config_path = get_config_path(mat_a.size(1), mat_b.size(1), out_dtype);
            try {
                json config = read_json_config(config_path);
                int current_m = mat_a.size(0);
                int nearest_m = find_nearest_m(config, current_m);
                if (nearest_m != -1) {
                    std::string key = "M=" + std::to_string(nearest_m);
                    int config_id = config[key].get<int>();
                    if (out_dtype == torch::kBFloat16) {
                        sm89_dispatch_shape_explicit<cutlass::bfloat16_t>(
                            out, mat_a, mat_b, scales_a, scales_b, bias, config_id);

                    } else {
                        sm89_dispatch_shape_explicit<cutlass::half_t>(
                            out, mat_a, mat_b, scales_a, scales_b, bias, config_id);
                    }
                    return out;
                }
            } catch (const std::exception& e) {
                std::cerr << "Failed to read config, using default dispatch: " << e.what() << std::endl;
            }
        }
        
        if (out_dtype == torch::kBFloat16) {
            sm89_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
        } else {
            sm89_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
        }
  } else {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "No implemented int8_scaled_mm for current compute capability: ", sm_version);
  }


  return out;
}


template <typename OutType>
float test_config(int config_id, torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
    const torch::Tensor& scales_a, const torch::Tensor& scales_b, const c10::optional<torch::Tensor>& bias) {
    const int NUM_WARMUP = 25;
    const int NUM_TEST = 100;
    // warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        sm89_dispatch_shape_explicit<OutType>(out, mat_a, mat_b, scales_a, scales_b, bias, config_id);
    }
    
    float total_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < NUM_TEST; i++) {
        cudaEventRecord(start);
        sm89_dispatch_shape_explicit<OutType>(out, mat_a, mat_b, scales_a, scales_b, bias, config_id);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_time += elapsed_time;
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return total_time / NUM_TEST;
}

template <typename OutType>
int sm89_dispatch_shape_profile(const torch::Tensor& mat_a, const torch::Tensor& mat_b,
    const torch::Tensor& scales_a, const torch::Tensor& scales_b, 
    const torch::Dtype& out_dtype, const c10::optional<torch::Tensor>& bias) {
    torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));
    float min_time = std::numeric_limits<float>::max();
    int best_config = -1;
#ifdef SGL_DEBUG_BUILD
    for (int i = 1; i <= MAX_CONFIG_ID; i++) {
#else
    for (int i = 1; i <= MAX_CONFIG_ID; i++) {
#endif
        try {
            float elapsed_time = test_config<OutType>(i, out, mat_a, mat_b, scales_a, scales_b, bias);
            #ifdef SGL_DEBUG_BUILD
            std::cout << "batch_size: " << mat_a.size(0) << ", config_id: " << i << ", time: " << elapsed_time << "ms" << std::endl;
            #endif
            if (elapsed_time < min_time) {
                min_time = elapsed_time;
                best_config = i;
            }
        } catch (const std::exception& e) {
            continue;
        }
    }
    return best_config;
}


void fp8_scaled_mm_profile(const torch::Tensor& mat_a, const torch::Tensor& mat_b, 
    const torch::Tensor& scales_a, const torch::Tensor& scales_b, 
    const torch::Dtype& out_dtype, const c10::optional<torch::Tensor>& bias) {
    
    TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
    TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
    TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
    TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
    TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
    TORCH_CHECK(mat_b.stride(0) == 1, "mat_a must be a column major tensor");
    TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

    TORCH_CHECK((mat_a.size(1) * mat_a.element_size()) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");
    TORCH_CHECK((mat_b.size(0) * mat_b.element_size()) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
    TORCH_CHECK(mat_a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
    TORCH_CHECK(mat_b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
    TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

    TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
    TORCH_CHECK(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
    TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
    TORCH_CHECK(scales_b.is_contiguous(), "scales_b must be contiguous");
    TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
    TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

    if (bias) {
        TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dtype() == out_dtype, "bias dtype must match output dtype");
    }

    torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));
    TORCH_CHECK((out.size(1) * out.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");
    
    std::string config_path = get_config_path(mat_a.size(1), mat_b.size(1), out_dtype);
    int best_config = -1;
    bool need_profile = true;
    
    try {
        json config = read_json_config(config_path);
        // construct key
        std::string key = "M=" + std::to_string(mat_a.size(0));
        
        // check if key exists
        if (config.contains(key)) {
            best_config = config[key].get<int>();
            need_profile = false;
        }
    } catch (const std::exception& e) {
        // if read failed, create new json object
        need_profile = true;
    }
    
    // if need profile, run profile and update config
    if (need_profile) {
        if (out_dtype == torch::kBFloat16) {
            best_config = sm89_dispatch_shape_profile<cutlass::bfloat16_t>(mat_a, mat_b, scales_a, scales_b, out_dtype, bias);
        } else {
            best_config = sm89_dispatch_shape_profile<cutlass::half_t>(mat_a, mat_b, scales_a, scales_b, out_dtype, bias);
        }
        if (best_config != -1) {
            try {
                // read existing config or create new config
                json config;
                try {
                    config = read_json_config(config_path);
                } catch (...) {
                    // if file not exists, use empty json object
                }
                
                // update config
                std::string key = "M=" + std::to_string(mat_a.size(0));
                config[key] = best_config;
                
                // save config
                std::ofstream o(config_path);
                o << std::setw(4) << config << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to save config: " << e.what() << std::endl;
            }
        }
    }
}