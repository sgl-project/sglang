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

#include <sgl_kernel/utils.cuh>

#include "fp8_per_tensor_common.cuh"

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
// clang-format on

namespace sglang {

using namespace cute;

template <
    typename OutElementType,
    typename CtaShape,
    typename WarpShape,
    int Stages,
    bool WithBias,
    typename FP8MathOperator = cutlass::arch::OpMultiplyAdd,
    typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct JitGemmFp8RowwiseSm89 {
  using ElementType = cutlass::float_e4m3_t;

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
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm89;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int EVTEpilogueStages = 1;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::
      OutputTileThreadLayout<CtaShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

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

template <typename GemmType, bool WithBias>
void launch_sm89_fp8_rowwise_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  using Gemm = typename GemmType::Gemm;
  using ElementT = typename GemmType::ElementA;
  using ElementOutput = typename GemmType::ElementOutput;
  using ElementComputeEpilogue = float;

  const int32_t m = a.size(0);
  const int32_t n = b.size(1);
  const int32_t k = a.size(1);

  const int64_t lda = a.stride(0);
  const int64_t ldb = b.stride(1);
  const int64_t ldc = out.stride(0);

  auto ptr_a = static_cast<const ElementT*>(a.data_ptr());
  auto ptr_b = static_cast<const ElementT*>(b.data_ptr());
  auto ptr_d = static_cast<ElementOutput*>(out.data_ptr());
  auto ptr_scales_a = static_cast<const ElementComputeEpilogue*>(scales_a.data_ptr());
  auto ptr_scales_b = static_cast<const ElementComputeEpilogue*>(scales_b.data_ptr());

  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k},
      1,
      {},
      ptr_a,
      ptr_b,
      nullptr,
      nullptr,
      m * k,
      n * k,
      m * n,
      m * n,
      lda,
      ldb,
      ldc,
      ldc);

  if constexpr (WithBias) {
    auto ptr_bias = static_cast<const ElementOutput*>(bias.value().data_ptr());
    args.epilogue = {
        {{{}, {ptr_scales_b, ElementComputeEpilogue(0), {_0{}, _1{}, _0{}}}, {}},
         {ptr_scales_a, ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
         {ptr_bias, ElementOutput(0), {_0{}, _1{}, _0{}}},
         {}},
        {ptr_d, {n, _1{}, _0{}}}};
  } else {
    args.epilogue = {
        {{{}, {ptr_scales_b, ElementComputeEpilogue(0), {_0{}, _1{}, _0{}}}, {}},
         {ptr_scales_a, ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
         {}},
        {ptr_d, {n, _1{}, _0{}}}};
  }

  Gemm gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  const size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace_tensor = host::alloc_workspace_tensor(workspace_size, a.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm_op(args, workspace, stream));
}

template <typename OutType, typename CtaShape, typename WarpShape, int Stages>
void sm89_fp8_pertensor_dispatch_bias(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  if (bias.has_value()) {
    using GemmWithBias = JitGemmFp8RowwiseSm89<OutType, CtaShape, WarpShape, Stages, true>;
    return launch_sm89_fp8_rowwise_scaled_mm<GemmWithBias, true>(out, a, b, scales_a, scales_b, bias, stream);
  }
  using GemmNoBias = JitGemmFp8RowwiseSm89<OutType, CtaShape, WarpShape, Stages, false>;
  return launch_sm89_fp8_rowwise_scaled_mm<GemmNoBias, false>(out, a, b, scales_a, scales_b, bias, stream);
}

template <typename OutType>
void sm89_fp8_pertensor_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  const uint32_t m = a.size(0);
  const uint32_t n = out.size(1);

#define SGL_SM89_DISPATCH(CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, STAGES) \
  return sm89_fp8_pertensor_dispatch_bias<                                     \
      OutType,                                                                 \
      cutlass::gemm::GemmShape<CTA_M, CTA_N, CTA_K>,                           \
      cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                        \
      STAGES>(out, a, b, scales_a, scales_b, bias, stream)

  if (m == 1) {
    if (n <= 8192) {
      SGL_SM89_DISPATCH(16, 64, 128, 16, 64, 64, 7);
    }
    SGL_SM89_DISPATCH(32, 64, 128, 16, 64, 64, 5);
  }
  if (m <= 16) {
    if (n <= 8192) {
      SGL_SM89_DISPATCH(16, 64, 128, 16, 64, 64, 4);
    }
    if (n <= 16384) {
      SGL_SM89_DISPATCH(32, 64, 128, 16, 64, 64, 5);
    }
    SGL_SM89_DISPATCH(16, 64, 128, 16, 64, 64, 7);
  }
  if (m <= 64) {
    if (n <= 16384) {
      SGL_SM89_DISPATCH(32, 64, 128, 16, 64, 64, 7);
    }
    SGL_SM89_DISPATCH(16, 64, 128, 16, 64, 64, 7);
  }
  if (m <= 128) {
    if (n <= 8192) {
      SGL_SM89_DISPATCH(64, 64, 128, 32, 64, 64, 4);
    }
    if (n <= 16384) {
      SGL_SM89_DISPATCH(64, 64, 128, 32, 64, 64, 5);
    }
    SGL_SM89_DISPATCH(32, 64, 128, 16, 64, 64, 5);
  }
  if (m <= 256) {
    if (n <= 8192) {
      SGL_SM89_DISPATCH(128, 64, 64, 64, 32, 64, 5);
    }
    if (n <= 16384) {
      SGL_SM89_DISPATCH(64, 128, 64, 64, 32, 64, 7);
    }
    SGL_SM89_DISPATCH(128, 64, 128, 64, 32, 128, 4);
  }
  if (m <= 512) {
    if (n <= 16384) {
      SGL_SM89_DISPATCH(128, 128, 64, 64, 32, 64, 2);
    }
    SGL_SM89_DISPATCH(128, 128, 64, 64, 32, 64, 4);
  }
  if (n <= 8192) {
    SGL_SM89_DISPATCH(128, 128, 64, 64, 32, 64, 3);
  }
  SGL_SM89_DISPATCH(128, 128, 64, 64, 32, 64, 2);

#undef SGL_SM89_DISPATCH
}

}  // namespace sglang
