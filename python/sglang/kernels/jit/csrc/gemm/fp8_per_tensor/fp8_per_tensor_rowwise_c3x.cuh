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

template <
    typename ArchTag,
    typename OutType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename EpilogueScheduleType,
    typename EpilogueTileType,
    bool WithBias,
    bool ScalarA = false,
    typename TileScheduler = void>
struct JitGemmFp8RowwiseC3x {
  using ElementType = cutlass::float_e4m3_t;
  using TileShape = CTAShape;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using ElementComputeEpilogue = float;
  using VectorScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;
  using ScalarScaleA = cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementComputeEpilogue>;
  using ScaleA = std::conditional_t<ScalarA, ScalarScaleA, VectorScaleA>;

  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Bias = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, TileShape, OutType, OutType, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1MulAdd = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiply_add, OutType, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using Compute1Mul = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, OutType, float, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute = std::conditional_t<
      WithBias,
      cutlass::epilogue::fusion::Sm90EVT<Compute1MulAdd, ScaleA, EVTCompute0, Bias>,
      cutlass::epilogue::fusion::Sm90EVT<Compute1Mul, ScaleA, EVTCompute0>>;
  using ArgumentType = typename EVTCompute::Arguments;

  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;

  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutType>::value;

  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ElementCompute = float;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      OutType,
      LayoutD,
      AlignmentD,
      EpilogueScheduleType,
      EVTCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
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

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static ArgumentType prepare_args(
      tvm::ffi::TensorView scales_a, tvm::ffi::TensorView scales_b, tvm::ffi::Optional<tvm::ffi::TensorView> bias) {
    typename ScaleA::Arguments a_args = [&] {
      auto* a_ptr = static_cast<const float*>(scales_a.data_ptr());
      if constexpr (ScalarA) {
        // Sm90ScalarBroadcast takes the scale by pointer, not by value.
        return typename ScalarScaleA::Arguments{{}, {a_ptr}, {}};
      } else {
        return typename VectorScaleA::Arguments{a_ptr};
      }
    }();
    typename ScaleB::Arguments b_args{static_cast<const float*>(scales_b.data_ptr())};
    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};

    if constexpr (WithBias) {
      typename Bias::Arguments bias_args{static_cast<const OutType*>(bias.value().data_ptr())};
      return ArgumentType{a_args, evt0_args, bias_args, {}};
    } else {
      return ArgumentType{a_args, evt0_args, {}};
    }
  }
};

template <typename GemmType>
void launch_c3x_fp8_rowwise_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream,
    typename GemmType::Gemm::GemmKernel::TileSchedulerArguments scheduler = {}) {
  using Gemm = typename GemmType::Gemm;
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementT = typename Gemm::ElementA;
  using ElementOutput = typename Gemm::ElementD;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = StrideC;

  const int32_t m = a.size(0);
  const int32_t n = b.size(1);
  const int32_t k = a.size(1);

  auto ptr_a = static_cast<const ElementT*>(a.data_ptr());
  auto ptr_b = static_cast<const ElementT*>(b.data_ptr());
  auto ptr_d = static_cast<ElementOutput*>(out.data_ptr());

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  typename GemmKernel::MainloopArguments mainloop_args{ptr_a, stride_a, ptr_b, stride_b};
  typename GemmKernel::EpilogueArguments epilogue_args{
      GemmType::prepare_args(scales_a, scales_b, bias), ptr_d, stride_c, ptr_d, stride_d};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.device().device_id;
  hw_info.sm_count = static_cast<int>(host::runtime::get_sm_count(hw_info.device_id));

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler,
  };

  Gemm gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  const size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace_tensor = host::alloc_workspace_tensor(workspace_size, a.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm_op.run(args, workspace, stream));
}

}  // namespace sglang
