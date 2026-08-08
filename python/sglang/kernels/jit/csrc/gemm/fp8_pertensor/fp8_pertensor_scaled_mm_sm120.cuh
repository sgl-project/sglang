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

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

using namespace host;

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                                        \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

template <int CtaM, int CtaN, int CtaK>
void launch_fp8_pertensor_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b_nk,
    tvm::ffi::TensorView scale_a,
    tvm::ffi::TensorView scale_b,
    cudaStream_t stream) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  // No C operand: the epilogue only rescales the accumulator.
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementD = cutlass::bfloat16_t;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;

  // The broadcast multiplies both device-side scales, so callers need no alpha kernel.
  using ScaleAB =
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementCompute, Stride<_0, _0, _0>, 2, cutlass::multiplies>;
  using ApplyScale = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, ElementD, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<ApplyScale, ScaleAB, cutlass::epilogue::fusion::Sm90AccFetch>;

  using MmaTileShape_MNK = Shape<Int<CtaM>, Int<CtaN>, Int<CtaK>>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;
  using KernelScheduleTag = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using TileSchedulerTag = void;

  const int m = static_cast<int>(mat_a.size(0));
  const int k = static_cast<int>(mat_a.size(1));
  const int n = static_cast<int>(mat_b_nk.size(0));
  constexpr int l = 1;

  auto* a_ptr = static_cast<ElementA*>(mat_a.data_ptr());
  auto* b_ptr = static_cast<ElementB*>(mat_b_nk.data_ptr());
  auto* d_ptr = static_cast<ElementD*>(out.data_ptr());
  auto const* scale_a_ptr = static_cast<ElementCompute const*>(scale_a.data_ptr());
  auto const* scale_b_ptr = static_cast<ElementCompute const*>(scale_b.data_ptr());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      MmaTileShape_MNK,
      ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueEVT>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape_MNK,
      ClusterShape_MNK,
      StageCount,
      KernelScheduleTag>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileSchedulerTag>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  // EVT args are children-first; the broadcast takes {scalars, pointers, strides}.
  typename EpilogueEVT::Arguments epilogue_thread_args{{{}, {scale_a_ptr, scale_b_ptr}, {}}, {}, {}};

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {a_ptr, stride_A, b_ptr, stride_B},
      {epilogue_thread_args, nullptr, stride_C, d_ptr, stride_D}};

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(arguments));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace_tensor = alloc_workspace_tensor(workspace_size, mat_a.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
  CUTLASS_CHECK(gemm.run(stream));
}

// Below M=24 the deeper pipeline of the smaller CtaK wins; above it CTA count does.
inline void fp8_pertensor_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b_nk,
    tvm::ffi::TensorView scale_a,
    tvm::ffi::TensorView scale_b,
    cudaStream_t stream) {
  const int m = static_cast<int>(mat_a.size(0));
  if (m < 24) {
    launch_fp8_pertensor_scaled_mm<64, 128, 64>(out, mat_a, mat_b_nk, scale_a, scale_b, stream);
  } else {
    launch_fp8_pertensor_scaled_mm<64, 64, 128>(out, mat_a, mat_b_nk, scale_a, scale_b, stream);
  }
}

inline void fp8_pertensor_scaled_mm_sm120(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b_nk,
    tvm::ffi::TensorView scale_a,
    tvm::ffi::TensorView scale_b) {
  RuntimeCheck(mat_a.device().device_type == kDLCUDA, "mat_a must be a CUDA tensor");
  RuntimeCheck(mat_b_nk.device().device_type == kDLCUDA, "mat_b_nk must be a CUDA tensor");
  RuntimeCheck(out.device().device_type == kDLCUDA, "out must be a CUDA tensor");
  RuntimeCheck(scale_a.device().device_type == kDLCUDA, "scale_a must be a CUDA tensor");
  RuntimeCheck(scale_b.device().device_type == kDLCUDA, "scale_b must be a CUDA tensor");

  RuntimeCheck(mat_a.dim() == 2, "mat_a must be a 2D tensor [M, K]");
  RuntimeCheck(mat_b_nk.dim() == 2, "mat_b_nk must be a 2D tensor [N, K]");
  RuntimeCheck(out.dim() == 2, "out must be a 2D tensor [M, N]");
  RuntimeCheck(scale_a.numel() == 1, "scale_a must be a scalar tensor");
  RuntimeCheck(scale_b.numel() == 1, "scale_b must be a scalar tensor");

  RuntimeCheck(mat_a.stride(1) == 1, "mat_a must be row-major contiguous in K");
  RuntimeCheck(mat_b_nk.stride(1) == 1, "mat_b_nk must be row-major contiguous in K");
  RuntimeCheck(out.stride(1) == 1, "out must be row-major contiguous in N");

  RuntimeCheck(mat_a.size(1) == mat_b_nk.size(1), "mat_a and mat_b_nk K dims must match");
  RuntimeCheck(out.size(0) == mat_a.size(0), "out M must match mat_a M");
  RuntimeCheck(out.size(1) == mat_b_nk.size(0), "out N must match mat_b_nk N");

  RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_a.dtype()), "mat_a must be Float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_b_nk.dtype()), "mat_b_nk must be Float8_e4m3fn");
  RuntimeCheck(host::is_type<float>(scale_a.dtype()), "scale_a must be Float32");
  RuntimeCheck(host::is_type<float>(scale_b.dtype()), "scale_b must be Float32");
  RuntimeCheck(host::is_type<bf16_t>(out.dtype()), "out must be BFloat16");

  const cudaStream_t stream = LaunchKernel::resolve_device(mat_a.device());
  fp8_pertensor_dispatch_shape(out, mat_a, mat_b_nk, scale_a, scale_b, stream);
}

#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
