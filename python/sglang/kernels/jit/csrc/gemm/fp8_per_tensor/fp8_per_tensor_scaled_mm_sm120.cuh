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

#include "fp8_per_tensor_rowwise_c3x.cuh"

namespace sglang {

template <typename OutType, bool WithBias, bool ScalarA>
void sm120_fp8_pertensor_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  const int m = a.size(0);
  using ArchTag = cutlass::arch::Sm120;
  using ClusterShape = Shape<_1, _1, _1>;
  using PingpongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueScheduleAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using EpilogueTileAuto = cutlass::epilogue::collective::EpilogueTileAuto;

  if (m <= 16) {
    using GemmM16 = JitGemmFp8RowwiseC3x<
        ArchTag,
        OutType,
        Shape<_16, _64, _128>,
        ClusterShape,
        PingpongSchedule,
        EpilogueScheduleAuto,
        Shape<_16, _32>,
        WithBias,
        ScalarA>;
    return launch_c3x_fp8_rowwise_scaled_mm<GemmM16>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (m <= 32) {
    using GemmM32 = JitGemmFp8RowwiseC3x<
        ArchTag,
        OutType,
        Shape<_32, _64, _128>,
        ClusterShape,
        PingpongSchedule,
        EpilogueScheduleAuto,
        Shape<_32, _32>,
        WithBias,
        ScalarA>;
    return launch_c3x_fp8_rowwise_scaled_mm<GemmM32>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (m <= 256) {
    using GemmM64 = JitGemmFp8RowwiseC3x<
        ArchTag,
        OutType,
        Shape<_64, _64, _128>,
        ClusterShape,
        PingpongSchedule,
        EpilogueScheduleAuto,
        EpilogueTileAuto,
        WithBias,
        ScalarA>;
    return launch_c3x_fp8_rowwise_scaled_mm<GemmM64>(out, a, b, scales_a, scales_b, bias, stream);
  }

  using GemmDefault = JitGemmFp8RowwiseC3x<
      ArchTag,
      OutType,
      Shape<_128, _128, _128>,
      ClusterShape,
      cutlass::gemm::collective::KernelScheduleAuto,
      EpilogueScheduleAuto,
      EpilogueTileAuto,
      WithBias,
      ScalarA>;
  return launch_c3x_fp8_rowwise_scaled_mm<GemmDefault>(out, a, b, scales_a, scales_b, bias, stream);
}

template <typename OutType>
void sm120_fp8_pertensor_dispatch_bias(
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
      return sm120_fp8_pertensor_dispatch_shape<OutType, true, true>(out, a, b, scales_a, scales_b, bias, stream);
    }
    return sm120_fp8_pertensor_dispatch_shape<OutType, true, false>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (scalar_a) {
    return sm120_fp8_pertensor_dispatch_shape<OutType, false, true>(out, a, b, scales_a, scales_b, bias, stream);
  }
  return sm120_fp8_pertensor_dispatch_shape<OutType, false, false>(out, a, b, scales_a, scales_b, bias, stream);
}

}  // namespace sglang
