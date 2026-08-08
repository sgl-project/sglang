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
void sm100_fp8_pertensor_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
  using ArchTag = cutlass::arch::Sm100;
  using MainloopScheduleAuto = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueScheduleAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using EpilogueTileAuto = cutlass::epilogue::collective::EpilogueTileAuto;

  using Gemm16 = JitGemmFp8RowwiseC3x<
      ArchTag,
      OutType,
      Shape<_64, _64, _128>,
      Shape<_1, _4, _1>,
      MainloopScheduleAuto,
      EpilogueScheduleAuto,
      EpilogueTileAuto,
      WithBias,
      ScalarA>;
  using Gemm64 = JitGemmFp8RowwiseC3x<
      ArchTag,
      OutType,
      Shape<_64, _64, _128>,
      Shape<_1, _1, _1>,
      MainloopScheduleAuto,
      EpilogueScheduleAuto,
      EpilogueTileAuto,
      WithBias,
      ScalarA>;
  using Gemm256 = JitGemmFp8RowwiseC3x<
      ArchTag,
      OutType,
      Shape<_128, _128, _128>,
      Shape<_2, _1, _1>,
      MainloopScheduleAuto,
      EpilogueScheduleAuto,
      EpilogueTileAuto,
      WithBias,
      ScalarA>;
  using GemmDefault = JitGemmFp8RowwiseC3x<
      ArchTag,
      OutType,
      Shape<_256, _128, _64>,
      Shape<_2, _2, _1>,
      MainloopScheduleAuto,
      EpilogueScheduleAuto,
      EpilogueTileAuto,
      WithBias,
      ScalarA>;

  const uint32_t m = a.size(0);
  const uint32_t mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));

  if (mp2 <= 16) {
    return launch_c3x_fp8_rowwise_scaled_mm<Gemm16>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (mp2 <= 64) {
    return launch_c3x_fp8_rowwise_scaled_mm<Gemm64>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (mp2 <= 256) {
    return launch_c3x_fp8_rowwise_scaled_mm<Gemm256>(out, a, b, scales_a, scales_b, bias, stream);
  }
  return launch_c3x_fp8_rowwise_scaled_mm<GemmDefault>(out, a, b, scales_a, scales_b, bias, stream);
}

template <typename OutType>
void sm100_fp8_pertensor_dispatch_bias(
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
      return sm100_fp8_pertensor_dispatch_shape<OutType, true, true>(out, a, b, scales_a, scales_b, bias, stream);
    }
    return sm100_fp8_pertensor_dispatch_shape<OutType, true, false>(out, a, b, scales_a, scales_b, bias, stream);
  }
  if (scalar_a) {
    return sm100_fp8_pertensor_dispatch_shape<OutType, false, true>(out, a, b, scales_a, scales_b, bias, stream);
  }
  return sm100_fp8_pertensor_dispatch_shape<OutType, false, false>(out, a, b, scales_a, scales_b, bias, stream);
}

}  // namespace sglang
