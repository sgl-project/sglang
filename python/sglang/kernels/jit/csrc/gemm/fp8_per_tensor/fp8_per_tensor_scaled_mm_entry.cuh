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

#include <sgl_kernel/utils.cuh>

#if SGL_CUDA_ARCH >= 1200
#include "fp8_per_tensor_scaled_mm_sm120.cuh"
#elif SGL_CUDA_ARCH >= 1000
#include "fp8_per_tensor_scaled_mm_sm100.cuh"
#elif SGL_CUDA_ARCH >= 900
#include "fp8_per_tensor_scaled_mm_sm90.cuh"
#elif SGL_CUDA_ARCH >= 890
#include "fp8_per_tensor_scaled_mm_sm89.cuh"
#else
#error "fp8_per_tensor_scaled_mm requires SM89 or later"
#endif

namespace sglang {

// Scalar (per-tensor) A scales rely on a scalar-broadcast epilogue: SM90 selects
// one at runtime, SM100/SM120 instantiate one. SM89's visitor is per-row only.
#if SGL_CUDA_ARCH >= 1000 && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
inline constexpr bool kScalarAScaleSupported = true;
#elif SGL_CUDA_ARCH >= 900 && SGL_CUDA_ARCH < 1000 && defined(CUDA_VERSION) && CUDA_VERSION >= 12000
inline constexpr bool kScalarAScaleSupported = true;
#else
inline constexpr bool kScalarAScaleSupported = false;
#endif

template <typename OutType>
void fp8_per_tensor_dispatch_arch(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    cudaStream_t stream) {
#if SGL_CUDA_ARCH >= 1200
  sm120_fp8_pertensor_dispatch_bias<OutType>(out, a, b, scales_a, scales_b, bias, stream);
#elif SGL_CUDA_ARCH >= 1000
  sm100_fp8_pertensor_dispatch_bias<OutType>(out, a, b, scales_a, scales_b, bias, stream);
#elif SGL_CUDA_ARCH >= 900
  sm90_fp8_pertensor_dispatch_shape<OutType>(out, a, b, scales_a, scales_b, bias, stream);
#else
  sm89_fp8_pertensor_dispatch_shape<OutType>(out, a, b, scales_a, scales_b, bias, stream);
#endif
}

void fp8_per_tensor_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias) {
  using namespace host;

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

  RuntimeCheck(
      scales_a.numel() == 1 || scales_a.numel() == mat_a.size(0),
      "scales_a must contain either one scalar scale or one scale per row; got ",
      scales_a.numel(),
      " elements for M=",
      mat_a.size(0));
  // A single A scale broadcast across M rows needs a scalar-broadcast epilogue,
  // which only the SM90 and SM100+ paths provide. SM89 stays per-row only.
  RuntimeCheck(
      scales_a.numel() != 1 || mat_a.size(0) == 1 || kScalarAScaleSupported,
      "scalar scales_a with M > 1 is unsupported on SM",
      SGL_CUDA_ARCH / 10,
      " for this build; got M=",
      mat_a.size(0));
  RuntimeCheck(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  RuntimeCheck(scales_a.IsContiguous(), "scales_a must be contiguous");
  RuntimeCheck(scales_b.IsContiguous(), "scales_b must be contiguous");
  RuntimeCheck(is_type<float>(scales_a.dtype()), "scales_a must be Float32");
  RuntimeCheck(is_type<float>(scales_b.dtype()), "scales_b must be Float32");

  RuntimeCheck(
      (out.size(1) * (out.dtype().bits / 8)) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  if (bias.has_value()) {
    RuntimeCheck(bias.value().numel() == mat_b.size(1), "size of bias is not matched");
    RuntimeCheck(bias.value().IsContiguous(), "bias must be contiguous");
    RuntimeCheck(
        bias.value().dtype().code == out.dtype().code && bias.value().dtype().bits == out.dtype().bits,
        "bias dtype must match output dtype");
  }

  const cudaStream_t stream = LaunchKernel::resolve_device(mat_a.device());

  if (is_type<bf16_t>(out.dtype())) {
    fp8_per_tensor_dispatch_arch<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias, stream);
  } else if (is_type<fp16_t>(out.dtype())) {
    fp8_per_tensor_dispatch_arch<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias, stream);
  } else {
    Panic("out_dtype must be Half or BFloat16");
  }
}

}  // namespace sglang
