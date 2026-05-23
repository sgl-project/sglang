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

#include "nvfp4_scaled_mm_common.cuh"
#include "nvfp4_scaled_mm_sm100.cuh"
#include "nvfp4_scaled_mm_sm120.cuh"

void cutlass_scaled_fp4_mm_sm100a_sm120a(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha) {
  RuntimeCheck(A.device().device_type == kDLCUDA, "a must be a CUDA tensor");
  RuntimeCheck(B.device().device_type == kDLCUDA, "b must be a CUDA tensor");
  RuntimeCheck(A_sf.device().device_type == kDLCUDA, "scale_a must be a CUDA tensor");
  RuntimeCheck(B_sf.device().device_type == kDLCUDA, "scale_b must be a CUDA tensor");
  RuntimeCheck(alpha.device().device_type == kDLCUDA, "alpha must be a CUDA tensor");
  RuntimeCheck(D.device().device_type == kDLCUDA, "out must be a CUDA tensor");

  RuntimeCheck(A.device() == B.device(), "a and b must be on same device");
  RuntimeCheck(A.device() == A_sf.device(), "a and scale_a must be on same device");
  RuntimeCheck(A.device() == B_sf.device(), "a and scale_b must be on same device");
  RuntimeCheck(A.device() == alpha.device(), "a and alpha must be on same device");
  RuntimeCheck(A.device() == D.device(), "a and out must be on same device");

  RuntimeCheck(A.is_contiguous(), "a must be contiguous");
  RuntimeCheck(B.is_contiguous(), "b must be contiguous");
  RuntimeCheck(A_sf.is_contiguous(), "scale_a must be contiguous");
  RuntimeCheck(B_sf.is_contiguous(), "scale_b must be contiguous");
  RuntimeCheck(alpha.is_contiguous(), "alpha must be contiguous");
  RuntimeCheck(D.is_contiguous(), "out must be contiguous");

  RuntimeCheck(host::is_type<uint8_t>(A.dtype()), "a must be uint8");
  RuntimeCheck(host::is_type<uint8_t>(B.dtype()), "b must be uint8");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(A_sf.dtype()), "scale_a must be float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(B_sf.dtype()), "scale_b must be float8_e4m3fn");
  RuntimeCheck(host::is_type<float>(alpha.dtype()), "alpha must be float32");

  RuntimeCheck(A.dim() == 2, "a must be a matrix");
  RuntimeCheck(B.dim() == 2, "b must be a matrix");
  RuntimeCheck(A_sf.dim() == 2, "scale_a must be a matrix");
  RuntimeCheck(B_sf.dim() == 2, "scale_b must be a matrix");
  RuntimeCheck(alpha.numel() == 1, "alpha must have exactly one element");

  RuntimeCheck(
      A.size(1) == B.size(1),
      "a and b shapes cannot be multiplied (",
      A.size(0),
      "x",
      A.size(1),
      " and ",
      B.size(0),
      "x",
      B.size(1),
      ")");

  const auto m = static_cast<int64_t>(A.size(0));
  const auto n = static_cast<int64_t>(B.size(0));
  const auto k = static_cast<int64_t>(A.size(1) * 2);

  RuntimeCheck(D.dim() == 2, "out must be 2D");
  RuntimeCheck(D.size(0) == m, "out first dim must equal m");
  RuntimeCheck(D.size(1) == n, "out second dim must equal n");

  constexpr int alignment = 32;
  RuntimeCheck(k % alignment == 0, "Expected k to be divisible by ", alignment, ", but got k: ", k);
  RuntimeCheck(n % alignment == 0, "Expected n to be divisible by ", alignment, ", but got n: ", n);

  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  const int64_t rounded_m = round_up(m, 128);
  const int64_t rounded_n = round_up(n, 128);
  const int64_t rounded_k = round_up(k / 16, 4);

  RuntimeCheck(
      A_sf.size(1) == B_sf.size(1),
      "scale_a and scale_b shapes cannot be multiplied (",
      A_sf.size(0),
      "x",
      A_sf.size(1),
      " and ",
      B_sf.size(0),
      "x",
      B_sf.size(1),
      ")");
  RuntimeCheck(
      A_sf.size(0) == rounded_m && A_sf.size(1) == rounded_k,
      "scale_a must be padded/swizzled to shape (",
      rounded_m,
      "x",
      rounded_k,
      "), got (",
      A_sf.size(0),
      "x",
      A_sf.size(1),
      ")");
  RuntimeCheck(
      B_sf.size(0) == rounded_n && B_sf.size(1) == rounded_k,
      "scale_b must be padded/swizzled to shape (",
      rounded_n,
      "x",
      rounded_k,
      "), got (",
      B_sf.size(0),
      "x",
      B_sf.size(1),
      ")");

  const cudaStream_t stream = LaunchKernel::resolve_device(A.device());
  const int sm_version = getSMVersion(A.device().device_id);

  if (sm_version >= 120) {
    if (host::is_type<fp16_t>(D.dtype())) {
      cutlass_fp4_f16_gemm_dispatch_sm120(
          D, A, B, A_sf, B_sf, alpha, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
    } else if (host::is_type<bf16_t>(D.dtype())) {
      cutlass_fp4_bf16_gemm_dispatch_sm120(
          D, A, B, A_sf, B_sf, alpha, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
    } else {
      Panic("Unsupported output data type of nvfp4 mm sm120");
    }
  } else {
    if (host::is_type<fp16_t>(D.dtype())) {
      cutlassFp4GemmDispatchSm100<cutlass::half_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else if (host::is_type<bf16_t>(D.dtype())) {
      cutlassFp4GemmDispatchSm100<cutlass::bfloat16_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else if (host::is_type<float>(D.dtype())) {
      cutlassFp4GemmDispatchSm100<float>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else {
      Panic("Unsupported output data type of nvfp4 mm");
    }
  }
}
