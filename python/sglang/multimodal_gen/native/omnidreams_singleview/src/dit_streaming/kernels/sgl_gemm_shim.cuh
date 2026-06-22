// SPDX-License-Identifier: Apache-2.0
// C++ direct-call shim into sgl-kernel's fp8_scaled_mm (exported symbol, default
// visibility). sgl-kernel ships no C++ header, so we declare the signature and
// link common_ops.abi3.so. Wrapped in a namespace to avoid ODR clashes.
#pragma once
#include <ATen/Functions.h>
#include <c10/util/Optional.h>
#include <torch/types.h>

// sgl-kernel's exported op (sgl_kernel_ops.h signature; symbol lives in
// sgl_kernel/sm100/common_ops.abi3.so). Declared in the GLOBAL namespace so its
// mangled name matches sgl-kernel's exported ::fp8_scaled_mm (the wheel ships no
// C++ header). Must NOT be inside namespace omnidreams_singleview.
at::Tensor fp8_scaled_mm(
    const at::Tensor& mat_a, const at::Tensor& mat_b,
    const at::Tensor& scales_a, const at::Tensor& scales_b,
    const at::ScalarType& out_dtype,
    const c10::optional<at::Tensor>& bias);

namespace omnidreams_singleview {

// Drop-in replacement for cutlass_linear_layer_rcr_fp8_colscale_bf16 (no GELU):
//   out[M,N] = bf16( (a8[M,K] @ b8[K,N]) * colscale[N] )
// a8: row-major [M,K] fp8 ; weight stored as [N,K] -> pass .t() for [K,N]
// col-major. colscale[N] becomes scales_b (per-output-col). scales_a = ones[M].
// fp8_scaled_mm allocates its own bf16 output on the current CUDA stream.
inline at::Tensor sgl_linear_rcr_fp8_colscale_bf16(
    const void* a8_ptr, const void* b8_ptr, const void* colscale_ptr,
    int M, int K, int N, cudaStream_t stream) {
  // fp8_scaled_mm runs on at::cuda::getCurrentCUDAStream(); the native engine's
  // stream is current during the forward, so no explicit stream wiring needed.
  (void)stream;
  auto opts_fp8 = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);
  auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  at::Tensor a8 = torch::from_blob(const_cast<void*>(a8_ptr), {M, K}, opts_fp8);
  at::Tensor b8 = torch::from_blob(const_cast<void*>(b8_ptr), {N, K}, opts_fp8);
  at::Tensor colscale = torch::from_blob(const_cast<void*>(colscale_ptr), {N}, opts_f32);
  at::Tensor ones = torch::ones({M}, opts_f32);
  return fp8_scaled_mm(a8, b8.t(), ones, colscale, torch::kBFloat16, c10::nullopt);
}


// Bare FP8 GEMM (scales_b=1, no colscale) -> bf16 output for unfused post-op
// paths (#2 gelu_fp8, #3 residual_bf16, #4 residual_ln_to_fp8).
inline at::Tensor sgl_linear_rcr_fp8_bare(
    const void* a8_ptr, const void* b8_ptr,
    int M, int K, int N) {
  auto opts_fp8 = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);
  auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  at::Tensor a8 = torch::from_blob(const_cast<void*>(a8_ptr), {M, K}, opts_fp8);
  at::Tensor b8 = torch::from_blob(const_cast<void*>(b8_ptr), {N, K}, opts_fp8);
  at::Tensor ones_a = torch::ones({M}, opts_f32);
  at::Tensor ones_b = torch::ones({N}, opts_f32);
  return fp8_scaled_mm(a8, b8.t(), ones_a, ones_b, torch::kBFloat16, c10::nullopt);
}

// Bare FP8 GEMM (scales_b=1, no colscale) -> bf16 output for unfused post-op
// paths (#2 gelu_fp8, #3 residual_bf16, #4 residual_ln_to_fp8).
inline at::Tensor sgl_linear_rcr_fp8_bare(
    const void* a8_ptr, const void* b8_ptr,
    int M, int K, int N) {
  auto opts_fp8 = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);
  auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  at::Tensor a8 = torch::from_blob(const_cast<void*>(a8_ptr), {M, K}, opts_fp8);
  at::Tensor b8 = torch::from_blob(const_cast<void*>(b8_ptr), {N, K}, opts_fp8);
  at::Tensor ones_a = torch::ones({M}, opts_f32);
  at::Tensor ones_b = torch::ones({N}, opts_f32);
  return fp8_scaled_mm(a8, b8.t(), ones_a, ones_b, torch::kBFloat16, c10::nullopt);
}
}  // namespace omnidreams_singleview
