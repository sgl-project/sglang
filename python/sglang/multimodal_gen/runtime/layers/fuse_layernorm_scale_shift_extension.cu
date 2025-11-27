// PyTorch extension binding for fuse_layernorm_scale_shift.h (CUTLASS-based layernorm)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

// CUTLASS
#if __has_include("/workspace/cutlass/include/cutlass/cutlass.h")
#include "/workspace/cutlass/include/cutlass/cutlass.h"
#include "/workspace/cutlass/include/cutlass/layout/tensor.h"
#include "/workspace/cutlass/include/cutlass/tensor_coord.h"
#include "/workspace/cutlass/include/cutlass/tensor_ref.h"
#else
#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#endif

// Local kernel header (templated CUDA kernels and launcher)
#include "fuse_layernorm_scale_shift.h"

#include <cuda_fp16.h>

namespace {

template <typename T>
torch::Tensor layernorm_cutlass(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1, "last dim of x must be contiguous (stride 1)");

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);

  const bool has_gamma = gamma_opt.has_value() && gamma_opt->defined();
  const bool has_beta = beta_opt.has_value() && beta_opt->defined();

  const T* gamma_ptr = nullptr;
  const T* beta_ptr = nullptr;

  if (has_gamma) {
    const auto& g = *gamma_opt;
    TORCH_CHECK(g.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(g.dtype() == x.dtype(), "gamma must have same dtype as x");
    TORCH_CHECK(g.dim() == 1 && g.numel() == N, "gamma must be shape [N]");
    TORCH_CHECK(g.stride(0) == 1, "gamma must be contiguous");
    gamma_ptr = reinterpret_cast<const T*>(g.data_ptr());
  }
  if (has_beta) {
    const auto& b = *beta_opt;
    TORCH_CHECK(b.is_cuda(), "beta must be CUDA");
    TORCH_CHECK(b.dtype() == x.dtype(), "beta must have same dtype as x");
    TORCH_CHECK(b.dim() == 1 && b.numel() == N, "beta must be shape [N]");
    TORCH_CHECK(b.stride(0) == 1, "beta must be contiguous");
    beta_ptr = reinterpret_cast<const T*>(b.data_ptr());
  }

  auto y = torch::empty_like(x);

  // Build CUTLASS TensorRefs (layout info is unused by the header launcher, but keep it correct)
  cutlass::MatrixCoord size(static_cast<int>(M), static_cast<int>(N));
  auto layout_row_major = cutlass::layout::RowMajor(static_cast<int>(N));

  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_in(
      reinterpret_cast<T*>(x.data_ptr()), layout_row_major);
  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_out(
      reinterpret_cast<T*>(y.data_ptr()), layout_row_major);

  // If gamma/beta are null, provide dummy 1/0 vectors by allocating temporary tensors
  torch::Tensor gamma_fallback, beta_fallback;
  if (!has_gamma) {
    gamma_fallback = torch::ones({N}, x.options());
    gamma_ptr = reinterpret_cast<const T*>(gamma_fallback.data_ptr());
  }
  if (!has_beta) {
    beta_fallback = torch::zeros({N}, x.options());
    beta_ptr = reinterpret_cast<const T*>(beta_fallback.data_ptr());
  }
  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_gamma(
      const_cast<T*>(gamma_ptr), cutlass::layout::RowMajor(static_cast<int>(N)));
  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_beta(
      const_cast<T*>(beta_ptr), cutlass::layout::RowMajor(static_cast<int>(N)));

  auto stream = at::cuda::getCurrentCUDAStream();
  cutlass::layernorm<T>(
      size,
      ref_out,
      ref_in,
      ref_gamma,
      ref_beta,
      stream.stream());
  return y;
}

template <typename T>
torch::Tensor launch_fuse_layernorm_scale_shift_fused_impl(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    const torch::Tensor& scale,
    const torch::Tensor& shift) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(scale.is_cuda() && shift.is_cuda(), "scale/shift must be CUDA");
  TORCH_CHECK(x.dim() == 2 && scale.dim() == 2 && shift.dim() == 2, "x/scale/shift must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1 && scale.stride(-1) == 1 && shift.stride(-1) == 1,
              "last dim of x/scale/shift must be contiguous (stride 1)");
  TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D [N]");

  TORCH_CHECK(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x, scale, shift must have same dtype");
  TORCH_CHECK(x.dtype() == gamma.dtype() && x.dtype() == beta.dtype(), "x, gamma, beta must have same dtype");

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  TORCH_CHECK(scale.size(0) == M && scale.size(1) == N, "scale must be shape [M, N]");
  TORCH_CHECK(shift.size(0) == M && shift.size(1) == N, "shift must be shape [M, N]");
  TORCH_CHECK(gamma.numel() == N && beta.numel() == N, "gamma/beta must be length N");

  auto y = torch::empty_like(x);

  cutlass::MatrixCoord size(static_cast<int>(M), static_cast<int>(N));
  auto layout_row_major = cutlass::layout::RowMajor(static_cast<int>(N));

  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_in(
      reinterpret_cast<T*>(const_cast<void*>(x.data_ptr())), layout_row_major);
  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_out(
      reinterpret_cast<T*>(y.data_ptr()), layout_row_major);
  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_gamma(
      reinterpret_cast<T*>(const_cast<void*>(gamma.data_ptr())), cutlass::layout::RowMajor(static_cast<int>(N)));
  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_beta(
      reinterpret_cast<T*>(const_cast<void*>(beta.data_ptr())), cutlass::layout::RowMajor(static_cast<int>(N)));
  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_scale(
      reinterpret_cast<T*>(const_cast<void*>(scale.data_ptr())), layout_row_major);
  cutlass::TensorRef<T, cutlass::layout::RowMajor> ref_shift(
      reinterpret_cast<T*>(const_cast<void*>(shift.data_ptr())), layout_row_major);

  auto stream = at::cuda::getCurrentCUDAStream();
  cutlass::layernorm_fused_scale_shift<T>(
      size,
      ref_out,
      ref_in,
      ref_gamma,
      ref_beta,
      ref_scale,
      ref_shift,
      stream.stream());
  return y;
}

} // namespace

torch::Tensor fuse_layernorm_scale_shift_infer(
    torch::Tensor x,
    c10::optional<torch::Tensor> gamma,
    c10::optional<torch::Tensor> beta) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  // Fallback for bf16: compute in fp32 then cast back to bf16
  if (x.dtype() == torch::kBFloat16) {
    auto x_f = x.to(torch::kFloat32);
    c10::optional<torch::Tensor> gamma_f =
        gamma.has_value() && gamma->defined() ? c10::optional<torch::Tensor>(gamma->to(torch::kFloat32)) : c10::nullopt;
    c10::optional<torch::Tensor> beta_f =
        beta.has_value() && beta->defined() ? c10::optional<torch::Tensor>(beta->to(torch::kFloat32)) : c10::nullopt;
    auto y_f = layernorm_cutlass<float>(x_f, gamma_f, beta_f);
    return y_f.to(torch::kBFloat16);
  }
  if (x.dtype() == torch::kFloat32) {
    return layernorm_cutlass<float>(x, gamma, beta);
  } else if (x.dtype() == torch::kFloat16) {
    // at::Half storage maps to __half on device; cast pointer in templated path
    return layernorm_cutlass<half>(x, gamma, beta);
  } else {
    TORCH_CHECK(false, "Unsupported dtype. Use float32 or float16.");
  }
}

torch::Tensor fuse_layernorm_scale_shift_fused_infer(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor scale,
    torch::Tensor shift) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  // Fallback for bf16: compute in fp32 then cast back
  if (x.dtype() == torch::kBFloat16) {
    auto x_f = x.to(torch::kFloat32);
    auto g_f = gamma.to(torch::kFloat32);
    auto b_f = beta.to(torch::kFloat32);
    auto s_f = scale.to(torch::kFloat32);
    auto sh_f = shift.to(torch::kFloat32);
    auto y_f = launch_fuse_layernorm_scale_shift_fused_impl<float>(x_f, g_f, b_f, s_f, sh_f);
    return y_f.to(torch::kBFloat16);
  }
  if (x.dtype() == torch::kFloat32) {
    return launch_fuse_layernorm_scale_shift_fused_impl<float>(x, gamma, beta, scale, shift);
  } else if (x.dtype() == torch::kFloat16) {
    return launch_fuse_layernorm_scale_shift_fused_impl<half>(x, gamma, beta, scale, shift);
  } else {
    TORCH_CHECK(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layernorm_cutlass", &fuse_layernorm_scale_shift_infer, "CUTLASS device LayerNorm (inference)");
  m.def("fuse_layernorm_scale_shift", &fuse_layernorm_scale_shift_fused_infer, "CUTLASS device LayerNorm fused with scale/shift (inference)");
}


