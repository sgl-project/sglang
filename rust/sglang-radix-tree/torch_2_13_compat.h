#pragma once

#include <stdexcept>
#include <torch/version.h>

#if TORCH_VERSION_MAJOR > 2 ||                                                 \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 13)
// Keep tch 0.24's removed alignment wrappers as explicit runtime errors.
#define align_as(...)                                                          \
  alias();                                                                     \
  throw std::runtime_error("align_as is unavailable in PyTorch 2.13+")
#define align_tensors(...)                                                     \
  autograd::variable_list{};                                                   \
  throw std::runtime_error("align_tensors is unavailable in PyTorch 2.13+")
#endif

#if TORCH_VERSION_MAJOR > 2 ||                                                 \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 14)
// PyTorch 2.14 drops the deprecated `cholesky` and `qr` aliases that tch 0.24
// still binds; their linalg replacements differ in error handling and, for
// `qr`, in signature, so stub them out rather than silently remapping.
#include <ATen/core/Tensor.h>

#include <tuple>

namespace torch {

inline at::Tensor cholesky_unavailable(const at::Tensor &, bool) {
  throw std::runtime_error(
      "cholesky is unavailable in PyTorch 2.14+; use linalg_cholesky");
}

inline at::Tensor &cholesky_out_unavailable(at::Tensor &, const at::Tensor &,
                                            bool) {
  throw std::runtime_error(
      "cholesky_out is unavailable in PyTorch 2.14+; use linalg_cholesky_out");
}

inline std::tuple<at::Tensor, at::Tensor> qr_unavailable(const at::Tensor &,
                                                         bool) {
  throw std::runtime_error("qr is unavailable in PyTorch 2.14+; use linalg_qr");
}

inline std::tuple<at::Tensor &, at::Tensor &>
qr_out_unavailable(at::Tensor &, at::Tensor &, const at::Tensor &, bool) {
  throw std::runtime_error(
      "qr_out is unavailable in PyTorch 2.14+; use linalg_qr_out");
}

} // namespace torch

// Defined after the ATen include above so the declarations these replace are
// never themselves rewritten.
#define cholesky(...) cholesky_unavailable(__VA_ARGS__)
#define cholesky_out(...) cholesky_out_unavailable(__VA_ARGS__)
#define qr(...) qr_unavailable(__VA_ARGS__)
#define qr_out(...) qr_out_unavailable(__VA_ARGS__)
#endif
