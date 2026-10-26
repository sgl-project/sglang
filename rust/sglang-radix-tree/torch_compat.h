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
// tch 0.24 still binds the `cholesky` and `qr` aliases PyTorch 2.14 removed;
// linalg differs in errors and qr's signature, so stub rather than remap.
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

// Safe as global macros only because no PyTorch 2.14 header uses these names;
// the include above declares at::Tensor for the signatures.
#define cholesky(...) cholesky_unavailable(__VA_ARGS__)
#define cholesky_out(...) cholesky_out_unavailable(__VA_ARGS__)
#define qr(...) qr_unavailable(__VA_ARGS__)
#define qr_out(...) qr_out_unavailable(__VA_ARGS__)
#endif
