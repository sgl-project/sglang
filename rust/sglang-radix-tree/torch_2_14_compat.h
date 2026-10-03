#pragma once

// Shims for the C++ torch:: wrappers that PyTorch 2.14 dropped but tch 0.24's
// generated bindings still emit. Retire this header once tch-rs publishes a
// release regenerated against PyTorch 2.14 — then the generated bindings
// will call at::linalg_cholesky / at::linalg_qr directly and these shims
// become dead code. As of 2026-10-01 the latest tch main (4227b89, "Update
// for pytorch 2.13") still emits torch::cholesky / torch::qr.

#include <torch/version.h>

#if TORCH_VERSION_MAJOR > 2 ||                                                 \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 14)
#include <tuple>
#include <ATen/ops/linalg_cholesky.h>
#include <ATen/ops/linalg_qr.h>
namespace torch {
inline at::Tensor cholesky(const at::Tensor& self, bool upper = false) {
  return at::linalg_cholesky(self, upper);
}
inline at::Tensor& cholesky_out(at::Tensor& out, const at::Tensor& self,
                                bool upper = false) {
  return at::linalg_cholesky_out(out, self, upper);
}
inline ::std::tuple<at::Tensor, at::Tensor> qr(const at::Tensor& self,
                                               bool some = true) {
  return at::linalg_qr(self, some ? "reduced" : "complete");
}
inline ::std::tuple<at::Tensor&, at::Tensor&> qr_out(at::Tensor& Q,
                                                    at::Tensor& R,
                                                    const at::Tensor& self,
                                                    bool some = true) {
  return at::linalg_qr_out(Q, R, self, some ? "reduced" : "complete");
}
} // namespace torch
#endif
