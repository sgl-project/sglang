// SPDX-License-Identifier: Apache-2.0
// SageAttention-3 extension registration. Built as the separate sage3_ops
// library (SM120a-only); see cmake/sage3.cmake. The kernels themselves live in
// csrc/attention/sage3.cu, which #includes upstream SageAttention .cu sources.

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

#include "sgl_kernel_ops.h"  // REGISTER_EXTENSION macro

namespace sgl_kernel {
std::vector<at::Tensor> sage3_mha_fwd(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& sfq,
    const at::Tensor& sfk,
    const at::Tensor& sfv,
    const at::Tensor& delta_s,
    int64_t unpadded_k,
    std::optional<at::Tensor> out_,
    double softmax_scale,
    bool is_causal,
    bool per_block_mean,
    bool is_bf16);
void sage3_scaled_fp4_quant(
    const at::Tensor& input,
    const at::Tensor& output,
    const at::Tensor& output_sf,
    int64_t tensor_layout,
    int64_t variant);
}  // namespace sgl_kernel

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  // From csrc/attention/sage3 (SageAttention-3 Blackwell FP4)
  m.def(
      "sage3_mha_fwd(Tensor! q, Tensor k, Tensor v, "
      "Tensor sfq, Tensor sfk, Tensor sfv, Tensor delta_s, "
      "int unpadded_k, Tensor? out, "
      "float softmax_scale, bool is_causal, bool per_block_mean, bool is_bf16) "
      "-> Tensor[]");
  m.impl("sage3_mha_fwd", torch::kCUDA, &sgl_kernel::sage3_mha_fwd);

  m.def(
      "scaled_fp4_quant(Tensor input, Tensor! output, Tensor! output_sf, "
      "int tensor_layout, int variant) -> ()");
  m.impl("scaled_fp4_quant", torch::kCUDA, &sgl_kernel::sage3_scaled_fp4_quant);
}

REGISTER_EXTENSION(sage3_ops)
