#pragma once

// SiTU (SoftCap-GLU), the one definition K3 uses.
//
//   gate_out = beta * tanh(gate / beta) * sigmoid(gate)
//   up_out   = linear_beta * tanh(up / linear_beta)     (identity if !kHasLinearBeta)
//   out      = gate_out * up_out
//
// Unlike SiLU no external swiglu clamp is needed: the tanh softcap bounds the
// output to |beta * linear_beta| (< FP8_E4M3_MAX), which is what makes the fp8
// post-quant path safe.
//
// This header exists because the formula was written twice -- inlined in
// situ_and_mul.cuh and again as an fp32x2 helper in
// situ_and_mul_masked_post_quant.cuh. It lives next to those two consumers
// rather than in a tree-wide header.

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

namespace sglang {

namespace kimi_k3 {

/// One SiTU element. `sigmoid_fast` is `1/(1+expf(-x))` (math.cuh), i.e. the
/// same expression both call sites used before they were folded together.
template <bool kHasLinearBeta>
SGL_DEVICE float situ_activate(float g, float u, float beta, float inv_beta, float linear_beta, float inv_linear_beta) {
  const float gate_out = beta * tanhf(g * inv_beta) * device::math::sigmoid_fast(g);
  float up_out;
  if constexpr (kHasLinearBeta) {
    up_out = linear_beta * tanhf(u * inv_linear_beta);
  } else {
    up_out = u;
  }
  return gate_out * up_out;
}

}  // namespace kimi_k3

}  // namespace sglang
