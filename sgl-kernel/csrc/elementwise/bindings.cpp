#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Forward declarations of CUDA implementations in fused_layernorm_scale_shift.cu
torch::Tensor device_layernorm(torch::Tensor x,
                               const c10::optional<torch::Tensor>& gamma,
                               const c10::optional<torch::Tensor>& beta);

torch::Tensor device_layernorm_fuse_scale_shift(torch::Tensor x,
                                                torch::Tensor gamma,
                                                torch::Tensor beta,
                                                torch::Tensor scale,
                                                torch::Tensor shift);

torch::Tensor device_scale_residual_layernorm_fuse_scale_shift(torch::Tensor residual,
                                                               torch::Tensor x,
                                                               torch::Tensor gamma,
                                                               torch::Tensor beta,
                                                               torch::Tensor scale,
                                                               torch::Tensor shift,
                                                               const c10::optional<torch::Tensor>& gate_opt);

PYBIND11_MODULE(fused_layernorm_scale_shift, m) {
  m.def("device_layernorm",
        &device_layernorm,
        "LayerNorm on device (supports float32/float16/bfloat16). "
        "x: [M, N], gamma/beta: [N] (optional). Returns y with same shape/dtype as x.");
  m.def("device_layernorm_fuse_scale_shift",
        &device_layernorm_fuse_scale_shift,
        "LayerNorm then fused scale/shift on device. "
        "x: [M, N], gamma/beta: [N], scale/shift: [M, N] or [B, F, 1, N]. "
        "Returns y with same shape/dtype as x.");
  m.def("device_scale_residual_layernorm_fuse_scale_shift",
        &device_scale_residual_layernorm_fuse_scale_shift,
        "Fused residual (+ optional gate) + LayerNorm + scale/shift on device. "
        "residual/x: [M, N], gamma/beta: [N], "
        "scale/shift: [M, N] or [B, F, 1, N], "
        "gate: None, [M, N], [B, 1, N], or [B, F, 1, N]. "
        "Returns y with same shape/dtype as x.");
}