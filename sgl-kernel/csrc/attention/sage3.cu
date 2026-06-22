// SPDX-License-Identifier: Apache-2.0
// SageAttention-3 Blackwell FP4 attention, contributed to sgl-kernel.
//
// Wraps upstream SageAttention (thu-ml/SageAttention, commit d1a57a5) by directly
// #including its .cu sources. Each upstream .cu registers its own standalone
// pybind module; we suppress that with the PYBIND11_MODULE neutralization trick
// (a UNIQUE stub name per #include — a shared name would collide because both
// files define it).
//
// Exposes two ops under the sgl_kernel torch library:
//   sgl_kernel.sage3_mha_fwd(...)      — upstream mha_fwd (FP4 attention forward)
//   sgl_kernel.scaled_fp4_quant(...)   — upstream scaled_fp4_quant family (variant 0/1/2)
//
// SM120a only (arch-conditional FP4 MMA). Built as a separate sage3_ops library
// (cmake/sage3.cmake) gated on SM120a, NOT added to the shared SOURCES — adding
// it there would break the sm90/sm100 builds on the arch-conditional kernels.
//
// SageAttention source is fetched via the FetchContent in cmake/sage3.cmake and
// exposed as ${repo-sageattention_SOURCE_DIR}.

#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Neutralize upstream api.cu's PYBIND11_MODULE (defines mha_fwd).
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) \
  static void sgl_sage3_api_stub(pybind11::module_& variable)
#include "sageattention3_blackwell/sageattn3/blackwell/api.cu"

// Neutralize upstream fp4_quantization_4d.cu's PYBIND11_MODULE (defines
// scaled_fp4_quant / _permute / _trans). Use a UNIQUE stub name — reusing the
// api.cu stub name would trigger "already defined".
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) \
  static void sgl_sage3_quant_stub(pybind11::module_& variable)
#include "sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu"

#undef PYBIND11_MODULE

namespace sgl_kernel {

// FP4 attention forward pass (Blackwell SM120a).
// q/k/v: uint8 [B, H, M, D/2] (FP4 packed); *_sf: fp8 e4m3 block scales;
// delta_s: bf16/fp32 [B, H, Mq, Mk]. Returns {out [B, H, Mq, D] bf16, softmax_lse}.
std::vector<at::Tensor> sage3_mha_fwd(
    at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& sfq, const at::Tensor& sfk, const at::Tensor& sfv,
    const at::Tensor& delta_s, int64_t unpadded_k,
    std::optional<at::Tensor> out_,
    double softmax_scale, bool is_causal, bool per_block_mean, bool is_bf16) {
  c10::optional<at::Tensor> out_opt = out_;
  return mha_fwd(q, k, v, sfq, sfk, sfv, delta_s,
                 static_cast<int>(unpadded_k), out_opt,
                 static_cast<float>(softmax_scale),
                 is_causal, per_block_mean, is_bf16);
}

// FP4 per-token quantization (3 layout variants). input: bf16 [B,H,M,D];
// tensor_layout=1 (BMHD). out: uint8 FP4, out_sf: fp8 e4m3 scales [B,H,M,D/16].
// variant: 0=plain, 1=permute (for K), 2=trans (for V).
void sage3_scaled_fp4_quant(
    const at::Tensor& input, const at::Tensor& output,
    const at::Tensor& output_sf, int64_t tensor_layout, int64_t variant) {
  switch (variant) {
    case 0: scaled_fp4_quant(input, output, output_sf, static_cast<int>(tensor_layout)); break;
    case 1: scaled_fp4_quant_permute(input, output, output_sf, static_cast<int>(tensor_layout)); break;
    case 2: scaled_fp4_quant_trans(input, output, output_sf, static_cast<int>(tensor_layout)); break;
    default: TORCH_CHECK(false, "sage3_scaled_fp4_quant: variant must be 0|1|2");
  }
}

}  // namespace sgl_kernel
