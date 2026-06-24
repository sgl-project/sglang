/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streaming_dit_bindings.h"

#include "pyext/streaming_dit_bridge.h"
#include <cuda_runtime.h>

namespace py = pybind11;

namespace omnidreams_singleview {

void bind_optimized_dit(py::module_& module) {
  module.def(
      "optimized_dit_forward",
      &::optimized_dit_forward,
      py::arg("x_new"),
      py::arg("condition_mask_patched"),
      py::arg("hdmap_patched"),
      py::arg("timesteps"),
      py::arg("rope_emb"),
      py::arg("k_cross_caches"),
      py::arg("v_cross_caches"),
      py::arg("k_self_caches"),
      py::arg("v_self_caches"),
      py::arg("self_attn_write_start"),
      py::arg("weights"),
      py::arg("config"));
  module.def(
      "sage3_quantize_cross_kv_bf16",
      &sage3_quantize_cross_kv_bf16,
      py::arg("k_bmhd"),
      py::arg("v_bmhd"));
  module.def(
      "cosmos_test_linear_fp8",
      &cosmos_test_linear_fp8,
      py::arg("input"),
      py::arg("weight_fp8_u8"),
      py::arg("weight_scale") = py::none(),
      py::arg("bias") = py::none(),
      py::arg("gelu") = false);
  module.def(
      "cosmos_test_linear_fp8_out_fp8",
      &cosmos_test_linear_fp8_out_fp8,
      py::arg("input_bf16"),
      py::arg("weight_fp8_u8"),
      py::arg("weight_scale"));
  module.def(
      "cosmos_test_linear_fp8_gelu_out_fp8",
      &cosmos_test_linear_fp8_gelu_out_fp8,
      py::arg("input_bf16"),
      py::arg("weight_fp8_u8"),
      py::arg("weight_scale"),
      py::arg("output_scale") = 1.0,
      py::arg("alias_output") = false);
  module.def(
      "cosmos_test_linear_fp8_scaled_bf16",
      &cosmos_test_linear_fp8_scaled_bf16,
      py::arg("input_bf16"),
      py::arg("weight_fp8_u8"),
      py::arg("weight_scale"));
  module.def(
      "cosmos_test_linear_fp8_residual_scaled_bf16",
      &cosmos_test_linear_fp8_residual_scaled_bf16,
      py::arg("input_bf16"),
      py::arg("weight_fp8_u8"),
      py::arg("alpha"),
      py::arg("residual_bf16"));
  module.def(
      "cosmos_test_fp8_sdpa_selection",
      &cosmos_test_fp8_sdpa_selection,
      py::arg("B"),
      py::arg("Mq"),
      py::arg("Mk"),
      py::arg("H"),
      py::arg("D"));
  module.def(
      "cosmos_test_fp8_dense_ref_sdpa",
      &cosmos_test_fp8_dense_ref_sdpa,
      py::arg("q_fp8_u8"),
      py::arg("k_fp8_u8"),
      py::arg("v_fp8_u8"),
      py::arg("causal"));
  module.def(
      "cosmos_test_fp8_cudnn_sdpa",
      &cosmos_test_fp8_cudnn_sdpa,
      py::arg("q_fp8_u8"),
      py::arg("k_fp8_u8"),
      py::arg("v_fp8_u8"),
      py::arg("causal"));
  module.def(
      "cosmos_test_fp8_attention_backend",
      &cosmos_test_fp8_attention_backend,
      py::arg("q_fp8_u8"),
      py::arg("k_fp8_u8"),
      py::arg("v_fp8_u8"),
      py::arg("causal"),
      py::arg("backend"));
  module.def("sage3_is_built", &sage3_is_built);
  module.def("sage3_is_runtime_supported", &sage3_is_runtime_supported, py::arg("device"));
  module.def("optimized_dit_supports_block_mod_cache", []() { return true; });
  module.def("optimized_dit_supports_hdmap_cache", []() { return true; });
}

}  // namespace omnidreams_singleview
