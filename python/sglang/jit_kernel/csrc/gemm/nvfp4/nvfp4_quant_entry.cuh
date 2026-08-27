/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

void scaled_fp4_quant_sm100a_sm120a(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output_sf,
    tvm::ffi::TensorView input_sf);

void scaled_fp4_experts_quant_sm100a(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView output_scale,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView input_global_scale,
    tvm::ffi::TensorView input_offset_by_experts,
    tvm::ffi::TensorView output_scale_offset_by_experts);

void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView output_scale,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView input_global_scale,
    tvm::ffi::TensorView mask,
    bool use_silu_and_mul);

void scaled_fp4_quant(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output_sf,
    tvm::ffi::TensorView input_sf) {
  scaled_fp4_quant_sm100a_sm120a(output, input, output_sf, input_sf);
}

void scaled_fp4_experts_quant(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView output_scale,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView input_global_scale,
    tvm::ffi::TensorView input_offset_by_experts,
    tvm::ffi::TensorView output_scale_offset_by_experts) {
  scaled_fp4_experts_quant_sm100a(
      output, output_scale, input, input_global_scale, input_offset_by_experts, output_scale_offset_by_experts);
}

void silu_and_mul_scaled_fp4_experts_quant(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView output_scale,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView input_global_scale,
    tvm::ffi::TensorView mask,
    bool use_silu_and_mul) {
  silu_and_mul_scaled_fp4_experts_quant_sm100a(output, output_scale, input, input_global_scale, mask, use_silu_and_mul);
}
