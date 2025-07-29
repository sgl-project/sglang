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
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From csrc/gemm - only per_token_quant_fp8
   */
  m.def("sgl_per_token_quant_fp8(Tensor input, Tensor output_q, Tensor output_s) -> ()");
  m.impl("sgl_per_token_quant_fp8", torch::kCUDA, &sgl_per_token_quant_fp8);
}

REGISTER_EXTENSION(common_ops)
