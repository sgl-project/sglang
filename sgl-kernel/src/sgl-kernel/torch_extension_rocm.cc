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
#include <torch/library.h>

#include "sgl_kernels_ops.h"

TORCH_LIBRARY_EXPAND(sgl_kernels, m) {
  // moe_align_block_size
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! token_cnts_buffer, Tensor! cumsum_buffer) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);
}

REGISTER_EXTENSION(_kernels)
