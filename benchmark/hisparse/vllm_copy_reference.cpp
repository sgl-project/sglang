// SPDX-License-Identifier: Apache-2.0
// Bind the unmodified vLLM Ascend adapter to its private original OPP.
#include "aclnn_torch_adapter/op_api_common.h"
#include <ATen/ATen.h>
#include <array>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#define vllm_ascend original_copy_reference
#include "attention/fused_scatter_copy_sparse_flash_attention/fused_scatter_copy_sparse_flash_attention_torch_adpt.h"
#undef vllm_ascend

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

TORCH_LIBRARY(vllm_copy_reference, m) {
  m.def("run(Tensor query_rope, Tensor query, Tensor actual_seq_lengths_query, "
        "Tensor actual_seq_lengths_kv, Tensor num_cache_tokens, Tensor "
        "topk_dst_slots, "
        "Tensor topk_src_ids, Tensor topk_miss_counts, Tensor miss_src_ids, "
        "Tensor miss_dst_slots, Tensor miss_counts, Tensor hbm_block_table, "
        "Tensor dram_block_table, Tensor(a!) hbm_k_rope, Tensor(b!) "
        "hbm_kv_cache, "
        "Tensor dram_k_rope, Tensor dram_kv_cache, float scale_value, "
        "Tensor(c!) attention_out) -> ()");
  m.impl(
      "run", torch::kPrivateUse1,
      &original_copy_reference::npu_fused_scatter_copy_sparse_flash_attention);
}
