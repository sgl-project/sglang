// SPDX-License-Identifier: Apache-2.0
// Standalone registration of the upstream 19-tensor LIM ABI for SGLang.
#include <torch/extension.h>
#include <torch/library.h>

#include <array>
#include <cstring>
#include <limits.h>
#include <tuple>
#include <unistd.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/fused_lightning_indexer_manage/fused_lightning_indexer_manage_torch_adpt.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

// Other optional Ascend operators may already own this torch namespace.
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
  ops.def(
      "npu_fused_lightning_indexer_manage("
      "Tensor index_weights, Tensor query_dequant_scale, Tensor query, "
      "Tensor index_key_dequant_scale, Tensor index_key_cache, "
      "Tensor index_block_table, Tensor actual_seq_lengths_query, "
      "Tensor actual_seq_lengths_key, Tensor offload_seq_lengths_key, "
      "Tensor num_cache_tokens, Tensor request_state, Tensor req_pool_entries, "
      "Tensor(a!) cache_slots_pool, Tensor(b!) topk_src_ids, "
      "Tensor(c!) topk_dst_slots, Tensor(d!) topk_miss_counts, "
      "Tensor(e!) miss_src_ids, Tensor(f!) miss_dst_slots, "
      "Tensor(g!) miss_counts) -> ()");
  ops.impl("npu_fused_lightning_indexer_manage", torch::kPrivateUse1,
           &vllm_ascend::npu_fused_lightning_indexer_manage);
}
