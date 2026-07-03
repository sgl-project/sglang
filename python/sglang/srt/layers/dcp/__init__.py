# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Decode Context Parallel (DCP) — consolidated home for the primitives that
were previously split between layers/attention/utils.py (PR #25090, Triton/MHA)
and layers/utils/dcp_utils.py (PR #14194, FlashInfer-MLA).

The two ``cp_lse_ag_out_rs`` variants are kept distinct (``_mha`` torch/all-reduce,
``_mla`` Triton/reduce-scatter) because their bodies are backend-forced.

Only the symbols imported by code OUTSIDE this subpackage are re-exported here.
Package-internal helpers (the @triton.jit kernels, ``CPTritonContext``,
``correct_attn_out``, ``create_dcp_kv_indices``, ``update_kv_lens_and_indices``,
``_all_gather_dcp_kv_cache``) stay private to their submodules — import them from
``sglang.srt.layers.dcp.{kernels,comm}`` if ever needed internally."""

from sglang.srt.layers.dcp.base import (
    DecodeContextParallelStrategyKind,
    get_dcp_strategy,
    init_dcp_strategy,
    is_dcp_enabled,
)
from sglang.srt.layers.dcp.comm import (
    all_gather_kv_cache_for_dcp,
    all_gather_kv_cache_for_mha_chunk_extend,
    all_gather_kv_cache_for_mha_extend,
    all_gather_kv_cache_for_mla_extend,
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mha,
    cp_lse_ag_out_rs_mla,
    dcp_enabled,
    get_attention_dcp_rank,
    get_attention_dcp_world_size,
)
from sglang.srt.layers.dcp.kernels import create_triton_kv_indices_for_dcp_triton
from sglang.srt.layers.dcp.layout import (
    filter_dcp_local_kv_indices,
    get_dcp_lens,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.layers.dcp.utils import (
    dcp_build_decode_metadata,
    dcp_gather_decode_query,
    dcp_merge_attention,
    dcp_plan_decode_metadata,
    dcp_shard_decode_kv_indices,
    dcp_update_local_decode_kv_lens,
)

# NOTE: strategy.py and planner.py are intentionally NOT imported here. planner
# depends on server_args (get_global_server_args), and strategy imports planner;
# this package-init executes at module-load time for every eager importer of the DCP
# primitives — triton_backend, mem_cache.memory_pool, mem_cache.triton_ops.mla_buffer,
# mem_cache.kv_cache_builder, the FlashInfer-MLA / FlashMLA backends, and the deepseek
# forward methods. Keeping the init server_args-free avoids a load-time import edge.
# The concrete strategy is reached via get_dcp_strategy() (which lazily imports it);
# import planner functions from sglang.srt.layers.dcp.planner directly if needed.

__all__ = [
    "DecodeContextParallelMetadata",
    "DecodeContextParallelStrategyKind",
    "all_gather_kv_cache_for_dcp",
    "all_gather_kv_cache_for_mha_chunk_extend",
    "all_gather_kv_cache_for_mha_extend",
    "all_gather_kv_cache_for_mla_extend",
    "all_gather_q_for_mla_decode",
    "cp_lse_ag_out_rs_mha",
    "cp_lse_ag_out_rs_mla",
    "create_triton_kv_indices_for_dcp_triton",
    "dcp_build_decode_metadata",
    "dcp_enabled",
    "dcp_gather_decode_query",
    "dcp_merge_attention",
    "dcp_plan_decode_metadata",
    "dcp_shard_decode_kv_indices",
    "dcp_update_local_decode_kv_lens",
    "filter_dcp_local_kv_indices",
    "get_attention_dcp_rank",
    "get_attention_dcp_world_size",
    "get_dcp_lens",
    "get_dcp_strategy",
    "init_dcp_strategy",
    "is_dcp_enabled",
    "update_local_kv_lens_for_dcp",
]
