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
``_mla`` Triton/reduce-scatter) because their bodies are backend-forced."""

from sglang.srt.layers.cp.dcp.comm import (
    _all_gather_dcp_kv_cache,
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
from sglang.srt.layers.cp.dcp.kernels import (
    CPTritonContext,
    _correct_attn_cp_out_kernel,
    correct_attn_out,
    create_dcp_kv_indices,
    create_triton_kv_indices_for_dcp_triton,
    update_kv_lens_and_indices,
)
from sglang.srt.layers.cp.dcp.layout import (
    filter_dcp_local_kv_indices,
    get_dcp_lens,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.cp.dcp.metadata import DecodeContextParallelMetadata

# NOTE: planner.py is intentionally NOT imported here. It depends on server_args,
# and this package-init runs whenever attention/utils.py (Triton path) imports
# comm/kernels/layout. Keeping the init free of server_args avoids a module-load
# import edge. Import planner functions from sglang.srt.layers.cp.dcp.planner.

__all__ = [
    "CPTritonContext",
    "DecodeContextParallelMetadata",
    "_all_gather_dcp_kv_cache",
    "_correct_attn_cp_out_kernel",
    "all_gather_kv_cache_for_dcp",
    "all_gather_kv_cache_for_mha_chunk_extend",
    "all_gather_kv_cache_for_mha_extend",
    "all_gather_kv_cache_for_mla_extend",
    "all_gather_q_for_mla_decode",
    "correct_attn_out",
    "cp_lse_ag_out_rs_mha",
    "cp_lse_ag_out_rs_mla",
    "create_dcp_kv_indices",
    "create_triton_kv_indices_for_dcp_triton",
    "dcp_enabled",
    "filter_dcp_local_kv_indices",
    "get_attention_dcp_rank",
    "get_attention_dcp_world_size",
    "get_dcp_lens",
    "update_kv_lens_and_indices",
    "update_local_kv_lens_for_dcp",
]
