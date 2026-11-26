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
The other helpers stay in their defining modules: the @triton.jit kernels,
``CPTritonContext``, ``correct_attn_out``, ``create_dcp_kv_indices`` and
``update_kv_lens_and_indices`` in ``sglang.kernels.ops.attention.dcp_kernels``,
``_all_gather_dcp_kv_cache`` in ``sglang.srt.layers.dcp.comm``."""

from sglang.kernels.ops.attention.dcp_kernels import (
    create_triton_kv_indices_for_dcp_triton,
)
from sglang.srt.layers.dcp.comm import (
    all_gather_kv_cache_for_dcp,
    all_gather_kv_cache_for_mha_chunk_extend,
    all_gather_kv_cache_for_mha_extend,
    all_gather_kv_cache_for_mla_extend,
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mha,
    cp_lse_ag_out_rs_mla,
    dcp_a2a_lse_reduce,
    dcp_push_combine_fits,
    dcp_push_combine_reduce,
    init_dcp_push_combine,
    init_fi_a2a_workspace,
)
from sglang.srt.layers.dcp.layout import (
    filter_dcp_local_kv_indices,
    get_dcp_lens,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata

# planner.py is not imported here: it imports model_executor.forward_context, and
# this package init runs at module load for every importer of the DCP primitives.
# Import planner functions from sglang.srt.layers.dcp.planner directly.

__all__ = [
    "DecodeContextParallelMetadata",
    "dcp_a2a_lse_reduce",
    "dcp_push_combine_fits",
    "dcp_push_combine_reduce",
    "init_dcp_push_combine",
    "init_fi_a2a_workspace",
    "all_gather_kv_cache_for_dcp",
    "all_gather_kv_cache_for_mha_chunk_extend",
    "all_gather_kv_cache_for_mha_extend",
    "all_gather_kv_cache_for_mla_extend",
    "all_gather_q_for_mla_decode",
    "cp_lse_ag_out_rs_mha",
    "cp_lse_ag_out_rs_mla",
    "create_triton_kv_indices_for_dcp_triton",
    "filter_dcp_local_kv_indices",
    "get_dcp_lens",
    "update_local_kv_lens_for_dcp",
]
