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

"""Backwards-compatibility shim.

DCP primitives moved to ``sglang.srt.layers.cp.dcp`` (consolidated with the
PR #25090 Triton/MHA path). Import from there directly. This module re-exports
the original public names; ``cp_lse_ag_out_rs`` here is the MLA (Triton /
reduce-scatter) variant, now ``cp_lse_ag_out_rs_mla``.

TODO(dcp-refactor P3b): delete once all importers move to layers.cp.dcp.
"""

from sglang.srt.layers.cp.dcp.comm import (  # noqa: F401
    _all_gather_dcp_kv_cache,
    all_gather_kv_cache_for_dcp,
    all_gather_kv_cache_for_mha_chunk_extend,
    all_gather_kv_cache_for_mha_extend,
    all_gather_kv_cache_for_mla_extend,
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla as cp_lse_ag_out_rs,
    dcp_enabled,
    get_attention_dcp_rank,
    get_attention_dcp_world_size,
)
from sglang.srt.layers.cp.dcp.kernels import (  # noqa: F401
    CPTritonContext,
    _correct_attn_cp_out_kernel,
    correct_attn_out,
    create_dcp_kv_indices,
    update_kv_lens_and_indices,
)
from sglang.srt.layers.cp.dcp.layout import (  # noqa: F401
    filter_dcp_local_kv_indices,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.cp.dcp.metadata import (  # noqa: F401
    DecodeContextParallelMetadata,
)
from sglang.srt.layers.cp.dcp.planner import (  # noqa: F401
    plan_dcp_decode_metadata,
    prepare_decode_context_parallel_metadata,
)
