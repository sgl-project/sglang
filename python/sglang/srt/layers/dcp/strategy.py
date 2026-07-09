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

"""``DecodeContextParallelStrategy`` — the decode context parallel strategy.

STANDALONE: it does NOT inherit the prefill ``ContextParallelStrategy``. Decode-CP is
an orthogonal parallelism on the ``_DCP`` group (owner rule ``pos % dcp_size == rank``)
applied during decode. The class is structurally aligned with the CP-v2 strategies for
familiarity — the package mirrors CP's ``base``/``strategy``/``utils``/``__init__``
layout and ``init_/get_/is_`` singleton naming (see ``layers/dcp/base.py`` and
``layers/dcp/utils.py``) — but shares no code with them (no prefill ``shard_*`` /
``run_attention`` / ``materialize_full_kv``, no prefill-axis ``cp_rank``).

Its methods are a behavior-preserving *seam* over the ``layers/dcp`` primitives
(comm/layout/planner) — no new logic. Imported lazily (only by ``init_dcp_strategy``),
so pulling ``planner`` (-> ``server_args``) adds no load-time edge to the DCP package
init.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.layers.dcp.base import DecodeContextParallelStrategyKind
from sglang.srt.layers.dcp.comm import (
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mha,
    cp_lse_ag_out_rs_mla,
)
from sglang.srt.layers.dcp.layout import (
    filter_dcp_local_kv_indices,
    get_dcp_lens,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.dcp.planner import (
    plan_dcp_decode_metadata,
    prepare_decode_context_parallel_metadata,
)


class DecodeContextParallelStrategy:
    """Decode context parallel on the ``_DCP`` group (owner rule pos % N == rank)."""

    name = "decode_context_parallel"
    kind = DecodeContextParallelStrategyKind.DECODE

    def __init__(self, dcp_size: int):
        self.dcp_size = dcp_size

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if self.dcp_size <= 1:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        return forward_mode is None or forward_mode.is_decode()

    # -- KV layout -------------------------------------------------------------
    def local_decode_kv_lens(
        self, lens: Any, dcp_size: int, dcp_rank: int, start: Any = None
    ) -> Any:
        return get_dcp_lens(lens, dcp_size, dcp_rank, start)

    def update_local_decode_kv_lens(self, kv_len_arr: Any) -> None:
        update_local_kv_lens_for_dcp(kv_len_arr)

    def shard_decode_kv_indices(self, kv_indices: Any) -> Any:
        return filter_dcp_local_kv_indices(kv_indices=kv_indices)

    # -- metadata --------------------------------------------------------------
    def build_decode_metadata(self, **kwargs: Any) -> Any:
        return prepare_decode_context_parallel_metadata(**kwargs)

    def plan_decode_metadata(self, **kwargs: Any) -> Any:
        return plan_dcp_decode_metadata(**kwargs)

    # -- attention -------------------------------------------------------------
    def gather_decode_query(self, q_nope_out: Any, q_pe: Any) -> Any:
        return all_gather_q_for_mla_decode(q_nope_out, q_pe)

    def merge_decode_attention(
        self,
        cp_attn_out: Any,
        cp_attn_lse: Any,
        cp_group: Any,
        *,
        backend: str,
        return_lse: bool = False,
        ctx: Optional[Any] = None,
    ) -> Any:
        if backend == "mha":
            return cp_lse_ag_out_rs_mha(
                cp_attn_out, cp_attn_lse, cp_group, return_lse=return_lse
            )
        if backend == "mla":
            return cp_lse_ag_out_rs_mla(cp_attn_out, cp_attn_lse, cp_group, ctx=ctx)
        raise ValueError(
            f"unknown decode-CP attention backend {backend!r}; expected 'mha' or 'mla'"
        )
