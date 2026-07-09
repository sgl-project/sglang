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

"""Runtime facade for the decode context parallel (DCP) strategy.

Mirrors ``layers/cp/utils.py``: backends call these thin, None-safe wrappers instead
of reaching ``get_dcp_strategy()`` directly, so the strategy-existence guard lives in
one place. When DCP is not configured each wrapper is a no-op / identity that matches
the call site's original inline guard, so callers keep their surrounding mode/platform
gates (``dcp_enabled()``, ``dcp_size > 1``) and route only the operation here.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.layers.dcp.base import get_dcp_strategy


def dcp_shard_decode_kv_indices(kv_indices: Any) -> Any:
    """Keep only this rank's owned KV indices; identity when DCP is off."""
    strategy = get_dcp_strategy()
    if strategy is None:
        return kv_indices
    return strategy.shard_decode_kv_indices(kv_indices)


def dcp_update_local_decode_kv_lens(kv_len_arr: Any) -> None:
    """In-place per-rank KV length; no-op when DCP is off."""
    strategy = get_dcp_strategy()
    if strategy is not None:
        strategy.update_local_decode_kv_lens(kv_len_arr)


def dcp_build_decode_metadata(**kwargs: Any) -> Any:
    """Build per-forward DCP decode metadata; None when DCP is off."""
    strategy = get_dcp_strategy()
    if strategy is None:
        return None
    return strategy.build_decode_metadata(**kwargs)


def dcp_plan_decode_metadata(**kwargs: Any) -> Any:
    """Plan/replay per-rank decode kv-len + index buffers; no-op when DCP is off."""
    strategy = get_dcp_strategy()
    if strategy is None:
        return None
    return strategy.plan_decode_metadata(**kwargs)


def dcp_gather_decode_query(q_nope_out: Any, q_pe: Any) -> Any:
    """All-gather sharded decode query heads (MLA); identity when DCP is off."""
    strategy = get_dcp_strategy()
    if strategy is None:
        return q_nope_out, q_pe
    return strategy.gather_decode_query(q_nope_out, q_pe)


def dcp_merge_attention(
    cp_attn_out: Any,
    cp_attn_lse: Any,
    cp_group: Any,
    *,
    backend: str,
    return_lse: bool = False,
    ctx: Optional[Any] = None,
) -> Any:
    """Merge per-rank partial attention via LSE rescale; identity when DCP is off.

    ``backend`` in {"mha", "mla"}.
    """
    strategy = get_dcp_strategy()
    if strategy is None:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out
    return strategy.merge_decode_attention(
        cp_attn_out,
        cp_attn_lse,
        cp_group,
        backend=backend,
        return_lse=return_lse,
        ctx=ctx,
    )
