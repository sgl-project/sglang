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

"""Next-DSA-layer prefix AllGather into the shared ``dcp_kv_buffer``.

Kick sits on a DSA layer after the current attention has released the shared
buffer. The AG runs on a side CUDA stream; the next DSA layer's
``all_gather_kv_cache_for_mla_extend`` waits / consumes (skip sync prefix AG,
append local K only) or falls back to the synchronous path.

Env
---
``SGLANG_ENABLE_DCP_KV_BUFFER_PREFETCH=1``
    Master switch (default off).

``SGLANG_DCP_KV_BUFFER_PREFETCH_KICK=after_ffn|after_attn``
    Where a DSA layer kicks the *next* DSA prefix AG (default ``after_ffn``).

    - ``after_ffn``: after ``ffn_boundary.finish`` (MegaMoE + MHC post done);
      intended cover = following ~3 KDA layers. Prefer this if MegaMoE A2A
      contends with DCP AG.
    - ``after_attn``: after ``attn_boundary.finish``, before MLP; cover includes
      this layer's MegaMoE + following KDA. Use for A/B under long prefixes.

``SGLANG_DCP_KV_BUFFER_PREFETCH_FIRST=1``
    Also kick the *first* DSA layer at model-forward entry (after embed /
    PP handoff). That layer has no previous DSA kicker; without this its
    prefix AG stays fully exposed. Default on when the master switch is on.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch

from sglang.srt.layers.dcp.comm import all_gather_kv_cache_for_dcp
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_ENABLE = get_bool_env_var("SGLANG_ENABLE_DCP_KV_BUFFER_PREFETCH", "false")
_DEBUG = get_bool_env_var("SGLANG_DCP_KV_BUFFER_PREFETCH_DEBUG", "false")
_KICK_RAW = (
    os.environ.get("SGLANG_DCP_KV_BUFFER_PREFETCH_KICK", "after_ffn").strip().lower()
)
_KICK_POINT: Literal["after_attn", "after_ffn"] = (
    "after_attn" if _KICK_RAW == "after_attn" else "after_ffn"
)
# First-DSA kick defaults on with the master switch; set =0 to isolate A/B of
# mid-stack kick points from the independent first-layer ledger.
_FIRST_LAYER_KICK = get_bool_env_var(
    "SGLANG_DCP_KV_BUFFER_PREFETCH_FIRST", "true" if _ENABLE else "false"
)

KickPoint = Literal["after_attn", "after_ffn"]


@dataclass
class _PendingPrefetch:
    layer_id: int
    event: torch.cuda.Event
    prefix_sum: int


_stream: Optional[torch.cuda.Stream] = None
_pending: Optional[_PendingPrefetch] = None


def dcp_kv_buffer_prefetch_enabled() -> bool:
    return _ENABLE


def get_dcp_kv_buffer_prefetch_kick_point() -> KickPoint:
    return _KICK_POINT


def dcp_kv_buffer_prefetch_first_layer_enabled() -> bool:
    return _ENABLE and _FIRST_LAYER_KICK


def should_kick_dcp_kv_buffer_prefetch_at(point: KickPoint) -> bool:
    return _ENABLE and _KICK_POINT == point


def _get_stream() -> torch.cuda.Stream:
    global _stream
    if _stream is None:
        _stream = torch.cuda.Stream()
    return _stream


def cancel_pending_dcp_kv_buffer_prefetch() -> None:
    """Drain any in-flight prefetch so the shared buffer can be rewritten."""
    global _pending
    if _pending is None:
        return
    torch.cuda.current_stream().wait_event(_pending.event)
    _pending = None


def kick_dcp_kv_buffer_prefix_prefetch(
    *,
    token_to_kv_pool,
    next_attn_mqa,
    extend_prefix_lens_cpu: List[int],
    dcp_local_prefix_kv_indices: torch.Tensor,
    dcp_extend_prefix_lens_sum: int,
    dcp_kv_buffer: torch.Tensor,
) -> bool:
    """Start async prefix AG for ``next_attn_mqa.layer_id`` into ``dcp_kv_buffer``.

    Returns True if a prefetch was kicked or is already pending for that layer.
    Prefetch writes only the gathered prefix rows; local extend K is appended
    later by ``all_gather_kv_cache_for_mla_extend``.
    """
    if not _ENABLE:
        return False
    if not get_parallel().dcp_enabled:
        return False
    if dcp_kv_buffer is None or dcp_local_prefix_kv_indices is None:
        return False
    if dcp_extend_prefix_lens_sum is None:
        return False
    if extend_prefix_lens_cpu is None:
        return False

    layer_id = int(next_attn_mqa.layer_id)
    global _pending
    if _pending is not None:
        if _pending.layer_id == layer_id:
            return True
        # Shared buffer: finish the previous prefetch before reuse.
        cancel_pending_dcp_kv_buffer_prefetch()

    prefix_tokens = sum(int(x) for x in extend_prefix_lens_cpu)

    stream = _get_stream()
    stream.wait_stream(torch.cuda.current_stream())
    event = torch.cuda.Event()

    with torch.cuda.stream(stream):
        if prefix_tokens == 0:
            pass
        else:
            cache_k_nope, cache_k_rope = token_to_kv_pool.layer_router.read(
                layer_id
            ).read_mla(
                next_attn_mqa,
                dcp_local_prefix_kv_indices,
            )
            extend_prefix_lens = torch.tensor(extend_prefix_lens_cpu, dtype=torch.int32)
            gathered_kv = all_gather_kv_cache_for_dcp(
                cache_k_nope,
                cache_k_rope,
                extend_prefix_lens,
                prefix_starts_cpu=torch.zeros_like(extend_prefix_lens),
            )
            if gathered_kv.shape[0] != dcp_extend_prefix_lens_sum:
                raise RuntimeError(
                    "DCP prefetch gathered prefix length mismatch: "
                    f"gathered={gathered_kv.shape[0]}, "
                    f"expected={dcp_extend_prefix_lens_sum}."
                )
            dcp_kv_buffer[:dcp_extend_prefix_lens_sum] = gathered_kv
        event.record(stream)

    _pending = _PendingPrefetch(
        layer_id=layer_id,
        event=event,
        prefix_sum=int(dcp_extend_prefix_lens_sum),
    )
    if _DEBUG:
        logger.info(
            "dcp_kv_buffer_prefetch kick layer=%s prefix_sum=%s",
            layer_id,
            dcp_extend_prefix_lens_sum,
        )
    return True


def try_consume_dcp_kv_buffer_prefix_prefetch(layer_id: int) -> bool:
    """Wait for a pending prefetch targeting ``layer_id``.

    Returns True iff the prefix region of ``dcp_kv_buffer`` is ready so the
    caller may skip the sync prefix AllGather and only append local K.
    A pending prefetch for a *different* layer is drained (buffer must not be
    half-owned) and the caller falls back to the sync path.
    """
    if not _ENABLE:
        return False

    global _pending
    if _pending is None:
        return False
    if _pending.layer_id != int(layer_id):
        cancel_pending_dcp_kv_buffer_prefetch()
        return False

    torch.cuda.current_stream().wait_event(_pending.event)
    if _DEBUG:
        logger.info(
            "dcp_kv_buffer_prefetch consume layer=%s prefix_sum=%s",
            layer_id,
            _pending.prefix_sum,
        )
    _pending = None
    return True


def maybe_kick_next_dsa_dcp_kv_buffer_prefetch(
    *,
    forward_batch,
    next_attn_mqa,
    when: KickPoint,
    token_to_kv_pool=None,
) -> bool:
    """Model-side helper: kick only when env kick-point + DCP metadata allow it."""
    if not should_kick_dcp_kv_buffer_prefetch_at(when):
        return False
    return _maybe_kick_impl(
        forward_batch=forward_batch,
        next_attn_mqa=next_attn_mqa,
        token_to_kv_pool=token_to_kv_pool,
    )


def maybe_kick_first_dsa_dcp_kv_buffer_prefetch(
    *,
    forward_batch,
    first_attn_mqa,
    token_to_kv_pool=None,
) -> bool:
    """Kick the first DSA layer at forward entry (embed / PP handoff).

    Independent of ``SGLANG_DCP_KV_BUFFER_PREFETCH_KICK``: mid-stack A/B does
    not cover the first DSA. Best cover is early layers before the first DSA.
    """
    if not dcp_kv_buffer_prefetch_first_layer_enabled():
        return False
    return _maybe_kick_impl(
        forward_batch=forward_batch,
        next_attn_mqa=first_attn_mqa,
        token_to_kv_pool=token_to_kv_pool,
    )


def _maybe_kick_impl(
    *,
    forward_batch,
    next_attn_mqa,
    token_to_kv_pool=None,
) -> bool:
    if not _ENABLE or next_attn_mqa is None:
        return False
    if forward_batch is None or forward_batch.attn_dcp_metadata is None:
        return False
    if not forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
        return False

    meta = forward_batch.attn_dcp_metadata
    if meta.dcp_kv_buffer is None or meta.dcp_local_prefix_kv_indices is None:
        return False
    if forward_batch.extend_prefix_lens_cpu is None:
        return False

    if token_to_kv_pool is None:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        token_to_kv_pool = get_token_to_kv_pool()

    return kick_dcp_kv_buffer_prefix_prefetch(
        token_to_kv_pool=token_to_kv_pool,
        next_attn_mqa=next_attn_mqa,
        extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
        dcp_local_prefix_kv_indices=meta.dcp_local_prefix_kv_indices,
        dcp_extend_prefix_lens_sum=meta.dcp_extend_prefix_lens_sum,
        dcp_kv_buffer=meta.dcp_kv_buffer,
    )
