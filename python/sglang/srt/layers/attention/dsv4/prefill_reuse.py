"""Query-row top-K reuse for the DeepSeek-V4 C4 indexer on the prefill fast path.

Adjacent prefill query rows select heavily overlapping top-K sets, so the
non-paged fast path scores one leader row per window of rows and broadcasts its
selection to the rest of the window. The leader is the first row of its window,
so no row reuses a selection that saw later tokens.

Presets are layer-tiered: the window is chosen per C4 indexer layer, and
``deep10-inf-w4`` lets the deepest layers share one selection per chunk while
the rest keep a window of 4. The same query-group sharing on DSA indexers is
PIVOT-Reuse (arXiv:2607.24593).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4 import (
    plan_topk_v2,
    topk_transform_paged,
    topk_transform_paged_v2,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
    from sglang.srt.layers.attention.dsv4.metadata import NonPagedIndexerPlan
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

logger = logging.getLogger(__name__)

# DeepSeek-V4-Flash has C4 indexers on the even layers 2..42; these are the ten deepest.
_DEEP10_LAYERS = (24, 26, 28, 30, 32, 34, 36, 38, 40, 42)


class ReusePreset(msgspec.Struct, frozen=True, kw_only=True):
    window: int
    # These layers share one selection per chunk instead of using `window`.
    deep_layers: tuple[int, ...] = ()

    def window_for(self, *, layer_id: int, query_rows: int) -> int:
        return query_rows if layer_id in self.deep_layers else self.window


PRESETS: dict[str, ReusePreset] = {
    "uniform-w4": ReusePreset(window=4),
    "deep10-inf-w4": ReusePreset(window=4, deep_layers=_DEEP10_LAYERS),
}


def resolve_prefill_reuse_preset(name: Optional[str]) -> Optional[ReusePreset]:
    if name is None:
        return None
    preset = PRESETS[name]
    logger.info("DeepSeek V4 prefill top-K reuse enabled: %s = %s", name, preset)
    return preset


# The fast path fails closed silently, so these counters are the only evidence
# that reuse ran; rows_saved / rows_total should match the preset.
_dose = {"calls": 0, "rows_total": 0, "rows_saved": 0}


def _record_dose(*, query_rows: int, num_leaders: int) -> None:
    _dose["calls"] += 1
    _dose["rows_total"] += query_rows
    _dose["rows_saved"] += query_rows - num_leaders
    interval = envs.SGLANG_LOG_DSV4_PREFILL_REUSE_INTERVAL.get()
    if interval > 0 and _dose["calls"] % interval == 0:
        logger.info(
            "[dsv4-prefill-reuse] pid=%d calls=%d rows_total=%d rows_saved=%d "
            "frac=%.4f",
            os.getpid(),
            _dose["calls"],
            _dose["rows_total"],
            _dose["rows_saved"],
            _dose["rows_saved"] / _dose["rows_total"],
        )


def maybe_apply_reuse(
    *,
    preset: ReusePreset,
    topk_backend: DSATopKBackend,
    forward_nonpaged_indexer: Callable[..., torch.Tensor],
    c4_indexer: C4Indexer,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    plan: NonPagedIndexerPlan,
    q_indexer: torch.Tensor,
    weights: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    c4_sparse_page_indices: torch.Tensor,
    raw_indices: Optional[torch.Tensor],
    compressed_page_size: int,
) -> bool:
    """Score only the window leaders and broadcast their top-K to every row.

    Returns True once the page (and raw) indices of all rows are written, so the
    caller must skip its own scoring and top-K; False means nothing was touched.
    """
    # Only the sgl-kernel transforms are mirrored below.
    if topk_backend.is_torch() or topk_backend.is_flashinfer():
        return False
    query_rows = q_indexer.shape[0]
    window = preset.window_for(layer_id=c4_indexer.layer_id, query_rows=query_rows)
    num_leaders = -(-query_rows // window)
    if num_leaders >= query_rows:
        return False

    device = c4_sparse_page_indices.device
    leader_rows = torch.arange(0, query_rows, window, device=device)
    row_to_leader = torch.arange(query_rows, device=device) // window

    # Rebuilding the plan would fail its row-count checks, so derive it; only ke is per row.
    leader_ke = plan.ke.index_select(0, leader_rows)
    leader_plan = dataclasses.replace(
        plan, ks=torch.zeros_like(leader_ke), ke=leader_ke, query_rows=num_leaders
    )
    logits = forward_nonpaged_indexer(
        q_indexer=q_indexer.index_select(0, leader_rows),
        weights=weights.index_select(0, leader_rows),
        c4_indexer=c4_indexer,
        token_to_kv_pool=token_to_kv_pool,
        plan=leader_plan,
    )

    leader_seq_lens = c4_seq_lens.index_select(0, leader_rows)
    leader_page_table = page_table.index_select(0, leader_rows)
    # -1 like the production buffer: the transform leaves unfilled slots alone,
    # and the broadcast below copies whole rows.
    leader_pages = torch.full(
        (num_leaders, c4_sparse_page_indices.shape[1]),
        -1,
        dtype=c4_sparse_page_indices.dtype,
        device=device,
    )
    if topk_backend.should_use_topk_v2() and raw_indices is None:
        # The cached top-K plan covers all rows, so the leaders need their own.
        topk_transform_paged_v2(
            logits,
            leader_seq_lens,
            leader_page_table,
            leader_pages,
            compressed_page_size,
            plan_topk_v2(leader_seq_lens),
        )
        leader_raw = None
    else:
        leader_raw = None if raw_indices is None else torch.empty_like(leader_pages)
        topk_transform_paged(
            logits,
            leader_seq_lens,
            leader_page_table,
            leader_pages,
            compressed_page_size,
            leader_raw,
        )

    # Followers longer than their leader read its -1 padding as part of their top-K;
    # flash_mla_sparse_fwd skips -1, and compressed_base is 0 on this single-request path.
    c4_sparse_page_indices.copy_(leader_pages.index_select(0, row_to_leader))
    # Raw indices are sequence positions, not row-relative, so they broadcast as-is.
    if leader_raw is not None:
        raw_indices[:query_rows].copy_(leader_raw.index_select(0, row_to_leader))
    _record_dose(query_rows=query_rows, num_leaders=num_leaders)
    return True
