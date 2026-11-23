"""ROCm per-request sliding-window KV ring wiring for GQA DSpark drafts."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_rows_per_req_from_config,
)
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    parse_dflash_draft_config,
)

logger = logging.getLogger(__name__)


def rocm_dspark_gqa_ring_unsupported_reason(
    load_draft_hf_config: Callable[[], Any],
    *,
    disaggregation_mode: str,
    draft_window_size: Optional[int],
) -> Optional[str]:
    """Return why a ROCm Triton GQA DSpark ring cannot be enabled, or None."""
    if disaggregation_mode != "null":
        return "ROCm GQA DSpark draft KV ring does not support PD disaggregation"
    if draft_window_size is not None:
        return "--speculative-draft-window-size keeps the compact draft table"
    draft_hf_config = load_draft_hf_config()
    if parse_dflash_draft_config(draft_hf_config=draft_hf_config).is_mla:
        return "ROCm GQA DSpark ring does not serve MLA drafts"
    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        return f"not every draft layer is sliding_attention ({layer_types})"
    window_size = get_dflash_attention_sliding_window_size(draft_hf_config)
    if window_size is None or int(window_size) <= 0:
        return "the draft has no positive sliding_window"
    return None


def build_dspark_draft_kv_ring(
    *,
    draft_hf_config,
    draft_pool,
    req_to_token_rows: int,
    device,
    log_rank_zero: bool,
) -> DFlashDraftKVRing:
    """Create the per-request KV ring sized for a GQA or MLA DSpark draft."""
    is_mla = parse_dflash_draft_config(draft_hf_config=draft_hf_config).is_mla
    parallel = get_parallel()
    ring = DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req_from_config(draft_hf_config),
        dcp_size=parallel.attn_dcp_size if is_mla else 1,
        req_to_token_rows=req_to_token_rows,
        device=device,
        window_left=get_dflash_attention_sliding_window_size(draft_hf_config),
        dcp_rank=parallel.attn_dcp_rank if is_mla else 0,
        pool=draft_pool,
    )
    if log_rank_zero:
        logger.info(
            "DSPARK draft KV ring: %d rows per request (%d positions), "
            "%d req_to_token rows, draft pool %d rows.",
            ring.rows_per_req,
            ring.span,
            req_to_token_rows,
            draft_pool.size,
        )
    return ring


def bind_dspark_draft_kv_ring(
    *,
    ring: Optional[DFlashDraftKVRing],
    kv_injector,
    proposer,
) -> None:
    kv_injector.draft_kv_ring = ring
    proposer.draft_kv_ring = ring
