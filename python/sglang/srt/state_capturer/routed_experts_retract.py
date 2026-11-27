"""Keep a retracted request's routed experts and force its re-prefill to them.

The capturer's host buffer is indexed by KV slot. Retraction releases the request's
slots and the resumed request re-prefills into new ones with freshly selected
experts, so the response would carry rows its earlier tokens were not sampled with.
The rows are copied onto the request just before the release; the re-prefill's
gates take them (routed_experts_forcing.py), so the rebuilt KV and the returned
rows are both the ones the tokens were sampled with.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_resources
from sglang.srt.state_capturer.routed_experts_forcing import (
    ForcingBatch,
    build_forcing_batch,
)
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return envs.SGLANG_ENABLE_ROUTED_EXPERTS_RETRACT_FORCING.get()


def completed_len(req: Req) -> int:
    """Positions whose forward result has been processed, so whose row is written.

    The forward at position p yields token p + 1 and output_ids only grows when a
    result is processed; kv_committed_len advances at allocation, ahead of that,
    but bounds a prefill that has run only some of its chunks.
    """
    return max(
        0,
        min(
            req.kv.kv_committed_len,
            len(req.origin_input_ids) + len(req.output_ids) - 1,
        ),
    )


def _captures(req: Req) -> bool:
    capturer = get_resources().experts_capturer
    return (
        req.return_routed_experts and capturer is not None and capturer.has_host_cache
    )


def keep_rows_before_release(*, req: Req, req_to_token_pool: ReqToTokenPool) -> None:
    """Copy the request's rows off its KV slots; call before the slots are released."""
    if not enabled() or not _captures(req) or not req.kv.holds_kv:
        return
    if req.input_embeds is not None:
        # This retraction discards the generated tokens and restarts generation.
        req.routed_experts_kept = None
        return
    kept = req.routed_experts_kept
    have = 0 if kept is None else kept.shape[0]
    done = completed_len(req)
    # Append only: rows below `have` were re-prefilled since the last retraction, on
    # slots the cache may have rebound to another request's KV.
    if done > have:
        slots = req_to_token_pool.req_to_token[req.kv.req_pool_idx, have:done].cpu()
        new = get_resources().experts_capturer.host_cache.buffer[slots].clone()
        req.routed_experts_kept = new if kept is None else torch.cat([kept, new])
    # Rows below len(origin_input_ids) - 1 sample no token: a prefill that has
    # not reached them yet has lost nothing.
    known = len(req.origin_input_ids) + len(req.output_ids) - 1
    unwritten = known - max(done, have, len(req.origin_input_ids) - 1)
    if unwritten > 0:
        logger.warning(
            "Request %s was retracted with %d sampled tokens whose routed experts "
            "were not captured yet; the re-prefill selects them again.",
            req.rid,
            unwritten,
        )


def build_extend_forcing(
    *, reqs: Sequence[Req], prefix_lens: Sequence[int], extend_lens: Sequence[int]
) -> Optional[ForcingBatch]:
    """The kept rows an extend batch recomputes, or None if it recomputes none."""
    if not enabled():
        return None
    return build_forcing_batch(
        kept_rows=[req.routed_experts_kept for req in reqs],
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        pin_memory=is_pin_memory_available(),
    )


def _rows_differing(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    differing = (a != b).flatten(1).any(dim=1)
    rows = differing.nonzero(as_tuple=True)[0]
    if rows.numel():
        # A gate keeps its own column order for a row it already agrees with.
        differing[rows] = (
            (a[rows].sort(dim=-1).values != b[rows].sort(dim=-1).values)
            .flatten(1)
            .any(dim=1)
        )
    return differing


def check_forced_extend(
    *,
    reqs: Sequence[Req],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    captured: torch.Tensor,
) -> None:
    """Compare a forced extend's captured rows with the kept rows it was forced to.

    captured is the forward's own [tokens, layers, top_k] capture in forward row
    order: the cache may already have rebound a chunked request's slots to
    another request's KV, so the slot table cannot locate these rows.
    """
    offset = 0
    for req, prefix_len, extend_len in zip(reqs, prefix_lens, extend_lens, strict=True):
        kept = req.routed_experts_kept
        num_forced = (
            0
            if kept is None
            else min(prefix_len + extend_len, kept.shape[0]) - prefix_len
        )
        if num_forced > 0:
            mismatched = int(
                _rows_differing(
                    captured[offset : offset + num_forced],
                    kept[prefix_len : prefix_len + num_forced],
                ).sum()
            )
            if mismatched:
                logger.error(
                    "routed-experts forcing: %d of the %d rows of request %s forced "
                    "from position %d were captured with other experts",
                    mismatched,
                    num_forced,
                    req.rid,
                    prefix_len,
                )
        offset += extend_len


def overlay_kept_rows(
    *, req: Req, rows: Optional[torch.Tensor], start_len: int
) -> Optional[torch.Tensor]:
    """Put the kept rows back over rows [start_len, start_len + len(rows)).

    The slot table no longer finds all of them: the cache rebinds a request's slots
    to an identical prefix another request inserted first.
    """
    kept = req.routed_experts_kept
    if kept is None or rows is None:
        return rows
    n = min(kept.shape[0] - start_len, rows.shape[0])
    if n > 0:
        rows[:n] = kept[start_len : start_len + n]
    return rows
