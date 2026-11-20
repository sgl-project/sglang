"""Helpers shared by the rankers and `SchedulePlanner`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

# Arbitrary; bounds per-round prefix work on the waiting queue.
WAITING_QUEUE_PREFIX_MATCH_MAX = 128


def prefill_work(tree_cache: BasePrefixCache, r: Req) -> int:
    """The prompt and output tokens of ``r`` not yet cached on device or staged
    from storage, at least 1."""
    cached = max(
        r.num_matched_prefix_tokens,
        tree_cache.staged_storage_prefix_end(r.cache_request_handle),
    )
    return max(1, len(r.origin_input_ids) + len(r.output_ids) - cached)
