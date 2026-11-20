"""The interface every ranker implements, the prefix matching it asks
`SchedulePlanner` to run before ranking, and helpers both share."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Union

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


@dataclass(frozen=True)
class NoMatch:
    """Rank without reading prefix matches."""


@dataclass(frozen=True)
class FullMatch:
    """Match every waiting request, then deduplicate shared uncached prefixes:
    of the requests sharing one, the first by ``leader_key`` (or by queue order)
    runs and the rest are temporarily deprioritized."""

    leader_key: Optional[Callable[[Req], float]] = None


@dataclass(frozen=True)
class HeadWindowMatch:
    """Match a request only when first seen and rank the queue, then match and
    deduplicate the first ``window`` requests, rank them, and move the
    deprioritized ones to the end of the queue."""

    window: int
    leader_key: Callable[[Req], float]


MatchPlan = Union[NoMatch, FullMatch, HeadWindowMatch]

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


class BaseRanker:
    """Orders the waiting queue for admission.

    A ranker reads requests, their prefix matches, and the tree cache; it never
    matches prefixes or writes request or tree state (except gamma-Ada's
    first-seen stamp). `SchedulePlanner` does both,
    following ``match_plan``.
    """

    match_plan: MatchPlan = NoMatch()
    # Above this many waiting requests, rank as plain FCFS without matching.
    fcfs_fallback_queue_len: Optional[int] = None
    # Admission order follows uncached prefill work (``prefill_rank``):
    # `SchedulePlanner` polls storage prefetches before ranking and caps the
    # active prefill chunk to leave room for requests ranked ahead of it.
    ranks_by_prefill_work: bool = False

    def before_rank(self, waiting_queue: List[Req]) -> None:
        """Called at the start of each pass that ranks with this ranker, before
        any matching."""

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Sort ``waiting_queue`` in place, admitted first. ``temporary_deprioritized``
        holds the rids that share an uncached prefix with a request ahead."""
        raise NotImplementedError

    def prefill_rank(self, r: Req, work: int, processed_tokens: int) -> float:
        """The admission rank of ``r`` with ``work`` prefill tokens left; lower
        goes first."""
        raise NotImplementedError
