from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.managers.schedule_policy import WAITING_QUEUE_PREFIX_MATCH_MAX
from sglang.srt.mem_cache.common import peek_uncached_tokens
from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder, SchedulePolicy
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)

# Load-sized chunks shrink only after this many sizing steps per stage of
# lower load.
_DIP_STEPS_PER_STAGE = 4


class PPChunkSizer:
    """Sizes pipeline-parallel prefill chunks from the pending uncached tokens
    (--pp-min-chunked-prefill-size, --pp-chunk-overhead-tokens)."""

    def __init__(
        self,
        min_chunk: int,
        overhead_tokens: Optional[int],
        pp_size: int,
        tree_cache: BasePrefixCache,
        policy: SchedulePolicy,
    ):
        self.min_chunk = min_chunk
        self.overhead_tokens = overhead_tokens
        self.pp_size = pp_size
        self.tree_cache = tree_cache
        self.policy = policy
        # Uncached tokens pending when this step's chunk was sized (load rule).
        self.pending_tokens = 0
        # Load rule: the size held through a dip, and the steps the dip has lasted.
        self.held_chunk = 0
        self.dip_steps = 0
        if tree_cache.peek_prefix_len(RadixKey(array("q"))) is None:
            logger.warning(
                "--pp-min-chunked-prefill-size: %s cannot probe waiting requests "
                "without changing the cache, so a step that needs a probe keeps "
                "the full chunk.",
                type(tree_cache).__name__,
            )

    def chunk_size(
        self, chunk: int, chunked_req: Optional[Req], waiting_queue: List[Req]
    ) -> int:
        """Use the small prefill chunk unless a pending request has at least a
        full chunk of uncached tokens: small chunks spread a batch with few new
        tokens over every stage, while every chunk of a long prefill rereads the
        KV cache of its whole prefix. A waiting request's hint is probed again
        only when it is missing, or long and the cache changed since, so a
        short request is probed about once while it waits. Every PP rank sees
        the same queue and tree, so all ranks pick the same size."""
        if self.overhead_tokens is not None:
            return self._chunk_size_for_load(chunk, chunked_req, waiting_queue)
        if (
            chunked_req is not None
            and chunked_req.seqlen - chunked_req.prefix_len >= chunk
        ):
            return chunk
        # The policy matched every waiting request this step: read those results.
        matched = self.policy.waiting_queue_prefix_matched(waiting_queue)
        for req in waiting_queue[:WAITING_QUEUE_PREFIX_MATCH_MAX]:
            uncached = self._uncached_tokens(req, matched, recheck_from=chunk)
            if uncached is None or uncached >= chunk:
                return chunk
        return min(self.min_chunk, chunk)

    def _chunk_size_for_load(
        self, full_chunk: int, chunked_req: Optional[Req], waiting_queue: List[Req]
    ) -> int:
        """Size the chunk from U, the pending uncached tokens: chunk C costs
        about overhead + C, so U takes (U / C + pp - 1) * (overhead + C). The
        pipeline runs at the pace of its largest chunk, so the chunked request
        counts the tokens it was admitted with, not just its rest."""
        cap = self._min_pending_for_full(full_chunk)
        pending = 0
        if chunked_req is not None:
            # The rest outgrows the admitted count when a host load-back fell
            # short at admission and the shortfall is prefilled again.
            pending = max(
                chunked_req.seqlen - chunked_req.prefix_len,
                chunked_req.pp_uncached_hint or 0,
            )
        matched = self.policy.waiting_queue_prefix_matched(waiting_queue)
        for req in waiting_queue[:WAITING_QUEUE_PREFIX_MATCH_MAX]:
            if pending >= cap:
                break
            # Every request adds to U, so any hint of the small chunk is rechecked.
            uncached = self._uncached_tokens(req, matched, recheck_from=self.min_chunk)
            if uncached is None:
                return full_chunk
            pending += uncached
        self.pending_tokens = pending
        return self._hold_chunk(
            self._chunk_for_pending(pending, full_chunk), full_chunk
        )

    def _uncached_tokens(
        self, req: Req, matched: bool, recheck_from: int
    ) -> Optional[int]:
        """req's uncached tokens from this step's match, else from its hint,
        probed again when missing, or at least recheck_from tokens and the cache
        version changed since. None when the cache cannot probe."""
        if matched:
            return self._record_uncached(req)
        uncached = req.pp_uncached_hint
        if uncached is None or uncached >= recheck_from:
            version = self.tree_cache.prefix_cache_version
            if version is None or version != req.pp_uncached_hint_version:
                uncached = peek_uncached_tokens(req, self.tree_cache)
                if uncached is not None:
                    req.pp_uncached_hint = uncached
                    req.pp_uncached_hint_version = version
        return uncached

    def _hold_chunk(self, chunk: int, full_chunk: int) -> int:
        """Grow at once, shrink only after a lasting dip: a steady load dips
        between one prefill's end and the next request, and mixed sizes run at
        the pace of the largest while paying every small one's overhead. Steps,
        unlike time, advance alike on every PP rank."""
        if chunk >= self.held_chunk:
            self.held_chunk, self.dip_steps = chunk, 0
        else:
            self.dip_steps += 1
            if self.dip_steps > _DIP_STEPS_PER_STAGE * self.pp_size:
                self.held_chunk, self.dip_steps = chunk, 0
        return min(self.held_chunk, full_chunk)

    def _chunk_for_pending(self, pending: int, full_chunk: int) -> int:
        """Double the small chunk, the last step only up to full_chunk, while
        the larger chunk gets pending through the pipeline faster."""
        chunk = self.min_chunk
        while chunk < full_chunk:
            bigger = min(2 * chunk, full_chunk)
            if pending < self._min_pending_to_grow(chunk, bigger):
                break
            chunk = bigger
        return min(chunk, full_chunk)

    def _min_pending_for_full(self, full_chunk: int) -> int:
        """Pending uncached tokens from which _chunk_for_pending reaches
        full_chunk: the threshold of its last step, 0 if it takes none."""
        chunk, pending = self.min_chunk, 0
        while chunk < full_chunk:
            bigger = min(2 * chunk, full_chunk)
            pending = self._min_pending_to_grow(chunk, bigger)
            chunk = bigger
        return pending

    def _min_pending_to_grow(self, chunk: int, bigger: int) -> int:
        """Pending uncached tokens from which bigger beats chunk:
        pending * overhead >= (pp - 1) * chunk * bigger, which is
        2 * (pp - 1) * chunk^2 for a doubling."""
        stages = self.pp_size - 1
        return -(-stages * chunk * bigger // self.overhead_tokens)

    @staticmethod
    def _record_uncached(req: Req) -> int:
        """Store req's latest match as its hint, unversioned: a match can count
        tokens the probe does not, so the next recheck probes."""
        req.pp_uncached_hint = max(1, req.seqlen - req.num_matched_prefix_tokens)
        req.pp_uncached_hint_version = None
        return req.pp_uncached_hint

    def resize_after_match(
        self, req: Req, adder: PrefillAdder, chunk: int, full_chunk: int
    ) -> int:
        """After admission matched req, refresh its hint and return the batch's
        chunk size, growing the adder's budget when a stale hint undercounted
        req (binary rule: it needs the full chunk; load rule: resized from the
        corrected load)."""
        hint = req.pp_uncached_hint
        uncached = self._record_uncached(req)
        if self.overhead_tokens is None:
            target = full_chunk if uncached >= full_chunk else chunk
        else:
            self.pending_tokens += max(0, uncached - (hint or 0))
            target = self._chunk_for_pending(self.pending_tokens, full_chunk)
            if target > chunk:
                self._hold_chunk(target, full_chunk)
        if target > chunk:
            adder.grow_chunk_budget(target - chunk)
            return target
        return chunk
