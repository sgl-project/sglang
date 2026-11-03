"""Decode time-slicing for first-token-starved waiters (--uniboost-preempt-wait-s).

Under KV-memory pressure the queue-wait tail is set by memory, not by order:
the waiting head needs device KV that long-running decodes hold, and output
lengths are heavy-tailed, so the request that has decoded the most is expected
to hold its KV the longest. When a request still waiting for its first token
has queued for wait_s and the pool cannot fit it, the scheduler releases the
longest-attained decodes into the radix tree (the finished-request KV
lifecycle, so each resumes from its own cached KV) and puts the waiter at the
head of the next admission pass.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

import msgspec

from sglang.srt.layers.cp.base import is_cp_sharded_padded
from sglang.srt.layers.cp.utils import from_cp_sharded
from sglang.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS,
    running_request_token_reserve,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)

# Arbitrary: one plan per second keeps a released decode's KV from being
# counted twice before the next admission pass has consumed it.
PREEMPT_COOLDOWN_S = 1.0
# Arbitrary bound on the waiter's head-of-queue slot (passes are tens of ms)
# when the released memory still does not fit it.
RESERVE_MAX_PASSES = 256


class DecodePreemptionPlan(msgspec.Struct, frozen=True, kw_only=True):
    target_rid: str
    target_wait_s: float
    need_tokens: int
    deficit_tokens: int
    # Indices into the running batch's reqs, in release order.
    victim_indices: tuple[int, ...]
    victim_output_lens: tuple[int, ...]
    freed_tokens: int


def available_and_evictable_tokens(
    *, allocator: BaseTokenToKVPoolAllocator, tree_cache: BasePrefixCache
) -> int:
    """PrefillBudget's free pool for a full-attention tree: free plus evictable
    KV tokens, the allocator side in global (CP-unsharded) tokens."""
    available = allocator.available_size()
    if is_cp_sharded_padded():
        available = from_cp_sharded(available)
    return available + tree_cache.evictable_size()


def admission_need_tokens(req: Req, page_size: int) -> int:
    """Tokens PrefillAdder.add_one_req charges a waiting request: everything
    past its device prefix (host hits load into fresh device slots), the
    clipped decode reservation and one page of alignment."""
    max_new = min(
        max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
        CLIP_MAX_NEW_TOKENS,
    )
    seq_len = len(req.origin_input_ids) + len(req.output_ids)
    return seq_len - req.prefix_len + max_new + page_size


def is_preemptible(req: Req, *, quantum_tokens: int, max_per_req: int) -> bool:
    # retract_decode aborts beam groups instead of retracting them, and an
    # input_embeds request that has streamed output cannot re-prefill.
    if req.finished() or req.beam_group is not None or not req.resumable_by_reprefill:
        return False
    if req.num_preemptions >= max_per_req:
        return False
    # The quantum doubles per release, so every request keeps progressing.
    decoded = len(req.output_ids) - req.preempt_resume_output_len
    return decoded >= quantum_tokens << req.num_preemptions


def _starving_target(
    waiting_queue: Iterable[Req],
    *,
    now_s: float,
    wait_s: float,
    preferred_rid: Optional[str],
) -> Optional[Req]:
    def starving(req: Req) -> bool:
        # Generated output means the first token was delivered (a released or
        # retracted decode); only first-token waiters trigger a release.
        return (
            not req.output_ids
            and req.uniboost_arrival_s is not None
            and now_s - req.uniboost_arrival_s >= wait_s
        )

    queue = list(waiting_queue)
    if preferred_rid is not None:
        for req in queue:
            if req.rid == preferred_rid and starving(req):
                return req
    for req in queue:
        if starving(req):
            return req
    return None


def plan_decode_preemption(
    *,
    now_s: float,
    waiting_queue: Iterable[Req],
    running_reqs: Sequence[Req],
    have_tokens: int,
    new_token_ratio: float,
    page_size: int,
    wait_s: float,
    quantum_tokens: int,
    max_per_req: int,
    preferred_rid: Optional[str] = None,
) -> Optional[DecodePreemptionPlan]:
    """The fewest longest-attained decodes whose release lets the first
    starving waiter fit, or None when it already fits (its wait has another
    cause) or no eligible set covers the deficit. have_tokens is the pool's
    available plus evictable KV tokens."""
    target = _starving_target(
        waiting_queue, now_s=now_s, wait_s=wait_s, preferred_rid=preferred_rid
    )
    if target is None:
        return None
    reserves = [running_request_token_reserve(r, new_token_ratio) for r in running_reqs]
    need = admission_need_tokens(target, page_size)
    deficit = need - (have_tokens - sum(reserves))
    if deficit <= 0:
        return None

    # Least attained service first: the most decoded output; heavy-tailed
    # outputs make it the longest expected remaining hold.
    candidates = sorted(
        (
            i
            for i, r in enumerate(running_reqs)
            if is_preemptible(r, quantum_tokens=quantum_tokens, max_per_req=max_per_req)
        ),
        key=lambda i: (
            -len(running_reqs[i].output_ids),
            -len(running_reqs[i].origin_input_ids),
        ),
    )
    chosen: List[int] = []
    freed = 0.0
    for i in candidates:
        req = running_reqs[i]
        chosen.append(i)
        freed += len(req.origin_input_ids) + len(req.output_ids) + reserves[i]
        if freed >= deficit:
            return DecodePreemptionPlan(
                target_rid=target.rid,
                target_wait_s=now_s - target.uniboost_arrival_s,
                need_tokens=need,
                deficit_tokens=int(math.ceil(deficit)),
                victim_indices=tuple(chosen),
                victim_output_lens=tuple(
                    len(running_reqs[j].output_ids) for j in chosen
                ),
                freed_tokens=int(freed),
            )
    return None


class DecodePreemptor:
    """Per-scheduler state around plan_decode_preemption: the cooldown, the
    waiter's head-of-queue reservation and each request's release count and
    decode progress since its latest admission."""

    def __init__(
        self, *, wait_s: float, quantum_tokens: int, max_per_req: int, dp_rank: int
    ) -> None:
        self.wait_s = wait_s
        self.quantum_tokens = quantum_tokens
        self.max_per_req = max_per_req
        self.dp_rank = dp_rank
        self._last_plan_s = -math.inf
        self._target_rid: Optional[str] = None
        self._target_passes_left = 0
        self.num_events = 0
        self.num_victims = 0

    @classmethod
    def maybe_create(
        cls,
        *,
        wait_s: float,
        quantum_tokens: int,
        max_per_req: int,
        dp_rank: int,
        tree_supports_mamba: bool,
        tree_supports_swa: bool,
    ) -> Optional[DecodePreemptor]:
        if wait_s <= 0:
            return None
        # Releasing an unfinished request into the tree is a plain checkpoint
        # only for full attention; SWA and mamba keep per-request state outside it.
        if tree_supports_mamba or tree_supports_swa:
            raise ValueError(
                "--uniboost-preempt-wait-s requires a full-attention radix cache "
                "(no SWA or mamba components)"
            )
        return cls(
            wait_s=wait_s,
            quantum_tokens=quantum_tokens,
            max_per_req=max_per_req,
            dp_rank=dp_rank,
        )

    def maybe_plan(
        self,
        *,
        now_s: float,
        waiting_queue: Iterable[Req],
        running_reqs: Sequence[Req],
        have_tokens: int,
        new_token_ratio: float,
        page_size: int,
    ) -> Optional[DecodePreemptionPlan]:
        if now_s - self._last_plan_s < PREEMPT_COOLDOWN_S:
            return None
        return plan_decode_preemption(
            now_s=now_s,
            waiting_queue=waiting_queue,
            running_reqs=running_reqs,
            have_tokens=have_tokens,
            new_token_ratio=new_token_ratio,
            page_size=page_size,
            wait_s=self.wait_s,
            quantum_tokens=self.quantum_tokens,
            max_per_req=self.max_per_req,
            preferred_rid=self._target_rid,
        )

    def on_preempted(
        self, plan: DecodePreemptionPlan, preempted: Sequence[Req], now_s: float
    ) -> None:
        self._last_plan_s = now_s
        self._target_rid = plan.target_rid
        self._target_passes_left = RESERVE_MAX_PASSES
        for req in preempted:
            req.num_preemptions += 1
        self.num_events += 1
        self.num_victims += len(preempted)
        logger.info(
            "decode preemption [dp %d]: waiter %s queued %.1f s needs %d tok "
            "(deficit %d); released %d decode(s) with output %s tok, ~%d tok "
            "freed; totals events=%d victims=%d",
            self.dp_rank,
            plan.target_rid,
            plan.target_wait_s,
            plan.need_tokens,
            plan.deficit_tokens,
            len(preempted),
            list(plan.victim_output_lens),
            plan.freed_tokens,
            self.num_events,
            self.num_victims,
        )

    def promote_target(self, waiting_queue: Iterable[Req]) -> List[Req]:
        """The sorted queue with the reserved waiter moved to its head, so the
        released memory goes to it and the admission scan stops at it instead
        of letting later requests take that memory."""
        order = list(waiting_queue)
        if self._target_rid is None:
            return order
        idx = next((i for i, r in enumerate(order) if r.rid == self._target_rid), None)
        self._target_passes_left -= 1
        if idx is None or self._target_passes_left < 0:
            self._target_rid = None
            return order
        return [order[idx], *order[:idx], *order[idx + 1 :]]

    def on_admitted(self, reqs: Sequence[Req]) -> None:
        for req in reqs:
            req.preempt_resume_output_len = len(req.output_ids)
            if req.rid == self._target_rid:
                self._target_rid = None
