from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Set

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policies.base import BaseRanker, FullMatch, NoMatch
from sglang.srt.managers.schedule_policies.common import (
    WAITING_QUEUE_PREFIX_MATCH_MAX,
    prefill_work,
)
from sglang.srt.managers.schedule_policies.dfs_weight import DfsWeightRanker
from sglang.srt.managers.schedule_policies.fcfs import FcfsRanker, PriorityFcfsRanker
from sglang.srt.managers.schedule_policies.hrrn import HrrnRanker
from sglang.srt.managers.schedule_policies.lof import LofRanker
from sglang.srt.managers.schedule_policies.lpm import LpmRanker
from sglang.srt.managers.schedule_policies.policy import SchedulePolicy
from sglang.srt.managers.schedule_policies.random_order import RandomRanker
from sglang.srt.managers.schedule_policies.routing_key import RoutingKeyRanker
from sglang.srt.managers.schedule_policies.shortest_prefill import (
    ShortestPrefillFirstRanker,
)
from sglang.srt.managers.schedule_policies.uniboost import UniBoostRanker
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InsertParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.common import match_kv_cache, touch_waiting_prefix
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import get_disagg, get_memory
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_policies.uniboost_gamma_ada import (
        GammaAdaController,
    )

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)

# Bound the walk cost using scheduling passes, which agree across TP ranks.
WAITING_PREFIX_REFRESH_INTERVAL = 32


class SchedulePlanner:
    """Orders the waiting queue for admission under the configured policy."""

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
        shortest_prefill_aging_factor: float = 0.0,
        bounded_lpm_enabled: bool = True,
        *,
        uniboost_gamma: float = 10.0,
        enable_uniboost_gamma_ada: bool = False,
        uniboost_rank_invariant: bool = False,
        uniboost_step_clock: Optional[Callable[[], int]] = None,
    ):
        self.on_bounded_lpm_state_change: Optional[Callable[[bool], None]] = None
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        if not math.isfinite(shortest_prefill_aging_factor) or (
            shortest_prefill_aging_factor < 0
        ):
            raise ValueError(
                f"shortest_prefill_aging_factor must be finite and >= 0, got {shortest_prefill_aging_factor}"
            )
        if (
            shortest_prefill_aging_factor
            and self.policy != SchedulePolicy.SHORTEST_PREFILL_FIRST
        ):
            raise ValueError(
                "shortest_prefill_aging_factor requires schedule_policy=shortest-prefill-first"
            )
        self.tree_cache = tree_cache
        priority_sign = 1 if schedule_low_priority_values_first else -1
        if self.policy == SchedulePolicy.FCFS:
            self.ranker = (
                PriorityFcfsRanker(priority_sign)
                if enable_priority_scheduling
                else FcfsRanker()
            )
        elif self.policy == SchedulePolicy.LOF:
            self.ranker = LofRanker(enable_priority_scheduling, priority_sign)
        elif self.policy == SchedulePolicy.RANDOM:
            self.ranker = RandomRanker()
        elif self.policy == SchedulePolicy.ROUTING_KEY:
            self.ranker = RoutingKeyRanker()
        elif self.policy == SchedulePolicy.LPM:
            self.ranker = LpmRanker(
                tree_cache, enable_hierarchical_cache, bounded_lpm_enabled
            )
        elif self.policy == SchedulePolicy.DFS_WEIGHT:
            self.ranker = DfsWeightRanker(tree_cache)
        elif self.policy == SchedulePolicy.HRRN:
            self.ranker = HrrnRanker()
        elif self.policy == SchedulePolicy.SHORTEST_PREFILL_FIRST:
            self.ranker = ShortestPrefillFirstRanker(
                tree_cache, shortest_prefill_aging_factor
            )
        elif self.policy == SchedulePolicy.UNIBOOST:
            self.ranker = UniBoostRanker(
                tree_cache,
                uniboost_gamma,
                enable_uniboost_gamma_ada,
                uniboost_rank_invariant,
                uniboost_step_clock,
            )
        if enable_uniboost_gamma_ada and self.policy != SchedulePolicy.UNIBOOST:
            raise ValueError(
                "enable_uniboost_gamma_ada requires schedule_policy=uniboost"
            )
        self._lpm = self.ranker if isinstance(self.ranker, LpmRanker) else None
        self.uniboost_gamma_ada: Optional[GammaAdaController] = (
            self.ranker.uniboost_gamma_ada
            if isinstance(self.ranker, UniBoostRanker)
            else None
        )
        # What a ranker ranks as above its fcfs_fallback_queue_len.
        self._fcfs_fallback = FcfsRanker()
        self._waiting_prefix_refresh_calls = 0

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache.create_simulated()

    def calc_priority(
        self,
        waiting: Iterable[Req],
        running_reqs: Optional[List[Req]] = None,
        processed_tokens: int = 0,
        bootstrap_queue: Iterable[Req] = (),
    ) -> List[Req]:
        """The waiting requests in the order the policy admits them."""
        waiting_queue = list(waiting)
        if self.bounded_lpm_engaged(len(waiting_queue)):
            if (
                self._lpm.refresh_bounded_members(waiting_queue)
                and self.on_bounded_lpm_state_change is not None
            ):
                self.on_bounded_lpm_state_change(self._lpm.bounded_active)

        if self.bounded_lpm_active:
            members, tail = self._lpm.split_waiting(waiting_queue)
            # Keep the existing FCFS refresh cadence/cap for non-members.
            # Touch them first so matching the current batch makes its prefixes
            # more recent. Fast-match caches retain the existing match-anyway path.
            if (
                self.tree_cache.supports_fast_match_prefix()
                and get_disagg().disaggregation_mode != "decode"
            ):
                for r in tail:
                    r.init_next_round_input(self.tree_cache)
            self._touch_waiting_prefixes(tail, bootstrap_queue)
            if not members:
                return waiting_queue
            deprioritized = self._compute_prefix_matches(members)
            return self._lpm.bounded_order(deprioritized, tail)

        ranker = self._active_ranker(waiting_queue)
        ranker.before_rank(waiting_queue)
        plan = ranker.match_plan
        if isinstance(plan, NoMatch):
            # Cache-agnostic policies sort without matching; match anyway so the load
            # snapshot sees num_matched_prefix_tokens. Decode never prefills.
            if (
                self.tree_cache.supports_fast_match_prefix()
                and get_disagg().disaggregation_mode != "decode"
            ):
                for r in waiting_queue:
                    match_kv_cache(r, self.tree_cache)
            ranker.sort(
                waiting_queue,
                set(),
                processed_tokens=processed_tokens,
                running_reqs=running_reqs,
            )
        elif isinstance(plan, FullMatch):
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, plan.leader_key
            )
            ranker.sort(
                waiting_queue,
                temporary_deprioritized,
                processed_tokens=processed_tokens,
                running_reqs=running_reqs,
            )
        else:
            self._match_head_window(
                waiting_queue, ranker, processed_tokens, running_reqs
            )

        # A full match already touched every request while matching.
        if not isinstance(plan, FullMatch):
            self._touch_waiting_prefixes(waiting_queue, bootstrap_queue)
        return waiting_queue

    @property
    def bounded_lpm_active(self) -> bool:
        """Whether admission is limited to bounded LPM's current batch."""
        return self._lpm is not None and self._lpm.bounded_active

    def bounded_lpm_engaged(self, num_waiting: int) -> bool:
        """Whether bounded LPM manages a pass over ``num_waiting`` requests."""
        return self._lpm is not None and self._lpm.bounded_engaged(num_waiting)

    def retire_waiting_members(self, reqs: Iterable[Req]) -> None:
        if self._lpm is not None:
            self._lpm.retire_waiting_members(reqs)

    def mark_admission_skipped(self, req: Req) -> None:
        if self._lpm is not None:
            self._lpm.mark_admission_skipped(req)

    def admission_candidates(self, waiting: Iterable[Req]) -> Iterable[Req]:
        """Yield current members first, then at most one fixed fallback batch.

        The caller reports temporary skips. A resource/budget break closes this
        generator before it can expose next-batch candidates. Matching the next
        batch is lazy, so the normal path still sorts at most 64 requests.
        """
        if not self.bounded_lpm_active:
            yield from waiting
            return
        yield from self._lpm.admission_members()
        next_members = self._lpm.fallback_members(waiting)
        if next_members:
            deprioritized = self._compute_prefix_matches(next_members)
            self._lpm.sort(next_members, deprioritized)
            yield from next_members

    def allows_admission(self, req: Req) -> bool:
        return self._lpm is None or self._lpm.allows_admission(req)

    def _touch_waiting_prefixes(
        self,
        waiting_queue: List[Req],
        bootstrap_queue: Iterable[Req] = (),
    ) -> None:
        # Head last, so LRU evicts in reverse admission order; other eviction
        # strategies ignore recency.
        bootstrap_max = envs.SGLANG_WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX.get()
        if (
            not (waiting_queue or (bootstrap_queue and bootstrap_max > 0))
            or not envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.get()
            or get_disagg().disaggregation_mode == "decode"
            or get_memory().radix_eviction_policy.lower()
            not in ("lru", "joint_lru", "state_cost")
            or self.tree_cache.supports_fast_match_prefix()
            or not self.tree_cache.supports_prefix_sharing()
        ):
            return
        refresh = (
            self._waiting_prefix_refresh_calls % WAITING_PREFIX_REFRESH_INTERVAL == 0
        )
        self._waiting_prefix_refresh_calls += 1
        if not refresh:
            return
        # Bootstrap-queued requests run after the waiting queue, so touch them first.
        for r in reversed(list(bootstrap_queue)[:bootstrap_max]):
            touch_waiting_prefix(r, self.tree_cache)
        for r in reversed(waiting_queue[:WAITING_QUEUE_PREFIX_MATCH_MAX]):
            touch_waiting_prefix(r, self.tree_cache)

    def _active_ranker(self, waiting_queue: List[Req]) -> BaseRanker:
        limit = self.ranker.fcfs_fallback_queue_len
        if limit is not None and len(waiting_queue) > limit:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return self._fcfs_fallback
        return self.ranker

    def waiting_queue_prefix_matched(self, waiting_queue: List[Req]) -> bool:
        if self.bounded_lpm_active:
            return False
        ranker = self._active_ranker(waiting_queue)
        return (
            not isinstance(ranker.match_plan, NoMatch)
            or self.tree_cache.supports_fast_match_prefix()
        )

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> SchedulePolicy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = SchedulePolicy(policy)
        except ValueError:
            raise ValueError(f"Unknown schedule_policy: {policy=}")
        if policy_enum.cache_aware and getattr(tree_cache, "disable", True):
            if policy_enum == SchedulePolicy.UNIBOOST:
                # UniBoost still works without a prefix cache: every
                # prompt token counts as uncached work.
                return policy_enum
            if policy_enum == SchedulePolicy.SHORTEST_PREFILL_FIRST:
                raise ValueError(
                    "Experimental shortest-prefill-first requires prefix caching"
                )
            # If tree_cache is disabled, fall back to a cache-agnostic policy.
            return SchedulePolicy.FCFS
        return policy_enum

    def _compute_prefix_matches(
        self,
        waiting_queue: List[Req],
        leader_key: Optional[Callable[[Req], float]] = None,
    ) -> Set[int]:
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()

        # In-batch prefix caching: of the requests with almost nothing cached that
        # share a prefix, only one runs, so the shared prefix is computed once.
        candidates = []
        for r in waiting_queue:
            prefix_ids = r.origin_input_ids + r.output_ids
            match_kv_cache(r, self.tree_cache, prefix_ids)
            if r.prefix_len <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                # Only the first DEPRIORITIZE_THRESHOLD tokens can decide a match.
                key_ids = prefix_ids[:IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD]
                candidates.append((r, key_ids))

        if leader_key is not None:
            # The first of a shared-prefix group runs.
            candidates.sort(key=lambda c: leader_key(c[0]))

        for r, key_ids in candidates:
            extra_key = r.extra_key
            cache_salt = r.cache_salt
            match_result = self.waiting_queue_radix_tree.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids=key_ids,
                        extra_key=extra_key,
                        cache_salt=cache_salt,
                    )
                )
            )
            if envs.SGLANG_RADIX_FORCE_MISS.get():
                match_result = zero_match_result(
                    self.waiting_queue_radix_tree, match_result, extra_key=extra_key
                )
            if (
                match_result.device_prefix_len
                >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
            ):
                temporary_deprioritized.add(r.rid)
            else:
                # Insert with a dummy key
                self.waiting_queue_radix_tree.insert(
                    InsertParams(
                        key=RadixKey(
                            token_ids=key_ids,
                            extra_key=extra_key,
                            cache_salt=cache_salt,
                        ),
                        value=torch.empty(len(key_ids), dtype=torch.bool),
                    )
                )
        return temporary_deprioritized

    def _match_head_window(
        self,
        waiting_queue: List[Req],
        ranker: BaseRanker,
        processed_tokens: int,
        running_reqs: Optional[List[Req]],
    ) -> None:
        plan = ranker.match_plan

        def rank(queue: List[Req], temporary_deprioritized: Set[int]) -> None:
            ranker.sort(
                queue,
                temporary_deprioritized,
                processed_tokens=processed_tokens,
                running_reqs=running_reqs,
            )

        if self.tree_cache.disable:
            rank(waiting_queue, set())
            return
        if len(waiting_queue) > plan.window:
            # Past the head window, a request is matched only when first seen
            # (new or retracted: last_node is None); its count may go stale.
            for r in waiting_queue:
                if r.last_node is None:
                    match_kv_cache(r, self.tree_cache)
            rank(waiting_queue, set())
        window = waiting_queue[: plan.window]
        deprioritized = self._compute_prefix_matches(window, plan.leader_key)
        rank(window, deprioritized)
        # As in LPM and SPF, a request sharing a leader's uncached prefix waits
        # at the end of the queue until the leader has cached it.
        waiting_queue[:] = (
            [r for r in window if r.rid not in deprioritized]
            + waiting_queue[plan.window :]
            + [r for r in window if r.rid in deprioritized]
        )

    def poll_storage_prefetches(self, waiting_queue: Iterable[Req]) -> None:
        """Observe finished storage prefetches before ranking. Completion is
        otherwise observed only when admission reaches a request, which a
        request still ranked by its unloaded prefix may never do."""
        if (
            not self.ranker.ranks_by_prefill_work
            or not self.tree_cache.can_poll_prefetch_progress()
        ):
            return
        self.tree_cache.poll_prefetch_progress(
            [req.cache_request_handle for req in waiting_queue]
        )

    def shortest_prefill_chunk_limit(
        self,
        chunked_req: Req,
        waiting_queue: List[Req],
        budget: int,
        page_size: int,
        *,
        processed_tokens: int,
    ) -> Optional[int]:
        """Cap the active prefill chunk to reserve tokens for waiting requests
        that rank ahead of it under shortest-prefill-first or UniBoost."""
        if not self.ranker.ranks_by_prefill_work or budget < 2 * page_size:
            return None
        remaining = len(chunked_req.full_untruncated_fill_ids) - chunked_req.prefix_len
        chunked_rank = self.ranker.prefill_rank(
            chunked_req, remaining, processed_tokens
        )
        reserved = 0
        for req in waiting_queue:
            work = prefill_work(self.tree_cache, req)
            charge = ceil_div(work, page_size) * page_size
            # Admission also charges the chunk a hit's encoder replay rows. It
            # re-matches the prefix, but a longer page-aligned match adds at most
            # as many replay rows as it drops extend tokens, so this still fits.
            cached = len(req.origin_input_ids) + len(req.output_ids) - work
            charge += PrefillAdder._replay_len(req, cached)
            if (
                self.ranker.prefill_rank(req, work, processed_tokens) >= chunked_rank
                or reserved + charge > budget - page_size
            ):
                break
            reserved += charge
        if not reserved:
            return None
        # Page alignment keeps continuation boundaries allocator-compatible.
        return (budget - reserved) // page_size * page_size
