"""Highest response ratio next."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

from sglang.srt.managers.schedule_policies.base import BaseRanker, FullMatch
from sglang.srt.managers.schedule_policies.common import WAITING_QUEUE_PREFIX_MATCH_MAX

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class HrrnRanker(BaseRanker):
    match_plan = FullMatch()
    fcfs_fallback_queue_len = WAITING_QUEUE_PREFIX_MATCH_MAX

    @staticmethod
    def _uncached_len(r: Req) -> int:
        """Number of tokens that must actually be prefilled for this req
        (all cache levels — device + host via hicache — counted as cached)."""
        return max(0, len(r.origin_input_ids) - r.num_matched_prefix_tokens)

    def sort(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        *,
        processed_tokens: int = 0,
        running_reqs: Optional[List[Req]] = None,
    ) -> None:
        """Highest Response Ratio Next, with token-based aging.

        Equivalence with classic HRRN when throughput is constant:
            ratio = 1 + wait_sec / est_prefill_time
                  = 1 + (processed_tokens - arrival_processed_tokens) / uncached

        Caller (Scheduler) contract:
          - Maintain a monotonically increasing counter of prefill tokens processed so far
            (accumulate batch.extend_num_tokens per forward). Pass it in as `processed_tokens`.
          - Snapshot `req.arrival_processed_tokens = counter` when the req enters waiting_queue
            (move_bootstrapped for disagg prefill, _add_request_to_queue for unified).

        Call sites that omit `processed_tokens` (dllm, disagg decode)
        degrade to rid-lexicographic order; those queues carry no prefill work.
        """

        def _key(r: Req):
            rid = r.rid
            if rid in temporary_deprioritized:
                return (float("inf"), rid)
            uncached = HrrnRanker._uncached_len(r)
            if uncached <= 0:
                # No prefill work; drain immediately.
                return (-float("inf"), rid)
            waited_tokens = max(0, processed_tokens - r.arrival_processed_tokens)
            ratio_delta = waited_tokens / uncached
            return (-ratio_delta, rid)

        waiting_queue.sort(key=_key)
