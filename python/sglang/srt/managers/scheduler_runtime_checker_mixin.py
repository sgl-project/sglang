from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.metrics_collector import QueueCount

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:
    def _maybe_log_idle_metrics(self: Scheduler):
        """Collect and log metrics every 30 seconds during idle."""
        if (
            not self.current_scheduler_metrics_enabled
            or time.perf_counter() <= self.metrics_collector.last_log_time + 30
        ):
            return

        self.get_pool_stats().update_scheduler_stats(self.stats)
        self.stats.num_streaming_sessions = self._streaming_session_count()
        self.stats.streaming_session_held_tokens = self._session_held_tokens()

        priority_enabled = self.enable_priority_scheduling
        self.stats.num_running_reqs = QueueCount.from_reqs(
            self.running_batch.reqs, priority_enabled
        )
        self.stats.gen_throughput = 0
        self.stats.num_queue_reqs = QueueCount.from_reqs(
            self.waiting_queue, priority_enabled
        )
        self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                self.disagg_prefill_bootstrap_queue.queue, priority_enabled
            )
            self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                self.disagg_prefill_inflight_queue, priority_enabled
            )
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                self.disagg_decode_prealloc_queue.queue, priority_enabled
            )
            self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                self.disagg_decode_transfer_queue.queue, priority_enabled
            )
        self.metrics_collector.log_stats(self.stats)
