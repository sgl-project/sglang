"""Unit tests for KV metric block accounting."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_components import kv_events_publisher
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAX_TOTAL_NUM_TOKENS = 8192
TOKEN_USAGE = 0.5


def emit_metrics(page_size):
    stats = SimpleNamespace(
        num_running_reqs=SimpleNamespace(total=0),
        num_queue_reqs=SimpleNamespace(total=0),
        token_usage=TOKEN_USAGE,
        cache_hit_rate=0.0,
    )
    publisher = SchedulerKvEventsPublisher(
        kv_events_config=None,
        ps=SimpleNamespace(dp_rank=None),
        attn_tp_rank=0,
        attn_cp_rank=0,
        attn_dp_rank=0,
        dp_rank=None,
        tree_cache=SimpleNamespace(),
        send_metrics_from_scheduler=SimpleNamespace(closed=False),
        max_running_requests=16,
        max_total_num_tokens=MAX_TOTAL_NUM_TOKENS,
        page_size=page_size,
        get_stats=lambda: stats,
    )
    publisher.enable_kv_cache_events = True

    with patch.object(kv_events_publisher, "sock_send") as sock_send:
        publisher.emit_kv_metrics()

    return sock_send.call_args.args[1]


class TestKvMetricsBlockUnits(CustomTestCase):
    def test_paged_kv_reports_pages_not_tokens(self):
        metrics = emit_metrics(page_size=16)
        num_blocks = MAX_TOTAL_NUM_TOKENS // 16

        self.assertEqual(metrics.kv_total_blocks, num_blocks)
        self.assertEqual(metrics.kv_active_blocks, int(TOKEN_USAGE * num_blocks))

    def test_page_size_one_is_unchanged(self):
        metrics = emit_metrics(page_size=1)

        self.assertEqual(metrics.kv_total_blocks, MAX_TOTAL_NUM_TOKENS)
        self.assertEqual(
            metrics.kv_active_blocks, int(TOKEN_USAGE * MAX_TOTAL_NUM_TOKENS)
        )


if __name__ == "__main__":
    unittest.main()
