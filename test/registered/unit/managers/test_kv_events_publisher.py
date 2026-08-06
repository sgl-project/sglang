import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.kv_events_publisher import (  # noqa: E402
    SchedulerKvEventsPublisher,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerKvEventsPublisher(unittest.TestCase):
    def setUp(self):
        self.mock_get_stats = MagicMock()
        # Mock token_usage to be 0.5 (50% full)
        self.mock_get_stats.return_value.token_usage = 0.5
        self.mock_get_stats.return_value.num_running_reqs.total = 0
        self.mock_get_stats.return_value.num_queue_reqs.total = 0
        self.mock_get_stats.return_value.cache_hit_rate = 0.0

        self.mock_ps = MagicMock()
        self.mock_ps.attn_tp_rank = 0
        self.mock_ps.attn_cp_rank = 0
        self.mock_ps.attn_dp_rank = 0
        self.mock_ps.dp_rank = 0
        self.mock_ps.pp_rank = 0

        self.mock_tree_cache = MagicMock()
        self.mock_send_metrics = MagicMock()
        self.mock_send_metrics.closed = False

    def test_emit_kv_metrics_page_size_1(self):
        """Verify backward compatibility when page_size == 1."""
        publisher = SchedulerKvEventsPublisher(
            kv_events_config=None,
            ps=self.mock_ps,
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_dp_rank=0,
            dp_rank=0,
            tree_cache=self.mock_tree_cache,
            send_metrics_from_scheduler=self.mock_send_metrics,
            max_running_requests=10,
            max_total_num_tokens=8192,
            page_size=1,
            get_stats=self.mock_get_stats,
            enable_kv_cache_events=True,
        )

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler_components.kv_events_publisher.sock_send"
        ) as mock_sock_send:
            publisher.emit_kv_metrics()
            mock_sock_send.assert_called_once()
            metrics = mock_sock_send.call_args[0][1]

            self.assertEqual(metrics.kv_total_blocks, 8192)
            self.assertEqual(metrics.kv_active_blocks, 4096)  # 0.5 * 8192

    def test_emit_kv_metrics_page_size_greater_than_1(self):
        """Verify block metrics when page_size > 1."""
        publisher = SchedulerKvEventsPublisher(
            kv_events_config=None,
            ps=self.mock_ps,
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_dp_rank=0,
            dp_rank=0,
            tree_cache=self.mock_tree_cache,
            send_metrics_from_scheduler=self.mock_send_metrics,
            max_running_requests=10,
            max_total_num_tokens=8192,
            page_size=16,
            get_stats=self.mock_get_stats,
            enable_kv_cache_events=True,
        )

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler_components.kv_events_publisher.sock_send"
        ) as mock_sock_send:
            publisher.emit_kv_metrics()
            mock_sock_send.assert_called_once()
            metrics = mock_sock_send.call_args[0][1]

            self.assertEqual(metrics.kv_total_blocks, 512)  # 8192 // 16
            self.assertEqual(metrics.kv_active_blocks, 256)  # 0.5 * 512


if __name__ == "__main__":
    unittest.main()
