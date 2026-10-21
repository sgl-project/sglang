"""Scheduler idle behavior: on_idle's stalled-path load publish is wall-clock
bounded, and a Rust-server rank parks on its request ring only when it is safe.

A non-storage no-batch stall spins on_idle without sleeping, so the gate must
cap the O(queue) get_loads for both the DP-balancing writer and the load
socket. CPU-only: builds a bare Scheduler with mocked collaborators, like
test_scheduler_flush_cache.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestOnIdleStallPublish(CustomTestCase):
    def _stalled_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.enable_hicache_storage = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_lmcache = False
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=False)  # stalled, not idle
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s.dp_attn_adapter = MagicMock()
        s._last_stall_publish_ts = float("-inf")
        return s

    def test_stall_publish_is_wall_clock_bounded_by_the_floor(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            # A spinning stall inside the floor publishes once.
            mono.return_value = 100.0
            for _ in range(100):
                s.on_idle()
            self.assertEqual(s.publish_load_snapshot.call_count, 1)
            self.assertEqual(s.load_publisher.publish_load_stat.call_count, 1)
            # After the floor elapses the next stall publishes again.
            mono.return_value = 100.10  # > LOAD_STALL_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)


class _RecordingRequestRing:
    def __init__(self):
        self.park_timeouts_ms = []

    def wait_request(self, timeout_ms: int) -> None:
        self.park_timeouts_ms.append(timeout_ms)


class TestRustServerIdlePark(CustomTestCase):
    def test_attention_dp_rank_parks_only_with_sleep_on_idle(self):
        """An idle attention-DP rank must not park on its ring by default: a request
        on a peer's listener never wakes it, yet the peer's next step gathers every
        DP rank. --sleep-on-idle and non-attention-DP ranks still park."""
        for attn_dp_enabled, sleep_on_idle, parks in (
            (True, False, False),
            (True, True, True),
            (False, False, True),
        ):
            with self.subTest(
                attn_dp_enabled=attn_dp_enabled, sleep_on_idle=sleep_on_idle
            ):
                ring = _RecordingRequestRing()
                s = Scheduler.__new__(Scheduler)
                s._engine_paused = False
                s.get_rust_server_class = lambda: SimpleNamespace(
                    launch=lambda _scheduler: ring
                )
                parallel = SimpleNamespace(
                    pp_rank=0,
                    attn_tp_rank=0,
                    attn_cp_rank=0,
                    attn_dp_enabled=attn_dp_enabled,
                )
                with (
                    envs.SGLANG_RUST_SERVER.override(True),
                    patch(
                        "sglang.srt.managers.scheduler.get_parallel",
                        return_value=parallel,
                    ),
                    patch(
                        "sglang.srt.managers.scheduler.get_device",
                        return_value=SimpleNamespace(sleep_on_idle=sleep_on_idle),
                    ),
                ):
                    s.maybe_init_rust_server()
                    s.maybe_sleep_on_idle()
                self.assertEqual(bool(ring.park_timeouts_ms), parks)


if __name__ == "__main__":
    unittest.main()
