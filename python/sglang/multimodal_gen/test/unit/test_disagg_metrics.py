# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disagg pipeline metrics (Phase 4 P3)."""

import time
import unittest

from sglang.multimodal_gen.runtime.disaggregation.metrics import (
    DisaggMetrics,
)


class TestDisaggMetrics(unittest.TestCase):
    """Test DisaggMetrics tracking and snapshot."""

    def test_initial_snapshot(self):
        """Fresh metrics should have zero counts."""
        m = DisaggMetrics(role="encoder")
        s = m.snapshot()
        self.assertEqual(s.role, "encoder")
        self.assertEqual(s.requests_completed, 0)
        self.assertEqual(s.requests_failed, 0)
        self.assertEqual(s.requests_in_flight, 0)
        self.assertEqual(s.requests_timed_out, 0)
        self.assertEqual(s.queue_depth, 0)
        self.assertAlmostEqual(s.throughput_rps, 0.0)
        self.assertGreater(s.uptime_s, 0.0)

    def test_request_lifecycle(self):
        """Start -> complete should increment counts and track latency."""
        m = DisaggMetrics(role="denoising")
        m.record_request_start("req-001")

        s = m.snapshot()
        self.assertEqual(s.requests_in_flight, 1)
        self.assertEqual(s.requests_completed, 0)

        time.sleep(0.05)
        m.record_request_complete("req-001")

        s = m.snapshot()
        self.assertEqual(s.requests_in_flight, 0)
        self.assertEqual(s.requests_completed, 1)
        self.assertGreater(s.last_latency_s, 0.04)
        self.assertGreater(s.avg_latency_s, 0.04)
        self.assertGreater(s.max_latency_s, 0.04)

    def test_multiple_requests(self):
        """Track multiple concurrent requests."""
        m = DisaggMetrics(role="decoder")

        m.record_request_start("r1")
        m.record_request_start("r2")
        m.record_request_start("r3")
        self.assertEqual(m.snapshot().requests_in_flight, 3)

        m.record_request_complete("r1")
        m.record_request_complete("r2")
        self.assertEqual(m.snapshot().requests_in_flight, 1)
        self.assertEqual(m.snapshot().requests_completed, 2)

        m.record_request_failed("r3")
        self.assertEqual(m.snapshot().requests_in_flight, 0)
        self.assertEqual(m.snapshot().requests_failed, 1)

    def test_timeout_tracking(self):
        m = DisaggMetrics(role="encoder")
        m.record_request_start("r1")
        m.record_request_timeout("r1")

        s = m.snapshot()
        self.assertEqual(s.requests_timed_out, 1)
        self.assertEqual(s.requests_in_flight, 0)

    def test_queue_depth(self):
        m = DisaggMetrics(role="encoder")
        m.update_queue_depth(5)
        self.assertEqual(m.snapshot().queue_depth, 5)
        m.update_queue_depth(0)
        self.assertEqual(m.snapshot().queue_depth, 0)

    def test_throughput(self):
        """Throughput should reflect completions within the window."""
        m = DisaggMetrics(role="denoising")
        # Complete 5 requests
        for i in range(5):
            m.record_request_start(f"r{i}")
            m.record_request_complete(f"r{i}")

        s = m.snapshot()
        self.assertEqual(s.requests_completed, 5)
        # Should have positive throughput (5 completions in last 60s window)
        self.assertGreater(s.throughput_rps, 0.0)

    def test_to_dict(self):
        """Snapshot should serialize to a dict with all expected keys."""
        m = DisaggMetrics(role="encoder")
        m.record_request_start("r1")
        m.record_request_complete("r1")

        d = m.snapshot().to_dict()
        expected_keys = {
            "role",
            "requests_completed",
            "requests_failed",
            "requests_in_flight",
            "requests_timed_out",
            "queue_depth",
            "last_latency_s",
            "avg_latency_s",
            "max_latency_s",
            "throughput_rps",
            "uptime_s",
        }
        self.assertEqual(set(d.keys()), expected_keys)
        self.assertEqual(d["role"], "encoder")
        self.assertEqual(d["requests_completed"], 1)

    def test_max_latency_tracks_worst_case(self):
        m = DisaggMetrics(role="encoder")

        m.record_request_start("fast")
        m.record_request_complete("fast")
        fast_latency = m.snapshot().max_latency_s

        m.record_request_start("slow")
        time.sleep(0.1)
        m.record_request_complete("slow")

        s = m.snapshot()
        self.assertGreater(s.max_latency_s, fast_latency)
        self.assertGreater(s.max_latency_s, 0.09)


if __name__ == "__main__":
    unittest.main()
