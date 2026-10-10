"""Timeout observations count once without changing KV failure handling."""

import threading
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

from prometheus_client import CollectorRegistry, Counter

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKVTransferTimeoutMetrics(CustomTestCase):
    def setUp(self):
        self.registry = CollectorRegistry()
        self.counter = Counter(
            "sglang:kv_transfer_timeouts_total",
            "Timeout observations.",
            ["stage"],
            registry=self.registry,
        )
        for stage in ("bootstrap", "transfer"):
            self.counter.labels(stage=stage).inc(0)

    def connection(self, *, enabled=True):
        manager = SimpleNamespace(
            bootstrap_timeout=10,
            waiting_timeout=10,
            kv_transfer_timeouts=self.counter if enabled else None,
            request_status={7: KVPoll.Bootstrapping},
            failure_records={},
            failure_lock=threading.Lock(),
            deferred_bootstrap=None,
            defer_decode_allocation=False,
        )
        manager.record_failure = MethodType(CommonKVManager.record_failure, manager)
        manager.update_status = MethodType(CommonKVManager.update_status, manager)
        return SimpleNamespace(
            kv_mgr=manager,
            bootstrap_room=7,
            init_time=80,
            abort_notified=False,
            bootstrap_infos=[{"rank_ip": "127.0.0.1", "rank_port": 2001}],
            invalidate_cached_bootstrap_infos=Mock(),
            _send_abort_notification=Mock(),
            ensure_abort_notified=Mock(),
            _owns_bootstrap_room=True,
        )

    def count(self, stage):
        return self.registry.get_sample_value(
            "sglang:kv_transfer_timeouts_total", {"stage": stage}
        )

    @staticmethod
    def timeout_paths():
        return (
            (CommonKVSender._check_bootstrap_timeout, "bootstrap"),
            (CommonKVReceiver._check_waiting_timeout, "transfer"),
        )

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=100)
    def test_timeout_counts_once_and_preserves_failure(self, _time):
        for check, stage in self.timeout_paths():
            with self.subTest(stage=stage):
                connection = self.connection()
                for _ in range(3):
                    self.assertEqual(check(connection), KVPoll.Failed)
                self.assertEqual(self.count(stage), 1)
                self.assertEqual(connection.kv_mgr.request_status[7], KVPoll.Failed)
                self.assertIn(
                    "timed out after 20.0s", connection.kv_mgr.failure_records[7]
                )
                if stage == "transfer":
                    self.assertTrue(connection.abort_notified)
                    connection._send_abort_notification.assert_called_once_with()
                    self.assertEqual(
                        connection.invalidate_cached_bootstrap_infos.call_count, 3
                    )

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=100)
    def test_unstarted_and_before_deadline_do_not_count(self, _time):
        for check, stage in self.timeout_paths():
            with self.subTest(stage=stage):
                connection = self.connection()
                for init_time in (None, 95):
                    connection.init_time = init_time
                    self.assertIsNone(check(connection))
                self.assertEqual(self.count(stage), 0)
                self.assertEqual(
                    connection.kv_mgr.request_status, {7: KVPoll.Bootstrapping}
                )
                self.assertEqual(connection.kv_mgr.failure_records, {})
                # The existing deadline is inclusive.
                connection.init_time = 90
                self.assertEqual(check(connection), KVPoll.Failed)
                self.assertEqual(self.count(stage), 1)

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=100)
    def test_disabled_metrics_preserve_timeout_behavior(self, _time):
        for check, stage in self.timeout_paths():
            with self.subTest(stage=stage):
                connection = self.connection(enabled=False)
                self.assertEqual(check(connection), KVPoll.Failed)
                self.assertEqual(check(connection), KVPoll.Failed)
                self.assertEqual(self.count(stage), 0)
                self.assertEqual(connection.kv_mgr.request_status[7], KVPoll.Failed)

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=100)
    def test_deferred_bootstrap_uses_the_existing_completion_clock(self, _time):
        connection = self.connection()
        connection.kv_mgr.defer_decode_allocation = True
        for completed_at in (None, 95):
            connection._prefill_complete_time = completed_at
            self.assertIsNone(CommonKVSender._check_bootstrap_timeout(connection))
        self.assertEqual(self.count("bootstrap"), 0)
        connection._prefill_complete_time = 90
        self.assertEqual(
            CommonKVSender._check_bootstrap_timeout(connection), KVPoll.Failed
        )
        self.assertEqual(self.count("bootstrap"), 1)

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=100)
    def test_new_connection_with_reused_room_counts_again(self, _time):
        for check, stage in self.timeout_paths():
            with self.subTest(stage=stage):
                first = self.connection()
                second = self.connection()
                second.kv_mgr = first.kv_mgr
                for connection in (first, second):
                    self.assertEqual(check(connection), KVPoll.Failed)
                    self.assertEqual(check(connection), KVPoll.Failed)
                self.assertEqual(self.count(stage), 2)

    @patch("sglang.srt.disaggregation.common.conn.time.time", return_value=100)
    def test_existing_failure_record_does_not_suppress_timeout(self, _time):
        for check, stage in self.timeout_paths():
            with self.subTest(stage=stage):
                connection = self.connection()
                connection.kv_mgr.failure_records[7] = "Earlier failure"
                self.assertEqual(check(connection), KVPoll.Failed)
                self.assertEqual(self.count(stage), 1)

    def test_abort_is_not_a_timeout(self):
        for cls in (CommonKVSender, CommonKVReceiver):
            with self.subTest(cls=cls):
                connection = self.connection()
                cls.abort(connection)
                self.assertEqual(connection.conclude_state, KVPoll.Failed)
                self.assertEqual(
                    connection.kv_mgr.failure_records[7], "Aborted by AbortReq."
                )
                self.assertEqual(self.count("bootstrap"), 0)
                self.assertEqual(self.count("transfer"), 0)

    def test_terminal_receiver_poll_does_not_observe_a_timeout(self):
        for status in (KVPoll.Success, KVPoll.Failed):
            with self.subTest(status=status):
                connection = self.connection()
                connection.conclude_state = status
                self.assertEqual(CommonKVReceiver._poll(connection), status)
                self.assertEqual(self.count("transfer"), 0)
                self.assertEqual(connection.kv_mgr.failure_records, {})


if __name__ == "__main__":
    unittest.main()
