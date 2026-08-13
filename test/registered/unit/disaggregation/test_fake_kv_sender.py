import unittest
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFakeKVSender(unittest.TestCase):
    """Test FakeKVSender queue accumulation fix and lifecycle behavior."""

    def setUp(self):
        """Create a FakeKVSender instance for testing."""
        self.mgr = MagicMock()
        self.sender = FakeKVSender(
            mgr=self.mgr,
            bootstrap_addr="fake_addr:1234",
            bootstrap_room=42,
            dest_tp_ranks=[0],
            pp_rank=0,
        )

    def test_first_poll_returns_waiting_for_input(self):
        """First poll should return WaitingForInput for proper handshake."""
        result = self.sender.poll()
        self.assertEqual(result, KVPoll.WaitingForInput)
        self.assertEqual(self.sender.poll_count, 1)

    def test_multiple_polls_without_send_return_waiting_for_input(self):
        """Polls 2-99 should return WaitingForInput without auto-completing."""
        for i in range(1, 99):
            result = self.sender.poll()
            self.assertEqual(result, KVPoll.WaitingForInput)
            self.assertEqual(self.sender.poll_count, i)

    def test_auto_abort_after_threshold(self):
        """After 100 polls without send(), should auto-abort to Failed."""
        # Poll 99 times (still WaitingForInput)
        for i in range(99):
            result = self.sender.poll()
            self.assertEqual(result, KVPoll.WaitingForInput)

        # 100th poll should auto-abort
        result = self.sender.poll()
        self.assertEqual(result, KVPoll.Failed)
        self.assertEqual(self.sender.poll_count, 100)
        self.assertEqual(self.sender.conclude_state, KVPoll.Failed)

        # Subsequent polls should return cached Failed
        result = self.sender.poll()
        self.assertEqual(result, KVPoll.Failed)

    def test_normal_send_flow(self):
        """Normal flow: poll -> send -> poll should work correctly."""
        # First poll returns WaitingForInput
        result = self.sender.poll()
        self.assertEqual(result, KVPoll.WaitingForInput)

        # Send KV data
        kv_indices = np.array([0, 1, 2], dtype=np.int32)
        self.sender.send(kv_indices)
        self.assertTrue(self.sender.has_sent)
        self.assertEqual(self.sender.poll_count, 0)  # Reset after send

        # Next poll should return Success
        result = self.sender.poll()
        self.assertEqual(result, KVPoll.Success)
        self.assertEqual(self.sender.conclude_state, KVPoll.Success)

        # Subsequent polls return cached Success
        result = self.sender.poll()
        self.assertEqual(result, KVPoll.Success)

    def test_send_resets_poll_count(self):
        """send() should reset poll_count to prevent carry-over."""
        # Poll multiple times
        for _ in range(5):
            self.sender.poll()
        self.assertEqual(self.sender.poll_count, 5)

        # Send should reset counter
        kv_indices = np.array([0], dtype=np.int32)
        self.sender.send(kv_indices)
        self.assertEqual(self.sender.poll_count, 0)

    def test_send_without_poll(self):
        """send() can be called without prior polling."""
        kv_indices = np.array([0, 1], dtype=np.int32)
        self.sender.send(kv_indices)
        self.assertTrue(self.sender.has_sent)
        self.assertEqual(self.sender.poll_count, 0)

        # Next poll should return Success
        result = self.sender.poll()
        self.assertEqual(result, KVPoll.Success)

    def test_conclude_state_caching(self):
        """Once conclude_state is set, it should be returned immediately."""
        # Trigger auto-abort
        for _ in range(100):
            self.sender.poll()

        self.assertEqual(self.sender.conclude_state, KVPoll.Failed)

        # All subsequent polls should return cached state without incrementing poll_count
        for _ in range(10):
            result = self.sender.poll()
            self.assertEqual(result, KVPoll.Failed)
            self.assertEqual(self.sender.poll_count, 100)  # Unchanged

    def test_abort_sets_failed_state(self):
        """abort() should set conclude_state to Failed."""
        self.sender.abort()
        self.assertEqual(self.sender.conclude_state, KVPoll.Failed)

        # Subsequent polls should return Failed
        result = self.sender.poll()
        self.assertEqual(result, KVPoll.Failed)

    def test_init_method(self):
        """init() should execute without errors."""
        self.sender.init(kv_indices=[0, 1, 2], aux_index=5)
        # Should not raise any exceptions

    def test_get_transfer_metric(self):
        """get_transfer_metric() should return empty metric."""
        metric = self.sender.get_transfer_metric()
        self.assertIsNone(metric.transfer_latency_s)
        self.assertIsNone(metric.alloc_latency_s)
        self.assertIsNone(metric.transfer_total_bytes)

    def test_failure_exception(self):
        """failure_exception() should raise exception."""
        with self.assertRaises(Exception) as ctx:
            self.sender.failure_exception()
        self.assertIn("Fake KVSender Exception", str(ctx.exception))

    def test_configurable_threshold(self):
        """Threshold should be configurable via environment variable."""
        # Test with custom threshold of 50
        with envs.SGLANG_DISAGGREGATION_FAKE_AUTO_ABORT_THRESHOLD.override(50):
            custom_sender = FakeKVSender(
                mgr=self.mgr,
                bootstrap_addr="fake_addr:1234",
                bootstrap_room=42,
                dest_tp_ranks=[0],
                pp_rank=0,
            )
            self.assertEqual(custom_sender._auto_abort_threshold, 50)

            # Poll 49 times - should still return WaitingForInput
            for i in range(49):
                result = custom_sender.poll()
                self.assertEqual(result, KVPoll.WaitingForInput)

            # 50th poll should auto-abort
            result = custom_sender.poll()
            self.assertEqual(result, KVPoll.Failed)
            self.assertEqual(custom_sender.poll_count, 50)


if __name__ == "__main__":
    unittest.main()
