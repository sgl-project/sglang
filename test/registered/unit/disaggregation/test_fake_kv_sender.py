import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVSender
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFakeKVSender(unittest.TestCase):
    """A FakeKVSender whose send() never comes must not pin the prefill
    inflight queue forever, and a healthy request must never be failed early."""

    def setUp(self):
        self.mgr = self._make_mgr()
        self.sender = self._make_sender()

    def _make_mgr(self) -> FakeKVManager:
        return FakeKVManager(
            args=KVArgs(),
            disaggregation_mode=DisaggregationMode.PREFILL,
            server_args=MagicMock(),
        )

    def _make_sender(self, mgr=None) -> FakeKVSender:
        return FakeKVSender(
            mgr=self.mgr if mgr is None else mgr,
            bootstrap_addr="fake_addr:1234",
            bootstrap_room=42,
            dest_tp_ranks=[0],
            pp_rank=0,
        )

    def _expire_deadline(self, sender: FakeKVSender, by: float = 1.0) -> None:
        """Push the armed deadline `by` seconds past the timeout."""
        sender.waiting_since -= sender.waiting_timeout + by

    def test_polls_before_send_stay_waiting_for_input(self):
        """Repeated polling must not conclude on its own: the scheduler polls a
        waiting request once per loop iteration, and those are healthy requests."""
        self.sender.init(3, 5)
        for _ in range(1000):
            self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(self.sender.conclude_state)

    def test_normal_send_flow(self):
        self.sender.init(3, 5)
        self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)

        self.sender.send(np.array([0, 1, 2], dtype=np.int32))
        self.assertTrue(self.sender.has_sent)

        self.assertEqual(self.sender.poll(), KVPoll.Success)
        self.assertEqual(self.sender.conclude_state, KVPoll.Success)
        # Cached afterwards.
        self.assertEqual(self.sender.poll(), KVPoll.Success)

    def test_zero_page_last_chunk_still_sends(self):
        """A zero-page last chunk must not be gated out: skipping it leaves the
        sender in WaitingForInput forever and pins the prefill inflight queue."""
        self.assertTrue(self.sender.should_send_kv_chunk(0, last_chunk=True))
        self.assertFalse(self.sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(self.sender.should_send_kv_chunk(3, last_chunk=False))

    def test_fully_cached_request_concludes(self):
        """Regression: a request whose last chunk carries no pages still reaches
        a terminal poll state instead of accumulating in the inflight queue."""
        self.sender.init(0, 0)
        page_indices = np.array([], dtype=np.int32)
        # Mirrors the send_kv_chunk gate in SchedulerDisaggregationPrefillMixin.
        if self.sender.should_send_kv_chunk(len(page_indices), True):
            self.sender.send(page_indices)
        self.assertEqual(self.sender.poll(), KVPoll.Success)

    def test_send_without_init_or_poll(self):
        self.sender.send(np.array([0, 1], dtype=np.int32))
        self.assertEqual(self.sender.poll(), KVPoll.Success)

    def test_waiting_timeout_fails_a_stuck_sender(self):
        self.sender.init(1, 0)
        self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)

        self._expire_deadline(self.sender)
        self.assertEqual(self.sender.poll(), KVPoll.Failed)
        self.assertEqual(self.sender.conclude_state, KVPoll.Failed)
        # Terminal: stays Failed even if send() arrives late.
        self.sender.send(np.array([0], dtype=np.int32))
        self.assertEqual(self.sender.poll(), KVPoll.Failed)

    def test_poll_does_not_read_the_timeout_off_the_manager(self):
        """Regression: a FAKE_BOOTSTRAP_HOST req on a real transfer backend pairs
        FakeKVSender with that backend's KVManager, which defines no
        waiting_timeout in prefill mode. Reading the knob off the manager raised
        AttributeError in the scheduler loop on the first poll after init()."""

        class ManagerWithoutWaitingTimeout:
            pass

        sender = self._make_sender(mgr=ManagerWithoutWaitingTimeout())
        sender.init(1, 0)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

        self._expire_deadline(sender)
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_queue_and_prefill_time_is_not_charged_to_the_deadline(self):
        """The scheduler stops polling between init() and the last chunk, so the
        deadline starts at the first poll. Arming it at init() instead would fail
        a healthy request that merely queued for longer than the timeout."""
        self.sender.init(1, 0)
        self.assertIsNone(self.sender.waiting_since)

        after_init = time.monotonic()
        self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)
        self.assertGreaterEqual(self.sender.waiting_since, after_init)

    def test_no_timeout_before_init(self):
        """A sender still in the bootstrap queue has no deadline, matching the
        real backends, which cover that window with the bootstrap timeout."""
        self.assertFalse(self.sender.inited)
        for _ in range(100):
            self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(self.sender.waiting_since)

    def test_timeout_is_configurable(self):
        """The knob is read once, when the sender is built."""
        with envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(600):
            sender = self._make_sender()
        self.assertEqual(sender.waiting_timeout, 600)

        sender.init(1, 0)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        sender.waiting_since -= 599
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        sender.waiting_since -= 2
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_abort_sets_failed_state(self):
        self.sender.abort()
        self.assertEqual(self.sender.conclude_state, KVPoll.Failed)
        self.assertEqual(self.sender.poll(), KVPoll.Failed)

    def test_get_transfer_metric(self):
        metric = self.sender.get_transfer_metric()
        self.assertIsNone(metric.transfer_latency_s)
        self.assertIsNone(metric.alloc_latency_s)
        self.assertIsNone(metric.transfer_total_bytes)

    def test_failure_exception(self):
        with self.assertRaises(Exception) as ctx:
            self.sender.failure_exception()
        self.assertIn("Fake KVSender Exception", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
