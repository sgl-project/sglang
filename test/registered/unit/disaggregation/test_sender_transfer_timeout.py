"""The prefill-side deadline for a KV transfer whose decode never completes it."""

import time
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 7


def make_sender(status, *, last_chunk_sent=True, wait_start=None, waiting_timeout=300):
    """A CommonKVSender past bootstrap, with only the manager surface poll() touches."""
    statuses = {ROOM: status}
    failures = []
    manager = SimpleNamespace(
        waiting_timeout=waiting_timeout,
        bootstrap_timeout=300,
        defer_decode_allocation=False,
        reports_transfer_latency=False,
        request_status=statuses,
        _staging_outstanding={},
        check_status=lambda room: statuses[room],
        update_status=lambda room, s: statuses.__setitem__(room, s),
        record_failure=lambda room, reason: failures.append(reason),
    )
    sender = CommonKVSender.__new__(CommonKVSender)
    sender.kv_mgr = manager
    sender.bootstrap_room = ROOM
    sender.conclude_state = None
    sender.init_time = 0.0  # bootstrap finished long ago; irrelevant past Bootstrapping
    sender._prefill_complete_time = None
    sender._last_chunk_sent = last_chunk_sent
    sender._transfer_wait_start = wait_start
    sender._transfer_start_time = None
    sender._transfer_metric = SimpleNamespace(transfer_latency_s=None)
    sender.trace_ctx = SimpleNamespace(trace_req_finish=lambda: None)
    return sender, statuses, failures


class TestSenderTransferTimeout(unittest.TestCase):
    def test_transfer_the_decode_never_completed_times_out(self):
        sender, statuses, failures = make_sender(KVPoll.Transferring, wait_start=0.0)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(statuses[ROOM], KVPoll.Failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertEqual(len(failures), 1)
        self.assertIn("KVPoll.Transferring", failures[0])

    def test_last_chunk_queued_but_not_yet_picked_up_also_times_out(self):
        sender, _, failures = make_sender(KVPoll.WaitingForInput, wait_start=0.0)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertIn("KVPoll.WaitingForInput", failures[0])

    def test_first_poll_past_bootstrap_arms_the_clock_without_failing(self):
        # The clock starts when the room is first seen past Bootstrapping with
        # the last chunk sent, so parking time (prefill_complete) and prefill
        # compute time never count against the transfer.
        sender, _, failures = make_sender(KVPoll.Transferring)
        before = time.time()
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertIsNotNone(sender._transfer_wait_start)
        self.assertGreaterEqual(sender._transfer_wait_start, before)
        self.assertEqual(failures, [])

    def test_no_deadline_while_prefill_is_still_producing_chunks(self):
        sender, _, failures = make_sender(KVPoll.WaitingForInput, last_chunk_sent=False)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(sender._transfer_wait_start)
        self.assertEqual(failures, [])

    def test_a_transfer_within_the_deadline_keeps_waiting(self):
        sender, _, failures = make_sender(KVPoll.Transferring, wait_start=time.time())
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertEqual(failures, [])

    def test_deadline_follows_the_waiting_timeout(self):
        sender, _, failures = make_sender(
            KVPoll.Transferring, wait_start=time.time() - 400, waiting_timeout=600
        )
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertEqual(failures, [])

    def test_bootstrapping_is_left_to_the_bootstrap_timeout(self):
        sender, _, failures = make_sender(KVPoll.Bootstrapping, wait_start=0.0)
        sender.init_time = time.time()
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.assertEqual(failures, [])

    def test_success_is_not_touched(self):
        sender, _, failures = make_sender(KVPoll.Success, wait_start=0.0)
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
