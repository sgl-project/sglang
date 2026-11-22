"""Unit tests for srt/distributed/pp_sync_window."""

import threading
import time
import unittest

from sglang.srt.distributed.pp_sync_window import PPReceiveAhead, PPSyncSendWindow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def _round(window: PPSyncSendWindow, sends: int) -> list[_FakeWork]:
    """One scheduler round: open it, then issue this round's sends."""
    window.begin_round()
    works = [_FakeWork() for _ in range(sends)]
    window.add(works)
    return works


class TestPPSyncSendWindow(CustomTestCase):
    """Sends inside the window are never waited on, so a rank is not tied to the
    next rank's progress; sends older than the window are, so the backlog stays
    bounded (the leak #28916 fixed with the single-round drain)."""

    def test_rounds_within_window_are_not_waited(self):
        window = PPSyncSendWindow(rounds=3)
        rounds = [_round(window, sends=2) for _ in range(3)]
        self.assertFalse(any(w.waited for works in rounds for w in works))
        self.assertEqual(len(window), 3)

        _round(window, sends=1)
        self.assertTrue(all(w.waited for w in rounds[0]))
        self.assertFalse(any(w.waited for works in rounds[1:] for w in works))
        self.assertEqual(len(window), 3)

    def test_window_one_is_the_single_round_drain(self):
        window = PPSyncSendWindow(rounds=1)
        first = _round(window, sends=2)
        self.assertFalse(any(w.waited for w in first))

        second = _round(window, sends=1)
        self.assertTrue(all(w.waited for w in first))
        self.assertFalse(any(w.waited for w in second))
        self.assertEqual(len(window), 1)

    def test_non_positive_depth_still_keeps_the_open_round(self):
        """A window of zero rounds would pop the round it just opened and leave
        nowhere to record the sends."""
        window = PPSyncSendWindow(rounds=0)
        first = _round(window, sends=1)
        self.assertFalse(first[0].waited)
        _round(window, sends=1)
        self.assertTrue(first[0].waited)


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.001)


class TestPPReceiveAhead(CustomTestCase):
    """The thread posts the channel's receives ahead of the scheduler, in order
    and at most ``depth`` messages ahead; a failed receive fails every pop after
    the messages received before it."""

    def test_pops_in_receive_order_and_reads_at_most_depth_ahead(self):
        calls = []
        gate = threading.Event()

        def recv_one():
            gate.wait()
            calls.append(len(calls))
            return calls[-1]

        receiver = PPReceiveAhead(recv_one=recv_one, depth=2, name="t-recv")
        self.assertFalse(receiver.ready())
        gate.set()
        _wait_until(lambda: len(calls) == 3)
        time.sleep(0.05)
        # Two queued, a third received and waiting for room.
        self.assertEqual(len(calls), 3)
        self.assertTrue(receiver.ready())
        self.assertEqual([receiver.pop() for _ in range(5)], [0, 1, 2, 3, 4])

    def test_a_failed_receive_fails_this_and_every_later_pop(self):
        messages = iter([0])

        def recv_one():
            return next(messages)

        receiver = PPReceiveAhead(recv_one=recv_one, depth=4, name="t-recv-fail")
        self.assertEqual(receiver.pop(), 0)
        for _ in range(2):
            with self.assertRaises(RuntimeError) as ctx:
                receiver.pop()
            self.assertIsInstance(ctx.exception.__cause__, StopIteration)


if __name__ == "__main__":
    unittest.main()
