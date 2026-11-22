"""Unit tests for srt/distributed/pp_sync_window."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import (
    host_send,
    track_host_sends,
    wait_host_sends,
)
from sglang.srt.distributed.pp_sync_window import PPSyncSendWindow
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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


class _GlooSendWork:
    """A gloo send Work: completed only once waited, and a second wait never returns."""

    def __init__(self):
        self.waits = 0

    def wait(self):
        assert self.waits == 0, "a gloo send waited twice blocks"
        self.waits += 1

    def is_completed(self):
        return self.waits > 0


class TestHostSendsBeforeProfilerStop(CustomTestCase):
    """A gloo send started under the torch profiler ends its RecordFunction in
    wait(); waited after profiler.stop(), that write lands in freed memory. The
    stop waits the scheduler thread's un-waited sends first, each exactly once."""

    def tearDown(self):
        parallel_state._tracked_host_sends = None

    def test_stop_waits_each_unwaited_send_once_and_the_window_skips_it_later(self):
        untracked = host_send(_GlooSendWork(), torch.zeros(1))
        track_host_sends()
        window = PPSyncSendWindow(rounds=2)
        rounds = []
        for _ in range(3):
            window.begin_round()
            rounds.append(
                [host_send(_GlooSendWork(), torch.zeros(1)) for _ in range(2)]
            )
            window.add(rounds[-1])
        works = [p2p.work for sends in rounds for p2p in sends]
        # The window already waited the oldest round; the stop must not wait it again.
        self.assertEqual([w.waits for w in works], [1, 1, 0, 0, 0, 0])

        self.assertEqual(wait_host_sends(), 4)
        self.assertEqual([w.waits for w in works], [1] * 6)
        self.assertEqual(untracked.work.waits, 0)
        # The window's own later waits reach none of the drained sends.
        for _ in range(3):
            window.begin_round()
        self.assertEqual([w.waits for w in works], [1] * 6)
        self.assertEqual(wait_host_sends(), 0)

    def test_only_the_scheduler_thread_sends_are_tracked(self):
        track_host_sends()
        other = []
        thread = threading.Thread(
            target=lambda: other.append(host_send(_GlooSendWork(), torch.zeros(1)))
        )
        thread.start()
        thread.join()
        self.assertEqual(wait_host_sends(), 0)
        self.assertEqual(other[0].work.waits, 0)

    def test_pp_loops_run_the_step_predicate_at_the_iteration_boundary(self):
        manager = object.__new__(SchedulerProfilerManager)
        manager.at_boundary = True
        manager._boundary_batch = None
        manager._run_profile_predicate = Mock()
        batch = SimpleNamespace()

        manager._profile_batch_predicate(batch)
        manager._run_profile_predicate.assert_not_called()
        manager.run_at_boundary()
        manager._run_profile_predicate.assert_called_once_with(batch)
        manager.run_at_boundary()
        manager._run_profile_predicate.assert_called_once_with(batch)


if __name__ == "__main__":
    unittest.main()
