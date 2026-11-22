"""Unit tests for srt/distributed/pp_sync_window."""

import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import (
    host_send,
    track_host_sends,
    wait_host_sends,
)
from sglang.srt.distributed.pp_sync_window import PPSyncSendWindow
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import profiler_manager
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.observability.profile_utils import ProfileManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

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
    """A gloo send Work: a second wait blocks until the group's timeout."""

    def __init__(self, fail=False):
        self.waits = 0
        self.fail = fail

    def wait(self, timeout=None):
        assert self.waits == 0, "a gloo send waited twice blocks"
        self.waits += 1
        if self.fail:
            raise RuntimeError("Timed out waiting for send operation to complete")


class TestHostSendsBeforeProfilerStop(CustomTestCase):
    """The profiler stop waits the tracking thread's un-waited sends, each once;
    any later wait of the same send returns at once."""

    def tearDown(self):
        parallel_state._tracked_host_sends = None
        parallel_state._tracking_thread = None

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
        self.assertEqual([w.waits for w in works], [1, 1, 0, 0, 0, 0])

        wait_host_sends(timedelta(seconds=1))
        self.assertEqual([w.waits for w in works], [1] * 6)
        self.assertEqual(untracked.work.waits, 0)
        for _ in range(3):
            window.begin_round()
        self.assertEqual([w.waits for w in works], [1] * 6)

    def test_only_the_tracking_thread_sends_are_tracked(self):
        tracked = []

        def stage_thread():
            track_host_sends()
            tracked.append(host_send(_GlooSendWork(), torch.zeros(1)))

        thread = threading.Thread(target=stage_thread)
        thread.start()
        thread.join()
        other = host_send(_GlooSendWork(), torch.zeros(1))
        wait_host_sends(timedelta(seconds=1))
        self.assertEqual((tracked[0].work.waits, other.work.waits), (1, 0))

    def test_a_peer_that_never_receives_fails_the_stop_and_leaves_no_tracking(self):
        track_host_sends()
        send = host_send(_GlooSendWork(fail=True), torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError, "were not received"):
            wait_host_sends(timedelta(seconds=1))
        self.assertFalse(send.waited)
        self.assertIsNone(parallel_state._tracked_host_sends)

        manager = object.__new__(SchedulerProfilerManager)
        manager.__dict__.update(
            profile_in_progress=True,
            torch_profiler=Mock(),
            torch_profiler_output_dir=Path(tempfile.mkdtemp()),
            profile_prefix=None,
        )
        profiler = manager.torch_profiler
        with (
            published_topology(),
            envs.SGLANG_PROFILE_V2.override(False),
            patch.object(
                profiler_manager, "wait_host_sends", side_effect=RuntimeError("gone")
            ),
            self.assertRaises(RuntimeError),
        ):
            manager._stop_profile()
        profiler.stop.assert_not_called()
        self.assertFalse(manager.profile_in_progress)
        self.assertIsNone(manager.torch_profiler)

        # SGLANG_PROFILE_V2: a failed stop leaves no capture behind either.
        with published_topology():
            v2 = ProfileManager(cpu_group=None)
        v2.profiler = Mock(stop=Mock(side_effect=RuntimeError("gone")))
        v2.profiler_kwargs = {"output_dir": "unused"}
        with self.assertRaises(RuntimeError):
            v2._do_stop()
        self.assertIsNone(v2.profiler)

    def test_only_pp_loops_defer_the_step_predicate_to_the_iteration_boundary(self):
        for at_boundary in (False, True):
            manager = object.__new__(SchedulerProfilerManager)
            manager.at_boundary = at_boundary
            manager._boundary_batch = None
            manager._run_profile_predicate = Mock()
            batch = SimpleNamespace()

            manager._profile_batch_predicate(batch)
            self.assertEqual(
                manager._run_profile_predicate.call_count, 0 if at_boundary else 1
            )
            manager.run_at_boundary()
            manager._run_profile_predicate.assert_called_once_with(batch)


if __name__ == "__main__":
    unittest.main()
