"""Tests for SchedulerActorMonitor (requires Ray, no GPU).

Stand-in actors replace the scheduler actors: ``run_event_loop`` blocks, like
the real one, until the test ends it or kills the actor.

Usage:
    python -m pytest test/manual/test_ray_scheduler_monitor.py -v
"""

from __future__ import annotations

import queue
import threading
import unittest

import ray

from sglang.srt.ray.scheduler_monitor import SchedulerActorMonitor
from sglang.test.test_utils import CustomTestCase

_EXIT_TIMEOUT_S = 30


@ray.remote(num_cpus=0)
class _FakeSchedulerActor:
    def __init__(self):
        self._stop = threading.Event()

    def run_event_loop(self, fail: bool = False) -> None:
        self._stop.wait()
        if fail:
            raise ValueError("scheduler crashed")

    def stop(self) -> None:
        self._stop.set()


class TestSchedulerActorMonitor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=4, include_dashboard=False)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def _start(self, *, num_actors: int = 2, fail: bool = False):
        # max_concurrency lets stop() run while run_event_loop blocks.
        actors = [
            _FakeSchedulerActor.options(max_concurrency=2).remote()
            for _ in range(num_actors)
        ]
        refs = [actor.run_event_loop.remote(fail=fail) for actor in actors]
        errors: queue.Queue = queue.Queue()
        monitor = SchedulerActorMonitor(event_loop_refs=refs, on_exit=errors.put)
        monitor.start()
        self.addCleanup(monitor.stop)
        for actor in actors:
            self.addCleanup(ray.kill, actor)
        return actors, monitor, errors

    def test_one_killed_actor_is_reported_while_others_run(self):
        """A dead TP rank must be reported even though the other ranks keep running."""
        actors, _, errors = self._start()
        ray.kill(actors[0])
        error = errors.get(timeout=_EXIT_TIMEOUT_S)
        self.assertIn("scheduler actor died", str(error))

    def test_crashed_event_loop_is_reported(self):
        actors, _, errors = self._start(num_actors=1, fail=True)
        ray.get(actors[0].stop.remote())
        error = errors.get(timeout=_EXIT_TIMEOUT_S)
        self.assertIn("scheduler actor died", str(error))
        self.assertIn("scheduler crashed", str(error))

    def test_returned_event_loop_is_reported(self):
        actors, _, errors = self._start(num_actors=1)
        ray.get(actors[0].stop.remote())
        error = errors.get(timeout=_EXIT_TIMEOUT_S)
        self.assertIn("event loop exited", str(error))

    def test_exit_after_stop_is_not_reported(self):
        """Engine shutdown kills the actors; that must not count as a crash."""
        actors, monitor, errors = self._start(num_actors=1)
        monitor.stop()
        ray.kill(actors[0])
        monitor._thread.join(timeout=_EXIT_TIMEOUT_S)
        self.assertFalse(monitor._thread.is_alive())
        self.assertTrue(errors.empty())


if __name__ == "__main__":
    unittest.main()
