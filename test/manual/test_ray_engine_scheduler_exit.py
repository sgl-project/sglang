"""Tests for RayEngine's scheduler-exit monitor and health probe (requires Ray, no GPU).

Stand-in actors replace the scheduler actors: ``run_event_loop`` blocks, like
the real one, until the test ends it or kills the actor. A stand-in tokenizer
manager records what the engine forwards to it and answers, or ignores, probes.

Usage:
    python -m pytest test/manual/test_ray_engine_scheduler_exit.py -v
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import unittest
from types import SimpleNamespace

import ray

from sglang.srt.ray.engine import RayEngine
from sglang.srt.runtime_context import get_context
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


class TestRayEngineSchedulerExit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=4, include_dashboard=False)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def _start(
        self, *, num_actors: int = 2, fail: bool = False, handle_loop: bool = True
    ):
        # max_concurrency lets stop() run while run_event_loop blocks.
        actors = [
            _FakeSchedulerActor.options(max_concurrency=2).remote()
            for _ in range(num_actors)
        ]
        refs = [actor.run_event_loop.remote(fail=fail) for actor in actors]
        # Each message records whether handle_scheduler_exit ran on the handle loop.
        messages: queue.Queue = queue.Queue()
        loop = loop_thread = None
        if handle_loop:
            loop = asyncio.new_event_loop()
            loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
            loop_thread.start()
            self.addCleanup(loop.call_soon_threadsafe, loop.stop)
        engine = RayEngine.__new__(RayEngine)
        engine._scheduler_monitor_stopped = threading.Event()
        engine._scheduler_exit_error = None
        engine._scheduler_init_result = SimpleNamespace(event_loop_refs=refs)
        engine.tokenizer_manager = SimpleNamespace(
            event_loop=loop,
            scheduler_exit_error=None,
            handle_scheduler_exit=lambda message: messages.put(
                (threading.current_thread() is loop_thread, message)
            ),
        )
        monitor = threading.Thread(target=engine._monitor_schedulers, daemon=True)
        monitor.start()
        self.addCleanup(engine._scheduler_monitor_stopped.set)
        for actor in actors:
            self.addCleanup(ray.kill, actor)
        return actors, engine, monitor, messages

    def _assert_reported(self, engine, messages, expected: str):
        on_handle_loop, message = messages.get(timeout=_EXIT_TIMEOUT_S)
        self.assertTrue(on_handle_loop)
        self.assertIn(expected, message)
        self.assertIn(expected, engine.tokenizer_manager.scheduler_exit_error)
        with self.assertRaisesRegex(RuntimeError, expected):
            asyncio.run(engine.check_health())

    def test_one_killed_actor_is_reported_while_others_run(self):
        """A dead TP rank must be reported even though the other ranks keep running."""
        actors, engine, _, messages = self._start()
        ray.kill(actors[0])
        self._assert_reported(engine, messages, "scheduler actor died")

    def test_crashed_event_loop_is_reported(self):
        actors, engine, _, messages = self._start(num_actors=1, fail=True)
        ray.get(actors[0].stop.remote())
        self._assert_reported(engine, messages, "scheduler crashed")

    def test_returned_event_loop_is_reported(self):
        actors, engine, _, messages = self._start(num_actors=1)
        ray.get(actors[0].stop.remote())
        self._assert_reported(engine, messages, "event loop exited")

    def test_exit_before_any_request_sets_flag_without_handle_loop(self):
        """No request has started the handle loop, so there is nothing to fail;
        the flag alone makes the first request get 503."""
        actors, engine, monitor, messages = self._start(num_actors=1, handle_loop=False)
        ray.kill(actors[0])
        monitor.join(timeout=_EXIT_TIMEOUT_S)
        self.assertFalse(monitor.is_alive())
        self.assertTrue(messages.empty())
        self.assertIn(
            "scheduler actor died", engine.tokenizer_manager.scheduler_exit_error
        )
        with self.assertRaisesRegex(RuntimeError, "scheduler actor died"):
            asyncio.run(engine.check_health())

    def test_exit_after_shutdown_is_not_reported(self):
        """shutdown() kills the actors; that must not count as a crash."""
        actors, engine, monitor, messages = self._start(num_actors=1)
        engine._scheduler_monitor_stopped.set()
        ray.kill(actors[0])
        monitor.join(timeout=_EXIT_TIMEOUT_S)
        self.assertFalse(monitor.is_alive())
        self.assertTrue(messages.empty())
        self.assertIsNone(engine._scheduler_exit_error)
        self.assertIsNone(engine.tokenizer_manager.scheduler_exit_error)


class _ProbeTokenizerManager:
    """Answers a probe like a live scheduler, or never answers like a hung one."""

    is_generation = True

    def __init__(self, *, answers: bool):
        self.answers = answers
        self.last_receive_tstamp = time.time()
        self.rid_to_state = {}

    async def generate_request(self, obj):
        self.rid_to_state[obj.rid] = obj
        if self.answers:
            await asyncio.sleep(0.05)
            self.last_receive_tstamp = time.time()
            yield {}
        await asyncio.Event().wait()


class TestRayEngineHealthProbe(CustomTestCase):
    def _engine(self, *, answers: bool) -> RayEngine:
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        engine = RayEngine.__new__(RayEngine)
        engine._scheduler_exit_error = None
        engine.tokenizer_manager = _ProbeTokenizerManager(answers=answers)
        return engine

    def test_answered_probe_passes(self):
        engine = self._engine(answers=True)
        asyncio.run(engine.check_health(timeout_s=5))
        self.assertEqual(engine.tokenizer_manager.rid_to_state, {})

    def test_hung_scheduler_fails_probe(self):
        """A scheduler that is alive but stuck never answers; check_health must raise."""
        engine = self._engine(answers=False)
        with self.assertRaisesRegex(RuntimeError, "did not answer a health probe"):
            asyncio.run(engine.check_health(timeout_s=0.3))
        self.assertEqual(engine.tokenizer_manager.rid_to_state, {})

    def test_exit_during_probe_fails_it(self):
        """Output that arrives after a scheduler exit (sent before it, or by
        surviving ranks) must not pass a probe that was already waiting."""
        engine = self._engine(answers=False)

        async def run():
            async def exit_then_output():
                await asyncio.sleep(0.05)
                engine._scheduler_exit_error = RuntimeError(
                    "SGLang scheduler actor died"
                )
                engine.tokenizer_manager.last_receive_tstamp = time.time()

            exiting = asyncio.create_task(exit_then_output())
            try:
                await engine.check_health(timeout_s=5)
            finally:
                await exiting

        with self.assertRaisesRegex(RuntimeError, "scheduler actor died"):
            asyncio.run(run())
        self.assertEqual(engine.tokenizer_manager.rid_to_state, {})


if __name__ == "__main__":
    unittest.main()
