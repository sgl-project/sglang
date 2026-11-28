"""The frontend stall watchdog exits the HTTP process when its event loop stops
running, or when user requests are in flight and the scheduler does not answer
a probe; it stays quiet while requests legitimately wait."""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # noqa: E402
from sglang.srt.managers import frontend_stall_watchdog as fsw  # noqa: E402
from sglang.srt.managers.frontend_stall_watchdog import (  # noqa: E402
    FrontendStallWatchdog,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus  # noqa: E402
from sglang.srt.utils.aio_rwlock import RWLock  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _state(age: float = 5.0):
    return SimpleNamespace(
        time_stats=SimpleNamespace(created_time=time.perf_counter() - age)
    )


class _FakeTokenizerManager:
    def __init__(self, rids=("user-1",)):
        self.server_status = ServerStatus.Up
        self.gracefully_exit = False
        self.is_pause = False
        self.model_update_lock = RWLock()
        self.rid_to_state = {rid: _state() for rid in rids}
        self.last_receive_tstamp = time.time()
        self._subprocess_watchdog = MagicMock()
        self.dump_requests_before_crash = MagicMock()


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _quiet_exit_steps():
    """Patch the steps that would touch real processes."""
    return (
        patch.object(fsw, "kill_process_tree"),
        patch.object(fsw, "pyspy_dump_schedulers"),
        patch.object(fsw.faulthandler, "dump_traceback"),
    )


class TestBlockedEventLoop(CustomTestCase):
    def _watchdog(self, tm, clock):
        return FrontendStallWatchdog(
            tokenizer_manager=tm,
            timeout=60.0,
            probe_fn=MagicMock(),
            exit_fn=MagicMock(),
            clock=clock,
        )

    def test_a_loop_silent_past_the_timeout_is_a_stall(self):
        clock = _Clock()
        wd = self._watchdog(_FakeTokenizerManager(), clock)
        clock.now += 59
        self.assertIsNone(wd._check())
        clock.now += 2
        self.assertIn("event loop did not run", wd._check())

    def test_startup_and_drain_are_not_stalls(self):
        """Warmup and a SIGTERM drain may keep the loop busy; neither is a hang."""
        for name, change in [
            ("starting", {"server_status": ServerStatus.Starting}),
            ("draining", {"gracefully_exit": True}),
        ]:
            with self.subTest(name):
                clock = _Clock()
                tm = _FakeTokenizerManager()
                for key, value in change.items():
                    setattr(tm, key, value)
                wd = self._watchdog(tm, clock)
                clock.now += 600
                self.assertIsNone(wd._check())

    def test_unhealthy_status_does_not_disarm_the_check(self):
        """/health marks the server UnHealthy on the very stalls this watchdog
        exists for; that state must not turn the watchdog off."""
        clock = _Clock()
        tm = _FakeTokenizerManager()
        tm.server_status = ServerStatus.UnHealthy
        wd = self._watchdog(tm, clock)
        clock.now += 61
        self.assertIsNotNone(wd._check())


class TestFire(CustomTestCase):
    def test_fire_dumps_kills_children_and_exits_nonzero(self):
        tm = _FakeTokenizerManager()
        exit_fn = MagicMock()
        wd = FrontendStallWatchdog(
            tokenizer_manager=tm, timeout=60.0, probe_fn=MagicMock(), exit_fn=exit_fn
        )
        kill, pyspy, dump = _quiet_exit_steps()
        with kill as kill_tree, pyspy, dump:
            wd._fire("test")
        exit_fn.assert_called_once_with(1)
        self.assertEqual(tm.server_status, ServerStatus.UnHealthy)
        tm.dump_requests_before_crash.assert_called_once_with()
        tm._subprocess_watchdog.stop.assert_called_once_with()
        _, kwargs = kill_tree.call_args
        # The process itself exits through exit_fn; killing it here would be a
        # no-op for PID 1 and would skip the non-zero exit code.
        self.assertFalse(kwargs["include_parent"])

    def test_exit_still_happens_when_diagnostics_hang_or_fail(self):
        tm = _FakeTokenizerManager()
        tm.dump_requests_before_crash.side_effect = RuntimeError("disk full")
        exit_fn = MagicMock()
        wd = FrontendStallWatchdog(
            tokenizer_manager=tm,
            timeout=60.0,
            probe_fn=MagicMock(),
            exit_fn=exit_fn,
            diagnostics_timeout=0.1,
        )
        never = threading.Event()
        with (
            patch.object(fsw, "pyspy_dump_schedulers", lambda **_: never.wait()),
            patch.object(fsw, "kill_process_tree", side_effect=RuntimeError("reap")),
            patch.object(fsw.faulthandler, "dump_traceback"),
        ):
            wd._fire("test")
        never.set()
        exit_fn.assert_called_once_with(1)


class TestProbe(CustomTestCase):
    def _run(self, tm, probe_result, seconds=0.3, on_probe=None):
        """Run the watchdog's loop tasks and monitor thread for `seconds`."""
        calls = []
        exited = threading.Event()

        async def probe_fn(timeout):
            calls.append(timeout)
            if on_probe is not None:
                on_probe(tm)
            return probe_result

        async def main():
            wd = FrontendStallWatchdog(
                tokenizer_manager=tm,
                timeout=60.0,
                probe_fn=probe_fn,
                probe_interval=0.01,
                heartbeat_interval=0.01,
                monitor_interval=0.01,
                exit_fn=lambda code: exited.set(),
            )
            wd.start()
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and not exited.is_set():
                await asyncio.sleep(0.01)
            wd.stop()
            return wd

        kill, pyspy, dump = _quiet_exit_steps()
        with kill, pyspy, dump:
            # main() returns once the monitor thread has exited through exit_fn.
            wd = asyncio.run(main())
        return calls, exited.is_set(), wd

    def test_unanswered_probe_with_requests_in_flight_exits(self):
        calls, exited, _ = self._run(_FakeTokenizerManager(), probe_result=False)
        self.assertTrue(calls)
        self.assertEqual(calls[0], 60.0)
        self.assertTrue(exited)

    def test_answered_probe_does_not_exit(self):
        calls, exited, wd = self._run(_FakeTokenizerManager(), probe_result=True)
        self.assertTrue(calls)
        self.assertFalse(exited)
        self.assertIsNone(wd._failure)

    def test_no_probe_while_requests_legitimately_wait(self):
        """Paused, draining, mid weight update, starting, or idle: a probe would
        wait at the same gate as the requests and read as a stall."""
        writer_lock = RWLock()
        writer_lock._writer_active = True
        cases = {
            "idle": dict(rids=()),
            "only health probes": dict(rids=(f"{HEALTH_CHECK_RID_PREFIX}_x",)),
            "paused": dict(is_pause=True),
            "draining": dict(gracefully_exit=True),
            "weight update": dict(model_update_lock=writer_lock),
            "starting": dict(server_status=ServerStatus.Starting),
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                tm = _FakeTokenizerManager(rids=overrides.pop("rids", ("user-1",)))
                for key, value in overrides.items():
                    setattr(tm, key, value)
                calls, exited, _ = self._run(tm, probe_result=False, seconds=0.1)
                self.assertEqual(calls, [])
                self.assertFalse(exited)

    def test_pause_that_begins_mid_probe_is_not_a_stall(self):
        def start_pause(tm):
            tm.is_pause = True

        calls, exited, wd = self._run(
            _FakeTokenizerManager(), probe_result=False, on_probe=start_pause
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(exited)
        self.assertIsNone(wd._failure)


if __name__ == "__main__":
    unittest.main()
