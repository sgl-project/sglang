"""The scheduler stall watchdog crashes a scheduler that holds requests but
launches no forward step for `SGLANG_STALL_WATCHDOG_TIMEOUT` seconds.

The forward watchdog only times a batch in flight; a waiting queue that never
yields a batch is the window this watchdog covers.
"""

import signal
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.invariant_checker import (  # noqa: E402
    _scheduler_has_unscheduled_work,
    create_scheduler_stall_watchdog,
)
from sglang.srt.utils import watchdog as watchdog_module  # noqa: E402
from sglang.srt.utils.watchdog import WatchdogRaw  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Stop(Exception):
    pass


class _FakeTime:
    """perf_counter/sleep pair; sleeping advances the clock instantly."""

    def __init__(self, limit: float = 10_000.0):
        self.now = 1000.0
        self.limit = limit

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.now > self.limit:
            raise _Stop()


def _make_watchdog(clock, **kwargs):
    """Build a WatchdogRaw without its thread; the test drives `_watchdog_once`."""
    parent = MagicMock()
    with (
        patch.object(watchdog_module.threading, "Thread"),
        patch.object(watchdog_module.psutil, "Process") as process,
    ):
        process.return_value.parent.return_value = parent
        wd = WatchdogRaw(debug_name="test", **kwargs)
    return wd, parent


def _run_once(wd, clock):
    with (
        patch.object(watchdog_module, "time", clock),
        patch.object(watchdog_module, "pyspy_dump_schedulers"),
    ):
        try:
            wd._watchdog_once()
        except _Stop:
            pass


class TestWatchdogResetOnActivate(CustomTestCase):
    def _fire_time(self, reset_on_activate: bool):
        clock = _FakeTime()
        detected = []
        wd, parent = _make_watchdog(
            clock,
            get_counter=lambda: 0,
            # Inactive for 100 s, then active with a frozen counter.
            is_active=lambda: clock.now >= 1100.0,
            watchdog_timeout=10.0,
            dump_info=lambda: detected.append(clock.now) or "",
            reset_on_activate=reset_on_activate,
            poll_interval=1.0,
        )
        _run_once(wd, clock)
        parent.send_signal.assert_called_once_with(signal.SIGQUIT)
        return detected[0]

    def test_reset_counts_only_active_time(self):
        fired_at = self._fire_time(reset_on_activate=True)
        self.assertGreaterEqual(fired_at, 1110.0)
        self.assertLess(fired_at, 1113.0)

    def test_without_reset_the_inactive_stretch_counts(self):
        # Existing behavior, kept for the forward watchdog: it fires on the
        # first active sample after a long inactive stretch.
        self.assertLess(self._fire_time(reset_on_activate=False), 1102.0)

    def test_progress_while_active_never_fires(self):
        clock = _FakeTime(limit=1500.0)
        wd, parent = _make_watchdog(
            clock,
            get_counter=lambda: int(clock.now),
            is_active=lambda: True,
            watchdog_timeout=10.0,
            reset_on_activate=True,
            poll_interval=1.0,
        )
        _run_once(wd, clock)
        parent.send_signal.assert_not_called()

    def test_default_poll_interval_is_half_the_timeout(self):
        wd, _ = _make_watchdog(
            _FakeTime(),
            get_counter=lambda: 0,
            is_active=lambda: False,
            watchdog_timeout=8.0,
        )
        self.assertEqual(wd.poll_interval, 4.0)


def _scheduler(**overrides):
    fields = dict(
        is_initializing=False,
        pause_state=SimpleNamespace(paused=False),
        cur_batch_for_debug=None,
        waiting_queue=[],
        chunked_req=None,
        running_batch=SimpleNamespace(is_empty=lambda: True, reqs=[]),
        forward_ct=7,
        watchdog_heartbeats=0,
        dump_requests_before_crash=MagicMock(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestUnscheduledWork(CustomTestCase):
    def test_active_only_when_requests_are_held_without_a_batch(self):
        non_empty = SimpleNamespace(is_empty=lambda: False, reqs=[object()])
        cases = [
            ("idle", {}, False),
            ("waiting", {"waiting_queue": [object()]}, True),
            ("running", {"running_batch": non_empty}, True),
            ("chunked", {"chunked_req": object()}, True),
            (
                "batch in flight",
                {"waiting_queue": [object()], "cur_batch_for_debug": object()},
                False,
            ),
            (
                "paused",
                {
                    "waiting_queue": [object()],
                    "pause_state": SimpleNamespace(paused=True),
                },
                False,
            ),
            (
                "initializing",
                {"waiting_queue": [object()], "is_initializing": True},
                False,
            ),
        ]
        for name, overrides, expected in cases:
            with self.subTest(name):
                self.assertEqual(
                    _scheduler_has_unscheduled_work(_scheduler(**overrides)),
                    expected,
                )


class TestStallWatchdogFactory(CustomTestCase):
    def _create(self, scheduler, timeout=60.0):
        with (
            patch.object(watchdog_module.threading, "Thread"),
            patch.object(watchdog_module.psutil, "Process"),
        ):
            return create_scheduler_stall_watchdog(scheduler, watchdog_timeout=timeout)

    def test_configuration(self):
        scheduler = _scheduler()
        wd = self._create(scheduler)
        self.assertTrue(wd.reset_on_activate)
        self.assertEqual(wd.poll_interval, 5.0)
        self.assertIs(wd.before_crash, scheduler.dump_requests_before_crash)
        self.assertEqual(wd.get_counter(), (7, 0))
        self.assertEqual(self._create(scheduler, timeout=8.0).poll_interval, 2.0)

    def test_is_active_fails_closed(self):
        # Missing queue fields make the check raise; it must report inactive.
        wd = self._create(
            SimpleNamespace(
                is_initializing=False, dump_requests_before_crash=MagicMock()
            )
        )
        self.assertFalse(wd.is_active())

    def test_dump_info_without_a_batch(self):
        checker = SimpleNamespace(
            _check_all_pools=lambda stats: (False, ["[full] total=8"])
        )
        scheduler = _scheduler(
            waiting_queue=[object(), object()],
            invariant_checker=checker,
            pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: None),
        )
        info = self._create(scheduler).dump_info()
        self.assertIn("waiting=2", info)
        self.assertIn("forward_ct=7", info)
        self.assertIn("[full] total=8", info)

    def test_dump_info_survives_pool_stats_failure(self):
        info = self._create(_scheduler()).dump_info()
        self.assertIn("pool stats unavailable", info)


class TestInitStallWatchdog(CustomTestCase):
    def _init(self, disagg_mode="null"):
        s = Scheduler.__new__(Scheduler)
        with (
            patch.object(
                scheduler_module,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=disagg_mode),
            ),
            patch.object(scheduler_module, "create_scheduler_stall_watchdog") as create,
        ):
            s.maybe_init_stall_watchdog()
        return s, create

    def test_disabled_by_default(self):
        with envs.SGLANG_STALL_WATCHDOG_TIMEOUT.override(-1):
            s, create = self._init()
        self.assertIsNone(s.stall_watchdog)
        create.assert_not_called()

    def test_armed_with_the_timeout(self):
        with envs.SGLANG_STALL_WATCHDOG_TIMEOUT.override(60):
            s, create = self._init()
        create.assert_called_once_with(s, watchdog_timeout=60.0)
        self.assertIs(s.stall_watchdog, create.return_value)

    def test_not_armed_under_pd_disaggregation(self):
        with envs.SGLANG_STALL_WATCHDOG_TIMEOUT.override(60):
            s, create = self._init(disagg_mode="prefill")
        self.assertIsNone(s.stall_watchdog)
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
