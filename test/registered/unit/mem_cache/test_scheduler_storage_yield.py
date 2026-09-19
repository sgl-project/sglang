"""The scheduler must release the GIL on iterations that launch no batch.

A HiCache storage backend's transfers are driven by daemon threads in the
scheduler process -- the prefetch daemon, or a backend's own progress thread.
An iteration that ran a forward already yielded, because the launch releases
the GIL; an iteration that scheduled nothing spins pure Python, and CPython
will not preempt it for a thread it cannot see waiting.

The yield #38504 added covers one of the paths that can reach a no-batch
iteration. The overlap loops do not reach it -- ``on_idle`` is skipped there
while ``last_batch`` is still pending -- and neither does a rank with no idle
sleeper, which is every rank that is not pp0/attn-tp0/attn-cp0 plus all of
them when --sleep-on-idle is off (the default).

Measured on a storage backend fetching from a peer: with the loop quiet on the
peer's side the fetch missed 5 of 8 times and blew a 2s budget at 5.07s, while
the same fetch from a busy peer passed every time -- the forward was the only
thing releasing the GIL.

    python -m pytest test/registered/unit/mem_cache/test_scheduler_storage_yield.py -v
"""

from __future__ import annotations

import importlib
import inspect
import unittest
from unittest.mock import patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeScheduler:
    """Just enough scheduler to exercise the yield helper in isolation."""

    _yield_to_storage_threads = Scheduler._yield_to_storage_threads
    maybe_sleep_on_idle = Scheduler.maybe_sleep_on_idle

    def __init__(self, idle_sleeper=None, enable_hicache_storage=True):
        self.idle_sleeper = idle_sleeper
        self.enable_hicache_storage = enable_hicache_storage


class _RecordingSleeper:
    def __init__(self):
        self.calls = 0

    def maybe_sleep(self):
        self.calls += 1


class StorageYieldTest(unittest.TestCase):
    def test_no_storage_costs_nothing(self):
        """An engine with no L3 tier has no thread to yield to, and must not pay
        a syscall per no-batch iteration."""
        scheduler = _FakeScheduler(enable_hicache_storage=False)
        with patch("sglang.srt.managers.scheduler.time.sleep") as slept:
            scheduler._yield_to_storage_threads()
        slept.assert_not_called()

    def test_missing_sleeper_still_yields(self):
        """--sleep-on-idle defaults off, and init_idle_sleeper nulls the sleeper
        on every rank that is not pp0/attn-tp0/attn-cp0. Neither may be a hole."""
        scheduler = _FakeScheduler(idle_sleeper=None)
        with patch("sglang.srt.managers.scheduler.time.sleep") as slept:
            scheduler.maybe_sleep_on_idle()
        slept.assert_called_once_with(0)

    def test_sleeper_parks_instead_of_sleeping(self):
        """With a sleeper the park IS the yield; adding a sleep on top would
        double the idle latency for no gain."""
        sleeper = _RecordingSleeper()
        scheduler = _FakeScheduler(idle_sleeper=sleeper)
        with patch("sglang.srt.managers.scheduler.time.sleep") as slept:
            scheduler.maybe_sleep_on_idle()
        slept.assert_not_called()
        self.assertEqual(sleeper.calls, 1)


class EventLoopYieldCoverageTest(unittest.TestCase):
    """Every loop must yield on a no-batch iteration, including the one where
    ``on_idle`` is skipped because ``last_batch`` is still pending."""

    _OVERLAP_LOOPS = (
        ("sglang.srt.managers.scheduler", "Scheduler", "event_loop_overlap"),
        (
            "sglang.srt.disaggregation.decode",
            "SchedulerDisaggregationDecodeMixin",
            "event_loop_overlap_disagg_decode",
        ),
        (
            "sglang.srt.disaggregation.prefill",
            "SchedulerDisaggregationPrefillMixin",
            "event_loop_overlap_disagg_prefill",
        ),
    )

    def test_overlap_loops_yield_when_on_idle_is_skipped(self):
        for module_name, class_name, loop_name in self._OVERLAP_LOOPS:
            with self.subTest(loop=loop_name):
                module = importlib.import_module(module_name)
                owner = getattr(module, class_name)
                source = inspect.getsource(getattr(owner, loop_name))
                self.assertIn("_yield_to_storage_threads", source)


if __name__ == "__main__":
    unittest.main()
