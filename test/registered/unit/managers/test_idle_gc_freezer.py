"""The idle GC freezer collects, then freezes, once per idle period, only after
the period has lasted the configured time; a cache reset collects and refreezes
at once."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.srt.managers.scheduler_components import idle_gc_freezer
from sglang.srt.managers.scheduler_components.idle_gc_freezer import IdleGCFreezer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestIdleGCFreezer(unittest.TestCase):
    def test_freezes_once_per_idle_period(self):
        now = [100.0]
        gc_ops = Mock()
        freezer = IdleGCFreezer(idle_secs=5.0)
        with (
            patch.object(
                idle_gc_freezer, "time", SimpleNamespace(monotonic=lambda: now[0])
            ),
            patch.object(
                idle_gc_freezer,
                "gc",
                SimpleNamespace(collect=gc_ops.collect, unfreeze=gc_ops.unfreeze),
            ),
            patch.object(idle_gc_freezer, "freeze_gc", gc_ops.freeze_gc),
        ):
            freezer.on_idle(forward_ct=7)
            now[0] += 4.9
            freezer.on_idle(forward_ct=7)
            self.assertEqual(gc_ops.mock_calls, [])

            now[0] += 0.1
            for _ in range(3):
                freezer.on_idle(forward_ct=7)
            once = [call.collect(), call.freeze_gc("Scheduler")]
            self.assertEqual(gc_ops.mock_calls, once)

            # A launched batch starts a new idle period.
            freezer.on_idle(forward_ct=8)
            now[0] += 5.0
            freezer.on_idle(forward_ct=8)
            self.assertEqual(gc_ops.mock_calls, once + once)

            # A cache reset collects the dropped tree and refreezes at once.
            freezer.on_cache_reset()
            now[0] += 5.0
            freezer.on_idle(forward_ct=8)
            self.assertEqual(gc_ops.mock_calls, once + once + [call.unfreeze()] + once)


if __name__ == "__main__":
    unittest.main()
