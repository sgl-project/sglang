"""A NO_TOKEN caused by the buffer-only write-backlog hold must not latch
``batch_is_full`` until a request finishes: admission reopens as soon as the
hold shrinks (writes staged or swept), which is the recovery the throttle
promises."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerWriteBacklogLatch(CustomTestCase):
    def _scheduler(self, *, hold_tokens: int, hold_at_full: int) -> Scheduler:
        """Just enough of a Scheduler for the admission gate at the top of
        _get_new_batch_prefill_raw; an empty waiting queue returns right after
        the gate, so the flag's value afterwards is the gate's verdict."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        scheduler.chunked_req = None
        scheduler.waiting_queue = []
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.write_backlog_hold_tokens.return_value = hold_tokens
        scheduler._write_backlog_hold_at_full = hold_at_full
        return scheduler

    def _running_batch(self) -> ScheduleBatch:
        return ScheduleBatch(reqs=[MagicMock()], batch_is_full=True)

    def test_flag_clears_once_the_hold_shrinks(self):
        scheduler = self._scheduler(hold_tokens=30, hold_at_full=40)
        batch = self._running_batch()
        self.assertEqual(
            scheduler._get_new_batch_prefill_raw(None, batch), (None, batch)
        )
        self.assertFalse(batch.batch_is_full)
        self.assertEqual(scheduler._write_backlog_hold_at_full, 0)

    def test_flag_holds_while_the_hold_has_not_shrunk(self):
        for hold_tokens, hold_at_full in ((40, 40), (50, 40), (0, 0), (40, 0)):
            with self.subTest(hold_tokens=hold_tokens, hold_at_full=hold_at_full):
                scheduler = self._scheduler(
                    hold_tokens=hold_tokens, hold_at_full=hold_at_full
                )
                batch = self._running_batch()
                scheduler._get_new_batch_prefill_raw(None, batch)
                # Memory-bound NO_TOKENs (hold_at_full == 0) keep the stock
                # latch: only a finished or retracted request reopens them.
                self.assertTrue(batch.batch_is_full)
                self.assertEqual(scheduler._write_backlog_hold_at_full, hold_at_full)


if __name__ == "__main__":
    unittest.main()
