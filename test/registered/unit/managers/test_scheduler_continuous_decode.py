import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestSchedulerContinuousDecode(CustomTestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.num_continuous_decode_steps = 4
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.chunked_req = None
        return scheduler

    def _make_batch(self, is_decode: bool, is_empty: bool = False):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: is_decode),
            is_empty=lambda: is_empty,
        )

    def test_can_run_continuous_decode_true_for_decode_batch(self):
        scheduler = self._new_scheduler()
        batch = self._make_batch(is_decode=True)

        self.assertTrue(scheduler._can_run_continuous_decode(batch))

    def test_can_run_continuous_decode_false_with_pending_waiting_queue(self):
        scheduler = self._new_scheduler()
        scheduler.waiting_queue = [object()]
        batch = self._make_batch(is_decode=True)

        self.assertFalse(scheduler._can_run_continuous_decode(batch))

    def test_can_run_continuous_decode_false_when_disabled(self):
        scheduler = self._new_scheduler()
        scheduler.num_continuous_decode_steps = 1
        batch = self._make_batch(is_decode=True)

        self.assertFalse(scheduler._can_run_continuous_decode(batch))

    def test_event_loop_normal_runs_multiple_continuous_decode_steps(self):
        class _LoopExit(Exception):
            pass

        scheduler = self._new_scheduler()
        scheduler.num_continuous_decode_steps = 4
        scheduler.self_check_during_idle = MagicMock()
        scheduler.self_check_during_busy = MagicMock()
        scheduler.cancel_bubble_timer = MagicMock()
        scheduler.process_input_requests = MagicMock()
        scheduler.running_batch = MagicMock()

        # First scheduler loop runs normally; second loop exits the infinite loop.
        scheduler.recv_requests = MagicMock(side_effect=[[], _LoopExit()])

        first_batch = self._make_batch(is_decode=True)
        next_batches = [
            self._make_batch(is_decode=True),
            self._make_batch(is_decode=True),
            self._make_batch(is_decode=True),
        ]
        scheduler.get_next_batch_to_run = MagicMock(return_value=first_batch)
        scheduler.update_running_batch = MagicMock(side_effect=next_batches)
        scheduler.run_batch = MagicMock(return_value=object())
        scheduler.process_batch_result = MagicMock()

        with self.assertRaises(_LoopExit):
            scheduler.event_loop_normal()

        # 1 initial decode + (num_continuous_decode_steps - 1) continuous decodes.
        self.assertEqual(scheduler.run_batch.call_count, 4)
        self.assertEqual(scheduler.process_batch_result.call_count, 4)
        self.assertEqual(scheduler.update_running_batch.call_count, 3)


if __name__ == "__main__":
    unittest.main()
