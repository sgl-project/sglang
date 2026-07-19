import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestWarBarrierPosition(unittest.TestCase):
    def test_overlap_loop_applies_war_barrier_after_each_launch(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.gracefully_exit = False
        scheduler.request_receiver = MagicMock()
        scheduler.request_receiver.recv_requests.side_effect = [[], [], StopIteration]
        scheduler.process_input_requests = MagicMock()
        scheduler._engine_paused = False
        scheduler.running_batch = MagicMock(name="initial_running_batch")
        first_batch = MagicMock(name="first_batch")
        second_batch = MagicMock(name="second_batch")
        first_result = MagicMock(name="first_result")
        second_result = MagicMock(name="second_result")
        scheduler.get_next_batch_to_run = MagicMock(
            side_effect=[
                SimpleNamespace(
                    running_batch=MagicMock(name="first_running_batch"),
                    batch_to_run=first_batch,
                ),
                SimpleNamespace(
                    running_batch=MagicMock(name="second_running_batch"),
                    batch_to_run=second_batch,
                ),
            ]
        )
        scheduler.is_disable_overlap_for_batch = MagicMock(return_value=False)
        calls = MagicMock()
        scheduler._apply_war_barrier = calls.war_barrier
        scheduler.run_batch = calls.run_batch
        scheduler.run_batch.side_effect = [first_result, second_result]
        scheduler.process_batch_result = calls.process_result
        scheduler.is_generation = False
        scheduler.last_batch = None

        with self.assertRaises(StopIteration):
            scheduler.event_loop_overlap()

        # The WAR barrier must be applied right after each launch, before the
        # previous result is processed.
        self.assertEqual(
            calls.mock_calls,
            [
                unittest.mock.call.run_batch(first_batch),
                unittest.mock.call.war_barrier(),
                unittest.mock.call.run_batch(second_batch),
                unittest.mock.call.war_barrier(),
                unittest.mock.call.process_result(
                    first_batch.copy.return_value, first_result
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
