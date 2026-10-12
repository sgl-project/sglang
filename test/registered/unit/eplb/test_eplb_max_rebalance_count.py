import unittest
from unittest.mock import Mock, patch

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_NUM_ITERATIONS = 3


class TestEPLBMaxRebalanceCount(unittest.TestCase):
    def setUp(self):
        self.recorder = Mock()
        recorder_patch = patch(
            "sglang.srt.eplb.eplb_manager.get_global_expert_distribution_recorder",
            return_value=self.recorder,
        )
        recorder_patch.start()
        self.addCleanup(recorder_patch.stop)

    def _make_manager(self, *, max_count: int, outcomes: list[bool]):
        manager = EPLBManager.__new__(EPLBManager)
        manager._rebalance_num_iterations = _NUM_ITERATIONS
        manager._rebalance_max_count = max_count
        manager._recorder_stopped_by_freeze = False
        manager.rebalance_calls = 0
        outcomes_iter = iter(outcomes)

        def rebalance():
            manager.rebalance_calls += 1
            yield
            return next(outcomes_iter)

        manager.rebalance = rebalance
        manager.reset_generator()
        return manager

    def _run_windows(self, manager: EPLBManager, num_windows: int):
        # Each window is the counting passes plus the one pass rebalance() yields.
        for _ in range(num_windows * (_NUM_ITERATIONS + 1)):
            manager.on_forward_pass_end()

    def test_freezes_after_max_count_executed_rebalances(self):
        manager = self._make_manager(max_count=2, outcomes=[True, True])
        self._run_windows(manager, num_windows=10)
        self.assertEqual(manager.rebalance_calls, 2)
        self.recorder.stop_record.assert_called_once()

    def test_skipped_rebalances_do_not_count(self):
        manager = self._make_manager(max_count=2, outcomes=[False, True, False, True])
        self._run_windows(manager, num_windows=3)
        self.assertEqual(manager.rebalance_calls, 3)
        self.recorder.stop_record.assert_not_called()
        self._run_windows(manager, num_windows=10)
        self.assertEqual(manager.rebalance_calls, 4)
        self.recorder.stop_record.assert_called_once()

    def test_zero_means_no_limit(self):
        manager = self._make_manager(max_count=0, outcomes=[True] * 20)
        self._run_windows(manager, num_windows=20)
        self.assertEqual(manager.rebalance_calls, 20)
        self.recorder.stop_record.assert_not_called()

    def test_reset_after_freeze_restarts_recorder_and_budget(self):
        manager = self._make_manager(max_count=1, outcomes=[True, True])
        self._run_windows(manager, num_windows=3)
        self.assertEqual(manager.rebalance_calls, 1)
        self.recorder.start_record.assert_not_called()

        manager.reset_generator()
        self.recorder.start_record.assert_called_once()
        self._run_windows(manager, num_windows=3)
        self.assertEqual(manager.rebalance_calls, 2)
        self.assertEqual(self.recorder.stop_record.call_count, 2)


if __name__ == "__main__":
    unittest.main()
