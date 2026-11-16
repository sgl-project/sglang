"""Cadence of EPLBManager rebalances with --eplb-first-rebalance-num-iterations."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import Mock, patch

from sglang.srt.eplb import eplb_manager as eplb_manager_module
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.test.test_utils import CustomTestCase


def _make_manager(
    *,
    rebalance_num_iterations: int,
    first_rebalance_num_iterations: Optional[int],
) -> EPLBManager:
    moe = SimpleNamespace(
        eplb_rebalance_num_iterations=rebalance_num_iterations,
        eplb_first_rebalance_num_iterations=first_rebalance_num_iterations,
        eplb_rebalance_layers_per_chunk=None,
        expert_distribution_recorder_buffer_size=rebalance_num_iterations,
    )
    recorder = Mock(recording=True)
    with (
        patch.object(
            eplb_manager_module, "get_exec", return_value=SimpleNamespace(moe=moe)
        ),
        patch.object(
            eplb_manager_module,
            "get_global_expert_distribution_recorder",
            return_value=recorder,
        ),
    ):
        manager = EPLBManager(
            model_config=Mock(),
            get_model=Mock(),
            get_expert_location_updater=Mock(),
            get_expert_backup_client=Mock(),
            get_weight_updater=Mock(),
        )
    return manager


def _rebalance_passes(manager: EPLBManager, num_passes: int) -> List[int]:
    """Reset the generator, run num_passes passes, return the 1-based passes that rebalance."""
    starts: List[int] = []
    current_pass = [0]

    def fake_rebalance():
        starts.append(current_pass[0])
        return
        yield

    manager.rebalance = fake_rebalance
    manager.reset_generator()
    for i in range(1, num_passes + 1):
        current_pass[0] = i
        manager.on_forward_pass_end()
    return starts


class TestEPLBFirstRebalance(CustomTestCase):
    def test_unset_keeps_regular_cadence(self):
        manager = _make_manager(
            rebalance_num_iterations=10, first_rebalance_num_iterations=None
        )
        self.assertEqual(_rebalance_passes(manager, 45), [11, 21, 31, 41])

    def test_first_rebalance_moves_only_the_first(self):
        manager = _make_manager(
            rebalance_num_iterations=10, first_rebalance_num_iterations=3
        )
        self.assertEqual(_rebalance_passes(manager, 45), [4, 14, 24, 34, 44])

    def test_generator_reset_after_first_window_uses_regular_cadence(self):
        manager = _make_manager(
            rebalance_num_iterations=10, first_rebalance_num_iterations=3
        )
        self.assertEqual(_rebalance_passes(manager, 5), [4])
        # Each call resets the generator, as disable/enable_rebalance and elastic EP do.
        self.assertEqual(_rebalance_passes(manager, 25), [11, 21])

    def test_reset_before_first_window_keeps_first_delay(self):
        manager = _make_manager(
            rebalance_num_iterations=10, first_rebalance_num_iterations=3
        )
        self.assertEqual(_rebalance_passes(manager, 2), [])
        self.assertEqual(_rebalance_passes(manager, 15), [4, 14])


if __name__ == "__main__":
    unittest.main()
