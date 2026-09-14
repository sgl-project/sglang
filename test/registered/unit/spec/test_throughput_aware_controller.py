import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.throughput_aware_controller import (
    ThroughputAwareAdaptiveController,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class FakeWorker:
    def __init__(self):
        self.speculative_num_steps = 3
        self.model_config = SimpleNamespace(context_len=4096)
        self.build_calls = []

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
        self.build_calls.append((speculative_num_steps, cuda_graph_bs))
        return SpecRuntimeState(
            speculative_num_steps,
            speculative_num_draft_tokens,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def apply_runtime_state(self, state):
        self.speculative_num_steps = state.speculative_num_steps


class TestThroughputAwareController(unittest.TestCase):
    def make_controller(self, **settings):
        config = {
            "window_size": 2,
            "update_interval": 2,
            "1": {"candidate_steps": [1, 3]},
            "8": {"candidate_steps": [1]},
            **settings,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            controller = ThroughputAwareAdaptiveController(FakeWorker(), str(path))
        controller.init_states([1, 4, 8, 16])
        return controller

    def test_policy_interface_builds_pruned_runtime_states(self):
        controller = self.make_controller()
        self.assertIs(controller.params, controller)
        self.assertEqual(
            controller.worker.build_calls, [(1, [1, 4, 8, 16]), (3, [1, 4])]
        )
        self.assertEqual(controller.worker.speculative_num_steps, 3)

    def test_feedback_defers_switch_until_next_decode(self):
        controller = self.make_controller()
        controller._cost_table.set(1, 1, 1.0)
        controller._cost_table.set(1, 3, 10.0)
        controller.on_verify_complete([3, 3], 1)
        controller.activate_step_by_batch(1)
        self.assertEqual(controller.worker.speculative_num_steps, 3)
        controller.on_verify_complete([3, 3], 1)
        self.assertEqual(controller.worker.speculative_num_steps, 3)
        controller.activate_step_by_batch(1)
        self.assertEqual(controller.worker.speculative_num_steps, 1)
        self.assertEqual(
            controller._tracker.snapshot_position_rates(3), [1.0, 1.0, 1.0]
        )
        self.assertTrue(controller._tracker.is_position_extrapolated(1))

    def test_hysteresis_keeps_current_step(self):
        controller = self.make_controller()
        controller._cost_table.set(1, 1, 2 / 1.05)
        controller._cost_table.set(1, 3, 4.0)
        for _ in range(2):
            controller.on_verify_complete([3], 1)
        controller.activate_step_by_batch(1)
        self.assertEqual(controller.worker.speculative_num_steps, 3)

    def test_batch_change_selects_captured_state_before_tracker_warmup(self):
        controller = self.make_controller()
        # BS=5 pads to the captured BS=8, whose only allowed step is 1.
        controller.activate_step_by_batch(5)
        self.assertEqual(controller.worker.speculative_num_steps, 1)
        self.assertEqual(controller._batch_count, 0)

    def test_profile_grid_uses_resolved_buckets_and_request_limit(self):
        controller = self.make_controller(profile_run_batch_sizes=[1, 2, 4, 8, 16])
        self.assertEqual(controller._build_profile_grid(8), {1: [1, 4, 8], 3: [1, 4]})
        controller.set_cuda_graph_bs(None)
        self.assertEqual(controller._build_profile_grid(8), {})

    def test_profile_context_leaves_decode_headroom(self):
        controller = self.make_controller(profile_run_seq_len=4096)
        self.assertEqual(controller._resolve_profile_seq_len(), 4096 - 15 * 4 - 16)
        controller.worker.model_config.context_len = 32
        with self.assertRaisesRegex(ValueError, "headroom"):
            controller._resolve_profile_seq_len()

    def test_invalid_config_fails_before_runtime_state_building(self):
        for setting in (
            {"window_size": True},
            {"switch_hysteresis": -1},
            {"profile_run_n_measure": 0},
        ):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                self.make_controller(**setting)

    def test_profiling_populates_costs_and_restores_initial_step(self):
        controller = self.make_controller()
        session = SimpleNamespace(measure=lambda: 2.5)
        with patch(
            "sglang.srt.speculative.spec_profiling_session.SpecProfilingSession",
            return_value=session,
        ) as factory:
            controller.run_profiling(object(), max_running_requests=4)
        self.assertEqual(factory.call_count, 4)
        self.assertEqual(controller._cost_table.lookup(4, 3), 2.5)
        self.assertEqual(controller.worker.speculative_num_steps, 3)
        self.assertEqual(controller._batch_count, 0)


if __name__ == "__main__":
    unittest.main()
