import unittest

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestAdaptiveSpeculativeParams(CustomTestCase):
    def test_initial_steps_snap_to_nearest_candidate_preferring_larger_step(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=2,
            config={"candidate_steps": [1, 3, 7]},
        )

        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)
        self.assertEqual(params.last_batch_accept_len, 0.0)
        self.assertEqual(params.last_batch_wasted_draft_ratio, 0.0)

    def test_update_respects_warmup_and_interval(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 2,
            },
        )

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.last_batch_wasted_draft_ratio, 1.0)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_empty_batches_do_not_consume_warmup_or_shift_steps(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 1,
            },
        )

        self.assertFalse(params.update([]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)
        self.assertEqual(params.last_batch_accept_len, 0.0)
        self.assertEqual(params.last_batch_wasted_draft_ratio, 0.0)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.last_batch_accept_len, 0.0)
        self.assertEqual(params.last_batch_wasted_draft_ratio, 1.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_update_scales_up_across_candidates(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=1,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_update_can_scale_down_across_candidates_in_one_recompute(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=7,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_exact_rise_threshold_does_not_upshift(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertFalse(params.update([2, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.5)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_exact_drop_threshold_does_downshift(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
                "up_hysteresis": 0.5,
            },
        )

        self.assertTrue(params.update([0, 1]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.5)

    def test_hysteresis_can_prevent_premature_upshift(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.75,
            },
        )

        self.assertFalse(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)

    def test_down_hysteresis_can_prevent_premature_downshift(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=7,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": -0.75,
            },
        )

        self.assertFalse(params.update([2, 2]))
        self.assertEqual(params.current_steps, 7)

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

    def test_multi_batch_sequence_can_ramp_up_then_back_down(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 0.5,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)
        self.assertEqual(params.ema_accept_len, 3.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 1.5)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 0.75)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.375)
        self.assertEqual(params.last_batch_accept_len, 0.0)
        self.assertEqual(params.last_batch_wasted_draft_ratio, 1.0)

    def test_wasted_draft_ratio_uses_active_tier_capacity(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            config={
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 1,
            },
        )

        self.assertFalse(params.update([1, 3], draft_capacity_per_req=3))
        self.assertAlmostEqual(params.last_batch_wasted_draft_ratio, 1 / 3)

    def test_wasted_draft_ratio_handles_zero_capacity(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=1,
            config={"candidate_steps": [1, 3, 7]},
        )

        self.assertFalse(params.update([0, 0], draft_capacity_per_req=0))
        self.assertEqual(params.last_batch_wasted_draft_ratio, 0.0)


class _FakeAdaptiveWorker:
    def __init__(self, initial_steps: int = 1):
        self.speculative_num_steps = initial_steps
        self.applied_steps = []
        self.built_steps = []

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState:
        self.built_steps.append(speculative_num_steps)
        return _make_runtime_state(
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        self.speculative_num_steps = state.speculative_num_steps
        self.applied_steps.append(state.speculative_num_steps)


def _make_runtime_state(
    speculative_num_steps: int,
    speculative_num_draft_tokens: int | None = None,
) -> SpecRuntimeState:
    if speculative_num_draft_tokens is None:
        speculative_num_draft_tokens = speculative_num_steps + 1
    return SpecRuntimeState(
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        draft_attn_backend=None,
        cuda_graph_runner=None,
        target_attn_backend=object(),
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
    )


class TestAdaptiveControllerMetrics(CustomTestCase):
    def test_metrics_snapshot_tracks_acceptance_and_switches(self):
        worker = _FakeAdaptiveWorker(initial_steps=1)
        controller = AdaptiveController(worker)
        controller.params.ema_alpha = 1.0
        controller.params.warmup_batches = 0
        controller.params.update_interval = 1
        controller.params.up_hysteresis = 0.0
        controller.register(_make_runtime_state(1))
        controller.init_states()

        self.assertEqual(worker.built_steps, [3, 7])
        self.assertEqual(worker.applied_steps, [1])
        self.assertEqual(
            controller.get_metrics(),
            {
                "enabled": 1,
                "current_steps": 1,
                "previous_steps": 0,
                "num_tier_switches": 0,
                "ema_accept_len": 0.0,
                "last_batch_accept_len": 0.0,
                "wasted_draft_ratio": 0.0,
            },
        )

        controller.on_verify_complete([1, 1])
        metrics = controller.get_metrics()

        self.assertEqual(worker.applied_steps, [1, 3])
        self.assertEqual(metrics["current_steps"], 3)
        self.assertEqual(metrics["previous_steps"], 1)
        self.assertEqual(metrics["num_tier_switches"], 1)
        self.assertEqual(metrics["ema_accept_len"], 1.0)
        self.assertEqual(metrics["last_batch_accept_len"], 1.0)
        self.assertEqual(metrics["wasted_draft_ratio"], 0.0)

    def test_initial_activation_is_not_counted_as_tier_switch(self):
        worker = _FakeAdaptiveWorker(initial_steps=3)
        controller = AdaptiveController(worker)
        controller.register(_make_runtime_state(3))
        controller.init_states()

        metrics = controller.get_metrics()
        self.assertEqual(metrics["enabled"], 1)
        self.assertEqual(metrics["current_steps"], 3)
        self.assertEqual(metrics["previous_steps"], 0)
        self.assertEqual(metrics["num_tier_switches"], 0)


if __name__ == "__main__":
    unittest.main()
