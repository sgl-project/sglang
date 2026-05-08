import unittest

from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestAdaptiveSpeculativeParams(unittest.TestCase):
    def test_initial_steps_added_to_candidates_when_missing(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=2,
            config={"candidate_steps": [1, 3, 7]},
        )

        self.assertEqual(params.candidate_steps, [1, 2, 3, 7])
        self.assertEqual(params.current_steps, 2)
        self.assertEqual(params.ema_accept_len, 1.0)

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

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

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


if __name__ == "__main__":
    unittest.main()
