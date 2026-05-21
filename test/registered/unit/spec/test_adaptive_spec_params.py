import json
import tempfile
import unittest

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    build_per_bs_params,
    resolve_candidate_steps_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestAdaptiveSpeculativeParams(unittest.TestCase):
    def _make_params_from_config(self, initial_steps: int, config: dict):
        """Create params from a dict (the per-slot config passed by _init_per_bs)."""
        return AdaptiveSpeculativeParams(initial_steps=initial_steps, bs_cfg=config)

    def test_params_from_dict(self):
        params = self._make_params_from_config(
            5,
            {
                "candidate_steps": [1, 5],
                "ema_alpha": 0.75,
                "warmup_batches": 2,
            },
        )
        self.assertEqual(params.candidate_steps, [1, 5])
        self.assertEqual(params.ema_alpha, 0.75)
        self.assertEqual(params.warmup_batches, 2)

    def test_params_loads_config_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "candidate_steps": [1, 5],
                    "ema_alpha": 0.75,
                    "warmup_batches": 2,
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=5, bs_cfg=f.name)

        self.assertEqual(params.candidate_steps, [1, 5])
        self.assertEqual(params.ema_alpha, 0.75)
        self.assertEqual(params.warmup_batches, 2)

    def test_initial_steps_snaps_to_middle_when_missing(self):
        params = self._make_params_from_config(2, {"candidate_steps": [1, 3, 7]})

        self.assertEqual(params.candidate_steps, [1, 3, 7])
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

    def test_update_respects_warmup_and_interval(self):
        params = self._make_params_from_config(
            3,
            {
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
        params = self._make_params_from_config(
            3,
            {
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
        params = self._make_params_from_config(
            1,
            {
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
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_exact_rise_threshold_does_not_upshift(self):
        params = self._make_params_from_config(
            3,
            {
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
        params = self._make_params_from_config(
            3,
            {
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
        params = self._make_params_from_config(
            3,
            {
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
        params = self._make_params_from_config(
            7,
            {
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
        params = self._make_params_from_config(
            3,
            {
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

    def test_ceiling_coeff_caps_steps(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "ceiling_coeff": 1.0,
            },
        )
        # Force low ema to trigger ceiling
        params.ema_accept_len = 1.0
        self.assertTrue(params.update([1, 1]))
        # ceiling = ceil(1.0 * 1.0) = 1, target capped to 1
        self.assertEqual(params.current_steps, 1)

    def test_ceiling_disabled_by_default(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )
        self.assertEqual(params.ceiling_coeff, 0)


class TestBuildPerBsParams(unittest.TestCase):
    def test_default_config_loads(self):
        bs_list, bs_params = build_per_bs_params()
        self.assertEqual(bs_list, [1, 8])
        self.assertEqual(bs_params[1].candidate_steps, [1, 3, 7])
        self.assertEqual(bs_params[8].candidate_steps, [1])

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5], "up_hysteresis": 0.3},
                    "32": {"candidate_steps": [1, 2]},
                },
                f,
            )
            f.flush()
            bs_list, bs_params = build_per_bs_params(f.name)
        self.assertEqual(bs_list, [1, 32])
        self.assertEqual(bs_params[1].candidate_steps, [1, 5])
        self.assertEqual(bs_params[1].up_hysteresis, 0.3)
        self.assertEqual(bs_params[32].candidate_steps, [1, 2])

    def test_invalid_config_falls_back_to_default(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"not_a_bs": "bad"}, f)
            f.flush()
            bs_list, bs_params = build_per_bs_params(f.name)
        self.assertEqual(bs_list, [1, 8])

    def test_invalid_steps_falls_back_to_default(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": "bad"}}, f)
            f.flush()
            bs_list, bs_params = build_per_bs_params(f.name)
        self.assertEqual(bs_list, [1, 8])

    def test_global_hysteresis_inherited(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3]},
                },
                f,
            )
            f.flush()
            _, bs_params = build_per_bs_params(f.name)
        self.assertEqual(bs_params[1].up_hysteresis, 0.5)

    def test_entry_hysteresis_overrides_global(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3], "up_hysteresis": 0.1},
                },
                f,
            )
            f.flush()
            _, bs_params = build_per_bs_params(f.name)
        self.assertEqual(bs_params[1].up_hysteresis, 0.1)

    def test_ceiling_coeff_passthrough(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {"1": {"candidate_steps": [1, 3], "ceiling_coeff": 2.5}},
                f,
            )
            f.flush()
            _, bs_params = build_per_bs_params(f.name)
        self.assertEqual(bs_params[1].ceiling_coeff, 2.5)


class TestResolveCandidateSteps(unittest.TestCase):
    def test_default_config(self):
        steps = resolve_candidate_steps_from_config(initial_steps=3)
        self.assertIn(1, steps)
        self.assertIn(3, steps)
        self.assertIn(7, steps)

    def test_initial_steps_always_included(self):
        steps = resolve_candidate_steps_from_config(initial_steps=99)
        self.assertIn(99, steps)

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": [2, 4]}}, f)
            f.flush()
            steps = resolve_candidate_steps_from_config(
                initial_steps=3, cfg_path=f.name
            )
        self.assertEqual(steps, [2, 3, 4])

    def test_invalid_config_falls_back(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"bad": "config"}, f)
            f.flush()
            steps = resolve_candidate_steps_from_config(
                initial_steps=3, cfg_path=f.name
            )
        # Falls back to default, which has steps [1,3,7] + initial_steps=3
        self.assertIn(1, steps)
        self.assertIn(7, steps)

    def test_empty_steps_falls_back(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": []}}, f)
            f.flush()
            steps = resolve_candidate_steps_from_config(
                initial_steps=3, cfg_path=f.name
            )
        # Falls back to default
        self.assertIn(1, steps)
        self.assertIn(7, steps)

    def test_zero_steps_falls_back(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": [0]}}, f)
            f.flush()
            steps = resolve_candidate_steps_from_config(
                initial_steps=3, cfg_path=f.name
            )
        # Falls back to default (step=0 not supported yet)
        self.assertIn(1, steps)
        self.assertIn(7, steps)


if __name__ == "__main__":
    unittest.main()
