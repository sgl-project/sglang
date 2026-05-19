import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.speculative.adaptive_runtime_state import AdaptiveController
from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    adaptive_unsupported_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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


def _supported_server_args(**overrides):
    """Minimal SimpleNamespace with the attributes adaptive_unsupported_reason reads."""
    base = dict(
        speculative_algorithm="EAGLE",
        enable_dp_attention=False,
        disable_overlap_schedule=True,
        enable_multi_layer_eagle=False,
        enable_two_batch_overlap=False,
        enable_pdmux=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestAdaptiveUnsupportedReason(unittest.TestCase):
    def test_topk_greater_than_one_is_supported(self):
        # topk != 1 used to disable adaptive; tree EAGLE is now supported.
        for topk in (2, 4, 8):
            args = _supported_server_args(speculative_eagle_topk=topk)
            self.assertIsNone(adaptive_unsupported_reason(args))

    def test_non_eagle_algorithm_still_disables(self):
        args = _supported_server_args(speculative_algorithm="STANDALONE")
        self.assertIsNotNone(adaptive_unsupported_reason(args))


class _StubWorker:
    """Minimal AdaptiveSpecWorker implementer for controller-construction tests.

    Mirrors EAGLEWorker's pool / num_draft_tokens formulas so the controller
    sees realistic values without touching real GPU paths. build/apply are
    stubbed but must exist to satisfy the Protocol's structural shape.
    """

    def __init__(self, *, speculative_num_steps, speculative_num_draft_tokens, topk):
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.topk = topk

    def get_draft_pool_size(self, num_steps):
        if num_steps < 1:
            return 0
        return self.topk + (num_steps - 1) * self.topk * self.topk

    def get_num_draft_tokens(self, num_steps):
        if self.topk == 1:
            return num_steps + 1
        return self.speculative_num_draft_tokens

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens
    ):
        del speculative_num_steps, speculative_num_draft_tokens
        raise NotImplementedError

    def apply_runtime_state(self, state):
        del state
        raise NotImplementedError


class TestValidateCandidateSteps(unittest.TestCase):
    def _make_config(self, candidate_steps):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"candidate_steps": candidate_steps, "warmup_batches": 1}, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_topk_one_passes_validation(self):
        # server_args enforces num_draft_tokens = num_steps + 1 at topk=1,
        # so the per-tier budget always matches that tier's pool size.
        worker = _StubWorker(
            speculative_num_steps=1, speculative_num_draft_tokens=2, topk=1
        )
        cfg = self._make_config([1, 3])
        AdaptiveController(worker, config_path=cfg)  # must not raise

    def test_topk_one_default_candidate_steps_passes(self):
        # Regression: at topk=1 with the default candidate_steps=[1, 3, 7],
        # the smallest tier (s=1) must validate against its own
        # num_draft_tokens (=2 by the chain rule), not against the worker's
        # initial num_draft_tokens (which tracks the user's --speculative-num-steps
        # and could be much larger).
        worker = _StubWorker(
            speculative_num_steps=3, speculative_num_draft_tokens=4, topk=1
        )
        cfg = self._make_config([1, 3, 7])
        AdaptiveController(worker, config_path=cfg)  # must not raise

    def test_undersized_pool_raises(self):
        # topk=4, candidate_steps=[1, 3], num_draft_tokens=20
        # min_step=1 -> pool = 4; required >= 19 -> must raise.
        worker = _StubWorker(
            speculative_num_steps=1, speculative_num_draft_tokens=20, topk=4
        )
        cfg = self._make_config([1, 3])
        with self.assertRaisesRegex(ValueError, "draft pool of only 4"):
            AdaptiveController(worker, config_path=cfg)

    def test_sufficient_pool_accepts(self):
        # topk=4, candidate_steps=[2, 3], num_draft_tokens=20
        # min_step=2 -> pool = 4 + 1*16 = 20; required >= 19 -> OK.
        worker = _StubWorker(
            speculative_num_steps=3, speculative_num_draft_tokens=20, topk=4
        )
        cfg = self._make_config([2, 3])
        AdaptiveController(worker, config_path=cfg)  # must not raise

    def test_boundary_pool_equals_required_passes(self):
        # topk=4, candidate_steps=[2, 3], num_draft_tokens=21
        # min_step=2 -> pool = 20; required = 20 -> exactly at boundary, OK.
        worker = _StubWorker(
            speculative_num_steps=3, speculative_num_draft_tokens=21, topk=4
        )
        cfg = self._make_config([2, 3])
        AdaptiveController(worker, config_path=cfg)

    def test_off_by_one_above_boundary_raises(self):
        # topk=4, candidate_steps=[2, 3], num_draft_tokens=22
        # min_step=2 -> pool = 20; required = 21 -> fails by exactly 1.
        worker = _StubWorker(
            speculative_num_steps=3, speculative_num_draft_tokens=22, topk=4
        )
        cfg = self._make_config([2, 3])
        with self.assertRaisesRegex(ValueError, "draft pool of only 20"):
            AdaptiveController(worker, config_path=cfg)


if __name__ == "__main__":
    unittest.main()
