"""Unit tests for srt/managers/scheduler_components/new_token_ratio_tracker.py

Pure-logic tests for the new-token-ratio bookkeeping the scheduler uses to
estimate how many new tokens to reserve. Covers the arithmetic and clamping in
``from_server_args``, the decay floor in ``decay_step``, ``reset``, and the
zero-division-safe estimate produced after a retract.

No server / engine launch — env values are pinned via the ``envs`` field
``override`` context manager so the arithmetic is deterministic.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(num_output_ids: int, max_new_tokens: int) -> SimpleNamespace:
    """Minimal stand-in for a ``Req``.

    ``estimate_new_token_ratio_after_retract`` only reads ``len(r.output_ids)``
    and ``r.sampling_params.max_new_tokens``.
    """
    return SimpleNamespace(
        output_ids=[0] * num_output_ids,
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
    )


class TestNewTokenRatioTrackerFromServerArgs(CustomTestCase):
    """``from_server_args`` derives init / min / decay from envs + conservativeness."""

    def _build(self, *, conservativeness, init=0.7, factor=0.14, steps=600):
        server_args = SimpleNamespace(schedule_conservativeness=conservativeness)
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_INIT_NEW_TOKEN_RATIO.override(init))
            stack.enter_context(envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.override(factor))
            stack.enter_context(envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.override(steps))
            return NewTokenRatioTracker.from_server_args(server_args)

    def test_nominal_values(self):
        tracker = self._build(conservativeness=1.0)
        # init = min(0.7 * 1.0, 1.0) = 0.7
        self.assertAlmostEqual(tracker.init, 0.7)
        # min = min(0.7 * 0.14, 1.0) = 0.098
        self.assertAlmostEqual(tracker.min, 0.098)
        # decay = (0.7 - 0.098) / 600
        self.assertAlmostEqual(tracker.decay, (0.7 - 0.098) / 600)
        # current starts at init
        self.assertAlmostEqual(tracker.current, tracker.init)

    def test_conservativeness_scales_init(self):
        tracker = self._build(conservativeness=0.5)
        # init = min(0.7 * 0.5, 1.0) = 0.35
        self.assertAlmostEqual(tracker.init, 0.35)
        self.assertAlmostEqual(tracker.min, 0.35 * 0.14)

    def test_init_clamped_to_one(self):
        # 0.7 * 2.0 = 1.4 -> clamped to 1.0
        tracker = self._build(conservativeness=2.0)
        self.assertAlmostEqual(tracker.init, 1.0)
        self.assertAlmostEqual(tracker.current, 1.0)
        self.assertAlmostEqual(tracker.min, 0.14)

    def test_min_clamped_to_one(self):
        # init=0.7, factor=2.0 -> 0.7*2.0=1.4 -> min clamped to 1.0
        tracker = self._build(conservativeness=1.0, factor=2.0)
        self.assertAlmostEqual(tracker.init, 0.7)
        self.assertAlmostEqual(tracker.min, 1.0)
        # decay goes negative when the floor exceeds init
        self.assertAlmostEqual(tracker.decay, (0.7 - 1.0) / 600)


class TestNewTokenRatioTrackerDecayAndReset(CustomTestCase):
    """``decay_step`` walks ``current`` down toward ``min`` and stops there."""

    def test_decay_step_decreases_then_floors_at_min(self):
        tracker = NewTokenRatioTracker(init=0.7, min=0.1, decay=0.2, current=0.7)

        tracker.decay_step()
        self.assertAlmostEqual(tracker.current, 0.5)
        tracker.decay_step()
        self.assertAlmostEqual(tracker.current, 0.3)
        # 0.3 - 0.2 = 0.1 == min
        tracker.decay_step()
        self.assertAlmostEqual(tracker.current, 0.1)
        # Already at the floor: further steps clamp instead of going below min.
        tracker.decay_step()
        self.assertAlmostEqual(tracker.current, 0.1)

    def test_decay_step_clamps_when_current_already_below_min(self):
        # Defensive: current below min should snap up to min, never decrease.
        tracker = NewTokenRatioTracker(init=0.7, min=0.5, decay=0.2, current=0.4)
        tracker.decay_step()
        self.assertAlmostEqual(tracker.current, 0.5)

    def test_reset_restores_init(self):
        tracker = NewTokenRatioTracker(init=0.7, min=0.1, decay=0.2, current=0.7)
        tracker.decay_step()
        self.assertNotAlmostEqual(tracker.current, tracker.init)
        tracker.reset()
        self.assertAlmostEqual(tracker.current, 0.7)


class TestEstimateNewTokenRatioAfterRetract(CustomTestCase):
    """``estimate_new_token_ratio_after_retract`` is a static, env-driven estimate."""

    def test_empty_reqs_returns_zero(self):
        # (0 + retract * 0) / (0 + 1) == 0.0, and no ZeroDivisionError.
        with envs.SGLANG_RETRACT_DECODE_STEPS.override(20):
            self.assertEqual(
                NewTokenRatioTracker.estimate_new_token_ratio_after_retract([]),
                0.0,
            )

    def test_nominal_estimate(self):
        reqs = [_make_req(5, 100), _make_req(3, 50)]
        with envs.SGLANG_RETRACT_DECODE_STEPS.override(20):
            # (5 + 3 + 20 * 2) / (100 + 50 + 1) = 48 / 151
            self.assertAlmostEqual(
                NewTokenRatioTracker.estimate_new_token_ratio_after_retract(reqs),
                48 / 151,
            )

    def test_estimate_clamped_to_one(self):
        # Numerator dominates -> ratio would exceed 1.0, must clamp.
        reqs = [_make_req(200, 10)]
        with envs.SGLANG_RETRACT_DECODE_STEPS.override(20):
            # (200 + 20) / (10 + 1) = 20.0 -> clamped to 1.0
            self.assertEqual(
                NewTokenRatioTracker.estimate_new_token_ratio_after_retract(reqs),
                1.0,
            )

    def test_zero_max_new_tokens_does_not_divide_by_zero(self):
        # max_new_tokens == 0 and no output ids: denominator is +1 guarded.
        reqs = [_make_req(0, 0)]
        with envs.SGLANG_RETRACT_DECODE_STEPS.override(0):
            self.assertEqual(
                NewTokenRatioTracker.estimate_new_token_ratio_after_retract(reqs),
                0.0,
            )

    def test_retract_decode_steps_contributes_per_request(self):
        # Two reqs with no decoded tokens isolate the retract term: 2 * retract.
        reqs = [_make_req(0, 1000), _make_req(0, 1000)]
        with envs.SGLANG_RETRACT_DECODE_STEPS.override(20):
            # (0 + 20 * 2) / (2000 + 1) = 40 / 2001
            self.assertAlmostEqual(
                NewTokenRatioTracker.estimate_new_token_ratio_after_retract(reqs),
                40 / 2001,
            )


if __name__ == "__main__":
    unittest.main()
