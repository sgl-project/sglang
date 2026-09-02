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


def _override_new_token_ratio_envs(
    *,
    init: float,
    min_factor: float,
    decay_steps: int,
):
    stack = ExitStack()
    stack.enter_context(envs.SGLANG_INIT_NEW_TOKEN_RATIO.override(init))
    stack.enter_context(envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.override(min_factor))
    stack.enter_context(envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.override(decay_steps))
    return stack


class TestNewTokenRatioTracker(CustomTestCase):
    def test_from_server_args_computes_initial_min_and_decay(self):
        server_args = SimpleNamespace(schedule_conservativeness=0.5)

        with _override_new_token_ratio_envs(init=0.8, min_factor=0.25, decay_steps=3):
            tracker = NewTokenRatioTracker.from_server_args(server_args)

        self.assertEqual(tracker.init, 0.4)
        self.assertEqual(tracker.min, 0.1)
        self.assertAlmostEqual(tracker.decay, 0.1)
        self.assertEqual(tracker.current, tracker.init)

    def test_from_server_args_clamps_init_and_min_to_one(self):
        server_args = SimpleNamespace(schedule_conservativeness=2.0)

        with _override_new_token_ratio_envs(init=0.8, min_factor=2.0, decay_steps=10):
            tracker = NewTokenRatioTracker.from_server_args(server_args)

        self.assertEqual(tracker.init, 1.0)
        self.assertEqual(tracker.min, 1.0)
        self.assertEqual(tracker.decay, 0.0)
        self.assertEqual(tracker.current, 1.0)

    def test_decay_step_does_not_go_below_min_and_reset_restores_init(self):
        tracker = NewTokenRatioTracker(init=0.6, min=0.2, decay=0.15, current=0.6)

        tracker.decay_step()
        tracker.decay_step()
        tracker.decay_step()
        tracker.decay_step()

        self.assertEqual(tracker.current, 0.2)

        tracker.reset()

        self.assertEqual(tracker.current, 0.6)

    def test_estimate_new_token_ratio_after_retract_uses_decode_steps(self):
        reqs = [
            SimpleNamespace(
                output_ids=[1, 2],
                sampling_params=SimpleNamespace(max_new_tokens=10),
            ),
            SimpleNamespace(
                output_ids=[3, 4, 5],
                sampling_params=SimpleNamespace(max_new_tokens=20),
            ),
        ]

        with envs.SGLANG_RETRACT_DECODE_STEPS.override(4):
            ratio = NewTokenRatioTracker.estimate_new_token_ratio_after_retract(reqs)

        self.assertAlmostEqual(ratio, 13 / 31)

    def test_estimate_new_token_ratio_after_retract_is_capped_at_one(self):
        reqs = [
            SimpleNamespace(
                output_ids=[1, 2, 3, 4, 5],
                sampling_params=SimpleNamespace(max_new_tokens=1),
            )
        ]

        with envs.SGLANG_RETRACT_DECODE_STEPS.override(20):
            ratio = NewTokenRatioTracker.estimate_new_token_ratio_after_retract(reqs)

        self.assertEqual(ratio, 1.0)

    def test_estimate_new_token_ratio_after_retract_handles_empty_reqs(self):
        ratio = NewTokenRatioTracker.estimate_new_token_ratio_after_retract([])

        self.assertEqual(ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
