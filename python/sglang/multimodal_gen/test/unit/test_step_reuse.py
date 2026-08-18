# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the step-reuse contract (no GPU/torch required).

These tests exercise :mod:`sglang.multimodal_gen.runtime.cache.step_reuse`
directly with plain Python floats standing in for denoising predictions and
similarity observations, since the contract is intentionally agnostic to the
tensor library used by a real model adapter.
"""

import pytest

from sglang.multimodal_gen.runtime.cache.step_reuse import (
    FORCE_FIRST_ONE,
    FORCE_FIRST_TWO,
    FORCE_TERMINAL,
    StepReuseController,
    StepReuseError,
    StepReusePolicy,
    StepSideEffectContract,
)


def _threshold_decider(threshold):
    def decide_reuse(state, observation):
        previous = state.real_history[-1]
        return abs(observation - previous) < threshold

    return decide_reuse


def _make_controller(**policy_kwargs):
    defaults = dict(
        policy_name="test-policy",
        observation_point="post_cfg_velocity",
        history_size=4,
        max_skip_steps=2,
    )
    defaults.update(policy_kwargs)
    policy = StepReusePolicy(**defaults)
    return StepReuseController(policy, decide_reuse=_threshold_decider(0.05))


class TestStepReusePolicyValidation:
    def test_rejects_negative_history_size(self):
        with pytest.raises(ValueError):
            StepReusePolicy(
                policy_name="p",
                observation_point="x",
                history_size=-1,
                max_skip_steps=0,
            )

    def test_rejects_negative_max_skip_steps(self):
        with pytest.raises(ValueError):
            StepReusePolicy(
                policy_name="p",
                observation_point="x",
                history_size=0,
                max_skip_steps=-1,
            )

    def test_rejects_unknown_force_points(self):
        with pytest.raises(ValueError):
            StepReusePolicy(
                policy_name="p",
                observation_point="x",
                history_size=0,
                max_skip_steps=0,
                force_real_steps=frozenset({"bogus"}),
            )


class TestFirstRealForwardNeverSkips:
    def test_no_reuse_before_any_real_prediction(self):
        controller = _make_controller()
        scope = ("req-1",)
        assert controller.should_reuse(scope, 0, total_steps=5) is False

    def test_first_real_forward_cannot_open_skip_window(self):
        # Nothing in history yet, so decide_reuse must not even be consulted.
        controller = _make_controller()
        scope = ("req-1",)
        opened = controller.record_real(scope, prediction="p0", observation=1.0)
        assert opened is False
        assert controller.should_reuse(scope, 1, total_steps=5) is False


class TestSkipWindowLifecycle:
    def test_similar_observation_opens_skip_window_within_budget(self):
        controller = _make_controller(max_skip_steps=2)
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        opened = controller.record_real(scope, prediction="p1", observation=1.01)
        assert opened is True

        assert controller.should_reuse(scope, 2, total_steps=10) is True
        assert controller.get_reused_prediction(scope) == "p1"
        controller.record_reuse(scope)

        assert controller.should_reuse(scope, 3, total_steps=10) is True
        controller.record_reuse(scope)

        # Budget exhausted after max_skip_steps consecutive reuses.
        assert controller.should_reuse(scope, 4, total_steps=10) is False

    def test_dissimilar_observation_keeps_window_closed(self):
        controller = _make_controller(max_skip_steps=2)
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        opened = controller.record_real(scope, prediction="p1", observation=50.0)
        assert opened is False
        assert controller.should_reuse(scope, 2, total_steps=10) is False

    def test_record_reuse_without_budget_raises(self):
        controller = _make_controller()
        scope = ("req-1",)
        with pytest.raises(StepReuseError):
            controller.record_reuse(scope)


class TestReusedPredictionsNeverEnterHistory:
    def test_reuse_does_not_perturb_similarity_history(self):
        controller = _make_controller(max_skip_steps=3, history_size=4)
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        controller.record_real(scope, prediction="p1", observation=1.0)
        assert list(controller._state(scope).real_history) == [1.0, 1.0]

        controller.record_reuse(scope)
        controller.record_reuse(scope)
        # Reused steps must never be appended as new observations.
        assert list(controller._state(scope).real_history) == [1.0, 1.0]


class TestForcedRealSteps:
    def test_first_one_step_is_forced_real(self):
        controller = _make_controller(
            max_skip_steps=5, force_real_steps=frozenset({FORCE_FIRST_ONE})
        )
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        controller.record_real(scope, prediction="p1", observation=1.0)
        # Unlike "first_two", only step_index=0 is forced -- step_index=1
        # may already reuse once a skip window is open.
        assert controller.should_reuse(scope, 1, total_steps=10) is True

    def test_first_two_steps_are_forced_real(self):
        controller = _make_controller(
            max_skip_steps=5, force_real_steps=frozenset({FORCE_FIRST_TWO})
        )
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        controller.record_real(scope, prediction="p1", observation=1.0)
        # Even though a skip window is open, step_index=1 is within "first_two".
        assert controller.should_reuse(scope, 1, total_steps=10) is False
        assert controller.should_reuse(scope, 2, total_steps=10) is True

    def test_terminal_step_is_forced_real(self):
        controller = _make_controller(
            max_skip_steps=5, force_real_steps=frozenset({FORCE_TERMINAL})
        )
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        controller.record_real(scope, prediction="p1", observation=1.0)
        total_steps = 5
        assert controller.should_reuse(scope, total_steps - 1, total_steps) is False
        assert controller.should_reuse(scope, total_steps - 2, total_steps) is True

    def test_side_effect_contract_forces_terminal_write(self):
        controller = _make_controller(max_skip_steps=5)
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        controller.record_real(scope, prediction="p1", observation=1.0)
        side_effects = StepSideEffectContract(
            terminal_write_required=True, write_tags=frozenset({"kv_commit"})
        )
        total_steps = 4
        assert (
            controller.should_reuse(
                scope, total_steps - 1, total_steps, side_effects=side_effects
            )
            is False
        )
        # Non-terminal steps are unaffected by the side-effect contract.
        assert (
            controller.should_reuse(
                scope, total_steps - 2, total_steps, side_effects=side_effects
            )
            is True
        )


class TestIndependentScopes:
    def test_scopes_do_not_share_state(self):
        controller = _make_controller(max_skip_steps=2)
        cond_scope = ("req-1", "cfg_positive")
        uncond_scope = ("req-1", "cfg_negative")

        controller.record_real(cond_scope, prediction="pc0", observation=1.0)
        controller.record_real(cond_scope, prediction="pc1", observation=1.0)
        assert controller.should_reuse(cond_scope, 2, total_steps=10) is True
        assert controller.should_reuse(uncond_scope, 2, total_steps=10) is False

    def test_reset_clears_only_target_scope(self):
        controller = _make_controller(max_skip_steps=2)
        scope_a = ("req-a",)
        scope_b = ("req-b",)
        controller.record_real(scope_a, prediction="pa", observation=1.0)
        controller.record_real(scope_b, prediction="pb", observation=1.0)

        controller.reset(scope_a)
        assert controller.should_reuse(scope_a, 0, total_steps=5) is False
        with pytest.raises(StepReuseError):
            controller.get_reused_prediction(scope_a)
        # scope_b untouched.
        assert controller.get_reused_prediction(scope_b) == "pb"


class TestMetricsAndValidation:
    def test_metrics_track_real_and_reused_steps(self):
        controller = _make_controller(max_skip_steps=2)
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        controller.record_real(scope, prediction="p1", observation=1.0)
        controller.record_reuse(scope)

        metrics = controller.metrics(scope)
        assert metrics["real_forwards"] == 2
        assert metrics["reused_steps"] == 1
        assert metrics["total_steps_seen"] == 3

    def test_should_reuse_rejects_out_of_range_step_index(self):
        controller = _make_controller()
        scope = ("req-1",)
        with pytest.raises(StepReuseError):
            controller.should_reuse(scope, -1, total_steps=5)
        with pytest.raises(StepReuseError):
            controller.should_reuse(scope, 5, total_steps=5)

    def test_should_reuse_rejects_non_positive_total_steps(self):
        controller = _make_controller()
        scope = ("req-1",)
        with pytest.raises(StepReuseError):
            controller.should_reuse(scope, 0, total_steps=0)


class TestZeroSkipBudgetNeverOpensWindow:
    def test_max_skip_steps_zero_always_forces_real(self):
        controller = _make_controller(max_skip_steps=0)
        scope = ("req-1",)
        controller.record_real(scope, prediction="p0", observation=1.0)
        opened = controller.record_real(scope, prediction="p1", observation=1.0)
        assert opened is False
        assert controller.should_reuse(scope, 2, total_steps=10) is False
