"""Effort budgets must not erase live requests or mix completion populations."""

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler_components.reasoning_effort_budget import (
    HISTORY_SIZE,
    MAX_HISTORY_GROUPS,
    ReasoningEffortBudget,
    normalize_effort_hint,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def request(effort="medium", generated=0, maximum=1000, **overrides):
    return SimpleNamespace(
        **(
            dict(
                reasoning_effort_hint=effort,
                reasoning_budget_observed=False,
                reasoning_budget_weight_version="v1",
                lora_id=None,
                output_ids=array("q", [1]) * generated,
                sampling_params=SimpleNamespace(
                    max_new_tokens=maximum, ignore_eos=False
                ),
                finished_reason=SimpleNamespace(is_error=False),
            )
            | overrides
        )
    )


class TestReasoningEffortBudget(CustomTestCase):
    def test_disabled_fixed_length_and_unknown_effort_keep_their_contracts(self):
        for mode in ("off", "scaled", "adaptive"):
            budget = ReasoningEffortBudget(mode, initial=0.7, minimum=0.098)
            for ratio in (0.01, 0.098, 0.7, 1.0):
                self.assertEqual(budget.ratio_for(request(None), ratio), ratio)
                fixed = request("max")
                fixed.sampling_params.ignore_eos = True
                self.assertEqual(budget.ratio_for(fixed, ratio), 1.0)
                if mode == "off":
                    self.assertEqual(budget.ratio_for(request("max"), ratio), ratio)

    def test_tier_order_and_decay_speed(self):
        budget = ReasoningEffortBudget("scaled", initial=0.7, minimum=0.098)
        low, medium, high = (request(t) for t in ("low", "medium", "high"))
        before = [budget.ratio_for(r, 0.6) for r in (low, medium, high)]
        after = [budget.ratio_for(r, 0.5) for r in (low, medium, high)]
        self.assertLess(before[0], before[1])
        self.assertLess(before[1], before[2])
        drops = [a - b for a, b in zip(before, after)]
        self.assertGreater(drops[0], drops[1])
        self.assertGreater(drops[1], drops[2])
        self.assertEqual(budget.ratio_for(request(None), 0.2), 0.2)

    def test_history_changes_decay_and_overrun_recovers(self):
        short = ReasoningEffortBudget("adaptive", initial=0.7, minimum=0.098)
        long = ReasoningEffortBudget("adaptive", initial=0.7, minimum=0.098)
        for _ in range(8):
            short.observe(request(generated=200))
            long.observe(request(generated=900))
        running = request(generated=100)
        self.assertLess(short.ratio_for(running, 0.6), long.ratio_for(running, 0.6))
        self.assertEqual(short.ratio_for(request(generated=250), 0.2), 1.0)

    def test_completion_is_recorded_once_and_groups_are_isolated(self):
        budget = ReasoningEffortBudget("adaptive", initial=0.7, minimum=0.098)
        req = request("low", generated=200)
        for _ in range(8):
            budget.observe(req)
        # Replaying one completion cannot satisfy the eight-sample warmup.
        baseline = ReasoningEffortBudget("scaled", initial=0.7, minimum=0.098)
        self.assertEqual(
            budget.ratio_for(request("low"), 0.4),
            baseline.ratio_for(request("low"), 0.4),
        )
        for _ in range(7):
            budget.observe(request("low", generated=200))
        self.assertNotEqual(
            budget.ratio_for(request("low"), 0.4),
            baseline.ratio_for(request("low"), 0.4),
        )
        for req in (
            request("high"),
            request("low", lora_id="adapter"),
            request("low", reasoning_budget_weight_version="v2"),
        ):
            self.assertEqual(budget.ratio_for(req, 0.4), baseline.ratio_for(req, 0.4))

    def test_numeric_effort_and_invalid_hints(self):
        for value, tier in (
            (1, "low"),
            (50, "low"),
            (75, "high"),
            (100, "max"),
            (0.75, "high"),
            ("no_think", "none"),
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_effort_hint(value), tier)
        for value in (True, -1, 101, float("nan"), "unknown"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_effort_hint(value))

    def test_unfinished_aborted_and_forced_length_requests_do_not_train(self):
        budget = ReasoningEffortBudget("adaptive", initial=0.7, minimum=0.098)
        for reason in (None, FINISH_ABORT()):
            for _ in range(8):
                budget.observe(request(generated=100, finished_reason=reason))
        fixed = request(generated=100)
        fixed.sampling_params.ignore_eos = True
        budget.observe(fixed)
        self.assertFalse(budget.histories)

    def test_history_window_and_adapter_cardinality_are_bounded(self):
        budget = ReasoningEffortBudget("adaptive", initial=0.7, minimum=0.098)
        for _ in range(HISTORY_SIZE):
            budget.observe(request(generated=900))
        for _ in range(HISTORY_SIZE):
            budget.observe(request(generated=200))
        history = next(iter(budget.histories.values()))
        self.assertEqual(history.total / len(history.lengths), 200)
        # Loading many adapters must not retain unbounded retired populations.
        for index in range(MAX_HISTORY_GROUPS + 1):
            budget.observe(request(generated=100, lora_id=str(index)))
        self.assertEqual(len(budget.histories), MAX_HISTORY_GROUPS)
        self.assertNotIn(("v1", None, "medium"), budget.histories)

    def test_limits_above_the_estimation_cap_do_not_accelerate_decay(self):
        budget = ReasoningEffortBudget(
            "adaptive", initial=0.7, minimum=0.098, estimation_cap=4096
        )
        for _ in range(8):
            budget.observe(request(generated=5000, maximum=16000))
        first = budget.ratio_for(request(generated=100, maximum=16000), 0.6)
        second = budget.ratio_for(request(generated=100, maximum=32000), 0.6)
        self.assertEqual(first, second)
        self.assertGreater(first, 0.6)


if __name__ == "__main__":
    unittest.main()
