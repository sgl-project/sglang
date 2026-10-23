"""CPU regression tests for frontend choice logprob normalization."""

import unittest

from sglang.lang.backend.runtime_endpoint import compute_normalized_prompt_logprobs
from sglang.lang.choices import token_length_normalized
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNormalizedPromptLogprobs(CustomTestCase):
    def test_all_zero_logprobs(self):
        self.assertEqual(
            compute_normalized_prompt_logprobs([[0.0, 1, "a"], [0.0, 2, "b"]]),
            0.0,
        )

    def test_none_is_excluded_but_zero_is_counted(self):
        self.assertEqual(
            compute_normalized_prompt_logprobs(
                [[None, 0, "prefix"], [0.0, 1, "a"], [-2.0, 2, "b"]]
            ),
            -1.0,
        )

    def test_negative_logprobs_are_unchanged(self):
        self.assertEqual(
            compute_normalized_prompt_logprobs(
                [[None, 0, "prefix"], [-1.0, 1, "a"], [-3.0, 2, "b"]]
            ),
            -2.0,
        )

    def test_zero_logprob_contributes_to_choice_ranking(self):
        logprobs = [
            [[None, 0, "prefix"], [0.0, 1, "a"], [-2.0, 2, "b"]],
            [[-1.5, 3, "c"], [-1.5, 4, "d"]],
        ]
        decision = token_length_normalized(
            choices=["A", "B"],
            normalized_prompt_logprobs=[
                compute_normalized_prompt_logprobs(tokens) for tokens in logprobs
            ],
            input_token_logprobs=logprobs,
            output_token_logprobs=[[], []],
            unconditional_token_logprobs=None,
        )
        self.assertEqual(decision.decision, "A")
        self.assertEqual(decision.meta_info["normalized_prompt_logprobs"], [-1.0, -1.5])


if __name__ == "__main__":
    unittest.main()
