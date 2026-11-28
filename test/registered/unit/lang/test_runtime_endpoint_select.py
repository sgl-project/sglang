"""Unit tests for RuntimeEndpoint.select logprob handling, run against a stubbed
server response (no server, no model weights)."""

import unittest
from unittest.mock import patch

from sglang.lang.backend.runtime_endpoint import (
    RuntimeEndpoint,
    compute_normalized_prompt_logprobs,
)
from sglang.lang.choices import token_length_normalized
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _tokens(*logprobs):
    """Build input_token_logprobs entries: [logprob, token_id, token_text]."""
    return [[lp, i, f"t{i}"] for i, lp in enumerate(logprobs)]


class TestComputeNormalizedPromptLogprobs(CustomTestCase):
    def test_zero_logprob_is_counted(self):
        # A token with probability 1 has logprob 0.0. It is a real value and must
        # stay in the mean: (-0.5 + 0.0 + 0.0) / 3, not -0.5 / 1.
        result = compute_normalized_prompt_logprobs(_tokens(-0.5, 0.0, 0.0))
        self.assertAlmostEqual(result, -0.5 / 3)

    def test_negative_zero_logprob_is_counted(self):
        result = compute_normalized_prompt_logprobs(_tokens(-0.3, -0.0))
        self.assertAlmostEqual(result, -0.15)

    def test_missing_first_logprob_is_skipped(self):
        # The first prompt token has no logprob (None) and is not part of the mean.
        result = compute_normalized_prompt_logprobs(_tokens(None, -0.2, -0.4))
        self.assertAlmostEqual(result, -0.3)

    def test_single_zero_logprob_does_not_divide_by_zero(self):
        self.assertEqual(compute_normalized_prompt_logprobs(_tokens(0.0)), 0.0)


class TestSelectWithZeroLogprob(CustomTestCase):
    def test_select_prefers_the_choice_with_the_higher_mean(self):
        """Choice "a" has the higher mean logprob, -1/6 against -0.3. Dropping its
        two 0.0 entries used to score it -0.5 and make select pick "b"."""
        endpoint = RuntimeEndpoint.__new__(RuntimeEndpoint)
        state = type("State", (), {"text_": "prompt"})()

        def reply(input_token_logprobs):
            return {
                "meta_info": {
                    "input_token_logprobs": input_token_logprobs,
                    "output_token_logprobs": [],
                }
            }

        responses = [
            {"meta_info": {"prompt_tokens": 4}},
            [
                reply(_tokens(-0.5, 0.0, 0.0)),
                reply(_tokens(-0.3, -0.3, -0.3)),
            ],
        ]
        with patch.object(
            RuntimeEndpoint, "_generate_http_request", side_effect=responses
        ):
            decision = endpoint.select(
                state,
                choices=["a", "b"],
                temperature=0,
                choices_method=token_length_normalized,
            )

        self.assertEqual(decision.decision, "a")


if __name__ == "__main__":
    unittest.main()
