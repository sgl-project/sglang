"""Regression tests for prefix determinism result propagation."""

import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from sglang.test import test_deterministic as deterministic
from sglang.test import test_deterministic_utils as deterministic_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeterministicPrefixResults(CustomTestCase):
    def setUp(self):
        super().setUp()
        response = {
            "text": "same output",
            "meta_info": {"output_token_logprobs": [[-0.1, 42, " token"]]},
        }
        self.responses = {
            index: [copy.deepcopy(response), copy.deepcopy(response)]
            for index in range(4)
        }

    def _run_prefix(self, *, return_logprob=True):
        args = deterministic.BenchArgs(
            test_mode="prefix", n_start=8, n_trials=1, return_logprob=return_logprob
        )
        responses = self.responses
        if not return_logprob:
            responses = {
                index: [response["text"] for response in group]
                for index, group in responses.items()
            }
        with (
            patch.object(deterministic, "send_prefix", return_value=responses),
            redirect_stdout(io.StringIO()),
        ):
            return deterministic.test_deterministic(args)

    def test_equal_text_tokens_and_logprobs_pass(self):
        self.assertTrue(all(result == 1 for result in self._run_prefix()))

    def test_logprob_mismatch_with_identical_text_and_token_fails(self):
        self.responses[2][1]["meta_info"]["output_token_logprobs"][0][0] = -0.2
        self.assertFalse(all(result == 1 for result in self._run_prefix()))

    def test_token_id_mismatch_with_identical_text_fails(self):
        self.responses[2][1]["meta_info"]["output_token_logprobs"][0][1] = 43
        self.assertFalse(all(result == 1 for result in self._run_prefix()))

    def test_logprob_length_mismatch_fails(self):
        self.responses[2][1]["meta_info"]["output_token_logprobs"].append(
            [-0.3, 43, " other"]
        )
        self.assertFalse(all(result == 1 for result in self._run_prefix()))

    def test_insufficient_responses_fail(self):
        original = self.responses[2]
        for count in (0, 1):
            with self.subTest(response_count=count):
                self.responses[2] = original[:count]
                self.assertFalse(all(result == 1 for result in self._run_prefix()))

    def test_empty_logprob_sequences_fail(self):
        for response in self.responses[2]:
            response["meta_info"]["output_token_logprobs"] = []
        self.assertFalse(all(result == 1 for result in self._run_prefix()))

    def test_text_mismatch_still_fails_when_scores_match(self):
        self.responses[2][1]["text"] = "different output"
        self.assertFalse(all(result == 1 for result in self._run_prefix()))

    def test_text_only_results_are_unchanged(self):
        self.responses[2][1]["meta_info"]["output_token_logprobs"][0][0] = -0.2
        self.assertEqual(self._run_prefix(return_logprob=False), [1, 1, 1, 1])
        self.responses[2][1]["text"] = "different output"
        self.assertEqual(self._run_prefix(return_logprob=False), [1, 1, 2, 1])

    def test_registered_caller_rejects_logprob_mismatch(self):
        self.responses[2][1]["meta_info"]["output_token_logprobs"][0][0] = -0.2
        case = deterministic_utils.TestDeterministicBase(
            methodName="test_prefix_with_logprobs"
        )
        with (
            patch.object(deterministic, "send_prefix", return_value=self.responses),
            redirect_stdout(io.StringIO()),
            self.assertRaises(AssertionError),
        ):
            case.test_prefix_with_logprobs()


if __name__ == "__main__":
    unittest.main()
