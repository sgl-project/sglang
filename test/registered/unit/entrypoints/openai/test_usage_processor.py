"""
Unit tests for sglang.srt.entrypoints.openai.usage_processor.UsageProcessor.

Covers:
  - calculate_token_usage
  - calculate_response_usage (batch / n_choices)
  - calculate_streaming_usage (streaming / n_choices)

All tests run on CPU with no server or model weights required.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Stub out sgl_kernel before any sglang import so tests run on CPU-only runners.
for _mod in ("sgl_kernel", "sgl_kernel.kvcacheio"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails, UsageInfo
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> dict:
    return {
        "meta_info": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        }
    }


# ---------------------------------------------------------------------------
# calculate_token_usage
# ---------------------------------------------------------------------------


class TestCalculateTokenUsage(unittest.TestCase):
    def test_basic_totals(self):
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=5
        )
        self.assertIsInstance(result, UsageInfo)
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 5)
        self.assertEqual(result.total_tokens, 15)

    def test_total_is_sum_of_prompt_and_completion(self):
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=100, completion_tokens=200
        )
        self.assertEqual(result.total_tokens, 300)

    def test_no_cached_tokens_details(self):
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=5
        )
        self.assertIsNone(result.prompt_tokens_details)

    def test_with_cached_tokens_details(self):
        details = PromptTokensDetails(cached_tokens=7)
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=5, cached_tokens=details
        )
        self.assertIsNotNone(result.prompt_tokens_details)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 7)

    def test_zero_tokens(self):
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=0, completion_tokens=0
        )
        self.assertEqual(result.total_tokens, 0)


# ---------------------------------------------------------------------------
# calculate_response_usage
# ---------------------------------------------------------------------------


class TestCalculateResponseUsage(unittest.TestCase):
    def test_single_response(self):
        responses = [_resp(prompt_tokens=20, completion_tokens=10)]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(result.completion_tokens, 10)
        self.assertEqual(result.total_tokens, 30)

    def test_multiple_responses_n_choices_1(self):
        # With n_choices=1 every response counts for prompt
        responses = [
            _resp(prompt_tokens=10, completion_tokens=5),
            _resp(prompt_tokens=10, completion_tokens=3),
        ]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(result.completion_tokens, 8)

    def test_n_choices_2_only_first_of_each_pair_counts_for_prompt(self):
        # n=2 choices: responses[0] and responses[1] are choices for prompt A.
        # prompt_tokens should be counted once (from index 0 only, not index 1).
        responses = [
            _resp(prompt_tokens=50, completion_tokens=5),  # idx 0 — counts for prompt
            _resp(prompt_tokens=50, completion_tokens=3),  # idx 1 — skipped for prompt
        ]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        self.assertEqual(result.prompt_tokens, 50)  # only index 0
        self.assertEqual(result.completion_tokens, 8)  # all choices sum

    def test_n_choices_2_two_prompts(self):
        # 4 responses, n_choices=2: two prompts × two choices each.
        # prompt_tokens at indices 0 and 2 count; indices 1 and 3 are skipped.
        responses = [
            _resp(prompt_tokens=10, completion_tokens=1),  # prompt A, choice 0
            _resp(prompt_tokens=10, completion_tokens=2),  # prompt A, choice 1
            _resp(prompt_tokens=20, completion_tokens=3),  # prompt B, choice 0
            _resp(prompt_tokens=20, completion_tokens=4),  # prompt B, choice 1
        ]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        self.assertEqual(result.prompt_tokens, 30)  # 10 + 20
        self.assertEqual(result.completion_tokens, 10)  # 1+2+3+4

    def test_cache_report_disabled_no_details(self):
        responses = [_resp(prompt_tokens=10, completion_tokens=5, cached_tokens=3)]
        result = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=False
        )
        self.assertIsNone(result.prompt_tokens_details)

    def test_cache_report_enabled_zero_cached_no_details(self):
        # _details_if_cached returns None when count == 0
        responses = [_resp(prompt_tokens=10, completion_tokens=5, cached_tokens=0)]
        result = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=True
        )
        self.assertIsNone(result.prompt_tokens_details)

    def test_cache_report_enabled_with_cached_tokens(self):
        responses = [_resp(prompt_tokens=10, completion_tokens=5, cached_tokens=4)]
        result = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=True
        )
        self.assertIsNotNone(result.prompt_tokens_details)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 4)


# ---------------------------------------------------------------------------
# calculate_streaming_usage
# ---------------------------------------------------------------------------


class TestCalculateStreamingUsage(unittest.TestCase):
    def test_basic_single_choice(self):
        prompt_tokens = {0: 15}
        completion_tokens = {0: 7}
        cached_tokens = {0: 0}
        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            n_choices=1,
        )
        self.assertEqual(result.prompt_tokens, 15)
        self.assertEqual(result.completion_tokens, 7)
        self.assertEqual(result.total_tokens, 22)

    def test_n_choices_2_only_even_indices_count_for_prompt(self):
        # Indices 0 and 2 are "first of each prompt" (idx % 2 == 0).
        prompt_tokens = {0: 10, 1: 10, 2: 20, 3: 20}
        completion_tokens = {0: 1, 1: 2, 2: 3, 3: 4}
        cached_tokens = {0: 0, 1: 0, 2: 0, 3: 0}
        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            n_choices=2,
        )
        self.assertEqual(result.prompt_tokens, 30)  # 10 + 20
        self.assertEqual(result.completion_tokens, 10)  # 1+2+3+4

    def test_cache_report_disabled_no_details(self):
        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            completion_tokens={0: 5},
            cached_tokens={0: 3},
            n_choices=1,
            enable_cache_report=False,
        )
        self.assertIsNone(result.prompt_tokens_details)

    def test_cache_report_enabled_zero_cached_no_details(self):
        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            completion_tokens={0: 5},
            cached_tokens={0: 0},
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertIsNone(result.prompt_tokens_details)

    def test_cache_report_enabled_with_cached_tokens(self):
        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            completion_tokens={0: 5},
            cached_tokens={0: 6},
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertIsNotNone(result.prompt_tokens_details)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 6)

    def test_cache_report_n_choices_2_only_even_indices_sum(self):
        # Only indices where idx % n_choices == 0 contribute to cached total.
        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 10},
            completion_tokens={0: 1, 1: 2},
            cached_tokens={0: 5, 1: 5},
            n_choices=2,
            enable_cache_report=True,
        )
        self.assertIsNotNone(result.prompt_tokens_details)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 5)  # only idx 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
