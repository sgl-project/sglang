"""Unit tests for srt/entrypoints/openai/usage_processor.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.test_utils import CustomTestCase


class TestDetailsIfCached(CustomTestCase):
    """Tests for UsageProcessor._details_if_cached."""

    def test_positive_count_returns_details(self):
        result = UsageProcessor._details_if_cached(42)
        self.assertIsInstance(result, PromptTokensDetails)
        self.assertEqual(result.cached_tokens, 42)

    def test_zero_count_returns_none(self):
        self.assertIsNone(UsageProcessor._details_if_cached(0))

    def test_negative_count_returns_none(self):
        # Negative counts are treated like zero (no cached tokens)
        self.assertIsNone(UsageProcessor._details_if_cached(-1))


class TestCalculateTokenUsage(CustomTestCase):
    """Tests for UsageProcessor.calculate_token_usage."""

    def test_basic_usage(self):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertIsNone(usage.prompt_tokens_details)
        self.assertEqual(usage.reasoning_tokens, 0)

    def test_with_reasoning_tokens(self):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=5, completion_tokens=15, reasoning_tokens=7
        )
        self.assertEqual(usage.reasoning_tokens, 7)
        self.assertEqual(usage.total_tokens, 20)

    def test_with_cached_tokens(self):
        details = PromptTokensDetails(cached_tokens=3)
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=5, cached_tokens=details
        )
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 3)

    def test_zero_tokens(self):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=0, completion_tokens=0
        )
        self.assertEqual(usage.total_tokens, 0)


class TestCalculateResponseUsage(CustomTestCase):
    """Tests for UsageProcessor.calculate_response_usage."""

    @staticmethod
    def _make_response(
        prompt_tokens, completion_tokens, cached_tokens=0, reasoning_tokens=0
    ):
        return {
            "meta_info": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
        }

    def test_single_response(self):
        responses = [self._make_response(10, 20)]
        usage = UsageProcessor.calculate_response_usage(responses)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)

    def test_multiple_responses_n1(self):
        """Multiple responses with n_choices=1: each is a separate prompt."""
        responses = [
            self._make_response(10, 5),
            self._make_response(20, 8),
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        # prompt_tokens sums every response (step = n_choices = 1)
        self.assertEqual(usage.prompt_tokens, 30)
        self.assertEqual(usage.completion_tokens, 13)

    def test_multiple_choices_per_prompt(self):
        """With n_choices=2, only every other response contributes to prompt_tokens."""
        responses = [
            self._make_response(10, 5),  # choice 0 of prompt 0 -> prompt counted
            self._make_response(10, 8),  # choice 1 of prompt 0 -> prompt NOT counted
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 13)

    def test_cache_report_disabled(self):
        responses = [self._make_response(10, 20, cached_tokens=5)]
        usage = UsageProcessor.calculate_response_usage(
            responses, enable_cache_report=False
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_cache_report_enabled_with_cached_tokens(self):
        responses = [self._make_response(10, 20, cached_tokens=5)]
        usage = UsageProcessor.calculate_response_usage(
            responses, enable_cache_report=True
        )
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 5)

    def test_cache_report_enabled_zero_cached(self):
        """When cache report is enabled but no tokens are cached, details should be None."""
        responses = [self._make_response(10, 20, cached_tokens=0)]
        usage = UsageProcessor.calculate_response_usage(
            responses, enable_cache_report=True
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_reasoning_tokens_summed(self):
        responses = [
            self._make_response(10, 5, reasoning_tokens=3),
            self._make_response(10, 5, reasoning_tokens=4),
        ]
        usage = UsageProcessor.calculate_response_usage(responses)
        self.assertEqual(usage.reasoning_tokens, 7)

    def test_missing_meta_keys_default_to_zero(self):
        """Responses missing optional meta_info keys should not raise."""
        responses = [{"meta_info": {}}]
        usage = UsageProcessor.calculate_response_usage(responses)
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)

    def test_empty_responses(self):
        usage = UsageProcessor.calculate_response_usage([])
        self.assertEqual(usage.total_tokens, 0)


class TestCalculateStreamingUsage(CustomTestCase):
    """Tests for UsageProcessor.calculate_streaming_usage."""

    def test_single_choice(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            reasoning_tokens={0: 3},
            completion_tokens={0: 20},
            cached_tokens={0: 0},
            n_choices=1,
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.reasoning_tokens, 3)
        self.assertEqual(usage.total_tokens, 30)

    def test_multiple_choices(self):
        """With n_choices=2, only indices 0, 2, 4, ... count for prompt_tokens."""
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 10, 2: 20, 3: 20},
            reasoning_tokens={0: 1, 1: 2, 2: 3, 3: 4},
            completion_tokens={0: 5, 1: 6, 2: 7, 3: 8},
            cached_tokens={0: 0, 1: 0, 2: 0, 3: 0},
            n_choices=2,
        )
        # prompt: index 0 (10) + index 2 (20) = 30
        self.assertEqual(usage.prompt_tokens, 30)
        # completion: all summed = 5+6+7+8 = 26
        self.assertEqual(usage.completion_tokens, 26)
        # reasoning: all summed = 1+2+3+4 = 10
        self.assertEqual(usage.reasoning_tokens, 10)

    def test_cache_report_enabled(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            reasoning_tokens={0: 0},
            completion_tokens={0: 5},
            cached_tokens={0: 4},
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 4)

    def test_cache_report_disabled(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            reasoning_tokens={0: 0},
            completion_tokens={0: 5},
            cached_tokens={0: 4},
            n_choices=1,
            enable_cache_report=False,
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_cached_tokens_only_counted_for_first_choice(self):
        """With n_choices=2, cached_tokens at index 1 should not count."""
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 10},
            reasoning_tokens={0: 0, 1: 0},
            completion_tokens={0: 5, 1: 5},
            cached_tokens={0: 3, 1: 99},
            n_choices=2,
            enable_cache_report=True,
        )
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 3)

    def test_empty_mappings(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={},
            reasoning_tokens={},
            completion_tokens={},
            cached_tokens={},
            n_choices=1,
        )
        self.assertEqual(usage.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
