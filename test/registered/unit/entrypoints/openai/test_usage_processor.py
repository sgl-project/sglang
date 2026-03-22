"""Unit tests for UsageProcessor — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor


class TestDetailsIfCached(unittest.TestCase):

    def test_zero_returns_none(self):
        self.assertIsNone(UsageProcessor._details_if_cached(0))

    def test_negative_returns_none(self):
        self.assertIsNone(UsageProcessor._details_if_cached(-1))

    def test_positive_returns_details(self):
        result = UsageProcessor._details_if_cached(42)
        self.assertIsInstance(result, PromptTokensDetails)
        self.assertEqual(result.cached_tokens, 42)


class TestCalculateTokenUsage(unittest.TestCase):

    def test_total_equals_sum(self):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertIsNone(usage.prompt_tokens_details)

    def test_cached_details_propagated(self):
        cached = PromptTokensDetails(cached_tokens=5)
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20, cached_tokens=cached
        )
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 5)


class TestCalculateResponseUsage(unittest.TestCase):

    def _make_response(self, prompt_tokens=10, completion_tokens=5, cached_tokens=0):
        return {
            "meta_info": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
            }
        }

    def test_single_response(self):
        responses = [self._make_response(prompt_tokens=10, completion_tokens=5)]
        usage = UsageProcessor.calculate_response_usage(responses)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 5)
        self.assertEqual(usage.total_tokens, 15)

    def test_n_choices_skips_duplicate_prompt_counting(self):
        """With n_choices=2 the source loops range(0, len, n_choices), so only
        index 0 should contribute prompt_tokens, not index 1."""
        responses = [
            self._make_response(prompt_tokens=10, completion_tokens=3),
            self._make_response(prompt_tokens=99, completion_tokens=4),
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 7)

    def test_n_choices_with_four_responses(self):
        """4 responses with n_choices=2: prompt counted at index 0 and 2 only."""
        responses = [
            self._make_response(prompt_tokens=10, completion_tokens=1),
            self._make_response(prompt_tokens=99, completion_tokens=2),
            self._make_response(prompt_tokens=20, completion_tokens=3),
            self._make_response(prompt_tokens=99, completion_tokens=4),
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        self.assertEqual(usage.prompt_tokens, 30)
        self.assertEqual(usage.completion_tokens, 10)

    def test_cache_report_disabled_omits_details(self):
        responses = [self._make_response(cached_tokens=5)]
        usage = UsageProcessor.calculate_response_usage(
            responses, enable_cache_report=False
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_cache_report_enabled(self):
        responses = [self._make_response(cached_tokens=5)]
        usage = UsageProcessor.calculate_response_usage(
            responses, enable_cache_report=True
        )
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 5)

    def test_cache_report_zero_cached_stays_none(self):
        """_details_if_cached returns None for 0, so prompt_tokens_details should be None."""
        responses = [self._make_response(cached_tokens=0)]
        usage = UsageProcessor.calculate_response_usage(
            responses, enable_cache_report=True
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_missing_meta_info_keys_default_to_zero(self):
        responses = [{"meta_info": {}}]
        usage = UsageProcessor.calculate_response_usage(responses)
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)


class TestCalculateStreamingUsage(unittest.TestCase):

    def test_single_choice(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            completion_tokens={0: 20},
            cached_tokens={0: 0},
            n_choices=1,
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)

    def test_multi_choice_prompt_filtering(self):
        """Prompt tokens are only summed for indices where idx % n_choices == 0,
        while completion tokens are summed across all indices."""
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 99, 2: 10, 3: 99},
            completion_tokens={0: 5, 1: 5, 2: 5, 3: 5},
            cached_tokens={0: 0, 1: 0, 2: 0, 3: 0},
            n_choices=2,
        )
        self.assertEqual(usage.prompt_tokens, 20)
        self.assertEqual(usage.completion_tokens, 20)

    def test_streaming_cache_report(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            completion_tokens={0: 5},
            cached_tokens={0: 3},
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 3)

    def test_streaming_cache_report_disabled(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            completion_tokens={0: 5},
            cached_tokens={0: 3},
            n_choices=1,
            enable_cache_report=False,
        )
        self.assertIsNone(usage.prompt_tokens_details)


if __name__ == "__main__":
    unittest.main()
