"""Unit tests for srt/entrypoints/openai/usage_processor.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails, UsageInfo
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.test_utils import CustomTestCase


class TestDetailsIfCached(CustomTestCase):
    """Tests for UsageProcessor._details_if_cached()."""

    def test_positive_count_returns_details(self):
        """Positive cached count returns PromptTokensDetails."""
        result = UsageProcessor._details_if_cached(10)
        self.assertIsInstance(result, PromptTokensDetails)
        self.assertEqual(result.cached_tokens, 10)

    def test_zero_count_returns_none(self):
        """Zero cached count returns None (keeps JSON slim)."""
        result = UsageProcessor._details_if_cached(0)
        self.assertIsNone(result)

    def test_negative_count_returns_none(self):
        """Negative cached count returns None."""
        result = UsageProcessor._details_if_cached(-1)
        self.assertIsNone(result)


class TestCalculateTokenUsage(CustomTestCase):
    """Tests for UsageProcessor.calculate_token_usage()."""

    def test_basic_usage(self):
        """Basic prompt + completion tokens are summed correctly."""
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=100, completion_tokens=50
        )
        self.assertIsInstance(result, UsageInfo)
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 50)
        self.assertEqual(result.total_tokens, 150)
        self.assertIsNone(result.prompt_tokens_details)

    def test_zero_tokens(self):
        """Zero tokens for both fields."""
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=0, completion_tokens=0
        )
        self.assertEqual(result.total_tokens, 0)

    def test_with_cached_tokens(self):
        """Cached tokens details are included when provided."""
        cached = PromptTokensDetails(cached_tokens=30)
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=100, completion_tokens=50, cached_tokens=cached
        )
        self.assertIsNotNone(result.prompt_tokens_details)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 30)

    def test_without_cached_tokens(self):
        """No cached_tokens argument means None prompt_tokens_details."""
        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=5
        )
        self.assertIsNone(result.prompt_tokens_details)


class TestCalculateResponseUsage(CustomTestCase):
    """Tests for UsageProcessor.calculate_response_usage()."""

    def _make_response(self, prompt_tokens, completion_tokens, cached_tokens=0):
        return {
            "meta_info": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
            }
        }

    def test_single_response(self):
        """Single response with n_choices=1."""
        responses = [self._make_response(50, 20)]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=1)

        self.assertEqual(result.prompt_tokens, 50)
        self.assertEqual(result.completion_tokens, 20)
        self.assertEqual(result.total_tokens, 70)

    def test_multiple_responses_single_choice(self):
        """Multiple responses each with n_choices=1 sums all prompts and completions."""
        responses = [
            self._make_response(50, 20),
            self._make_response(30, 10),
        ]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=1)

        self.assertEqual(result.prompt_tokens, 80)
        self.assertEqual(result.completion_tokens, 30)
        self.assertEqual(result.total_tokens, 110)

    def test_multiple_choices_prompt_counted_once_per_group(self):
        """With n_choices=2, prompt tokens from every other response are summed."""
        responses = [
            self._make_response(100, 10),  # index 0 -> counted for prompt
            self._make_response(100, 15),  # index 1 -> skipped for prompt
        ]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=2)

        # Only index 0 counted for prompt (100), both for completion (10+15)
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 25)
        self.assertEqual(result.total_tokens, 125)

    def test_four_responses_two_choices(self):
        """Two prompts with n_choices=2 each -> 4 responses total."""
        responses = [
            self._make_response(50, 10),  # prompt 1, choice 1
            self._make_response(50, 12),  # prompt 1, choice 2
            self._make_response(60, 8),  # prompt 2, choice 1
            self._make_response(60, 9),  # prompt 2, choice 2
        ]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=2)

        # prompt: index 0 (50) + index 2 (60) = 110
        self.assertEqual(result.prompt_tokens, 110)
        # completion: 10+12+8+9 = 39
        self.assertEqual(result.completion_tokens, 39)
        self.assertEqual(result.total_tokens, 149)

    def test_cache_report_disabled(self):
        """When enable_cache_report=False, no prompt_tokens_details."""
        responses = [self._make_response(50, 20, cached_tokens=10)]
        result = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=False
        )
        self.assertIsNone(result.prompt_tokens_details)

    def test_cache_report_enabled_with_cached_tokens(self):
        """When enable_cache_report=True and cached_tokens > 0, details are included."""
        responses = [self._make_response(50, 20, cached_tokens=15)]
        result = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=True
        )
        self.assertIsNotNone(result.prompt_tokens_details)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 15)

    def test_cache_report_enabled_zero_cached(self):
        """When enable_cache_report=True but cached_tokens=0, details are None."""
        responses = [self._make_response(50, 20, cached_tokens=0)]
        result = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=True
        )
        self.assertIsNone(result.prompt_tokens_details)

    def test_cache_report_with_multiple_choices(self):
        """Cached tokens with n_choices=2 only sums from first choice of each group."""
        responses = [
            self._make_response(50, 10, cached_tokens=20),
            self._make_response(50, 12, cached_tokens=20),
        ]
        result = UsageProcessor.calculate_response_usage(
            responses, n_choices=2, enable_cache_report=True
        )
        # Only index 0 cached_tokens (20), index 1 skipped
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 20)

    def test_missing_meta_fields_default_to_zero(self):
        """Missing prompt_tokens/completion_tokens in meta_info default to 0."""
        responses = [{"meta_info": {}}]
        result = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)
        self.assertEqual(result.total_tokens, 0)


class TestCalculateStreamingUsage(CustomTestCase):
    """Tests for UsageProcessor.calculate_streaming_usage()."""

    def test_single_choice_streaming(self):
        """Basic streaming usage with one choice."""
        prompt_tokens = {0: 100}
        completion_tokens = {0: 50}
        cached_tokens = {0: 10}

        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens, completion_tokens, cached_tokens, n_choices=1
        )

        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 50)
        self.assertEqual(result.total_tokens, 150)
        # cache report disabled by default
        self.assertIsNone(result.prompt_tokens_details)

    def test_multiple_choices_prompt_dedup(self):
        """With n_choices=2, only even-indexed entries are summed for prompt."""
        prompt_tokens = {0: 100, 1: 100}
        completion_tokens = {0: 20, 1: 25}
        cached_tokens = {0: 10, 1: 10}

        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens, completion_tokens, cached_tokens, n_choices=2
        )

        # Only index 0 for prompt (100)
        self.assertEqual(result.prompt_tokens, 100)
        # All completions: 20+25=45
        self.assertEqual(result.completion_tokens, 45)
        self.assertEqual(result.total_tokens, 145)

    def test_four_entries_two_choices(self):
        """Four entries with n_choices=2: two prompts, four completions."""
        prompt_tokens = {0: 50, 1: 50, 2: 60, 3: 60}
        completion_tokens = {0: 10, 1: 12, 2: 8, 3: 9}
        cached_tokens = {0: 5, 1: 5, 2: 3, 3: 3}

        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens, completion_tokens, cached_tokens, n_choices=2
        )

        # Prompt: index 0 (50) + index 2 (60) = 110
        self.assertEqual(result.prompt_tokens, 110)
        # Completion: 10+12+8+9 = 39
        self.assertEqual(result.completion_tokens, 39)
        self.assertEqual(result.total_tokens, 149)

    def test_cache_report_enabled(self):
        """Cached token details included when enable_cache_report=True."""
        prompt_tokens = {0: 100}
        completion_tokens = {0: 50}
        cached_tokens = {0: 30}

        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            n_choices=1,
            enable_cache_report=True,
        )

        self.assertIsNotNone(result.prompt_tokens_details)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 30)

    def test_cache_report_enabled_zero_cached(self):
        """When enable_cache_report=True but all cached=0, details are None."""
        prompt_tokens = {0: 100}
        completion_tokens = {0: 50}
        cached_tokens = {0: 0}

        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            n_choices=1,
            enable_cache_report=True,
        )

        self.assertIsNone(result.prompt_tokens_details)

    def test_cache_report_with_multiple_choices(self):
        """Cached tokens with n_choices=2 only sums even-indexed entries."""
        prompt_tokens = {0: 100, 1: 100}
        completion_tokens = {0: 20, 1: 25}
        cached_tokens = {0: 15, 1: 15}

        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            n_choices=2,
            enable_cache_report=True,
        )

        # Only index 0 cached (15)
        self.assertEqual(result.prompt_tokens_details.cached_tokens, 15)

    def test_empty_dicts(self):
        """Empty dictionaries produce zero-count usage."""
        result = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={},
            completion_tokens={},
            cached_tokens={},
            n_choices=1,
        )
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)
        self.assertEqual(result.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
