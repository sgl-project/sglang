"""Unit tests for srt/entrypoints/openai/usage_processor.py"""

import unittest

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails, UsageInfo
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


class TestDetailsIfCached(CustomTestCase):
    def test_positive_count_returns_details(self):
        result = UsageProcessor._details_if_cached(5)
        self.assertIsInstance(result, PromptTokensDetails)
        self.assertEqual(result.cached_tokens, 5)

    def test_zero_count_returns_none(self):
        self.assertIsNone(UsageProcessor._details_if_cached(0))

    def test_negative_count_returns_none(self):
        self.assertIsNone(UsageProcessor._details_if_cached(-1))


class TestCalculateTokenUsage(CustomTestCase):
    def test_basic_usage(self):
        info = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20
        )
        self.assertEqual(info.prompt_tokens, 10)
        self.assertEqual(info.completion_tokens, 20)
        self.assertEqual(info.total_tokens, 30)

    def test_with_reasoning_tokens(self):
        info = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20, reasoning_tokens=5
        )
        self.assertEqual(info.reasoning_tokens, 5)
        self.assertEqual(info.total_tokens, 30)

    def test_with_cached_tokens(self):
        cached = PromptTokensDetails(cached_tokens=3)
        info = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20, cached_tokens=cached
        )
        self.assertEqual(info.prompt_tokens_details, cached)
        self.assertEqual(info.prompt_tokens_details.cached_tokens, 3)

    def test_default_reasoning_tokens(self):
        info = UsageProcessor.calculate_token_usage(prompt_tokens=5, completion_tokens=5)
        self.assertEqual(info.reasoning_tokens, 0)


class TestCalculateResponseUsage(CustomTestCase):
    def _make_response(self, prompt_tokens, completion_tokens, **extra):
        return {"meta_info": {"prompt_tokens": prompt_tokens,
                              "completion_tokens": completion_tokens, **extra}}

    def test_single_response(self):
        responses = [self._make_response(10, 20)]
        info = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(info.prompt_tokens, 10)
        self.assertEqual(info.completion_tokens, 20)
        self.assertEqual(info.total_tokens, 30)

    def test_multiple_prompts_with_n_choices(self):
        """Prompt tokens only counted once per group of n_choices."""
        responses = [
            self._make_response(10, 5),   # prompt 0, choice 0
            self._make_response(10, 6),   # prompt 0, choice 1
            self._make_response(20, 7),   # prompt 1, choice 0
        ]
        info = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        # prompts: 10 + 20 = 30 (only every 2nd)
        # completions: 5 + 6 + 7 = 18
        self.assertEqual(info.prompt_tokens, 30)
        self.assertEqual(info.completion_tokens, 18)

    def test_with_reasoning_tokens(self):
        responses = [
            self._make_response(10, 20, reasoning_tokens=3),
            self._make_response(10, 20, reasoning_tokens=4),
        ]
        info = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(info.prompt_tokens, 20)
        self.assertEqual(info.reasoning_tokens, 7)

    def test_with_cache_report(self):
        responses = [
            self._make_response(10, 20, cached_tokens=2),
        ]
        info = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=True
        )
        self.assertIsNotNone(info.prompt_tokens_details)
        self.assertEqual(info.prompt_tokens_details.cached_tokens, 2)

    def test_with_cache_report_zero_cached(self):
        """Zero cached tokens should not produce details (keeps JSON slim)."""
        responses = [
            self._make_response(10, 20, cached_tokens=0),
        ]
        info = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=True
        )
        self.assertIsNone(info.prompt_tokens_details)

    def test_cache_report_disabled(self):
        responses = [
            self._make_response(10, 20, cached_tokens=5),
        ]
        info = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=False
        )
        self.assertIsNone(info.prompt_tokens_details)

    def test_missing_tokens_default_to_zero(self):
        responses = [{"meta_info": {}}]
        info = UsageProcessor.calculate_response_usage(responses)
        self.assertEqual(info.prompt_tokens, 0)
        self.assertEqual(info.completion_tokens, 0)
        self.assertEqual(info.total_tokens, 0)

    def test_prompt_dedup_with_n_choices(self):
        """Prompt tokens from same prompt group should not be double-counted."""
        responses = [
            self._make_response(100, 10),  # idx 0 → prompt counted
            self._make_response(100, 10),  # idx 1 → skipped (same prompt)
            self._make_response(100, 10),  # idx 2 → prompt counted
            self._make_response(100, 10),  # idx 3 → skipped
        ]
        info = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        self.assertEqual(info.prompt_tokens, 200)  # 100 + 100
        self.assertEqual(info.completion_tokens, 40)  # all 4 counted


class TestCalculateStreamingUsage(CustomTestCase):
    def test_basic_streaming(self):
        prompt = {0: 10, 1: 10, 2: 20}
        completion = {0: 5, 1: 5, 2: 7}
        info = UsageProcessor.calculate_streaming_usage(
            prompt_tokens=prompt,
            reasoning_tokens={},
            completion_tokens=completion,
            cached_tokens={},
            n_choices=2,
        )
        # prompts: idx 0 (10) + idx 2 (20) = 30
        # completions: 5+5+7 = 17
        self.assertEqual(info.prompt_tokens, 30)
        self.assertEqual(info.completion_tokens, 17)

    def test_with_reasoning(self):
        prompt = {0: 10, 1: 10}
        reasoning = {0: 3, 1: 2}
        completion = {0: 5, 1: 5}
        info = UsageProcessor.calculate_streaming_usage(
            prompt_tokens=prompt,
            reasoning_tokens=reasoning,
            completion_tokens=completion,
            cached_tokens={},
            n_choices=1,
        )
        self.assertEqual(info.reasoning_tokens, 5)

    def test_with_cache(self):
        prompt = {0: 10}
        completion = {0: 20}
        cached = {0: 3}
        info = UsageProcessor.calculate_streaming_usage(
            prompt_tokens=prompt,
            reasoning_tokens={},
            completion_tokens=completion,
            cached_tokens=cached,
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertIsNotNone(info.prompt_tokens_details)
        self.assertEqual(info.prompt_tokens_details.cached_tokens, 3)

    def test_cache_zero_not_reported(self):
        prompt = {0: 10}
        completion = {0: 20}
        cached = {0: 0}
        info = UsageProcessor.calculate_streaming_usage(
            prompt_tokens=prompt,
            reasoning_tokens={},
            completion_tokens=completion,
            cached_tokens=cached,
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertIsNone(info.prompt_tokens_details)

    def test_cache_disabled(self):
        prompt = {0: 10}
        completion = {0: 20}
        cached = {0: 5}
        info = UsageProcessor.calculate_streaming_usage(
            prompt_tokens=prompt,
            reasoning_tokens={},
            completion_tokens=completion,
            cached_tokens=cached,
            n_choices=1,
            enable_cache_report=False,
        )
        self.assertIsNone(info.prompt_tokens_details)

    def test_empty_completions(self):
        info = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={},
            reasoning_tokens={},
            completion_tokens={},
            cached_tokens={},
            n_choices=1,
        )
        self.assertEqual(info.total_tokens, 0)
        self.assertEqual(info.prompt_tokens, 0)
        self.assertEqual(info.completion_tokens, 0)


if __name__ == "__main__":
    unittest.main()
