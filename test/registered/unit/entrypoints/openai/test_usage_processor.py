"""Unit tests for srt/entrypoints/openai/usage_processor.py"""

import unittest

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class TestUsageProcessor(CustomTestCase):
    def test_calculate_token_usage_basic(self):
        """Test basic token usage calculation without optional fields."""
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertIsNone(usage.prompt_tokens_details)
        self.assertEqual(usage.reasoning_tokens, 0)

    def test_calculate_token_usage_with_reasoning(self):
        """Test token usage calculation with reasoning tokens."""
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20, reasoning_tokens=5
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertEqual(usage.reasoning_tokens, 5)

    def test_calculate_token_usage_with_cached_details(self):
        """Test token usage calculation with cached token details."""
        cached_details = PromptTokensDetails(cached_tokens=15)
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10, completion_tokens=20, cached_tokens=cached_details
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 15)

    def test_details_if_cached_zero(self):
        """Test _details_if_cached returns None for zero count."""
        result = UsageProcessor._details_if_cached(0)
        self.assertIsNone(result)

    def test_details_if_cached_negative(self):
        """Test _details_if_cached returns None for negative count."""
        result = UsageProcessor._details_if_cached(-1)
        self.assertIsNone(result)

    def test_details_if_cached_positive(self):
        """Test _details_if_cached returns details for positive count."""
        result = UsageProcessor._details_if_cached(10)
        self.assertIsNotNone(result)
        self.assertEqual(result.cached_tokens, 10)

    def test_calculate_response_usage_single_choice(self):
        """Test response usage calculation with single choice."""
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "reasoning_tokens": 5,
                }
            }
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.reasoning_tokens, 5)
        self.assertEqual(usage.total_tokens, 30)

    def test_calculate_response_usage_multiple_choices(self):
        """Test response usage calculation with n>1 choices."""
        # 2 prompts, each with 2 choices (4 responses total)
        responses = [
            {"meta_info": {"prompt_tokens": 10, "completion_tokens": 5}},  # prompt 0
            {"meta_info": {"prompt_tokens": 10, "completion_tokens": 7}},  # prompt 0
            {"meta_info": {"prompt_tokens": 15, "completion_tokens": 8}},  # prompt 1
            {"meta_info": {"prompt_tokens": 15, "completion_tokens": 9}},  # prompt 1
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        # Only count prompt tokens from indices 0, 2 (first choice of each prompt)
        self.assertEqual(usage.prompt_tokens, 25)  # 10 + 15
        # Sum all completion tokens
        self.assertEqual(usage.completion_tokens, 29)  # 5 + 7 + 8 + 9
        self.assertEqual(usage.total_tokens, 54)

    def test_calculate_response_usage_with_cache_report(self):
        """Test response usage with cache reporting enabled."""
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "cached_tokens": 5,
                }
            }
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=True
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 5)

    def test_calculate_response_usage_no_cache_report(self):
        """Test response usage with cache reporting disabled."""
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "cached_tokens": 5,
                }
            }
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=False
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_calculate_response_usage_missing_fields(self):
        """Test response usage handles missing meta_info fields gracefully."""
        responses = [{"meta_info": {}}]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertEqual(usage.reasoning_tokens, 0)

    def test_calculate_streaming_usage_single_choice(self):
        """Test streaming usage calculation with single choice."""
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            reasoning_tokens={0: 5},
            completion_tokens={0: 20},
            cached_tokens={0: 3},
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.reasoning_tokens, 5)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 3)

    def test_calculate_streaming_usage_multiple_choices(self):
        """Test streaming usage with n>1 choices."""
        # 2 prompts with 2 choices each (indices 0,1,2,3)
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 10, 2: 15, 3: 15},
            reasoning_tokens={0: 2, 1: 3, 2: 4, 3: 5},
            completion_tokens={0: 5, 1: 7, 2: 8, 3: 9},
            cached_tokens={0: 1, 1: 1, 2: 2, 3: 2},
            n_choices=2,
            enable_cache_report=True,
        )
        # Only indices 0, 2 (idx % 2 == 0) for prompt tokens
        self.assertEqual(usage.prompt_tokens, 25)  # 10 + 15
        # Sum all reasoning tokens
        self.assertEqual(usage.reasoning_tokens, 14)  # 2 + 3 + 4 + 5
        # Sum all completion tokens
        self.assertEqual(usage.completion_tokens, 29)  # 5 + 7 + 8 + 9
        # Only indices 0, 2 for cached tokens
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 3)  # 1 + 2

    def test_calculate_streaming_usage_no_cache_report(self):
        """Test streaming usage without cache reporting."""
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10},
            reasoning_tokens={0: 5},
            completion_tokens={0: 20},
            cached_tokens={0: 3},
            n_choices=1,
            enable_cache_report=False,
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_calculate_streaming_usage_empty_dicts(self):
        """Test streaming usage with empty token dictionaries."""
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={},
            reasoning_tokens={},
            completion_tokens={},
            cached_tokens={},
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.reasoning_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertIsNone(usage.prompt_tokens_details)


if __name__ == "__main__":
    unittest.main()
