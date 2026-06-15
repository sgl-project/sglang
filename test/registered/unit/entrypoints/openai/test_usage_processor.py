"""Unit tests for entrypoints/openai/usage_processor.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")

import unittest

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.test_utils import CustomTestCase


def _resp(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, cached_tokens=0):
    meta = {}
    if prompt_tokens:
        meta["prompt_tokens"] = prompt_tokens
    if completion_tokens:
        meta["completion_tokens"] = completion_tokens
    if reasoning_tokens:
        meta["reasoning_tokens"] = reasoning_tokens
    if cached_tokens:
        meta["cached_tokens"] = cached_tokens
    return {"meta_info": meta}


class TestDetailsIfCached(CustomTestCase):
    def test_positive_count_returns_details(self):
        details = UsageProcessor._details_if_cached(5)
        self.assertIsInstance(details, PromptTokensDetails)
        self.assertEqual(details.cached_tokens, 5)

    def test_zero_or_negative_returns_none(self):
        self.assertIsNone(UsageProcessor._details_if_cached(0))
        self.assertIsNone(UsageProcessor._details_if_cached(-1))


class TestCalculateResponseUsage(CustomTestCase):
    def test_n_choices_1_every_response_counts_for_prompt(self):
        responses = [
            _resp(prompt_tokens=100, completion_tokens=10),
            _resp(prompt_tokens=200, completion_tokens=20),
            _resp(prompt_tokens=300, completion_tokens=30),
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(usage.prompt_tokens, 600)
        self.assertEqual(usage.completion_tokens, 60)

    def test_n_choices_2_prompt_skipped_on_odd_indices(self):
        responses = [
            _resp(prompt_tokens=100, completion_tokens=10),
            _resp(prompt_tokens=100, completion_tokens=10),
            _resp(prompt_tokens=200, completion_tokens=10),
            _resp(prompt_tokens=200, completion_tokens=10),
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        self.assertEqual(usage.prompt_tokens, 300)
        self.assertEqual(usage.completion_tokens, 40)

    def test_cache_report_enabled(self):
        responses = [
            _resp(prompt_tokens=100, completion_tokens=10, cached_tokens=50),
            _resp(prompt_tokens=100, completion_tokens=10, cached_tokens=50),
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses, n_choices=2, enable_cache_report=True
        )
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 50)

    def test_cache_report_disabled_ignores_cached_tokens(self):
        responses = [
            _resp(cached_tokens=999),
            _resp(cached_tokens=999),
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=False
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_reasoning_tokens_sum_all_regardless_of_stride(self):
        responses = [
            _resp(reasoning_tokens=50),
            _resp(reasoning_tokens=50),
            _resp(reasoning_tokens=50),
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=3)
        self.assertEqual(usage.reasoning_tokens, 150)

    def test_n_choices_greater_than_response_count(self):
        responses = [
            _resp(prompt_tokens=100, completion_tokens=10),
            _resp(prompt_tokens=100, completion_tokens=10),
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=5)
        self.assertEqual(usage.prompt_tokens, 100)

    def test_missing_meta_keys_default_to_zero(self):
        responses = [{"meta_info": {}}, {"meta_info": {}}]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=1)
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)

    def test_n_choices_zero_raises_valueerror(self):
        with self.assertRaises(ValueError):
            UsageProcessor.calculate_response_usage(
                [_resp(prompt_tokens=100)], n_choices=0
            )


class TestCalculateStreamingUsage(CustomTestCase):
    def test_prompt_tokens_stride(self):
        prompt = {0: 100, 1: 100, 2: 200, 3: 200}
        reasoning = {0: 10, 1: 10, 2: 10, 3: 10}
        completion = {0: 5, 1: 5, 2: 5, 3: 5}
        usage = UsageProcessor.calculate_streaming_usage(
            prompt, reasoning, completion, {}, n_choices=2
        )
        self.assertEqual(usage.prompt_tokens, 300)
        self.assertEqual(usage.reasoning_tokens, 40)
        self.assertEqual(usage.completion_tokens, 20)

    def test_cache_report_enabled(self):
        prompt = {0: 100}
        completion = {0: 10}
        reasoning = {0: 0}
        cached = {0: 60, 1: 60}
        usage = UsageProcessor.calculate_streaming_usage(
            prompt, reasoning, completion, cached, n_choices=2, enable_cache_report=True
        )
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 60)

    def test_cache_report_disabled_ignores_cached(self):
        cached = {0: 60}
        usage = UsageProcessor.calculate_streaming_usage(
            {0: 100}, {0: 0}, {0: 10}, cached, n_choices=1, enable_cache_report=False
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_sparse_keys(self):
        prompt = {0: 100, 5: 200}
        completion = {2: 10, 7: 20}
        reasoning = {3: 5}
        usage = UsageProcessor.calculate_streaming_usage(
            prompt, reasoning, completion, {}, n_choices=2
        )
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 30)
        self.assertEqual(usage.reasoning_tokens, 5)

    def test_n_choices_zero_raises_zerodivisionerror(self):
        with self.assertRaises(ZeroDivisionError):
            UsageProcessor.calculate_streaming_usage(
                {0: 100}, {0: 0}, {0: 10}, {}, n_choices=0
            )


if __name__ == "__main__":
    unittest.main()
