# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for UsageProcessor — no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUsageProcessor(unittest.TestCase):
    def test_calculate_response_usage_single_response(self):
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "reasoning_tokens": 2,
                }
            }
        ]
        usage = UsageProcessor.calculate_response_usage(responses)
        self.assertEqual(usage.prompt_tokens, 3)
        self.assertEqual(usage.completion_tokens, 5)
        self.assertEqual(usage.total_tokens, 8)
        self.assertEqual(usage.reasoning_tokens, 2)
        self.assertIsNone(usage.prompt_tokens_details)

    def test_calculate_response_usage_n_choices_counts_prompt_once(self):
        # Two prompts, each with two choices (n=2).
        responses = [
            {"meta_info": {"prompt_tokens": 10, "completion_tokens": 1}},
            {"meta_info": {"prompt_tokens": 10, "completion_tokens": 2}},
            {"meta_info": {"prompt_tokens": 20, "completion_tokens": 3}},
            {"meta_info": {"prompt_tokens": 20, "completion_tokens": 4}},
        ]
        usage = UsageProcessor.calculate_response_usage(responses, n_choices=2)
        self.assertEqual(usage.prompt_tokens, 30)
        self.assertEqual(usage.completion_tokens, 10)
        self.assertEqual(usage.total_tokens, 40)

    def test_calculate_response_usage_cache_report_enabled(self):
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "cached_tokens": 4,
                }
            }
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses, enable_cache_report=True
        )
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 4)

    def test_calculate_response_usage_cache_report_zero_is_omitted(self):
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                }
            }
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses, enable_cache_report=True
        )
        # _details_if_cached suppresses the details object when the count is 0.
        self.assertIsNone(usage.prompt_tokens_details)

    def test_calculate_response_usage_multimodal_counts_with_cache(self):
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "cached_tokens": 2,
                }
            }
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses,
            enable_cache_report=True,
            image_tokens=3,
            audio_tokens=1,
            video_tokens=1,
        )
        details = usage.prompt_tokens_details
        self.assertIsNotNone(details)
        self.assertEqual(details.cached_tokens, 2)
        self.assertEqual(details.image_tokens, 3)
        self.assertEqual(details.audio_tokens, 1)
        self.assertEqual(details.video_tokens, 1)

    def test_calculate_response_usage_missing_meta_info_keys_default_to_zero(self):
        responses = [{"meta_info": {}}]
        usage = UsageProcessor.calculate_response_usage(responses)
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertEqual(usage.reasoning_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)

    def test_calculate_streaming_usage_basic(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 5},
            reasoning_tokens={0: 2},
            completion_tokens={0: 3, 1: 4},
            cached_tokens={0: 10, 1: 0},
            n_choices=1,
            enable_cache_report=True,
        )
        self.assertEqual(usage.prompt_tokens, 15)
        self.assertEqual(usage.completion_tokens, 7)
        self.assertEqual(usage.reasoning_tokens, 2)
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 10)

    def test_calculate_streaming_usage_n_choices_filters_prompt_and_cache(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 99, 2: 20, 3: 99},
            reasoning_tokens={},
            completion_tokens={0: 1, 1: 2, 2: 3, 3: 4},
            cached_tokens={0: 5, 1: 99, 2: 6, 3: 99},
            n_choices=2,
            enable_cache_report=True,
        )
        # Only indices divisible by n_choices belong to the first (charged) choice.
        self.assertEqual(usage.prompt_tokens, 30)
        self.assertEqual(usage.completion_tokens, 10)
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 11)

    def test_calculate_streaming_usage_cache_report_disabled(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 1},
            reasoning_tokens={},
            completion_tokens={0: 1},
            cached_tokens={0: 5},
            n_choices=1,
            enable_cache_report=False,
        )
        self.assertIsNone(usage.prompt_tokens_details)

    def test_calculate_token_usage_no_details_without_multimodal_or_cache(self):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=1, completion_tokens=2
        )
        self.assertIsNone(usage.prompt_tokens_details)
        self.assertEqual(usage.total_tokens, 3)

    def test_calculate_token_usage_multimodal_details_without_cache(self):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=1, completion_tokens=2, image_tokens=3
        )
        details = usage.prompt_tokens_details
        self.assertIsInstance(details, PromptTokensDetails)
        self.assertEqual(details.image_tokens, 3)
        self.assertEqual(details.cached_tokens, 0)

    def test_calculate_token_usage_reasoning_tokens_preserved(self):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=1,
            completion_tokens=2,
            reasoning_tokens=4,
        )
        self.assertEqual(usage.reasoning_tokens, 4)


if __name__ == "__main__":
    unittest.main()
