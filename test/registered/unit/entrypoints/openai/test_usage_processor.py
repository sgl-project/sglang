import unittest

from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUsageProcessor(CustomTestCase):
    def test_cache_details_follow_reporting_flag(self):
        for enable_cache_report, cached_tokens in (
            (False, 7),
            (True, 0),
            (True, 7),
        ):
            with self.subTest(
                enable_cache_report=enable_cache_report,
                cached_tokens=cached_tokens,
            ):
                usages = (
                    UsageProcessor.calculate_response_usage(
                        [{"meta_info": {"cached_tokens": cached_tokens}}],
                        enable_cache_report=enable_cache_report,
                    ),
                    UsageProcessor.calculate_streaming_usage(
                        prompt_tokens={0: 0},
                        reasoning_tokens={0: 0},
                        completion_tokens={0: 0},
                        cached_tokens={0: cached_tokens},
                        n_choices=1,
                        enable_cache_report=enable_cache_report,
                    ),
                )
                for usage in usages:
                    details = usage.prompt_tokens_details
                    if enable_cache_report:
                        self.assertIsNotNone(details)
                        self.assertEqual(details.cached_tokens, cached_tokens)
                    else:
                        self.assertIsNone(details)


if __name__ == "__main__":
    unittest.main()
