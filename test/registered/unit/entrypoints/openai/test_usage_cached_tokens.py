"""When cache reporting is enabled, cached_tokens=0 must still be reported.

Returning None for a 0 count makes "cache reporting on, nothing cached"
indistinguishable from "cache reporting off" for the client.
"""

import unittest

from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(3, "base-a-test-cpu")


def _responses(cached_tokens):
    return [
        {
            "meta_info": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": cached_tokens,
            }
        }
    ]


class TestCachedTokensReporting(unittest.TestCase):
    def test_zero_reported_when_enabled(self):
        usage = UsageProcessor.calculate_response_usage(
            _responses(0), enable_cache_report=True
        )
        self.assertIsNotNone(usage.prompt_tokens_details)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 0)

    def test_nonzero_reported_when_enabled(self):
        usage = UsageProcessor.calculate_response_usage(
            _responses(7), enable_cache_report=True
        )
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 7)

    def test_not_reported_when_disabled(self):
        usage = UsageProcessor.calculate_response_usage(
            _responses(0), enable_cache_report=False
        )
        self.assertIsNone(usage.prompt_tokens_details)


if __name__ == "__main__":
    unittest.main()
