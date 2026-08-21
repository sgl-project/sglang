import unittest

from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(3, "base-a-test-cpu")


def _response_usage(cached_tokens, enabled):
    return UsageProcessor.calculate_response_usage(
        [{"meta_info": {"cached_tokens": cached_tokens}}],
        enable_cache_report=enabled,
    )


def _streaming_usage(cached_tokens, enabled):
    return UsageProcessor.calculate_streaming_usage(
        prompt_tokens={0: 0},
        reasoning_tokens={0: 0},
        completion_tokens={0: 0},
        cached_tokens={0: cached_tokens},
        n_choices=1,
        enable_cache_report=enabled,
    )


class TestCachedTokensReporting(unittest.TestCase):
    def test_cache_details_follow_reporting_flag(self):
        for calculate in (_response_usage, _streaming_usage):
            for enabled, cached_tokens, expected in (
                (False, 0, None),
                (False, 7, None),
                (True, 0, {"cached_tokens": 0}),
                (True, 7, {"cached_tokens": 7}),
            ):
                with self.subTest(
                    calculate=calculate.__name__,
                    enabled=enabled,
                    cached_tokens=cached_tokens,
                ):
                    usage = calculate(cached_tokens, enabled)
                    actual = usage.model_dump(exclude_none=True).get(
                        "prompt_tokens_details"
                    )
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
