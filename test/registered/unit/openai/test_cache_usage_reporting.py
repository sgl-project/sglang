import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.openai import serving_chat
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCacheReadReporting(unittest.TestCase):
    def assert_cached(self, usage, expected):
        raw = usage.model_dump()
        details = raw["prompt_tokens_details"]
        if expected is None:
            self.assertIsNone(details)
        else:
            self.assertEqual(details["cached_tokens"], expected)
        self.assertEqual(
            raw["total_tokens"], raw["prompt_tokens"] + raw["completion_tokens"]
        )

    def response(self, count, enabled=True):
        meta = {"prompt_tokens": 100, "completion_tokens": 7}
        if count is not None:
            meta["cached_tokens"] = count
        return UsageProcessor.calculate_response_usage(
            [{"meta_info": meta}], enable_cache_report=enabled
        )

    def test_nonstream_known_zero_positive_and_unknown(self):
        for count in (0, 40, None):
            with self.subTest(count=count):
                self.assert_cached(self.response(count), count)

    def test_disabled_reporting_remains_absent(self):
        for count in (0, 40, None):
            with self.subTest(count=count):
                self.assert_cached(self.response(count, enabled=False), None)
                self.assert_cached(self.stream(count, enabled=False), None)

    def stream(self, count, enabled=True):
        cached = {} if count is None else {0: count}
        return UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 100},
            reasoning_tokens={0: 2},
            completion_tokens={0: 7},
            cached_tokens=cached,
            n_choices=1,
            enable_cache_report=enabled,
        )

    def test_stream_known_zero_positive_and_unknown(self):
        for count in (0, 40, None):
            with self.subTest(count=count):
                self.assert_cached(self.stream(count), count)

    def test_multiple_choices_do_not_double_count_prompt_cache(self):
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "cached_tokens": n,
                }
            }
            for n in (0, 99, 40, 99)
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses, n_choices=2, enable_cache_report=True
        )
        self.assert_cached(usage, 40)
        self.assertEqual(usage.prompt_tokens, 200)
        self.assertEqual(usage.completion_tokens, 28)
        streamed = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 100, 1: 100, 2: 100, 3: 100},
            reasoning_tokens={},
            completion_tokens={0: 7, 1: 7, 2: 7, 3: 7},
            cached_tokens={0: 0, 1: 99, 2: 40, 3: 99},
            n_choices=2,
            enable_cache_report=True,
        )
        self.assertEqual(usage.model_dump(), streamed.model_dump())

    def test_unknown_prompt_cache_does_not_become_partial_total(self):
        responses = [
            {
                "meta_info": {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "cached_tokens": 40,
                }
            },
            {"meta_info": {"prompt_tokens": 100, "completion_tokens": 7}},
        ]
        self.assert_cached(
            UsageProcessor.calculate_response_usage(
                responses, enable_cache_report=True
            ),
            None,
        )
        self.assert_cached(
            UsageProcessor.calculate_streaming_usage(
                prompt_tokens={0: 100, 1: 100},
                reasoning_tokens={},
                completion_tokens={0: 7, 1: 7},
                cached_tokens={0: 40},
                n_choices=1,
                enable_cache_report=True,
            ),
            None,
        )

    def test_invalid_negative_counts_do_not_become_hits_or_misses(self):
        self.assert_cached(self.response(-1), None)
        self.assert_cached(self.stream(-1), None)
        self.assert_cached(
            UsageProcessor.calculate_response_usage(
                [
                    {"meta_info": {"cached_tokens": 40}},
                    {"meta_info": {"cached_tokens": -1}},
                ],
                enable_cache_report=True,
            ),
            None,
        )

    def test_explicit_null_empty_and_unrelated_stream_counts_stay_unknown(self):
        self.assert_cached(
            UsageProcessor.calculate_response_usage(
                [{"meta_info": {"cached_tokens": None}}], enable_cache_report=True
            ),
            None,
        )
        self.assert_cached(
            UsageProcessor.calculate_response_usage([], enable_cache_report=True),
            None,
        )
        for prompts, cached in (({}, {}), ({0: 100}, {0: None}), ({0: 100}, {1: 40})):
            with self.subTest(prompts=prompts, cached=cached):
                self.assert_cached(
                    UsageProcessor.calculate_streaming_usage(
                        prompt_tokens=prompts,
                        reasoning_tokens={},
                        completion_tokens={},
                        cached_tokens=cached,
                        n_choices=1,
                        enable_cache_report=True,
                    ),
                    None,
                )

    def test_multimodal_details_do_not_invent_cache_counts(self):
        for enabled in (False, True):
            for count in (None, 0, 40):
                with self.subTest(enabled=enabled, count=count):
                    meta = {} if count is None else {"cached_tokens": count}
                    usage = UsageProcessor.calculate_response_usage(
                        [{"meta_info": meta}],
                        enable_cache_report=enabled,
                        image_tokens=11,
                        audio_tokens=12,
                        video_tokens=13,
                    )
                    expected = {
                        "image_tokens": 11,
                        "audio_tokens": 12,
                        "video_tokens": 13,
                    }
                    if enabled and count is not None:
                        expected["cached_tokens"] = count
                    self.assertEqual(
                        usage.model_dump()["prompt_tokens_details"], expected
                    )

    def test_continuous_usage_has_same_known_unknown_contract(self):
        server = object.__new__(serving_chat.OpenAIServingChat)
        for enabled in (False, True):
            for count in (0, 40, None):
                with self.subTest(enabled=enabled, count=count):
                    meta = {} if count is None else {"cached_tokens": count}
                    with patch.object(
                        serving_chat,
                        "get_serving",
                        return_value=SimpleNamespace(enable_cache_report=enabled),
                    ):
                        details = server._continuous_usage_cached_details(
                            {"meta_info": meta}
                        )
                    if enabled and count is not None:
                        if details is None:
                            self.fail("Known native cache count was omitted")
                        self.assertEqual(details.cached_tokens, count)
                    else:
                        self.assertIsNone(details)


if __name__ == "__main__":
    unittest.main()
