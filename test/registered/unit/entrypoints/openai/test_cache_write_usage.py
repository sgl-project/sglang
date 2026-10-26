"""Measured cache writes must retain their meaning on the API wire."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.anthropic.serving import (  # noqa: E402
    _anthropic_usage_from_openai,
)
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ResponsesResponse,
)
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCacheWriteUsage(CustomTestCase):
    def _usage(self, value):
        return UsageProcessor.calculate_response_usage(
            [
                {
                    "meta_info": {
                        "prompt_tokens": 20,
                        "completion_tokens": 2,
                        "cached_tokens": 4,
                        "cache_write_tokens": value,
                    }
                }
            ],
            enable_cache_report=True,
        )

    def test_unknown_zero_and_measured_writes(self):
        for value in (None, 0, 16):
            with self.subTest(value=value):
                usage = self._usage(value)
                details = usage.model_dump()["prompt_tokens_details"]
                if value is None:
                    self.assertNotIn("cache_write_tokens", details)
                else:
                    self.assertEqual(details["cache_write_tokens"], value)
                response = ResponsesResponse.model_construct(usage=usage)
                details = response.model_dump()["usage"]["input_tokens_details"]
                self.assertEqual(details["cache_write_tokens"], value or 0)
                self.assertEqual(
                    details["cache_write_tokens_reported"], value is not None
                )
                anthropic = _anthropic_usage_from_openai(
                    usage, include_input=True, include_output=True
                )
                self.assertEqual(anthropic.input_tokens, 16 - (value or 0))
                self.assertEqual(anthropic.cache_creation_input_tokens, value)

    def test_terminal_stream_snapshot_replaces_earlier_counts(self):
        for value in (8, 16):
            usage = UsageProcessor.calculate_streaming_usage(
                prompt_tokens={0: 20},
                reasoning_tokens={0: 0},
                completion_tokens={0: 2},
                cached_tokens={0: 4},
                cache_write_tokens={0: value},
                n_choices=1,
                enable_cache_report=True,
            )
            self.assertEqual(usage.prompt_tokens_details.cache_write_tokens, value)

    def test_invalid_or_per_choice_totals_remain_unknown(self):
        for value in (-1, True, 17):
            self.assertIsNone(
                self._usage(value).prompt_tokens_details.cache_write_tokens
            )
        usage = UsageProcessor.calculate_response_usage(
            [
                {
                    "meta_info": {
                        "prompt_tokens": 20,
                        "cached_tokens": 4,
                        "cache_write_tokens": 16,
                    }
                }
            ]
            * 2,
            n_choices=2,
            enable_cache_report=True,
        )
        self.assertIsNone(usage.prompt_tokens_details.cache_write_tokens)

    def test_encoding_output_multi_choice_continuous_writes_remain_unknown(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        for n in (1, 2):
            with self.subTest(n=n):
                server = OpenAIServingChat.__new__(OpenAIServingChat)
                server.chat_encoding_spec = None
                seen = []
                adapter = SimpleNamespace(
                    stream_chat_chunks=lambda **kwargs: (
                        seen.append(kwargs["usage"]) or []
                    )
                )
                server._encoding_output = lambda: adapter
                request = ChatCompletionRequest(
                    model="test", messages=[{"role": "user", "content": "test"}], n=n
                )

                async def collect():
                    async for _ in server._generate_stream_content(
                        {
                            "meta_info": {
                                "prompt_tokens": 20,
                                "cached_tokens": 4,
                                "cache_write_tokens": 16,
                            }
                        },
                        0,
                        request,
                        0,
                        {},
                        {},
                        {},
                        {},
                        None,
                        None,
                        True,
                        {},
                        {},
                        {},
                    ):
                        pass

                with patch(
                    "sglang.srt.entrypoints.openai.serving_chat.get_serving",
                    return_value=SimpleNamespace(enable_cache_report=True),
                ):
                    asyncio.run(collect())
                self.assertEqual(len(seen), 1)
                details = seen[0]["prompt_tokens_details"]
                if n == 1:
                    self.assertEqual(details["cache_write_tokens"], 16)
                else:
                    self.assertNotIn("cache_write_tokens", details)


if __name__ == "__main__":
    unittest.main()
