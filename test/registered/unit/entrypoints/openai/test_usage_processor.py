import json
import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionStreamResponse,
    UsageInfo,
)
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

    def test_reasoning_tokens_in_completion_tokens_details(self):
        # OpenAI clients and gateways read usage.completion_tokens_details.reasoning_tokens;
        # the top-level reasoning_tokens stays for existing clients.
        usages = (
            UsageProcessor.calculate_response_usage(
                [{"meta_info": {"completion_tokens": 58, "reasoning_tokens": 55}}]
            ),
            UsageProcessor.calculate_streaming_usage(
                prompt_tokens={0: 45},
                reasoning_tokens={0: 55},
                completion_tokens={0: 58},
                cached_tokens={0: 0},
                n_choices=1,
            ),
            UsageProcessor.calculate_token_usage(
                prompt_tokens=45, completion_tokens=58, reasoning_tokens=55
            ),
        )
        for usage in usages:
            with self.subTest(usage=usage):
                self.assertEqual(usage.reasoning_tokens, 55)
                self.assertEqual(usage.completion_tokens_details.reasoning_tokens, 55)
                dumped = usage.model_dump()
                self.assertEqual(
                    dumped["completion_tokens_details"]["reasoning_tokens"], 55
                )
                self.assertLessEqual(
                    usage.completion_tokens_details.reasoning_tokens,
                    usage.completion_tokens,
                )

    def test_completion_tokens_details_present_without_reasoning(self):
        # Always an object (OpenAI types it as one, not nullable), with 0 reasoning tokens.
        for usage in (
            UsageProcessor.calculate_response_usage(
                [{"meta_info": {"completion_tokens": 3}}]
            ),
            UsageProcessor.calculate_streaming_usage(
                prompt_tokens={0: 1},
                reasoning_tokens={0: 0},
                completion_tokens={0: 3},
                cached_tokens={0: 0},
                n_choices=1,
            ),
            UsageProcessor.calculate_token_usage(prompt_tokens=1, completion_tokens=3),
        ):
            with self.subTest(usage=usage):
                self.assertEqual(usage.reasoning_tokens, 0)
                self.assertEqual(
                    usage.model_dump()["completion_tokens_details"],
                    {"reasoning_tokens": 0},
                )

    def test_completion_tokens_details_sums_parallel_choices(self):
        # n=2: like completion_tokens, the detail aggregates every choice.
        usages = (
            UsageProcessor.calculate_response_usage(
                [
                    {"meta_info": {"completion_tokens": 9, "reasoning_tokens": 5}},
                    {"meta_info": {"completion_tokens": 11, "reasoning_tokens": 7}},
                ],
                n_choices=2,
            ),
            UsageProcessor.calculate_streaming_usage(
                prompt_tokens={0: 4, 1: 4},
                reasoning_tokens={0: 5, 1: 7},
                completion_tokens={0: 9, 1: 11},
                cached_tokens={0: 0, 1: 0},
                n_choices=2,
            ),
        )
        for usage in usages:
            with self.subTest(usage=usage):
                self.assertEqual(usage.completion_tokens, 20)
                self.assertEqual(usage.completion_tokens_details.reasoning_tokens, 12)
                self.assertEqual(usage.reasoning_tokens, 12)

    def test_completion_tokens_details_never_exceed_completion_tokens(self):
        # The parser's count can exceed the completion; the standard field is
        # clamped, the top-level field keeps the raw count.
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=5, completion_tokens=48, reasoning_tokens=51
        )
        self.assertEqual(usage.completion_tokens_details.reasoning_tokens, 48)
        self.assertEqual(usage.reasoning_tokens, 51)

    def test_usage_without_generation_omits_completion_tokens_details(self):
        # Embeddings, score and classify build UsageInfo directly; the key is left
        # out rather than sent as null, which OpenAI's schema does not allow.
        usage = UsageInfo(prompt_tokens=3, total_tokens=3)
        self.assertNotIn("completion_tokens_details", usage.model_dump())
        self.assertNotIn("completion_tokens_details", usage.model_dump_json())

    def test_chat_response_and_final_stream_chunk_carry_the_detail(self):
        # The serialized wire shapes, not only the UsageInfo helper.
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=45, completion_tokens=58, reasoning_tokens=55
        )
        response = ChatCompletionResponse(
            id="x",
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
        chunk = ChatCompletionStreamResponse(id="x", model="m", choices=[], usage=usage)
        completion_chunk = CompletionStreamResponse(
            id="x", model="m", choices=[], usage=usage
        )
        for payload in (
            response.model_dump_json(),
            chunk.model_dump_json(),
            # The completions final chunk is dumped without None fields.
            completion_chunk.model_dump_json(exclude_none=True),
        ):
            with self.subTest(payload=payload[:40]):
                self.assertEqual(
                    json.loads(payload)["usage"]["completion_tokens_details"],
                    {"reasoning_tokens": 55},
                )


if __name__ == "__main__":
    unittest.main()
