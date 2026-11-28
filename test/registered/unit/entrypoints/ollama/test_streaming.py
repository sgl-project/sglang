import asyncio
import json
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi import Request
from fastapi.responses import StreamingResponse

from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaChatStreamResponse,
    OllamaGenerateRequest,
    OllamaGenerateStreamResponse,
    OllamaMessage,
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeTokenizerManager:
    served_model_name = "test-model"

    def __init__(self, texts: list[str], incremental: bool):
        self.texts = texts
        self.incremental_streaming_output = incremental
        self.tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: [1, 2, 3]
        )

    async def generate_request(
        self, request: GenerateReqInput, raw_request: Request
    ) -> AsyncIterator[dict]:
        for index, text in enumerate(self.texts):
            yield {
                "text": text,
                "meta_info": {
                    "finish_reason": (
                        {"type": "stop"} if index == len(self.texts) - 1 else None
                    )
                },
            }


class TestOllamaStreaming(CustomTestCase):
    async def _assert_stream(
        self,
        endpoint: str,
        incremental: bool,
        texts: list[str],
        expected: list[str],
        event_count: int | None,
    ) -> None:
        serving = OllamaServing(FakeTokenizerManager(texts, incremental))
        raw_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": f"/api/{endpoint}",
                "headers": [],
            }
        )
        if endpoint == "chat":
            request = OllamaChatRequest(
                model="test-model",
                messages=[OllamaMessage(role="user", content="Hi")],
                stream=True,
            )
            response = await serving.handle_chat(request, raw_request)
        else:
            request = OllamaGenerateRequest(
                model="test-model", prompt="Hi", stream=True
            )
            response = await serving.handle_generate(request, raw_request)

        self.assertIsInstance(response, StreamingResponse)
        self.assertEqual(response.media_type, "application/x-ndjson")
        events = []
        contents = []
        async for line in response.body_iterator:
            self.assertIsInstance(line, bytes)
            self.assertTrue(line.endswith(b"\n"))
            event = json.loads(line)
            events.append(event)
            if endpoint == "chat":
                parsed = OllamaChatStreamResponse.model_validate(event)
                self.assertEqual(parsed.message.role, "assistant")
                contents.append(parsed.message.content)
            else:
                parsed = OllamaGenerateStreamResponse.model_validate(event)
                contents.append(parsed.response)
            self.assertEqual(parsed.model, "test-model")
            self.assertTrue(parsed.created_at)
            self.assertIsInstance(event["done"], bool)

        self.assertTrue(events)
        self.assertEqual(
            [event["done"] for event in events], [False] * (len(events) - 1) + [True]
        )
        self.assertEqual(contents[-1], "")
        self.assertEqual(events[-1]["done_reason"], "stop")
        self.assertEqual([text for text in contents if text], expected)
        self.assertEqual("".join(contents), "".join(expected))
        if event_count is not None:
            self.assertEqual(len(events), event_count)

    def _check_modes(
        self,
        cumulative: list[str],
        incremental: list[str],
        expected: list[str],
        event_count: int | None = None,
    ) -> None:
        for endpoint in ("chat", "generate"):
            for is_incremental, texts in ((False, cumulative), (True, incremental)):
                with self.subTest(endpoint=endpoint, incremental=is_incremental):
                    asyncio.run(
                        self._assert_stream(
                            endpoint, is_incremental, texts, expected, event_count
                        )
                    )

    def test_empty_terminal_reply(self):
        self._check_modes(
            ["Hello", "Hello ", "Hello world", "Hello world"],
            ["Hello", " ", "world", ""],
            ["Hello", " ", "world"],
        )

    def test_text_in_terminal_reply(self):
        self._check_modes(
            ["Hello", "Hello world"], ["Hello", " world"], ["Hello", " world"]
        )

    def test_single_terminal_reply_with_text(self):
        self._check_modes(["Hello"], ["Hello"], ["Hello"], event_count=2)

    def test_single_empty_terminal_reply(self):
        self._check_modes([""], [""], [], event_count=1)

    def test_unicode_and_empty_replies(self):
        self._check_modes(
            ["", "你", "你🙂", "你🙂", "你🙂世界"],
            ["", "你", "🙂", "", "世界"],
            ["你", "🙂", "世界"],
        )

    def test_repeated_text(self):
        self._check_modes(
            ["哈", "哈哈", "哈哈🙂"], ["哈", "哈", "🙂"], ["哈", "哈", "🙂"]
        )


if __name__ == "__main__":
    unittest.main()
