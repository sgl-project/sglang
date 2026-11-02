"""Replay fixed model text through real parsers and OpenAI response builders."""

import asyncio
import json
import unittest
from unittest.mock import Mock

from utils import (
    MockTemplateManager,
    MockTokenizerManager,
    StreamFixture,
    engine_chunk,
    event_payloads,
    make_serving,
)

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestInvalidToolJSON(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", tool_call_parser="qwen"), role="tokenizer"
        )
        self.function = {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        }
        self.wire = (
            "I'll edit the file.\n<tool_call>\n<function=edit>\n"
            "<parameter=text>synthetic payload</parameter>\n</function>\n</tool_call>"
        )

    def chat(self, stream):
        manager = MockTokenizerManager()
        manager.server_args.tool_call_parser = "qwen"
        chat = OpenAIServingChat(manager, MockTemplateManager())
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Edit the fixture."}],
            tools=[{"type": "function", "function": self.function}],
            stream=stream,
        )

        async def generate(*args, **kwargs):
            chunk = engine_chunk(self.wire, finish=True)
            chunk["meta_info"]["weight_version"] = "default"
            yield {**chunk, "index": 0}

        manager.generate_request = generate
        return chat, request, GenerateReqInput(input_ids=[1], sampling_params={})

    def test_chat_non_streaming_reports_server_error(self):
        chat, request, adapted = self.chat(stream=False)
        response = asyncio.run(
            chat._handle_non_streaming_request(adapted, request, Mock())
        )
        self.assertEqual(response.status_code, 500)
        error = json.loads(response.body)
        self.assertEqual(error["type"], "InternalServerError")
        self.assertIn("Invalid JSON", error["message"])
        self.assertNotIn("synthetic payload", error["message"])

    def test_chat_streaming_reports_error_without_successful_finish(self):
        chat, request, adapted = self.chat(stream=True)

        async def collect():
            return [
                chunk
                async for chunk in chat._generate_chat_stream(adapted, request, Mock())
            ]

        chunks = asyncio.run(collect())
        events = [
            json.loads(c[6:])
            for c in chunks
            if c.startswith("data: ") and c != "data: [DONE]\n\n"
        ]
        errors = [event["error"] for event in events if "error" in event]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["code"], 500)
        self.assertIn("Invalid JSON", errors[0]["message"])
        self.assertFalse(
            any(c.get("finish_reason") for e in events for c in e.get("choices", []))
        )
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

    def responses_request(self):
        return ResponsesRequest(
            model="x",
            input="Edit the fixture.",
            store=False,
            tools=[{"type": "function", **self.function}],
        )

    def test_responses_non_streaming_reports_server_error(self):
        serving = make_serving()
        serving.tool_call_parser = "qwen"
        request = self.responses_request()
        context = SimpleContext()
        context.last_output = engine_chunk(self.wire, finish=True)

        async def empty_generator():
            for _ in ():
                yield None

        response = asyncio.run(
            serving.responses_full_generator(
                request,
                sampling_params={},
                result_generator=empty_generator(),
                context=context,
                model_name="x",
                tokenizer=serving.tokenizer_manager.tokenizer,
                request_metadata=RequestResponseMetadata(request_id=request.request_id),
                require_reasoning=False,
            )
        )
        self.assertEqual(response.status_code, 500)
        error = json.loads(response.body)["error"]
        self.assertEqual(error["type"], "server_error")
        self.assertIn("Invalid JSON", error["message"])

    def test_responses_streaming_fails_instead_of_completing(self):
        serving = make_serving()
        serving.tool_call_parser = "qwen"
        events = event_payloads(
            StreamFixture(serving, self.responses_request()).run(
                [
                    engine_chunk(self.wire, finish=True),
                ]
            )
        )
        failed = [event for event in events if event["type"] == "response.failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["response"]["error"]["code"], "server_error")
        self.assertIn("Invalid JSON", failed[0]["response"]["error"]["message"])
        self.assertNotIn("response.completed", [event["type"] for event in events])


if __name__ == "__main__":
    unittest.main()
