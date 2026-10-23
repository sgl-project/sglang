"""Reject malformed V4.1 calls visibly without GPU inference or tool execution."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask

from sglang.srt.entrypoints.openai import serving_chat
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, Tool
from sglang.srt.function_call.core_types import ToolCallParseError
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MARKER = "\uff5cDSML\uff5c"
TOOLS = [
    Tool(
        type="function",
        function={
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "max_output_tokens": {"type": "integer"},
                },
                "required": ["cmd"],
            },
        },
    )
]


def invoke(body, name="exec_command"):
    return f'<{MARKER} invoke name="{name}">\n{body}\n</{MARKER} invoke>'


def block(*calls):
    return f"<{MARKER} calls>\n" + "\n".join(calls) + f"\n</{MARKER} calls>"


def parameter(name, value, string="true"):
    return f'<{MARKER} parameter name="{name}" string="{string}">{value}</{MARKER} parameter>'


VALID_BODY = (
    parameter("cmd", "must-not-execute")
    + "\n"
    + parameter("max_output_tokens", "60000", "false")
)
# Exact malformed header from the live failure, with all command content removed.
MALFORMED_BODY = parameter("cmd", "must-not-execute") + (
    f'\n<{MARKER} parameter name="max_output_tokens": 60000</{MARKER} parameter>'
)
MALFORMED = block(invoke(MALFORMED_BODY))


class ToolParseErrorTest(CustomTestCase):
    def parser(self, tools=TOOLS):
        return FunctionCallParser(tools, "deepseekv41")

    def test_observed_header_is_not_silently_dropped(self):
        with self.assertRaisesRegex(
            ToolCallParseError, "Model generated.*malformed"
        ) as caught:
            self.parser().parse_non_stream(MALFORMED)
        self.assertNotIn("must-not-execute", str(caught.exception))
        self.assertNotIn(MARKER, str(caught.exception))

    def test_stream_rejects_at_every_chunk_alignment(self):
        for size in (1, 2, 7, 32, len(MALFORMED)):
            with self.subTest(size=size):
                parser = self.parser()
                emitted = []
                with self.assertRaises(ToolCallParseError):
                    for start in range(0, len(MALFORMED), size):
                        _, calls = parser.parse_stream_chunk(
                            MALFORMED[start : start + size]
                        )
                        emitted.extend(calls)
                    parser.parse_stream_end()
                self.assertEqual(emitted, [])

    def test_valid_calls_and_parallel_order_unchanged(self):
        text = block(invoke(VALID_BODY), invoke('{"cmd":"second"}'))
        _, expected = self.parser().parse_non_stream(text)
        self.assertEqual(
            [json.loads(call.parameters) for call in expected],
            [
                {"cmd": "must-not-execute", "max_output_tokens": 60000},
                {"cmd": "second"},
            ],
        )
        for size in (1, 7, len(text)):
            parser = self.parser()
            actual = []
            for start in range(0, len(text), size):
                _, calls = parser.parse_stream_chunk(text[start : start + size])
                actual.extend(calls)
            actual.extend(parser.parse_stream_end()[1])
            self.assertEqual(
                [json.loads(call.parameters) for call in actual],
                [json.loads(call.parameters) for call in expected],
            )
            self.assertEqual([call.tool_index for call in actual], [0, 1])

    def test_no_repair_of_broken_json_or_parameter_tags(self):
        for body in (
            '{"cmd":',
            "[]",
            f'<{MARKER} parameter name="cmd">must-not-execute</{MARKER} parameter>',
            parameter("cmd", "first") + MALFORMED_BODY,
        ):
            with self.subTest(body=body), self.assertRaises(ToolCallParseError):
                self.parser().parse_non_stream(block(invoke(body)))

    def test_unclosed_invoke_rejected_at_end(self):
        text = f'<{MARKER} calls>\n<{MARKER} invoke name="exec_command">\n' + VALID_BODY
        with self.assertRaises(ToolCallParseError):
            self.parser().parse_non_stream(text)
        parser = self.parser()
        self.assertEqual(parser.parse_stream_chunk(text)[1], [])
        with self.assertRaises(ToolCallParseError):
            parser.parse_stream_end()

    def test_malformed_invoke_header_rejected(self):
        text = block(
            f"<{MARKER} invoke name=exec_command>{VALID_BODY}</{MARKER} invoke>"
        )
        with self.assertRaises(ToolCallParseError):
            self.parser().parse_non_stream(text)
        parser = self.parser()
        self.assertEqual(parser.parse_stream_chunk(text)[1], [])
        with self.assertRaises(ToolCallParseError):
            parser.parse_stream_end()

    def test_qualified_names_and_zero_argument_calls_unchanged(self):
        tool = Tool(
            type="function",
            function={"name": "ops::inspect", "parameters": {}},
        )
        for text in (
            block(invoke("", "ops::inspect")),
            block(f'<{MARKER} invoke name="ops::inspect"/>'),
        ):
            _, calls = self.parser([tool]).parse_non_stream(text)
            self.assertEqual(
                [(call.name, call.parameters) for call in calls],
                [("ops::inspect", "{}")],
            )

    def test_mixed_valid_and_malformed_headers_never_succeed(self):
        valid = invoke(VALID_BODY)
        malformed = f"<{MARKER} invoke name=exec_command>{VALID_BODY}</{MARKER} invoke>"
        for calls in ((valid, malformed), (malformed, valid)):
            text = block(*calls)
            with self.subTest(calls=calls):
                with self.assertRaises(ToolCallParseError):
                    self.parser().parse_non_stream(text)
                for size in (1, 7, len(text)):
                    parser = self.parser()
                    with self.assertRaises(ToolCallParseError):
                        for start in range(0, len(text), size):
                            parser.parse_stream_chunk(text[start : start + size])
                        parser.parse_stream_end()

    def test_new_request_is_healthy_after_rejection(self):
        with self.assertRaises(ToolCallParseError):
            self.parser().parse_non_stream(MALFORMED)
        self.assertEqual(
            len(self.parser().parse_non_stream(block(invoke(VALID_BODY)))[1]), 1
        )

    def test_preceding_valid_streamed_call_not_replayed(self):
        parser = self.parser()
        _, calls = parser.parse_stream_chunk(block(invoke(VALID_BODY)))
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ToolCallParseError):
            parser.parse_stream_chunk(MALFORMED)
        self.assertEqual(len(calls), 1)

    def test_custom_tool_bridge_never_receives_rejected_arguments(self):
        tools = [
            Tool(
                type="function",
                function={
                    "name": "write_text",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                    },
                },
            )
        ]
        with self.assertRaises(ToolCallParseError):
            self.parser(tools).parse_non_stream(
                block(invoke(MALFORMED_BODY, "write_text"))
            )

    def test_plain_text_and_no_tools_unchanged(self):
        self.assertEqual(
            self.parser().parse_non_stream("No tool is needed."),
            ("No tool is needed.", []),
        )
        self.assertEqual(self.parser([]).parse_non_stream(MALFORMED), (MALFORMED, []))

    def test_other_deepseek_policies_unchanged(self):
        for detector in (DeepSeekV32Detector(), DeepSeekV4Detector()):
            with self.subTest(detector=type(detector).__name__):
                text = MALFORMED
                text = text.replace(" calls", detector.tool_calls_block_name)
                text = text.replace(" invoke", detector.invoke_tag_name)
                text = text.replace(" parameter", detector.parameter_tag_name)
                self.assertFalse(detector.raise_on_malformed_tool_calls)
                self.assertEqual(detector.detect_and_parse(text, TOOLS).calls, [])


class ToolParseServingTest(unittest.IsolatedAsyncioTestCase):
    async def check_response(self, streaming):
        # Execute real serving/error/SSE methods; replace only generation and unrelated
        # model/usage setup. No model, CUDA, gateway or actual tool is invoked.
        server = object.__new__(serving_chat.OpenAIServingChat)
        server.tool_call_parser = "deepseekv41"
        server.reasoning_parser = None
        generated, aborted = [], []

        async def generate_request(*args):
            generated.append(True)
            yield {
                "text": MALFORMED,
                "meta_info": {
                    "id": "rejected-call",
                    "prompt_tokens": 12,
                    "completion_tokens": 20,
                    "finish_reason": {"type": "stop", "matched": 1},
                },
            }

        server.tokenizer_manager = SimpleNamespace(
            tokenizer=None,
            generate_request=generate_request,
            create_abort_task=lambda _: BackgroundTask(lambda: aborted.append(True)),
        )
        server._should_return_input_ids = lambda _: False
        server._should_return_output_ids = lambda _: False
        server._reported_prompt_tokens = lambda meta: meta["prompt_tokens"]
        server._build_chat_response = lambda request, ret, created: (
            server._process_tool_calls(
                ret[0]["text"], request.tools, ret[0]["meta_info"]["finish_reason"]
            )
        )

        async def generate_content(**kwargs):
            async for chunk in server._process_tool_call_stream(
                index=kwargs["index"],
                delta=kwargs["content"]["text"],
                parser_dict=kwargs["parser_dict"],
                content=kwargs["content"],
                request=kwargs["request"],
                has_tool_calls=kwargs["has_tool_calls"],
                created=kwargs["created"],
                flush=True,
            ):
                yield chunk

        server._generate_stream_content = generate_content
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(raw: Request):
            request = ChatCompletionRequest(**await raw.json())
            handler = (
                server._handle_streaming_request
                if request.stream
                else (server._handle_non_streaming_request)
            )
            return await handler(SimpleNamespace(), request, raw)

        config = SimpleNamespace(stream_response_default_include_usage=False)
        with patch.object(serving_chat, "get_serving", return_value=config):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "Use a tool."}],
                        "tools": [tool.model_dump() for tool in TOOLS],
                        "stream": streaming,
                    },
                )
        self.assertEqual(generated, [True])  # No implicit whole-response retry.
        if streaming:
            self.assertEqual(
                response.status_code, 200
            )  # Headers preceded the parser failure.
            data = [
                line[6:]
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            self.assertEqual(data[-1], "[DONE]")
            events = [json.loads(item) for item in data[:-1]]
            errors = [event["error"] for event in events if "error" in event]
            self.assertEqual(len(errors), 1)
            self.assertFalse(
                any(
                    choice.get("finish_reason")
                    for event in events
                    for choice in event.get("choices", [])
                )
            )
            self.assertFalse(
                any(
                    choice.get("delta", {}).get("tool_calls")
                    for event in events
                    for choice in event.get("choices", [])
                )
            )
            error = errors[0]
            self.assertEqual(aborted, [True])
        else:
            self.assertEqual(response.status_code, 400)
            error = response.json()
        self.assertEqual(error["type"], "ToolCallParseError")
        self.assertEqual(error["code"], 400)
        self.assertIn("Model generated a malformed", error["message"])
        self.assertNotIn("must-not-execute", response.text)
        self.assertNotIn(MARKER, response.text)

    async def test_nonstream_reports_error_not_success(self):
        await self.check_response(False)

    async def test_stream_reports_error_and_aborts_generation(self):
        await self.check_response(True)


if __name__ == "__main__":
    unittest.main()
