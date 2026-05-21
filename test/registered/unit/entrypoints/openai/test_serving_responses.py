from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import json
import unittest
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class _MockTokenizerManager:
    def __init__(self):
        self.model_config = Mock(
            is_multimodal=False,
            context_len=4096,
        )
        self.model_config.get_default_sampling_params = Mock(return_value={})
        self.model_config.hf_config = Mock(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
        )
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            stream_response_default_include_usage=False,
            tokenizer_metrics_allowed_custom_labels=None,
            context_length=4096,
            allow_auto_truncate=False,
            incremental_streaming_output=True,
        )
        self.num_reserved_tokens = 0
        self.request_logger = Mock(log_requests=False, log_requests_level=0)
        self.create_abort_task = Mock(return_value=None)
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.decode.return_value = "decoded"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1


class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None
        self.force_reasoning = False


class ServingResponsesTestCase(unittest.TestCase):
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.responses = OpenAIServingResponses(self.tm, self.template_manager)
        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    def _collect_async_iterable(self, async_iterable):
        async def run():
            items = []
            async for item in async_iterable:
                items.append(item)
            return items

        loop = get_or_create_event_loop()
        return loop.run_until_complete(run())

    def _parse_response_stream(self, chunks):
        events = []
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                events.append("[DONE]")
                continue
            lines = chunk.strip().splitlines()
            event_type = None
            data = None
            for line in lines:
                if line.startswith("event: "):
                    event_type = line[len("event: ") :]
                elif line.startswith("data: "):
                    data = json.loads(line[len("data: ") :])
            if event_type is not None:
                events.append((event_type, data))
        return events

    def test_handle_raw_request_uses_native_path_for_native_payload(self):
        payload = {"model": "x", "input": "hello", "stream": True}
        native_result = Mock()

        with patch.object(
            self.responses, "create_responses", AsyncMock(return_value=native_result)
        ) as native_mock, patch.object(
            self.responses,
            "_handle_bridged_request",
            AsyncMock(return_value=Mock()),
        ) as bridge_mock:
            result = get_or_create_event_loop().run_until_complete(
                self.responses.handle_raw_request(payload, self.fastapi_request)
            )

        self.assertIs(result, native_result)
        native_mock.assert_awaited_once()
        bridge_mock.assert_not_awaited()

    def test_handle_raw_request_routes_codex_payloads_to_bridge(self):
        payloads = [
            {
                "model": "x",
                "input": "hello",
                "stream": True,
                "text": {
                    "format": {"type": "json_schema", "schema": {"type": "object"}}
                },
            },
            {
                "model": "x",
                "input": "hello",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object"},
                    }
                ],
            },
            {
                "model": "x",
                "input": "hello",
                "stream": True,
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "functions": [
                            {"name": "exec_command", "parameters": {"type": "object"}}
                        ],
                    }
                ],
            },
        ]

        for payload in payloads:
            with self.subTest(payload=payload), patch.object(
                self.responses,
                "_handle_bridged_request",
                AsyncMock(return_value="bridge"),
            ) as bridge_mock, patch.object(
                self.responses, "create_responses", AsyncMock(return_value="native")
            ) as native_mock:
                result = get_or_create_event_loop().run_until_complete(
                    self.responses.handle_raw_request(payload, self.fastapi_request)
                )
                self.assertEqual(result, "bridge")
                bridge_mock.assert_awaited_once()
                native_mock.assert_not_awaited()

    def test_translate_instructions_and_text_input(self):
        request, _, _ = self.responses._translate_bridged_request(
            {
                "model": "x",
                "stream": True,
                "instructions": "Be terse.",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    }
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                        },
                    }
                },
                "reasoning": {"effort": "low"},
            }
        )

        self.assertEqual(request.messages[0].role, "system")
        self.assertEqual(request.messages[0].content, "Be terse.")
        self.assertEqual(request.messages[1].role, "user")
        self.assertEqual(request.messages[1].content, "hello")
        self.assertEqual(request.reasoning_effort, "low")
        self.assertIsNotNone(request.response_format)
        self.assertEqual(request.response_format.type, "json_schema")

    def test_translate_function_call_output_history(self):
        messages = self.responses._translate_bridged_messages(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": {"temperature": "21C"},
                },
            ],
            instructions=None,
        )

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"], '{"city": "Paris"}'
        )
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "call_1")
        self.assertEqual(messages[1]["content"], '{"temperature": "21C"}')

    def test_translate_namespace_tools_and_reverse_map(self):
        request, namespace_map, bridge_warnings = (
            self.responses._translate_bridged_request(
                {
                    "model": "x",
                    "stream": True,
                    "input": "hello",
                    "tools": [
                        {
                            "type": "namespace",
                            "namespace": "functions",
                            "functions": [
                                {
                                    "name": "exec_command",
                                    "description": "Run a command",
                                    "parameters": {"type": "object", "properties": {}},
                                }
                            ],
                        }
                    ],
                }
            )
        )

        self.assertEqual(request.tools[0].function.name, "functions.exec_command")
        self.assertEqual(
            namespace_map["functions.exec_command"],
            {"namespace": "functions", "name": "exec_command"},
        )
        self.assertEqual(bridge_warnings, [])

    def test_translate_custom_tool_to_string_input_schema(self):
        request, _, _ = self.responses._translate_bridged_request(
            {
                "model": "x",
                "stream": True,
                "input": "hello",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "Patch files",
                    }
                ],
            }
        )

        parameters = request.tools[0].function.parameters
        self.assertEqual(parameters["required"], ["input"])
        self.assertEqual(parameters["properties"]["input"]["type"], "string")

    def test_translate_skips_web_search_tool_with_warning(self):
        request, _, bridge_warnings = self.responses._translate_bridged_request(
            {
                "model": "x",
                "stream": True,
                "input": "hello",
                "tools": [
                    {"type": "web_search"},
                    {
                        "type": "function",
                        "name": "exec_command",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            }
        )

        self.assertEqual(len(request.tools), 1)
        self.assertEqual(request.tools[0].function.name, "exec_command")
        self.assertEqual(
            bridge_warnings,
            [
                "Skipping unsupported tool type 'web_search' in /v1/responses bridge mode."
            ],
        )

    def test_translate_skips_image_generation_tool_with_warning(self):
        request, _, bridge_warnings = self.responses._translate_bridged_request(
            {
                "model": "x",
                "stream": True,
                "input": "hello",
                "tools": [
                    {"type": "image_generation"},
                    {
                        "type": "function",
                        "name": "exec_command",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            }
        )

        self.assertEqual(len(request.tools), 1)
        self.assertEqual(request.tools[0].function.name, "exec_command")
        self.assertEqual(
            bridge_warnings,
            [
                "Skipping unsupported tool type 'image_generation' in /v1/responses bridge mode."
            ],
        )

    def test_translate_rejects_tool_choice_for_skipped_web_search(self):
        with self.assertRaisesRegex(
            ValueError, "tool_choice selects unsupported tool type 'web_search'"
        ):
            self.responses._translate_bridged_request(
                {
                    "model": "x",
                    "stream": True,
                    "input": "hello",
                    "tools": [{"type": "web_search"}],
                    "tool_choice": {"type": "web_search"},
                }
            )

    def test_translate_rejects_tool_choice_for_skipped_image_generation(self):
        with self.assertRaisesRegex(
            ValueError,
            "tool_choice selects unsupported tool type 'image_generation'",
        ):
            self.responses._translate_bridged_request(
                {
                    "model": "x",
                    "stream": True,
                    "input": "hello",
                    "tools": [{"type": "image_generation"}],
                    "tool_choice": {"type": "image_generation"},
                }
            )

    def test_bridge_stream_adapts_plain_text_chat_stream(self):
        chat_request = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hello"}], stream=True
        )

        async def fake_chat_stream():
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[{"index":0,"finish_reason":"stop","delta":{}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"reasoning_tokens":1,"prompt_tokens_details":{"cached_tokens":1}}}\n\n'
            yield "data: [DONE]\n\n"

        chunks = self._collect_async_iterable(
            self.responses._bridge_chat_stream(
                fake_chat_stream(),
                {"model": "x", "stream": True},
                chat_request,
                response_id="resp_test",
                created_time=1,
                namespace_map={},
                bridge_warnings=[],
            )
        )
        events = self._parse_response_stream(chunks)

        self.assertEqual(
            [event[0] if isinstance(event, tuple) else event for event in events],
            [
                "response.created",
                "response.output_item.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_item.done",
                "response.completed",
                "[DONE]",
            ],
        )
        completed = events[-2][1]["response"]
        self.assertEqual(completed["usage"]["input_tokens"], 3)
        self.assertEqual(completed["usage"]["output_tokens"], 2)
        self.assertEqual(
            completed["usage"]["output_tokens_details"]["reasoning_tokens"], 1
        )

    def test_bridge_stream_buffers_tool_call_arguments(self):
        chat_request = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hello"}], stream=True
        )

        async def fake_chat_stream():
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"exec_command","arguments":"{\\"cmd\\""}}]}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"ls\\"}"}}]}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
            yield "data: [DONE]\n\n"

        chunks = self._collect_async_iterable(
            self.responses._bridge_chat_stream(
                fake_chat_stream(),
                {"model": "x", "stream": True},
                chat_request,
                response_id="resp_test",
                created_time=1,
                namespace_map={},
                bridge_warnings=[],
            )
        )
        events = self._parse_response_stream(chunks)

        self.assertEqual(events[1][0], "response.output_item.done")
        self.assertEqual(events[1][1]["item"]["type"], "function_call")
        self.assertEqual(events[1][1]["item"]["call_id"], "call_1")
        self.assertEqual(events[1][1]["item"]["name"], "exec_command")
        self.assertEqual(events[1][1]["item"]["arguments"], '{"cmd":"ls"}')

    def test_bridge_stream_restores_namespace_on_tool_call(self):
        chat_request = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hello"}], stream=True
        )

        async def fake_chat_stream():
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"functions.exec_command","arguments":"{}"}}]}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
            yield "data: [DONE]\n\n"

        chunks = self._collect_async_iterable(
            self.responses._bridge_chat_stream(
                fake_chat_stream(),
                {"model": "x", "stream": True},
                chat_request,
                response_id="resp_test",
                created_time=1,
                namespace_map={
                    "functions.exec_command": {
                        "namespace": "functions",
                        "name": "exec_command",
                    }
                },
                bridge_warnings=[],
            )
        )
        events = self._parse_response_stream(chunks)

        self.assertEqual(events[1][1]["item"]["namespace"], "functions")
        self.assertEqual(events[1][1]["item"]["name"], "exec_command")

    def test_bridge_stream_surfaces_bridge_warnings_in_metadata(self):
        chat_request = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hello"}], stream=True
        )

        async def fake_chat_stream():
            yield 'data: {"id":"chat_1","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
            yield "data: [DONE]\n\n"

        chunks = self._collect_async_iterable(
            self.responses._bridge_chat_stream(
                fake_chat_stream(),
                {"model": "x", "stream": True, "metadata": {"source": "codex"}},
                chat_request,
                response_id="resp_test",
                created_time=1,
                namespace_map={},
                bridge_warnings=[
                    "Skipping unsupported tool type 'web_search' in /v1/responses bridge mode."
                ],
            )
        )
        events = self._parse_response_stream(chunks)

        created = events[0][1]["response"]
        completed = events[1][1]["response"]
        self.assertEqual(created["metadata"]["source"], "codex")
        self.assertIn("bridge_warnings", created["metadata"])
        self.assertEqual(created["metadata"], completed["metadata"])

    def test_convert_chat_error_response_uses_nested_responses_error(self):
        chat_error = OpenAIServingChat.create_error_response(
            self.responses, "boom", err_type="BadRequestError", status_code=422
        )

        response = self.responses._convert_chat_error_response(chat_error)
        payload = json.loads(response.body)

        self.assertEqual(payload["error"]["message"], "boom")
        self.assertEqual(payload["error"]["type"], "BadRequestError")
        self.assertEqual(payload["error"]["code"], 422)

    def test_v1_responses_route_streams_bridge_events_in_order(self):
        app = FastAPI()
        app.state.openai_serving_responses = self.responses

        @app.post("/v1/responses")
        async def responses_route(request: dict, raw_request: Request):
            result = (
                await raw_request.app.state.openai_serving_responses.handle_raw_request(
                    request, raw_request
                )
            )
            if hasattr(result, "__aiter__"):
                return StreamingResponse(result, media_type="text/event-stream")
            return result

        async def fake_chat_stream():
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[{"index":0,"delta":{"content":" there"}}]}\n\n'
            yield 'data: {"id":"chat_1","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}\n\n'
            yield "data: [DONE]\n\n"

        chat_request = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hello"}], stream=True
        )

        with patch.object(
            OpenAIServingChat,
            "_convert_to_internal_request",
            return_value=(Mock(), chat_request),
        ), patch.object(
            OpenAIServingChat,
            "_handle_streaming_request",
            AsyncMock(
                return_value=StreamingResponse(
                    fake_chat_stream(), media_type="text/event-stream"
                )
            ),
        ):
            client = TestClient(app)
            with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "x",
                    "input": "hello",
                    "stream": True,
                    "text": {"format": {"type": "text"}},
                },
            ) as response:
                lines = [
                    line.decode("utf-8") if isinstance(line, bytes) else line
                    for line in response.iter_lines()
                ]

        event_lines = [line for line in lines if line.startswith("event: ")]
        data_lines = [line for line in lines if line == "data: [DONE]"]

        self.assertEqual(
            event_lines,
            [
                "event: response.created",
                "event: response.output_item.added",
                "event: response.output_text.delta",
                "event: response.output_text.delta",
                "event: response.output_item.done",
                "event: response.completed",
            ],
        )
        self.assertEqual(data_lines, ["data: [DONE]"])
