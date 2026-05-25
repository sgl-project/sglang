"""Unit tests for the OpenAI Responses API serving path."""

try:
    import torch

    # Avoid importing torch._inductor while collecting CPU-only serving tests.
    _ORIGINAL_TORCH_COMPILE = torch.compile

    def _identity_compile(fn=None, **kwargs):
        if fn is None:
            return lambda inner_fn: inner_fn
        return fn

    torch.compile = _identity_compile
except ImportError:
    torch = None
    _ORIGINAL_TORCH_COMPILE = None

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest
from unittest.mock import Mock, patch

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci

if torch is not None:
    torch.compile = _ORIGINAL_TORCH_COMPILE

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingResponses."""

    def __init__(self, *, is_multimodal: bool = False):
        self.model_config = Mock(is_multimodal=is_multimodal, context_len=4096)
        self.model_config.get_default_sampling_params.return_value = {}
        self.model_config.hf_config = Mock(
            model_type="llama", architectures=["LlamaForCausalLM"]
        )
        self.server_args = Mock(
            enable_cache_report=False,
            reasoning_parser=None,
            stream_response_default_include_usage=False,
            tokenizer_metrics_allowed_custom_labels=None,
            tool_call_parser=None,
        )
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1
        self.num_reserved_tokens = 0
        self.generate_request = Mock()
        self.create_abort_task = Mock()


class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name = "llama-3"
        self.jinja_template_content_format = None
        self.completion_template_name = None
        self.reasoning_config = None
        self.force_reasoning = False


def _make_serving(*, is_multimodal: bool = False) -> OpenAIServingResponses:
    return OpenAIServingResponses(
        _MockTokenizerManager(is_multimodal=is_multimodal), _MockTemplateManager()
    )


class ServingResponsesTestCase(unittest.TestCase):
    def test_previous_response_replays_assistant_text_not_instructions(self):
        serving = _make_serving()
        prev_response = Mock(id="resp_prev")
        prev_response.output = [
            ResponseReasoningItem(
                id="rs_prev",
                summary=[],
                type="reasoning",
                content=None,
                status=None,
            ),
            ResponseOutputMessage(
                id="msg_prev",
                content=[
                    ResponseOutputText(
                        text="first answer part",
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    ),
                    ResponseOutputText(
                        text="second answer part",
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]
        serving.msg_store["resp_prev"] = [{"role": "user", "content": "old input"}]

        request = ResponsesRequest(
            model="x",
            instructions="Be brief",
            previous_response_id="resp_prev",
            input="new input",
            store=False,
        )

        messages = serving._construct_input_messages(request, prev_response)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "old input"},
                {
                    "role": "assistant",
                    "content": "first answer part\nsecond answer part",
                },
                {"role": "user", "content": "new input"},
            ],
        )

    def test_responses_input_parts_are_normalized_for_chat_templates(self):
        serving = _make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            store=False,
        )

        messages = serving._construct_input_messages(request)

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "http://example.com/cat.png",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
        )

    def test_responses_request_accepts_function_tools(self):
        request = ResponsesRequest(
            model="x",
            input="call the tool",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                    "strict": True,
                }
            ],
            store=False,
        )

        self.assertEqual(request.tools[0].type, "function")
        self.assertEqual(request.tools[0].name, "lookup")
        self.assertTrue(request.tools[0].strict)

    def test_make_request_passes_function_tools_to_chat_processing(self):
        serving = _make_serving()
        seen = {}

        def fake_process(chat_request, is_multimodal):
            seen["tools"] = chat_request.tools
            seen["tool_choice"] = chat_request.tool_choice
            seen["parallel_tool_calls"] = chat_request.parallel_tool_calls
            return MessageProcessingResult(
                prompt="prompt",
                prompt_ids=[1, 2, 3],
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=["</s>"],
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

        serving._process_messages = Mock(side_effect=fake_process)
        request = ResponsesRequest(
            model="x",
            input="call the tool",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice="required",
            parallel_tool_calls=False,
            store=False,
        )

        messages, request_prompts, engine_prompts, processed = asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )

        self.assertEqual(messages, [{"role": "user", "content": "call the tool"}])
        self.assertEqual(request_prompts, [[1, 2, 3]])
        self.assertEqual(engine_prompts, [[1, 2, 3]])
        self.assertEqual(seen["tools"][0].function.name, "lookup")
        self.assertEqual(seen["tool_choice"], "required")
        self.assertFalse(seen["parallel_tool_calls"])
        self.assertEqual(processed.tool_call_constraint[0], "json_schema")

    def test_sampling_params_include_processed_stop_and_tool_constraint(self):
        request = ResponsesRequest(model="x", input="call the tool", store=False)

        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            stop=["</s>"],
            tool_call_constraint=("json_schema", {"type": "object"}),
        )

        self.assertEqual(params["stop"], ["</s>"])
        self.assertEqual(params["json_schema"], '{"type": "object"}')

    def test_full_response_uses_dict_meta_info_for_usage(self):
        serving = _make_serving()
        context = SimpleContext()
        context.last_output = {
            "text": "done",
            "meta_info": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "cached_tokens": 3,
                "reasoning_tokens": 2,
            },
        }
        request = ResponsesRequest(
            model="x",
            input="hello",
            request_id="resp_usage",
            store=False,
        )
        metadata = RequestResponseMetadata(request_id=request.request_id)

        async def empty_generator():
            if False:
                yield None

        response = asyncio.run(
            serving.responses_full_generator(
                request,
                sampling_params={},
                result_generator=empty_generator(),
                context=context,
                model_name="x",
                tokenizer=serving.tokenizer_manager.tokenizer,
                request_metadata=metadata,
                created_time=123,
            )
        )

        self.assertEqual(response.usage.prompt_tokens, 11)
        self.assertEqual(response.usage.completion_tokens, 7)
        self.assertEqual(response.usage.reasoning_tokens, 2)
        self.assertEqual(metadata.final_usage_info, response.usage)

    def test_multimodal_create_responses_sends_text_and_media_to_engine(self):
        serving = _make_serving(is_multimodal=True)
        captured = {}

        serving._process_messages = Mock(
            return_value=MessageProcessingResult(
                prompt="rendered multimodal prompt",
                prompt_ids=[9, 9, 9],
                image_data=["http://example.com/cat.png"],
                audio_data=None,
                video_data=None,
                modalities=["image"],
                stop=[],
            )
        )

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["request_prompt"] = request_prompt
            captured["adapted_request"] = adapted_request
            context.append_output(
                {
                    "text": "looks like a cat",
                    "meta_info": {
                        "prompt_tokens": 5,
                        "completion_tokens": 4,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe it"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            request_id="resp_mm",
            store=False,
        )

        response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        self.assertEqual(captured["request_prompt"], "rendered multimodal prompt")
        self.assertEqual(captured["adapted_request"].text, "rendered multimodal prompt")
        self.assertIsNone(captured["adapted_request"].input_ids)
        self.assertEqual(
            captured["adapted_request"].image_data, ["http://example.com/cat.png"]
        )
        self.assertEqual(captured["adapted_request"].modalities, ["image"])

    def test_function_tool_requires_name(self):
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": "function"}],
                store=False,
            )
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": "function", "name": ""}],
                store=False,
            )

    def test_function_call_input_item_becomes_assistant_tool_call(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "lookup",
                "arguments": '{"key": "val"}',
                "status": "completed",
            }
        )
        self.assertEqual(
            normalized,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key": "val"}',
                        },
                    }
                ],
            },
        )

    def test_function_call_output_input_item_becomes_tool_message(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "42",
            }
        )
        self.assertEqual(
            normalized,
            {"role": "tool", "tool_call_id": "call_abc", "content": "42"},
        )

    def test_unknown_input_item_type_raises(self):
        with self.assertRaises(ValueError):
            OpenAIServingResponses._normalize_response_message_for_chat(
                {"type": "web_search_call", "id": "ws_1"}
            )

    def test_tool_call_and_constraint_conflict_raises(self):
        request = ResponsesRequest(model="x", input="hi", store=False)
        with self.assertRaises(ValueError):
            request.to_sampling_params(
                default_max_tokens=128,
                default_params={"json_schema": '{"type": "object"}'},
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

    def test_to_sampling_params_structural_tag_with_model_dump(self):
        class _FakeStructuralTag:
            def model_dump(self, by_alias=False):
                return {"type": "structural_tag"}

        request = ResponsesRequest(model="x", input="hi", store=False)
        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            tool_call_constraint=("structural_tag", _FakeStructuralTag()),
        )
        self.assertEqual(params["structural_tag"], '{"type": "structural_tag"}')

    def test_function_tool_call_output_items_extracted_via_parser(self):
        """Tool-call text in the assistant output must surface as
        ResponseFunctionToolCall items, not leak through as raw markup."""
        from openai.types.responses.response_function_tool_call import (
            ResponseFunctionToolCall,
        )

        from sglang.srt.function_call.core_types import ToolCallItem

        serving = _make_serving()
        serving.tool_call_parser = "qwen3_coder"

        request = ResponsesRequest(
            model="x",
            input="weather?",
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        fake_call = ToolCallItem(
            tool_index=0,
            name="get_weather",
            parameters='{"city": "Beijing"}',
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = (
                "trailing text",
                [fake_call],
            )
            output_items = serving._make_response_output_items(
                request, "raw model output with <tool_call>", tokenizer=Mock()
            )

        tool_calls = [
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Beijing"}')

        message_items = [
            item for item in output_items if isinstance(item, ResponseOutputMessage)
        ]
        self.assertEqual(len(message_items), 1)
        self.assertEqual(message_items[0].content[0].text, "trailing text")

    def test_responses_request_accepts_extended_tool_types(self):
        # The ResponseTool literal must accept every type that the OpenAI
        # Responses spec advertises so clients (Codex CLI, web-search agents)
        # don't fail at FastAPI validation when the server has no execution
        # path for that built-in type yet.
        for tool_type in (
            "web_search",
            "web_search_preview",
            "code_interpreter",
            "file_search",
            "image_generation",
            "computer_use_preview",
            "local_shell",
            "mcp",
            "custom",
            "namespace",
        ):
            request = ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": tool_type}],
                store=False,
            )
            self.assertEqual(request.tools[0].type, tool_type)

    def test_namespace_tool_accepts_inner_tools_list(self):
        request = ResponsesRequest(
            model="x",
            input="hi",
            tools=[
                {
                    "type": "namespace",
                    "name": "codex",
                    "tools": [
                        {"type": "function", "name": "apply_patch"},
                        {"type": "function", "name": "shell"},
                    ],
                }
            ],
            store=False,
        )
        self.assertEqual(request.tools[0].type, "namespace")
        self.assertEqual(len(request.tools[0].tools), 2)
        self.assertEqual(request.tools[0].tools[0]["name"], "apply_patch")

    def test_non_harmony_stream_emits_typed_sse_events(self):
        serving = _make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        request_metadata = RequestResponseMetadata(request_id=request.request_id)

        async def fake_generator():
            for text, ctoks in [("Hel", 1), ("Hello", 2), ("Hello world", 4)]:
                yield {
                    "text": text,
                    "meta_info": {
                        "id": "rid",
                        "prompt_tokens": 5,
                        "completion_tokens": ctoks,
                        "cached_tokens": 0,
                        "reasoning_tokens": 0,
                        "finish_reason": (
                            {"type": "stop"} if text == "Hello world" else None
                        ),
                    },
                }

        async def collect():
            events = []
            async for chunk in serving.responses_stream_generator_non_harmony(
                request,
                sampling_params={},
                result_generator=fake_generator(),
                model_name="x",
                tokenizer=Mock(),
                request_metadata=request_metadata,
            ):
                events.append(chunk)
            return events

        events = asyncio.run(collect())
        # Parse out the typed event names from the SSE envelope.
        types = [
            line[len("event: ") :].strip()
            for chunk in events
            for line in chunk.splitlines()
            if line.startswith("event: ")
        ]
        self.assertEqual(types[0], "response.created")
        self.assertEqual(types[1], "response.in_progress")
        self.assertIn("response.output_item.added", types)
        self.assertIn("response.content_part.added", types)
        self.assertIn("response.output_text.delta", types)
        self.assertIn("response.output_text.done", types)
        self.assertIn("response.content_part.done", types)
        self.assertIn("response.output_item.done", types)
        self.assertEqual(types[-1], "response.completed")
        # ``sequence_number`` must be monotonic and contiguous across the
        # whole stream so clients can detect dropped events.
        seqs = []
        for chunk in events:
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    seqs.append(json.loads(line[len("data: ") :])["sequence_number"])
        self.assertEqual(seqs, list(range(len(seqs))))

    def test_no_tool_call_extraction_when_tool_choice_none(self):
        serving = _make_serving()
        serving.tool_call_parser = "qwen3_coder"

        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            tool_choice="none",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            output_items = serving._make_response_output_items(
                request, "just a plain answer", tokenizer=Mock()
            )
            parser_cls.assert_not_called()

        self.assertEqual(len(output_items), 1)
        self.assertIsInstance(output_items[0], ResponseOutputMessage)


if __name__ == "__main__":
    unittest.main()
