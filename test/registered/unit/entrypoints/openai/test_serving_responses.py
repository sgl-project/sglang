import asyncio
import unittest
from unittest.mock import Mock, patch

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai_harmony import Message, Role
from utils import make_serving

from sglang.srt.entrypoints.context import (
    HarmonyContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from sglang.srt.entrypoints.harmony_utils import (
    get_streamable_parser_for_assistant,
    render_for_completion,
)
from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_responses import (
    OpenAIServingResponses,
    _build_output_text_logprobs,
    _should_emit_normal_text_as_message,
)
from sglang.srt.entrypoints.openai.utils import to_responses_output_text_logprobs
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.parser.template_detection import ReasoningToggleConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class InputMessageConstructionTestCase(CustomTestCase):
    def test_previous_response_replays_assistant_text_not_instructions(self):
        serving = make_serving()
        prev_response = Mock(id="resp_prev")
        prev_response.output = [
            ResponseReasoningItem(
                id="rs_prev", summary=[], type="reasoning", content=None, status=None
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

    def test_input_parts_normalized_for_chat_templates(self):
        serving = make_serving()
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

    def test_previous_response_id_input_list_does_not_call_copy_module(self):
        serving = make_serving()
        serving.use_harmony = True
        prev = Mock(id="resp_prev")
        prev.output = [
            ResponseFunctionToolCall(
                arguments="{}",
                call_id="call_x",
                name="t",
                type="function_call",
                id="fc_x",
                status="completed",
            )
        ]
        request = ResponsesRequest(
            model="x",
            input=[{"role": "user", "content": "hi"}],
            previous_response_id="resp_prev",
            store=False,
        )
        try:
            serving._construct_input_messages_with_harmony(request, prev)
        except TypeError as exc:
            self.fail(f"copy() module-call regression: {exc}")
        except Exception:
            pass


class ChatToolForwardingTestCase(CustomTestCase):
    def test_make_request_passes_function_tools_to_chat_processing(self):
        serving = make_serving()
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

    def test_required_tool_choice_without_function_tool_returns_400(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[{"type": "web_search"}, {"type": "mcp"}],
            store=False,
        )
        result = asyncio.run(serving.create_responses(request, raw_request=None))
        self.assertEqual(getattr(result, "status_code", None), 400)

    def test_kimi_k3_request_uses_chat_encoder_fields(self):
        serving = make_serving()
        serving.chat_encoding_spec = "kimi_k3"
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [4, 5, 6]
        request = ResponsesRequest(
            model="x",
            input="Explain <|kimi_image_placeholder|>",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice="required",
            reasoning={"effort": "high"},
            store=False,
        )

        _, request_prompts, engine_prompts, _ = asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )

        call = serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
        self.assertEqual(
            call.args[0][0]["content"], "Explain <| kimi_image_placeholder |>"
        )
        self.assertEqual(call.kwargs["thinking_effort"], "high")
        self.assertEqual(call.kwargs["tool_choice"], "required")
        self.assertEqual(call.kwargs["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(request_prompts, [[4, 5, 6]])
        self.assertEqual(engine_prompts, [[4, 5, 6]])


class ReasoningRequestForwardingTestCase(unittest.TestCase):
    def test_create_responses_uses_processed_reasoning_state(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.default_chat_template_kwargs = {"thinking": False}
        serving.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        rendered = MessageProcessingResult(
            prompt="prompt",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["adapted_request"] = adapted_request
            context.append_output(
                {
                    "text": "done",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input="answer",
            request_id="resp_reasoning",
            store=False,
        )

        with (
            patch.object(
                serving, "_apply_conversation_template", return_value=rendered
            ),
            patch(
                "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
            ) as parser_cls,
        ):
            parser_cls.return_value.parse_non_stream.return_value = (None, "done")
            response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        self.assertFalse(captured["adapted_request"].require_reasoning)
        self.assertFalse(parser_cls.call_args.kwargs["force_reasoning"])


class SkipSpecialTokensForwardingTestCase(CustomTestCase):
    """The skip_special_tokens override from _process_messages must reach the
    engine sampling params; muse's channel markers die in detok otherwise."""

    def _create_responses_sampling_params(self, serving):
        serving.default_chat_template_kwargs = None
        rendered = MessageProcessingResult(
            prompt="prompt",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["sampling_params"] = sampling_params
            context.append_output(
                {
                    "text": "done",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input="answer",
            request_id="resp_skip_special",
            store=False,
        )

        with (
            patch.object(
                serving, "_apply_conversation_template", return_value=rendered
            ),
            patch(
                "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
            ) as parser_cls,
        ):
            parser_cls.return_value.parse_non_stream.return_value = (None, "done")
            response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        return captured["sampling_params"]

    def test_marker_preserving_parser_disables_skip_special_tokens(self):
        serving = make_serving()
        serving.reasoning_parser = "muse"
        params = self._create_responses_sampling_params(serving)
        self.assertFalse(params["skip_special_tokens"])

    def test_default_parser_keeps_skip_special_tokens(self):
        serving = make_serving()
        params = self._create_responses_sampling_params(serving)
        # The chat request's True is a synthesized default (ResponsesRequest has
        # no such field), so leave it unset for --preferred-sampling-params.
        self.assertNotIn("skip_special_tokens", params)


class InputItemNormalizationTestCase(CustomTestCase):
    def test_function_call_becomes_assistant_tool_call(self):
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

    def test_developer_role_becomes_system(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {"role": "developer", "content": "Be terse."}
        )
        self.assertEqual(normalized, {"role": "system", "content": "Be terse."})

    def test_function_call_output_becomes_tool_message(self):
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


class FullResponseUsageTestCase(CustomTestCase):
    def test_full_response_uses_dict_meta_info_for_usage(self):
        serving = make_serving()
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
            model="x", input="hello", request_id="resp_usage", store=False
        )
        metadata = RequestResponseMetadata(request_id=request.request_id)

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
                request_metadata=metadata,
                created_time=123,
                require_reasoning=False,
            )
        )

        self.assertEqual(response.usage.prompt_tokens, 11)
        self.assertEqual(response.usage.completion_tokens, 7)
        self.assertEqual(response.usage.reasoning_tokens, 2)
        self.assertEqual(metadata.final_usage_info, response.usage)


class MultimodalRequestTestCase(CustomTestCase):
    def test_text_only_create_responses_rejects_media_before_generation(self):
        serving = make_serving()
        serving._process_messages = Mock()
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
            store=False,
        )

        response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"received unsupported content type 'image_url'", response.body)
        serving._process_messages.assert_not_called()
        serving.tokenizer_manager.generate_request.assert_not_called()

    def test_multimodal_create_responses_sends_text_and_media_to_engine(self):
        serving = make_serving(is_multimodal=True)
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


class OutputItemsTestCase(CustomTestCase):
    def setUp(self):
        # qwen3_coder is the default for this class; the one no-native-parser
        # case overrides it.
        self.serving = make_serving()
        self.serving.tool_call_parser = "qwen3_coder"

    def _function_tool_request(self):
        return ResponsesRequest(
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

    def test_function_tool_call_extracted_via_parser(self):
        serving = self.serving
        fake_call = ToolCallItem(
            tool_index=0, name="get_weather", parameters='{"city": "Beijing"}'
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = ("trailing text", [fake_call])
            output_items = serving._make_response_output_items(
                self._function_tool_request(),
                "raw model output with <tool_call>",
                tokenizer=Mock(),
                require_reasoning=False,
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

    def test_prose_emitted_before_tool_call_item(self):
        serving = self.serving
        fake_call = ToolCallItem(
            tool_index=0, name="get_weather", parameters='{"city": "Beijing"}'
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = (
                "I'll check the weather.",
                [fake_call],
            )
            output_items = serving._make_response_output_items(
                self._function_tool_request(),
                "raw model output",
                tokenizer=Mock(),
                require_reasoning=False,
            )

        types = [type(item).__name__ for item in output_items]
        self.assertEqual(types, ["ResponseOutputMessage", "ResponseFunctionToolCall"])

    def test_required_tool_choice_parses_json_array_without_native_parser(self):
        serving = self.serving
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
            store=False,
        )
        raw = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        output_items = serving._make_response_output_items(
            request, raw, tokenizer=Mock(), require_reasoning=False
        )

        tool_calls = [
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Beijing"}')
        self.assertEqual(
            [item for item in output_items if isinstance(item, ResponseOutputMessage)],
            [],
        )

    def test_no_tool_call_extraction_when_tool_choice_none(self):
        serving = self.serving
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
                request,
                "just a plain answer",
                tokenizer=Mock(),
                require_reasoning=False,
            )
            parser_cls.assert_not_called()

        self.assertEqual(len(output_items), 1)
        self.assertIsInstance(output_items[0], ResponseOutputMessage)


class HarmonyResponsesTestCase(CustomTestCase):
    def test_developer_message_skips_unsupported_tool_types(self):
        from sglang.srt.entrypoints.harmony_utils import get_developer_message
        from sglang.srt.entrypoints.openai.protocol import ResponseTool

        tools = [
            ResponseTool(
                type="function",
                name="get_weather",
                description="Look up weather.",
                parameters={"type": "object"},
            ),
            ResponseTool(type="web_search"),
            ResponseTool(type="namespace", name="codex"),
            ResponseTool(type="mcp"),
        ]
        msg = get_developer_message(instructions="be helpful", tools=tools)
        self.assertIsNotNone(msg)


class StatusFromFinishReasonTestCase(CustomTestCase):
    def test_only_length_maps_to_incomplete(self):
        fn = OpenAIServingResponses._status_from_finish_reason
        self.assertEqual(fn({"type": "length"}), "incomplete")
        self.assertEqual(fn("length"), "incomplete")
        for other in ({"type": "stop"}, {"type": "tool_calls"}, "stop", None):
            self.assertEqual(fn(other), "completed", other)


class BuildOutputTextLogprobsTestCase(CustomTestCase):
    def test_tokens_and_top_logprobs_are_converted(self):
        meta_info = {
            "output_token_logprobs": [(-0.1, 10, "Hello"), (-0.2, 11, " world")],
            "output_top_logprobs": [
                [(-0.1, 10, "Hello"), (-2.0, 12, "Hi")],
                [(-0.2, 11, " world"), (-3.0, 13, " earth")],
            ],
        }
        out = _build_output_text_logprobs(meta_info)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].token, "Hello")
        self.assertEqual(out[0].logprob, -0.1)
        self.assertEqual(out[0].bytes, list("Hello".encode("utf-8")))
        self.assertEqual(len(out[0].top_logprobs), 2)
        self.assertEqual(out[0].top_logprobs[0].token, "Hello")
        self.assertEqual(out[1].token, " world")

    def test_no_top_logprobs_yields_empty_lists(self):
        meta_info = {
            "output_token_logprobs": [(-0.5, 7, "hi")],
            "output_top_logprobs": None,
        }
        out = _build_output_text_logprobs(meta_info)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].top_logprobs, [])


class ChatToolChoiceConversionTestCase(CustomTestCase):
    def test_conversion(self):
        fn = OpenAIServingResponses._chat_tool_choice
        for s in ("auto", "required", "none"):
            self.assertEqual(fn(s), s)
        # Input is an effective_tool_choice() result, so the only object form
        # reaching here is a named function; degrading the rest to "auto"
        # happens there, once, so the echoed and the honored value agree.
        self.assertEqual(
            fn({"type": "function", "name": "get_weather"}),
            {"type": "function", "function": {"name": "get_weather"}},
        )


class ShouldEmitNormalTextTestCase(CustomTestCase):
    def test_whitespace_suppressed_only_while_a_tool_is_open(self):
        emit = _should_emit_normal_text_as_message
        self.assertFalse(emit("", any_tool_call_in_progress=False))
        # whitespace between tool blocks is an inter-call separator, not content
        self.assertFalse(emit("\n", any_tool_call_in_progress=True))
        self.assertTrue(emit("\n", any_tool_call_in_progress=False))
        self.assertTrue(emit("hello", any_tool_call_in_progress=True))


class EnginePassthroughTestCase(CustomTestCase):
    """Both flags cross hops with no type contract, and dropping either fails
    silently."""

    def _capture(self, serving, request):
        # Let the real _process_messages run: it is the hop that turns
        # skip_special_tokens off, so mocking it would make that assertion vacuous.
        # chat_template_name=None routes it through the tokenizer's template
        # (mocked) instead of the conversation registry, which has no fixture entry.
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["adapted_request"] = adapted_request
            captured["sampling_params"] = sampling_params
            context.append_output(
                {
                    "text": "ok",
                    "meta_info": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        asyncio.run(serving.create_responses(request))
        return captured

    def test_require_reasoning_forwarded_when_reasoning_parser_configured(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )

        captured = self._capture(
            serving, ResponsesRequest(model="x", input="hi", store=False)
        )

        self.assertTrue(captured["adapted_request"].require_reasoning)

    def test_prefilled_think_template_opens_the_parser(self):
        """``force_reasoning`` is a template property, not a request one, so it
        drives the parser but never the engine flag -- as on the chat path."""
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.template_manager.force_reasoning = True

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            parser_cls.return_value.parse_non_stream.return_value = (None, "hi")
            serving._make_response_output_items(
                ResponsesRequest(model="x", input="hi", store=False),
                "hi",
                tokenizer=Mock(),
                require_reasoning=False,
            )

        self.assertTrue(parser_cls.call_args.kwargs["force_reasoning"])

    def test_require_reasoning_false_without_reasoning_parser(self):
        serving = make_serving()
        serving.reasoning_parser = None

        captured = self._capture(
            serving, ResponsesRequest(model="x", input="hi", store=False)
        )

        self.assertFalse(captured["adapted_request"].require_reasoning)

    def test_skip_special_tokens_disabled_for_tool_requests(self):
        # _process_messages turns it off so tool-call markers survive detokenize;
        # create_responses must re-apply it to the engine sampling dict.
        serving = make_serving()
        serving.tool_call_parser = "qwen25"

        captured = self._capture(
            serving,
            ResponsesRequest(
                model="x",
                input="weather",
                store=False,
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    }
                ],
            ),
        )

        self.assertFalse(captured["sampling_params"]["skip_special_tokens"])


class CancelIdempotencyTestCase(CustomTestCase):
    def test_cancelling_a_terminal_response_returns_it_not_an_error(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        for status in ("cancelled", "completed"):
            serving = make_serving()
            resp = ResponsesResponse.from_request(
                ResponsesRequest(model="x", input="hi", store=False),
                sampling_params={},
                model_name="x",
                created_time=0,
                output=[],
                status=status,
                usage=None,
            )
            serving.response_store[resp.id] = resp

            out = asyncio.run(serving.cancel_responses(resp.id))

            self.assertIs(out, resp, status)
            self.assertEqual(out.status, status)


class HarmonyLogprobsTestCase(CustomTestCase):
    """Non-streaming harmony logprobs: capture final-channel content tokens
    and attach them to the final ResponseOutputText."""

    @staticmethod
    def _final_answer_tokens(answer_text: str) -> list[int]:
        """Return the assistant-generated harmony token ids (turn header +
        final-channel content + <|return|>) for *answer_text*."""
        sys_msg = Message.from_role_and_content(Role.SYSTEM, "x")
        user_msg = Message.from_role_and_content(Role.USER, "q")
        final_msg = Message.from_role_and_content(Role.ASSISTANT, answer_text)
        final_msg = final_msg.with_channel("final")
        all_toks = render_for_completion([sys_msg, user_msg, final_msg])
        prompt_toks = render_for_completion([sys_msg, user_msg])
        return all_toks[len(prompt_toks) :]

    @staticmethod
    def _engine_output(token_ids: list[int]) -> dict:
        """Build an engine chunk whose logprob token_text is the delta each
        token emits (from an independent parser pass) so captured texts
        reconstruct the visible answer; structural tokens get a sentinel."""
        parser = get_streamable_parser_for_assistant()
        token_logprobs = []
        for i, tok in enumerate(token_ids):
            parser.process(tok)
            delta = parser.last_content_delta
            token_logprobs.append((-0.1 * i, tok, delta if delta else f"STRUCT{i}"))
        return {
            "output_ids": token_ids,
            "meta_info": {
                "output_token_logprobs": token_logprobs,
                "output_top_logprobs": None,
                "prompt_tokens": 5,
                "completion_tokens": len(token_ids),
                "cached_tokens": 0,
            },
        }

    def test_append_output_buckets_only_final_channel_content(self):
        gen_toks = self._final_answer_tokens("Hello!")
        context = HarmonyContext(
            [
                Message.from_role_and_content(Role.SYSTEM, "x"),
                Message.from_role_and_content(Role.USER, "q"),
            ],
            {},
        )
        context.append_output(self._engine_output(gen_toks))

        captured = context.final_token_logprobs
        # Only the final-channel content tokens survive; their texts reconstruct
        # the visible answer, proving no reasoning/structural tokens leaked in.
        self.assertEqual("".join(lp[2] for lp in captured), "Hello!")
        self.assertTrue(captured)
        self.assertFalse(any(lp[2].startswith("STRUCT") for lp in captured))

    def test_make_output_items_attaches_logprobs_only_when_requested(self):
        serving = make_serving()
        gen_toks = self._final_answer_tokens("Hello!")
        context = HarmonyContext(
            [
                Message.from_role_and_content(Role.SYSTEM, "x"),
                Message.from_role_and_content(Role.USER, "q"),
            ],
            {},
        )
        context.append_output(self._engine_output(gen_toks))

        final_logprobs = to_responses_output_text_logprobs(
            context.final_token_logprobs, context.final_top_logprobs
        )
        items = serving._make_response_output_items_with_harmony(
            context, final_logprobs=final_logprobs
        )
        msg = [i for i in items if isinstance(i, ResponseOutputMessage)]
        self.assertEqual(len(msg), 1)
        text_part = msg[0].content[0]
        self.assertIsInstance(text_part, ResponseOutputText)
        self.assertIsNotNone(text_part.logprobs)
        self.assertEqual(len(text_part.logprobs), len(context.final_token_logprobs))

        # No logprobs when the caller passes None (request didn't ask for them).
        items_none = serving._make_response_output_items_with_harmony(
            context, final_logprobs=None
        )
        text_part_none = [
            i for i in items_none if isinstance(i, ResponseOutputMessage)
        ][0].content[0]
        self.assertIsNone(text_part_none.logprobs)


class StreamingHarmonyLogprobsTestCase(CustomTestCase):
    """Streaming harmony logprobs: bucketing across incremental and cumulative
    chunks, plus per-chunk delta tracking for the delta event."""

    @staticmethod
    def _token_info(answer_text: str):
        """Return (gen_toks, info) where info[i] = (delta, channel) for token i,
        from an independent parser pass."""
        gen_toks = HarmonyLogprobsTestCase._final_answer_tokens(answer_text)
        parser = get_streamable_parser_for_assistant()
        info = []
        for tok in gen_toks:
            parser.process(tok)
            info.append((parser.last_content_delta, parser.current_channel))
        return gen_toks, info

    @staticmethod
    def _lp(i: int, tok: int, delta):
        return (-0.1 * i, tok, delta if delta else f"STRUCT{i}")

    def test_incremental_chunks_bucket_and_track_delta(self):
        gen_toks, info = self._token_info("Hello!")
        ctx = StreamingHarmonyContext(
            [
                Message.from_role_and_content(Role.SYSTEM, "x"),
                Message.from_role_and_content(Role.USER, "q"),
            ],
            {},
        )
        for i, tok in enumerate(gen_toks):
            delta, channel = info[i]
            ctx.append_output(
                {
                    "output_ids": [tok],
                    "meta_info": {"output_token_logprobs": [self._lp(i, tok, delta)]},
                }
            )
            # delta_token_logprobs holds the final-content tokens added in this
            # chunk (one token per chunk here); empty when the chunk added none.
            if channel == "final" and delta:
                self.assertEqual(len(ctx.delta_token_logprobs), 1)
                self.assertEqual(
                    ctx.delta_token_logprobs[0][2], self._lp(i, tok, delta)[2]
                )
            else:
                self.assertEqual(ctx.delta_token_logprobs, [])

        self.assertEqual("".join(lp[2] for lp in ctx.final_token_logprobs), "Hello!")

    def test_cumulative_chunks_slice_logprobs_without_duplication(self):
        gen_toks, info = self._token_info("Hello!")
        all_lps = [self._lp(i, tok, info[i][0]) for i, tok in enumerate(gen_toks)]
        ctx = StreamingHarmonyContext(
            [
                Message.from_role_and_content(Role.SYSTEM, "x"),
                Message.from_role_and_content(Role.USER, "q"),
            ],
            {},
        )
        # Cumulative streaming: each chunk re-sends all tokens so far; the
        # context must process only the new slice each step.
        for n in range(1, len(gen_toks) + 1):
            ctx.append_output(
                {
                    "output_ids": gen_toks[:n],
                    "meta_info": {
                        "output_token_logprobs": all_lps[:n],
                        "completion_tokens": n,
                    },
                }
            )
        # Correct slicing => each final-content token captured exactly once.
        self.assertEqual("".join(lp[2] for lp in ctx.final_token_logprobs), "Hello!")

    def test_multi_token_chunk_delta_slice_keeps_every_token(self):
        """Regression guard for the harmony streaming delta path: when a single
        chunk carries several final-channel content tokens (normal batched
        decode, or a multi-byte char spanning decode steps), delta_token_logprobs
        must hold one entry per token -- not just the last. Previously the delta
        event pinned a single logprob to the whole multi-token delta string, so
        len(logprobs) != len(delta tokens) and aligning clients got garbage."""
        gen_toks, info = self._token_info("Hello wonderful world")
        final_deltas = [
            info[i][0]
            for i in range(len(gen_toks))
            if info[i][1] == "final" and info[i][0]
        ]
        # The answer must actually span >=2 final tokens for the guard to bite.
        self.assertGreaterEqual(len(final_deltas), 2)
        all_lps = [self._lp(i, tok, info[i][0]) for i, tok in enumerate(gen_toks)]
        ctx = StreamingHarmonyContext(
            [
                Message.from_role_and_content(Role.SYSTEM, "x"),
                Message.from_role_and_content(Role.USER, "q"),
            ],
            {},
        )
        # Incremental mode (no completion_tokens): the whole batch is new in a
        # single append_output call.
        ctx.append_output(
            {
                "output_ids": gen_toks,
                "meta_info": {"output_token_logprobs": all_lps},
            }
        )
        # Every final-content token of this chunk is in the per-chunk slice...
        self.assertEqual([lp[2] for lp in ctx.delta_token_logprobs], final_deltas)
        # ...and it matches the fully-accumulated list, since this is the only
        # chunk fed so far.
        self.assertEqual(
            [lp[2] for lp in ctx.delta_token_logprobs],
            [lp[2] for lp in ctx.final_token_logprobs],
        )
        # The delta text is the concatenation of every final token's piece -- one
        # per logprob entry -- not just the last token's. This is what keeps the
        # delta string aligned with len(delta_token_logprobs); the parser's own
        # last_content_delta would only be "world" here.
        self.assertEqual(ctx.delta_text, "".join(final_deltas))
        # Sanity: delta_text is strictly longer than the last token's piece, which
        # is the value the pre-fix code would have shipped.
        self.assertGreater(len(ctx.delta_text), len(final_deltas[-1]))


class StreamingLogprobsAcceptedTestCase(CustomTestCase):
    def test_stream_with_logprobs_include_not_rejected(self):
        # Streaming + logprobs used to short-circuit to a 400 at validation
        # ("logprobs are not supported in streaming mode"). That rejection must
        # stay lifted. The other stream tests drive the generator directly and
        # bypass validation, so this is the only case guarding the rejection.
        #
        # The mock tokenizer fails later in request prep and create_responses
        # returns that as an ORJSONResponse -- but its message is never the
        # logprobs-rejection text. If the rejection were re-added, this would
        # return a 400 whose message starts with "logprobs are not supported".
        import orjson
        from fastapi.responses import ORJSONResponse

        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            stream=True,
            include=["message.output_text.logprobs"],
        )
        result = asyncio.run(serving.create_responses(request))

        if isinstance(result, ORJSONResponse):
            body = orjson.loads(result.body)
            # Error envelope nests under "error"; read the message wherever it is.
            err = body.get("error", body)
            self.assertNotIn("logprobs are not supported", err["message"])


if __name__ == "__main__":
    unittest.main()
