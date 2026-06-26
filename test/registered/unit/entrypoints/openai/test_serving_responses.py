import asyncio
import unittest
from unittest.mock import Mock, patch

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from utils import collect_stream_events, event_payloads, event_types, make_serving

from sglang.srt.entrypoints.context import SimpleContext
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
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class InputMessageConstructionTestCase(unittest.TestCase):
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


class ChatToolForwardingTestCase(unittest.TestCase):
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


class InputItemNormalizationTestCase(unittest.TestCase):
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


class FullResponseUsageTestCase(unittest.TestCase):
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


class MultimodalRequestTestCase(unittest.TestCase):
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


class OutputItemsTestCase(unittest.TestCase):
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
        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
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
        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
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
                self._function_tool_request(), "raw model output", tokenizer=Mock()
            )

        types = [type(item).__name__ for item in output_items]
        self.assertEqual(types, ["ResponseOutputMessage", "ResponseFunctionToolCall"])

    def test_required_tool_choice_parses_json_array_without_native_parser(self):
        serving = make_serving()
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
            request, raw, tokenizer=Mock()
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
        serving = make_serving()
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


class HarmonyResponsesTestCase(unittest.TestCase):
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


class StatusFromFinishReasonTestCase(unittest.TestCase):
    def test_only_length_maps_to_incomplete(self):
        fn = OpenAIServingResponses._status_from_finish_reason
        self.assertEqual(fn({"type": "length"}), "incomplete")
        self.assertEqual(fn("length"), "incomplete")
        for other in ({"type": "stop"}, {"type": "tool_calls"}, "stop", None):
            self.assertEqual(fn(other), "completed", other)


class BuildOutputTextLogprobsTestCase(unittest.TestCase):
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


class ChatToolChoiceConversionTestCase(unittest.TestCase):
    def test_conversion(self):
        fn = OpenAIServingResponses._chat_tool_choice
        for s in ("auto", "required", "none"):
            self.assertEqual(fn(s), s)
        self.assertEqual(
            fn({"type": "function", "name": "get_weather"}),
            {"type": "function", "function": {"name": "get_weather"}},
        )
        self.assertEqual(fn({"type": "web_search"}), "auto")  # non-function -> auto


class ShouldEmitNormalTextTestCase(unittest.TestCase):
    def test_whitespace_suppressed_only_while_a_tool_is_open(self):
        emit = _should_emit_normal_text_as_message
        self.assertFalse(emit("", any_tool_call_in_progress=False))
        # whitespace between tool blocks is an inter-call separator, not content
        self.assertFalse(emit("\n", any_tool_call_in_progress=True))
        self.assertTrue(emit("\n", any_tool_call_in_progress=False))
        self.assertTrue(emit("hello", any_tool_call_in_progress=True))


class CancelIdempotencyTestCase(unittest.TestCase):
    def test_cancel_terminal_response_returns_it_not_error(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        serving = make_serving()
        resp = ResponsesResponse.from_request(
            ResponsesRequest(model="x", input="hi", store=False),
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="cancelled",
            usage=None,
        )
        serving.response_store[resp.id] = resp
        out = asyncio.run(serving.cancel_responses(resp.id))
        self.assertIs(out, resp)
        self.assertEqual(out.status, "cancelled")


class StreamingLogprobsRejectionTestCase(unittest.TestCase):
    def test_stream_with_logprobs_include_rejected(self):
        import orjson

        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            stream=True,
            include=["message.output_text.logprobs"],
        )
        result = asyncio.run(serving.create_responses(request))
        self.assertEqual(result.status_code, 400)
        body = orjson.loads(result.body)
        self.assertIn("streaming mode", body["error"]["message"])


class MultiToolCallStreamingOrderTestCase(unittest.TestCase):
    def test_prior_tool_call_done_before_next_added(self):
        from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

        serving = make_serving()
        serving.tool_call_parser = "qwen3_coder"
        serving.reasoning_parser = None

        det = Qwen3CoderDetector()
        s, e = det.tool_call_start_token, det.tool_call_end_token
        fp, fe = det.tool_call_prefix, det.function_end_token
        pp, pe = det.parameter_prefix, det.parameter_end_token
        tool1 = f"{s}{fp}get_weather>{pp}city>Beijing{pe}{fe}{e}"
        tool2 = f"{s}{fp}get_time>{pp}tz>UTC{pe}{fe}{e}"
        full = tool1 + "\n" + tool2

        async def fake_gen():
            yield {"text": tool1, "meta_info": {}}
            yield {"text": tool1 + "\n", "meta_info": {}}
            yield {"text": full, "meta_info": {}}
            yield {
                "text": full,
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                },
            }

        request = ResponsesRequest(
            model="x",
            input="weather and time",
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
                {
                    "type": "function",
                    "name": "get_time",
                    "parameters": {"type": "object"},
                },
            ],
        )

        async def run():
            stream = serving.responses_stream_generator_non_harmony(
                request,
                {},
                fake_gen(),
                "x",
                Mock(),
                RequestResponseMetadata(request_id="r"),
            )
            return await collect_stream_events(stream)

        events = asyncio.run(run())
        seq = list(zip(event_types(events), event_payloads(events)))

        def position(pred):
            return next(i for i, (t, p) in enumerate(seq) if pred(t, p))

        done0 = position(
            lambda t, p: t == "response.output_item.done" and p["output_index"] == 0
        )
        added1 = position(
            lambda t, p: t == "response.output_item.added" and p["output_index"] == 1
        )
        self.assertLess(done0, added1)

        fc_done_items = [
            p["item"]
            for t, p in seq
            if t == "response.output_item.done"
            and p["item"].get("type") == "function_call"
        ]
        self.assertEqual(len(fc_done_items), 2)
        names = sorted(item["name"] for item in fc_done_items)
        self.assertEqual(names, ["get_time", "get_weather"])


if __name__ == "__main__":
    unittest.main()
