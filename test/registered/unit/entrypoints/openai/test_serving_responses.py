import argparse
import asyncio
import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import orjson
import pytest
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai_harmony import Conversation, Message, Role, ToolNamespaceConfig
from utils import StreamFixture, engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.context import (
    HarmonyContext,
    SimpleContext,
)
from sglang.srt.entrypoints.harmony_utils import get_encoding
from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
)
from sglang.srt.entrypoints.openai.serving_responses import (
    OpenAIServingResponses,
    _build_output_text_logprobs,
    _should_emit_normal_text_as_message,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.parser.template_detection import ReasoningToggleConfig
from sglang.srt.runtime_context import get_serving, publish, reset_context
from sglang.srt.sampling.sampling_params import (
    REQUEST_REASONING_END_TOKEN_IDS_KEY,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        serving.msg_store["resp_prev"] = [
            {"role": "user", "content": "old input"},
            *[item.model_dump(exclude_none=True) for item in prev_response.output],
        ]

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
                    "content": [
                        {"type": "text", "text": "first answer part"},
                        {"type": "text", "text": "second answer part"},
                    ],
                },
                {"role": "user", "content": "new input"},
            ],
        )

    def test_stored_tool_turn_matches_client_replay_without_old_instructions(self):
        publish(
            ServerArgs(model_path="dummy", enable_response_store=True), role="tokenizer"
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                serving = make_serving()
                serving.reasoning_parser = "deepseek-r1"
                serving.tool_call_parser = None
                request = ResponsesRequest(
                    model="x",
                    input="old input",
                    instructions="OLD INSTRUCTION",
                    tools=[{"type": "function", "name": "lookup"}],
                    tool_choice="required",
                    store=True,
                    stream=stream,
                )
                chunk = engine_chunk(
                    '<think>secret plan</think>[{"name":"lookup","parameters":{}}]',
                    finish=True,
                )
                if stream:
                    StreamFixture(serving, request).run([chunk])
                    response = serving.response_store[request.request_id]
                else:
                    context = SimpleContext()
                    context.append_output(chunk)

                    async def empty():
                        if False:
                            yield

                    response = asyncio.run(
                        serving.responses_full_generator(
                            request,
                            {},
                            empty(),
                            context,
                            "x",
                            Mock(),
                            RequestResponseMetadata(request_id=request.request_id),
                            require_reasoning=False,
                        )
                    )
                call = next(
                    item for item in response.output if item.type == "function_call"
                )
                result = {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "answer 42",
                }
                for instructions in ("NEW INSTRUCTION", None):
                    followup = ResponsesRequest(
                        model="x",
                        previous_response_id=response.id,
                        input=[result],
                        instructions=instructions,
                        store=False,
                    )
                    explicit = ResponsesRequest(
                        model="x",
                        input=[{"role": "user", "content": "old input"}]
                        + [item.model_dump() for item in response.output]
                        + [result],
                        instructions=instructions,
                        store=False,
                    )
                    messages = serving._construct_input_messages(followup, response)
                    self.assertEqual(
                        messages, serving._construct_input_messages(explicit)
                    )
                    self.assertNotIn("OLD INSTRUCTION", str(messages))
                    self.assertIn("secret plan", str(messages))
                    self.assertIn(call.call_id, str(messages))
                self.assertEqual(
                    [item["type"] for item in serving.msg_store[response.id][1:]],
                    ["reasoning", "function_call"],
                )

    def test_harmony_instructions_are_rebuilt_for_each_request(self):
        serving = make_serving()
        serving.use_harmony = True
        previous = Mock(id="resp_previous", output=[])
        first = ResponsesRequest(
            model="x", input="old input", instructions="OLD INSTRUCTION"
        )
        messages = serving._construct_input_messages_with_harmony(first, None)
        serving.msg_store[previous.id] = messages[2:]
        for instructions in ("NEW INSTRUCTION", None):
            request = ResponsesRequest(
                model="x",
                input="next",
                instructions=instructions,
                previous_response_id=previous.id,
            )
            actual = serving._construct_input_messages_with_harmony(request, previous)
            expected = serving._construct_input_messages_with_harmony(request, None)
            self.assertEqual(actual[:2], expected[:2])
            self.assertEqual(actual[2:], messages[2:] + expected[2:])

    def test_harmony_replays_output_text_and_encoded_reasoning(self):
        from sglang.srt.entrypoints.harmony_utils import parse_response_input
        from sglang.srt.entrypoints.openai.responses_adapters import (
            encode_reasoning_state,
        )

        message = parse_response_input(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "assistant-only secret 42"},
                ],
                "phase": "final_answer",
            },
            [],
        )
        self.assertEqual(message.content[0].text, "assistant-only secret 42")
        reasoning = parse_response_input(
            {
                "type": "reasoning",
                "encrypted_content": encode_reasoning_state("private plan"),
            },
            [],
        )
        self.assertEqual(reasoning.content[0].text, "private plan")
        self.assertEqual(reasoning.channel, "analysis")

    def test_harmony_replays_dict_tool_calls_and_results_in_one_input(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "type": "function_call",
                    "name": "lookup",
                    "call_id": "call_1",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "output_text", "text": "result"}],
                },
            ],
            store=False,
        )
        messages = serving._construct_input_messages_with_harmony(request, None)
        self.assertEqual(messages[-1].author.name, "functions.lookup")
        self.assertEqual(messages[-1].content[0].text, "result")

    def test_harmony_message_channels_map_to_phases(self):
        from sglang.srt.entrypoints.harmony_utils import (
            parse_output_message,
            parse_response_input,
        )

        for phase in ("commentary", "final_answer"):
            message = parse_response_input(
                {
                    "role": "assistant",
                    "content": "answer",
                    "phase": phase,
                },
                [],
            )
            (item,) = parse_output_message(message)
            self.assertEqual(item.phase, phase)
            self.assertEqual(item.content[0].text, "answer")

    def test_replay_preserves_different_assistant_phases(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {"role": "assistant", "content": "working", "phase": "commentary"},
                {"role": "assistant", "content": "answer", "phase": "final_answer"},
            ],
            store=False,
        )
        messages = serving._construct_input_messages(request)
        self.assertEqual([m["phase"] for m in messages], ["commentary", "final_answer"])
        self.assertEqual([m["content"] for m in messages], ["working", "answer"])

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

    def test_harmony_forced_choices_explain_missing_routing_constraints(self):
        serving = make_serving()
        serving.use_harmony = True
        for choice in ("none", "required", {"type": "function", "name": "lookup"}):
            request = ResponsesRequest(
                model="x",
                input="hi",
                tool_choice=choice,
                tools=[{"type": "function", "name": "lookup"}],
                store=False,
            )
            response = asyncio.run(serving.create_responses(request))
            self.assertEqual(response.status_code, 400)
            self.assertIn(b"recipient", response.body)
            self.assertIn(b"tool_choice", response.body)
        serving.tokenizer_manager.generate_request.assert_not_called()

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

    def test_k2_output_parser_reuses_effective_template_default(self):
        serving = make_serving()
        serving.reasoning_parser = "k2_horizon"
        serving.default_chat_template_kwargs = {"reasoning_effort": "low"}
        serving.template_manager.chat_template_name = None
        serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [4, 5, 6]
        request = ResponsesRequest(
            model="IFM/K2-Horizon-7B",
            input="hi",
            # Template kwargs are the final render inputs, so the server default
            # below takes precedence over this API convenience field.
            reasoning={"effort": "medium"},
            store=False,
        )

        asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )

        render_call = serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
        self.assertEqual(render_call.kwargs["reasoning_effort"], "low")
        self.assertEqual(request.chat_template_kwargs["reasoning_effort"], "low")

        output_items = serving._make_response_output_items(
            request,
            "work</ifm|think_faster>\nanswer",
            tokenizer=Mock(),
            require_reasoning=True,
        )
        self.assertEqual(output_items[0].content[0].text, "work")
        self.assertEqual(output_items[1].content[0].text, "\nanswer")


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
            reasoning_end_token_ids=[41, 42],
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
        self.assertEqual(
            captured["adapted_request"].sampling_params["custom_params"][
                REQUEST_REASONING_END_TOKEN_IDS_KEY
            ],
            [41, 42],
        )
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

    def test_developer_role_becomes_labelled_system(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {"role": "developer", "content": "Be terse."}
        )
        self.assertEqual(
            normalized,
            {"role": "system", "content": "Developer instructions:\nBe terse."},
        )

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

    def test_multimodal_token_first_specs_route_through_prompt_ids(self):
        """Bug regression: token-first encoders leave prompt == "" with
        non-empty prompt_ids; forwarding the empty text 400s in
        _tokenize_texts, so the multimodal branch must forward prompt_ids."""
        for spec in ("inkling", "kimi_k3"):
            with self.subTest(spec=spec):
                serving = make_serving(is_multimodal=True)
                serving.chat_encoding_spec = spec
                serving._process_messages = Mock(
                    return_value=MessageProcessingResult(
                        prompt="",
                        prompt_ids=[4, 5, 6],
                        image_data=None,
                        audio_data=None,
                        video_data=None,
                        modalities=[],
                        stop=[],
                    )
                )
                request = ResponsesRequest(model="x", input="hi", store=False)

                _, request_prompts, engine_prompts, _ = asyncio.run(
                    serving._make_request(
                        request, None, serving.tokenizer_manager.tokenizer
                    )
                )

                self.assertEqual(engine_prompts, [[4, 5, 6]])
                self.assertEqual(request_prompts, [[4, 5, 6]])


class OutputItemsTestCase(CustomTestCase):
    def setUp(self):
        # qwen3_coder is the default for this class; the one no-native-parser
        # case overrides it.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
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
        self.assertEqual(output_items[0].phase, "commentary")

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

    def test_required_tool_choice_skips_json_fallback_for_native_parser(self):
        """muse reports parses_required_natively, so required output must not
        be pushed through the orjson JSON-array fallback (mirrors chat)."""
        serving = self.serving
        serving.tool_call_parser = "muse"
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

        self.assertEqual(
            [
                item
                for item in output_items
                if isinstance(item, ResponseFunctionToolCall)
            ],
            [],
        )
        message_items = [
            item for item in output_items if isinstance(item, ResponseOutputMessage)
        ]
        self.assertEqual(len(message_items), 1)
        self.assertEqual(message_items[0].content[0].text, raw)

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

    def _capture(self, serving, request, raw_request=None):
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
        asyncio.run(serving.create_responses(request, raw_request=raw_request))
        return captured

    def test_pd_routing_fields_forwarded_to_engine(self):
        serving = make_serving()
        raw_request = Mock(headers={"x-data-parallel-rank": "2"}, state=Mock())

        captured = self._capture(
            serving,
            ResponsesRequest(
                model="x",
                input="hi",
                bootstrap_host="10.0.0.1",
                bootstrap_port=8998,
                bootstrap_room=42,
                routed_dp_rank=1,
                disagg_prefill_dp_rank=0,
                store=False,
            ),
            raw_request=raw_request,
        )

        adapted_request = captured["adapted_request"]
        self.assertEqual(adapted_request.bootstrap_host, "10.0.0.1")
        self.assertEqual(adapted_request.bootstrap_port, 8998)
        self.assertEqual(adapted_request.bootstrap_room, 42)
        self.assertEqual(adapted_request.routed_dp_rank, 2)
        self.assertEqual(adapted_request.disagg_prefill_dp_rank, 0)

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
        publish(
            ServerArgs(model_path="dummy", enable_response_store=True), role="tokenizer"
        )
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


class StreamingLogprobsRejectionTestCase(CustomTestCase):
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


STORE_DISABLED_MESSAGE = (
    "Response store is disabled. Stateful Responses require "
    "--enable-response-store on a standalone server; response storage "
    "is unavailable in PD mode."
)
STORE_PD_MESSAGE = (
    "--enable-response-store is not supported with "
    "--disaggregation-mode=prefill or decode; response storage must "
    "remain disabled in PD mode."
)


@pytest.fixture(autouse=True)
def isolated_response_config():
    reset_context()
    yield
    reset_context()


@pytest.fixture
def response_serving():
    def build(enabled=False, mode="null", harmony=False):
        reset_context()
        publish(
            ServerArgs(
                model_path="dummy",
                enable_response_store=enabled,
                disaggregation_mode=mode,
            ),
            role="tokenizer",
        )
        serving = make_serving()
        serving.use_harmony = harmony
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        serving.template_manager.jinja_template_content_format = "string"
        serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        serving.tokenizer_manager.abort_request = Mock()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        async def generate(*args, **kwargs):
            chunk = engine_chunk("ok", finish=True)
            if harmony:
                chunk["output_ids"] = get_encoding().render_conversation(
                    Conversation.from_messages(
                        [
                            Message.from_role_and_content(
                                Role.ASSISTANT, "ok"
                            ).with_channel("final")
                        ]
                    )
                )
                chunk["meta_info"]["completion_tokens"] = len(chunk["output_ids"])
            yield chunk

        serving.tokenizer_manager.generate_request = Mock(side_effect=generate)
        return serving

    return build


async def create_response_result(serving, request):
    result = await serving.create_responses(request)
    if request.stream:
        payloads = event_payloads([event async for event in result])
        assert payloads[0]["type"] == "response.created"
        assert payloads[-1]["type"] == "response.completed"
        return ResponsesResponse.model_validate(payloads[-1]["response"])
    return result


def assert_response_error(response, param, message=STORE_DISABLED_MESSAGE, status=400):
    assert response.status_code == status
    assert orjson.loads(response.body) == {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": status,
        }
    }


@pytest.mark.parametrize(
    "mode,enabled", [("null", False), ("null", True), ("decode", True)]
)
def test_response_store_configuration(mode, enabled):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = ["--model-path", "dummy", "--disaggregation-mode", mode]
    if enabled:
        argv.append("--enable-response-store")
    args = ServerArgs.from_cli_args(parser.parse_args(argv))
    if mode != "null":
        with pytest.raises(ValueError) as error:
            publish(args, role="tokenizer")
        assert str(error.value) == STORE_PD_MESSAGE
    else:
        publish(args, role="tokenizer")
        serving = make_serving()
        assert get_serving().enable_response_store is enabled
        assert serving.enable_response_store is enabled
        assert not serving.is_disaggregated


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_response_store_persistence_and_continuation(response_serving, stream, enabled):
    serving = response_serving(enabled=enabled, harmony=not stream)

    async def run():
        first = await create_response_result(
            serving, ResponsesRequest(model="x", input="first", stream=stream)
        )
        assert first.status == "completed"
        assert first.output[0].content[0].text == "ok"
        assert first.store is True
        assert not serving.background_tasks
        if not enabled:
            assert not serving.response_store and not serving.msg_store
            return
        assert serving.response_store[first.id].output == first.output
        history = list(serving.msg_store[first.id])
        assert len(history) == 2
        second = await create_response_result(
            serving,
            ResponsesRequest(
                model="x",
                input="next",
                previous_response_id=first.id,
                store=False,
                stream=stream,
            ),
        )
        if serving.use_harmony:
            generated_request = (
                serving.tokenizer_manager.generate_request.call_args.args[0]
            )
            prompt = get_encoding().decode(generated_request.input_ids)
        else:
            prompt = str(
                serving.tokenizer_manager.tokenizer.apply_chat_template.call_args.args[
                    0
                ]
            )
        assert all(text in prompt for text in ("first", "ok", "next"))
        assert second.status == "completed"
        assert second.output[0].content[0].text == "ok"
        assert second.store is False
        assert set(serving.response_store) == {first.id}
        assert set(serving.msg_store) == {first.id}
        assert serving.msg_store[first.id] == history

    asyncio.run(run())


@pytest.mark.parametrize(
    "enabled,fields,param,message",
    [
        (
            False,
            {"previous_response_id": ""},
            "previous_response_id",
            STORE_DISABLED_MESSAGE,
        ),
        (False, {}, "background", STORE_DISABLED_MESSAGE),
        (True, {}, "store", "background=true requires store=true."),
    ],
    ids=["empty-predecessor", "disabled-background", "background-without-store"],
)
def test_response_store_admission(response_serving, enabled, fields, param, message):
    serving = response_serving(enabled=enabled, mode="null" if enabled else "prefill")
    serving._make_request = AsyncMock(side_effect=AssertionError("prompt reached"))
    serving.response_store = Mock()
    serving.msg_store = Mock()
    request = ResponsesRequest(
        model="unknown", input="hi", background=True, stream=True, store=None, **fields
    )
    assert_response_error(
        asyncio.run(serving.create_responses(request)), param, message
    )
    serving._make_request.assert_not_called()
    serving.tokenizer_manager.generate_request.assert_not_called()
    assert not serving.response_store.mock_calls
    assert not serving.msg_store.mock_calls
    assert not serving.background_tasks


@pytest.mark.parametrize("enabled", [False, True])
def test_response_store_read_endpoints(response_serving, enabled):
    serving = response_serving(enabled=enabled)

    async def run():
        if not enabled:
            serving.response_store = Mock()
            serving.background_tasks = Mock()
            for method in (serving.retrieve_responses, serving.cancel_responses):
                assert_response_error(await method("bad"), "response_id")
            assert not serving.response_store.mock_calls
            assert not serving.background_tasks.mock_calls
        else:
            response = await create_response_result(
                serving, ResponsesRequest(model="x", input="hi")
            )
            assert await serving.retrieve_responses(response.id) is response
            for response_id, status in (("bad", 400), ("resp_missing", 404)):
                for method in (serving.retrieve_responses, serving.cancel_responses):
                    assert (await method(response_id)).status_code == status
                request = ResponsesRequest(
                    model="x", input="hi", previous_response_id=response_id
                )
                assert (await serving.create_responses(request)).status_code == status
        serving.tokenizer_manager.abort_request.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize(
    "outcome", ["success", "failure", "cancel_queued", "cancel_running"]
)
def test_background_task_lifecycle(response_serving, outcome):
    serving = response_serving(enabled=True)
    original_generate = serving.tokenizer_manager.generate_request

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(*args, **kwargs):
            entered.set()
            await release.wait()
            if outcome == "failure":
                raise ValueError("generation failed")
            async for chunk in original_generate(*args, **kwargs):
                yield chunk

        serving.tokenizer_manager.generate_request = generate
        request = ResponsesRequest(model="x", input="hi", background=True)
        queued = await serving.create_responses(request)
        assert queued.status == "queued"
        assert serving.response_store[queued.id] is queued
        assert request.request_id in serving.msg_store
        task = serving.background_tasks[queued.id]
        if outcome != "cancel_queued":
            await entered.wait()
            assert (await serving.retrieve_responses(queued.id)).status == "in_progress"
        if outcome.startswith("cancel"):
            cancelled = await serving.cancel_responses(queued.id)
            assert cancelled.status == "cancelled"
            serving.tokenizer_manager.abort_request.assert_called_once_with(
                rid=queued.id
            )
            assert task.done()
            if outcome == "cancel_queued":
                assert task.cancelled()
        else:
            release.set()
            await task
            assert serving.response_store[queued.id].status == (
                "failed" if outcome == "failure" else "completed"
            )
        assert not serving.background_tasks

    asyncio.run(run())


def test_active_stream_cancel_and_final_history(response_serving):
    serving = response_serving(enabled=True, harmony=True)

    async def run():
        request = ResponsesRequest(model="x", input="hi", background=True, stream=True)
        stream = await serving.create_responses(request)
        assert "response.created" in await anext(stream)
        assert not serving.background_tasks
        assert (await serving.cancel_responses(request.request_id)).status_code == 404
        serving.tokenizer_manager.abort_request.assert_not_called()
        events = [event async for event in stream]
        assert event_payloads(events)[-1]["type"] == "response.completed"
        assert serving.response_store[request.request_id].status == "completed"
        assert len(serving.msg_store[request.request_id]) == 2

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_completion_preserves_cancelled_response(response_serving, stream):
    serving = response_serving(enabled=True, harmony=not stream)
    request = ResponsesRequest(model="x", input="hi", stream=stream)
    cancelled = ResponsesResponse.from_request(
        request,
        {},
        model_name="x",
        created_time=0,
        output=[],
        status="cancelled",
        usage=None,
    )
    serving.response_store[request.request_id] = cancelled
    serving.msg_store[request.request_id] = ["original"]
    if stream:
        StreamFixture(serving, request).run([engine_chunk("ok", finish=True)])
    else:
        messages = serving._construct_input_messages_with_harmony(request, None)
        context = HarmonyContext(messages, {})

        async def generate():
            async for chunk in serving.tokenizer_manager.generate_request():
                context.append_output(chunk)
                yield context

        response = asyncio.run(
            serving.responses_full_generator(
                request,
                {},
                generate(),
                context,
                "x",
                Mock(),
                RequestResponseMetadata(request_id=request.request_id),
                require_reasoning=False,
            )
        )
        assert response.status == "completed"
    assert serving.response_store[request.request_id] is cancelled
    assert serving.msg_store[request.request_id] == ["original"]


def test_pd_builtin_tool_admission(response_serving):
    serving = response_serving(mode="prefill", harmony=True)
    serving.tool_server = Mock()
    serving.supports_browsing = True
    request = ResponsesRequest(model="x", input="hi", tools=[{"type": "web_search"}])
    result = asyncio.run(serving.create_responses(request))
    assert result.status_code == 400
    assert orjson.loads(result.body)["error"]["param"] == "tools"
    serving.tool_server.get_tool_session.assert_not_called()
    serving.tokenizer_manager.generate_request.assert_not_called()


def test_standalone_builtin_tools_without_storage(response_serving):
    serving = response_serving(harmony=True)
    tool_session = Mock()
    tool_session.call_tool = AsyncMock(
        return_value=Mock(content=[Mock(text="Search result: 42")])
    )

    @asynccontextmanager
    async def session(name):
        yield tool_session

    turns = iter(
        [
            Message.from_role_and_content(Role.ASSISTANT, '{"query":"answer"}')
            .with_channel("commentary")
            .with_recipient("browser.search"),
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 42."
            ).with_channel("final"),
        ]
    )

    async def generate(*args, **kwargs):
        chunk = engine_chunk("", finish=True)
        chunk["output_ids"] = get_encoding().render_conversation(
            Conversation.from_messages([next(turns)])
        )
        if serving.tokenizer_manager.generate_request.call_count == 1:
            # The parser starts inside the prompt's open assistant header.
            chunk["output_ids"] = chunk["output_ids"][2:]
        chunk["meta_info"]["completion_tokens"] = len(chunk["output_ids"])
        yield chunk

    serving.tokenizer_manager.generate_request = Mock(side_effect=generate)
    serving.tool_server = Mock()
    serving.tool_server.get_tool_session = session
    serving.tool_server.get_tool_description.return_value = ToolNamespaceConfig(
        name="browser", description="Browser", tools=[]
    )
    serving.supports_browsing = True
    request = ResponsesRequest(model="x", input="hi", tools=[{"type": "web_search"}])
    response = asyncio.run(create_response_result(serving, request))
    assert isinstance(response, ResponsesResponse), response.body
    tool_session.call_tool.assert_awaited_once_with("search", {"query": "answer"})
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "The answer is 42."
    assert serving.tokenizer_manager.generate_request.call_count == 2
    continuation = serving.tokenizer_manager.generate_request.call_args_list[1].args[0]
    assert "Search result: 42" in get_encoding().decode(continuation.input_ids)
    assert not serving.msg_store and not serving.response_store


def test_pd_tool_continuation_stops_before_side_effect(response_serving):
    serving = response_serving(mode="decode")
    context = Mock()
    context.need_builtin_tool_call.return_value = True
    context.call_tool = AsyncMock()

    async def run():
        async for _ in serving._generate_with_builtin_tools(
            "resp_tool", "hi", Mock(), {}, context
        ):
            pass

    with pytest.raises(ValueError, match="disaggregation"):
        asyncio.run(run())
    context.call_tool.assert_not_awaited()
    assert serving.tokenizer_manager.generate_request.call_count == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
