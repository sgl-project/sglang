import asyncio
import unittest
from copy import deepcopy
from unittest.mock import Mock

import orjson
from openai_harmony import Conversation, Role
from utils import (
    StreamFixture,
    create_response_result,
    engine_chunk,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.harmony_utils import get_encoding
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, ResponsesResponse
from sglang.srt.entrypoints.openai.responses_adapters import (
    decode_custom_tool_input,
    decode_custom_tool_input_prefix,
    decode_reasoning_state,
    encode_custom_tool_input,
    encode_reasoning_state,
    label_developer_content,
    response_tools_to_chat_tools,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CUSTOM_TOOL = {
    "type": "custom",
    "name": "emit_command",
    "description": "Emit a shell command.",
    "format": {"type": "text"},
}


def _custom_request(**kwargs) -> ResponsesRequest:
    payload = {
        "model": "x",
        "input": "run pwd",
        "tools": [CUSTOM_TOOL],
        "tool_choice": "required",
        "store": False,
    }
    payload.update(kwargs)
    return ResponsesRequest(**payload)


class CustomToolAdapterTestCase(CustomTestCase):
    def test_payload_survives_encode_decode(self):
        for payload in ("pwd", 'echo "hi"', "a\nb\tc", "naïve 😀", "{not json}"):
            self.assertEqual(
                decode_custom_tool_input(encode_custom_tool_input(payload)), payload
            )
        self.assertEqual(decode_custom_tool_input("pwd"), "pwd")

    def test_prefix_decode_tracks_a_growing_buffer(self):
        payload = 'echo "a\nb" 😀'
        arguments = encode_custom_tool_input(payload)
        seen = ""
        for end in range(len(arguments) + 1):
            prefix = decode_custom_tool_input_prefix(arguments[:end])
            # Monotonic, and never runs ahead of the finished value.
            self.assertTrue(prefix.startswith(seen), (seen, prefix))
            self.assertTrue(payload.startswith(prefix), (payload, prefix))
            seen = prefix
        self.assertEqual(seen, payload)
        self.assertEqual(decode_custom_tool_input_prefix('{"city": "Beijing"}'), "")


class CustomToolShimTestCase(CustomTestCase):
    def test_custom_tool_becomes_a_single_string_function_tool(self):
        request = _custom_request()
        (tool,) = response_tools_to_chat_tools(request.tools)
        self.assertEqual(tool.function.name, "emit_command")
        self.assertEqual(list(tool.function.parameters["properties"]), ["input"])
        self.assertEqual(tool.function.parameters["required"], ["input"])

        nameless = ResponsesRequest(
            model="x", input="hi", tools=[{"type": "custom"}], store=False
        )
        self.assertEqual(response_tools_to_chat_tools(nameless.tools), [])

    def test_grammar_format_is_described_to_the_model(self):
        request = _custom_request(
            tools=[
                {
                    **CUSTOM_TOOL,
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": 'start: "pwd"',
                    },
                }
            ]
        )
        (tool,) = response_tools_to_chat_tools(request.tools)
        self.assertIn("lark", tool.function.description)
        self.assertIn('start: "pwd"', tool.function.description)

    def test_required_tool_choice_accepts_a_custom_tool(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        request = _custom_request()
        output_items = serving._make_response_output_items(
            request,
            '[{"name": "emit_command", "parameters": {"input": "pwd"}}]',
            tokenizer=Mock(),
            require_reasoning=False,
        )
        (item,) = output_items
        self.assertEqual(item.type, "custom_tool_call")
        self.assertEqual(item.name, "emit_command")
        self.assertEqual(item.input, "pwd")
        self.assertTrue(item.call_id)

    def test_named_choice_parses_json_in_full_and_stream_responses(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        for tool_type in ("function", "custom"):
            for nested in (False, True):
                with self.subTest(tool_type=tool_type, nested=nested):
                    name = "emit_command"
                    choice = {"type": tool_type}
                    choice.update(
                        {"function": {"name": name}} if nested else {"name": name}
                    )
                    request = _custom_request(
                        tools=[{"type": tool_type, "name": name}],
                        tool_choice=choice,
                        stream=True,
                    )
                    raw = '[{"name":"emit_command","parameters":{"input":"pwd"}}]'
                    (item,) = serving._make_response_output_items(
                        request, raw, tokenizer=Mock(), require_reasoning=False
                    )
                    self.assertEqual(
                        item.type,
                        f"{tool_type}_tool_call"
                        if tool_type == "custom"
                        else "function_call",
                    )
                    events = StreamFixture(serving, request).run(
                        [engine_chunk(raw[:30]), engine_chunk(raw, 2, finish=True)]
                    )
                    (stream_item,) = find_completed_event(events)["response"]["output"]
                    self.assertEqual(stream_item["type"], item.type)
                    self.assertEqual(stream_item["name"], name)
                    field = "input" if tool_type == "custom" else "arguments"
                    self.assertEqual(stream_item[field], getattr(item, field))
                    self.assertNotIn("response.output_text.delta", event_types(events))

    def test_named_choice_rejects_an_undeclared_tool_before_generation(self):
        serving = make_serving()
        for stream in (False, True):
            request = _custom_request(
                tool_choice={"type": "custom", "name": "missing"}, stream=stream
            )
            result = asyncio.run(serving.create_responses(request))
            self.assertEqual(result.status_code, 400)
            self.assertIn(b"tool_choice", result.body)
        serving.tokenizer_manager.generate_request.assert_not_called()


class CustomToolReplayTestCase(CustomTestCase):
    def test_custom_tool_call_replays_through_the_shim(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "emit_command",
                "input": "pwd",
            }
        )
        self.assertEqual(message["role"], "assistant")
        (call,) = message["tool_calls"]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["function"]["name"], "emit_command")
        self.assertEqual(decode_custom_tool_input(call["function"]["arguments"]), "pwd")

    def test_custom_tool_call_output_becomes_a_tool_message(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "/workspace",
            }
        )
        self.assertEqual(
            message,
            {"role": "tool", "tool_call_id": "call_1", "content": "/workspace"},
        )

        parts = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": [{"type": "output_text", "text": "/work"}, {"text": "space"}],
            }
        )
        self.assertEqual(parts["content"], "/workspace")


class CustomToolStreamTestCase(CustomTestCase):
    def _stream(self, chunks):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        request = _custom_request(stream=True)
        return StreamFixture(serving, request).run(chunks)

    def test_input_deltas_reconstruct_the_final_payload(self):
        emitted = '[{"name": "emit_command", "parameters": {"input": "pwd"}}]'
        chunks = [engine_chunk(emitted[:i], i) for i in range(1, len(emitted))] + [
            engine_chunk(emitted, len(emitted), finish=True)
        ]

        events = self._stream(chunks)
        pairs = list(zip(event_types(events), event_payloads(events)))
        deltas = "".join(
            p["delta"] for t, p in pairs if t == "response.custom_tool_call_input.delta"
        )
        done = [p for t, p in pairs if t == "response.custom_tool_call_input.done"]

        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["input"], "pwd")
        self.assertEqual(deltas, "pwd")

        added = [p for t, p in pairs if t == "response.output_item.added"]
        item_done = [p for t, p in pairs if t == "response.output_item.done"]
        self.assertEqual(len(added), 1)
        self.assertEqual(len(item_done), 1)
        self.assertEqual(added[0]["item"]["type"], "custom_tool_call")
        self.assertEqual(added[0]["item"]["id"], item_done[0]["item"]["id"])

        final = find_completed_event(events)["response"]
        (item,) = [i for i in final["output"] if i["type"] == "custom_tool_call"]
        self.assertEqual(item["name"], "emit_command")
        self.assertEqual(item["input"], "pwd")
        self.assertTrue(item["call_id"])
        self.assertNotIn(
            "response.function_call_arguments.delta", [t for t, _ in pairs]
        )


GLM47_CALL = (
    "<tool_call>emit_command"
    "<arg_key>input</arg_key><arg_value>pwd</arg_value>"
    "</tool_call>"
)


class CustomToolGlm47FormatTestCase(CustomTestCase):
    """The shim has to survive a real model-native tool-call format, not just the
    JSON array the ``required`` constraint produces."""

    def _serving(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = "glm47"
        return serving

    def test_non_streaming_glm47_call_becomes_a_custom_tool_call(self):
        serving = self._serving()
        request = _custom_request(tool_choice="auto")
        (item,) = serving._make_response_output_items(
            request, GLM47_CALL, tokenizer=Mock(), require_reasoning=False
        )
        self.assertEqual(item.type, "custom_tool_call")
        self.assertEqual(item.name, "emit_command")
        self.assertEqual(item.input, "pwd")

    def test_streaming_glm47_call_reconstructs_the_payload(self):
        serving = self._serving()
        request = _custom_request(tool_choice="auto", stream=True)
        chunks = [engine_chunk(GLM47_CALL[:i], i) for i in range(1, len(GLM47_CALL))]
        chunks.append(engine_chunk(GLM47_CALL, len(GLM47_CALL), finish=True))

        events = StreamFixture(serving, request).run(chunks)
        pairs = list(zip(event_types(events), event_payloads(events)))
        deltas = "".join(
            p["delta"] for t, p in pairs if t == "response.custom_tool_call_input.delta"
        )
        done = [p for t, p in pairs if t == "response.custom_tool_call_input.done"]

        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["input"], "pwd")
        self.assertEqual(deltas, done[0]["input"])

        final = find_completed_event(events)["response"]
        (item,) = [i for i in final["output"] if i["type"] == "custom_tool_call"]
        self.assertEqual(item["input"], "pwd")


class ReasoningEncryptedContentTestCase(CustomTestCase):
    def test_state_survives_encode_decode(self):
        for text in ("", "step one\nstep two", "naïve 😀"):
            self.assertEqual(decode_reasoning_state(encode_reasoning_state(text)), text)
        self.assertIsNone(decode_reasoning_state("not-ours"))
        self.assertIsNone(decode_reasoning_state(None))

    def test_reasoning_item_carries_the_blob_only_when_included(self):
        without = OpenAIServingResponses._make_reasoning_item(
            ResponsesRequest(model="x", input="hi", store=False),
            "because",
            item_id="rs_1",
            status=None,
        )
        self.assertIsNone(without.encrypted_content)

        with_blob = OpenAIServingResponses._make_reasoning_item(
            ResponsesRequest(
                model="x",
                input="hi",
                store=False,
                include=["reasoning.encrypted_content"],
            ),
            "because",
            item_id="rs_1",
            status=None,
        )
        self.assertEqual(decode_reasoning_state(with_blob.encrypted_content), "because")

    def test_blob_only_reasoning_item_replays(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "reasoning",
                "summary": [],
                "content": [],
                "encrypted_content": encode_reasoning_state("because the sky"),
            }
        )
        self.assertEqual(
            message, {"role": "assistant", "reasoning_content": "because the sky"}
        )

    def test_streamed_reasoning_item_carries_the_blob(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            include=["reasoning.encrypted_content"],
        )
        events = StreamFixture(serving, request, require_reasoning=True).run(
            [
                engine_chunk("because", 1),
                engine_chunk("because</think>answer", 2, finish=True),
            ]
        )
        final = find_completed_event(events)["response"]
        (item,) = [i for i in final["output"] if i["type"] == "reasoning"]
        self.assertEqual(decode_reasoning_state(item["encrypted_content"]), "because")

    def test_non_streaming_reasoning_item_carries_the_blob(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            include=["reasoning.encrypted_content"],
        )
        output_items = serving._make_response_output_items(
            request,
            "because</think>answer",
            tokenizer=Mock(),
            require_reasoning=True,
        )
        self.assertEqual(
            decode_reasoning_state(output_items[0].encrypted_content), "because"
        )


class DeveloperMessageTestCase(CustomTestCase):
    def test_content_is_labelled(self):
        self.assertEqual(
            label_developer_content("Be terse."),
            "Developer instructions:\nBe terse.",
        )
        self.assertEqual(
            label_developer_content(
                [{"type": "input_text", "text": "Be terse."}, {"type": "input_image"}]
            ),
            [
                {"type": "input_text", "text": "Developer instructions:\nBe terse."},
                {"type": "input_image"},
            ],
        )

    def test_developer_block_follows_instructions_in_the_system_message(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            store=False,
            instructions="Respond in English.",
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": "Reply with exactly OK."}
                    ],
                },
                {"role": "user", "content": "Reply with exactly NO."},
            ],
        )
        messages = serving._construct_input_messages(request, None)
        self.assertEqual(
            messages[0],
            {
                "role": "system",
                "content": (
                    "Respond in English.\n\n"
                    "Developer instructions:\nReply with exactly OK."
                ),
            },
        )
        self.assertEqual(messages[1]["role"], "user")


class ModelValidationTestCase(CustomTestCase):
    def test_model_validation(self):
        serving = make_serving()
        error = serving._validate_model("__no_such_model__")
        self.assertIsNotNone(error)
        self.assertEqual(error.status_code, 404)
        self.assertIsNone(serving._validate_model(None))
        self.assertIsNone(serving._validate_model("x"))
        self.assertIsNone(serving._validate_model("x:my-adapter"))


class AdditionalToolsTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)

    def test_additional_tools_stay_at_their_input_position(self):
        """An inline tool inventory must survive without changing the old prefix."""
        serving = make_serving()
        tool = {
            "type": "function",
            "name": "lookup",
            "description": "Read a value.",
            "parameters": {"type": "object", "properties": {}},
        }
        inventory = {"type": "additional_tools", "role": "developer", "tools": [tool]}
        request = ResponsesRequest(
            model="x",
            instructions="Review the proposed action.",
            input=[
                {"role": "user", "content": "INPUT_BEFORE_TOOLS"},
                inventory,
                {"role": "developer", "content": "LATER_INSTRUCTION"},
                {"role": "user", "content": "INPUT_AFTER_TOOLS"},
            ],
        )
        original = deepcopy(request.model_dump())
        for encoding in (None, "kimi_k3", "dsv41"):
            with self.subTest(encoding=encoding):
                serving.chat_encoding_spec = encoding
                messages = serving._construct_input_messages(request)
                self.assertEqual(
                    [message["role"] for message in messages],
                    ["system", "user", "system", "system", "user"],
                )
                self.assertEqual(messages[0]["content"], request.instructions)
                self.assertEqual(messages[1]["content"], "INPUT_BEFORE_TOOLS")
                self.assertEqual(
                    messages[3]["content"], "Developer instructions:\nLATER_INSTRUCTION"
                )
                self.assertEqual(messages[4]["content"], "INPUT_AFTER_TOOLS")
                function = messages[2]["tools"][0]["function"]
                self.assertEqual(function["name"], "lookup")
                self.assertEqual(function["parameters"], tool["parameters"])
                if encoding is None:
                    self.assertIn('"name": "lookup"', messages[2]["content"])
                else:
                    self.assertEqual(messages[2]["content"], "")
                if encoding == "dsv41":
                    from sglang.srt.entrypoints.openai.encoding_dsv41 import (
                        encode_messages,
                    )

                    prompt = encode_messages(messages, thinking_mode="chat")
                    self.assertLess(
                        prompt.index("INPUT_BEFORE_TOOLS"), prompt.index("lookup")
                    )
                    self.assertLess(
                        prompt.index("lookup"), prompt.index("INPUT_AFTER_TOOLS")
                    )
                    self.assertEqual(prompt.count('"name": "lookup"'), 1)
                self.assertEqual(request.model_dump(), original)

    def test_deepseek_additional_tools_repro_and_stored_continuation(self):
        """Inline tool definitions must not reject the request or hide its tool call."""
        publish(
            ServerArgs(model_path="dummy", enable_response_store=True), role="tokenizer"
        )
        serving = make_serving()
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        serving.template_manager.jinja_template_content_format = "string"
        serving.reasoning_parser = None
        serving.tool_call_parser = "deepseekv41"
        serving.chat_encoding_spec = "dsv41"
        serving._dsv41_default_reasoning_effort = "high"
        model = "deepseek-ai/DeepSeek-V4.1-Flash"
        serving.tokenizer_manager.served_model_name = model

        async def generate(*args, **kwargs):
            raw = (
                '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
                '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
                "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
            )
            yield engine_chunk(raw[:80])
            yield engine_chunk(raw, 2, finish=True)

        serving.tokenizer_manager.generate_request = Mock(side_effect=generate)
        for stream in (False, True):
            request = ResponsesRequest(
                model=model,
                input=[
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [
                            {
                                "type": "function",
                                "name": "get_weather",
                                "description": "Get the current weather for a city.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": "What is the weather in Paris? Use the tool.",
                    },
                ],
                stream=stream,
            )
            response = asyncio.run(create_response_result(serving, request))
            self.assertIsInstance(response, ResponsesResponse)
            (call,) = response.output
            self.assertEqual((call.type, call.name), ("function_call", "get_weather"))
            self.assertEqual(orjson.loads(call.arguments), {"city": "Paris"})
            self.assertEqual(response.tools, [])
            prompt = serving.tokenizer_manager.tokenizer.encode.call_args.args[0]
            self.assertIn('"name": "get_weather"', prompt)
            self.assertIn('"city": {"type": "string"}', prompt)
            continuation = ResponsesRequest(
                model=model,
                previous_response_id=response.id,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": "Sunny",
                    }
                ],
                tool_choice={"type": "function", "name": "get_weather"},
                stream=stream,
            )
            next_response = asyncio.run(create_response_result(serving, continuation))
            self.assertIsInstance(next_response, ResponsesResponse)
            self.assertEqual(next_response.output[0].name, "get_weather")

    def test_ambiguous_inline_tool_names_are_rejected_before_generation(self):
        serving = make_serving()
        tool = {"type": "function", "name": "lookup"}
        inventory = {"type": "additional_tools", "role": "developer", "tools": [tool]}
        for top_tools, items in (
            ([tool], [inventory]),
            ([], [inventory, inventory]),
            ([tool, tool], []),
        ):
            with self.subTest(top_tools=top_tools):
                request = ResponsesRequest(model="x", tools=top_tools, input=items)
                response = asyncio.run(serving.create_responses(request))
                self.assertEqual(response.status_code, 400)
                self.assertIn(b"Tool names must be unique", response.body)
                serving.tokenizer_manager.generate_request.assert_not_called()

    def test_additional_tools_do_not_replace_top_level_tools(self):
        serving = make_serving()
        serving.chat_encoding_spec = "dsv41"
        request = ResponsesRequest(
            model="x",
            tools=[{"type": "function", "name": "initial"}],
            input=[
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "function", "name": "later"}],
                },
                {"role": "user", "content": "call later"},
            ],
        )
        messages = serving._construct_input_messages(request)
        self.assertEqual(messages[0], {"role": "system", "content": ""})
        self.assertEqual(messages[1]["tools"][0]["function"]["name"], "later")
        self.assertEqual(
            [tool.name for tool in serving._effective_response_tools(request)],
            ["initial", "later"],
        )

    def test_additional_tools_replay_matches_previous_response(self):
        serving = make_serving()
        history = [
            {"role": "user", "content": "before"},
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{"type": "function", "name": "lookup"}],
            },
            {"role": "assistant", "content": "ready"},
        ]
        serving.msg_store["resp_previous"] = deepcopy(history)
        followup = ResponsesRequest(
            model="x", previous_response_id="resp_previous", input="call lookup"
        )
        replay = ResponsesRequest(
            model="x", input=history + [{"role": "user", "content": "call lookup"}]
        )
        self.assertEqual(
            serving._construct_input_messages(followup),
            serving._construct_input_messages(replay),
        )
        self.assertEqual(
            serving._effective_response_tools(followup),
            serving._effective_response_tools(replay),
        )
        self.assertEqual(serving.msg_store["resp_previous"], history)

    def test_inline_function_and_custom_calls_survive_full_and_stream_output(self):
        serving = make_serving()
        serving.tool_call_parser = None
        for tool_type in ("function", "custom"):
            for choice in ("required", {"type": tool_type, "name": "lookup"}, "none"):
                with self.subTest(tool_type=tool_type, choice=choice):
                    request = ResponsesRequest(
                        model="x",
                        input=[
                            {
                                "type": "additional_tools",
                                "role": "developer",
                                "tools": [{"type": tool_type, "name": "lookup"}],
                            },
                            {"role": "user", "content": "call lookup"},
                        ],
                        tool_choice=choice,
                        store=False,
                    )
                    raw = '[{"name":"lookup","parameters":{"input":"value"}}]'
                    (full,) = serving._make_response_output_items(
                        request, raw, tokenizer=Mock(), require_reasoning=False
                    )
                    events = event_payloads(
                        StreamFixture(serving, request).run(
                            [engine_chunk(raw[:20]), engine_chunk(raw, 2, finish=True)]
                        )
                    )
                    completed = next(
                        e for e in events if e["type"] == "response.completed"
                    )
                    (streamed,) = completed["response"]["output"]
                    if choice == "none":
                        self.assertEqual(full.type, "message")
                        self.assertEqual(streamed["type"], "message")
                        continue
                    expected_type = (
                        "custom_tool_call" if tool_type == "custom" else "function_call"
                    )
                    self.assertEqual(full.type, expected_type)
                    self.assertEqual(streamed["type"], expected_type)
                    self.assertEqual(full.name, "lookup")
                    self.assertEqual(streamed["name"], "lookup")
                    field = "input" if tool_type == "custom" else "arguments"
                    self.assertEqual(getattr(full, field), streamed[field])
                    self.assertEqual(
                        getattr(full, field),
                        "value" if tool_type == "custom" else '{"input": "value"}',
                    )

    def test_harmony_additional_tools_are_developer_messages_in_place(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {"role": "user", "content": "before"},
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "description": "Read a value.",
                        }
                    ],
                },
                {"role": "user", "content": "after"},
            ],
        )
        messages = serving._construct_input_messages_with_harmony(request, None)
        self.assertEqual(
            [m.author.role for m in messages],
            [Role.SYSTEM, Role.DEVELOPER, Role.USER, Role.DEVELOPER, Role.USER],
        )
        rendered = get_encoding().decode(
            get_encoding().render_conversation(Conversation.from_messages(messages))
        )
        self.assertLess(rendered.index("before"), rendered.index("lookup"))
        self.assertLess(rendered.index("lookup"), rendered.index("after"))
        serving.msg_store["resp_previous"] = messages[2:]
        followup = ResponsesRequest(
            model="x", previous_response_id="resp_previous", input="call lookup"
        )
        self.assertEqual(
            serving._effective_response_tools(followup),
            serving._effective_response_tools(request),
        )

        request.input[1]["tools"][0]["type"] = "custom"
        with self.assertRaisesRegex(ValueError, "function tools only"):
            serving._construct_input_messages_with_harmony(request, None)


if __name__ == "__main__":
    unittest.main()
