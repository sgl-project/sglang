import asyncio
import unittest
from unittest.mock import Mock

from utils import (
    StreamFixture,
    engine_chunk,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from sglang.srt.entrypoints.openai.responses_adapters import (
    decode_custom_tool_input,
    decode_custom_tool_input_prefix,
    decode_reasoning_state,
    encode_custom_tool_input,
    encode_reasoning_state,
    label_developer_content,
    split_namespaced_call,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
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
        (tool,) = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertEqual(tool.function.name, "emit_command")
        self.assertEqual(list(tool.function.parameters["properties"]), ["input"])
        self.assertEqual(tool.function.parameters["required"], ["input"])

        nameless = ResponsesRequest(
            model="x", input="hi", tools=[{"type": "custom"}], store=False
        )
        self.assertEqual(
            OpenAIServingResponses._response_tools_to_chat_tools(nameless), []
        )

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
        (tool,) = OpenAIServingResponses._response_tools_to_chat_tools(request)
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


NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "weather",
    "description": "Weather services.",
    "tools": [
        {"name": "lookup", "parameters": {"type": "object"}},
        {"name": "wait", "strict": True},
    ],
}


def _namespace_request(**kwargs) -> ResponsesRequest:
    payload = {
        "model": "x",
        "input": "weather in Paris",
        "tools": [NAMESPACE_TOOL],
        "tool_choice": "required",
        "store": False,
    }
    payload.update(kwargs)
    return ResponsesRequest(**payload)


class NamespaceDeclarationTestCase(CustomTestCase):
    def test_members_flatten_to_qualified_function_tools(self):
        # Pre-namespace support the declaration passed validation and was
        # silently dropped, so the model never saw a callable and never
        # called it.
        request = _namespace_request(tool_choice="auto")
        tools = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertEqual(
            [t.function.name for t in tools], ["weather.lookup", "weather.wait"]
        )
        # Inner description wins; the namespace description fills members
        # that omit one; strict flags survive the flattening.
        self.assertEqual(tools[0].function.description, "Weather services.")
        self.assertEqual(tools[0].function.parameters, {"type": "object"})
        self.assertTrue(tools[1].function.strict)

    def test_only_declared_namespaces_split_on_call(self):
        # An exact declaration outranks a prefix split, so a plain function
        # whose name contains a dot is never reinterpreted; undeclared
        # dotted names pass through unchanged.
        request = _namespace_request(
            tool_choice="auto",
            tools=[NAMESPACE_TOOL, {"type": "function", "name": "weather.lookup"}],
        )
        OpenAIServingResponses._response_tools_to_chat_tools(request)
        declared = {"weather.lookup"}
        self.assertEqual(
            split_namespaced_call("weather.lookup", {"weather"}, declared),
            ("weather.lookup", None),
        )
        self.assertEqual(
            split_namespaced_call("weather.wait", {"weather"}, declared),
            ("wait", "weather"),
        )
        self.assertEqual(
            split_namespaced_call("unrelated.dot", {"weather"}, declared),
            ("unrelated.dot", None),
        )


class NamespaceReplayTestCase(CustomTestCase):
    def test_namespaced_function_call_replay_requalifies(self):
        # The chat template must see the flattened name the model was shown
        # at declaration time, or the replayed turn is unrecognizable.
        serving = make_serving()
        message = serving._normalize_response_message_for_chat(
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "namespace": "weather",
                "arguments": "{}",
            }
        )
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "weather.lookup")


class NamespaceOutputTestCase(CustomTestCase):
    def setUp(self):
        self.serving = make_serving()
        self.serving.reasoning_parser = None
        self.serving.tool_call_parser = None

    def test_required_call_splits_name_and_namespace(self):
        request = _namespace_request()
        output = self.serving._make_response_output_items(
            request,
            '[{"name": "weather.lookup", "parameters": {"city": "SF"}}]',
            tokenizer=Mock(),
            require_reasoning=False,
        )
        item = output[0].model_dump()
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["name"], "lookup")
        self.assertEqual(item["namespace"], "weather")

    def test_namespaced_item_survives_response_serialization(self):
        # The SDK unions type ``output`` through ``ResponseFunctionToolCall``,
        # which serializes a subclass instance without ``namespace`` unless
        # the widened arm leads; a response echoing a namespaced call must
        # keep the field.
        item = self.serving._make_tool_call_item(
            "weather.lookup",
            "{}",
            custom_names=frozenset(),
            namespaces={"weather"},
            declared_names=frozenset(),
        )
        response = ResponsesResponse(model="x", status="completed", output=[item])
        self.assertEqual(response.model_dump()["output"][0]["namespace"], "weather")


class NamespaceStreamTestCase(CustomTestCase):
    def setUp(self):
        self.serving = make_serving()
        self.serving.reasoning_parser = None
        self.serving.tool_call_parser = None

    def test_added_and_done_items_split_the_qualified_name(self):
        request = _namespace_request(stream=True)
        payload = '[{"name": "weather.lookup", "parameters": {"city": "SF"}}]'
        chunks = []
        sent = 0
        while sent < len(payload):
            sent += min(9, len(payload) - sent)
            chunks.append(
                engine_chunk(payload[:sent], sent, finish=sent == len(payload))
            )
        events = StreamFixture(self.serving, request).run(chunks)
        types = event_types(events)
        payloads = event_payloads(events)

        self.assertIn("response.function_call_arguments.delta", types)
        added = [
            p for t, p in zip(types, payloads) if t == "response.output_item.added"
        ]
        self.assertEqual(added[0]["item"]["name"], "lookup")
        self.assertEqual(added[0]["item"]["namespace"], "weather")
        done = [p for t, p in zip(types, payloads) if t == "response.output_item.done"]
        self.assertEqual(done[0]["item"]["namespace"], "weather")
        completed = find_completed_event(events)["response"]
        self.assertEqual(completed["output"][0]["namespace"], "weather")


class NamespaceHarmonyTestCase(CustomTestCase):
    def setUp(self):
        self.serving = make_serving()
        self.serving.use_harmony = True

    def test_developer_message_renders_flattened_members(self):
        from sglang.srt.entrypoints.harmony_utils import get_developer_message
        from sglang.srt.entrypoints.openai.protocol import ResponseTool

        tools = [ResponseTool.model_validate(NAMESPACE_TOOL)]
        dev_msg = get_developer_message("be helpful", tools)
        rendered = str(dev_msg.to_dict())
        for name in ("weather.lookup", "weather.wait"):
            self.assertIn(name, rendered)

    def test_output_items_split_the_qualified_name(self):
        from openai_harmony import Message, Role

        from sglang.srt.entrypoints.context import HarmonyContext

        request = _namespace_request(tool_choice="auto")
        namespaces = {"weather"}
        declared = frozenset()
        calls = [
            Message.from_role_and_content(Role.ASSISTANT, '{"city": "SF"}')
            .with_channel("commentary")
            .with_recipient("functions.weather.lookup")
            .with_content_type("json"),
            Message.from_role_and_content(Role.ASSISTANT, "{}")
            .with_channel("commentary")
            .with_recipient("functions.unrelated.name")
            .with_content_type("json"),
        ]
        context = HarmonyContext(calls, {})
        context.num_init_messages = 0
        output = self.serving._make_response_output_items_with_harmony(
            context, namespaces, declared
        )

        namespaced = output[0].model_dump()
        self.assertEqual(namespaced["name"], "lookup")
        self.assertEqual(namespaced["namespace"], "weather")
        # An undeclared dotted recipient keeps its full name unsplit.
        self.assertEqual(output[1].name, "unrelated.name")

    def test_namespaced_replay_requalifies_the_recipient(self):
        from sglang.srt.entrypoints.harmony_utils import parse_response_input

        msg = parse_response_input(
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "namespace": "weather",
                "arguments": "{}",
            },
            [],
        )
        self.assertEqual(msg.recipient, "functions.weather.lookup")


if __name__ == "__main__":
    unittest.main()
