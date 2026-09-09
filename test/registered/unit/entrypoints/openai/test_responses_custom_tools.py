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

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.responses_adapters import (
    decode_custom_tool_input,
    decode_custom_tool_input_prefix,
    decode_reasoning_state,
    encode_custom_tool_input,
    encode_reasoning_state,
    label_developer_content,
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

    def test_bare_text_falls_back_to_raw_arguments(self):
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

    def test_prefix_decode_ignores_a_foreign_buffer(self):
        self.assertEqual(decode_custom_tool_input_prefix('{"city": "Beijing"}'), "")


class CustomToolShimTestCase(CustomTestCase):
    def test_custom_tool_becomes_a_single_string_function_tool(self):
        request = _custom_request()
        (tool,) = OpenAIServingResponses._response_tools_to_chat_tools(request)
        self.assertEqual(tool.function.name, "emit_command")
        self.assertEqual(list(tool.function.parameters["properties"]), ["input"])
        self.assertEqual(tool.function.parameters["required"], ["input"])

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

    def test_nameless_custom_tool_is_skipped(self):
        request = ResponsesRequest(
            model="x", input="hi", tools=[{"type": "custom"}], store=False
        )
        self.assertEqual(
            OpenAIServingResponses._response_tools_to_chat_tools(request), []
        )

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

    def test_function_tools_still_report_json_arguments(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        request = _custom_request(
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ]
        )
        output_items = serving._make_response_output_items(
            request,
            '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]',
            tokenizer=Mock(),
            require_reasoning=False,
        )
        (item,) = output_items
        self.assertEqual(item.type, "function_call")
        self.assertEqual(item.arguments, '{"city": "Beijing"}')


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

    def test_tool_output_content_parts_are_flattened(self):
        message = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": [{"type": "output_text", "text": "/work"}, {"text": "space"}],
            }
        )
        self.assertEqual(message["content"], "/workspace")


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

        final = find_completed_event(events)["response"]
        (item,) = [i for i in final["output"] if i["type"] == "custom_tool_call"]
        self.assertEqual(item["name"], "emit_command")
        self.assertEqual(item["input"], "pwd")
        self.assertTrue(item["call_id"])
        self.assertNotIn(
            "response.function_call_arguments.delta", [t for t, _ in pairs]
        )

    def test_item_lifetime_events_pair_up(self):
        emitted = '[{"name": "emit_command", "parameters": {"input": "ls -la"}}]'
        events = self._stream(
            [engine_chunk(emitted, 1), engine_chunk(emitted, 2, finish=True)]
        )
        pairs = list(zip(event_types(events), event_payloads(events)))
        added = [p for t, p in pairs if t == "response.output_item.added"]
        done = [p for t, p in pairs if t == "response.output_item.done"]
        self.assertEqual(len(added), 1)
        self.assertEqual(len(done), 1)
        self.assertEqual(added[0]["item"]["type"], "custom_tool_call")
        self.assertEqual(added[0]["item"]["id"], done[0]["item"]["id"])
        self.assertEqual(done[0]["item"]["input"], "ls -la")


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

    def test_foreign_blob_is_rejected(self):
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
    def test_string_content_is_labelled(self):
        self.assertEqual(
            label_developer_content("Be terse."),
            "Developer instructions:\nBe terse.",
        )

    def test_first_text_part_is_labelled(self):
        self.assertEqual(
            label_developer_content(
                [{"type": "input_text", "text": "Be terse."}, {"type": "input_image"}]
            ),
            [
                {"type": "input_text", "text": "Developer instructions:\nBe terse."},
                {"type": "input_image"},
            ],
        )

    def test_label_is_prepended_when_no_text_part_exists(self):
        self.assertEqual(
            label_developer_content([{"type": "input_image"}]),
            [
                {"type": "input_text", "text": "Developer instructions:"},
                {"type": "input_image"},
            ],
        )

    def test_missing_content_is_left_alone(self):
        self.assertIsNone(label_developer_content(None))

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
    def test_unknown_model_is_rejected(self):
        serving = make_serving()
        error = serving._validate_model("__no_such_model__")
        self.assertIsNotNone(error)
        self.assertEqual(error.status_code, 404)

    def test_served_and_lora_qualified_names_are_accepted(self):
        serving = make_serving()
        self.assertIsNone(serving._validate_model(None))
        self.assertIsNone(serving._validate_model("x"))
        self.assertIsNone(serving._validate_model("x:my-adapter"))


if __name__ == "__main__":
    unittest.main()
