import unittest
from unittest.mock import Mock

from utils import StreamFixture, engine_chunk, make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def namespace(name):
    return {
        "type": "namespace",
        "name": name,
        "description": "Weather tools",
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "strict": True,
            }
        ],
    }


class ResponseNamespaceToolsTest(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.serving = make_serving()
        self.request = ResponsesRequest(
            model="test-model",
            input="weather",
            tools=[namespace("weather")],
            store=False,
        )

    def test_flattening_preserves_schemas_and_distinguishes_namespaces(self):
        self.request = ResponsesRequest(
            model="test-model",
            input="weather",
            tools=[namespace("weather"), namespace("travel")],
        )
        tools = self.serving._response_tools_to_chat_tools(self.request)
        self.assertEqual(
            [t.function.name for t in tools], ["weather.lookup", "travel.lookup"]
        )
        self.assertEqual(
            tools[0].function.parameters, namespace("weather")["tools"][0]["parameters"]
        )
        self.assertEqual(tools[0].function.description, "Look up the weather")
        self.assertTrue(tools[0].function.strict)

    def test_qualified_name_collision_is_rejected(self):
        request = ResponsesRequest(
            model="test-model",
            input="weather",
            tools=[
                namespace("weather"),
                {"type": "function", "name": "weather.lookup"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous function tool name"):
            self.serving._response_tools_to_chat_tools(request)

    def test_dotted_top_level_function_is_not_reinterpreted(self):
        request = ResponsesRequest(
            model="test-model",
            input="hi",
            tools=[{"type": "function", "name": "weather.lookup"}],
        )
        self.assertEqual(
            self.serving._response_function_identity(request, "weather.lookup"),
            {"name": "weather.lookup"},
        )

    def test_forced_namespace_tool_choice_is_qualified(self):
        self.request.tool_choice = {
            "type": "function",
            "name": "lookup",
            "namespace": "weather",
        }
        self.assertEqual(
            self.request.effective_tool_choice(),
            {"type": "function", "name": "weather.lookup"},
        )

    def test_nonstreaming_call_and_stateless_replay_preserve_namespace(self):
        self.request.tool_choice = "required"
        self.serving.tool_call_parser = None
        output = self.serving._make_response_output_items(
            self.request,
            '[{"name":"weather.lookup","parameters":{"city":"Paris"}}]',
            tokenizer=Mock(),
            require_reasoning=False,
        )
        call = output[0].model_dump(exclude_none=True)
        self.assertEqual(call["name"], "lookup")
        self.assertEqual(call["namespace"], "weather")
        for tool_output in ("sunny", [{"type": "input_text", "text": "sunny"}]):
            with self.subTest(tool_output=tool_output):
                replay = ResponsesRequest(
                    model="test-model",
                    store=False,
                    tools=[namespace("weather")],
                    input=[
                        {"role": "user", "content": "weather"},
                        call,
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": tool_output,
                        },
                    ],
                )
                messages = self.serving._construct_input_messages(replay)
                self.assertEqual(
                    messages[1]["tool_calls"][0]["function"]["name"], "weather.lookup"
                )
                self.assertEqual(messages[2]["tool_call_id"], call["call_id"])
        self.assertFalse(self.serving.response_store)
        self.assertFalse(self.serving.msg_store)

    def test_streaming_call_preserves_namespace_in_all_item_events(self):
        from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

        self.serving.tool_call_parser = "qwen3_coder"
        detector = Qwen3CoderDetector()
        text = (
            f"{detector.tool_call_start_token}{detector.tool_call_prefix}weather.lookup>"
            f"{detector.parameter_prefix}city>Paris{detector.parameter_end_token}"
            f"{detector.function_end_token}{detector.tool_call_end_token}"
        )
        events = StreamFixture(self.serving, self.request).run_seq(
            [engine_chunk(text), engine_chunk(text, finish=True)]
        )
        items = [
            p["item"]
            for t, p in events
            if t in ("response.output_item.added", "response.output_item.done")
            and p["item"]["type"] == "function_call"
        ]
        self.assertEqual(len(items), 2)
        for item in items:
            self.assertEqual((item["namespace"], item["name"]), ("weather", "lookup"))
        final = next(p["response"] for t, p in events if t == "response.completed")
        self.assertEqual(final["output"][-1]["namespace"], "weather")
        self.assertFalse(self.serving.response_store)

    def test_nested_namespaces_and_missing_member_names_are_rejected(self):
        for member in (namespace("nested"), {"type": "function"}):
            with self.subTest(member=member), self.assertRaises(ValueError):
                ResponsesRequest(
                    model="test-model",
                    input="hi",
                    tools=[{"type": "namespace", "name": "weather", "tools": [member]}],
                )


if __name__ == "__main__":
    unittest.main()
