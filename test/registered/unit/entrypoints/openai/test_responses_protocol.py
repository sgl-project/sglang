import unittest

from utils import make_serving  # noqa: F401 — bootstrap import

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, UsageInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class ResponsesRequestTestCase(unittest.TestCase):
    def test_function_tool_accepted(self):
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

    def test_function_tool_requires_name(self):
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x", input="hi", tools=[{"type": "function"}], store=False
            )
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": "function", "name": ""}],
                store=False,
            )

    def test_extended_tool_types_accepted(self):
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

    def test_namespace_tool_carries_inner_tools_list(self):
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


class ResponsesSamplingParamsTestCase(unittest.TestCase):
    def test_processed_stop_and_tool_constraint_propagate(self):
        request = ResponsesRequest(model="x", input="call the tool", store=False)
        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            stop=["</s>"],
            tool_call_constraint=("json_schema", {"type": "object"}),
        )
        self.assertEqual(params["stop"], ["</s>"])
        self.assertEqual(params["json_schema"], '{"type": "object"}')

    def test_constraint_conflict_raises(self):
        request = ResponsesRequest(model="x", input="hi", store=False)
        with self.assertRaises(ValueError):
            request.to_sampling_params(
                default_max_tokens=128,
                default_params={"json_schema": '{"type": "object"}'},
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

    def test_structural_tag_with_model_dump(self):
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


class ResponsesResponseFromRequestTestCase(unittest.TestCase):
    def test_parallel_tool_calls_false_preserved(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        request = ResponsesRequest(
            model="x", input="hi", parallel_tool_calls=False, store=False
        )
        response = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertFalse(response.parallel_tool_calls)


class ResponsesToolChoiceTestCase(unittest.TestCase):
    def test_flat_tool_choice_normalized_to_nested(self):
        from sglang.srt.entrypoints.openai.protocol import ToolChoice

        # OpenAI Responses API flat form {"type":"function","name":X}
        request = ResponsesRequest(
            model="x",
            input="call get_weather",
            tool_choice={"type": "function", "name": "get_weather"},
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            store=False,
        )
        self.assertIsInstance(request.tool_choice, ToolChoice)
        self.assertEqual(request.tool_choice.function.name, "get_weather")

    def test_string_tool_choice_unchanged(self):
        for choice in ("auto", "required", "none"):
            request = ResponsesRequest(
                model="x", input="hi", tool_choice=choice, store=False
            )
            self.assertEqual(request.tool_choice, choice)

    def test_nested_tool_choice_accepted(self):
        from sglang.srt.entrypoints.openai.protocol import ToolChoice

        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice={"type": "function", "function": {"name": "f"}},
            tools=[
                {
                    "type": "function",
                    "name": "f",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            store=False,
        )
        self.assertIsInstance(request.tool_choice, ToolChoice)
        self.assertEqual(request.tool_choice.function.name, "f")


if __name__ == "__main__":
    unittest.main()
