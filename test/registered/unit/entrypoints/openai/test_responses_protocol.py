import json
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

    def test_text_format_maps_to_json_schema_constraint(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        for fmt, expected in [
            ({"type": "json_schema", "name": "p", "schema": schema}, schema),
            ({"type": "json_object"}, {"type": "object"}),
        ]:
            req = ResponsesRequest(
                model="x", input="hi", store=False, text={"format": fmt}
            )
            params = req.to_sampling_params(default_max_tokens=128, default_params={})
            self.assertEqual(json.loads(params["json_schema"]), expected, fmt)
        plain = ResponsesRequest(
            model="x", input="hi", store=False, text={"format": {"type": "text"}}
        ).to_sampling_params(default_max_tokens=128, default_params={})
        self.assertNotIn("json_schema", plain)

    def test_text_format_conflicts_with_tool_constraint(self):
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "p",
                    "schema": {"type": "object"},
                }
            },
        )
        with self.assertRaises(ValueError):
            request.to_sampling_params(
                default_max_tokens=128,
                default_params={},
                tool_call_constraint=("json_schema", {"type": "object"}),
            )


class IncludeOutputLogprobsTestCase(unittest.TestCase):
    def test_detected_only_for_logprobs_include(self):
        def has(include):
            return ResponsesRequest(
                model="x", input="hi", store=False, include=include
            ).is_include_output_logprobs()

        self.assertTrue(has(["message.output_text.logprobs"]))
        self.assertFalse(has(None))
        self.assertFalse(has(["reasoning.encrypted_content"]))


class ThinkingControlTestCase(unittest.TestCase):
    def test_effort_values_accepted(self):
        for effort in ("none", "minimal", "low", "medium", "high"):
            req = ResponsesRequest(
                model="x", input="hi", store=False, reasoning={"effort": effort}
            )
            self.assertEqual(req.reasoning.effort, effort)

    def test_thinking_disabled_for_constrained_generations(self):
        # effort=none, JSON output, and forced tool calls all constrain the
        # generation so it can't interleave free reasoning -> thinking off.
        for kw in (
            {"reasoning": {"effort": "none"}},
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "p",
                        "schema": {"type": "object"},
                    }
                }
            },
            {"text": {"format": {"type": "json_object"}}},
            {"tool_choice": "required"},
            {"tool_choice": {"type": "function", "name": "f"}},
        ):
            ctk = ResponsesRequest(
                model="x", input="hi", store=False, **kw
            ).chat_template_kwargs
            self.assertEqual(
                (ctk["enable_thinking"], ctk["thinking"]), (False, False), kw
            )

    def test_thinking_untouched_otherwise(self):
        for kw in (
            {"reasoning": {"effort": "medium"}},
            {"text": {"format": {"type": "text"}}},
            {},
        ):
            req = ResponsesRequest(model="x", input="hi", store=False, **kw)
            self.assertIsNone(req.chat_template_kwargs, kw)

    def test_explicit_chat_template_kwargs_preserved(self):
        req = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            chat_template_kwargs={"enable_thinking": True},
        )
        self.assertTrue(req.chat_template_kwargs["enable_thinking"])


class ResponsesResponseFromRequestTestCase(unittest.TestCase):
    def test_requested_text_format_is_echoed(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            text={"format": {"type": "json_schema", "name": "p", "schema": schema}},
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
        self.assertEqual(response.text["format"]["type"], "json_schema")

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

    def test_incomplete_status_sets_incomplete_details(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        request = ResponsesRequest(model="x", input="hi", store=False)
        incomplete = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="incomplete",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertEqual(incomplete.status, "incomplete")
        self.assertEqual(incomplete.incomplete_details, {"reason": "max_output_tokens"})

        completed = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertIsNone(completed.incomplete_details)

    def test_usage_serialized_in_responses_shape(self):
        from sglang.srt.entrypoints.openai.protocol import (
            PromptTokensDetails,
            ResponsesResponse,
        )

        request = ResponsesRequest(model="x", input="hi", store=False)
        resp = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(
                prompt_tokens=11,
                completion_tokens=102,
                total_tokens=113,
                reasoning_tokens=7,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=3),
            ),
        )
        usage = resp.model_dump()["usage"]
        self.assertEqual(usage["input_tokens"], 11)
        self.assertEqual(usage["output_tokens"], 102)
        self.assertEqual(usage["total_tokens"], 113)
        self.assertEqual(usage["output_tokens_details"]["reasoning_tokens"], 7)
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 3)
        # Chat-style keys must be gone.
        self.assertNotIn("prompt_tokens", usage)
        self.assertNotIn("completion_tokens", usage)

    def test_effort_none_not_echoed_so_streaming_event_validates(self):
        import openai.types.responses as ort

        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        req = ResponsesRequest(
            model="x", input="hi", store=False, reasoning={"effort": "none"}
        )
        resp = ResponsesResponse.from_request(
            req,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="in_progress",
            usage=None,
        )
        # "none" is our request-side extension; OpenAI's Response schema rejects
        # it, so it must not be echoed (else the typed streaming event below
        # raises a ValidationError and the stream emits nothing).
        self.assertIsNone(resp.reasoning["effort"])
        ort.ResponseCreatedEvent(
            type="response.created", sequence_number=0, response=resp.model_dump()
        )


class InputItemStringIdTestCase(unittest.TestCase):
    """A response.output item replayed into input (string id + content) must
    keep its content rather than collapse to an item-reference."""

    def test_string_id_dropped_only_for_content_items(self):
        norm = ResponsesRequest._normalize_input_item_for_validation
        kept = norm(
            {
                "id": "msg_x",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        )
        self.assertNotIn("id", kept)
        self.assertTrue(kept["content"])
        # int ids and bare item-references are left alone
        self.assertEqual(norm({"id": 123, "content": [{"type": "text"}]})["id"], 123)
        ref = {"type": "item_reference", "id": "msg_ref"}
        self.assertEqual(norm(ref), ref)

    def test_request_accepts_replayed_output_item(self):
        # Construction must not raise (the bug returned 400).
        ResponsesRequest(
            model="x",
            store=False,
            input=[
                {
                    "id": "msg_x",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
        )


class ToolChoiceObjectFormTestCase(unittest.TestCase):
    def test_named_function_object_accepted(self):
        req = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            tool_choice={"type": "function", "name": "get_weather"},
        )
        self.assertEqual(req.tool_choice["name"], "get_weather")


if __name__ == "__main__":
    unittest.main()
