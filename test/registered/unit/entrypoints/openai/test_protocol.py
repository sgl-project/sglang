"""
Unit tests for srt/entrypoints/openai/protocol.py
"""

import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageGenericParam,
    ChatCompletionRequest,
    Function,
    Tool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import convert_json_schema_to_str

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")

_PROTOCOL_LOGGER = "sglang.srt.entrypoints.openai.protocol"


class TestChatCompletionRequest(CustomTestCase):
    def test_tool_choice_defaults_to_none_without_tools(self):
        """Test tool_choice defaults to none when no tools are provided."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(req.tool_choice, "none")

    def test_tool_choice_defaults_to_auto_with_tools(self):
        """Test tool_choice defaults to auto when tools are provided."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                Tool(
                    function=Function(
                        name="lookup_weather",
                        description="Get weather for a city.",
                    )
                )
            ],
        )
        self.assertEqual(req.tool_choice, "auto")

    def test_normalize_reasoning_inputs_maps_effort_and_enabled_string(self):
        """Test reasoning input maps effort and string-enabled flag to thinking."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "think carefully"}],
            reasoning={"reasoning_effort": "medium", "enabled": "On"},
        )

        self.assertEqual(req.reasoning_effort, "medium")
        self.assertEqual(req.chat_template_kwargs, {"thinking": True})

    def test_normalize_reasoning_inputs_none_effort_sets_disable_defaults(self):
        """Test none effort sets disable defaults without overriding explicit kwargs."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "no thinking"}],
            reasoning={"effort": "none"},
            chat_template_kwargs={"thinking": True},
        )

        self.assertEqual(req.reasoning_effort, "none")
        self.assertEqual(req.chat_template_kwargs["thinking"], True)
        self.assertEqual(req.chat_template_kwargs["enable_thinking"], False)

    def test_json_schema_response_format_migrates_legacy_schema_field(self):
        """Test legacy response_format.schema is migrated into json_schema."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={
                "type": "json_schema",
                "schema": {
                    "title": "MathSchema",
                    "type": "object",
                    "properties": {
                        "strict": {"type": "boolean", "default": True},
                        "answer": {"type": "number"},
                    },
                },
            },
        )

        self.assertEqual(req.response_format.type, "json_schema")
        self.assertIsNotNone(req.response_format.json_schema)
        self.assertEqual(req.response_format.json_schema.name, "MathSchema")
        self.assertTrue(req.response_format.json_schema.strict)
        self.assertEqual(
            req.response_format.json_schema.schema_,
            {
                "title": "MathSchema",
                "type": "object",
                "properties": {"answer": {"type": "number"}},
            },
        )

    def test_json_schema_response_format_already_set_is_not_overwritten(self):
        """Test existing response_format.json_schema is preserved as provided."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "MySchema", "schema": {"type": "string"}},
            },
        )
        self.assertEqual(req.response_format.json_schema.name, "MySchema")

    def test_to_sampling_params_applies_tool_call_constraint_when_no_conflict(self):
        """Test tool call constraints are applied when no decoding conflicts exist."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Need a tool call"}],
            max_completion_tokens=32,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            repetition_penalty=None,
        )

        params = req.to_sampling_params(
            stop=["</s>"],
            model_generation_config={
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.05,
                "repetition_penalty": 1.15,
            },
            tool_call_constraint=("json_schema", {"type": "object"}),
        )

        self.assertEqual(params["max_new_tokens"], 32)
        self.assertEqual(params["temperature"], 0.3)
        self.assertEqual(params["top_p"], 0.8)
        self.assertEqual(params["top_k"], 20)
        self.assertEqual(params["min_p"], 0.05)
        self.assertEqual(params["repetition_penalty"], 1.15)
        self.assertEqual(
            params["json_schema"], convert_json_schema_to_str({"type": "object"})
        )

    def test_to_sampling_params_skips_tool_call_constraint_when_constraints_exist(self):
        """Test tool call constraints are skipped when constrained decoding is set."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Need regex constrained output"}],
            max_completion_tokens=16,
            regex="\\d+",
        )

        with self.assertLogs(_PROTOCOL_LOGGER, level="WARNING") as logs:
            params = req.to_sampling_params(
                stop=[],
                model_generation_config={},
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

        self.assertEqual(params["regex"], "\\d+")
        self.assertNotIn("json_schema", params)
        self.assertTrue(
            any(
                "Constrained decoding is not compatible with tool calls." in msg
                for msg in logs.output
            )
        )

    def test_to_sampling_params_response_format_json_schema_sets_json_schema_param(
        self,
    ):
        """Test json_schema response_format sets the json_schema sampling param."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "structured output"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "MyObj",
                    "schema": {"type": "object"},
                    "strict": False,
                },
            },
        )

        params = req.to_sampling_params(stop=[], model_generation_config={})

        self.assertIn("json_schema", params)
        self.assertEqual(
            params["json_schema"],
            convert_json_schema_to_str({"type": "object"}),
        )

    def test_to_sampling_params_response_format_json_object_sets_open_schema(self):
        """Test json_object response_format sets an open object json_schema."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "return any JSON"}],
            response_format={"type": "json_object"},
        )

        params = req.to_sampling_params(stop=[], model_generation_config={})

        self.assertEqual(params["json_schema"], '{"type": "object"}')


class TestChatCompletionMessageGenericParam(CustomTestCase):
    def test_role_normalization_is_case_insensitive(self):
        """Test message roles are normalized to lowercase case-insensitively."""
        message = ChatCompletionMessageGenericParam(role="Assistant", content="ok")
        self.assertEqual(message.role, "assistant")

    def test_role_validation_rejects_unknown_role(self):
        """Test role validation rejects roles outside the allowed role set."""
        with self.assertRaises(ValidationError):
            ChatCompletionMessageGenericParam(role="guest", content="x")

    def test_all_valid_roles_are_accepted(self):
        """Test all supported message roles are accepted without validation errors."""
        for role in ("system", "assistant", "tool", "function", "developer"):
            with self.subTest(role=role):
                msg = ChatCompletionMessageGenericParam(role=role, content="x")
                self.assertEqual(msg.role, role)


if __name__ == "__main__":
    unittest.main()
