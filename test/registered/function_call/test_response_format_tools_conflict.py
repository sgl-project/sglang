"""
Unit tests for the validation guard that rejects the combination of
response_format (json_schema / json_object) with active tool calling.

When both are present, the grammar constraint applied by response_format
prevents the model from emitting tool-call tokens, so tool_calls is
silently always null.  SGLang now returns a 400-level error instead.

These tests exercise _validate_request() directly without a live server.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageUserParam,
    Function,
    JsonSchemaResponseFormat,
    ResponseFormat,
    Tool,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_MSG = ChatCompletionMessageUserParam(role="user", content="hi")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="multiply",
                description="Multiply two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            ),
        ),
    ]


def _json_schema_format():
    return ResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaResponseFormat(
            name="math_tool_answer",
            **{
                "schema": {
                    "type": "object",
                    "properties": {
                        "tools": {"type": "array", "items": {"type": "string"}},
                        "expression": {"type": "string"},
                        "result": {"type": "number"},
                    },
                    "required": ["tools", "expression", "result"],
                }
            },
        ),
    )


def _json_object_format():
    return ResponseFormat(type="json_object")


def _make_request(**kwargs):
    """Build a minimal ChatCompletionRequest with sensible defaults."""
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    defaults = dict(model="test-model", messages=[_USER_MSG])
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _make_serving():
    """Return an OpenAIServingChat instance whose external deps are mocked."""
    serving = MagicMock(spec=OpenAIServingChat)
    serving._validate_request = OpenAIServingChat._validate_request.__get__(
        serving, OpenAIServingChat
    )
    # _validate_request uses server_args.context_length and allow_auto_truncate
    serving.tokenizer_manager = MagicMock()
    serving.tokenizer_manager.server_args.context_length = None
    serving.tokenizer_manager.server_args.allow_auto_truncate = False
    return serving


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResponseFormatToolsConflict(unittest.TestCase):
    """Validate that response_format + active tool_choice is rejected."""

    def setUp(self):
        self.serving = _make_serving()

    # -- Conflict cases (should return an error string) ----------------------

    def test_json_schema_with_tool_choice_auto_returns_error(self):
        """The reproducer from the bug report: json_schema + tools + auto."""
        req = _make_request(
            tools=_make_tools(),
            tool_choice="auto",
            response_format=_json_schema_format(),
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err, "Expected a validation error but got None")
        self.assertIn("response_format", err)
        self.assertIn("tool", err.lower())

    def test_json_schema_with_tool_choice_required_returns_error(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="required",
            response_format=_json_schema_format(),
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)
        self.assertIn("response_format", err)

    def test_json_schema_with_named_tool_choice_returns_error(self):
        from sglang.srt.entrypoints.openai.protocol import (
            ToolChoice,
            ToolChoiceFuncName,
        )

        req = _make_request(
            tools=_make_tools(),
            tool_choice=ToolChoice(function=ToolChoiceFuncName(name="add")),
            response_format=_json_schema_format(),
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)

    def test_json_object_with_tool_choice_auto_returns_error(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="auto",
            response_format=_json_object_format(),
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)
        self.assertIn("response_format", err)

    def test_json_object_with_tool_choice_required_returns_error(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="required",
            response_format=_json_object_format(),
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)

    # -- Allowed cases (should return None) ----------------------------------

    def test_json_schema_with_tool_choice_none_is_allowed(self):
        """tool_choice=none means the model will not call tools → no conflict."""
        req = _make_request(
            tools=_make_tools(),
            tool_choice="none",
            response_format=_json_schema_format(),
        )
        err = self.serving._validate_request(req)
        self.assertIsNone(err, f"Expected no error but got: {err}")

    def test_json_object_with_tool_choice_none_is_allowed(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="none",
            response_format=_json_object_format(),
        )
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_no_response_format_with_tools_and_auto_is_allowed(self):
        """Plain tool calling without response_format is the normal happy path."""
        req = _make_request(tools=_make_tools(), tool_choice="auto")
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_json_schema_without_tools_is_allowed(self):
        """response_format alone (no tools) is perfectly valid."""
        req = _make_request(response_format=_json_schema_format())
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_json_object_without_tools_is_allowed(self):
        req = _make_request(response_format=_json_object_format())
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_text_response_format_with_tools_is_allowed(self):
        """type='text' sets no grammar constraint, so tool calling is fine."""
        req = _make_request(
            tools=_make_tools(),
            tool_choice="auto",
            response_format=ResponseFormat(type="text"),
        )
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_json_schema_missing_schema_field_is_caught_first(self):
        """The existing schema_ validation fires before our new check."""
        bad_format = ResponseFormat(type="json_schema", json_schema=None)
        req = _make_request(
            tools=_make_tools(),
            tool_choice="auto",
            response_format=bad_format,
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)
        # Could be either error; important is that we don't crash
        self.assertIsInstance(err, str)


if __name__ == "__main__":
    unittest.main()
