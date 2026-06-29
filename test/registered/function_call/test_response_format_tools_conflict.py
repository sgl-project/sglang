"""
Unit tests for the validation guard that rejects any output-constraint
parameter combined with active tool calling.

Any grammar constraint (response_format with json_schema / json_object /
structural_tag, or the SGLang-specific 'regex' / 'ebnf' fields) forces the
model to produce a specific shape and prevents it from emitting tool-call
tokens.  tool_calls is then silently null.  SGLang now returns a 400-level
error for all such combinations.

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
    serving.tokenizer_manager = MagicMock()
    serving.tokenizer_manager.server_args.context_length = None
    serving.tokenizer_manager.server_args.allow_auto_truncate = False
    return serving


# ---------------------------------------------------------------------------
# Tests — response_format conflicts
# ---------------------------------------------------------------------------


class TestResponseFormatToolsConflict(unittest.TestCase):
    """Validate that response_format + active tool_choice is rejected."""

    def setUp(self):
        self.serving = _make_serving()

    # -- json_schema conflict cases ------------------------------------------

    def test_json_schema_with_tool_choice_auto_returns_error(self):
        """The reproducer from the original bug report."""
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

    # -- json_object conflict cases ------------------------------------------

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

    # -- structural_tag conflict cases ---------------------------------------

    def test_structural_tag_with_tool_choice_auto_returns_error(self):
        """structural_tag response_format with active tool calling is rejected."""
        req = _make_request(
            tools=_make_tools(),
            tool_choice="auto",
            response_format=ResponseFormat(type="structural_tag"),
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)
        self.assertIn("structural_tag", err)

    def test_structural_tag_with_tool_choice_required_returns_error(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="required",
            response_format=ResponseFormat(type="structural_tag"),
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)

    # -- Allowed cases (should return None) ----------------------------------

    def test_json_schema_with_tool_choice_none_is_allowed(self):
        """tool_choice=none means the model will not call tools — no conflict."""
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

    def test_structural_tag_with_tool_choice_none_is_allowed(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="none",
            response_format=ResponseFormat(type="structural_tag"),
        )
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_no_response_format_with_tools_and_auto_is_allowed(self):
        """Plain tool calling without any constraint is the normal happy path."""
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
        """The existing schema_ validation fires before the conflict check."""
        bad_format = ResponseFormat(type="json_schema", json_schema=None)
        req = _make_request(
            tools=_make_tools(),
            tool_choice="auto",
            response_format=bad_format,
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)
        self.assertIsInstance(err, str)


# ---------------------------------------------------------------------------
# Tests — regex / ebnf conflicts
# ---------------------------------------------------------------------------


class TestRegexEbnfToolsConflict(unittest.TestCase):
    """Validate that regex/ebnf + active tool_choice is rejected."""

    def setUp(self):
        self.serving = _make_serving()

    # -- Conflict cases ------------------------------------------------------

    def test_regex_with_tool_choice_auto_returns_error(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="auto",
            regex=r"\d+",
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)
        self.assertIn("regex", err)

    def test_regex_with_tool_choice_required_returns_error(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="required",
            regex=r"\d+",
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)

    def test_ebnf_with_tool_choice_auto_returns_error(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="auto",
            ebnf='root ::= "yes" | "no"',
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)
        self.assertIn("ebnf", err)

    def test_ebnf_with_tool_choice_required_returns_error(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="required",
            ebnf='root ::= "yes" | "no"',
        )
        err = self.serving._validate_request(req)
        self.assertIsNotNone(err)

    # -- Allowed cases -------------------------------------------------------

    def test_regex_with_tool_choice_none_is_allowed(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="none",
            regex=r"\d+",
        )
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_ebnf_with_tool_choice_none_is_allowed(self):
        req = _make_request(
            tools=_make_tools(),
            tool_choice="none",
            ebnf='root ::= "yes" | "no"',
        )
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_regex_without_tools_is_allowed(self):
        req = _make_request(regex=r"\d+")
        err = self.serving._validate_request(req)
        self.assertIsNone(err)

    def test_ebnf_without_tools_is_allowed(self):
        req = _make_request(ebnf='root ::= "yes" | "no"')
        err = self.serving._validate_request(req)
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
