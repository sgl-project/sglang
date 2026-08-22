"""Unit tests for encoding_dsv32.py — no server, no model loading.

Tests cover encode_arguments_to_dsml, decode_dsml_to_arguments, render_tools,
find_last_user_index, render_message, drop_thinking_messages, encode_messages,
and _read_until_stop.
"""

import json
import re

from sglang.srt.entrypoints.openai.encoding_dsv32 import (
    DS32EncodingError,
    _read_until_stop,
    bos_token,
    decode_dsml_to_arguments,
    drop_thinking_messages,
    dsml_token,
    encode_arguments_to_dsml,
    encode_messages,
    eos_token,
    find_last_user_index,
    render_message,
    render_tools,
    thinking_end_token,
    thinking_start_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, parameters: dict = None) -> dict:
    """Return an OpenAI-format tool dict (matches what render_message receives)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool: {name}",
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


def _make_tool_call(name: str, arguments: str) -> dict:
    """Return an OpenAI-format tool_call dict."""
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse_dsml_args(dsml_str: str) -> dict:
    """Parse a DSML-encoded parameter string into a tool_args dict.

    Returns a mapping of parameter name -> (value, is_string) tuples,
    matching the format expected by decode_dsml_to_arguments.
    """
    pattern = (
        rf"<{dsml_token}parameter"
        rf' name="(.*?)" string="(true|false)">(.*?)</{dsml_token}parameter>'
    )
    matches = re.findall(pattern, dsml_str, flags=re.DOTALL)
    return {m[0]: (m[2], m[1]) for m in matches}


# ---------------------------------------------------------------------------
# TestEncodeArgumentsToDsml
# ---------------------------------------------------------------------------


class TestEncodeArgumentsToDsml(CustomTestCase):

    def test_string_parameter(self):
        """Scenario: Tool call with one string argument.
        Purpose: string values must carry string="true" and be emitted as-is."""
        tool_call = {"arguments": json.dumps({"city": "Beijing"})}
        result = encode_arguments_to_dsml(tool_call)
        self.assertIn(f"<{dsml_token}parameter", result)
        self.assertIn('name="city"', result)
        self.assertIn('string="true"', result)
        self.assertIn(">Beijing<", result)

    def test_int_parameter(self):
        """Scenario: Tool call with one integer argument.
        Purpose: non-string values must carry string="false"."""
        tool_call = {"arguments": json.dumps({"count": 42})}
        result = encode_arguments_to_dsml(tool_call)
        self.assertIn('name="count"', result)
        self.assertIn('string="false"', result)
        self.assertIn(">42<", result)

    def test_bool_parameter(self):
        """Scenario: Tool call with one boolean argument.
        Purpose: booleans are non-string so string="false"; JSON true/false used."""
        tool_call = {"arguments": json.dumps({"verbose": True})}
        result = encode_arguments_to_dsml(tool_call)
        self.assertIn('string="false"', result)
        # JSON representation of True is 'true'
        self.assertIn(">true<", result)

    def test_list_parameter(self):
        """Scenario: Tool call with a list argument.
        Purpose: lists are encoded as JSON strings with string="false"."""
        tool_call = {"arguments": json.dumps({"items": [1, 2, 3]})}
        result = encode_arguments_to_dsml(tool_call)
        self.assertIn('string="false"', result)
        self.assertIn("[1, 2, 3]", result)

    def test_dict_parameter(self):
        """Scenario: Tool call with a nested object argument.
        Purpose: objects are JSON-encoded and string="false"."""
        tool_call = {"arguments": json.dumps({"config": {"a": 1}})}
        result = encode_arguments_to_dsml(tool_call)
        self.assertIn('string="false"', result)
        self.assertIn('name="config"', result)

    def test_multiple_parameters(self):
        """Scenario: Tool call with multiple heterogeneous arguments.
        Purpose: each argument gets its own DSML parameter tag; order is preserved."""
        tool_call = {"arguments": json.dumps({"q": "hello", "n": 5})}
        result = encode_arguments_to_dsml(tool_call)
        lines = result.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn('name="q"', lines[0])
        self.assertIn('name="n"', lines[1])

    def test_empty_arguments(self):
        """Scenario: Tool call with no arguments ({}).
        Purpose: empty dict produces empty string, not a trailing newline."""
        tool_call = {"arguments": "{}"}
        result = encode_arguments_to_dsml(tool_call)
        self.assertEqual(result, "")

    def test_float_parameter(self):
        """Scenario: Tool call with a float argument.
        Purpose: floats are non-string, emitted as JSON number."""
        tool_call = {"arguments": json.dumps({"temperature": 0.7})}
        result = encode_arguments_to_dsml(tool_call)
        self.assertIn('string="false"', result)
        self.assertIn("0.7", result)

    def test_null_parameter(self):
        """Scenario: Tool call with a null/None argument.
        Purpose: None is serialised to JSON 'null' with string="false"."""
        tool_call = {"arguments": json.dumps({"value": None})}
        result = encode_arguments_to_dsml(tool_call)
        self.assertIn('string="false"', result)
        self.assertIn(">null<", result)


# ---------------------------------------------------------------------------
# TestDecodeDsmlToArguments
# ---------------------------------------------------------------------------


class TestDecodeDsmlToArguments(CustomTestCase):

    def test_string_decode(self):
        """Scenario: Decode one string argument.
        Purpose: string="true" wraps the value in JSON quotes."""
        result = decode_dsml_to_arguments("my_func", {"city": ("Beijing", "true")})
        self.assertEqual(result["name"], "my_func")
        parsed = json.loads(result["arguments"])
        self.assertEqual(parsed["city"], "Beijing")

    def test_number_decode(self):
        """Scenario: Decode one numeric argument.
        Purpose: string="false" leaves the value as raw JSON (integer)."""
        result = decode_dsml_to_arguments("my_func", {"count": ("42", "false")})
        parsed = json.loads(result["arguments"])
        self.assertEqual(parsed["count"], 42)

    def test_bool_decode(self):
        """Scenario: Decode one boolean argument.
        Purpose: string="false" allows 'true'/'false' to parse as Python booleans."""
        result = decode_dsml_to_arguments("my_func", {"flag": ("true", "false")})
        parsed = json.loads(result["arguments"])
        self.assertEqual(parsed["flag"], True)

    def test_roundtrip_string(self):
        """Scenario: Encode then decode a string argument.
        Purpose: roundtrip must be identity for string values."""
        original_args = json.dumps({"name": "Alice"})
        tool_call = {"arguments": original_args}
        dsml_str = encode_arguments_to_dsml(tool_call)
        tool_args = _parse_dsml_args(dsml_str)
        decoded = decode_dsml_to_arguments("f", tool_args)
        self.assertEqual(json.loads(decoded["arguments"])["name"], "Alice")

    def test_roundtrip_int(self):
        """Scenario: Encode then decode an integer argument.
        Purpose: roundtrip must be identity for integer values."""
        original_args = json.dumps({"n": 7})
        tool_call = {"arguments": original_args}
        dsml_str = encode_arguments_to_dsml(tool_call)
        tool_args = _parse_dsml_args(dsml_str)
        decoded = decode_dsml_to_arguments("f", tool_args)
        self.assertEqual(json.loads(decoded["arguments"])["n"], 7)

    def test_empty_args(self):
        """Scenario: Decode with no arguments.
        Purpose: empty dict produces valid JSON object {}."""
        result = decode_dsml_to_arguments("func", {})
        self.assertEqual(result["name"], "func")
        parsed = json.loads(result["arguments"])
        self.assertEqual(parsed, {})

    def test_multiple_args(self):
        """Scenario: Decode multiple arguments of mixed types.
        Purpose: all keys are present and correctly typed in the output."""
        tool_args = {
            "city": ("Paris", "true"),
            "days": ("3", "false"),
        }
        result = decode_dsml_to_arguments("weather", tool_args)
        parsed = json.loads(result["arguments"])
        self.assertEqual(parsed["city"], "Paris")
        self.assertEqual(parsed["days"], 3)


# ---------------------------------------------------------------------------
# TestRenderTools
# ---------------------------------------------------------------------------


class TestRenderTools(CustomTestCase):

    def test_single_tool_rendered(self):
        """Scenario: Render a single tool schema.
        Purpose: output must contain the dsml_token and function name."""
        tools = [_make_tool("search")]
        result = render_tools(tools)
        self.assertIn(dsml_token, result)
        self.assertIn("search", result)
        self.assertIn("<functions>", result)
        self.assertIn("</functions>", result)

    def test_multiple_tools_rendered(self):
        """Scenario: Render two tool schemas.
        Purpose: both tool names must appear in the output."""
        tools = [_make_tool("search"), _make_tool("calculator")]
        result = render_tools(tools)
        self.assertIn("search", result)
        self.assertIn("calculator", result)

    def test_thinking_tokens_in_template(self):
        """Scenario: Render tools in thinking mode context.
        Purpose: thinking_start_token and thinking_end_token must appear in instructions.
        """
        tools = [_make_tool("foo")]
        result = render_tools(tools)
        self.assertIn(thinking_start_token, result)
        self.assertIn(thinking_end_token, result)

    def test_empty_tools_list(self):
        """Scenario: Render an empty tools list.
        Purpose: template still renders without error; tool_schemas section is empty."""
        result = render_tools([])
        self.assertIn("<functions>", result)
        self.assertIn("</functions>", result)


# ---------------------------------------------------------------------------
# TestFindLastUserIndex
# ---------------------------------------------------------------------------


class TestFindLastUserIndex(CustomTestCase):

    def test_single_user_message(self):
        """Scenario: One user message at index 0.
        Purpose: returns 0."""
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(find_last_user_index(messages), 0)

    def test_user_at_end(self):
        """Scenario: Multiple messages; user is last.
        Purpose: returns the last index."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "follow-up"},
        ]
        self.assertEqual(find_last_user_index(messages), 2)

    def test_user_in_middle(self):
        """Scenario: User message is not the last message.
        Purpose: returns the index of the last user, not the last message."""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        self.assertEqual(find_last_user_index(messages), 0)

    def test_no_user(self):
        """Scenario: No user or developer messages.
        Purpose: returns -1."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "a"},
        ]
        self.assertEqual(find_last_user_index(messages), -1)

    def test_developer_counts_as_user(self):
        """Scenario: Developer role message present.
        Purpose: 'developer' is treated the same as 'user'."""
        messages = [
            {"role": "developer", "content": "dev instruction"},
        ]
        self.assertEqual(find_last_user_index(messages), 0)

    def test_developer_after_user(self):
        """Scenario: Developer message after a user message.
        Purpose: developer message is the last 'user-like' message."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "developer", "content": "dev"},
        ]
        self.assertEqual(find_last_user_index(messages), 1)

    def test_empty_messages(self):
        """Scenario: Empty message list.
        Purpose: returns -1, no crash."""
        self.assertEqual(find_last_user_index([]), -1)


# ---------------------------------------------------------------------------
# TestRenderMessage
# ---------------------------------------------------------------------------


class TestRenderMessage(CustomTestCase):

    # --- system role ---

    def test_system_role_basic(self):
        """Scenario: System message with plain content.
        Purpose: content appears verbatim; no extra wrapping tokens."""
        messages = [{"role": "system", "content": "You are helpful."}]
        result = render_message(0, messages, thinking_mode="chat")
        self.assertEqual(result, "You are helpful.")

    def test_system_role_empty_content(self):
        """Scenario: System message with no content.
        Purpose: renders as empty string without error."""
        messages = [{"role": "system", "content": None}]
        result = render_message(0, messages, thinking_mode="chat")
        self.assertEqual(result, "")

    def test_system_role_with_tools(self):
        """Scenario: System message carrying tools list.
        Purpose: tools section is appended after the system content."""
        tools = [_make_tool("search")]
        messages = [{"role": "system", "content": "System.", "tools": tools}]
        result = render_message(0, messages, thinking_mode="chat")
        self.assertIn("System.", result)
        self.assertIn(dsml_token, result)
        self.assertIn("search", result)

    # --- user role ---

    def test_user_role_basic_chat_mode(self):
        """Scenario: Basic user message in chat (non-thinking) mode.
        Purpose: content wrapped in user template; thinking_end_token appended."""
        messages = [{"role": "user", "content": "Hello"}]
        result = render_message(0, messages, thinking_mode="chat")
        self.assertIn("Hello", result)
        self.assertIn("<｜User｜>", result)
        self.assertIn("<｜Assistant｜>", result)
        self.assertIn(thinking_end_token, result)
        self.assertNotIn(thinking_start_token, result)

    def test_user_role_last_in_thinking_mode(self):
        """Scenario: Last user message in thinking mode.
        Purpose: thinking_start_token appended instead of thinking_end_token."""
        messages = [{"role": "user", "content": "Hello"}]
        result = render_message(0, messages, thinking_mode="thinking")
        self.assertIn(thinking_start_token, result)
        self.assertNotIn(thinking_end_token, result)

    def test_user_role_not_last_in_thinking_mode(self):
        """Scenario: Non-last user message in thinking mode.
        Purpose: thinking_end_token appended (not thinking_start_token)."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply", "reasoning_content": "reason"},
            {"role": "user", "content": "Second"},
        ]
        # index=0 is user but not the last user (index=2 is last)
        result = render_message(0, messages, thinking_mode="thinking")
        self.assertIn(thinking_end_token, result)
        self.assertNotIn(thinking_start_token, result)

    # --- assistant role ---

    def test_assistant_role_no_tools_chat(self):
        """Scenario: Plain assistant message in chat mode.
        Purpose: content and eos_token appear; no DSML function_calls block."""
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "Answer"},
        ]
        result = render_message(1, messages, thinking_mode="chat")
        self.assertIn("Answer", result)
        self.assertIn(eos_token, result)
        self.assertNotIn("function_calls", result)

    def test_assistant_role_with_tool_calls(self):
        """Scenario: Assistant message containing tool_calls.
        Purpose: DSML invoke block rendered; function name and dsml_token present."""
        tool_calls = [_make_tool_call("search", json.dumps({"query": "test"}))]
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]
        result = render_message(1, messages, thinking_mode="chat")
        self.assertIn(f"<{dsml_token}invoke", result)
        self.assertIn("search", result)
        self.assertIn(f"<{dsml_token}function_calls>", result)

    def test_assistant_thinking_mode_requires_reasoning(self):
        """Scenario: assistant after last user in thinking mode with no reasoning_content and no tool_calls.
        Purpose: DS32EncodingError raised — at least one must be present."""
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "no reasoning"},
        ]
        with self.assertRaises(DS32EncodingError):
            render_message(1, messages, thinking_mode="thinking")

    def test_assistant_thinking_mode_with_reasoning(self):
        """Scenario: Assistant after last user in thinking mode with reasoning_content.
        Purpose: thinking_end_token wraps reasoning; no error raised."""
        messages = [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Step by step...",
            },
        ]
        result = render_message(1, messages, thinking_mode="thinking")
        self.assertIn("Step by step...", result)
        self.assertIn(thinking_end_token, result)
        self.assertIn("Final answer", result)

    def test_assistant_thinking_mode_with_tool_calls_no_reasoning(self):
        """Scenario: Assistant has tool_calls but no reasoning_content in thinking mode.
        Purpose: tool_calls alone suffice — no DS32EncodingError."""
        tool_calls = [_make_tool_call("search", json.dumps({"q": "x"}))]
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]
        # Should NOT raise
        result = render_message(1, messages, thinking_mode="thinking")
        self.assertIn(f"<{dsml_token}function_calls>", result)

    # --- tool role ---

    def test_tool_role_first_output(self):
        """Scenario: First tool result following an assistant with one tool call.
        Purpose: <function_results> block opened and immediately closed for a single call.
        """
        tool_calls = [_make_tool_call("search", json.dumps({"q": "x"}))]
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "tool", "content": "result data"},
        ]
        result = render_message(2, messages, thinking_mode="chat")
        self.assertIn("<function_results>", result)
        self.assertIn("result data", result)
        self.assertIn("</function_results>", result)

    def test_tool_role_first_of_two(self):
        """Scenario: First of two tool results.
        Purpose: <function_results> opened but NOT closed at first result."""
        tool_calls = [
            _make_tool_call("search", json.dumps({"q": "x"})),
            _make_tool_call("calc", json.dumps({"expr": "1+1"})),
        ]
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "tool", "content": "result1"},
            {"role": "tool", "content": "result2"},
        ]
        result_first = render_message(2, messages, thinking_mode="chat")
        self.assertIn("<function_results>", result_first)
        self.assertNotIn("</function_results>", result_first)

    def test_tool_role_last_of_two(self):
        """Scenario: Last of two tool results.
        Purpose: </function_results> closed; thinking token appended."""
        tool_calls = [
            _make_tool_call("search", json.dumps({"q": "x"})),
            _make_tool_call("calc", json.dumps({"expr": "1+1"})),
        ]
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "tool", "content": "result1"},
            {"role": "tool", "content": "result2"},
        ]
        result_last = render_message(3, messages, thinking_mode="chat")
        self.assertNotIn("<function_results>", result_last)
        self.assertIn("</function_results>", result_last)
        self.assertIn(thinking_end_token, result_last)

    # --- developer role ---

    def test_developer_role_basic(self):
        """Scenario: Developer role message in chat mode.
        Purpose: content included; user template used; thinking_end_token appended."""
        messages = [{"role": "developer", "content": "Dev instruction"}]
        result = render_message(0, messages, thinking_mode="chat")
        self.assertIn("Dev instruction", result)
        self.assertIn("<｜User｜>", result)
        self.assertIn(thinking_end_token, result)

    def test_developer_role_empty_content_raises(self):
        """Scenario: Developer message with no content.
        Purpose: DS32EncodingError raised — developer messages must have content."""
        messages = [{"role": "developer", "content": ""}]
        with self.assertRaises(DS32EncodingError):
            render_message(0, messages, thinking_mode="chat")

    # --- error cases ---

    def test_invalid_index_negative(self):
        """Scenario: Negative index provided.
        Purpose: DS32EncodingError raised."""
        messages = [{"role": "user", "content": "hi"}]
        with self.assertRaises(DS32EncodingError):
            render_message(-1, messages, thinking_mode="chat")

    def test_invalid_index_out_of_bounds(self):
        """Scenario: Index beyond message list length.
        Purpose: DS32EncodingError raised."""
        messages = [{"role": "user", "content": "hi"}]
        with self.assertRaises(DS32EncodingError):
            render_message(1, messages, thinking_mode="chat")

    def test_invalid_thinking_mode(self):
        """Scenario: Unrecognised thinking_mode value.
        Purpose: DS32EncodingError raised for any value other than 'chat'/'thinking'."""
        messages = [{"role": "user", "content": "hi"}]
        with self.assertRaises(DS32EncodingError):
            render_message(0, messages, thinking_mode="invalid")

    def test_unknown_role_raises(self):
        """Scenario: Message with an unknown role string.
        Purpose: NotImplementedError raised."""
        messages = [{"role": "robot", "content": "beep"}]
        with self.assertRaises(NotImplementedError):
            render_message(0, messages, thinking_mode="chat")


# ---------------------------------------------------------------------------
# TestDropThinkingMessages
# ---------------------------------------------------------------------------


class TestDropThinkingMessages(CustomTestCase):

    def test_removes_reasoning_content_before_last_user(self):
        """Scenario: Assistant message with reasoning_content comes before last user.
        Purpose: reasoning_content stripped from that assistant message."""
        messages = [
            {"role": "user", "content": "First"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "internal reasoning",
            },
            {"role": "user", "content": "Second"},
        ]
        result = drop_thinking_messages(messages)
        # The assistant at index 1 is before last user index 2
        assistant_msg = result[1]
        self.assertNotIn("reasoning_content", assistant_msg)
        self.assertEqual(assistant_msg["content"], "answer")

    def test_preserves_reasoning_after_last_user(self):
        """Scenario: Assistant message with reasoning_content comes AFTER last user.
        Purpose: messages at or after last_user_idx are left untouched."""
        messages = [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "A",
                "reasoning_content": "kept",
            },
        ]
        result = drop_thinking_messages(messages)
        # last_user_idx = 0, assistant is at idx 1 which is >= last_user_idx
        self.assertIn("reasoning_content", result[1])
        self.assertEqual(result[1]["reasoning_content"], "kept")

    def test_preserves_user_messages_intact(self):
        """Scenario: User messages before and after last user.
        Purpose: user messages are copied through unmodified."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": "r"},
            {"role": "user", "content": "bye"},
        ]
        result = drop_thinking_messages(messages)
        self.assertEqual(result[0]["content"], "hi")
        self.assertEqual(result[2]["content"], "bye")

    def test_preserves_system_messages(self):
        """Scenario: System message before last user.
        Purpose: system messages are not modified."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Q"},
        ]
        result = drop_thinking_messages(messages)
        self.assertEqual(result[0]["content"], "sys")

    def test_preserves_tool_messages(self):
        """Scenario: Tool message before last user.
        Purpose: tool messages are passed through without modification."""
        tool_calls = [_make_tool_call("f", "{}")]
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "tool", "content": "r"},
            {"role": "user", "content": "Q2"},
        ]
        result = drop_thinking_messages(messages)
        # tool message at index 2 is before last_user_idx=3 but role=="tool"
        self.assertEqual(result[2]["content"], "r")

    def test_explicit_last_user_idx(self):
        """Scenario: Caller provides explicit last_user_idx.
        Purpose: function uses the provided value instead of computing it."""
        messages = [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "A",
                "reasoning_content": "should stay",
            },
        ]
        # Pretend last_user_idx is 0, so assistant at 1 is >= 0; reasoning kept
        result = drop_thinking_messages(messages, last_user_idx=0)
        self.assertIn("reasoning_content", result[1])

    def test_no_user_message(self):
        """Scenario: No user messages in the list.
        Purpose: last_user_idx=-1; all messages treated as 'at or after' last user;
        assistant reasoning preserved."""
        messages = [
            {
                "role": "assistant",
                "content": "hello",
                "reasoning_content": "preserved",
            },
        ]
        result = drop_thinking_messages(messages)
        self.assertIn("reasoning_content", result[0])

    def test_no_mutation_of_original(self):
        """Scenario: Run drop_thinking_messages and verify original is not mutated.
        Purpose: function must perform a copy of modified messages."""
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "r"},
            {"role": "user", "content": "Q2"},
        ]
        drop_thinking_messages(messages)
        # The original assistant message should still have reasoning_content
        self.assertIn("reasoning_content", messages[1])

    def test_boundary_at_last_user_index_kept(self):
        """Scenario: Message AT last_user_idx (the user message itself).
        Purpose: The user message at last_user_idx is kept intact (idx >= last_user_idx).
        """
        messages = [
            {"role": "user", "content": "Q"},
        ]
        result = drop_thinking_messages(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")


# ---------------------------------------------------------------------------
# TestEncodeMessages
# ---------------------------------------------------------------------------


class TestEncodeMessages(CustomTestCase):

    def test_simple_user_message_includes_bos(self):
        """Scenario: Single user message, no context.
        Purpose: output starts with bos_token and contains the user message."""
        messages = [{"role": "user", "content": "Hello"}]
        result = encode_messages(messages, thinking_mode="chat")
        self.assertTrue(result.startswith(bos_token))
        self.assertIn("Hello", result)

    def test_no_bos_when_context_provided(self):
        """Scenario: Caller provides non-empty context.
        Purpose: bos_token NOT prepended when context is given."""
        context = [{"role": "system", "content": "Sys"}]
        messages = [{"role": "user", "content": "Hello"}]
        result = encode_messages(messages, thinking_mode="chat", context=context)
        self.assertFalse(result.startswith(bos_token))

    def test_no_bos_when_add_default_bos_token_false(self):
        """Scenario: add_default_bos_token=False with no context.
        Purpose: bos_token must not appear at the start."""
        messages = [{"role": "user", "content": "Hello"}]
        result = encode_messages(
            messages, thinking_mode="chat", add_default_bos_token=False
        )
        self.assertFalse(result.startswith(bos_token))

    def test_thinking_mode_drop_strips_reasoning(self):
        """Scenario: Conversation with historical reasoning; drop_thinking=True.
        Purpose: reasoning_content stripped from history before encoding."""
        messages = [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "content": "A1",
                "reasoning_content": "secret",
            },
            {"role": "user", "content": "Q2"},
        ]
        result = encode_messages(messages, thinking_mode="thinking", drop_thinking=True)
        # reasoning_content should not appear in the encoded string
        self.assertNotIn("secret", result)

    def test_thinking_mode_no_drop_preserves_reasoning(self):
        """Scenario: Conversation with reasoning after last user; drop_thinking=False.
        Purpose: reasoning_content in an assistant message after the last user is
        included in the encoded output when drop_thinking=False."""
        messages = [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "A",
                "reasoning_content": "visible_reasoning",
            },
        ]
        # assistant is at index 1, last_user_idx=0, so index > last_user_idx
        result = encode_messages(
            messages, thinking_mode="thinking", drop_thinking=False
        )
        self.assertIn("visible_reasoning", result)

    def test_context_shifts_render_index(self):
        """Scenario: context + messages together form the full conversation.
        Purpose: only messages (not context) are rendered; combined correctly."""
        context = [{"role": "system", "content": "System prompt"}]
        messages = [{"role": "user", "content": "User question"}]
        result = encode_messages(
            messages, thinking_mode="chat", context=context, add_default_bos_token=True
        )
        # bos_token NOT added when context is non-empty
        self.assertFalse(result.startswith(bos_token))
        self.assertIn("User question", result)

    def test_empty_messages_list(self):
        """Scenario: Empty messages list with no context.
        Purpose: returns only bos_token (no crash)."""
        result = encode_messages([], thinking_mode="chat")
        self.assertEqual(result, bos_token)


# ---------------------------------------------------------------------------
# TestReadUntilStop
# ---------------------------------------------------------------------------


class TestReadUntilStop(CustomTestCase):

    def test_stop_found_at_start(self):
        """Scenario: Stop token is at the very beginning.
        Purpose: returns empty content and advances index past the stop token."""
        text = "STOP_here rest"
        new_idx, content, matched = _read_until_stop(0, text, ["STOP_"])
        self.assertEqual(content, "")
        self.assertEqual(matched, "STOP_")
        self.assertEqual(new_idx, len("STOP_"))

    def test_stop_found_in_middle(self):
        """Scenario: Stop token in the middle of the text.
        Purpose: content up to stop token returned; index past stop token."""
        text = "hello STOP world"
        new_idx, content, matched = _read_until_stop(0, text, ["STOP"])
        self.assertEqual(content, "hello ")
        self.assertEqual(matched, "STOP")
        self.assertEqual(new_idx, text.index("STOP") + len("STOP"))

    def test_no_stop_found(self):
        """Scenario: None of the stop tokens is present.
        Purpose: returns entire remaining text; matched_stop is None."""
        text = "no stop here"
        new_idx, content, matched = _read_until_stop(0, text, ["MISSING"])
        self.assertEqual(content, "no stop here")
        self.assertIsNone(matched)
        self.assertEqual(new_idx, len(text))

    def test_multiple_stops_picks_first(self):
        """Scenario: Two stop tokens both present; one appears earlier.
        Purpose: the stop token with the smallest position wins."""
        text = "abc FIRST later SECOND end"
        new_idx, content, matched = _read_until_stop(0, text, ["SECOND", "FIRST"])
        self.assertEqual(matched, "FIRST")
        self.assertEqual(content, "abc ")

    def test_start_index_respected(self):
        """Scenario: Non-zero start index pointing directly before stop token.
        Purpose: search begins from start index; content between start and stop is empty.
        """
        text = "SKIP_THIS_PARTSTOP content"
        # "SKIP_THIS_PART" is 14 chars; index=14 puts us right at "STOP"
        new_idx, content, matched = _read_until_stop(14, text, ["STOP"])
        self.assertEqual(content, "")
        self.assertEqual(matched, "STOP")

    def test_stop_at_end_of_text(self):
        """Scenario: Stop token is at the very end of the string.
        Purpose: content is everything before the stop; index advances to len(text)."""
        text = "hello world END"
        new_idx, content, matched = _read_until_stop(0, text, ["END"])
        self.assertEqual(content, "hello world ")
        self.assertEqual(matched, "END")
        self.assertEqual(new_idx, len(text))

    def test_empty_text(self):
        """Scenario: Empty text string.
        Purpose: returns empty content and None match without error."""
        new_idx, content, matched = _read_until_stop(0, "", ["STOP"])
        self.assertEqual(content, "")
        self.assertIsNone(matched)
        self.assertEqual(new_idx, 0)

    def test_empty_stop_list(self):
        """Scenario: Empty list of stop tokens.
        Purpose: returns full remaining text and None match."""
        text = "some text"
        new_idx, content, matched = _read_until_stop(0, text, [])
        self.assertEqual(content, "some text")
        self.assertIsNone(matched)

    def test_unicode_stop_token(self):
        """Scenario: Stop token contains Unicode characters (DeepSeek special tokens).
        Purpose: Unicode stop tokens must be found correctly."""
        text = f"prefix{eos_token}suffix"
        new_idx, content, matched = _read_until_stop(0, text, [eos_token])
        self.assertEqual(content, "prefix")
        self.assertEqual(matched, eos_token)


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
