"""Unit tests for Lfm2Detector — no server, no model loading."""

import json
import warnings

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestLfm2Detector(CustomTestCase):
    """Tests for LFM2 (Liquid Foundation Model 2) function call detector."""

    def setUp(self):
        """Set up test tools and detector."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="calculator",
                    description="Perform calculations",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression",
                            },
                        },
                        "required": ["expression"],
                    },
                ),
            ),
        ]
        self.detector = Lfm2Detector()

    # ==================== has_tool_call tests ====================

    def test_has_tool_call_false(self):
        """Test no false positives for regular text."""
        text = "The weather in Paris is nice today."
        self.assertFalse(self.detector.has_tool_call(text))

    def test_has_tool_call_partial_marker(self):
        """Test that partial markers are detected (start token present)."""
        text = '<|tool_call_start|>[get_weather(city="Paris")'
        self.assertTrue(self.detector.has_tool_call(text))

    # ==================== detect_and_parse tests (Pythonic format) ====================

    def test_detect_and_parse_pythonic_simple(self):
        """Test parsing a simple Pythonic format tool call."""
        text = '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Paris")

    def test_detect_and_parse_pythonic_invalid_escape(self):
        """An invalid Python escape (e.g. "\\d+") must not drop the tool call."""
        text = '<|tool_call_start|>[search(query="\\d+")]<|tool_call_end|>'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "\\d+")
        self.assertFalse(
            any(isinstance(w.message, SyntaxWarning) for w in caught),
            [str(w.message) for w in caught],
        )

    def test_detect_and_parse_pythonic_multiple_args(self):
        """Test parsing with multiple arguments."""
        text = '<|tool_call_start|>[get_weather(city="London", unit="celsius")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "London")
        self.assertEqual(params["unit"], "celsius")

    def test_detect_and_parse_pythonic_no_args(self):
        """Test parsing function with no arguments."""
        # Add a no-arg tool for this test
        tools_with_noarg = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]
        text = "<|tool_call_start|>[get_time()]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, tools_with_noarg)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_time")

    def test_detect_and_parse_pythonic_multiple_calls(self):
        """Test parsing multiple tool calls in one block."""
        text = '<|tool_call_start|>[get_weather(city="Paris"), search(query="restaurants")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

        params1 = json.loads(result.calls[0].parameters)
        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(params1["city"], "Paris")
        self.assertEqual(params2["query"], "restaurants")

    def test_detect_and_parse_with_normal_text_before(self):
        """Test parsing with normal text before the tool call."""
        text = 'Let me check the weather for you. <|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Let me check the weather for you.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_detect_and_parse_special_characters_in_value(self):
        """Test parsing with special characters in argument values."""
        text = (
            '<|tool_call_start|>[search(query="what\'s the weather?")]<|tool_call_end|>'
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertIn("weather", params["query"])

    # ==================== detect_and_parse tests (JSON format) ====================

    def test_detect_and_parse_json_simple(self):
        """Test parsing JSON format tool call."""
        text = '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Berlin"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Berlin")

    def test_detect_and_parse_json_multiple_calls(self):
        """Test parsing multiple JSON format tool calls."""
        text = '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "search", "arguments": {"query": "hotels"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_detect_and_parse_json_with_parameters_key(self):
        """Test parsing JSON format with 'parameters' key instead of 'arguments'."""
        text = '<|tool_call_start|>[{"name": "get_weather", "parameters": {"city": "Madrid"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Madrid")

    # ==================== Edge cases ====================

    def test_detect_and_parse_no_tool_call(self):
        """Test parsing text with no tool calls."""
        text = "This is just regular text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_unknown_function(self):
        """Test parsing with unknown function name - skipped by default (SGLANG_FORWARD_UNKNOWN_TOOLS=false)."""
        text = '<|tool_call_start|>[unknown_function(arg="value")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        # By default, unknown functions are skipped (consistent with other detectors)
        self.assertEqual(len(result.calls), 0)

    def test_detect_and_parse_empty_content(self):
        """Test parsing with empty content between markers."""
        text = "<|tool_call_start|><|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])

    def test_detect_and_parse_multiple_blocks(self):
        """Test parsing multiple separate tool call blocks."""
        text = '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|> Some text <|tool_call_start|>[search(query="food")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    # ==================== Streaming tests ====================
    # The LFM2 detector buffers until it sees complete <|tool_call_start|>...<|tool_call_end|>
    # blocks, then parses the complete block. This allows proper handling of both
    # JSON and Pythonic formats.

    def test_streaming_json_split_across_chunks(self):
        """Test streaming with JSON tool call split across multiple chunks - waits for complete block."""
        # Reset detector state
        self.detector = Lfm2Detector()

        # First chunk: start marker and partial JSON (no end token)
        chunk1 = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Should buffer and not emit calls yet (waiting for complete block)
        self.assertEqual(len(result1.calls), 0)
        self.assertEqual(result1.normal_text, "")

        # Second chunk: complete the JSON and end token
        chunk2 = '"Vienna"}}<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # Now should have the complete tool call
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")

    def test_streaming_json_normal_text_before_tool_call(self):
        """Test streaming with normal text before JSON tool call."""
        # Reset detector state
        self.detector = Lfm2Detector()

        chunk1 = "I'll check the weather. "
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Normal text should be returned
        self.assertIn("check the weather", result1.normal_text)

        chunk2 = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Amsterdam"}}<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(len(result2.calls), 1)

    def test_streaming_eot_token_filtering(self):
        """Test that end-of-turn token is filtered from normal text."""
        # Reset detector state
        self.detector = Lfm2Detector()

        # Send text that ends with tool call end token (JSON format)
        text = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Oslo"}}<|tool_call_end|>'
        result = self.detector.parse_streaming_increment(text, self.tools)

        # The normal_text should not contain the eot_token
        self.assertNotIn("<|tool_call_end|>", result.normal_text)

    # ==================== Pythonic streaming tests ====================

    def test_streaming_pythonic_split_across_chunks(self):
        """Test streaming with Pythonic tool call split across multiple chunks."""
        self.detector = Lfm2Detector()

        # First chunk: start marker and partial call
        chunk1 = '<|tool_call_start|>[get_weather(city="'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Should buffer and not emit calls yet
        self.assertEqual(len(result1.calls), 0)

        # Second chunk: complete the call
        chunk2 = 'Munich")]<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # Now should have the complete tool call
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result2.calls[0].parameters), {"city": "Munich"})

    def test_streaming_pythonic_multiple_calls(self):
        """Test streaming with multiple Pythonic tool calls."""
        self.detector = Lfm2Detector()

        text = '<|tool_call_start|>[get_weather(city="Paris"), search(query="hotels")]<|tool_call_end|>'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    # ==================== recovery tests (dropped-call regressions) ====================

    def test_multiline_string_argument_recovered(self):
        """A raw newline inside a string argument (multi-line shell command)
        is invalid Python, so ast.parse failed and the whole call was
        dropped. The value must round-trip with the newline intact."""
        text = (
            "<|tool_call_start|>[search(query='line one\nline two')]<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "line one\nline two")

    def test_nul_byte_in_string_argument_recovered(self):
        """A NUL byte anywhere makes ast.parse raise ValueError (not
        SyntaxError), so the call was dropped with no recovery path."""
        text = "<|tool_call_start|>[search(query='printf a\x00b')]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "printf a\x00b")

    def test_nested_quotes_recovered(self):
        """Unescaped same-style quotes nested in a shell command
        (sed -n '360,450p') read as string/number juxtaposition, a
        SyntaxError, so the call was dropped even though only one closing
        quote yields parseable text."""
        text = (
            "<|tool_call_start|>[search(query='sed -n '360,450p' f.py')]"
            "<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "sed -n '360,450p' f.py")

    def test_ambiguous_nested_quotes_not_guessed(self):
        """When a later string argument's closing quote is also a plausible
        closer, the nesting is genuinely ambiguous; recovery must NOT guess
        a reading (guards the recovery predicate degrading to greedy)."""
        text = (
            "<|tool_call_start|>[get_weather(city='echo 'hi', unit='celsius')]"
            "<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])

    def test_reserved_keyword_parameter_recovered(self):
        """A parameter named after a Python keyword (from=1) is a
        SyntaxError; the call was dropped. The original parameter name must
        be restored in the decoded arguments."""
        text = "<|tool_call_start|>[search(query='M.md', from=1)]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params, {"query": "M.md", "from": 1})

    def test_zero_padded_int_recovered(self):
        """Zero-padded ints (day=07) are a SyntaxError ("leading zeros in
        decimal integer literals"); the call was dropped."""
        text = "<|tool_call_start|>[get_weather(city='NYC', day=07)]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["day"], 7)

    def test_explicit_positive_number(self):
        """An explicitly signed positive number (+7) is UnaryOp(UAdd), which
        only had a USub branch, so the call was dropped."""
        text = "<|tool_call_start|>[search(query='x', limit=+7)]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["limit"], 7)

    def test_set_argument_decoded_as_list(self):
        """A set argument ({'a', 'b'}) raised in _get_parameter_value and
        dropped the call; JSON has no set type so it decodes as a list."""
        text = "<|tool_call_start|>[search(query={'a', 'b'})]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], ["a", "b"])

    def test_constant_fstring_argument(self):
        """A placeholder-free f-string (f'hello') parses as JoinedStr, not
        Constant, and dropped the call although it is a plain string."""
        text = "<|tool_call_start|>[search(query=f'hello')]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "hello")

    def test_bytes_argument_skips_only_that_call(self):
        """A bytes argument passed _get_parameter_value (it is an
        ast.Constant) and only failed later as TypeError inside json.dumps,
        which escaped the per-call handler and dropped every sibling call in
        the block."""
        text = (
            "<|tool_call_start|>[get_weather(city='SF'), search(query=b'z')]"
            "<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_non_finite_number_never_emits_invalid_json(self):
        """The literal 1e999 overflows to float inf, and json.dumps rendered
        it as Infinity — parameters that no JSON parser accepts. The call
        must be skipped instead; every emitted parameters string must be
        valid JSON."""
        text = "<|tool_call_start|>[search(query='x', limit=1e999)]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        for call in result.calls:
            json.loads(call.parameters)
        self.assertEqual(result.calls, [])

    def test_kwargs_unpack_merges_dict(self):
        """**-unpacked kwargs were skipped silently, emitting the call with
        arguments missing; a dict literal merges with later-binding-wins
        semantics instead, and non-dict operands reject the call."""
        text = "<|tool_call_start|>[search(**{'query': 'x'}, limit=2)]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params, {"query": "x", "limit": 2})

        bad = "<|tool_call_start|>[search(**[1, 2])]<|tool_call_end|>"
        self.assertEqual(self.detector.detect_and_parse(bad, self.tools).calls, [])

    def test_positional_argument_call_not_silently_corrupted(self):
        """get_weather('Paris', unit='celsius') used to silently drop
        'Paris' and emit a successful call with only {"unit": "celsius"} —
        a wrong execution instead of a visible failure. The call is
        rejected; a keyword-only sibling still comes through."""
        text = (
            "<|tool_call_start|>[search(query='x'), "
            "get_weather('Paris', unit='celsius')]<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

    def test_good_call_survives_unparsable_block(self):
        """A genuinely ambiguous nested quote makes the whole block a
        SyntaxError, so no call list exists and the parseable sibling died
        with the block, leaving the agent loop with no tool result."""
        text = (
            "<|tool_call_start|>[search(query='ok'), "
            "get_weather(city='x 'y', unit='c')]<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(json.loads(result.calls[0].parameters), {"query": "ok"})

    def test_swallowing_reading_rejected(self):
        """Closing the broken string late makes the text parse by absorbing
        the sibling call into the argument value, so the tool would run with
        corrupted arguments. Rejecting readings that lose calls leaves the
        correct early close and recovers both calls."""
        text = (
            "<|tool_call_start|>[search(query='x 'y'), "
            "get_weather(city='p 'q')]<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual([c.name for c in result.calls], ["search", "get_weather"])
        self.assertEqual(json.loads(result.calls[0].parameters), {"query": "x 'y"})
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "p 'q"})

    def test_unrecoverable_block_reports_no_calls(self):
        """Splitting must not fabricate calls: when no segment parses, the
        block yields no tool calls at all."""
        text = (
            "<|tool_call_start|>[search(query='x 'y' 'z), "
            "get_weather(city='p 'q' 'r)]<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])

    def test_streaming_recovers_multiline(self):
        """Streaming buffers the block and delegates to detect_and_parse;
        an incremental rewrite of the streaming path would bypass the
        recovery rewrites and re-drop multi-line commands."""
        text = (
            "<|tool_call_start|>[search(query='line one\nline two')]<|tool_call_end|>"
        )
        detector = Lfm2Detector()
        calls = []
        for i in range(0, len(text), 7):
            result = detector.parse_streaming_increment(text[i : i + 7], self.tools)
            calls.extend(result.calls)

        self.assertEqual(len(calls), 1)
        params = json.loads(calls[0].parameters)
        self.assertEqual(params["query"], "line one\nline two")

    def test_reserved_kwarg_suffix_parameter_not_rewritten(self):
        """A parameter literally named in_pyreservedkw_ must survive the
        normal parse path; only recovery-renamed kwargs get restored."""
        text = (
            "<|tool_call_start|>[search(query='x', in_pyreservedkw_=5)]"
            "<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params, {"query": "x", "in_pyreservedkw_": 5})

    def test_reserved_kwarg_with_nested_quote_recovered(self):
        """A keyword-named parameter holding a nested-quote command needs
        the rename and requote rewrites to compose."""
        text = (
            "<|tool_call_start|>[search(from='sed -n '1,5p' f.py')]" "<|tool_call_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params, {"from": "sed -n '1,5p' f.py"})

    # ==================== structure_info tests ====================

    def test_supports_structural_tag(self):
        """Test that LFM2 does not support structural tags (Pythonic format)."""
        # LFM2 uses Pythonic format which is not JSON-compatible,
        # so structural_tag constrained generation cannot be used
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info(self):
        """Test structure info for constrained generation."""
        info_func = self.detector.structure_info()
        info = info_func("get_weather")

        self.assertEqual(info.begin, "<|tool_call_start|>[get_weather(")
        self.assertEqual(info.end, ")]<|tool_call_end|>")
        self.assertEqual(info.trigger, "<|tool_call_start|>")


if __name__ == "__main__":
    import unittest

    unittest.main()
