import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.kimik2_detector import (
    KimiK2Detector as KimiK2FuncDetector,
)
from sglang.srt.function_call.kimik2_detector import (
    _strip_special_tokens,
)
from sglang.srt.parser.reasoning_parser import KimiK2Detector as KimiK2ReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


def _make_tool(name, parameters=None):
    """Helper to create a Tool with less boilerplate."""
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters=parameters
            or {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        ),
    )


def _collect_streaming_tool_calls(detector, chunks, tools):
    """Run streaming chunks through a detector and collect assembled tool calls."""
    tool_calls = []
    all_normal_text = ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        all_normal_text += result.normal_text
        for tc_chunk in result.calls:
            if tc_chunk.tool_index is not None:
                while len(tool_calls) <= tc_chunk.tool_index:
                    tool_calls.append({"name": "", "parameters": ""})
                tc = tool_calls[tc_chunk.tool_index]
                if tc_chunk.name:
                    tc["name"] = tc_chunk.name
                if tc_chunk.parameters:
                    tc["parameters"] += tc_chunk.parameters
    return tool_calls, all_normal_text


# ============================================================
# Part 1: KimiK2Detector (function call parsing) tests
# ============================================================


class TestKimiK2DetectorBasic(unittest.TestCase):
    """Basic non-streaming parsing tests for KimiK2Detector."""

    def setUp(self):
        self.tools = [
            _make_tool("ReadFile"),
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["city"],
                },
            ),
        ]
        self.detector = KimiK2FuncDetector()

    def test_single_tool_call(self):
        """Parse a single complete tool call."""
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/test.py"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "ReadFile")
        self.assertEqual(result.calls[0].parameters, '{"path": "/test.py"}')
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        """Parse two consecutive tool calls."""
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/a.py"}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.get_weather:1"
            '<|tool_call_argument_begin|>{"city": "Tokyo"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "ReadFile")
        self.assertEqual(result.calls[1].name, "get_weather")
        self.assertEqual(result.calls[1].parameters, '{"city": "Tokyo"}')

    def test_non_streaming_tool_index_is_local(self):
        """tool_index is the per-response 0-based position, not the model's :N suffix.

        The model may emit conversation-level ``:N`` counters (e.g. ``:5``, ``:6``)
        in a multi-turn conversation. The non-streaming parser must enumerate
        parsed calls locally (0, 1, ...) so that
        ``serving_chat._process_tool_call_id()`` can offset them by
        ``history_tool_calls_cnt`` without double-counting.
        """
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:5"
            '<|tool_call_argument_begin|>{"path": "/a.py"}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.get_weather:6"
            '<|tool_call_argument_begin|>{"city": "Tokyo"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[0].name, "ReadFile")
        self.assertEqual(result.calls[1].tool_index, 1)
        self.assertEqual(result.calls[1].name, "get_weather")

    def test_normal_text_before_tool_call(self):
        """Normal text before tool call markers is preserved."""
        text = (
            "Let me check the file."
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/test.py"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Let me check the file.")

    def test_no_tool_call(self):
        """Text without tool call markers returns as normal text."""
        text = "Just a normal response."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_has_tool_call(self):
        """has_tool_call correctly detects the presence of tool call markers."""
        self.assertTrue(
            self.detector.has_tool_call("<|tool_calls_section_begin|>stuff")
        )
        self.assertFalse(self.detector.has_tool_call("no markers here"))


class TestKimiK2DetectorHyphenatedNames(unittest.TestCase):
    """Test support for hyphenated function names (common in MCP tools)."""

    def setUp(self):
        self.tools = [
            _make_tool("mcp__portal__search-documents"),
            _make_tool("list-files"),
        ]
        self.detector = KimiK2FuncDetector()

    def test_hyphenated_name_non_streaming(self):
        """Parse tool call with hyphenated function name."""
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.mcp__portal__search-documents:0"
            '<|tool_call_argument_begin|>{"path": "/docs"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "mcp__portal__search-documents")

    def test_hyphenated_name_streaming(self):
        """Stream tool call with hyphenated function name."""
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.list-files:0"
            '<|tool_call_argument_begin|>{"path',
            '": "/home"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]
        tool_calls, _ = _collect_streaming_tool_calls(self.detector, chunks, self.tools)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "list-files")
        params = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params["path"], "/home")


class TestKimiK2DetectorStreaming(unittest.TestCase):
    """Streaming incremental parsing tests for KimiK2Detector."""

    def test_streaming_trailing_literal_left_angle_is_not_dropped(self):
        """A final literal '<' must remain in normal_text instead of being buffered away."""
        detector = KimiK2FuncDetector()

        result = detector.parse_streaming_increment("normal text <", [])

        self.assertEqual(result.normal_text, "normal text <")
        self.assertEqual(detector._buffer, "")

    def setUp(self):
        self.tools = [
            _make_tool("ReadFile"),
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ]

    def test_streaming_single_tool_call(self):
        """Stream a single tool call across multiple chunks."""
        detector = KimiK2FuncDetector()
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            "<|tool_call_argument_begin|>{",
            '"path": "/test.py"',
            "}",
            "<|tool_call_end|><|tool_calls_section_end|>",
        ]
        tool_calls, _ = _collect_streaming_tool_calls(detector, chunks, self.tools)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "ReadFile")
        self.assertEqual(tool_calls[0]["parameters"], '{"path": "/test.py"}')

    def test_streaming_multiple_tool_calls(self):
        """Stream two tool calls sequentially."""
        detector = KimiK2FuncDetector()
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/a.py"}',
            "<|tool_call_end|>",
            "<|tool_call_begin|>functions.get_weather:1"
            '<|tool_call_argument_begin|>{"city": "Paris"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]
        tool_calls, _ = _collect_streaming_tool_calls(detector, chunks, self.tools)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "ReadFile")
        self.assertEqual(tool_calls[1]["name"], "get_weather")
        self.assertEqual(json.loads(tool_calls[1]["parameters"]), {"city": "Paris"})

    def test_streaming_state_reset_after_completion(self):
        """Buffer and state reset after tool call completes."""
        detector = KimiK2FuncDetector()
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/x"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]
        for chunk in chunks:
            detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(detector._buffer, "")
        self.assertEqual(detector.current_tool_id, 1)


class TestKimiK2DetectorSpecialTokenLeakage(unittest.TestCase):
    """Verify special tokens are never leaked into normal_text output."""

    def setUp(self):
        self.tools = [_make_tool("ReadFile")]

    def test_no_leak_in_non_tool_text(self):
        """End tokens appearing without start tokens are stripped from output."""
        detector = KimiK2FuncDetector()
        result = detector.parse_streaming_increment(
            "normal text<|tool_calls_section_end|>", self.tools
        )
        self.assertNotIn("<|tool_calls_section_end|>", result.normal_text)
        self.assertIn("normal text", result.normal_text)

    def test_no_leak_of_argument_begin_token(self):
        """Argument begin token is stripped when leaked."""
        detector = KimiK2FuncDetector()
        result = detector.parse_streaming_increment(
            "text<|tool_call_argument_begin|>more", self.tools
        )
        self.assertNotIn("<|tool_call_argument_begin|>", result.normal_text)

    def test_no_leak_on_error_fallback(self):
        """On parse errors, normal_text fallback has tokens stripped."""
        cleaned = _strip_special_tokens(
            "leaked<|tool_calls_section_begin|>" "<|tool_call_end|>content"
        )
        self.assertEqual(cleaned, "leakedcontent")

    def test_strip_special_tokens_all_tokens(self):
        """All 5 known special tokens are stripped."""
        dirty = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>"
            "<|tool_call_argument_begin|>"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        self.assertEqual(_strip_special_tokens(dirty), "")

    def test_strip_preserves_normal_text(self):
        """Stripping doesn't affect normal text content."""
        text = "Hello world, this is normal text."
        self.assertEqual(_strip_special_tokens(text), text)


# ============================================================
# Part 2: KimiK2ReasoningDetector tests
# ============================================================


class TestKimiK2ReasoningDetectorNonStreaming(unittest.TestCase):
    """Non-streaming tests for KimiK2ReasoningDetector."""

    def test_normal_reasoning_with_think_end(self):
        """Standard case: <think>...</think> followed by tool call markers."""
        det = KimiK2ReasoningDetector()
        text = (
            "<think>I need to check the file.</think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/test.py"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = det.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "I need to check the file.")
        self.assertIn("<|tool_calls_section_begin|>", result.normal_text)

    def test_tool_call_inside_think_without_close_tag(self):
        """
        BUG FIX: Model outputs tool call markers inside <think> without </think>.

        This is the primary scenario that caused special token leakage.
        The model decides to call a tool while reasoning and directly outputs
        <|tool_calls_section_begin|> without first closing with </think>.
        """
        det = KimiK2ReasoningDetector()
        text = (
            "<think>Let me read this file..."
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/test.py"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = det.detect_and_parse(text)

        # Reasoning content must NOT contain tool call tokens
        self.assertNotIn("<|tool_calls_section_begin|>", result.reasoning_text)
        self.assertNotIn("<|tool_call_begin|>", result.reasoning_text)
        self.assertIn("Let me read this file...", result.reasoning_text)
        self.assertNotIn("<think>", result.reasoning_text)

        # Tool call markers must be in normal_text for downstream parsing
        self.assertIn("<|tool_calls_section_begin|>", result.normal_text)
        self.assertIn("<|tool_call_begin|>", result.normal_text)

    def test_no_reasoning_just_tool_call(self):
        """No <think> block, just tool call markers — pass through as normal_text."""
        det = KimiK2ReasoningDetector()
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/x"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = det.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "")
        self.assertIn("<|tool_calls_section_begin|>", result.normal_text)

    def test_normal_text_without_reasoning(self):
        """Plain text without reasoning or tool calls."""
        det = KimiK2ReasoningDetector()
        result = det.detect_and_parse("Hello, how can I help?")
        self.assertEqual(result.normal_text, "Hello, how can I help?")
        self.assertEqual(result.reasoning_text, "")


class TestKimiK2ReasoningDetectorStreaming(unittest.TestCase):
    """Streaming tests for KimiK2ReasoningDetector."""

    def _run_streaming(self, chunks, **kwargs):
        """Helper: run chunks through streaming detector, collect reasoning and normal text."""
        det = KimiK2ReasoningDetector(**kwargs)
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            r = det.parse_streaming_increment(chunk)
            all_reasoning += r.reasoning_text
            all_normal += r.normal_text
        return all_reasoning, all_normal

    def test_streaming_normal_think_then_tool_call(self):
        """Standard streaming: <think>...</think> then tool call markers."""
        reasoning, normal = self._run_streaming(
            [
                "<think>",
                "Analyzing the request...",
                "</think>",
                "<|tool_calls_section_begin|>",
                "<|tool_call_begin|>functions.ReadFile:0",
            ]
        )
        self.assertIn("Analyzing the request...", reasoning)
        self.assertIn("<|tool_calls_section_begin|>", normal)
        self.assertNotIn("<|tool_calls_section_begin|>", reasoning)

    def test_streaming_tool_call_inside_think(self):
        """
        BUG FIX (streaming): Tool call markers inside <think> without </think>.

        This is the streaming equivalent of the primary bug. The model streams
        reasoning content, then directly outputs tool call markers without </think>.
        """
        reasoning, normal = self._run_streaming(
            [
                "<think>",
                "I need to",
                " read the file.",
                "<|tool_calls_section_begin|>",
                "<|tool_call_begin|>functions.ReadFile:5",
                "<|tool_call_argument_begin|>",
                '{"path": "/Users/user/project/file.ts"}',
                "<|tool_call_end|>",
                "<|tool_calls_section_end|>",
            ]
        )

        # Reasoning is clean
        self.assertIn("I need to read the file.", reasoning)
        self.assertNotIn("<|tool_calls_section_begin|>", reasoning)
        self.assertNotIn("<|tool_call_begin|>", reasoning)
        self.assertNotIn("<think>", reasoning)

        # Tool call markers are in normal_text
        self.assertIn("<|tool_calls_section_begin|>", normal)
        self.assertIn("functions.ReadFile:5", normal)

    def test_streaming_tool_call_marker_in_single_chunk(self):
        """Tool call marker arrives in a single chunk while in reasoning mode."""
        reasoning, normal = self._run_streaming(
            [
                "<think>thinking...",
                '<|tool_calls_section_begin|><|tool_call_begin|>functions.ReadFile:0<|tool_call_argument_begin|>{"path": "/x"}',
            ]
        )
        self.assertIn("thinking...", reasoning)
        self.assertIn("<|tool_calls_section_begin|>", normal)

    def test_streaming_partial_marker_buffering(self):
        """
        Partial tool call marker at end of chunk is buffered to prevent
        premature streaming of marker characters as reasoning content.
        """
        det = KimiK2ReasoningDetector(stream_reasoning=True)

        # First chunk: reasoning + partial marker "<|tool_calls"
        det._in_reasoning = True
        det.stripped_think_start = True

        r1 = det.parse_streaming_increment("some reasoning")
        self.assertEqual(r1.reasoning_text, "some reasoning")

        # Chunk that ends with start of marker
        r2 = det.parse_streaming_increment("<|tool")
        # Partial marker should be buffered, not streamed
        self.assertNotIn("<|tool", r2.reasoning_text)

        # Complete the marker
        r3 = det.parse_streaming_increment("_calls_section_begin|>rest")
        # Now it should force-exit reasoning
        self.assertIn("<|tool_calls_section_begin|>", r3.normal_text)

    def test_streaming_no_reasoning_mode(self):
        """Normal text without reasoning passes through as normal_text."""
        reasoning, normal = self._run_streaming(
            [
                "Hello, I can help with that.",
                " What do you need?",
            ]
        )
        self.assertEqual(reasoning, "")
        self.assertIn("Hello, I can help with that.", normal)
        self.assertIn(" What do you need?", normal)

    def test_streaming_force_reasoning(self):
        """With force_reasoning, content before </think> is reasoning."""
        reasoning, normal = self._run_streaming(
            [
                "I should analyze this...",
                "</think>",
                "Here is the answer.",
            ],
            force_reasoning=True,
        )
        self.assertIn("I should analyze this...", reasoning)
        self.assertIn("Here is the answer.", normal)


# ============================================================
# Part 3: End-to-end integration tests
# ============================================================


class TestKimiK2EndToEnd(unittest.TestCase):
    """
    End-to-end tests simulating the full flow:
    reasoning parser -> tool call parser.

    These test the exact bug scenario from the issue: Kimi-K2.5 outputs
    tool call markers inside <think> blocks, which must be correctly
    split between reasoning and tool call parsers.
    """

    def setUp(self):
        self.tools = [
            _make_tool("ReadFile"),
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ]

    def test_e2e_streaming_reasoning_to_tool_call(self):
        """
        Full pipeline: streaming reasoning parser feeds into streaming tool call parser.

        Simulates the exact path through serving_chat.py:
        1. Model outputs <think>reasoning...<|tool_calls_section_begin|>...
        2. ReasoningParser splits: reasoning_text + normal_text
        3. FunctionCallParser receives normal_text and extracts tool calls
        """
        reasoning_det = KimiK2ReasoningDetector(stream_reasoning=True)
        tc_det = KimiK2FuncDetector()

        streaming_chunks = [
            "<think>",
            "I need to read the file",
            " to understand the code.",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.ReadFile:0",
            "<|tool_call_argument_begin|>",
            '{"path": "/Users/user/project/file.ts"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        all_reasoning = ""
        all_tc_calls = []

        for chunk in streaming_chunks:
            # Step 1: reasoning parser
            r = reasoning_det.parse_streaming_increment(chunk)
            all_reasoning += r.reasoning_text

            # Step 2: feed normal_text into tool call parser (like serving_chat.py does)
            if r.normal_text:
                tc_result = tc_det.parse_streaming_increment(r.normal_text, self.tools)
                all_tc_calls.extend(tc_result.calls)

        # Verify reasoning content
        self.assertIn("I need to read the file to understand the code.", all_reasoning)
        self.assertNotIn("<|", all_reasoning)

        # Verify tool calls were extracted
        name_calls = [c for c in all_tc_calls if c.name]
        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "ReadFile")

        param_calls = [c for c in all_tc_calls if c.parameters]
        full_params = "".join(c.parameters for c in param_calls)
        self.assertIn("/Users/user/project/file.ts", full_params)

    def test_e2e_non_streaming_reasoning_to_tool_call(self):
        """Non-streaming pipeline: reason parser then tool call parser."""
        reasoning_det = KimiK2ReasoningDetector()
        tc_det = KimiK2FuncDetector()

        text = (
            "<think>Let me check this file."
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/src/main.py"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )

        # Step 1: reasoning parser
        r = reasoning_det.detect_and_parse(text)
        self.assertIn("Let me check this file.", r.reasoning_text)
        self.assertNotIn("<|", r.reasoning_text)

        # Step 2: tool call parser on normal_text
        tc_result = tc_det.detect_and_parse(r.normal_text, self.tools)
        self.assertEqual(len(tc_result.calls), 1)
        self.assertEqual(tc_result.calls[0].name, "ReadFile")
        self.assertEqual(
            json.loads(tc_result.calls[0].parameters),
            {"path": "/src/main.py"},
        )

    def test_e2e_normal_think_close_then_tool_call(self):
        """Standard case with </think> — should also work correctly."""
        reasoning_det = KimiK2ReasoningDetector(stream_reasoning=True)
        tc_det = KimiK2FuncDetector()

        chunks = [
            "<think>",
            "Thinking about it...",
            "</think>",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_weather:0",
            '<|tool_call_argument_begin|>{"city": "London"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        all_reasoning = ""
        all_tc_calls = []

        for chunk in chunks:
            r = reasoning_det.parse_streaming_increment(chunk)
            all_reasoning += r.reasoning_text
            if r.normal_text:
                tc_result = tc_det.parse_streaming_increment(r.normal_text, self.tools)
                all_tc_calls.extend(tc_result.calls)

        self.assertIn("Thinking about it...", all_reasoning)
        name_calls = [c for c in all_tc_calls if c.name]
        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")

    def test_e2e_normal_think_close_then_content_overlap_tool_call(self):
        """Trailing content between ``</think>`` and the tool-call markers
        (e.g. ``"This is a content:"``) must be surfaced as ``normal_text``
        by the tool-call parser and not stripped — this is the exact bug the
        PR fixes.
        """
        reasoning_det = KimiK2ReasoningDetector(stream_reasoning=True)
        tc_det = KimiK2FuncDetector()

        chunks = [
            "<think>",
            "Thinking about it...",
            "</think>This is a ",
            "content:<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_weather:0",
            '<|tool_call_argument_begin|>{"city": "London"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        all_reasoning = ""
        all_content = ""

        toolcall_chunks = []
        for chunk in chunks:
            r = reasoning_det.parse_streaming_increment(chunk)
            all_reasoning += r.reasoning_text
            if r.normal_text:
                toolcall_chunks.append(r.normal_text)

        tool_calls, all_content = _collect_streaming_tool_calls(
            tc_det, toolcall_chunks, self.tools
        )

        self.assertEqual("Thinking about it...", all_reasoning)
        self.assertEqual("This is a content:", all_content)
        self.assertEqual(len(tool_calls), 1)
        first_call = tool_calls.pop()
        self.assertEqual(first_call["name"], "get_weather")
        self.assertEqual(first_call["parameters"], '{"city": "London"}')

    def test_e2e_normal_think_close_then_content_overlap_tool_call_multi_token(self):
        """Speculative decoding: a single chunk may contain normal text followed
        by tool-call markers and even the tool_call_begin/id. The normal-text
        prefix must still be emitted (not stripped) by the tool-call parser.
        """
        reasoning_det = KimiK2ReasoningDetector(stream_reasoning=True)
        tc_det = KimiK2FuncDetector()

        chunks = [
            "<think>",
            "Thinking about it...",
            "</think>This is a ",
            "content:<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "London"}<|tool_call_end|><|tool_calls_section_end|>',
        ]

        all_reasoning = ""
        all_content = ""
        toolcall_chunks = []

        for chunk in chunks:
            r = reasoning_det.parse_streaming_increment(chunk)
            all_reasoning += r.reasoning_text
            if r.normal_text:
                toolcall_chunks.append(r.normal_text)

        tool_calls, all_content = _collect_streaming_tool_calls(
            tc_det, toolcall_chunks, self.tools
        )

        self.assertEqual("Thinking about it...", all_reasoning)
        self.assertEqual("This is a content:", all_content)
        self.assertEqual(len(tool_calls), 1)
        first_call = tool_calls.pop()
        self.assertEqual(first_call["name"], "get_weather")
        self.assertEqual(first_call["parameters"], '{"city": "London"}')

    def test_e2e_normal_think_close_then_content_overlap_tool_call_multi_token_multi_calls(
        self,
    ):
        """Speculative decoding: a single chunk may contain normal text followed
        by tool-call markers and even the tool_call_begin/id. Additionally, this
        single chunk packs two complete tool-call sections back-to-back, which
        exercises the ``while True:`` drain loop introduced by this PR — both
        calls must be emitted from one ``parse_streaming_increment`` invocation.
        """
        reasoning_det = KimiK2ReasoningDetector(stream_reasoning=True)
        tc_det = KimiK2FuncDetector()

        chunks = [
            "<think>",
            "Thinking about it...",
            "</think>This is a ",
            'content:<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "London"}<|tool_call_end|><|tool_calls_section_end|><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>'
            '{"city": "Delhi"}<|tool_call_end|><|tool_calls_section_end|>',
        ]

        all_reasoning = ""
        all_content = ""
        toolcall_chunks = []

        for chunk in chunks:
            r = reasoning_det.parse_streaming_increment(chunk)
            all_reasoning += r.reasoning_text
            if r.normal_text:
                toolcall_chunks.append(r.normal_text)

        tool_calls, all_content = _collect_streaming_tool_calls(
            tc_det, toolcall_chunks, self.tools
        )

        self.assertEqual("Thinking about it...", all_reasoning)
        self.assertEqual("This is a content:", all_content)
        self.assertEqual(len(tool_calls), 2)
        first_call = tool_calls.pop(0)
        self.assertEqual(first_call["name"], "get_weather")
        self.assertEqual(first_call["parameters"], '{"city": "London"}')
        second_call = tool_calls.pop(0)
        self.assertEqual(second_call["name"], "get_weather")
        self.assertEqual(second_call["parameters"], '{"city": "Delhi"}')

    def test_e2e_chunk_split_invariance(self):
        """The detector must produce identical results across a few realistic
        chunking variants. Special tokens (e.g. ``<|tool_calls_section_begin|>``)
        are atomic and never split, so cuts only fall on token boundaries or
        inside JSON args.
        """
        prefix = "<think>Thinking about it...</think>This is a content:"
        call1 = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "London"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        call2 = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:1"
            '<|tool_call_argument_begin|>{"city": "Delhi"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )

        expected_reasoning = "Thinking about it..."
        expected_content = "This is a content:"
        expected_calls = [
            {"name": "get_weather", "parameters": '{"city": "London"}'},
            {"name": "get_weather", "parameters": '{"city": "Delhi"}'},
        ]

        variants = {
            # One complete tool call per chunk.
            "first_complete_then_second_complete": [prefix + call1, call2],
            # Both tool calls arrive in a single chunk.
            "both_in_one_chunk": [prefix + call1 + call2],
            # First call complete + second call partial (cut inside JSON args),
            # then the rest of the second call.
            "first_complete_second_partial_then_rest": [
                prefix
                + call1
                + "<|tool_calls_section_begin|>"
                + "<|tool_call_begin|>functions.get_weather:1"
                + '<|tool_call_argument_begin|>{"city": "De',
                'lhi"}<|tool_call_end|><|tool_calls_section_end|>',
            ],
            # First call partial (cut inside JSON args), then rest of first +
            # full second call.
            "first_partial_then_first_complete_second_complete": [
                prefix
                + "<|tool_calls_section_begin|>"
                + "<|tool_call_begin|>functions.get_weather:0"
                + '<|tool_call_argument_begin|>{"city": "Lon',
                'don"}<|tool_call_end|><|tool_calls_section_end|>' + call2,
            ],
        }

        for name, chunks in variants.items():
            with self.subTest(variant=name):
                reasoning_det = KimiK2ReasoningDetector(stream_reasoning=True)
                tc_det = KimiK2FuncDetector()
                all_reasoning = ""
                toolcall_chunks = []
                for chunk in chunks:
                    r = reasoning_det.parse_streaming_increment(chunk)
                    all_reasoning += r.reasoning_text
                    if r.normal_text:
                        toolcall_chunks.append(r.normal_text)
                tool_calls, all_content = _collect_streaming_tool_calls(
                    tc_det, toolcall_chunks, self.tools
                )
                self.assertEqual(all_reasoning, expected_reasoning)
                self.assertEqual(all_content, expected_content)
                self.assertEqual(tool_calls, expected_calls)

    def test_e2e_normal_text_between_two_tool_calls(self):
        """Normal text appearing BETWEEN two tool-call sections must be
        surfaced as ``normal_text``.
        """
        tc_det = KimiK2FuncDetector()
        chunks = [
            "Prefix text:"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "London"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            " Now calling next: "
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:1"
            "<|tool_call_argument_begin|>"
            '{"city":'
            ' "Delhi"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        ]

        tool_calls, all_content = _collect_streaming_tool_calls(
            tc_det, chunks, self.tools
        )

        self.assertIn("Prefix text:", all_content)
        self.assertIn("Now calling next:", all_content)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "London"}')
        self.assertEqual(tool_calls[1]["name"], "get_weather")
        self.assertEqual(tool_calls[1]["parameters"], '{"city": "Delhi"}')

    def test_e2e_unparsable_tool_id_does_not_wedge_stream(self):
        """A tool_call header with an unparsable ID must not wedge the
        streaming parser.
        """
        tc_det = KimiK2FuncDetector()

        # ``weird@id`` matches the broad ``[^\\s<|]+`` capture in
        # ``stream_tool_call_portion_regex`` but fails both the standard
        # ``name:idx`` form and the bare-counter form, so
        # ``_parse_tool_call_id`` returns ``(None, 0)``.
        chunks = [
            "normal text before",
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>weird@id"
            '<|tool_call_argument_begin|>{"city"'
            ': "London"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>",
            # A valid follow-up call: must still be parsed.
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "Delhi"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>",
        ]

        tool_calls, all_content = _collect_streaming_tool_calls(
            tc_det, chunks, self.tools
        )

        # The bad call may be skipped/logged, but the follow-up MUST be
        # emitted. The stream must not be wedged on the bad header.
        valid = [c for c in tool_calls if c.get("name") == "get_weather"]
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["parameters"], '{"city": "Delhi"}')
        self.assertIn("normal text before", all_content)

    def test_e2e_malformed_json_args_passes_through_to_client(self):
        """A tool call with malformed JSON args (e.g. unclosed brace,
        spelling mistake) is the **client's** problem to
        validate/repair — the parser's job is to locate boundaries and
        hand back the raw argument string. This mirrors the
        ``detect_and_parse`` (non-streaming) contract.

        Required behavior:

        1. The malformed call is emitted unchanged (raw bytes
           preserved, name + index intact) so the client can decide
           how to handle it (reject, repair, replay-prompt, etc).
        2. The stream is not wedged — the trailing valid call must
           still parse.
        3. ``current_tool_id`` advances normally — the trailing valid
           call sits at index 2, not 1 or 3.

        Asserts the SAME outcome under three chunk layouts:

        * **single-chunk / MTP path** — all three sections in one
          forward step (mimics speculative / multi-token-prediction).
        * **per-call split path** — one section per chunk.
        * **bad section split mid-payload** — the malformed section
          itself is fragmented across three chunks (header + partial
          args; more args; end-token + trailing valid call). Verifies
          the atomic-section buffer correctly defers emission until
          ``<|tool_call_end|>`` arrives.
        """
        good_args_0 = '{"city": "London"}'
        # JSON keyword (the model misspelled ``false``).
        bad_args = '{"city": "Bad", "valid": fasle'
        good_args_1 = '{"city": "Delhi"}'

        good_section_0 = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            f"<|tool_call_argument_begin|>{good_args_0}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        bad_section = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:1"
            f"<|tool_call_argument_begin|>{bad_args}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        good_section_1 = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:2"
            f"<|tool_call_argument_begin|>{good_args_1}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )

        layouts = {
            "mtp_single_chunk": [good_section_0 + bad_section + good_section_1],
            "per_call_chunks": [good_section_0, bad_section, good_section_1],
            "bad_section_split_mid_payload": [
                good_section_0,
                "<|tool_calls_section_begin|>"
                "<|tool_call_begin|>functions.get_weather:1"
                f'<|tool_call_argument_begin|>{{"city":',
                ' "Bad", "valid": fasle',
                "<|tool_call_end|>" "<|tool_calls_section_end|>" + good_section_1,
            ],
        }

        for layout_name, chunks in layouts.items():
            with self.subTest(layout=layout_name):
                tc_det = KimiK2FuncDetector()
                tool_calls, _ = _collect_streaming_tool_calls(
                    tc_det, chunks, self.tools
                )

                # Three contiguous tool calls (good, bad-passthrough, good).
                self.assertEqual(
                    len(tool_calls),
                    3,
                    f"[{layout_name}] expected 3 calls, got: {tool_calls!r}",
                )
                self.assertEqual(tool_calls[0]["name"], "get_weather")
                self.assertEqual(tool_calls[1]["name"], "get_weather")
                self.assertEqual(tool_calls[2]["name"], "get_weather")
                self.assertEqual(tool_calls[0]["parameters"], good_args_0)
                # Bad payload preserved byte-for-byte for the client.
                self.assertEqual(tool_calls[1]["parameters"], bad_args)
                self.assertEqual(tool_calls[2]["parameters"], good_args_1)

    def test_e2e_exception_mid_drain_preserves_accumulated_calls(self):
        """An exception raised mid-drain must not discard tool calls already
        finalized in the same ``parse_streaming_increment`` invocation.
        """
        import unittest.mock as mock

        tc_det = KimiK2FuncDetector()

        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "London"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:1"
            '<|tool_call_argument_begin|>{"city": "Delhi"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        ]

        # Force an exception on the SECOND resolution step by making
        # ``_resolve_function_name`` raise the second time it is invoked.
        call_count = {"n": 0}
        real_resolve = tc_det._resolve_function_name

        def flaky_resolve(function_id, tools, function_args=None):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise RuntimeError("forced mid-drain failure")
            return real_resolve(function_id, tools, function_args)

        with mock.patch.object(
            tc_det, "_resolve_function_name", side_effect=flaky_resolve
        ):
            tool_calls, all_content = _collect_streaming_tool_calls(
                tc_det, chunks, self.tools
            )

        # First call must survive the mid-drain exception.
        named = [c for c in tool_calls if c.get("name")]
        self.assertGreaterEqual(len(named), 1)
        self.assertEqual(named[0]["name"], "get_weather")
        self.assertEqual(named[0]["parameters"], '{"city": "London"}')
        # Already-finalized call payload must NOT leak into normal_text.
        self.assertNotIn('{"city": "London"}', all_content)
        self.assertNotIn("<|tool_call_begin|>", all_content)

    def test_e2e_multiple_tool_calls_without_think_close(self):
        """Multiple tool calls inside <think> without </think>."""
        reasoning_det = KimiK2ReasoningDetector(stream_reasoning=True)
        tc_det = KimiK2FuncDetector()

        chunks = [
            "<think>",
            "Let me check both files.",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>{"path": "/a.py"}',
            "<|tool_call_end|>",
            "<|tool_call_begin|>functions.ReadFile:1"
            '<|tool_call_argument_begin|>{"path": "/b.py"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        all_reasoning = ""
        all_tc_calls = []

        for chunk in chunks:
            r = reasoning_det.parse_streaming_increment(chunk)
            all_reasoning += r.reasoning_text
            if r.normal_text:
                tc_result = tc_det.parse_streaming_increment(r.normal_text, self.tools)
                all_tc_calls.extend(tc_result.calls)

        self.assertIn("Let me check both files.", all_reasoning)
        self.assertNotIn("<|", all_reasoning)

        name_calls = [c for c in all_tc_calls if c.name]
        self.assertEqual(len(name_calls), 2)
        self.assertEqual(name_calls[0].name, "ReadFile")
        self.assertEqual(name_calls[1].name, "ReadFile")


# ============================================================
# Part 2b: OpenAI streaming-spec compliance for ``tool_index``
# ============================================================


class TestKimiK2DetectorOpenAIIndexCompliance(unittest.TestCase):
    """The detector must emit ``tool_index`` as a dense, 0-based position
    within the *current response* (per the OpenAI streaming spec), regardless
    of the value of the model's conversation-level ``:N`` counter in the
    tool_call header. The serving layer is responsible for adding any
    history offset back when synthesizing the public ``id`` field.

    These tests pin the detector contract so multi-turn conversations
    (where the model continues an auto-incrementing counter across turns)
    can never produce sparse / non-zero-based ``index`` values in the
    streamed delta.
    """

    def setUp(self):
        self.tools = [
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ]

    def _run(self, chunks):
        det = KimiK2FuncDetector()
        events = []
        for chunk in chunks:
            r = det.parse_streaming_increment(chunk, self.tools)
            events.extend(r.calls)
        return events

    def test_single_call_with_nonzero_model_counter_starts_at_index_0(self):
        """Model continues a conversation-level counter (``:5``) across turns.
        The detector must still emit ``tool_index=0`` for the first call in
        this response — the model's ``:N`` suffix MUST NOT leak into ``index``.
        """
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:5"
            '<|tool_call_argument_begin|>{"city":',
            ' "Paris"}<|tool_call_end|><|tool_calls_section_end|>',
        ]
        events = self._run(chunks)

        self.assertGreaterEqual(len(events), 1)
        # Every emitted delta (name event + arg deltas) must use index 0.
        for ev in events:
            self.assertEqual(
                ev.tool_index,
                0,
                f"tool_index must be 0-based per response, got {ev.tool_index}",
            )
        first = events[0]
        self.assertEqual(first.name, "get_weather")

    def test_multi_call_response_uses_dense_0_based_indices(self):
        """Two calls in one response, model emits ``:7`` then ``:8``. The
        detector must emit ``tool_index=0`` then ``tool_index=1`` (dense,
        0-based), independent of the model's counter.
        """
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:7"
            '<|tool_call_argument_begin|>{"city": "London"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:8"
            '<|tool_call_argument_begin|>{"city": "Berlin"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        ]
        events = self._run(chunks)

        name_events = [e for e in events if e.name]
        self.assertEqual(len(name_events), 2)
        self.assertEqual(name_events[0].tool_index, 0)
        self.assertEqual(name_events[1].tool_index, 1)

        # Every delta for call N must carry tool_index == N.
        for ev in events:
            self.assertIn(ev.tool_index, (0, 1))

    def test_continuation_chunks_keep_same_tool_index(self):
        """Per OpenAI spec, all argument-delta chunks for a given call
        must share the same ``index``. Split the args across multiple
        chunks and verify ``tool_index`` stays at 0 throughout.
        """
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:10"
            '<|tool_call_argument_begin|>{"city":',
            ' "Pa',
            'ris"}',
            "<|tool_call_end|><|tool_calls_section_end|>",
        ]
        events = self._run(chunks)

        self.assertGreater(len(events), 1, "expected name event + arg deltas")
        for ev in events:
            self.assertEqual(ev.tool_index, 0)

        # Reassembled args must round-trip.
        joined = "".join(ev.parameters or "" for ev in events)
        self.assertEqual(joined, '{"city": "Paris"}')

    def test_three_calls_with_nonzero_model_counter_indices_are_dense(self):
        """Multi-turn worst case: model continues at ``:10`` and emits three
        calls in this response. Indices must be 0, 1, 2 — not 10, 11, 12.
        """
        sections = []
        for offset, city in enumerate(("Paris", "Berlin", "Madrid")):
            sections.append(
                "<|tool_calls_section_begin|>"
                f"<|tool_call_begin|>functions.get_weather:{10 + offset}"
                f'<|tool_call_argument_begin|>{{"city": "{city}"}}'
                "<|tool_call_end|>"
                "<|tool_calls_section_end|>"
            )
        events = self._run(["".join(sections)])

        name_events = [e for e in events if e.name]
        self.assertEqual([e.tool_index for e in name_events], [0, 1, 2])

    def test_bare_counter_id_also_starts_at_index_0(self):
        """Same invariant for the bare-counter ID form (model omits function
        name and emits just a numeric counter, e.g. ``:5``).
        """
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>5"
            '<|tool_call_argument_begin|>{"city": "Paris"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        ]
        events = self._run(chunks)

        self.assertGreaterEqual(len(events), 1)
        for ev in events:
            self.assertEqual(ev.tool_index, 0)


# ============================================================
# Part 3: Bare-counter tool call ID parsing
# ============================================================


class TestKimiK2BareCounterParsing(unittest.TestCase):
    """Tests for bare numeric tool_call_id format (e.g., '3' instead of 'functions.ReadFile:0')."""

    def setUp(self):
        self.detector = KimiK2FuncDetector()
        self.tools = [
            _make_tool("ReadFile"),
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["city"],
                },
            ),
        ]

    # --- _parse_tool_call_id ---

    def test_standard_format_with_functions_prefix(self):
        name, idx = self.detector._parse_tool_call_id(
            "functions.ReadFile:0", self.tools
        )
        self.assertEqual(name, "ReadFile")
        self.assertEqual(idx, 0)

    def test_standard_format_without_functions_prefix(self):
        name, idx = self.detector._parse_tool_call_id("ReadFile:1", self.tools)
        self.assertEqual(name, "ReadFile")
        self.assertEqual(idx, 1)

    def test_bare_counter_single_tool(self):
        single_tool = [_make_tool("search")]
        name, idx = self.detector._parse_tool_call_id(
            "3", single_tool, '{"query": "test"}'
        )
        self.assertEqual(name, "search")
        self.assertEqual(idx, 3)

    def test_bare_counter_infers_by_args(self):
        name, idx = self.detector._parse_tool_call_id(
            "0", self.tools, '{"city": "Tokyo"}'
        )
        self.assertEqual(name, "get_weather")
        self.assertEqual(idx, 0)

    def test_bare_counter_no_tools_returns_none(self):
        name, idx = self.detector._parse_tool_call_id("5", [], '{"x": 1}')
        self.assertIsNone(name)
        self.assertEqual(idx, 5)

    def test_bare_counter_no_args_multiple_tools_returns_none(self):
        name, idx = self.detector._parse_tool_call_id("2", self.tools, None)
        self.assertIsNone(name)
        self.assertEqual(idx, 2)

    def test_unexpected_format_returns_none(self):
        name, idx = self.detector._parse_tool_call_id("some_garbage", self.tools)
        self.assertIsNone(name)
        self.assertEqual(idx, 0)

    # --- _infer_tool_name ---

    def test_infer_no_tools(self):
        self.assertIsNone(self.detector._infer_tool_name([], '{"x": 1}'))

    def test_infer_single_tool(self):
        result = self.detector._infer_tool_name([_make_tool("only_one")], '{"x": 1}')
        self.assertEqual(result, "only_one")

    def test_infer_by_argument_overlap(self):
        result = self.detector._infer_tool_name(
            self.tools, '{"city": "Paris", "unit": "celsius"}'
        )
        self.assertEqual(result, "get_weather")

    def test_infer_malformed_json_returns_none(self):
        result = self.detector._infer_tool_name(self.tools, '{"city": "Par')
        self.assertIsNone(result)

    def test_infer_empty_args_returns_none(self):
        self.assertIsNone(self.detector._infer_tool_name(self.tools, None))
        self.assertIsNone(self.detector._infer_tool_name(self.tools, ""))

    def test_infer_no_matching_props_returns_none(self):
        tools_no_props = [
            _make_tool("a", {"type": "object"}),
            _make_tool("b", {"type": "object"}),
        ]
        result = self.detector._infer_tool_name(tools_no_props, '{"x": 1}')
        self.assertIsNone(result)

    # --- detect_and_parse with bare counter (end-to-end) ---

    def test_detect_and_parse_bare_counter(self):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>0"
            '<|tool_call_argument_begin|>{"city": "Tokyo"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].parameters, '{"city": "Tokyo"}')

    def test_detect_and_parse_bare_counter_skips_unknown(self):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>0"
            '<|tool_call_argument_begin|>{"unknown_key": "value"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        # No tool props match, _infer_tool_name returns None, call is skipped
        self.assertEqual(len(result.calls), 0)

    def test_streaming_bare_counter_single_tool(self):
        detector = KimiK2FuncDetector()
        single_tool = [_make_tool("search")]
        chunks = [
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>0"
            '<|tool_call_argument_begin|>{"path',
            '": "/test"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]
        tool_calls, _ = _collect_streaming_tool_calls(detector, chunks, single_tool)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "search")


if __name__ == "__main__":
    unittest.main()
