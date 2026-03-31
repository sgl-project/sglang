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

register_cpu_ci(1.0, "stage-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
