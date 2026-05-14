"""Unit tests for srt/parser/reasoning_parser.py"""

import unittest

from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    DeepSeekR1Detector,
    Gemma4Detector,
    Glm45Detector,
    HunyuanDetector,
    KimiDetector,
    KimiK2Detector,
    Nemotron3Detector,
    Qwen3Detector,
    ReasoningParser,
    StreamingParseResult,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="stage-a-test-cpu")


class TestStreamingParseResult(CustomTestCase):
    def test_init_default(self):
        """Test default initialization of StreamingParseResult."""
        result = StreamingParseResult()
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "")

    def test_init_with_values(self):
        """Test initialization with specific values."""
        result = StreamingParseResult("normal", "reasoning")
        self.assertEqual(result.normal_text, "normal")
        self.assertEqual(result.reasoning_text, "reasoning")


class TestBaseReasoningFormatDetector(CustomTestCase):
    def setUp(self):
        self.detector = BaseReasoningFormatDetector(
            think_start_token="<think>",
            think_end_token="</think>",
            force_reasoning=False,
            stream_reasoning=True,
        )

    def test_init(self):
        """Test initialization of BaseReasoningFormatDetector."""
        self.assertEqual(self.detector.think_start_token, "<think>")
        self.assertEqual(self.detector.think_end_token, "</think>")
        self.assertFalse(self.detector._in_reasoning)
        self.assertTrue(self.detector.stream_reasoning)
        self.assertEqual(self.detector._buffer, "")
        self.assertFalse(self.detector.stripped_think_start)

    def test_detect_and_parse_normal_text(self):
        """Test parsing normal text without reasoning."""
        text = "This is normal text"
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")

    def test_detect_and_parse_with_start_token(self):
        """Test parsing text starting with think token."""
        text = "<think>This is reasoning"
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "This is reasoning")
        self.assertEqual(result.normal_text, "")

    def test_detect_and_parse_complete_reasoning(self):
        """Test parsing complete reasoning block."""
        text = "<think>This is reasoning</think>This is normal"
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "This is reasoning")
        self.assertEqual(result.normal_text, "This is normal")

    def test_detect_and_parse_force_reasoning(self):
        """Test forced reasoning mode."""
        detector = BaseReasoningFormatDetector(
            "<think>", "</think>", force_reasoning=True
        )
        text = "This should be reasoning"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "This should be reasoning")
        self.assertEqual(result.normal_text, "")

    def test_parse_streaming_increment_normal(self):
        """Test streaming parse of normal text."""
        result = self.detector.parse_streaming_increment("Hello world")
        self.assertEqual(result.normal_text, "Hello world")
        self.assertEqual(result.reasoning_text, "")

    def test_parse_streaming_increment_partial_token(self):
        """Test streaming parse with partial token."""
        # Test partial start token
        result = self.detector.parse_streaming_increment("<thi")
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "")

        # Reset detector and test partial end token when in reasoning mode
        detector = BaseReasoningFormatDetector("<think>", "</think>")
        detector._in_reasoning = True
        result = detector.parse_streaming_increment("</thi")
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "")

    def test_parse_streaming_increment_complete_start(self):
        """Test streaming parse with complete start token."""
        result = self.detector.parse_streaming_increment("<think>")
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "")
        self.assertTrue(self.detector._in_reasoning)
        self.assertTrue(self.detector.stripped_think_start)

    def test_parse_streaming_increment_reasoning_content(self):
        """Test streaming parse of reasoning content."""
        # First add start token
        self.detector.parse_streaming_increment("<think>")

        # Then add reasoning content
        result = self.detector.parse_streaming_increment("reasoning content")
        self.assertEqual(result.reasoning_text, "reasoning content")
        self.assertEqual(result.normal_text, "")

    def test_parse_streaming_increment_end_token(self):
        """Test streaming parse with end token."""
        # Start reasoning mode
        self.detector.parse_streaming_increment("<think>")
        self.detector.parse_streaming_increment("reasoning")

        # End reasoning - the reasoning content accumulated in previous calls is cleared when end token is found
        result = self.detector.parse_streaming_increment("</think>normal text")
        self.assertEqual(result.reasoning_text, "")  # Buffer cleared, returns empty
        self.assertEqual(result.normal_text, "normal text")
        self.assertFalse(self.detector._in_reasoning)

    def test_parse_streaming_increment_no_stream_reasoning(self):
        """Test streaming parse without streaming reasoning."""
        detector = BaseReasoningFormatDetector(
            "<think>", "</think>", stream_reasoning=False
        )

        # Start reasoning mode
        detector.parse_streaming_increment("<think>")

        # Add reasoning content - should not return content
        result = detector.parse_streaming_increment("reasoning content")
        self.assertEqual(result.reasoning_text, "")
        self.assertEqual(result.normal_text, "")

    def test_parse_streaming_increment_mixed_content(self):
        """Test streaming parse with mixed content in one chunk."""
        result = self.detector.parse_streaming_increment(
            "<think>reasoning</think>normal"
        )
        self.assertEqual(result.reasoning_text, "reasoning")
        self.assertEqual(result.normal_text, "normal")


class TestDeepSeekR1Detector(CustomTestCase):
    def setUp(self):
        self.detector = DeepSeekR1Detector()

    def test_init(self):
        """Test DeepSeekR1Detector initialization."""
        self.assertEqual(self.detector.think_start_token, "<think>")
        self.assertEqual(self.detector.think_end_token, "</think>")
        self.assertTrue(self.detector._in_reasoning)  # force_reasoning=True
        self.assertTrue(self.detector.stream_reasoning)

    def test_init_no_stream_reasoning(self):
        """Test DeepSeekR1Detector with stream_reasoning=False."""
        detector = DeepSeekR1Detector(stream_reasoning=False)
        self.assertFalse(detector.stream_reasoning)

    def test_detect_and_parse_r1_format(self):
        """Test parsing DeepSeek-R1 format."""
        text = "I need to think about this. The answer is 42."
        result = self.detector.detect_and_parse(text)
        # Should be treated as reasoning because force_reasoning=True
        self.assertEqual(
            result.reasoning_text, "I need to think about this. The answer is 42."
        )
        self.assertEqual(result.normal_text, "")

    def test_detect_and_parse_with_end_token(self):
        """Test parsing with end token."""
        text = "I think this is the answer</think>The final answer is 42."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "I think this is the answer")
        self.assertEqual(result.normal_text, "The final answer is 42.")

    def test_detect_and_parse_with_start_token(self):
        """Test parsing deepseek-ai/DeepSeek-R1-0528 format, which generates the <think> token."""
        text = "<think>I need to think about this.</think>The answer is 42."
        result = self.detector.detect_and_parse(text)
        # Should be treated as reasoning because force_reasoning=True
        self.assertEqual(result.reasoning_text, "I need to think about this.")
        self.assertEqual(result.normal_text, "The answer is 42.")


class TestQwen3Detector(CustomTestCase):
    def setUp(self):
        self.detector = Qwen3Detector()

    def test_init(self):
        """Test Qwen3Detector initialization."""
        self.assertEqual(self.detector.think_start_token, "<think>")
        self.assertEqual(self.detector.think_end_token, "</think>")
        self.assertFalse(self.detector._in_reasoning)  # force_reasoning=False
        self.assertTrue(self.detector.stream_reasoning)

    def test_detect_and_parse_qwen3_format(self):
        """Test parsing Qwen3 format."""
        text = "<think>Let me think about this problem</think>The answer is 42."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "Let me think about this problem")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_detect_and_parse_without_thinking(self):
        """Test parsing without thinking (enable_thinking=False case)."""
        text = "Direct answer without thinking."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")


class TestQwen3ForcedReasoningDetector(CustomTestCase):
    def setUp(self):
        self.detector = Qwen3Detector(force_reasoning=True)

    def test_init(self):
        """Test Qwen3ForcedReasoningDetector initialization."""
        self.assertEqual(self.detector.think_start_token, "<think>")
        self.assertEqual(self.detector.think_end_token, "</think>")
        self.assertTrue(self.detector._in_reasoning)  # force_reasoning=True
        self.assertTrue(self.detector.stream_reasoning)

    def test_detect_and_parse_qwen3_forced_reasoning_format(self):
        """Test parsing Qwen3-ForcedReasoning format (no <think> start tag)."""
        text = "I need to think about this step by step.</think>The answer is 42."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(
            result.reasoning_text, "I need to think about this step by step."
        )
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_detect_and_parse_with_start_token(self):
        """Test parsing Qwen3-ForcedReasoning with optional <think> start tag."""
        text = "<think>I need to think about this.</think>The answer is 42."
        result = self.detector.detect_and_parse(text)
        # Should work because base class logic handles both force_reasoning=True OR start token
        self.assertEqual(result.reasoning_text, "I need to think about this.")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_streaming_qwen3_forced_reasoning_format(self):
        """Test streaming parse of Qwen3-ForcedReasoning format."""
        # First chunk without <think> start
        result = self.detector.parse_streaming_increment("I need to")
        self.assertEqual(result.reasoning_text, "I need to")
        self.assertEqual(result.normal_text, "")

        # More reasoning content
        result = self.detector.parse_streaming_increment(" think about this.")
        self.assertEqual(result.reasoning_text, " think about this.")
        self.assertEqual(result.normal_text, "")

        # End token with normal text
        result = self.detector.parse_streaming_increment("</think>The answer is 42.")
        self.assertEqual(result.reasoning_text, "")  # Buffer cleared
        self.assertEqual(result.normal_text, "The answer is 42.")


class TestKimiDetector(CustomTestCase):
    def setUp(self):
        self.detector = KimiDetector()

    def test_init(self):
        """Test KimiDetector initialization."""
        self.assertEqual(self.detector.think_start_token, "◁think▷")
        self.assertEqual(self.detector.think_end_token, "◁/think▷")
        self.assertFalse(self.detector._in_reasoning)
        self.assertTrue(self.detector.stream_reasoning)

    def test_detect_and_parse_kimi_format(self):
        """Test parsing Kimi format."""
        text = "◁think▷Let me consider this carefully◁/think▷The answer is 42."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "Let me consider this carefully")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_detect_and_parse_kimi_no_thinking(self):
        """Test parsing Kimi format without thinking."""
        text = "Direct answer without thinking tokens."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")

    def test_streaming_kimi_format(self):
        """Test streaming parse of Kimi format."""
        # Test partial token
        result = self.detector.parse_streaming_increment("◁thi")
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "")

        # Complete start token
        result = self.detector.parse_streaming_increment("nk▷Start")
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "Start")
        self.assertTrue(self.detector._in_reasoning)

        # Add reasoning content
        result = self.detector.parse_streaming_increment("thinking...")
        self.assertEqual(result.reasoning_text, "thinking...")
        self.assertEqual(result.normal_text, "")

        # End token - reasoning content is cleared when end token is processed
        result = self.detector.parse_streaming_increment("◁/think▷answer")
        self.assertEqual(result.reasoning_text, "")  # Buffer cleared
        self.assertEqual(result.normal_text, "answer")


class TestKimiK2Detector(CustomTestCase):
    """Test cases for KimiK2 detector with tool interruption support."""

    def setUp(self):
        self.detector = KimiK2Detector()

    def test_init(self):
        """Test KimiK2Detector initialization."""
        self.assertEqual(self.detector.think_start_token, "<think>")
        self.assertEqual(self.detector.think_end_token, "</think>")
        self.assertEqual(self.detector.tool_start_token, "<|tool_calls_section_begin|>")
        self.assertFalse(self.detector._in_reasoning)
        self.assertTrue(self.detector.stream_reasoning)

    def test_detect_and_parse_tool_interrupt(self):
        """Test parsing with Kimi-K2 tool-section interruption."""
        text = "<think>thinking<|tool_calls_section_begin|><|tool_call_begin|>"
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "thinking")
        self.assertEqual(
            result.normal_text, "<|tool_calls_section_begin|><|tool_call_begin|>"
        )

    def test_streaming_tool_interrupt(self):
        """Test streaming parse interrupted by tool section."""
        self.detector.parse_streaming_increment("<think>")
        result1 = self.detector.parse_streaming_increment("reasoning")
        self.assertEqual(result1.reasoning_text, "reasoning")
        self.assertEqual(result1.normal_text, "")

        result2 = self.detector.parse_streaming_increment(
            "<|tool_calls_section_begin|>"
        )
        self.assertEqual(result2.reasoning_text, "")
        self.assertEqual(result2.normal_text, "<|tool_calls_section_begin|>")

    def test_streaming_after_interrupt_is_normal(self):
        """After interruption, subsequent chunks should be normal text."""
        self.detector.parse_streaming_increment("<think>")
        self.detector.parse_streaming_increment("reasoning<|tool_calls_section_begin|>")
        result = self.detector.parse_streaming_increment("<|tool_call_begin|>")
        self.assertEqual(result.reasoning_text, "")
        self.assertEqual(result.normal_text, "<|tool_call_begin|>")


class TestGlm45Detector(CustomTestCase):
    """Test cases for GLM45 detector with tool interruption support."""

    def setUp(self):
        self.detector = Glm45Detector()

    def test_init(self):
        """Test Glm45Detector initialization."""
        self.assertEqual(self.detector.think_start_token, "<think>")
        self.assertEqual(self.detector.think_end_token, "</think>")
        self.assertEqual(self.detector.tool_start_token, "<tool_call>")
        self.assertFalse(self.detector._in_reasoning)
        self.assertTrue(self.detector.stream_reasoning)

    def test_detect_and_parse_normal_reasoning(self):
        """Test parsing normal reasoning block without tool interruption."""
        text = "<think>Let me think about this step by step</think>The answer is 42."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "Let me think about this step by step")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_detect_and_parse_tool_interrupt(self):
        """
        Test parsing with tool interruption.

        GLM45 can interrupt reasoning with tool token (<tool_call>) without closing </think>.
        Should split at the first occurrence of tool_start_token using find().
        """
        text = "<think>I need to think<tool_call>tool call data"
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "I need to think")
        self.assertEqual(result.normal_text, "<tool_call>tool call data")

    def test_detect_and_parse_multiple_tool_calls_find(self):
        """
        Test that find() finds the FIRST occurrence of tool_start_token.

        If multiple tool calls exist in buffer, should split at the first one.
        """
        text = "<think>thinking<tool_call>first tool<tool_call>second tool<tool_call>final tool"
        result = self.detector.detect_and_parse(text)
        # Should split at the first <tool_call>
        self.assertEqual(result.reasoning_text, "thinking")
        self.assertEqual(
            result.normal_text,
            "<tool_call>first tool<tool_call>second tool<tool_call>final tool",
        )

    def test_detect_and_parse_truncated_reasoning(self):
        """
        Test truncated reasoning without tool or end tag.

        Should return all content as reasoning_text.
        """
        text = "<think>This is incomplete"
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "This is incomplete")
        self.assertEqual(result.normal_text, "")

    def test_detect_and_parse_normal_text_only(self):
        """Test parsing text without reasoning block."""
        text = "Just the answer without any reasoning."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")

    def test_streaming_normal_flow(self):
        """Test streaming with normal reasoning flow."""
        # Start reasoning
        result1 = self.detector.parse_streaming_increment("<think>")
        self.assertEqual(result1.normal_text, "")
        self.assertEqual(result1.reasoning_text, "")
        self.assertTrue(self.detector._in_reasoning)

        # Reasoning content
        result2 = self.detector.parse_streaming_increment("thinking...")
        self.assertEqual(result2.normal_text, "")
        self.assertEqual(result2.reasoning_text, "thinking...")

        # End reasoning
        result3 = self.detector.parse_streaming_increment("</think>answer")
        self.assertEqual(result3.normal_text, "answer")
        self.assertEqual(result3.reasoning_text, "")
        self.assertFalse(self.detector._in_reasoning)

    def test_streaming_tool_interrupt_split_tokens(self):
        """
        Test streaming with tool interruption where tool token is split across chunks.

        This tests the buffer prefix logic that prevents partial emission of tool token.
        """
        # Start reasoning
        self.detector.parse_streaming_increment("<think>")

        # Add reasoning
        result1 = self.detector.parse_streaming_increment("thinking")
        self.assertEqual(result1.reasoning_text, "thinking")

        # Send partial tool token (should be buffered, not emitted)
        result2 = self.detector.parse_streaming_increment("<tool_call>")
        # Tool token is in buffer, causing switch to normal mode
        self.assertEqual(result2.reasoning_text, "")
        self.assertEqual(result2.normal_text, "<tool_call>")
        self.assertFalse(self.detector._in_reasoning)

        # Send tool args
        result3 = self.detector.parse_streaming_increment("tool args")
        self.assertEqual(result3.reasoning_text, "")
        self.assertEqual(result3.normal_text, "tool args")

    def test_streaming_no_stream_reasoning(self):
        """Test streaming without stream_reasoning enabled."""
        detector = Glm45Detector(stream_reasoning=False)

        # Start reasoning
        detector.parse_streaming_increment("<think>")

        # Reasoning content is buffered and not returned yet
        result = detector.parse_streaming_increment("thinking")
        self.assertEqual(result.reasoning_text, "")
        self.assertEqual(result.normal_text, "")

        # Tool interruption should still work - flushes buffered reasoning.
        # Note: when stream_reasoning=False, the <think> tag is stripped from the
        # local `current_text` variable but NOT from `self._buffer` (which is never
        # cleared in the non-streaming path). So the flushed reasoning content
        # includes the raw <think> tag.
        result = detector.parse_streaming_increment("<tool_call>tool call")
        self.assertEqual(result.reasoning_text, "<think>thinking")
        self.assertEqual(result.normal_text, "<tool_call>tool call")

    def test_streaming_empty_reasoning_with_tool(self):
        """Test empty reasoning block followed by tool call."""
        result1 = self.detector.parse_streaming_increment("<think>")
        result2 = self.detector.parse_streaming_increment("<tool_call>tool call")
        self.assertEqual(result2.reasoning_text, "")
        self.assertEqual(result2.normal_text, "<tool_call>tool call")

    def test_forced_reasoning_mode(self):
        """Test GLM45 with force_reasoning=True."""
        detector = Glm45Detector(force_reasoning=True)

        # Without start token, should still be in reasoning mode
        text = "This is reasoning"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "This is reasoning")
        self.assertEqual(result.normal_text, "")

        # Tool interruption should work with forced reasoning
        text = "More reasoning<tool_call>tool call"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "More reasoning")
        self.assertEqual(result.normal_text, "<tool_call>tool call")


class TestHunyuanDetector(CustomTestCase):
    """Test cases for Hunyuan detector with tool interruption support."""

    def setUp(self):
        self.detector = HunyuanDetector()

    def test_init(self):
        """Test HunyuanDetector initialization."""
        self.assertEqual(self.detector.think_start_token, "<think>")
        self.assertEqual(self.detector.think_end_token, "</think>")
        self.assertEqual(self.detector.tool_start_token, "<tool_calls>")
        self.assertFalse(self.detector._in_reasoning)
        self.assertTrue(self.detector.stream_reasoning)

    def test_detect_and_parse_normal_reasoning(self):
        """Test parsing normal reasoning block without tool interruption."""
        text = "<think>Let me think about this</think>The answer is 42."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "Let me think about this")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_detect_and_parse_without_thinking(self):
        """Test parsing without thinking tokens (no_think mode)."""
        text = "Direct answer without thinking."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")

    def test_detect_and_parse_tool_interrupt(self):
        """Test parsing with tool call interruption during reasoning."""
        text = "<think>I need to check<tool_calls><tool_call>get_weather<tool_sep></tool_call></tool_calls>"
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "I need to check")
        self.assertIn("<tool_calls>", result.normal_text)

    def test_streaming_normal_reasoning(self):
        """Test streaming parse of normal reasoning block."""
        self.detector.parse_streaming_increment("<think>")
        result1 = self.detector.parse_streaming_increment("reasoning content")
        self.assertEqual(result1.reasoning_text, "reasoning content")

        result2 = self.detector.parse_streaming_increment("</think>answer")
        self.assertEqual(result2.normal_text, "answer")
        self.assertFalse(self.detector._in_reasoning)

    def test_streaming_tool_interrupt(self):
        """Test streaming parse interrupted by tool call section."""
        self.detector.parse_streaming_increment("<think>")
        result1 = self.detector.parse_streaming_increment("thinking")
        self.assertEqual(result1.reasoning_text, "thinking")

        result2 = self.detector.parse_streaming_increment("<tool_calls>")
        self.assertEqual(result2.reasoning_text, "")
        self.assertEqual(result2.normal_text, "<tool_calls>")
        self.assertFalse(self.detector._in_reasoning)

    def test_streaming_after_interrupt_is_normal(self):
        """After tool interruption, subsequent chunks should be normal text."""
        self.detector.parse_streaming_increment("<think>")
        self.detector.parse_streaming_increment("reasoning<tool_calls>")
        result = self.detector.parse_streaming_increment("<tool_call>data")
        self.assertEqual(result.reasoning_text, "")
        self.assertEqual(result.normal_text, "<tool_call>data")

    def test_reasoning_parser_integration(self):
        """Test Hunyuan through ReasoningParser API."""
        parser = ReasoningParser("hunyuan")
        self.assertIsInstance(parser.detector, HunyuanDetector)

        # Non-streaming
        reasoning, normal = parser.parse_non_stream(
            "<think>thinking<tool_calls><tool_call>func<tool_sep></tool_call></tool_calls>"
        )
        self.assertEqual(reasoning, "thinking")
        self.assertIn("<tool_calls>", normal)

    def test_reasoning_parser_streaming(self):
        """Test Hunyuan streaming through ReasoningParser API."""
        parser = ReasoningParser("hunyuan")
        chunks = ["<think>", "reasoning", "<tool_calls>", "<tool_call>func"]
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            reasoning, normal = parser.parse_stream_chunk(chunk)
            if reasoning:
                all_reasoning += reasoning
            if normal:
                all_normal += normal

        self.assertEqual(all_reasoning, "reasoning")
        self.assertIn("<tool_calls>", all_normal)


class TestNemotron3Detector(CustomTestCase):
    def setUp(self):
        self.detector = Nemotron3Detector()

    def test_init(self):
        """Test Nemotron3Detector initialization."""
        self.assertEqual(self.detector.think_start_token, "<think>")
        self.assertEqual(self.detector.think_end_token, "</think>")
        self.assertFalse(self.detector._in_reasoning)
        self.assertTrue(self.detector.stream_reasoning)
        self.assertFalse(self.detector._force_nonempty_content)

    def test_detect_and_parse_complete_reasoning(self):
        """Test parsing complete reasoning block."""
        text = "<think>Let me think about this</think>The answer is 42."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "Let me think about this")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_detect_and_parse_no_thinking(self):
        """Test parsing without thinking tokens."""
        text = "Direct answer without thinking."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")

    def test_detect_and_parse_reasoning_only(self):
        """Test parsing when output is all reasoning (no content after </think>)."""
        text = "<think>All reasoning, no answer</think>"
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "All reasoning, no answer")
        self.assertEqual(result.normal_text, "")

    def test_force_nonempty_content_swaps_when_no_normal_text(self):
        """Test force_nonempty_content swaps reasoning to content when content is empty."""
        detector = Nemotron3Detector(force_nonempty_content=True)
        text = "<think>All reasoning, no answer</think>"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, "All reasoning, no answer")
        self.assertEqual(result.reasoning_text, "")

    def test_force_nonempty_content_no_swap_when_normal_text_exists(self):
        """Test force_nonempty_content does not swap when content already exists."""
        detector = Nemotron3Detector(force_nonempty_content=True)
        text = "<think>Reasoning here</think>The answer is 42."
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "Reasoning here")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_force_nonempty_content_truncated_reasoning(self):
        """Test force_nonempty_content with truncated reasoning (no end token)."""
        detector = Nemotron3Detector(force_nonempty_content=True)
        text = "<think>Truncated reasoning without end token"
        result = detector.detect_and_parse(text)
        # Truncated reasoning has no normal_text, so swap should occur
        self.assertEqual(result.normal_text, "Truncated reasoning without end token")
        self.assertEqual(result.reasoning_text, "")

    def test_force_nonempty_content_no_thinking_tokens(self):
        """Test force_nonempty_content with plain text (no thinking tokens)."""
        detector = Nemotron3Detector(force_nonempty_content=True)
        text = "Plain text without any thinking."
        result = detector.detect_and_parse(text)
        # Normal text already exists, no swap needed
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")


class TestGemma4Detector(CustomTestCase):
    def setUp(self):
        self.detector = Gemma4Detector()

    def test_init(self):
        """Test Gemma4Detector initialization."""
        self.assertEqual(self.detector.think_start_token, "<|channel>")
        self.assertEqual(self.detector.think_end_token, "<channel|>")
        self.assertEqual(self.detector.think_start_self_label, "thought\n")
        self.assertFalse(self.detector._in_reasoning)
        self.assertTrue(self.detector.stream_reasoning)

    def test_detect_and_parse_complete_reasoning(self):
        """Test parsing complete Gemma4 reasoning block (think_start_self_label is stripped)."""
        text = "<|channel>thought\nLet me think about this<channel|>The answer is 42."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "Let me think about this")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_detect_and_parse_without_thinking(self):
        """Test parsing without thinking (enable_thinking=False case)."""
        text = "Direct answer without thinking."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.reasoning_text, "")

    def test_detect_and_parse_reasoning_only(self):
        """Test parsing when output is all reasoning (no end token yet)."""
        text = "<|channel>thought\nStill thinking..."
        result = self.detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "Still thinking...")
        self.assertEqual(result.normal_text, "")

    def test_streaming_complete_flow(self):
        """Test streaming parse of Gemma4 reasoning flow."""
        chunks = [
            "<|channel>",
            "thought\nreasoning content",
            "<channel|>",
            "final answer",
        ]
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk)
            all_reasoning += result.reasoning_text
            all_normal += result.normal_text
        self.assertIn("reasoning content", all_reasoning)
        self.assertIn("final answer", all_normal)

    def test_streaming_full_start_sequence(self):
        """Test streaming with the full start sequence (token + self_label)."""
        # Gemma4 start sequence is "<|channel>thought\n", not just "<|channel>"
        result = self.detector.parse_streaming_increment("<|channel>thought\n")
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "")
        self.assertTrue(self.detector._in_reasoning)

        result = self.detector.parse_streaming_increment("reasoning content")
        self.assertEqual(result.reasoning_text, "reasoning content")
        self.assertEqual(result.normal_text, "")

    def test_streaming_partial_start_buffered(self):
        """Test that partial start sequence is buffered."""
        # "<|channel>" alone is a prefix of "<|channel>thought\n", so it's buffered
        result = self.detector.parse_streaming_increment("<|channel>")
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "")

    def test_streaming_end_token_mid_chunk(self):
        """Test end token arriving in the same chunk as reasoning content."""
        self.detector.parse_streaming_increment("<|channel>thought\n")
        result = self.detector.parse_streaming_increment(
            "some reasoning<channel|>the answer"
        )
        self.assertEqual(result.reasoning_text, "some reasoning")
        self.assertEqual(result.normal_text, "the answer")
        self.assertFalse(self.detector._in_reasoning)

    def test_streaming_split_end_token(self):
        """Test end token split across two chunks."""
        self.detector.parse_streaming_increment("<|channel>thought\n")
        self.detector.parse_streaming_increment("reasoning content")

        result1 = self.detector.parse_streaming_increment("<chan")
        self.assertEqual(result1.normal_text, "")

        result2 = self.detector.parse_streaming_increment("nel|>final answer")
        self.assertFalse(self.detector._in_reasoning)
        self.assertIn("final answer", result2.normal_text)

    def test_streaming_self_label_split_across_chunks(self):
        """Test self_label ('thought\\n') arriving separately from start token."""
        result1 = self.detector.parse_streaming_increment("<|channel>")
        self.assertEqual(result1.reasoning_text, "")
        self.assertEqual(result1.normal_text, "")

        result2 = self.detector.parse_streaming_increment("thought\n")
        self.assertTrue(self.detector._in_reasoning)

        result3 = self.detector.parse_streaming_increment("reasoning here")
        self.assertEqual(result3.reasoning_text, "reasoning here")

    def test_streaming_force_reasoning(self):
        """Test streaming with force_reasoning=True (no start token needed)."""
        detector = Gemma4Detector(force_reasoning=True)

        result1 = detector.parse_streaming_increment("reasoning content")
        self.assertEqual(result1.reasoning_text, "reasoning content")
        self.assertEqual(result1.normal_text, "")

        result2 = detector.parse_streaming_increment("<channel|>the answer")
        self.assertFalse(detector._in_reasoning)
        self.assertIn("the answer", result2.normal_text)

    def test_streaming_multiple_reasoning_chunks(self):
        """Test reasoning content arriving in many small chunks."""
        self.detector.parse_streaming_increment("<|channel>thought\n")

        all_reasoning = ""
        for chunk in ["Think", "ing ", "step ", "by ", "step."]:
            result = self.detector.parse_streaming_increment(chunk)
            all_reasoning += result.reasoning_text
            self.assertEqual(result.normal_text, "")
        self.assertEqual(all_reasoning, "Thinking step by step.")

    def test_force_reasoning(self):
        """Test Gemma4Detector with force_reasoning=True."""
        detector = Gemma4Detector(force_reasoning=True)
        text = "This should be reasoning<channel|>The answer."
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "This should be reasoning")
        self.assertEqual(result.normal_text, "The answer.")


class TestReasoningParser(CustomTestCase):
    def test_init_valid_model(self):
        """Test initialization with valid model types."""
        parser = ReasoningParser("deepseek-r1")
        self.assertIsInstance(parser.detector, DeepSeekR1Detector)

        parser = ReasoningParser("qwen3")
        self.assertIsInstance(parser.detector, Qwen3Detector)

        parser = ReasoningParser("kimi")
        self.assertIsInstance(parser.detector, KimiDetector)

        parser = ReasoningParser("kimi_k2")
        self.assertIsInstance(parser.detector, KimiK2Detector)

        parser = ReasoningParser("glm45")
        self.assertIsInstance(parser.detector, Glm45Detector)

        parser = ReasoningParser("hunyuan")
        self.assertIsInstance(parser.detector, HunyuanDetector)

        parser = ReasoningParser("gemma4")
        self.assertIsInstance(parser.detector, Gemma4Detector)

    def test_init_invalid_model(self):
        """Test initialization with invalid model type."""
        with self.assertRaises(ValueError) as context:
            ReasoningParser("invalid-model")
        self.assertIn("Unsupported model type", str(context.exception))

    def test_init_no_model(self):
        """Test initialization without model type."""
        with self.assertRaises(ValueError) as context:
            ReasoningParser(None)
        self.assertEqual(str(context.exception), "Model type must be specified")

    def test_parse_non_stream(self):
        """Test non-streaming parsing."""
        parser = ReasoningParser("qwen3")
        reasoning, normal = parser.parse_non_stream(
            "<think>Let me think</think>The answer is 42."
        )
        self.assertEqual(reasoning, "Let me think")
        self.assertEqual(normal, "The answer is 42.")

    def test_parse_stream_chunk(self):
        """Test streaming chunk parsing."""
        parser = ReasoningParser("qwen3")

        # First chunk with start token
        reasoning, normal = parser.parse_stream_chunk("<think>")
        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "")

        # Second chunk with reasoning content
        reasoning, normal = parser.parse_stream_chunk("thinking...")
        self.assertEqual(reasoning, "thinking...")
        self.assertEqual(normal, "")

        # Third chunk with end token and normal text
        reasoning, normal = parser.parse_stream_chunk("</think>answer")
        self.assertEqual(reasoning, "")  # Buffer cleared when end token processed
        self.assertEqual(normal, "answer")

    def test_case_insensitive_model_type(self):
        """Test case insensitive model type matching."""
        parser1 = ReasoningParser("DeepSeek-R1")
        parser2 = ReasoningParser("QWEN3")
        parser3 = ReasoningParser("Kimi")

        self.assertIsInstance(parser1.detector, DeepSeekR1Detector)
        self.assertIsInstance(parser2.detector, Qwen3Detector)
        self.assertIsInstance(parser3.detector, KimiDetector)

    def test_stream_reasoning_parameter(self):
        """Test stream_reasoning parameter is passed correctly."""
        parser = ReasoningParser("qwen3", stream_reasoning=False)
        self.assertFalse(parser.detector.stream_reasoning)

        parser = ReasoningParser("qwen3", stream_reasoning=True)
        self.assertTrue(parser.detector.stream_reasoning)

    def test_glm45_tool_interruption(self):
        """Test GLM45 tool interruption through ReasoningParser API."""
        parser = ReasoningParser("glm45")

        # Non-streaming: tool interrupt
        reasoning, normal = parser.parse_non_stream(
            "<think>thinking<tool_call>tool call"
        )
        self.assertEqual(reasoning, "thinking")
        self.assertEqual(normal, "<tool_call>tool call")

        # Streaming: tool interrupt
        parser = ReasoningParser("glm45")
        chunks = ["<think>", "reasoning", "<tool_call>", "tool args"]
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            reasoning, normal = parser.parse_stream_chunk(chunk)
            if reasoning:
                all_reasoning += reasoning
            if normal:
                all_normal += normal

        self.assertEqual(all_reasoning, "reasoning")
        self.assertEqual(all_normal, "<tool_call>tool args")

    def test_kimik2_tool_interruption(self):
        """Test Kimi-K2 tool interruption through ReasoningParser API."""
        parser = ReasoningParser("kimi_k2")

        # Non-streaming: tool interrupt
        reasoning, normal = parser.parse_non_stream(
            "<think>thinking<|tool_calls_section_begin|><|tool_call_begin|>"
        )
        self.assertEqual(reasoning, "thinking")
        self.assertEqual(normal, "<|tool_calls_section_begin|><|tool_call_begin|>")

        # Streaming: tool interrupt
        parser = ReasoningParser("kimi_k2")
        chunks = [
            "<think>",
            "reasoning",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
        ]
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            reasoning, normal = parser.parse_stream_chunk(chunk)
            if reasoning:
                all_reasoning += reasoning
            if normal:
                all_normal += normal

        self.assertEqual(all_reasoning, "reasoning")
        self.assertEqual(all_normal, "<|tool_calls_section_begin|><|tool_call_begin|>")


class TestIntegrationScenarios(CustomTestCase):
    """Integration tests for realistic usage scenarios."""

    def test_deepseek_r1_complete_response(self):
        """Test complete DeepSeek-R1 response parsing."""
        parser = ReasoningParser("deepseek-r1")
        text = "I need to solve this step by step. First, I'll analyze the problem. The given equation is x + 2 = 5. To solve for x, I subtract 2 from both sides: x = 5 - 2 = 3.</think>The answer is x = 3."

        reasoning, normal = parser.parse_non_stream(text)
        self.assertIn("step by step", reasoning)
        self.assertIn(
            "= 3", reasoning
        )  # The reasoning contains "x = 5 - 2 = 3" which has "= 3"
        self.assertEqual(normal, "The answer is x = 3.")

    def test_qwen3_streaming_scenario(self):
        """Test Qwen3 streaming scenario."""
        parser = ReasoningParser("qwen3")

        chunks = [
            "<think>",
            "Let me analyze this problem.",
            " I need to consider multiple factors.",
            "</think>",
            "Based on my analysis, the solution is to use a different approach.",
        ]

        all_reasoning = ""
        all_normal = ""

        for chunk in chunks:
            reasoning, normal = parser.parse_stream_chunk(chunk)
            all_reasoning += reasoning
            all_normal += normal

        self.assertIn("analyze", all_reasoning)
        self.assertIn("multiple factors", all_reasoning)
        self.assertIn("different approach", all_normal)

    def test_kimi_streaming_scenario(self):
        """Test Kimi streaming scenario."""
        parser = ReasoningParser("kimi")
        chunks = [
            "◁thi",
            "nk▷",
            "Let me analyze this problem.",
            " I need to consider multiple factors.",
            "◁/th",
            "ink▷",
            "The answer is 42.",
        ]
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            reasoning, normal = parser.parse_stream_chunk(chunk)
            all_reasoning += reasoning
            all_normal += normal

        self.assertIn("analyze", all_reasoning)
        self.assertIn("multiple factors", all_reasoning)
        self.assertIn("42", all_normal)

    def test_gemma4_complete_response(self):
        """Test complete Gemma4 response parsing (think_start_self_label stripped)."""
        parser = ReasoningParser("gemma4")
        text = "<|channel>thought\nI need to solve x + 2 = 5. Subtracting 2: x = 3.<channel|>The answer is x = 3."
        reasoning, normal = parser.parse_non_stream(text)
        self.assertIn("x = 3", reasoning)
        self.assertNotIn("thought\n", reasoning)
        self.assertEqual(normal, "The answer is x = 3.")

    def test_gemma4_streaming_scenario(self):
        """Test Gemma4 streaming scenario."""
        parser = ReasoningParser("gemma4")
        chunks = [
            "<|channel>",
            "thought\nLet me analyze.",
            " Multiple factors.",
            "<channel|>",
            "The solution is 42.",
        ]
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            reasoning, normal = parser.parse_stream_chunk(chunk)
            all_reasoning += reasoning
            all_normal += normal
        self.assertIn("analyze", all_reasoning)
        self.assertIn("Multiple factors", all_reasoning)
        self.assertIn("42", all_normal)

    def test_empty_reasoning_blocks(self):
        """Test handling of empty reasoning blocks."""
        parser = ReasoningParser("qwen3")
        text = "<think></think>Just the answer."

        reasoning, normal = parser.parse_non_stream(text)
        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "Just the answer.")

    def test_qwen3_forced_reasoning_complete_response(self):
        """Test complete Qwen3-ForcedReasoning response parsing."""
        parser = ReasoningParser("qwen3", force_reasoning=True)
        text = "Let me solve this step by step. The equation is x + 2 = 5. Subtracting 2 from both sides gives x = 3.</think>The solution is x = 3."

        reasoning, normal = parser.parse_non_stream(text)
        self.assertIn("step by step", reasoning)
        self.assertIn("x = 3", reasoning)
        self.assertEqual(normal, "The solution is x = 3.")

    def test_qwen3_forced_reasoning_streaming_scenario(self):
        """Test Qwen3-ForcedReasoning streaming scenario."""
        parser = ReasoningParser("qwen3", force_reasoning=True)

        chunks = [
            "I need to analyze",
            " this problem carefully.",
            " Let me break it down.",
            "</think>",
            "The final answer is 42.",
        ]

        all_reasoning = ""
        all_normal = ""

        for chunk in chunks:
            reasoning, normal = parser.parse_stream_chunk(chunk)
            all_reasoning += reasoning
            all_normal += normal

        self.assertIn("analyze", all_reasoning)
        self.assertIn("break it down", all_reasoning)
        self.assertIn("final answer", all_normal)


class TestBufferLossBugFix(CustomTestCase):
    """Test cases for the buffer loss bug fix in parse_streaming_increment."""

    def test_partial_end_tag_buffer_loss_bug(self):
        """
        Test the bug where partial end tag fragments are lost when followed by normal text.

        Bug scenario:
        1. _in_reasoning is False
        2. new_text is "</" (part of closing thinking tag)
        3. Fragment is stored in buffer and empty string is returned
        4. Next step: new_text is "answer", _in_reasoning still False
        5. Buffer is cleared and "answer" is returned directly
        6. The "</" from previous step is lost

        This test verifies the fix where the return was changed from:
        return StreamingParseResult(normal_text=new_text)
        to:
        return StreamingParseResult(normal_text=current_text)
        """
        detector = BaseReasoningFormatDetector("<think>", "</think>")

        # Step 1: Send partial end tag when not in reasoning mode
        # This should be buffered since it could be start of "</think>"
        result1 = detector.parse_streaming_increment("</")
        self.assertEqual(result1.normal_text, "")
        self.assertEqual(result1.reasoning_text, "")

        # Step 2: Send normal text that doesn't complete the end tag
        # Before fix: would return only "answer", losing the "</"
        # After fix: should return the complete buffered content "</answer"
        result2 = detector.parse_streaming_increment("answer")
        self.assertEqual(result2.normal_text, "</answer")
        self.assertEqual(result2.reasoning_text, "")

    def test_partial_start_tag_buffer_preservation(self):
        """
        Test that partial start tag fragments are properly preserved.
        """
        detector = BaseReasoningFormatDetector("<think>", "</think>")

        # Send partial start tag
        result1 = detector.parse_streaming_increment("<th")
        self.assertEqual(result1.normal_text, "")
        self.assertEqual(result1.reasoning_text, "")

        # Complete with non-matching text
        result2 = detector.parse_streaming_increment("is is text")
        self.assertEqual(result2.normal_text, "<this is text")
        self.assertEqual(result2.reasoning_text, "")

    def test_partial_end_tag_in_reasoning_mode(self):
        """
        Test partial end tag handling when already in reasoning mode.
        """
        detector = BaseReasoningFormatDetector("<think>", "</think>")

        # Enter reasoning mode
        detector.parse_streaming_increment("<think>")
        detector.parse_streaming_increment("some reasoning")

        # Send partial end tag
        result1 = detector.parse_streaming_increment("</")
        self.assertEqual(result1.normal_text, "")
        self.assertEqual(result1.reasoning_text, "")

        # Complete the end tag with normal text
        result2 = detector.parse_streaming_increment("think>normal text")
        self.assertEqual(result2.normal_text, "normal text")
        # The reasoning text should be empty since buffer was cleared when end tag was processed
        self.assertEqual(result2.reasoning_text, "")

    def test_multiple_partial_fragments(self):
        """
        Test handling of multiple partial fragments that don't match any tokens.
        """
        detector = BaseReasoningFormatDetector("<think>", "</think>")

        # Send multiple partial fragments
        result1 = detector.parse_streaming_increment("<")
        self.assertEqual(result1.normal_text, "")
        self.assertEqual(result1.reasoning_text, "")

        result2 = detector.parse_streaming_increment("/")
        self.assertEqual(result2.normal_text, "")
        self.assertEqual(result2.reasoning_text, "")

        result3 = detector.parse_streaming_increment("random>")
        self.assertEqual(result3.normal_text, "</random>")
        self.assertEqual(result3.reasoning_text, "")

    def test_edge_case_exact_token_match(self):
        """
        Test edge case where buffer content exactly matches a token.
        """
        detector = BaseReasoningFormatDetector("<think>", "</think>")

        # Build up the exact start token character by character
        detector.parse_streaming_increment("<")
        detector.parse_streaming_increment("t")
        detector.parse_streaming_increment("h")
        detector.parse_streaming_increment("i")
        detector.parse_streaming_increment("n")
        result = detector.parse_streaming_increment("k>")

        # Should enter reasoning mode
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.reasoning_text, "")
        self.assertTrue(detector._in_reasoning)
        self.assertTrue(detector.stripped_think_start)


class TestGptOssDetector(CustomTestCase):
    """Test cases for GptOssDetector which delegates to HarmonyParser."""

    def setUp(self):
        from sglang.srt.parser.reasoning_parser import GptOssDetector

        self.detector = GptOssDetector()

    def test_detect_and_parse_with_analysis_and_final(self):
        """Test one-shot parsing with analysis (reasoning) and final (normal) blocks."""
        text = "<|start|><|channel|>analysis<|message|>thinking hard<|end|><|channel|>final<|message|>the answer<|end|>"
        result = self.detector.detect_and_parse(text)
        self.assertIn("thinking hard", result.reasoning_text)
        self.assertIn("the answer", result.normal_text)

    def test_detect_and_parse_normal_only(self):
        """Test one-shot parsing with only final block."""
        text = "<|start|><|channel|>final<|message|>just the answer<|end|>"
        result = self.detector.detect_and_parse(text)
        self.assertIn("just the answer", result.normal_text)

    def test_streaming_analysis_then_final(self):
        """Test streaming parse across multiple chunks."""
        chunks = [
            "<|start|><|channel|>analysis<|message|>",
            "reasoning part",
            "<|end|>",
            "<|channel|>final<|message|>answer",
            "<|end|>",
        ]
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk)
            all_reasoning += result.reasoning_text
            all_normal += result.normal_text
        self.assertIn("reasoning part", all_reasoning)
        self.assertIn("answer", all_normal)

    def test_streaming_with_tool_call(self):
        """Test streaming parse with tool call events."""
        text = "<|start|><|channel|>analysis<|message|>think<|end|><|call|>tool_data<|return|><|channel|>final<|message|>result<|end|>"
        result = self.detector.detect_and_parse(text)
        self.assertIn("think", result.reasoning_text)
        self.assertIn("result", result.normal_text)


class TestMiniMaxAppendThinkDetector(CustomTestCase):
    """Test cases for MiniMaxAppendThinkDetector."""

    def setUp(self):
        from sglang.srt.parser.reasoning_parser import MiniMaxAppendThinkDetector

        self.detector = MiniMaxAppendThinkDetector()

    def test_detect_and_parse_prepends_think(self):
        """Test that detect_and_parse prepends <think> to the text."""
        result = self.detector.detect_and_parse("Hello world")
        self.assertEqual(result.normal_text, "<think>Hello world")

    def test_streaming_first_chunk_prepends_think(self):
        """Test that first streaming chunk gets <think> prepended."""
        result = self.detector.parse_streaming_increment("First chunk")
        self.assertEqual(result.normal_text, "<think>First chunk")

    def test_streaming_second_chunk_no_prepend(self):
        """Test that subsequent streaming chunks are passed through."""
        self.detector.parse_streaming_increment("First")
        result = self.detector.parse_streaming_increment("Second")
        self.assertEqual(result.normal_text, "Second")


class TestReasoningParserAdvanced(CustomTestCase):
    """Additional tests for ReasoningParser init edge cases."""

    def test_gpt_oss_model_type(self):
        """Test that gpt-oss model type creates GptOssDetector."""
        from sglang.srt.parser.reasoning_parser import GptOssDetector

        parser = ReasoningParser("gpt-oss")
        self.assertIsInstance(parser.detector, GptOssDetector)

    def test_minimax_append_think_model_type(self):
        """Test that minimax-append-think creates MiniMaxAppendThinkDetector."""
        from sglang.srt.parser.reasoning_parser import MiniMaxAppendThinkDetector

        parser = ReasoningParser("minimax-append-think")
        self.assertIsInstance(parser.detector, MiniMaxAppendThinkDetector)

    def test_qwen3_thinking_forces_reasoning(self):
        """Test that qwen3-thinking model type forces reasoning mode."""
        parser = ReasoningParser("qwen3-thinking")
        self.assertTrue(parser.detector._in_reasoning)

    def test_minimax_forces_reasoning(self):
        """Test that minimax model type forces reasoning mode.

        minimax maps to Qwen3Detector but ReasoningParser overrides
        force_reasoning=True, unlike the default Qwen3Detector behavior.
        """
        parser = ReasoningParser("minimax")
        self.assertIsInstance(parser.detector, Qwen3Detector)
        self.assertTrue(parser.detector._in_reasoning)

    def test_detector_map_aliases(self):
        """Test that all DetectorMap alias keys create the correct detector type."""
        # These are aliases that map to existing detector classes
        alias_tests = {
            "deepseek-v3": Qwen3Detector,
            "step3": DeepSeekR1Detector,
            "step3p5": DeepSeekR1Detector,
            "interns1": Qwen3Detector,
        }
        for model_type, expected_class in alias_tests.items():
            parser = ReasoningParser(model_type)
            self.assertIsInstance(
                parser.detector,
                expected_class,
                f"{model_type} should create {expected_class.__name__}",
            )

    def test_continue_final_message_with_request(self):
        """Test continue_final_message passes previous content to detector."""
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionMessageGenericParam,
            ChatCompletionMessageUserParam,
            ChatCompletionRequest,
        )

        request = ChatCompletionRequest(
            model="test",
            messages=[
                ChatCompletionMessageUserParam(role="user", content="Hi"),
                ChatCompletionMessageGenericParam(
                    role="assistant", content="Let me think..."
                ),
            ],
            continue_final_message=True,
        )
        parser = ReasoningParser("qwen3", request=request)
        self.assertTrue(parser.detector.continue_final_message)

    def test_force_nonempty_content_via_chat_template_kwargs(self):
        """Test that force_nonempty_content is passed via chat_template_kwargs."""
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionMessageUserParam,
            ChatCompletionRequest,
        )

        request = ChatCompletionRequest(
            model="test",
            messages=[
                ChatCompletionMessageUserParam(role="user", content="Hi"),
            ],
            chat_template_kwargs={"force_nonempty_content": True},
        )
        parser = ReasoningParser("nemotron_3", request=request)
        self.assertTrue(parser.detector._force_nonempty_content)


class TestContinueFinalMessage(CustomTestCase):
    """Test continue_final_message mode for BaseReasoningFormatDetector."""

    def test_continue_with_think_start_in_previous(self):
        """Test that previous_content with <think> sets _in_reasoning=True."""
        detector = BaseReasoningFormatDetector(
            "<think>",
            "</think>",
            force_reasoning=False,
            continue_final_message=True,
            previous_content="<think>some reasoning",
        )
        self.assertTrue(detector._in_reasoning)
        self.assertEqual(detector.previous_count, len("<think>some reasoning"))

    def test_continue_with_think_end_in_previous(self):
        """Test that previous_content with </think> sets _in_reasoning=False."""
        detector = BaseReasoningFormatDetector(
            "<think>",
            "</think>",
            force_reasoning=True,
            continue_final_message=True,
            previous_content="<think>done</think>normal",
        )
        # think_end_token in previous → _in_reasoning = False
        self.assertFalse(detector._in_reasoning)

    def test_continue_detect_parse_with_end_in_previous(self):
        """Test detect_and_parse when think_end_token is in previous_content only.
        This covers the branch where think_end is NOT in current text
        but IS in previous_content, so output is treated as normal_text."""
        detector = BaseReasoningFormatDetector(
            "<think>",
            "</think>",
            force_reasoning=True,
            continue_final_message=True,
            previous_content="<think>reasoning</think>",
        )
        # _in_reasoning is False (think_end in previous)
        # But force_reasoning was True → detect_and_parse still enters the
        # reasoning path because think_start is in previous_content.
        # However, since _in_reasoning=False and no think_start in new text,
        # it returns normal_text directly.
        result = detector.detect_and_parse("new content here")
        self.assertEqual(result.normal_text, "new content here")

    def test_continue_end_in_previous_new_text_has_start_but_no_end(self):
        """Test: think_end in previous, new text has think_start but no think_end.
        This produces: in_reasoning=True (from think_start in text),
        think_end NOT in processed_text, think_end IN previous_content,
        so it falls through to the else branch that returns normal_text."""
        detector = BaseReasoningFormatDetector(
            "<think>",
            "</think>",
            force_reasoning=False,
            continue_final_message=True,
            previous_content="earlier <think>old</think>old answer",
        )
        # _in_reasoning = False (think_end in previous overrides)
        self.assertFalse(detector._in_reasoning)
        # New text has <think> (triggers in_reasoning) but no </think>
        # think_end IS in previous_content → skips the truncated-reasoning branch
        # think_end NOT in processed_text → falls to else that returns normal_text
        result = detector.detect_and_parse("<think>continuing reasoning")
        self.assertEqual(result.normal_text, "continuing reasoning")
        self.assertEqual(result.reasoning_text, "")

    def test_continue_detect_parse_think_start_in_prev_but_end_also_in_prev(self):
        """Test detect_and_parse where both think tokens are in previous,
        and new text contains <think> to re-enter reasoning."""
        detector = BaseReasoningFormatDetector(
            "<think>",
            "</think>",
            force_reasoning=False,
            continue_final_message=True,
            previous_content="<think>old reasoning</think>old answer",
        )
        # _in_reasoning = False (end token in previous overrides start)
        self.assertFalse(detector._in_reasoning)
        # New text starts a fresh reasoning block
        result = detector.detect_and_parse("<think>new reasoning</think>new answer")
        self.assertEqual(result.reasoning_text, "new reasoning")
        self.assertEqual(result.normal_text, "new answer")

    def test_streaming_returns_empty_when_in_reasoning_and_end_buffered(self):
        """Test that streaming returns empty when buffer could be partial end token."""
        detector = BaseReasoningFormatDetector(
            "<think>", "</think>", force_reasoning=True, stream_reasoning=True
        )
        # In reasoning mode, send partial end token
        result = detector.parse_streaming_increment("</")
        self.assertEqual(result.reasoning_text, "")
        self.assertEqual(result.normal_text, "")
        # This goes through the path where _in_reasoning is True but buffer
        # is a prefix of think_end_token → returns empty


class TestGptOssDetectorToolCall(CustomTestCase):
    """Test GptOssDetector tool_call raw_text handling."""

    def test_detect_and_parse_tool_call_raw_text(self):
        """Test that tool_call events use raw_text when available."""
        from sglang.srt.parser.reasoning_parser import GptOssDetector

        detector = GptOssDetector()
        # Sequence with CALL...RETURN that produces tool_call events with raw_text
        text = (
            "<|start|><|channel|>analysis<|message|>think<|end|>"
            "<|call|>function_data<|return|>"
            "<|channel|>final<|message|>result<|end|>"
        )
        result = detector.detect_and_parse(text)
        self.assertIn("think", result.reasoning_text)
        # Tool call raw_text and/or final result should be in normal_text
        self.assertIn("result", result.normal_text)

    def test_streaming_tool_call_raw_text(self):
        """Test streaming parse with tool_call events preserving raw_text."""
        from sglang.srt.parser.reasoning_parser import GptOssDetector

        detector = GptOssDetector()
        chunks = [
            "<|start|><|channel|>analysis<|message|>reason<|end|>",
            "<|call|>tool_payload<|return|>",
            "<|channel|>final<|message|>done<|end|>",
        ]
        all_reasoning = ""
        all_normal = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk)
            all_reasoning += result.reasoning_text
            all_normal += result.normal_text
        self.assertIn("reason", all_reasoning)
        self.assertIn("done", all_normal)


class TestPoolsideV1Registered(CustomTestCase):
    """poolside_v1 (Laguna-XS.2) reuses the Qwen3 `<think>...</think>` envelope.
    Request dispatch differs (Mimo-style explicit `enable_thinking=True`,
    asserted in test_serving_chat.py), driven by
    `reasoning_default = "explicit_enable_thinking"` on the detector."""

    def test_registered_to_qwen3_subclass(self):
        cls = ReasoningParser.DetectorMap["poolside_v1"]
        self.assertTrue(issubclass(cls, Qwen3Detector))

    def test_explicit_enable_thinking_default(self):
        rp = ReasoningParser("poolside_v1", stream_reasoning=True)
        self.assertEqual(rp.detector.reasoning_default, "explicit_enable_thinking")
        self.assertTrue(rp.detector.thinks_internally)


if __name__ == "__main__":
    unittest.main()
