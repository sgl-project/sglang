import unittest

from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    DeepSeekR1Detector,
    KimiDetector,
    Qwen3Detector,
    ReasoningParser,
    StreamingParseResult,
)
from sglang.test.test_utils import CustomTestCase


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


class TestReasoningParser(CustomTestCase):
    def test_init_valid_model(self):
        """Test initialization with valid model types."""
        parser = ReasoningParser("deepseek-r1")
        self.assertIsInstance(parser.detector, DeepSeekR1Detector)

        parser = ReasoningParser("qwen3")
        self.assertIsInstance(parser.detector, Qwen3Detector)

        parser = ReasoningParser("kimi")
        self.assertIsInstance(parser.detector, KimiDetector)

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

        This test verifies the fix where line 108 was changed from:
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


if __name__ == "__main__":
    unittest.main()
