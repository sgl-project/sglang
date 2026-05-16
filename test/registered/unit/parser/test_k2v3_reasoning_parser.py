"""Unit tests for K2-v3 reasoning detector."""

import unittest

from sglang.srt.parser.reasoning_parser import K2V3Detector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestK2V3DetectorTokenSelection(CustomTestCase):
    """K2-v3 selects token pair based on reasoning_effort."""

    def test_high_effort_uses_think_tokens(self):
        detector = K2V3Detector(reasoning_effort="high")
        self.assertEqual(detector.think_start_token, "<think>")
        self.assertEqual(detector.think_end_token, "</think>")

    def test_medium_effort_uses_think_fast_tokens(self):
        detector = K2V3Detector(reasoning_effort="medium")
        self.assertEqual(detector.think_start_token, "<think_fast>")
        self.assertEqual(detector.think_end_token, "</think_fast>")

    def test_low_effort_uses_think_faster_tokens(self):
        detector = K2V3Detector(reasoning_effort="low")
        self.assertEqual(detector.think_start_token, "<think_faster>")
        self.assertEqual(detector.think_end_token, "</think_faster>")

    def test_none_effort_maps_to_high(self):
        detector = K2V3Detector(reasoning_effort="none")
        self.assertEqual(detector.think_start_token, "<think>")

    def test_unknown_effort_maps_to_high(self):
        detector = K2V3Detector(reasoning_effort="not-a-thing")
        self.assertEqual(detector.think_start_token, "<think>")

    def test_default_effort_is_high(self):
        detector = K2V3Detector()
        self.assertEqual(detector.think_start_token, "<think>")

    def test_force_reasoning_false_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires force_reasoning=True"):
            K2V3Detector(force_reasoning=False)


class TestK2V3DetectorParsing(CustomTestCase):
    """K2-v3 inherits the standard <think>...</think> state machine."""

    def test_medium_effort_parses_end_only_output(self):
        detector = K2V3Detector(reasoning_effort="medium")
        text = "reasoning here</think_fast>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "reasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_medium_effort_parses_think_fast_block(self):
        detector = K2V3Detector(reasoning_effort="medium")
        text = "<think_fast>reasoning here</think_fast>final answer"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "reasoning here")
        self.assertEqual(result.normal_text, "final answer")

    def test_low_effort_parses_think_faster_block(self):
        detector = K2V3Detector(reasoning_effort="low")
        text = "<think_faster>r</think_faster>a"
        result = detector.detect_and_parse(text)
        self.assertEqual(result.reasoning_text, "r")
        self.assertEqual(result.normal_text, "a")

    def test_streaming_medium_effort(self):
        detector = K2V3Detector(reasoning_effort="medium", force_reasoning=True)
        r1 = detector.parse_streaming_increment("partial reason")
        self.assertEqual(r1.reasoning_text, "partial reason")
        r2 = detector.parse_streaming_increment("ing</think_fast>answer")
        self.assertEqual(r2.reasoning_text, "ing")
        self.assertEqual(r2.normal_text, "answer")


class TestK2V3ParserIntegration(CustomTestCase):
    """ReasoningParser wires k2_v3 model_type to K2V3Detector with correct effort."""

    def test_parser_routes_to_k2v3_detector(self):
        parser = ReasoningParser(model_type="k2_v3")
        self.assertIsInstance(parser.detector, K2V3Detector)
        self.assertEqual(parser.detector.think_start_token, "<think>")

    def test_parser_rejects_force_reasoning_false(self):
        with self.assertRaisesRegex(ValueError, "requires force_reasoning=True"):
            ReasoningParser(model_type="k2_v3", force_reasoning=False)

    def test_parser_allows_force_reasoning_false_for_non_k2v3(self):
        parser = ReasoningParser(model_type="qwen3", force_reasoning=False)
        self.assertEqual(parser.detector.think_start_token, "<think>")

    def test_parser_forwards_reasoning_effort_medium(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="k2-v3",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        parser = ReasoningParser(model_type="k2_v3", request=req)
        self.assertEqual(parser.detector.think_start_token, "<think_fast>")
        self.assertEqual(parser.detector.think_end_token, "</think_fast>")

    def test_parser_forwards_reasoning_effort_low(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="k2-v3",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        parser = ReasoningParser(model_type="k2_v3", request=req)
        self.assertEqual(parser.detector.think_start_token, "<think_faster>")

    def test_parser_reads_top_level_reasoning_effort(self):
        """serving_chat.py pops reasoning_effort out of chat_template_kwargs
        and moves it to request.reasoning_effort before reaching the parser.
        The parser must read from the top-level field, not just the kwargs."""
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="k2-v3",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
        )
        parser = ReasoningParser(model_type="k2_v3", request=req)
        self.assertEqual(parser.detector.think_start_token, "<think_fast>")
        self.assertEqual(parser.detector.think_end_token, "</think_fast>")

    def test_parser_ignores_reasoning_effort_for_non_k2v3(self):
        """reasoning_effort kwarg must NOT be forwarded to Qwen3Detector.

        Qwen3Detector.__init__ does not accept reasoning_effort. If the
        ReasoningParser guard is ever dropped, building this parser would
        raise TypeError. Test the absence-of-leak property explicitly, not
        just the resulting token value.
        """
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        try:
            parser = ReasoningParser(model_type="qwen3", request=req)
        except TypeError as e:
            self.fail(
                "reasoning_effort was incorrectly forwarded to "
                f"Qwen3Detector: {e}"
            )
        self.assertEqual(parser.detector.think_start_token, "<think>")


if __name__ == "__main__":
    unittest.main()
