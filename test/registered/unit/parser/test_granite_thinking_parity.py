"""Parity tests for Granite thinking parser.

Verifies that the SGLang built-in GraniteThinkingDetector produces the exact
same (reasoning, content) outputs as the HF plugin granite_thinking_parser.py
shipped with ibm-granite/granite-4.2-30b, for both streaming and non-streaming.

Expected values were verified against the live HF plugin with the real Granite
4.2 tokenizer. No GPU or external dependencies needed to run these tests.
"""

import unittest

from sglang.srt.parser.reasoning_parser import GraniteThinkingDetector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (id, text, enable_thinking, force_nonempty_content,
#  expected_reasoning, expected_content)
# All cases produce identical results in both streaming and non-streaming.
PARITY_CASES = [
    (
        "leading_nl",
        "<think>reasoning</think>\nHello",
        True,
        False,
        "reasoning",
        "Hello",
    ),
    (
        "multi_nl",
        "<think>reasoning</think>\n\n\nHello",
        True,
        False,
        "reasoning",
        "Hello",
    ),
    ("simple_no_nl", "<think>r</think>c", True, False, "r", "c"),
    ("reasoning_only", "<think>reasoning</think>", True, False, "reasoning", ""),
    ("ws_only_content", "<think>reasoning</think>\n\n", True, False, "reasoning", ""),
    (
        "force_swap_empty",
        "<think>reasoning</think>\n\n",
        True,
        True,
        "reasoning",
        "",
    ),
    (
        "force_keep_real",
        "<think>reasoning</think>\nreal answer",
        True,
        True,
        "reasoning",
        "real answer",
    ),
    ("truncated", "<think>truncated reasoning", True, False, "truncated reasoning", ""),
    (
        "multiline",
        "<think>line1\nline2</think>\nresult1\nresult2",
        True,
        False,
        "line1\nline2",
        "result1\nresult2",
    ),
    ("empty_block", "<think></think>content", True, False, "", "content"),
    ("empty_block_nl", "<think></think>\ncontent", True, False, "", "content"),
    ("plain_force", "This is plain content", False, False, "", "This is plain content"),
    ("empty_reasoning_nl", "<think></think>\n\nHello", True, False, "", "Hello"),
    (
        "nested_nl_in_reasoning",
        "<think>think\n\n</think>\nresult",
        True,
        False,
        "think\n\n",
        "result",
    ),
    ("only_nl_content", "<think>r</think>\n", True, False, "r", ""),
    (
        "force_w_content",
        "<think>r</think>\n\nanswer",
        True,
        True,
        "r",
        "answer",
    ),
    ("enable_false_tags", "<think></think>content", False, False, "", "content"),
    (
        "tool_after_think",
        "<think>reasoning</think>\n<tool_call>fn</tool_call>",
        True,
        False,
        "reasoning",
        "<tool_call>fn</tool_call>",
    ),
]


def _run_sglang_non_streaming(text, enable_thinking, force_nonempty_content):
    det = GraniteThinkingDetector(
        force_nonempty_content=(force_nonempty_content or (not enable_thinking))
    )
    result = det.detect_and_parse(text)
    return result.reasoning_text, result.normal_text


def _run_sglang_streaming(text, enable_thinking, force_nonempty_content):
    det = GraniteThinkingDetector(
        force_nonempty_content=(force_nonempty_content or (not enable_thinking))
    )
    all_r, all_c = "", ""
    for ch in text:
        res = det.parse_streaming_increment(ch)
        all_r += res.reasoning_text
        all_c += res.normal_text
    end = det.finish()
    all_r += end.reasoning_text
    all_c += end.normal_text
    return all_r, all_c


class TestGraniteThinkingParityNonStreaming(CustomTestCase):
    """Non-streaming parity against hardcoded expected values."""

    def test_all_cases(self):
        for case_id, text, et, fnc, exp_r, exp_c in PARITY_CASES:
            with self.subTest(case_id=case_id):
                r, c = _run_sglang_non_streaming(text, et, fnc)
                self.assertEqual(r, exp_r, f"[{case_id}] reasoning")
                self.assertEqual(c, exp_c, f"[{case_id}] content")

    def test_truncated_force_nonempty(self):
        """Truncated reasoning (no </think>) with force_nonempty_content swaps."""
        r, c = _run_sglang_non_streaming("<think>truncated reasoning", True, True)
        self.assertEqual(r, "")
        self.assertEqual(c, "truncated reasoning")


class TestGraniteThinkingParityStreaming(CustomTestCase):
    """Streaming parity against hardcoded expected values."""

    def test_all_cases(self):
        for case_id, text, et, fnc, exp_r, exp_c in PARITY_CASES:
            with self.subTest(case_id=case_id):
                r, c = _run_sglang_streaming(text, et, fnc)
                self.assertEqual(r, exp_r, f"[{case_id}] streaming reasoning")
                self.assertEqual(c, exp_c, f"[{case_id}] streaming content")

    def test_truncated_force_nonempty(self):
        """Truncated reasoning with force_nonempty_content in streaming:
        reasoning is emitted incrementally, then finish() reclassifies the
        accumulated reasoning as content. Both fields contain the text."""
        r, c = _run_sglang_streaming("<think>truncated reasoning", True, True)
        self.assertEqual(r, "truncated reasoning")
        self.assertEqual(c, "truncated reasoning")


class TestGraniteThinkingStreamingMatchesNonStreaming(CustomTestCase):
    """Streaming and non-streaming produce the same outputs."""

    def test_all_cases(self):
        for case_id, text, et, fnc, _, _ in PARITY_CASES:
            with self.subTest(case_id=case_id):
                ns_r, ns_c = _run_sglang_non_streaming(text, et, fnc)
                s_r, s_c = _run_sglang_streaming(text, et, fnc)
                self.assertEqual(ns_r, s_r, f"[{case_id}] reasoning mismatch")
                self.assertEqual(ns_c, s_c, f"[{case_id}] content mismatch")


class TestGraniteThinkingReasoningParserIntegration(CustomTestCase):
    """ReasoningParser('granite_thinking_parser') routes correctly."""

    def test_non_stream(self):
        parser = ReasoningParser("granite_thinking_parser")
        r, c = parser.parse_non_stream("<think>thinking</think>\nThe answer is 42.")
        self.assertEqual(r, "thinking")
        self.assertEqual(c, "The answer is 42.")

    def test_stream(self):
        parser = ReasoningParser("granite_thinking_parser")
        all_r, all_c = "", ""
        for chunk in ["<think>", "thinking", "</think>", "\n", "The answer"]:
            r, c = parser.parse_stream_chunk(chunk)
            if r:
                all_r += r
            if c:
                all_c += c
        r, c = parser.parse_stream_end()
        if r:
            all_r += r
        if c:
            all_c += c
        self.assertEqual(all_r, "thinking")
        self.assertEqual(all_c, "The answer")

    def test_enable_thinking_false_maps_to_force_nonempty(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="granite-4.2-30b",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        parser = ReasoningParser("granite_thinking_parser", request=request)
        self.assertTrue(parser.detector._force_nonempty_content)


if __name__ == "__main__":
    unittest.main()
