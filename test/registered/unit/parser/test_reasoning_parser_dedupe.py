"""Unit tests for the leading-think_end-token stripping behaviour added to
BaseReasoningFormatDetector. This protects against duplicate `</think>` markers
produced by speculative decoding or by the reasoning-EOS redirect logit
processor triggering more than once.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest

from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    DeepSeekR1Detector,
    Qwen3Detector,
)
from sglang.test.test_utils import CustomTestCase


def _make_qwen3():
    return Qwen3Detector(stream_reasoning=True, force_reasoning=False)


def _make_qwen3_force():
    # force_reasoning=True so first chunk of stream is treated as in-reasoning
    return Qwen3Detector(stream_reasoning=True, force_reasoning=True)


class TestStripLeadingThinkEndHelper(CustomTestCase):
    def setUp(self):
        self.detector = BaseReasoningFormatDetector(
            think_start_token="<think>",
            think_end_token="</think>",
        )

    def test_no_change_when_no_leading_think_end(self):
        self.assertEqual(
            self.detector._strip_leading_think_end_tokens("hello world"),
            "hello world",
        )

    def test_strips_single_leading_think_end(self):
        self.assertEqual(
            self.detector._strip_leading_think_end_tokens("</think>answer"),
            "answer",
        )

    def test_strips_repeated_leading_think_end(self):
        self.assertEqual(
            self.detector._strip_leading_think_end_tokens("</think></think>answer"),
            "answer",
        )

    def test_strips_with_whitespace_between(self):
        self.assertEqual(
            self.detector._strip_leading_think_end_tokens(
                "  </think>\n</think>  real"
            ),
            "real",
        )

    def test_does_not_strip_mid_text_think_end(self):
        # `</think>` appearing in the middle is preserved (rare, but allowed).
        self.assertEqual(
            self.detector._strip_leading_think_end_tokens(
                "answer with </think> inside"
            ),
            "answer with </think> inside",
        )

    def test_handles_empty_text(self):
        self.assertEqual(self.detector._strip_leading_think_end_tokens(""), "")


class TestNonStreamingDedupe(CustomTestCase):
    def test_single_think_end_is_unchanged(self):
        d = _make_qwen3()
        result = d.detect_and_parse("<think>reasoning</think>real answer")
        self.assertEqual(result.reasoning_text, "reasoning")
        self.assertEqual(result.normal_text, "real answer")

    def test_duplicate_think_end_is_stripped(self):
        d = _make_qwen3()
        result = d.detect_and_parse("<think>reasoning</think></think>real answer")
        self.assertEqual(result.reasoning_text, "reasoning")
        self.assertEqual(result.normal_text, "real answer")

    def test_triple_think_end_is_stripped(self):
        d = _make_qwen3()
        result = d.detect_and_parse(
            "<think>reasoning</think></think>\n</think>real answer"
        )
        self.assertEqual(result.reasoning_text, "reasoning")
        self.assertEqual(result.normal_text, "real answer")

    def test_deepseek_r1_force_reasoning_dedupe(self):
        # DeepSeek-R1 always force_reasoning=True; raw stream may not have
        # think_start prepended.
        d = DeepSeekR1Detector(stream_reasoning=True)
        result = d.detect_and_parse("reasoning</think></think>answer")
        self.assertEqual(result.reasoning_text, "reasoning")
        self.assertEqual(result.normal_text, "answer")


class TestStreamingDedupe(CustomTestCase):
    def _drain(self, detector, chunks):
        reasoning, normal = [], []
        for chunk in chunks:
            r = detector.parse_streaming_increment(chunk)
            if r.reasoning_text:
                reasoning.append(r.reasoning_text)
            if r.normal_text:
                normal.append(r.normal_text)
        return "".join(reasoning), "".join(normal)

    def test_streaming_duplicate_in_single_chunk(self):
        d = _make_qwen3()
        reasoning, normal = self._drain(
            d, ["<think>reasoning</think></think>real answer"]
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(normal, "real answer")

    def test_streaming_duplicate_across_chunks(self):
        # First chunk closes reasoning normally; the second chunk arrives with
        # an extra `</think>` (e.g. spec-decoding double redirect).
        d = _make_qwen3_force()
        chunks = [
            "reasoning</think>part1 ",
            "</think>part2",
        ]
        reasoning, normal = self._drain(d, chunks)
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(normal, "part1 part2")

    def test_streaming_no_dedupe_for_legit_text(self):
        d = _make_qwen3_force()
        chunks = ["reasoning</think>", "real ", "answer"]
        reasoning, normal = self._drain(d, chunks)
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(normal, "real answer")

    def test_streaming_mid_text_think_end_is_preserved(self):
        d = _make_qwen3_force()
        chunks = [
            "reasoning</think>before ",
            "<code></think></code> after",
        ]
        reasoning, normal = self._drain(d, chunks)
        self.assertEqual(reasoning, "reasoning")
        # Mid-text </think> should NOT be stripped — only leading ones.
        # The leading `<code>` does not match, so nothing is stripped from
        # subsequent chunks. </think> in the middle is preserved.
        self.assertIn("</think>", normal)


if __name__ == "__main__":
    unittest.main()
