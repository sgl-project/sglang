"""Unit tests for the Agens reasoning detector (srt/parser/reasoning_parser.py)."""

import unittest

from sglang.srt.parser.reasoning_parser import AgensDetector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAgensDetector(CustomTestCase):
    def test_registered(self):
        self.assertIs(ReasoningParser.DetectorMap["agens"], AgensDetector)

    def test_prompt_opened_think_block(self):
        # The generation prompt already opened <|think|>.
        detector = AgensDetector(force_reasoning=True)
        result = detector.detect_and_parse(
            "step one, step two<|/think|>The answer is 4."
        )
        self.assertEqual(result.reasoning_text, "step one, step two")
        self.assertEqual(result.normal_text, "The answer is 4.")

    def test_explicit_think_block(self):
        detector = AgensDetector()
        result = detector.detect_and_parse("<|think|>hmm<|/think|>Done.")
        self.assertEqual(result.reasoning_text, "hmm")
        self.assertEqual(result.normal_text, "Done.")

    def test_call_closes_reasoning(self):
        detector = AgensDetector(force_reasoning=True)
        result = detector.detect_and_parse(
            "need the weather\n<|call|>\n<function=f>\n</function>\n<|/call|>"
        )
        self.assertEqual(result.reasoning_text.strip(), "need the weather")
        self.assertTrue(result.normal_text.startswith("<|call|>"))

    def test_streaming(self):
        detector = AgensDetector(force_reasoning=True)
        reasoning, normal = "", ""
        for chunk in ["abc ", "def<|/", "think|>", "Answer", "."]:
            res = detector.parse_streaming_increment(chunk)
            reasoning += res.reasoning_text or ""
            normal += res.normal_text or ""
        self.assertEqual(reasoning, "abc def")
        self.assertEqual(normal, "Answer.")


if __name__ == "__main__":
    unittest.main()
