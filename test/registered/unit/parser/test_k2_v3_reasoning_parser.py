import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ResponsesRequest,
)
from sglang.srt.parser.reasoning_parser import K2V3Detector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestK2V3ReasoningParser(CustomTestCase):
    def test_reasoning_effort_selects_ifm_pair(self):
        expected = {
            "high": ("<ifm|think>", "</ifm|think>"),
            "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
            "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
        }
        for effort, tokens in expected.items():
            with self.subTest(effort=effort):
                detector = K2V3Detector(reasoning_effort=effort)
                self.assertEqual(
                    (detector.think_start_token, detector.think_end_token), tokens
                )
        self.assertEqual(
            set(detector.request_selectable_think_end_tokens),
            {
                "</ifm|think>",
                "</ifm|think_fast>",
                "</ifm|think_faster>",
            },
        )

    def test_release_template_fallback_uses_medium_pair(self):
        # K2-Horizon-0.9B maps unsupported levels to <ifm|think_fast>.
        detector = K2V3Detector(reasoning_effort="none")
        self.assertEqual(detector.think_start_token, "<ifm|think_fast>")

    def test_end_only_output_preserves_newlines(self):
        result = K2V3Detector().detect_and_parse("\nreasoning\n</ifm|think>\nanswer")
        self.assertEqual(result.reasoning_text, "\nreasoning\n")
        self.assertEqual(result.normal_text, "\nanswer")

    def test_tool_group_implicitly_ends_malformed_reasoning(self):
        result = K2V3Detector().detect_and_parse(
            "reasoning\n<ifm|tool_calls><ifm|tool_call>x</ifm|tool_call>"
        )
        self.assertEqual(result.reasoning_text, "reasoning\n")
        self.assertTrue(result.normal_text.startswith("<ifm|tool_calls>"))

    def test_streaming_partial_tags(self):
        detector = K2V3Detector(reasoning_effort="medium")
        reasoning = ""
        normal = ""
        wire = "work</ifm|think_fast>\nanswer"
        for char in wire:
            result = detector.parse_streaming_increment(char)
            reasoning += result.reasoning_text
            normal += result.normal_text
        end = detector.finish()
        reasoning += end.reasoning_text
        normal += end.normal_text
        self.assertEqual(reasoning, "work")
        self.assertEqual(normal, "\nanswer")

    def test_reasoning_parser_reads_request_effort(self):
        request = ChatCompletionRequest(
            model="IFM/K2-Horizon-7B",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="low",
        )
        parser = ReasoningParser("k2_horizon", request=request)
        self.assertIsInstance(parser.detector, K2V3Detector)
        self.assertEqual(parser.detector.think_end_token, "</ifm|think_faster>")

    def test_template_kwarg_effort_has_rendering_precedence(self):
        request = ChatCompletionRequest(
            model="IFM/K2-Horizon-7B",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        parser = ReasoningParser("k2_horizon", request=request)
        self.assertEqual(parser.detector.think_end_token, "</ifm|think_faster>")

    def test_responses_request_effort_selects_non_stream_delimiter(self):
        end_tokens = {
            "high": "</ifm|think>",
            "medium": "</ifm|think_fast>",
            "low": "</ifm|think_faster>",
        }
        for effort, end_token in end_tokens.items():
            with self.subTest(effort=effort):
                request = ResponsesRequest(
                    model="IFM/K2-Horizon-7B",
                    input="hi",
                    reasoning={"effort": effort},
                    store=False,
                )
                parser = ReasoningParser(
                    "k2_horizon", stream_reasoning=False, request=request
                )
                self.assertEqual(
                    parser.parse_non_stream(f"work{end_token}\nanswer"),
                    ("work", "\nanswer"),
                )

    def test_force_reasoning_cannot_be_disabled(self):
        with self.assertRaisesRegex(ValueError, "requires force_reasoning=True"):
            ReasoningParser("k2_horizon", force_reasoning=False)


if __name__ == "__main__":
    unittest.main()
