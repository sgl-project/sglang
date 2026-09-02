import unittest

from sglang.srt.parser.template_detection import (
    detect_reasoning_parser,
    detect_reasoning_pattern,
    detect_tool_call_parser,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Tokenizer:
    def get_vocab(self):
        return {}


class TestK2V3TemplateDetection(CustomTestCase):
    TEMPLATE = """
    {% set tool_call_fmt = tool_call_format | default('xml') %}
    {% set effort = reasoning_effort | default('high') %}
    <ifm|think><ifm|think_fast><ifm|think_faster>
    <ifm|tool_calls><ifm|tool_call>
    """

    def test_detects_always_on_reasoning(self):
        force_reasoning, config = detect_reasoning_pattern(self.TEMPLATE)
        self.assertTrue(force_reasoning)
        self.assertIsNotNone(config)
        self.assertEqual(config.special_case, "always")

    def test_detects_single_public_parser_name(self):
        force_reasoning, config = detect_reasoning_pattern(self.TEMPLATE)
        self.assertEqual(
            detect_reasoning_parser(
                self.TEMPLATE, _Tokenizer(), config, force_reasoning
            ),
            "k2_v3",
        )
        self.assertEqual(
            detect_tool_call_parser(
                self.TEMPLATE, _Tokenizer(), config, force_reasoning
            ),
            "k2_v3",
        )


if __name__ == "__main__":
    unittest.main()
