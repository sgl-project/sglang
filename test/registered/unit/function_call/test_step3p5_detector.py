import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.step3p5_detector import Step3p5Detector
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStep3p5Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="bash",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "items": {"type": "array"},
                            "options": {"type": "object"},
                            "timeout": {"type": "integer"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "state": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

    def _parse_stream(self, chunks):
        detector = Step3p5Detector()
        normal_text = ""
        calls = {}
        name_counts = {}

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            for call in result.calls:
                self.assertGreaterEqual(call.tool_index, 0)
                state = calls.setdefault(
                    call.tool_index, {"name": None, "arguments": ""}
                )
                if call.name:
                    state["name"] = call.name
                    name_counts[call.tool_index] = (
                        name_counts.get(call.tool_index, 0) + 1
                    )
                state["arguments"] += call.parameters

        for index, state in calls.items():
            self.assertEqual(name_counts[index], 1)
            state["arguments"] = json.loads(state["arguments"])
        self.assertEqual(list(calls), list(range(len(calls))))
        return normal_text, calls

    def _assert_stream_invariant(self, source, expected_normal, expected_calls):
        expected = (expected_normal, expected_calls)
        self.assertEqual(self._parse_stream([source]), expected)
        self.assertEqual(self._parse_stream(list(source)), expected)
        for split in range(len(source) + 1):
            self.assertEqual(
                self._parse_stream([source[:split], source[split:]]),
                expected,
                msg=f"split={split}",
            )

    def test_reasoning_then_malformed_tool_non_stream(self):
        tool = "<tool_call><function bash><parameter=command>pwd</parameter></function></tool_call>"
        for suffix in ("", "</think>late close"):
            reasoning, content = ReasoningParser("step3p5").parse_non_stream(
                "inspect the repo" + tool + suffix
            )
            normal, calls = FunctionCallParser(self.tools, "step3p5").parse_non_stream(
                content
            )
            self.assertEqual(reasoning, "inspect the repo")
            self.assertEqual(normal, suffix)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].name, "bash")
            self.assertEqual(json.loads(calls[0].parameters), {"command": "pwd"})

    def test_reasoning_then_malformed_tool_stream_at_every_split(self):
        tool = "<tool_call><function bash><parameter=command>pwd</parameter></function></tool_call>"
        for suffix in ("", "</think>late close"):
            source = "inspect the repo" + tool + suffix
            chunkings = [list(source)] + [
                [source[:i], source[i:]] for i in range(len(source) + 1)
            ]
            for chunks in chunkings:
                reasoning_parser = ReasoningParser("step3p5")
                tool_parser = FunctionCallParser(self.tools, "step3p5")
                reasoning, normal, calls = "", "", []
                for chunk in chunks:
                    thinking, content = reasoning_parser.parse_stream_chunk(chunk)
                    reasoning += thinking
                    text, updates = tool_parser.parse_stream_chunk(content)
                    normal += text
                    calls.extend(updates)
                thinking, content = reasoning_parser.parse_stream_end()
                reasoning += thinking
                text, updates = tool_parser.parse_stream_chunk(content)
                normal += text
                calls.extend(updates)
                text, updates = tool_parser.parse_stream_end()
                normal += text
                calls.extend(updates)
                self.assertEqual(reasoning, "inspect the repo")
                self.assertEqual(normal, suffix)
                self.assertEqual([c.name for c in calls if c.name], ["bash"])
                self.assertTrue(all(c.tool_index == 0 for c in calls))
                self.assertEqual(
                    json.loads("".join(c.parameters for c in calls)), {"command": "pwd"}
                )

    def test_registry_isolated_from_qwen3_coder(self):
        self.assertIs(FunctionCallParser.ToolCallParserEnum["step3p5"], Step3p5Detector)
        self.assertIs(
            FunctionCallParser.ToolCallParserEnum["qwen3_coder"],
            Qwen3CoderDetector,
        )
        self.assertIsNot(Step3p5Detector, Qwen3CoderDetector)

    def test_canonical_xml_matches_qwen_parser(self):
        source = (
            "prefix<tool_call><function=bash>"
            "<parameter=command>pwd</parameter>"
            "<parameter=timeout>5</parameter>"
            "</function></tool_call>"
        )
        step_result = Step3p5Detector().detect_and_parse(source, self.tools)
        qwen_result = Qwen3CoderDetector().detect_and_parse(source, self.tools)
        self.assertEqual(step_result, qwen_result)
        self._assert_stream_invariant(
            source,
            "prefix",
            {0: {"name": "bash", "arguments": {"command": "pwd", "timeout": 5}}},
        )

    def test_missing_equals_in_known_function_tag(self):
        for function_tag in ("<function bash>", "<functionbash>"):
            source = (
                f"<tool_call>{function_tag}"
                "<parameter=command>pwd</parameter>"
                "</function></tool_call>"
            )
            self._assert_stream_invariant(
                source,
                "",
                {0: {"name": "bash", "arguments": {"command": "pwd"}}},
            )
            result = Step3p5Detector().detect_and_parse(source, self.tools)
            self.assertEqual(result.calls[0].name, "bash")
            self.assertEqual(json.loads(result.calls[0].parameters), {"command": "pwd"})

    def test_missing_right_angle_in_function_and_parameter_tags(self):
        source = (
            "<tool_call><function=bash\n"
            "<parameter=command\npwd</parameter>"
            "</function></tool_call>"
        )
        self._assert_stream_invariant(
            source,
            "",
            {0: {"name": "bash", "arguments": {"command": "pwd"}}},
        )
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "pwd"})

    def test_duplicate_parameter_prefix_is_repaired_when_boundary_is_known(self):
        source = (
            "<tool_call><function=bash>"
            "<parameter=parameter=command\npwd</parameter>"
            "</function></tool_call>"
        )
        self._assert_stream_invariant(
            source,
            "",
            {0: {"name": "bash", "arguments": {"command": "pwd"}}},
        )

    def test_missing_outer_tool_call_wrapper(self):
        source = "<function=bash><parameter=command>pwd</parameter></function>"
        parser = FunctionCallParser(self.tools, "step3p5")
        self.assertTrue(parser.has_tool_call(source))
        normal_text, calls = parser.parse_non_stream(source)
        self.assertEqual(normal_text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(json.loads(calls[0].parameters), {"command": "pwd"})
        self._assert_stream_invariant(
            source,
            "",
            {0: {"name": "bash", "arguments": {"command": "pwd"}}},
        )

    def test_missing_parameter_close_before_next_parameter(self):
        source = (
            "<tool_call><function=weather>"
            "<parameter=city>Dallas"
            "<parameter=state>TX</parameter>"
            "</function></tool_call>"
        )
        self._assert_stream_invariant(
            source,
            "",
            {
                0: {
                    "name": "weather",
                    "arguments": {"city": "Dallas", "state": "TX"},
                }
            },
        )

    def test_tool_close_repairs_open_parameter_and_function(self):
        source = "<tool_call><function=bash><parameter=command>pwd</tool_call>"
        self._assert_stream_invariant(
            source,
            "",
            {0: {"name": "bash", "arguments": {"command": "pwd"}}},
        )
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "pwd"})

    def test_adjacent_calls_in_one_mtp_chunk(self):
        source = (
            "<tool_call><function=bash>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call><tool_call><function=weather>"
            "<parameter=city>Paris</parameter>"
            "</function></tool_call>"
        )
        self._assert_stream_invariant(
            source,
            "",
            {
                0: {"name": "bash", "arguments": {"command": "pwd"}},
                1: {"name": "weather", "arguments": {"city": "Paris"}},
            },
        )

    def test_interleaved_text_is_preserved_in_both_modes(self):
        first = (
            "<tool_call><function=bash>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )
        second = (
            "<tool_call><function=weather>"
            "<parameter=city>Paris</parameter>"
            "</function></tool_call>"
        )
        source = f"hello{first}between{second}bye"
        expected_calls = {
            0: {"name": "bash", "arguments": {"command": "pwd"}},
            1: {"name": "weather", "arguments": {"city": "Paris"}},
        }
        self._assert_stream_invariant(source, "hellobetweenbye", expected_calls)

        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, "hellobetweenbye")
        self.assertEqual(
            {
                call.tool_index: {
                    "name": call.name,
                    "arguments": json.loads(call.parameters),
                }
                for call in result.calls
            },
            expected_calls,
        )

    def test_python_literal_container_parameters_remain_supported(self):
        source = (
            "<tool_call><function=bash>"
            "<parameter=options>{'check': True}</parameter>"
            "<parameter=items>[1, 2]</parameter>"
            "</function></tool_call>"
        )
        self._assert_stream_invariant(
            source,
            "",
            {
                0: {
                    "name": "bash",
                    "arguments": {"options": {"check": True}, "items": [1, 2]},
                }
            },
        )

    def test_unknown_malformed_function_is_not_repaired(self):
        source = (
            "<tool_call><function unknown>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(result.calls, [])
        normal_text, calls = self._parse_stream(list(source))
        self.assertEqual(normal_text, "")
        self.assertEqual(calls, {})

    def test_bare_unknown_function_is_preserved_as_text(self):
        source = (
            "prefix<function=unknown>"
            "<parameter=command>pwd</parameter>"
            "</function>suffix"
        )
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, source)
        self.assertEqual(result.calls, [])
        self.assertEqual(self._parse_stream(list(source)), (source, {}))

    def test_xml_like_prose_does_not_block_a_later_real_call(self):
        real_call = (
            "<tool_call><function=bash>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )
        prefix = "intro<functions>x</functions><function_name>y</function_name>"
        source = prefix + real_call
        self._assert_stream_invariant(
            source,
            prefix,
            {0: {"name": "bash", "arguments": {"command": "pwd"}}},
        )

    def test_partial_function_words_are_plain_stream_content(self):
        for source in (
            "This uses <functionality in prose",
            "This uses <functionbashful in prose",
            "This uses <function unknown in prose",
            "This discusses <parameterization forever",
        ):
            result = Step3p5Detector().detect_and_parse(source, self.tools)
            self.assertEqual(result.normal_text, source)
            self.assertEqual(result.calls, [])
            self.assertEqual(self._parse_stream(list(source)), (source, {}))

    def test_unmatched_closing_tags_are_preserved_as_text(self):
        source = "a</function>b</parameter>c</tool_call>d"
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, source)
        self.assertEqual(result.calls, [])
        self.assertEqual(self._parse_stream(list(source)), (source, {}))

    def test_private_use_text_has_no_special_meaning(self):
        source = "prefix\ue000step3p5-literal-lt\ue001suffix"
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, source)
        self.assertEqual(result.calls, [])
        self.assertEqual(self._parse_stream(list(source)), (source, {}))

    def test_literal_and_parser_segments_do_not_reorder_text(self):
        call = (
            "<tool_call><function=bash>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )
        source = f"a<</function>b{call}"
        self._assert_stream_invariant(
            source,
            "a<</function>b",
            {0: {"name": "bash", "arguments": {"command": "pwd"}}},
        )

    def test_non_stream_closes_confirmed_call_at_eos(self):
        source = "<tool_call><function=bash><parameter=command>pwd"
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "bash")
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "pwd"})

    def test_non_stream_does_not_invent_call_from_partial_opening_tag(self):
        for source in (
            "<tool_call><function=bash",
            "<tool_call><function=bash><parameter=command",
        ):
            result = Step3p5Detector().detect_and_parse(source, self.tools)
            self.assertEqual(result.normal_text, source)
            self.assertEqual(result.calls, [])

    def test_non_stream_partial_second_call_keeps_completed_first_call(self):
        first = (
            "<tool_call><function=bash>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )
        partial = "<tool_call><function=weather"
        result = Step3p5Detector().detect_and_parse(first + partial, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "bash")
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "pwd"})
        self.assertEqual(result.normal_text, partial)

    def test_redundant_close_after_synthetic_wrapper_is_consumed(self):
        for separator in ("", "\n", "\n  "):
            source = (
                "<function=bash><parameter=command>pwd</parameter>"
                f"</function>{separator}</tool_call>"
            )
            result = Step3p5Detector().detect_and_parse(source, self.tools)

            self.assertEqual(result.normal_text, "")
            self.assertEqual(len(result.calls), 1)
            self.assertEqual(result.calls[0].name, "bash")
            self.assertEqual(json.loads(result.calls[0].parameters), {"command": "pwd"})
            self._assert_stream_invariant(
                source,
                "",
                {0: {"name": "bash", "arguments": {"command": "pwd"}}},
            )

    def test_whitespace_after_synthetic_wrapper_is_preserved_without_close(self):
        source = "<function=bash></function>\n"
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, "\n")
        self.assertEqual(result.calls[0].name, "bash")

    def test_non_stream_conversion_error_falls_back_to_text(self):
        source = (
            "<tool_call><function=bash>"
            "<parameter=options>{1, 2}</parameter>"
            "</function></tool_call>"
        )
        result = Step3p5Detector().detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, source)
        self.assertEqual(result.calls, [])

    def test_function_like_text_inside_parameter_is_not_repaired(self):
        source = (
            "<tool_call><function=bash>"
            "<parameter=command>printf '&lt;function bash&gt;'</parameter>"
            "</function></tool_call>"
        )
        expected_arguments = {"command": "printf '&lt;function bash&gt;'"}
        self._assert_stream_invariant(
            source,
            "",
            {0: {"name": "bash", "arguments": expected_arguments}},
        )

        # The raw form is legal parameter text too. Keeping it verbatim is
        # important because shell/code-edit tools frequently carry XML-like
        # snippets as data.
        raw_source = source.replace("&lt;function bash&gt;", "<function bash>")
        self._assert_stream_invariant(
            raw_source,
            "",
            {
                0: {
                    "name": "bash",
                    "arguments": {"command": "printf '<function bash>'"},
                }
            },
        )
        result = Step3p5Detector().detect_and_parse(raw_source, self.tools)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"command": "printf '<function bash>'"},
        )

    def test_tool_call_start_text_inside_parameter_is_not_repaired(self):
        source = (
            "<tool_call><function=bash>"
            "<parameter=command>printf '<tool_call>'</parameter>"
            "</function></tool_call>"
        )
        self._assert_stream_invariant(
            source,
            "",
            {
                0: {
                    "name": "bash",
                    "arguments": {"command": "printf '<tool_call>'"},
                }
            },
        )

    def test_qwen_parser_does_not_gain_step_repair(self):
        source = (
            "<tool_call><function bash>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )
        step_result = Step3p5Detector().detect_and_parse(source, self.tools)
        qwen_result = Qwen3CoderDetector().detect_and_parse(source, self.tools)
        self.assertEqual(step_result.calls[0].name, "bash")
        self.assertEqual(qwen_result.calls, [])

    def test_structural_tag_contract_is_inherited(self):
        detector = Step3p5Detector()
        self.assertTrue(detector.supports_structural_tag())
        self.assertEqual(detector.get_structural_tag_name(), "qwen_3_coder")


if __name__ == "__main__":
    unittest.main()
