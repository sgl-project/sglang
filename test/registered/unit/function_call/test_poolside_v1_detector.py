"""Unit tests for PoolsideV1Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.poolside_v1_detector import PoolsideV1Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestPoolsideV1Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "count": {"type": "integer"},
                            "options": {"type": "object"},
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="now",
                    description="Current time",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]
        self.detector = PoolsideV1Detector()

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        text = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>SF</arg_value>\n</tool_call>"
        )
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("just a sentence."))

    # ==================== detect_and_parse ====================

    def test_single_tool_call_string_arg(self):
        text = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>San Francisco</arg_value>\n</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"location": "San Francisco"})

    def test_single_tool_call_mixed_types(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>location</arg_key>\n<arg_value>London</arg_value>\n"
            "<arg_key>count</arg_key>\n<arg_value>3</arg_value>\n"
            '<arg_key>options</arg_key>\n<arg_value>{"verbose": true}</arg_value>\n'
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["location"], "London")
        self.assertEqual(args["count"], 3)
        self.assertEqual(args["options"], {"verbose": True})

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>NYC</arg_value>\n</tool_call>\n"
            "<tool_call>search\n<arg_key>query</arg_key>\n"
            "<arg_value>pizza</arg_value>\n</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")
        self.assertEqual(json.loads(result.calls[1].parameters), {"query": "pizza"})

    def test_leading_text_extracted_as_normal(self):
        text = (
            "Sure, checking now. "
            "<tool_call>search\n<arg_key>query</arg_key>\n"
            "<arg_value>tacos</arg_value>\n</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Sure, checking now. ")
        self.assertEqual(len(result.calls), 1)

    def test_unknown_tool_dropped(self):
        text = (
            "<tool_call>nonexistent_fn\n<arg_key>x</arg_key>\n"
            "<arg_value>1</arg_value>\n</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_malformed_value_falls_back_to_string(self):
        text = (
            "<tool_call>get_weather\n<arg_key>options</arg_key>\n"
            "<arg_value>not_json</arg_value>\n</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["options"], "not_json")

    def test_zero_arg_call(self):
        text = "<tool_call>now\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "now")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_zero_arg_call_no_newline(self):
        """`<tool_call>now</tool_call>` (no `\\n` between name and close tag)."""
        text = "<tool_call>now</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "now")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_truncated_pre_value_emits_no_calls(self):
        """Regression: max-tokens cutoff mid-`<arg_value>` must drop the
        in-flight call. The closing-tag-anchored regex in detect_and_parse
        naturally drops these; this test locks in that contract so a future
        streaming-as-primitive refactor can't silently regress it."""
        text = "<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>San Fr"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(
            len(result.calls), 0, "truncated mid-arg_value must yield 0 calls"
        )

    def test_truncated_post_value_emits_no_calls(self):
        """Regression: cutoff after `</arg_value>` but before `</tool_call>`
        must drop the call. Without the closing-tag anchor, a partial
        emission could surface non-JSON parameters
        (`{"location": "SF"` with no closing brace)."""
        text = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>SF</arg_value>\n"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(
            len(result.calls),
            0,
            "truncated after arg_value but before </tool_call> must yield 0 calls",
        )

    def test_truncated_after_complete_call_keeps_complete(self):
        """A complete tool_call followed by a truncated second one must keep
        the complete one and drop only the truncated tail."""
        text = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>NYC</arg_value>\n</tool_call>\n"
            "<tool_call>search\n<arg_key>q"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"location": "NYC"})

    def test_streaming_malformed_no_name_does_not_hang(self):
        """Regression: malformed `<tool_call><arg_key>...` (no name, no \\n)
        used to spin in branch 2 with consume=0. Must drain to </tool_call>."""
        detector = PoolsideV1Detector()
        wire = "<tool_call><arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
        result = detector.parse_streaming_increment(wire, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_streaming_arg_tags_without_tool_call_wrapper(self):
        """Regression: a stray `<arg_key>...</arg_key><arg_value>...</arg_value>`
        arriving with no preceding `<tool_call>` used to crash branch 5 with
        IndexError on streamed_args_for_tool[-1] (current_tool_id == -1, list
        empty). The crash was silently masked by the outer except. The FSM
        should not enter the emission state without is_inside_tool_call."""
        detector = PoolsideV1Detector()
        wire = "<arg_key>k</arg_key><arg_value>v</arg_value>"
        # Capture log output to catch any masked exceptions.
        import logging

        with self.assertLogs(
            "sglang.srt.function_call.poolside_v1_detector", level="ERROR"
        ) as logs:
            result = detector.parse_streaming_increment(wire, self.tools)
            # Ensure the assertLogs context has at least one record (it errors
            # if not). If the FSM is correctly guarded, there's nothing to log,
            # so emit a benign one to satisfy the contract.
            logging.getLogger("sglang.srt.function_call.poolside_v1_detector").error(
                "sentinel"
            )
        # No errors other than our sentinel — i.e., the parser didn't crash.
        self.assertEqual(
            [r for r in logs.records if r.message != "sentinel"],
            [],
            "parser hit the broad except; FSM let an invalid state through",
        )
        self.assertEqual(len(result.calls), 0)

    # ==================== structure_info ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertEqual(info.trigger, "<tool_call>")
        self.assertIn("get_weather", info.begin)
        self.assertIn("</tool_call>", info.end)

    # ==================== Streaming ====================

    def test_streaming_single_call_chunked(self):
        detector = PoolsideV1Detector()
        chunks = [
            "<tool_",
            "call>get_weather\n<arg_key>",
            "location</arg_key>\n<arg_value>San Fr",
            "ancisco</arg_value>\n</tool_call>",
        ]
        names, params = self._collect(detector, chunks)
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(json.loads(params), {"location": "San Francisco"})

    def test_streaming_char_by_char_robustness(self):
        """Per-arg streaming under one-byte chunks. Values are emitted as a
        single `"key": value` fragment when `</arg_value>` arrives; this test
        proves the FSM doesn't leak trailing `<` / `</arg_v...` into the
        parameter delta as bytes arrive (partial-tag holdback in branch 6),
        and that the final reconstruction matches."""
        detector = PoolsideV1Detector()
        wire = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>hello world</arg_value>\n</tool_call>"
        )
        chunks = list(wire)
        names, params = self._collect(detector, chunks)
        self.assertEqual(names, ["get_weather"])
        decoded = json.loads(params)
        self.assertEqual(decoded, {"location": "hello world"})
        # And the emitted parameter delta itself contains no stray tag bytes.
        self.assertNotIn("<", params)
        self.assertNotIn(">", params)

    def test_streaming_index_is_sequential_not_tools_slot(self):
        """Regression: streaming emissions must use a per-response sequential
        index. If we emit the name with `tools_indices[name]` and the params
        with `current_tool_id`, OpenAI clients group chunks by `index` and
        split a `search`-only call (slot 1) into two broken calls."""
        detector = PoolsideV1Detector()
        # `search` is at tools-list slot 1, NOT 0.
        wire = (
            "<tool_call>search\n<arg_key>query</arg_key>\n"
            "<arg_value>tacos</arg_value>\n</tool_call>"
        )
        all_calls = []
        for c in list(wire):
            r = detector.parse_streaming_increment(c, self.tools)
            all_calls.extend(r.calls)
        # All chunks for this call must share the same index.
        indices = {c.tool_index for c in all_calls}
        self.assertEqual(
            indices,
            {0},
            f"streaming emitted mixed indices {indices}; OpenAI clients would "
            "split this into multiple broken calls",
        )
        names = [c.name for c in all_calls if c.name]
        params = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(names, ["search"])
        self.assertEqual(json.loads(params), {"query": "tacos"})

    def test_streaming_multiple_calls(self):
        detector = PoolsideV1Detector()
        wire = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>NYC</arg_value>\n</tool_call>\n"
            "<tool_call>search\n<arg_key>query</arg_key>\n"
            "<arg_value>pizza</arg_value>\n</tool_call>"
        )
        all_calls = []
        for chunk in [wire[i : i + 16] for i in range(0, len(wire), 16)]:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
        names = [c.name for c in all_calls if c.name]
        self.assertEqual(names, ["get_weather", "search"])
        # Each tool's argument deltas concatenate to a complete JSON object
        per_tool: dict = {}
        for c in all_calls:
            if c.parameters:
                per_tool.setdefault(c.tool_index, "")
                per_tool[c.tool_index] += c.parameters
        self.assertEqual(json.loads(per_tool[0]), {"location": "NYC"})
        self.assertEqual(json.loads(per_tool[1]), {"query": "pizza"})

    def test_streaming_zero_arg_call(self):
        detector = PoolsideV1Detector()
        wire = "<tool_call>now\n</tool_call>"
        names, params = self._collect(detector, list(wire))
        self.assertEqual(names, ["now"])
        # Either a single "{}" emission or a sequence whose join parses to {}
        self.assertEqual(json.loads(params or "{}"), {})

    def test_streaming_text_before_tool_call(self):
        detector = PoolsideV1Detector()
        chunks = [
            "Let me check. ",
            "<tool_call>search\n<arg_key>query</arg_key>\n",
            "<arg_value>foo</arg_value>\n</tool_call>",
        ]
        all_calls = []
        normal = ""
        for chunk in chunks:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
            normal += r.normal_text
        self.assertEqual(normal, "Let me check. ")
        names = [c.name for c in all_calls if c.name]
        self.assertEqual(names, ["search"])

    # ==================== Registry ====================

    def test_registered_in_function_call_parser(self):
        self.assertIn("poolside_v1", FunctionCallParser.ToolCallParserEnum)
        self.assertIs(
            FunctionCallParser.ToolCallParserEnum["poolside_v1"], PoolsideV1Detector
        )

    # ==================== Helpers ====================

    def _collect(self, detector, chunks):
        all_calls = []
        for chunk in chunks:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
        names = [c.name for c in all_calls if c.name]
        params = "".join(c.parameters for c in all_calls if c.parameters)
        return names, params


if __name__ == "__main__":
    import unittest

    unittest.main()
