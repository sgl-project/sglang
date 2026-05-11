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
        in-flight call, matching the old closing-tag-anchored regex behavior.
        Without the truncated-call filter in detect_and_parse, streaming-as-
        primitive surfaced a tool call with parameters="{}" on this input."""
        text = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n" "<arg_value>San Fr"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(
            len(result.calls), 0, "truncated mid-arg_value must yield 0 calls"
        )

    def test_truncated_post_value_emits_no_calls(self):
        """Regression: cutoff after `</arg_value>` but before `</tool_call>`
        used to surface a tool call with non-JSON parameters
        ('{"location": "SF"' with no closing brace). The filter must drop it."""
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

    def test_set_literal_falls_back_to_raw_string(self):
        """Regression: ast.literal_eval('{1,2,3}') returns a set, which
        json.dumps cannot serialize. Without the round-trip guard in
        _convert_param_value, the parse_streaming_increment loop would
        TypeError downstream. The guard rejects sets and falls back to the
        raw string (which then matches the underlying schema-string-typed
        treatment)."""
        tools_with_obj = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"options": {"type": "object"}},
                    },
                ),
            )
        ]
        detector = PoolsideV1Detector()
        text = (
            "<tool_call>get_weather\n<arg_key>options</arg_key>\n"
            "<arg_value>{1, 2, 3}</arg_value>\n</tool_call>"
        )
        result = detector.detect_and_parse(text, tools_with_obj)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        # set literal couldn't round-trip, so it's preserved as the raw
        # string (the only sane fallback).
        self.assertEqual(args["options"], "{1, 2, 3}")

    def test_truncated_after_complete_call_keeps_complete(self):
        """A complete tool_call followed by a truncated second one must keep
        the complete one and drop only the truncated tail — matching the old
        regex behavior on the same input."""
        text = (
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>NYC</arg_value>\n</tool_call>\n"
            "<tool_call>search\n<arg_key>q"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"location": "NYC"})

    def test_arg_key_without_value_emits_empty_call(self):
        """Non-streaming: malformed `<arg_key>K</arg_key></tool_call>` (no
        `<arg_value>`) yields a tool call with empty params — the orphan
        `<arg_key>` is dropped because the regex looks for key/value pairs.
        Locks in the contract the streaming FSM must match."""
        text = "<tool_call>get_weather\n<arg_key>location</arg_key></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_streaming_arg_key_without_value_closes_call(self):
        """Regression: malformed `<arg_key>K</arg_key></tool_call>` (no
        `<arg_value>`) used to leave the streaming FSM stuck in READING_VALUE
        — the bare-`<` discard ate `</tool_call>` byte-by-byte instead of
        recognizing it as a close. Worse: a *subsequent* tool call's
        `<arg_value>` would mis-attribute its content to the orphan
        `current_pending_key`, silently swallowing the second call's name.
        Both calls must be emitted with the orphan key dropped."""
        detector = PoolsideV1Detector()
        wire = (
            "<tool_call>get_weather\n<arg_key>location</arg_key></tool_call>"
            "<tool_call>search\n<arg_key>query</arg_key>\n"
            "<arg_value>tacos</arg_value>\n</tool_call>"
        )
        all_calls = []
        for chunk in [wire[i : i + 8] for i in range(0, len(wire), 8)]:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
        names = [c.name for c in all_calls if c.name]
        self.assertEqual(
            names,
            ["get_weather", "search"],
            "second call must not be swallowed when first is malformed",
        )
        per_tool: dict = {}
        for c in all_calls:
            if c.parameters:
                per_tool.setdefault(c.tool_index, "")
                per_tool[c.tool_index] += c.parameters
        # Orphan `location` key dropped — first call has empty params.
        self.assertEqual(json.loads(per_tool[0]), {})
        # Second call's value must NOT leak into first call's stale key.
        self.assertEqual(json.loads(per_tool[1]), {"query": "tacos"})

    def test_orphan_key_followed_by_new_key_uses_new_key(self):
        """Non-streaming: malformed `<arg_key>K1</arg_key><arg_key>K2</arg_key>
        <arg_value>V</arg_value>` (model emitted a key, then re-emitted a new
        key without a value for the first) yields `{K2: V}` — the orphan K1
        is dropped. Without the `[^<]` constraint in arg_pair_regex, the
        non-greedy `.*?` backtracks across the `</arg_key>` boundary and
        produces a junk key spanning both <arg_key> tags."""
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>location</arg_key>"
            "<arg_key>count</arg_key><arg_value>3</arg_value>"
            "\n</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(
            args,
            {"count": 3},
            f"orphan key 'location' should be dropped, count=3 should win, got {args}",
        )

    def test_streaming_orphan_key_followed_by_new_key_uses_new_key(self):
        """Regression: streaming on `<arg_key>K1</arg_key><arg_key>K2</arg_key>
        <arg_value>V</arg_value>` used to mis-attribute V to K1 — the bare-`<`
        discard ate the second `<arg_key>` as garbage and the value bound to
        the stale `current_pending_key`. With the orphan-key-replace branch
        in READING_VALUE, the new key wins and streaming matches the
        non-streaming regex path."""
        detector = PoolsideV1Detector()
        wire = (
            "<tool_call>get_weather\n"
            "<arg_key>location</arg_key>"
            "<arg_key>count</arg_key><arg_value>3</arg_value>"
            "\n</tool_call>"
        )
        all_calls = []
        for chunk in [wire[i : i + 8] for i in range(0, len(wire), 8)]:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
        names = [c.name for c in all_calls if c.name]
        self.assertEqual(names, ["get_weather"])
        params = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(
            json.loads(params),
            {"count": 3},
            "orphan key 'location' should be dropped; count=3 must win",
        )

    def test_streaming_malformed_no_name_does_not_hang(self):
        """Regression: malformed `<tool_call><arg_key>...` (no name, no \\n)
        used to spin in branch 2 with consume=0. Must drain to </tool_call>."""
        detector = PoolsideV1Detector()
        wire = "<tool_call><arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
        result = detector.parse_streaming_increment(wire, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_streaming_arg_tags_without_tool_call_wrapper(self):
        """Regression: stray `<arg_key>...</arg_key><arg_value>...</arg_value>`
        with no preceding `<tool_call>` used to crash with IndexError on
        `streamed_args_for_tool[-1]` (masked by the old broad except). The
        FSM's READING_VALUE state is unreachable from OUTSIDE, so this returns
        0 calls without raising — and now that the broad except is gone, any
        regression here would propagate as a real test failure."""
        detector = PoolsideV1Detector()
        wire = "<arg_key>k</arg_key><arg_value>v</arg_value>"
        result = detector.parse_streaming_increment(wire, self.tools)
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
