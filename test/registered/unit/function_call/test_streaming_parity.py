"""A detector must report the same tool calls however the output is chunked.

Chunk boundaries come from detokenization, so a client that streams cannot
avoid them: any disagreement with ``detect_and_parse`` reaches the caller.
"""

import json
from typing import Dict, List, Tuple

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(2.0, "base-a-test-cpu")


def _tool(name: str, properties: dict = None) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={"type": "object", "properties": properties or {}},
        ),
    )


# The second tool exists so that a call to it would get a non-zero index if a
# detector keyed tool_index off the tools list instead of the call order.
TOOLS = [_tool("get_weather"), _tool("f")]

# One generation per detector, each containing a tool call with empty or simple
# arguments -- the shape that used to be lost, mistyped, or misindexed.
CASES: Dict[str, str] = {
    "cohere_command4": (
        '<|START_ACTION|>[{"tool_name": "f", "parameters": {}}]<|END_ACTION|>'
    ),
    "gemma4": "<|tool_call>call:f{}<tool_call|>",
    "glm": (
        "<tool_call>get_weather\n"
        "<arg_key>get_weather</arg_key>\n"
        "<arg_value>123</arg_value>\n"
        "</tool_call>"
    ),
    "glm45": (
        "<tool_call>get_weather\n"
        "<arg_key>get_weather</arg_key>\n"
        "<arg_value>123</arg_value>\n"
        "</tool_call>"
    ),
    "glm47": (
        "<tool_call>get_weather"
        "<arg_key>get_weather</arg_key>"
        "<arg_value>123</arg_value>"
        "</tool_call>"
    ),
    "minimax-m2": (
        '<minimax:tool_call><invoke name="get_weather"></invoke></minimax:tool_call>'
    ),
    "mistral": (
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}, '
        '{"name": "get_weather", "arguments": {}}]'
    ),
    "step3": (
        "<｜tool_calls_begin｜>"
        "<｜tool_call_begin｜>function<｜tool_sep｜>"
        '<steptml:invoke name="get_weather"></steptml:invoke>'
        "<｜tool_call_end｜>"
        "<｜tool_call_begin｜>function<｜tool_sep｜>"
        '<steptml:invoke name="get_weather">'
        '<steptml:parameter name="get_weather">hello</steptml:parameter>'
        "</steptml:invoke>"
        "<｜tool_call_end｜>"
        "<｜tool_calls_end｜>"
    ),
}

Call = Tuple[int, str, object]


def _oneshot(parser_name: str, text: str) -> List[Call]:
    detector = FunctionCallParser.ToolCallParserEnum[parser_name]()
    calls = detector.detect_and_parse(text, TOOLS).calls
    return [(c.tool_index, c.name, json.loads(c.parameters or "{}")) for c in calls]


def _streamed(parser_name: str, chunks: List[str]) -> List[Call]:
    """Feed ``chunks`` in order and reassemble the calls the way a client does."""
    detector = FunctionCallParser.ToolCallParserEnum[parser_name]()
    names: Dict[int, str] = {}
    args: Dict[int, str] = {}
    order: List[int] = []
    # The trailing empty increments mirror the flushes the serving layer sends
    # when generation ends.
    for chunk in list(chunks) + ["", ""]:
        for call in detector.parse_streaming_increment(chunk, TOOLS).calls:
            if call.tool_index not in args:
                order.append(call.tool_index)
                args[call.tool_index] = ""
            if call.name:
                names[call.tool_index] = call.name
            if call.parameters:
                args[call.tool_index] += call.parameters
    return [(i, names.get(i), json.loads(args[i] or "{}")) for i in order]


class TestStreamingParity(CustomTestCase):
    def test_char_by_char_matches_one_shot(self):
        """One character per increment must not change the parsed calls."""
        for name, text in CASES.items():
            with self.subTest(parser=name):
                self.assertEqual(_streamed(name, list(text)), _oneshot(name, text))

    def test_single_chunk_matches_one_shot(self):
        """A whole generation in one increment must not be dropped."""
        for name, text in CASES.items():
            with self.subTest(parser=name):
                self.assertEqual(_streamed(name, [text]), _oneshot(name, text))

    def test_two_chunk_splits_match_one_shot(self):
        """No split point may change the parsed calls."""
        for name, text in CASES.items():
            expected = _oneshot(name, text)
            for cut in range(1, len(text)):
                with self.subTest(parser=name, cut=cut):
                    self.assertEqual(
                        _streamed(name, [text[:cut], text[cut:]]), expected
                    )


class TestParameterlessCalls(CustomTestCase):
    """A call with no arguments is still a call, and its arguments are ``{}``."""

    def test_step3_one_shot_keeps_parameterless_call(self):
        text = (
            "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
            '<steptml:invoke name="get_weather"></steptml:invoke>'
            "<｜tool_call_end｜><｜tool_calls_end｜>"
        )
        self.assertEqual(_oneshot("step3", text), [(0, "get_weather", {})])

    def test_minimax_m2_streams_empty_arguments(self):
        text = (
            '<minimax:tool_call><invoke name="get_weather"></invoke>'
            "</minimax:tool_call>"
        )
        self.assertEqual(_streamed("minimax-m2", list(text)), [(0, "get_weather", {})])


class TestToolIndexIsCallPosition(CustomTestCase):
    """``tool_index`` is the OpenAI ``index`` field: the call's position in the
    response. Tool-call ids derive from it, so two calls must never share one.
    """

    def test_repeated_tool_gets_distinct_indices(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}, '
            '{"name": "get_weather", "arguments": {}}]'
        )
        self.assertEqual([c[0] for c in _oneshot("mistral", text)], [0, 1])

    def test_first_call_is_index_zero_for_a_later_tool(self):
        text = '<|START_ACTION|>[{"tool_name": "f", "parameters": {}}]<|END_ACTION|>'
        # "f" is TOOLS[1], but it is the first (and only) call in the response.
        self.assertEqual(_oneshot("cohere_command4", text), [(0, "f", {})])


class TestUndeclaredArgumentTypes(CustomTestCase):
    """A value whose type the schema does not declare is typed from the literal,
    so an undeclared number must not reach the client quoted as a string.
    """

    def test_glm_number_survives_streaming(self):
        for parser in ("glm", "glm45", "glm47"):
            with self.subTest(parser=parser):
                streamed = _streamed(parser, list(CASES[parser]))
                self.assertEqual(streamed[0][2], {"get_weather": 123})


if __name__ == "__main__":
    import unittest

    unittest.main()
