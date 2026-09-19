"""Strict DSML parsing for the DeepSeek V3.2 / V4 detectors
(SGLANG_ENABLE_STRICT_DSML_TOOL_CALLS).

A complete invoke body that is not well-formed must never come back as a call
with `{}` arguments: the one-shot path returns the text as content with no
call, and the streaming path never leaves a client holding a tool name with
empty or `{}` arguments, nor a complete JSON object before the invoke body has
passed validation at the closer. With the knob off, today's behaviour is
unchanged.

    python test/registered/unit/function_call/test_deepseekv32_strict_dsml.py
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import (
    DeepSeekV32Detector,
    MalformedDSMLToolCall,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DSML = "｜DSML｜"
LOGGER = "sglang.srt.function_call.deepseekv32_detector"
CHUNK_SIZES = [1, 2, 3, 5, 7, 11, 23, 1000]

# (detector class, calls-block name) per format; invoke / parameter tags are shared.
FLAVORS = {
    "v4": (DeepSeekV4Detector, "tool_calls"),
    "v32": (DeepSeekV32Detector, "function_calls"),
}

# A model turn with a parameter tag missing both the DSML marker and the
# `string` attribute. Non-strict: [set_flag, "{}"]; the vendor parser raises.
MALFORMED_TURN = (
    "I'll call the `set_flag` function to enable the flag.\n\n"
    f"<{DSML}tool_calls>\n"
    f'<{DSML}invoke name="set_flag">\n'
    f'<parameter name="enabled">true</{DSML}parameter>\n'
    f"</{DSML}invoke>\n"
    f"</{DSML}tool_calls>"
)


def _strict(cls):
    with envs.SGLANG_ENABLE_STRICT_DSML_TOOL_CALLS.override(True):
        return cls()


def _tools():
    def tool(name, properties):
        return Tool(
            type="function",
            function=Function(
                name=name, parameters={"type": "object", "properties": properties}
            ),
        )

    return [
        tool("set_flag", {"enabled": {"type": "boolean"}}),
        tool("lookup_fixture", {"key": {"type": "string"}}),
        tool(
            "search",
            {
                "query": {"type": "string"},
                "topn": {"type": "integer"},
                "ratio": {"type": "number"},
            },
        ),
        tool("submit", {}),
    ]


def _param(name, is_string, value):
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _invoke(name, body):
    return f'<{DSML}invoke name="{name}">{body}</{DSML}invoke>'


def _block(flavor, *invokes, preamble=""):
    block = FLAVORS[flavor][1]
    return f"{preamble}<{DSML}{block}>\n" + "\n".join(invokes) + f"\n</{DSML}{block}>"


def _calls(result):
    return [(c.name, json.loads(c.parameters)) for c in result.calls]


def _assemble(calls):
    """Streamed ToolCallItems -> [(name, arguments)] per tool_index; the first
    item of each index must carry the name."""
    by_index, first = {}, {}
    for call in calls:
        first.setdefault(call.tool_index, call)
        entry = by_index.setdefault(call.tool_index, {"name": None, "args": ""})
        if call.name:
            entry["name"] = call.name
        entry["args"] += call.parameters or ""
    for index, call in first.items():
        assert call.name, f"tool_index {index}: arguments arrived before the name"
    return [(e["name"], json.loads(e["args"])) for _, e in sorted(by_index.items())]


def _stream_passes(detector, text, tools, chunk_size):
    return [
        detector.parse_streaming_increment(text[i : i + chunk_size], tools)
        for i in range(0, len(text), chunk_size)
    ]


def _stream(detector, text, tools, chunk_size):
    normal, calls = "", []
    for result in _stream_passes(detector, text, tools, chunk_size):
        normal += result.normal_text
        calls.extend(result.calls)
    return normal, calls


def _assert_names_travel_with_arguments(test, passes):
    """The item that carries a tool name carries the first non-empty argument
    delta itself, and no other item of the pass carries that name: serving
    yields one SSE chunk per item, so a name-only item followed by an argument
    item is two chunks, and a cut between them leaves the client name + ""."""
    for n, result in enumerate(passes):
        named = [c for c in result.calls if c.name]
        indices = [c.tool_index for c in named]
        test.assertEqual(len(indices), len(set(indices)), f"pass {n}: {result.calls}")
        for item in named:
            test.assertTrue(
                item.parameters,
                f"pass {n}: name {item.name!r} went out with parameters "
                f"{item.parameters!r} (name-only chunk): {result.calls}",
            )


def _malformed_bodies():
    return {
        "param_missing_dsml_marker_and_string_attr": f'\n<parameter name="key">alpha</{DSML}parameter>\n',
        "param_missing_string_attr_only": f'<{DSML}parameter name="key">alpha</{DSML}parameter>',
        "param_missing_dsml_marker_only": f'<parameter name="key" string="true">alpha</{DSML}parameter>',
        "unterminated_param_tag": f'<{DSML}parameter name="key" string="true">alpha',
        "truncated_json_body": '{"key": "alpha"',
        "invalid_json_body": '{"key": alpha}',
        "non_object_json_body": '["alpha"]',
        "good_param_then_stray_tag": _param("key", "true", "alpha")
        + f'\n<parameter name="x">1</{DSML}parameter>',
        "prose_in_body": "\nenable it please\n",
    }


# Text after a complete JSON object body that the closer rejects.
TRAILING_TEXT = {
    "prose": "\nnot json",
    "second_object": " {}",
    "stray_tag": f'\n<parameter name="x">1</{DSML}parameter>',
}


def _json_body_then(trailing):
    """A JSON object body followed by text the closer rejects, cut so the
    object is complete before the trailing text arrives: opener, object,
    trailing text, closer."""
    opener = f'<{DSML}tool_calls>\n<{DSML}invoke name="lookup_fixture">\n'
    closer = f"\n</{DSML}invoke>\n</{DSML}tool_calls>"
    chunks = [opener, '{"key": "alpha"}', trailing, closer]
    body = "\n" + '{"key": "alpha"}' + trailing + "\n"
    assert "".join(chunks) == _block("v4", _invoke("lookup_fixture", body))
    return chunks


class TestMalformedBodiesAreDropped(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    def test_malformed_turn_is_dropped_and_forwarded(self):
        detector = _strict(DeepSeekV4Detector)
        self.assertTrue(detector.has_tool_call(MALFORMED_TURN))
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            result = detector.detect_and_parse(MALFORMED_TURN, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, MALFORMED_TURN)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("Malformed DSML tool call for set_flag dropped", logs.output[0])
        self.assertIn("unparsed text inside the invoke body", logs.output[0])

    def test_every_malformed_body_in_every_flavour(self):
        for flavor, (cls, _) in FLAVORS.items():
            for label, body in _malformed_bodies().items():
                with self.subTest(flavor=flavor, shape=label):
                    detector = _strict(cls)
                    with self.assertRaises(MalformedDSMLToolCall):
                        detector._parse_parameters_from_xml(body)
                    text = _block(
                        flavor, _invoke("lookup_fixture", body), preamble="Sure.\n\n"
                    )
                    with self.assertLogs(LOGGER, level="WARNING") as logs:
                        result = detector.detect_and_parse(text, self.tools)
                    self.assertEqual(result.calls, [])
                    self.assertEqual(result.normal_text, text)
                    self.assertIn(
                        "Malformed DSML tool call for lookup_fixture dropped",
                        logs.output[-1],
                    )

    def test_a_block_with_one_malformed_member_yields_no_call(self):
        """Atomic, as the k2_v3 detector does: no partial recovery that keeps
        some calls and hides the malformed one in content."""
        text = _block(
            "v4",
            _invoke("lookup_fixture", _param("key", "true", "alpha")),
            _invoke("set_flag", f'<parameter name="enabled">true</{DSML}parameter>'),
        )
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            result = _strict(DeepSeekV4Detector).detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)
        self.assertIn("for set_flag dropped", logs.output[0])

    def test_error_names_the_reason(self):
        detector = _strict(DeepSeekV4Detector)
        with self.assertRaisesRegex(MalformedDSMLToolCall, "invalid JSON body"):
            detector._parse_parameters_from_xml('{"key": alpha}')
        with self.assertRaisesRegex(
            MalformedDSMLToolCall, "unparsed text inside the invoke body"
        ):
            detector._parse_parameters_from_xml("enable it")


class TestWellFormedBodiesUnchanged(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    def _one(self, flavor, name, body):
        text = _block(flavor, _invoke(name, body), preamble="Let me look.\n\n")
        result = _strict(FLAVORS[flavor][0]).detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Let me look.")
        return _calls(result)

    def test_parameter_shapes(self):
        cases = {
            "str": ("lookup_fixture", [("key", "true", "alpha")], {"key": "alpha"}),
            "two str": (
                "search",
                [("query", "true", "a b"), ("topn", "false", "3")],
                {"query": "a b", "topn": 3},
            ),
            "numbers": (
                "search",
                [("topn", "false", "10"), ("ratio", "false", "3.5")],
                {"topn": 10, "ratio": 3.5},
            ),
            "bool true": (
                "set_flag",
                [("enabled", "false", "true")],
                {"enabled": True},
            ),
            "bool false": (
                "set_flag",
                [("enabled", "false", "false")],
                {"enabled": False},
            ),
            "nested / quotes / <> / newline": (
                "search",
                [
                    ("query", "true", 'He said "hi" <b>&</b>\nnext'),
                    ("topn", "false", '{"name": "John", "age": 30}'),
                ],
                {
                    "query": 'He said "hi" <b>&</b>\nnext',
                    "topn": {"name": "John", "age": 30},
                },
            ),
            "empty string value": (
                "lookup_fixture",
                [("key", "true", "")],
                {"key": ""},
            ),
        }
        for flavor in FLAVORS:
            for label, (name, params, expected) in cases.items():
                with self.subTest(flavor=flavor, shape=label):
                    body = "\n" + "\n".join(_param(*p) for p in params) + "\n"
                    self.assertEqual(self._one(flavor, name, body), [(name, expected)])

    def test_zero_argument_forms(self):
        for flavor, (cls, _) in FLAVORS.items():
            with self.subTest(flavor=flavor):
                self.assertEqual(self._one(flavor, "submit", ""), [("submit", {})])
                self.assertEqual(
                    self._one(flavor, "submit", "\n   \n"), [("submit", {})]
                )
                text = _block(flavor, f'<{DSML}invoke name="submit"/>')
                result = _strict(cls).detect_and_parse(text, self.tools)
                self.assertEqual(_calls(result), [("submit", {})])

    def test_json_body_and_several_blocks(self):
        self.assertEqual(
            self._one("v4", "lookup_fixture", '\n{"key": "alpha"}\n'),
            [("lookup_fixture", {"key": "alpha"})],
        )
        two = (
            _block("v4", _invoke("lookup_fixture", _param("key", "true", "a")))
            + "\n"
            + _block("v4", _invoke("lookup_fixture", _param("key", "true", "b")))
        )
        result = _strict(DeepSeekV4Detector).detect_and_parse(two, self.tools)
        self.assertEqual(
            _calls(result),
            [("lookup_fixture", {"key": "a"}), ("lookup_fixture", {"key": "b"})],
        )

    def test_unknown_tool_is_still_dropped_by_parse_base_json(self):
        text = _block("v4", _invoke("nope", _param("key", "true", "a")))
        result = _strict(DeepSeekV4Detector).detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])


class TestStrictStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    def test_malformed_turn_streams_no_call_and_tracks_nothing(self):
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                detector = _strict(DeepSeekV4Detector)
                with self.assertLogs(LOGGER, level="WARNING") as logs:
                    normal, calls = _stream(
                        detector, MALFORMED_TURN, self.tools, chunk_size
                    )
                    normal += detector.finish(self.tools).normal_text
                self.assertEqual(calls, [])
                # The whole turn verbatim: the same content the one-shot path returns.
                self.assertEqual(normal, MALFORMED_TURN)
                # Nothing for the serving layer's end-of-stream back-fill.
                self.assertEqual(detector.prev_tool_call_arr, [])
                self.assertEqual(detector.streamed_args_for_tool, [])
                self.assertEqual(
                    sum("for set_flag dropped" in line for line in logs.output), 1
                )

    def test_good_call_before_a_malformed_one_survives_alone(self):
        text = _block(
            "v4",
            _invoke("lookup_fixture", _param("key", "true", "alpha")),
            _invoke("set_flag", f'<parameter name="enabled">true</{DSML}parameter>'),
        )
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                detector = _strict(DeepSeekV4Detector)
                with self.assertLogs(LOGGER, level="WARNING"):
                    normal, calls = _stream(detector, text, self.tools, chunk_size)
                self.assertEqual(
                    _assemble(calls), [("lookup_fixture", {"key": "alpha"})]
                )
                self.assertIn(f'<{DSML}invoke name="set_flag">', normal)
                self.assertEqual(len(detector.prev_tool_call_arr), 1)
                tracked = detector.prev_tool_call_arr[0]
                self.assertEqual(
                    tracked["arguments"], detector.streamed_args_for_tool[0]
                )
                self.assertEqual(json.loads(tracked["arguments"]), {"key": "alpha"})

    def test_malformed_close_after_a_good_parameter_never_leaves_executable_arguments(
        self,
    ):
        """What went out cannot be recalled, so at every chunk size either
        nothing of the call reached the client, or the name came with a
        non-empty JSON prefix that does not parse (never "" or `{}`); the
        tracked arguments equal what was streamed and the block is content."""
        body = (
            "\n"
            + _param("key", "true", "alpha")
            + f'\n<parameter name="x">1</{DSML}parameter>\n'
        )
        text = _block("v4", _invoke("lookup_fixture", body), preamble="On it.\n\n")
        seen_prefix = False
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                detector = _strict(DeepSeekV4Detector)
                with self.assertLogs(LOGGER, level="WARNING"):
                    passes = _stream_passes(detector, text, self.tools, chunk_size)
                _assert_names_travel_with_arguments(self, passes)
                normal = "".join(r.normal_text for r in passes)
                calls = [c for r in passes for c in r.calls]
                streamed = "".join(c.parameters for c in calls)
                if calls:
                    self.assertEqual(calls[0].name, "lookup_fixture")
                    self.assertTrue(streamed)
                    self.assertNotEqual(streamed, "{}")
                    with self.assertRaises(json.JSONDecodeError):
                        json.loads(streamed)
                    self.assertEqual(
                        detector.prev_tool_call_arr,
                        [{"name": "lookup_fixture", "arguments": streamed}],
                    )
                    self.assertEqual(detector.streamed_args_for_tool, [streamed])
                    seen_prefix = True
                else:
                    self.assertEqual(detector.prev_tool_call_arr, [])
                    self.assertEqual(detector.streamed_args_for_tool, [])
                self.assertEqual(normal.count("On it."), 1)
                self.assertIn(f'<parameter name="x">1</{DSML}parameter>', normal)
                self.assertEqual(detector.finish(self.tools).calls, [])
        self.assertTrue(seen_prefix, "no chunk size exercised the streamed-prefix arm")

    def test_malformed_then_good_in_one_block_yields_no_call(self):
        """Atomic like detect_and_parse: once an invoke of the block is
        malformed, the rest of the block is content up to the closer, nothing
        is tracked, and a block that follows parses again."""
        poisoned = _block(
            "v4",
            _invoke("set_flag", f'<parameter name="enabled">true</{DSML}parameter>'),
            _invoke("lookup_fixture", _param("key", "true", "alpha")),
            preamble="Two.\n\n",
        )
        later = _block("v4", _invoke("lookup_fixture", _param("key", "true", "beta")))
        text = poisoned + "\n" + later
        one_shot = _strict(DeepSeekV4Detector).detect_and_parse(poisoned, self.tools)
        self.assertEqual((one_shot.normal_text, one_shot.calls), (poisoned, []))
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                detector = _strict(DeepSeekV4Detector)
                with self.assertLogs(LOGGER, level="WARNING") as logs:
                    passes = _stream_passes(detector, poisoned, self.tools, chunk_size)
                    passes.append(detector.finish(self.tools))
                normal = "".join(r.normal_text for r in passes)
                self.assertEqual([c for r in passes for c in r.calls], [])
                self.assertEqual(normal, poisoned)
                self.assertEqual(detector.prev_tool_call_arr, [])
                self.assertEqual(
                    sum("for set_flag dropped" in line for line in logs.output), 1
                )
            with self.subTest(chunk_size=chunk_size, then="a later block"):
                detector = _strict(DeepSeekV4Detector)
                with self.assertLogs(LOGGER, level="WARNING"):
                    passes = _stream_passes(detector, text, self.tools, chunk_size)
                _assert_names_travel_with_arguments(self, passes)
                normal = "".join(r.normal_text for r in passes)
                self.assertEqual(
                    _assemble([c for r in passes for c in r.calls]),
                    [("lookup_fixture", {"key": "beta"})],
                )
                self.assertTrue(normal.startswith(poisoned))
                self.assertNotIn("beta", normal)

    def test_whitespace_only_text_is_held_not_released_as_content(self):
        """The "\\n\\n" before a calls block arrives as its own token: no
        `content: "\\n\\n"` delta ahead of the tool_calls deltas. Whitespace
        before prose flows out with the prose; at the end of the stream it
        comes out of finish()."""
        detector = _strict(DeepSeekV4Detector)
        self.assertEqual(
            detector.parse_streaming_increment("\n\n", self.tools).normal_text, ""
        )
        block = _block("v4", _invoke("lookup_fixture", _param("key", "true", "a")))
        passes = _stream_passes(detector, block, self.tools, 5)
        self.assertEqual("".join(r.normal_text for r in passes), "")
        self.assertEqual(
            _assemble([c for r in passes for c in r.calls]),
            [("lookup_fixture", {"key": "a"})],
        )

        detector = _strict(DeepSeekV4Detector)
        first = detector.parse_streaming_increment("\n", self.tools)
        second = detector.parse_streaming_increment("Hello", self.tools)
        self.assertEqual((first.normal_text, second.normal_text), ("", "\nHello"))

        detector = _strict(DeepSeekV4Detector)
        self.assertEqual(
            detector.parse_streaming_increment("Bye", self.tools).normal_text, "Bye"
        )
        self.assertEqual(
            detector.parse_streaming_increment("\n\n", self.tools).normal_text, ""
        )
        self.assertEqual(detector.finish(self.tools).normal_text, "\n\n")
        self.assertEqual(detector.finish(self.tools).normal_text, "")

    def test_cut_stream_with_an_unrecognised_body_tracks_nothing(self):
        """finish_reason length before the closer: no name went out, so nothing
        is left for the serving layer to complete with `{}`."""
        cut = MALFORMED_TURN[: MALFORMED_TURN.index(f"</{DSML}invoke>")]
        detector = _strict(DeepSeekV4Detector)
        _, calls = _stream(detector, cut, self.tools, 5)
        self.assertEqual(calls, [])
        self.assertEqual(detector.prev_tool_call_arr, [])
        self.assertEqual(detector.finish(self.tools).calls, [])

    def test_well_formed_streams_match_the_one_shot_result(self):
        samples = {
            "v4": _block(
                "v4",
                _invoke(
                    "search",
                    "\n"
                    + _param("query", "true", "WebNav")
                    + "\n"
                    + _param("topn", "false", "10")
                    + "\n",
                ),
                _invoke("lookup_fixture", '{"key": "alpha"}'),
                preamble="Checking.\n\n",
            ),
            "v32": _block(
                "v32", _invoke("search", _param("query", "true", 'He said "hi"\nnext'))
            ),
        }
        for flavor, text in samples.items():
            cls = FLAVORS[flavor][0]
            expected = _calls(_strict(cls).detect_and_parse(text, self.tools))
            self.assertTrue(expected)
            for chunk_size in CHUNK_SIZES:
                with self.subTest(flavor=flavor, chunk_size=chunk_size):
                    passes = _stream_passes(_strict(cls), text, self.tools, chunk_size)
                    _assert_names_travel_with_arguments(self, passes)
                    normal = "".join(r.normal_text for r in passes)
                    calls = [c for r in passes for c in r.calls]
                    self.assertEqual(_assemble(calls), expected)
                    self.assertNotIn(DSML, normal)

    def test_json_body_streams_the_name_with_the_json_prefix_in_one_item(self):
        """The forced-call shape (a calls block with one JSON object body): the
        first item of the call carries the name AND the JSON prefix at every
        chunk size, and streamed == tracked == the body."""
        body = '{"key": "alpha"}'
        text = _block("v4", _invoke("lookup_fixture", body), preamble="\n\n")
        expected = _calls(
            _strict(DeepSeekV4Detector).detect_and_parse(text, self.tools)
        )
        self.assertEqual(expected, [("lookup_fixture", {"key": "alpha"})])
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                detector = _strict(DeepSeekV4Detector)
                passes = _stream_passes(detector, text, self.tools, chunk_size)
                passes.append(detector.finish(self.tools))
                _assert_names_travel_with_arguments(self, passes)
                calls = [c for r in passes for c in r.calls]
                first = next(c for c in calls if c.tool_index == 0)
                self.assertEqual(first.name, "lookup_fixture")
                self.assertTrue(first.parameters.startswith("{"), first)
                self.assertEqual(sum(1 for c in calls if c.name), 1)
                self.assertEqual(_assemble(calls), expected)
                self.assertEqual("".join(c.parameters for c in calls), body)
                self.assertEqual(detector.streamed_args_for_tool[0], body)
                self.assertEqual(detector.prev_tool_call_arr[0]["arguments"], body)
                self.assertEqual("".join(r.normal_text for r in passes), "")

    def test_json_body_then_trailing_text_never_streams_a_complete_call(self):
        """A JSON object body followed by text the closer rejects, cut so the
        object is complete before the trailing text arrives. The first cut
        streamed the whole object (the common prefix of the two partial bodies
        includes the closing brace once text follows it) and the drop at the
        closer pinned the arguments to that complete, parsable object: the
        client held a complete call the one-shot path refuses. The character
        that completes the object is now held until the closer validates the
        body, so at every pass the client holds incomplete JSON, and a
        malformed close leaves it that way."""
        for label, trailing in TRAILING_TEXT.items():
            chunks = _json_body_then(trailing)
            text = "".join(chunks)
            one_shot = _strict(DeepSeekV4Detector).detect_and_parse(text, self.tools)
            self.assertEqual((one_shot.normal_text, one_shot.calls), (text, []))
            cuts = [chunks] + [
                [text[i : i + n] for i in range(0, len(text), n)] for n in CHUNK_SIZES
            ]
            seen_prefix = False
            for pieces in cuts:
                with self.subTest(trailing=label, chunks=len(pieces)):
                    detector = _strict(DeepSeekV4Detector)
                    passes, streamed = [], ""
                    with self.assertLogs(LOGGER, level="WARNING") as logs:
                        for piece in pieces:
                            result = detector.parse_streaming_increment(
                                piece, self.tools
                            )
                            passes.append(result)
                            streamed += "".join(c.parameters for c in result.calls)
                            # Never a complete object in flight, whatever the cut.
                            with self.assertRaises(json.JSONDecodeError):
                                json.loads(streamed)
                        passes.append(detector.finish(self.tools))
                    _assert_names_travel_with_arguments(self, passes)
                    calls = [c for r in passes for c in r.calls]
                    if calls:
                        self.assertEqual(calls[0].name, "lookup_fixture")
                        self.assertTrue(streamed.startswith("{"), streamed)
                        self.assertEqual(
                            detector.prev_tool_call_arr,
                            [{"name": "lookup_fixture", "arguments": streamed}],
                        )
                        self.assertEqual(detector.streamed_args_for_tool, [streamed])
                        seen_prefix = True
                    else:
                        self.assertEqual(detector.prev_tool_call_arr, [])
                        self.assertEqual(detector.streamed_args_for_tool, [])
                    # The block is content, verbatim, as the one-shot path returns it.
                    self.assertEqual("".join(r.normal_text for r in passes), text)
                    self.assertEqual(
                        sum("for lookup_fixture dropped" in l for l in logs.output),
                        1,
                    )
            self.assertTrue(seen_prefix, f"{label}: no cut streamed a prefix")

    def test_clean_bodies_release_the_closing_brace_only_with_the_closer(self):
        """The hold costs a well-formed call nothing but the timing of its
        last character: both modes assemble to the one-shot result, the
        streamed arguments equal the default mode's byte for byte, and in
        strict mode they stay incomplete JSON until the pass that carries the
        invoke closer."""
        samples = {
            "json": _block(
                "v4",
                _invoke("lookup_fixture", '{"key": "alpha"}'),
                preamble="Checking.\n\n",
            ),
            "xml": _block(
                "v4",
                _invoke(
                    "search",
                    "\n"
                    + _param("query", "true", "WebNav")
                    + "\n"
                    + _param("topn", "false", "10")
                    + "\n",
                ),
            ),
        }
        closer = f"</{DSML}invoke>"
        for label, text in samples.items():
            expected = _calls(DeepSeekV4Detector().detect_and_parse(text, self.tools))
            self.assertTrue(expected)
            closed_at = text.index(closer) + len(closer)
            for chunk_size in CHUNK_SIZES:
                with self.subTest(body=label, chunk_size=chunk_size):
                    strict, plain = _strict(DeepSeekV4Detector), DeepSeekV4Detector()
                    strict_calls, plain_calls, streamed, consumed = [], [], "", 0
                    for i in range(0, len(text), chunk_size):
                        piece = text[i : i + chunk_size]
                        consumed += len(piece)
                        result = strict.parse_streaming_increment(piece, self.tools)
                        strict_calls += result.calls
                        streamed += "".join(c.parameters for c in result.calls)
                        if consumed < closed_at:
                            with self.assertRaises(json.JSONDecodeError):
                                json.loads(streamed)
                        plain_calls += plain.parse_streaming_increment(
                            piece, self.tools
                        ).calls
                    self.assertEqual(_assemble(strict_calls), expected)
                    self.assertEqual(_assemble(plain_calls), expected)
                    self.assertEqual(
                        streamed, "".join(c.parameters for c in plain_calls)
                    )
                    self.assertEqual(
                        strict.streamed_args_for_tool, plain.streamed_args_for_tool
                    )

    def test_zero_argument_call_streams_its_name_at_the_closer(self):
        for flavor, (cls, _) in FLAVORS.items():
            for text in (
                _block(flavor, _invoke("submit", "\n")),
                _block(flavor, f'<{DSML}invoke name="submit"/>'),
            ):
                for chunk_size in CHUNK_SIZES:
                    with self.subTest(
                        flavor=flavor, chunk_size=chunk_size, text=text[:30]
                    ):
                        passes = _stream_passes(
                            _strict(cls), text, self.tools, chunk_size
                        )
                        _assert_names_travel_with_arguments(self, passes)
                        self.assertEqual(
                            _assemble([c for r in passes for c in r.calls]),
                            [("submit", {})],
                        )


class TestDefaultParsingUnchanged(CustomTestCase):
    """With the knob off (the default) the detectors behave as before."""

    def setUp(self):
        self.tools = _tools()

    def test_knob_is_off_by_default(self):
        self.assertFalse(envs.SGLANG_ENABLE_STRICT_DSML_TOOL_CALLS.get())
        self.assertFalse(DeepSeekV4Detector().strict)

    def test_malformed_body_still_parses_as_empty_arguments(self):
        result = DeepSeekV4Detector().detect_and_parse(MALFORMED_TURN, self.tools)
        self.assertEqual(_calls(result), [("set_flag", {})])
        self.assertEqual(
            result.normal_text, "I'll call the `set_flag` function to enable the flag."
        )
        self.assertEqual(
            DeepSeekV4Detector()._parse_parameters_from_xml('["alpha"]'), "{}"
        )

    def test_streaming_still_completes_a_json_body_ahead_of_trailing_text(self):
        """Knob off: the chunks of the strict trailing-text case stream as
        before (the name first with "", the object completes before the
        trailing text is seen, the closer adds nothing) and the one-shot
        parse of the same text is `{}`."""
        chunks = _json_body_then(TRAILING_TEXT["prose"])
        detector = DeepSeekV4Detector()
        items = [
            (c.tool_index, c.name, c.parameters)
            for piece in chunks
            for c in detector.parse_streaming_increment(piece, self.tools).calls
        ]
        self.assertEqual(
            items,
            [(0, "lookup_fixture", ""), (0, None, "{"), (0, None, '"key": "alpha"}')],
        )
        self.assertEqual(detector.streamed_args_for_tool, ['{"key": "alpha"}'])
        one_shot = DeepSeekV4Detector().detect_and_parse("".join(chunks), self.tools)
        self.assertEqual(_calls(one_shot), [("lookup_fixture", {})])

    def test_streaming_still_sends_the_name_first(self):
        text = _block("v4", _invoke("lookup_fixture", _param("key", "true", "a")))
        detector = DeepSeekV4Detector()
        passes = _stream_passes(detector, text, self.tools, 5)
        calls = [c for r in passes for c in r.calls]
        self.assertEqual((calls[0].name, calls[0].parameters), ("lookup_fixture", ""))
        self.assertEqual(_assemble(calls), [("lookup_fixture", {"key": "a"})])
        self.assertEqual(detector.finish(self.tools).normal_text, "")
        # Marker-free whitespace is released as content, not held.
        fresh = DeepSeekV4Detector()
        self.assertEqual(
            fresh.parse_streaming_increment("\n\n", self.tools).normal_text, "\n\n"
        )


if __name__ == "__main__":
    unittest.main()
