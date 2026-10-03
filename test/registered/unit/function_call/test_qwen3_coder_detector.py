"""Unit tests for Qwen3CoderDetector -- phantom tool-call over-capture.

qwen3_coder_detector emitted ``tool_calls`` from tool-call markup the model
only *quotes* (fenced examples, echoed payloads, reasoning about tool syntax).
Two failure layers, both covered here:

1. Bare ``<function=NAME>`` in prose (no wrapper). ``parse_streaming_increment``
   matched the structure tags anywhere in the buffer; the non-streaming
   ``detect_and_parse`` did not (it bails unless the ``<tool_call>`` wrapper is
   present). Fix: gate the streaming ``<function=`` / ``<parameter=`` /
   ``</function>`` / ``</tool_call>`` branches on ``self.is_inside_tool_call``.

2. ``<tool_call><function=NAME>`` where ``NAME`` is a placeholder / near-miss
   (``run_command`` for a declared ``run_commands``). Layer 1 still harvested
   it -- identically in stream and non-stream, because neither path validated
   the name. Fix: ``_is_declared_tool()`` -- in both paths a name that is not a
   declared tool is dropped (honoring ``SGLANG_FORWARD_UNKNOWN_TOOLS``), the
   same policy ``base_format_detector.parse_base_json`` uses for the JSON
   detectors.

No server, no model loading -- pure parser unit test.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# Tokens assembled from fragments so this source file never carries a
# contiguous tool call that a transcript-scanning agent could self-harvest.
FN_OPEN = "<" + "function="
FN_CLOSE = "<" + "/function" + ">"
PAR_OPEN = "<" + "parameter="
PAR_CLOSE = "<" + "/parameter" + ">"
TC_OPEN = "<" + "tool_call" + ">"
TC_CLOSE = "<" + "/tool_call" + ">"


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="run_commands",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            ),
        )
    ]


# (1) Bare, wrapper-less <function=...> the way docs abbreviate example markup.
BARE_PROSE = (
    "Here is what a call looks like:\n```text\n"
    + FN_OPEN
    + "..."
    + ">\n"
    + PAR_OPEN
    + "path"
    + ">\n/etc/hosts\n"
    + PAR_CLOSE
    + "\n(truncated -- no closing function tag)\n"
    + FN_OPEN
    + "example_function_name"
    + ">\n```\nThat is all.\n"
)

# (2) A full <tool_call> wrapper the model wrote around an *example*, naming a
# tool that was never declared (the classic near-miss: run_command[s]).
WRAPPED_EXAMPLE = (
    "For instance you might emit:\n```\n"
    + TC_OPEN
    + "\n"
    + FN_OPEN
    + "run_command"
    + ">\n"
    + PAR_OPEN
    + "command"
    + ">\nls\n"
    + PAR_CLOSE
    + "\n"
    + FN_CLOSE
    + "\n"
    + TC_CLOSE
    + "\n```\n-- but don't actually do that here.\n"
)

# (2b) Same, but inline in reasoning prose rather than a fence.
REASONING_MARKUP = (
    "If I were to emit "
    + TC_OPEN
    + FN_OPEN
    + "placeholder_tool"
    + ">"
    + FN_CLOSE
    + TC_CLOSE
    + " that would be wrong. Answer: 42."
)

# A genuine, wrapper-delimited invocation of a *declared* tool -- must parse.
GENUINE_CALL = (
    "sure\n"
    + TC_OPEN
    + "\n"
    + FN_OPEN
    + "run_commands"
    + ">\n"
    + PAR_OPEN
    + "command"
    + ">\necho hi\n"
    + PAR_CLOSE
    + "\n"
    + FN_CLOSE
    + "\n"
    + TC_CLOSE
    + "\ndone"
)


def _stream(text, tools, *, chunk=None):
    detector = Qwen3CoderDetector()
    pieces = (
        [text]
        if chunk is None
        else [text[i : i + chunk] for i in range(0, len(text), chunk)]
    )
    names, args = [], []
    for piece in pieces:
        r = detector.parse_streaming_increment(piece, tools)
        for c in r.calls or []:
            if getattr(c, "name", None):
                names.append(c.name)
            if getattr(c, "parameters", None):
                args.append(c.parameters)
    return names, "".join(args)


def _nonstream(text, tools):
    r = Qwen3CoderDetector().detect_and_parse(text, tools)
    return [c.name for c in r.calls or []]


class TestQwen3CoderPhantomToolCall(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    # ---- layer 1: bare markup in prose --------------------------------
    def test_bare_markup_nonstream(self):
        self.assertEqual(_nonstream(BARE_PROSE, self.tools), [])

    def test_bare_markup_stream(self):
        for chunk in (None, 1, 7):
            names, _ = _stream(BARE_PROSE, self.tools, chunk=chunk)
            self.assertEqual(names, [], f"chunk={chunk}: {names}")

    # ---- layer 2: wrapped example naming an undeclared tool ----------
    def test_wrapped_example_nonstream(self):
        self.assertEqual(_nonstream(WRAPPED_EXAMPLE, self.tools), [])

    def test_wrapped_example_stream(self):
        for chunk in (None, 1, 5, 13):
            names, _ = _stream(WRAPPED_EXAMPLE, self.tools, chunk=chunk)
            self.assertEqual(names, [], f"chunk={chunk}: {names}")

    def test_reasoning_markup_both_paths(self):
        self.assertEqual(_nonstream(REASONING_MARKUP, self.tools), [])
        names, _ = _stream(REASONING_MARKUP, self.tools, chunk=6)
        self.assertEqual(names, [])

    def test_forward_unknown_tools_still_forwards_wrapped(self):
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            self.assertEqual(_nonstream(WRAPPED_EXAMPLE, self.tools), ["run_command"])
            names, _ = _stream(WRAPPED_EXAMPLE, self.tools)
            self.assertEqual(names, ["run_command"])

    # ---- parity: the two paths must agree on every input ------------
    def test_stream_nonstream_name_parity(self):
        for text in (BARE_PROSE, WRAPPED_EXAMPLE, REASONING_MARKUP, GENUINE_CALL):
            s, _ = _stream(text, self.tools)
            ns = _nonstream(text, self.tools)
            self.assertEqual(sorted(s), sorted(ns), f"disagreement on {text[:40]!r}")

    # ---- no regression for genuine calls --------------------------
    def test_genuine_call_stream(self):
        for chunk in (None, 1, 4):
            names, args = _stream(GENUINE_CALL, self.tools, chunk=chunk)
            self.assertEqual(names, ["run_commands"], f"chunk={chunk}")
            self.assertIn("echo hi", args)

    def test_genuine_call_nonstream(self):
        self.assertEqual(_nonstream(GENUINE_CALL, self.tools), ["run_commands"])


if __name__ == "__main__":
    unittest.main()
