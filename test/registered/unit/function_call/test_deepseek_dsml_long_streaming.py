"""Long DSML tool calls streamed in small deltas (DeepSeek V3.2 / V4 / V4.1).

The streaming detector holds an open invoke in its buffer until the closer
arrives. These tests check that the open-invoke shortcut returns exactly what
a full re-scan of the buffer returns, and that the work per delta does not grow
with the length of the call.
"""

import json
import re
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DETECTORS = [DeepSeekV32Detector, DeepSeekV4Detector, DeepSeekV41Detector]
CHUNK_SIZES = [1, 3, 7, 18]


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name=name,
                parameters={"type": "object", "properties": {}},
            ),
        )
        for name in ("write_file", "lookup", "list_files")
    ]


def _code(detector, n_chars):
    """Code-like text with `<`, `</` and partial DSML closers, none complete."""
    pieces = [
        "def f(x):\n",
        "    return x < 3 and y </ 2\n",
        '    s = "<div>\\n"\n',
        f"    # {detector.invoke_end_token[:-1]} is not a closer\n",
        f"    # nor is <{detector.dsml_token}\n",
        "    # 中文 é\n",
    ]
    out = []
    while sum(map(len, out)) < n_chars:
        out.extend(pieces)
    return "".join(out)


def _transcript(detector_cls, n_chars):
    d = detector_cls()
    parameter = f"{d.dsml_token}{detector_cls.parameter_tag_name}"
    code = _code(d, n_chars)
    text = (
        "Let me write the file.\n\n"
        f"{d.bot_token}\n"
        f'{d.invoke_start_token} name="write_file">\n'
        f'<{parameter} name="path" string="true">src/app.py</{parameter}>\n'
        f'<{parameter} name="content" string="true">{code}</{parameter}>\n'
        f"{d.invoke_end_token}\n"
        f'{d.invoke_start_token} name="lookup">\n'
        + json.dumps({"query": code}, ensure_ascii=False)
        + f"\n{d.invoke_end_token}\n"
        f'{d.invoke_start_token} name="list_files"/>\n'
        f"{d.eot_token}"
    )
    expected = [
        ("write_file", {"path": "src/app.py", "content": code}),
        ("lookup", {"query": code}),
        ("list_files", {}),
    ]
    if d.strip_string_param_value:
        expected[0] = ("write_file", {"path": "src/app.py", "content": code.strip()})
    return text, expected


def _split(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


def _results(results):
    return [
        (r.normal_text, [(c.tool_index, c.name, c.parameters) for c in r.calls])
        for r in results
    ]


class TestDeepSeekDsmlLongStreaming(CustomTestCase):
    def test_incremental_matches_full_rescan(self):
        """Per delta, the results equal those of a detector that re-scans the
        whole buffer every time (open-invoke shortcut cleared before each delta)."""
        tools = _tools()
        for detector_cls in DETECTORS:
            text, expected = _transcript(detector_cls, 3000)
            for size in CHUNK_SIZES:
                with self.subTest(detector=detector_cls.__name__, chunk_size=size):
                    incremental, rescan = detector_cls(), detector_cls()
                    got, want = [], []
                    for chunk in _split(text, size):
                        got.append(incremental.parse_streaming_increment(chunk, tools))
                        rescan._open_invoke = None
                        want.append(rescan.parse_streaming_increment(chunk, tools))
                        self.assertEqual(incremental._buffer, rescan._buffer)
                    self.assertEqual(_results(got), _results(want))
                    self.assertEqual(
                        incremental.finish(tools).model_dump(),
                        rescan.finish(tools).model_dump(),
                    )

                    normal = "".join(r.normal_text for r in got)
                    calls = [c for r in got for c in r.calls]
                    self.assertEqual(normal, "Let me write the file.")
                    self.assertEqual([c.tool_index for c in calls], [0, 1, 2])
                    self.assertEqual(
                        [(c.name, json.loads(c.parameters)) for c in calls], expected
                    )

    def test_open_invoke_is_not_rescanned_per_delta(self):
        """Characters handed to re.search stay linear in the length of the call
        (a full re-scan per delta would be quadratic)."""
        tools = _tools()
        real_search = re.search
        for detector_cls in DETECTORS:
            text, expected = _transcript(detector_cls, 60_000)
            scanned = [0]

            def counting_search(pattern, string, flags=0):
                scanned[0] += len(string)
                return real_search(pattern, string, flags)

            with self.subTest(detector=detector_cls.__name__):
                detector = detector_cls()
                calls = []
                with patch.object(re, "search", counting_search):
                    for chunk in _split(text, 18):
                        calls.extend(
                            detector.parse_streaming_increment(chunk, tools).calls
                        )
                self.assertEqual(
                    [(c.name, json.loads(c.parameters)) for c in calls], expected
                )
                self.assertLess(scanned[0], 4 * len(text))


if __name__ == "__main__":
    unittest.main()
