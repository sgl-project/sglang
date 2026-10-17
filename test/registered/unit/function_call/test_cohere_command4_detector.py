"""Unit tests for CohereCommand4Detector streaming - no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _block(*names: str) -> str:
    items = ",".join(
        json.dumps(
            {
                "tool_call_id": str(i),
                "tool_name": name,
                "parameters": {"q": f"{name}{i}"},
            }
        )
        for i, name in enumerate(names)
    )
    return f"<|START_ACTION|>[{items}]<|END_ACTION|>"


class TestCohereCommand4Streaming(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="edit", parameters={"type": "object", "properties": {}}
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                ),
            ),
        ]
        self.detector = CohereCommand4Detector()

    def _stream(self, chunks):
        calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            calls.extend(result.calls)
        return [(call.tool_index, call.name) for call in calls]

    def test_index_is_call_position_not_tool_position(self):
        # "search" is the second tool in the request; the first call is index 0.
        self.assertEqual(self._stream([_block("search")]), [(0, "search")])

    def test_repeated_tool_gets_distinct_indices(self):
        self.assertEqual(
            self._stream([_block("search", "search")]), [(0, "search"), (1, "search")]
        )

    def test_indices_continue_across_blocks(self):
        self.assertEqual(
            self._stream([_block("search"), "\n", _block("edit")]),
            [(0, "search"), (1, "edit")],
        )

    def test_split_markers_still_index_from_zero(self):
        text = _block("search")
        chunks = [text[i : i + 5] for i in range(0, len(text), 5)]
        self.assertEqual(self._stream(chunks), [(0, "search")])

    def test_forwarded_unknown_tool_gets_valid_index(self):
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            self.assertEqual(self._stream([_block("read")]), [(0, "read")])

    def test_unknown_tool_dropped_by_default(self):
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            self.assertEqual(self._stream([_block("read", "search")]), [(0, "search")])

    def test_detector_state_tracks_emitted_calls(self):
        self._stream([_block("search", "edit")])
        self.assertEqual(
            [call["name"] for call in self.detector.prev_tool_call_arr],
            ["search", "edit"],
        )
        self.assertEqual(
            self.detector.streamed_args_for_tool,
            [json.dumps({"q": "search0"}), json.dumps({"q": "edit1"})],
        )
        self.assertEqual(self.detector.current_tool_id, 2)


if __name__ == "__main__":
    unittest.main()
