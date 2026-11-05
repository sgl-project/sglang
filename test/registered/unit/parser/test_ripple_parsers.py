"""Ripple's reasoning and tool-call parsers against its chat template's layout."""

import json
import unittest

from sglang.srt.configs.model_config import (
    get_hybrid_layer_ids,
    is_layer_types_hybrid_swa_model,
)
from sglang.srt.configs.ricursive import RicursiveConfig
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.ripple_detector import RippleDetector
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

CALL = '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>'
CHUNK_SIZES = (1, 2, 3, 5, 8, 13, 1000)


def chunks(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


def render_assistant(reasoning, content):
    """The assistant turn of the checkpoint's chat_template.jinja."""
    return f"<think>\n{reasoning}\n</think>\n{content}" if reasoning else content


class TestRippleReasoning(CustomTestCase):
    CASES = {
        "<think>\nstep 1\n\nstep 2\n</think>\nThe answer.": (
            "step 1\n\nstep 2",
            "The answer.",
        ),
        "<think>\nr\n</think>\n\n" + CALL: ("r", "\n" + CALL),
        "No reasoning, </think> and <think> are just text.": (
            "",
            "No reasoning, </think> and <think> are just text.",
        ),
        "<think>\ncut off\n": ("cut off\n", ""),
        "<thi": ("", "<thi"),
    }

    @staticmethod
    def _detector(**kwargs):
        return ReasoningParser("ripple", **kwargs).detector

    def test_streaming_matches_one_shot_at_any_chunk_size(self):
        for text, (reasoning, content) in self.CASES.items():
            result = self._detector().detect_and_parse(text)
            self.assertEqual(
                (result.reasoning_text, result.normal_text), (reasoning, content)
            )
            for size in CHUNK_SIZES:
                detector = self._detector()
                parts = [
                    detector.parse_streaming_increment(c) for c in chunks(text, size)
                ]
                parts.append(detector.finish())
                self.assertEqual(
                    "".join(p.reasoning_text for p in parts), reasoning, (text, size)
                )
                self.assertEqual(
                    "".join(p.normal_text for p in parts), content, (text, size)
                )

    def test_parsed_message_renders_back_to_generated_text(self):
        text = "<think>\nline 1\nline 2\n</think>\nThe answer.\n"
        result = self._detector().detect_and_parse(text)
        self.assertEqual(
            render_assistant(result.reasoning_text, result.normal_text), text
        )

    def test_reasoning_held_until_end_without_stream_reasoning(self):
        detector = self._detector(stream_reasoning=False)
        self.assertEqual(
            detector.parse_streaming_increment("<think>\nr").reasoning_text, ""
        )
        result = detector.parse_streaming_increment("\n</think>\nok")
        self.assertEqual((result.reasoning_text, result.normal_text), ("r", "ok"))


class TestRippleToolCalls(CustomTestCase):
    TOOLS = [
        Tool(
            type="function",
            function=Function(name="add", parameters={"type": "object"}),
        )
    ]

    def _stream(self, text, size):
        detector = RippleDetector()
        parts = [
            detector.parse_streaming_increment(c, self.TOOLS)
            for c in chunks(text, size)
        ]
        parts.append(detector.finish(self.TOOLS))
        content = "".join(p.normal_text for p in parts)
        calls = {}
        for call in (call for part in parts for call in part.calls):
            entry = calls.setdefault(call.tool_index, {"name": "", "arguments": ""})
            entry["name"] += call.name or ""
            entry["arguments"] += call.parameters or ""
        return content, [
            (c["name"], json.loads(c["arguments"])) for c in calls.values()
        ]

    def test_streaming_matches_one_shot_at_any_chunk_size(self):
        cases = {
            "\n" + CALL: "",
            "Let me add.\n" + CALL: "Let me add.",
            "Keeps\n\ninner lines.\n" + CALL + "\n" + CALL: "Keeps\n\ninner lines.",
            "Only text, trailing newline.\n": "Only text, trailing newline.\n",
            "ends in a partial tag <tool_c": "ends in a partial tag <tool_c",
        }
        for text, content in cases.items():
            result = RippleDetector().detect_and_parse(text, self.TOOLS)
            self.assertEqual(result.normal_text, content)
            expected = [(c.name, json.loads(c.parameters)) for c in result.calls]
            for size in CHUNK_SIZES:
                self.assertEqual(
                    self._stream(text, size), (content, expected), (text, size)
                )

    def test_constrained_call_is_in_the_template_layout(self):
        info = RippleDetector().structure_info()("add")
        self.assertEqual(info.begin + '{"a": 1, "b": 2}' + info.end, CALL)


class TestRicursiveConfig(CustomTestCase):
    def test_kv_pool_split_comes_from_layer_types(self):
        """The checkpoint also ships a precomputed split, which would skip the
        check of layer_types against the attention layers the model builds."""
        config = RicursiveConfig(
            num_hidden_layers=10,
            sliding_window=1024,
            is_hybrid_swa=True,
            hybrid_layer_pattern=[0] * 10,
        )
        self.assertTrue(is_layer_types_hybrid_swa_model(config.architectures, config))
        swa, full = get_hybrid_layer_ids(config.architectures, config)
        self.assertEqual(full, [4, 9])
        self.assertEqual(swa, [0, 1, 2, 3, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
