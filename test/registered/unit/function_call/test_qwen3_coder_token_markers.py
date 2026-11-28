"""qwen3_coder structural tags match <tool_call> as a token when possible (#45225)."""

import inspect
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call import base_format_detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.utils import strip_structural_tag_excludes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

_MARKER_IDS = {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}


class _Tokenizer:
    """Encodes the Qwen markers to one id each unless told to split them."""

    def __init__(self, split_markers: bool = False):
        self.split_markers = split_markers

    def encode(self, text, add_special_tokens=True):
        if text in _MARKER_IDS and not self.split_markers:
            return [_MARKER_IDS[text]]
        return [ord(c) for c in text]


def _tool() -> Tool:
    return Tool(
        type="function",
        function=Function(
            name="get_weather",
            strict=True,
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Seoul", "Tokyo"]}},
                "required": ["city"],
            },
        ),
    )


def _new_api(calls):
    def get_model_structural_tag(
        model,
        tools=None,
        tool_choice="auto",
        reasoning="enabled",
        parallel_tool_calls=True,
        token_markers=False,
    ):
        calls.append({"model": model, "token_markers": token_markers})
        return "tag"

    return get_model_structural_tag


def _old_api(calls):
    def get_model_structural_tag(
        model,
        tools=None,
        tool_choice="auto",
        reasoning="enabled",
        parallel_tool_calls=True,
    ):
        calls.append({"model": model})
        return "tag"

    return get_model_structural_tag


def _xgrammar_has_token_markers() -> bool:
    fn = base_format_detector.get_model_structural_tag
    return fn is not None and "token_markers" in inspect.signature(fn).parameters


class TestQwen3CoderTokenMarkers(unittest.TestCase):
    def _call(self, api_factory, detector):
        calls = []
        with patch.object(
            base_format_detector, "get_model_structural_tag", api_factory(calls)
        ):
            self.assertEqual(detector.get_structural_tag(tools=[_tool()]), "tag")
        self.assertEqual(len(calls), 1)
        return calls[0]

    def test_token_markers_with_single_token_markers(self):
        kwargs = self._call(_new_api, Qwen3CoderDetector(tokenizer=_Tokenizer()))
        self.assertEqual(kwargs["model"], "qwen_3_coder")
        self.assertIs(kwargs["token_markers"], True)

    def test_text_trigger_when_markers_split(self):
        detector = Qwen3CoderDetector(tokenizer=_Tokenizer(split_markers=True))
        self.assertIs(self._call(_new_api, detector)["token_markers"], False)

    def test_text_trigger_without_tokenizer(self):
        self.assertIs(
            self._call(_new_api, Qwen3CoderDetector())["token_markers"], False
        )

    def test_text_trigger_on_xgrammar_without_token_markers(self):
        kwargs = self._call(_old_api, Qwen3CoderDetector(tokenizer=_Tokenizer()))
        self.assertNotIn("token_markers", kwargs)

    def test_parser_passes_tokenizer_to_detector(self):
        parser = FunctionCallParser([_tool()], "qwen3_coder", tokenizer=_Tokenizer())
        self.assertTrue(parser.detector.use_structural_tag_token_markers())

    def test_strip_excludes_covers_exclude_tokens(self):
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "token_triggered_tags",
                "trigger_tokens": ["<tool_call>"],
                "tags": [],
                "exclude_tokens": ["<think>", "</think>"],
            },
        }
        strip_structural_tag_excludes(tag, ("<think>", "</think>"))
        self.assertEqual(tag["format"]["exclude_tokens"], [])

    @unittest.skipUnless(
        _xgrammar_has_token_markers(), "installed xgrammar has no token_markers"
    )
    def test_real_xgrammar_triggers_on_tool_call_token(self):
        parser = FunctionCallParser([_tool()], "qwen3_coder", tokenizer=_Tokenizer())
        kind, tag = parser.get_structure_constraint("auto")
        self.assertEqual(kind, "structural_tag")
        fmt = tag.model_dump(by_alias=True)["format"]
        self.assertEqual(fmt["type"], "token_triggered_tags")
        self.assertEqual(fmt["trigger_tokens"], ["<tool_call>"])
        self.assertEqual(
            fmt["tags"][0]["begin"], {"type": "token", "token": "<tool_call>"}
        )


if __name__ == "__main__":
    unittest.main()
